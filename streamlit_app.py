import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, date, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="StockSentry ML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
.prediction-box {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class StockSentryMLStreamlit:
    """Enhanced StockSentry with Streamlit integration"""

    def __init__(self, news_api_key=None):
        self.news_api_key = news_api_key
        self.models = {}
        self.best_model = None
        self.data = None
        self.model_metrics = {}

    def get_news_sentiment(self, company, date):
        """Get news sentiment with proper error handling"""
        if not self.news_api_key or self.news_api_key == "demo":
            # Return random sentiment for demo
            return np.random.uniform(-0.1, 0.1)

        url = f'https://newsapi.org/v2/everything?q={company}&from={date}&to={date}&sortBy=relevance&language=en&apiKey={self.news_api_key}'
        try:
            response = requests.get(url, timeout=10).json()
            sentiments = []
            for article in response.get('articles', []):
                if article.get('title'):
                    headline = article['title']
                    sentiment = TextBlob(headline).sentiment.polarity
                    sentiments.append(sentiment)

            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                return float(avg_sentiment)
            else:
                return 0.0
        except Exception as e:
            st.warning(f"News API error: {e}")
            return 0.0

    @st.cache_data
    def fetch_stock_data(_self, ticker, start_date, end_date):
        """Fetch stock data with caching"""
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            raise

    def prepare_features(self, ticker):
        """Prepare features for ML models"""
        if self.data is None:
            raise ValueError("No data available")

        features = []
        targets = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(len(self.data) - 1):
            try:
                # Update progress
                progress = (i + 1) / (len(self.data) - 1)
                progress_bar.progress(progress)
                status_text.text(f'Processing features: {i+1}/{len(self.data)-1}')

                current_date = self.data.loc[i, 'Date']
                if hasattr(current_date, 'strftime'):
                    date_str = current_date.strftime('%Y-%m-%d')
                else:
                    date_str = str(current_date)[:10]

                sentiment = self.get_news_sentiment(ticker, date_str)
                
                if isinstance(sentiment, (list, tuple, np.ndarray)):
                    sentiment = float(sentiment[0]) if len(sentiment) > 0 else 0.0
                else:
                    sentiment = float(sentiment)

                current_close = float(self.data.loc[i, 'Close'])
                next_close = float(self.data.loc[i + 1, 'Close'])

                feature_vector = [current_close, sentiment]
                features.append(feature_vector)
                targets.append(next_close)

            except Exception as e:
                continue

        progress_bar.empty()
        status_text.empty()
        
        return np.array(features), np.array(targets, dtype=float).reshape(-1)

    def initialize_models(self):
        """Initialize ML models"""
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=1.0, random_state=42, max_iter=2000),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0)
        }

        # Try to add XGBoost
        try:
            import xgboost as xgb
            self.models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        except ImportError:
            st.warning("XGBoost not available. Install with: pip install xgboost")

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Evaluate model performance"""
        try:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            metrics = {
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
            }

            if len(y_test) > 1:
                actual_direction = np.sign(np.diff(y_test))
                pred_direction = np.sign(np.diff(test_pred))
                directional_accuracy = np.mean(actual_direction == pred_direction)
                metrics['directional_accuracy'] = directional_accuracy
            else:
                metrics['directional_accuracy'] = 0.0

            return metrics, test_pred
        except Exception as e:
            st.error(f"Model evaluation error: {e}")
            return None, None

    def train_and_evaluate(self, ticker, start_date, end_date):
        """Complete training pipeline with Streamlit integration"""
        
        # Fetch data
        with st.spinner("Fetching stock data..."):
            self.data = self.fetch_stock_data(ticker, start_date, end_date)
            
        st.success(f"✅ Fetched {len(self.data)} days of data for {ticker}")

        # Prepare features
        with st.spinner("Preparing features and sentiment analysis..."):
            X, y = self.prepare_features(ticker)

        if len(X) == 0:
            st.error("No features could be prepared from the data")
            return None

        st.success(f"✅ Prepared {len(X)} feature samples")

        # Initialize models
        self.initialize_models()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate models
        results_container = st.container()
        with results_container:
            st.subheader("🤖 Model Training Results")
            
            results_data = []
            best_r2 = -np.inf
            best_model_name = None

            progress_bar = st.progress(0)
            
            for idx, (name, model) in enumerate(self.models.items()):
                progress_bar.progress((idx + 1) / len(self.models))
                
                with st.spinner(f"Training {name}..."):
                    metrics, predictions = self.evaluate_model(model, X_train, X_test, y_train, y_test)
                    
                if metrics:
                    results_data.append({
                        'Model': name,
                        'Test R²': f"{metrics['test_r2']:.4f}",
                        'Test MAE': f"${metrics['test_mae']:.2f}",
                        'Test RMSE': f"${metrics['test_rmse']:.2f}",
                        'Directional Accuracy': f"{metrics['directional_accuracy']:.2%}"
                    })
                    
                    self.model_metrics[name] = metrics
                    
                    if metrics['test_r2'] > best_r2:
                        best_r2 = metrics['test_r2']
                        self.best_model = model
                        best_model_name = name

            progress_bar.empty()

            # Display results table
            if results_data:
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Highlight best model
                st.markdown(f"""
                <div class="success-box">
                    <h4>🏆 Best Model: {best_model_name}</h4>
                    <p>R² Score: {best_r2:.4f}</p>
                </div>
                """, unsafe_allow_html=True)

        return self.best_model, X_test, y_test

    def predict_next_day(self, ticker):
        """Predict next day price"""
        if self.data is None or self.best_model is None:
            st.error("Please train the model first")
            return None

        try:
            latest_idx = len(self.data) - 1
            latest_date = self.data.loc[latest_idx, 'Date']

            if hasattr(latest_date, 'strftime'):
                date_str = latest_date.strftime('%Y-%m-%d')
            else:
                date_str = str(latest_date)[:10]

            latest_price = float(self.data.loc[latest_idx, 'Close'])
            latest_sentiment = self.get_news_sentiment(ticker, date_str)

            predicted_price = self.best_model.predict([[latest_price, latest_sentiment]])
            
            return float(predicted_price[0]), latest_price

        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

def create_price_chart(data, ticker):
    """Create interactive price chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines+markers',
        name='Close Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price Trend',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_sentiment_chart(data, sentiments, ticker):
    """Create sentiment analysis chart"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Price line
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='blue')),
        secondary_y=False,
    )
    
    # Sentiment line
    fig.add_trace(
        go.Scatter(x=data['Date'], y=sentiments, name="News Sentiment", 
                  line=dict(color='red', dash='dash')),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    
    fig.update_layout(title_text=f"{ticker} - Price vs News Sentiment")
    
    return fig

def create_prediction_chart(y_test, y_pred):
    """Create actual vs predicted chart"""
    fig = go.Figure()
    
    indices = list(range(len(y_test)))
    
    fig.add_trace(go.Scatter(
        x=indices,
        y=y_test,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=indices,
        y=y_pred,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Prices',
        xaxis_title='Test Sample',
        yaxis_title='Price (USD)',
        hovermode='x unified'
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 StockSentry ML</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Stock Price Prediction with Machine Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    news_api_key = st.sidebar.text_input(
        "News API Key (Optional)", 
        value="demo", 
        type="password",
        help="Get your free API key from newsapi.org. Use 'demo' for random sentiment data."
    )
    
    # Stock ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker", 
        value="AAPL",
        help="Enter a valid stock ticker (e.g., AAPL, GOOGL, TSLA)"
    ).upper()
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2023, 1, 1),
            max_value=date.today() - timedelta(days=30)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date(2023, 6, 30),
            min_value=start_date,
            max_value=date.today()
        )
    
    # Model selection options
    st.sidebar.subheader("🤖 Model Options")
    show_model_comparison = st.sidebar.checkbox("Show Model Comparison", value=True)
    show_visualizations = st.sidebar.checkbox("Show Visualizations", value=True)
    
    # Main content
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if ticker:
            # Initialize StockSentry
            stock_sentry = StockSentryMLStreamlit(news_api_key)
            
            try:
                # Train models
                result = stock_sentry.train_and_evaluate(ticker, start_date, end_date)
                
                if result is None:
                    st.error("Training failed. Please check your inputs and try again.")
                    return
                
                best_model, X_test, y_test = result
                
                # Make prediction
                st.subheader("🔮 Next Day Prediction")
                
                prediction_result = stock_sentry.predict_next_day(ticker)
                if prediction_result:
                    predicted_price, current_price = prediction_result
                    
                    change = predicted_price - current_price
                    change_pct = (change / current_price) * 100
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    # Prediction display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        st.metric("Predicted Price", f"${predicted_price:.2f}")
                    
                    with col3:
                        st.metric("Expected Change", f"{change_pct:+.2f}%", f"${change:+.2f}")
                    
                    # Prediction box
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>{direction} Price Prediction for {ticker}</h3>
                        <h2>${predicted_price:.2f}</h2>
                        <p>Expected change: {change_pct:+.2f}% (${change:+.2f})</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Visualizations
                if show_visualizations:
                    st.subheader("📊 Visualizations")
                    
                    # Price trend chart
                    st.subheader("💹 Price Trend")
                    price_fig = create_price_chart(stock_sentry.data, ticker)
                    st.plotly_chart(price_fig, use_container_width=True)
                    
                    # Sentiment analysis chart
                    if news_api_key != "demo":
                        st.subheader("🗞️ Sentiment Analysis")
                        with st.spinner("Analyzing news sentiment..."):
                            sentiments = []
                            for i in range(len(stock_sentry.data)):
                                date_str = stock_sentry.data.loc[i, 'Date'].strftime('%Y-%m-%d')
                                sentiment = stock_sentry.get_news_sentiment(ticker, date_str)
                                sentiments.append(sentiment)
                            
                            sentiment_fig = create_sentiment_chart(stock_sentry.data, sentiments, ticker)
                            st.plotly_chart(sentiment_fig, use_container_width=True)
                    
                    # Actual vs Predicted
                    if best_model is not None:
                        st.subheader("🎯 Model Performance")
                        y_pred = best_model.predict(X_test)
                        pred_fig = create_prediction_chart(y_test, y_pred)
                        st.plotly_chart(pred_fig, use_container_width=True)

                # Model comparison
                if show_model_comparison and stock_sentry.model_metrics:
                    st.subheader("⚖️ Model Comparison")
                    
                    # Create metrics comparison chart
                    models = list(stock_sentry.model_metrics.keys())
                    r2_scores = [stock_sentry.model_metrics[model]['test_r2'] for model in models]
                    mae_scores = [stock_sentry.model_metrics[model]['test_mae'] for model in models]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_r2 = go.Figure(data=[go.Bar(x=models, y=r2_scores)])
                        fig_r2.update_layout(title="R² Score Comparison", yaxis_title="R² Score")
                        st.plotly_chart(fig_r2, use_container_width=True)
                    
                    with col2:
                        fig_mae = go.Figure(data=[go.Bar(x=models, y=mae_scores)])
                        fig_mae.update_layout(title="MAE Comparison", yaxis_title="MAE ($)")
                        st.plotly_chart(fig_mae, use_container_width=True)

                # Download results
                st.subheader("💾 Export Results")
                
                if st.button("📥 Download Results as CSV"):
                    results_data = {
                        'Date': stock_sentry.data['Date'],
                        'Close_Price': stock_sentry.data['Close'],
                        'Ticker': ticker,
                        'Predicted_Next_Day': predicted_price if prediction_result else None
                    }
                    
                    results_df = pd.DataFrame(results_data)
                    csv = results_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please check your inputs and try again.")
        
        else:
            st.warning("Please enter a stock ticker symbol.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About StockSentry ML**")
    st.sidebar.info(
        "This app uses machine learning to predict stock prices based on historical data and news sentiment analysis. "
        "Predictions are for educational purposes only and should not be used as financial advice."
    )

if __name__ == "__main__":
    main()