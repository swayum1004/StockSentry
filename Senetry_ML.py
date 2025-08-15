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
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
from config import NEWS_API_URL
import logging
from datetime import datetime


class StockSentryML:
    """Enhanced StockSentry with multiple ML models and proper validation"""

    def __init__(self, news_api_key):
        self.news_api_key = news_api_key
        self.models = {}
        self.best_model = None
        self.data = None
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.info("StockSentryML initialized")

    def get_news_sentiment(self, company, date):
        """Get news sentiment with proper error handling"""
        if not self.news_api_key or self.news_api_key == "your_api_key_here":
            # Return random sentiment between -0.1 and 0.1 for demo purposes
            return np.random.uniform(-0.1, 0.1)

        url = (f'{NEWS_API_URL}?q={company}&from={date}&to={date}&sortBy=relevance&language=en&apiKey={self.news_api_key}')
        try:
            response = requests.get(url, timeout=10).json()
            sentiments = []
            for article in response.get('articles', []):
                if article.get('title'):
                    headline = article['title']
                    sentiment = TextBlob(headline).sentiment.polarity  # -1 to +1
                    sentiments.append(sentiment)

            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                return float(avg_sentiment)  # Ensure it's a single float
            else:
                return 0.0
        except Exception as e:
            return 0.0

    def fetch_stock_data(self, ticker, start_date="2023-01-01", end_date="2023-06-30"):
        """Fetch stock data with proper error handling"""
        try:
            self.data = yf.download(ticker, start=start_date, end=end_date)
            if self.data.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            self.data.reset_index(inplace=True)
            logging.info(f"Data fetched for {ticker}: {len(self.data)} rows")
            return self.data
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise

    def prepare_features(self, ticker):
        """Prepare features with fixed indexing"""
        if self.data is None:
            raise ValueError("No data available. Call fetch_stock_data first.")

        features = []
        targets = []

        for i in range(len(self.data) - 1):
            try:
                # Fixed indexing - access scalar values directly
                current_date = self.data.loc[i, 'Date']
                if hasattr(current_date, 'strftime'):
                    date_str = current_date.strftime('%Y-%m-%d')
                else:
                    date_str = str(current_date)[:10]  # Take first 10 chars (YYYY-MM-DD)

                # Get sentiment
                sentiment = self.get_news_sentiment(ticker, date_str)

                # Ensure sentiment is a single float
                if isinstance(sentiment, (list, tuple, np.ndarray)):
                    sentiment = float(sentiment[0]) if len(sentiment) > 0 else 0.0
                else:
                    sentiment = float(sentiment)

                # Get prices - handle both Series and scalar values
                current_close = self.data.loc[i, 'Close']
                next_close = self.data.loc[i + 1, 'Close']

                # Convert to float if needed
                if hasattr(current_close, 'iloc'):
                    current_close = float(current_close.iloc[0])
                else:
                    current_close = float(current_close)

                if hasattr(next_close, 'iloc'):
                    next_close = float(next_close.iloc[0])
                else:
                    next_close = float(next_close)


                # Create feature vector
                feature_vector = [current_close, sentiment]
                features.append(feature_vector)
                targets.append(next_close)

            except Exception as e:
                continue

        # Ensure targets is always 1D
        return np.array(features), np.array(targets, dtype=float).reshape(-1)

    def initialize_models(self):
        """Initialize multiple ML models"""
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=1.0, random_state=42, max_iter=2000),
            'SVR': SVR(kernel='rbf', C=1.0)
        }

        # Only include XGBoost if available
        try:
            import xgboost as xgb
            self.models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        except ImportError:
            pass


    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        try:
            # Train model
            model.fit(X_train, y_train)

            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)


            # Calculate metrics
            metrics = {
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
            }

            # Directional accuracy (financial metric)
            if len(y_test) > 1:
                actual_direction = np.sign(np.diff(y_test))
                pred_direction = np.sign(np.diff(test_pred))
                directional_accuracy = np.mean(actual_direction == pred_direction)
                metrics['directional_accuracy'] = directional_accuracy
            else:
                metrics['directional_accuracy'] = 0.0


            return metrics, test_pred
        except Exception as e:
            return None, None

    def train_with_cross_validation(self, X, y):
        """Train models with time-series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=3)
        results = {}

        for name, model in self.models.items():
            try:
                cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                    y_train_cv, y_val_cv = y[train_idx], y[val_idx]

                    model.fit(X_train_cv, y_train_cv)
                    val_pred = model.predict(X_val_cv)
                    cv_score = r2_score(y_val_cv, val_pred)
                    cv_scores.append(cv_score)

                results[name] = {
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'model': model
                }
            except Exception as e:
                continue

        return results


    def hyperparameter_tuning(self, X, y):
        """Tune hyperparameters for best models"""
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0]
            }
        }

        # Add XGBoost params if available
        if 'XGBoost' in self.models:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 4]
            }

        tuned_models = {}
        tscv = TimeSeriesSplit(n_splits=3)

        for name in param_grids.keys():
            if name in self.models:
                try:
                    base_model = self.models[name]
                    grid_search = GridSearchCV(
                        base_model,
                        param_grids[name],
                        cv=tscv,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )

                    grid_search.fit(X, y)
                    tuned_models[name] = grid_search.best_estimator_
                except Exception as e:
                    continue

        return tuned_models

    def create_ensemble(self, models):
        """Create ensemble model"""
        if len(models) < 2:
            return list(models.values())[0] if models else None

        estimators = [(name, model) for name, model in models.items()]
        ensemble = VotingRegressor(estimators=estimators)
        return ensemble

    def train_and_evaluate(self, ticker, start_date="2023-01-01", end_date="2023-06-30"):
        """Complete training and evaluation pipeline"""
        print(f"🔄 Training models for {ticker}...")

        try:
            # Fetching the  data and preparing  features
            self.fetch_stock_data(ticker, start_date, end_date)
            X, y = self.prepare_features(ticker)

            if len(X) == 0:
                raise ValueError("No features prepared")


            # 2. Initialize and train models
            self.initialize_models()
            logging.info(f"Models initialized: {list(self.models.keys())}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            # 3. Train models with cross-validation (silent)
            cv_results = self.train_with_cross_validation(X, y)

            # 4. Hyperparameter tuning (silent)
            tuned_models = self.hyperparameter_tuning(X, y)

            # 5. Evaluate all models and select best (silent)
            all_models = {**self.models, **tuned_models}
            best_r2 = -np.inf
            best_model_name = None

            for name, model in all_models.items():
                metrics, predictions = self.evaluate_model(model, X_train, X_test, y_train, y_test)
                if metrics and metrics['test_r2'] > best_r2:
                    best_r2 = metrics['test_r2']
                    self.best_model = model
                    best_model_name = name


            # 6. Try ensemble as backup
            if tuned_models:
                ensemble = self.create_ensemble(tuned_models)
                if ensemble:
                    ensemble_metrics, _ = self.evaluate_model(ensemble, X_train, X_test, y_train, y_test)
                    if ensemble_metrics and ensemble_metrics['test_r2'] > best_r2:
                        best_r2 = ensemble_metrics['test_r2']
                        self.best_model = ensemble
                        best_model_name = "Ensemble"


            print(f"✅ Best model: {best_model_name} (R² = {best_r2:.4f})")


            return self.best_model

        except Exception as e:
            print(f"⚠️  Training failed, using fallback model")
            # Fallback to simple Simple model
            self.best_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.best_model.fit(X_train, y_train)
            return self.best_model


    def predict_next_day(self, ticker):
        """Predict next day price using the best model"""
        if self.data is None or self.best_model is None:
            raise ValueError("Train the model first using train_and_evaluate()")

        try:
            # Get latest data
            latest_idx = len(self.data) - 1
            latest_date = self.data.loc[latest_idx, 'Date']

            if hasattr(latest_date, 'strftime'):
                date_str = latest_date.strftime('%Y-%m-%d')
            else:
                date_str = str(latest_date)[:10]

            latest_price = self.data.loc[latest_idx, 'Close']
            if hasattr(latest_price, 'iloc'):
                latest_price = float(latest_price.iloc[0])
            else:
                latest_price = float(latest_price)


            latest_sentiment = self.get_news_sentiment(ticker, date_str)

            # Predict using best model
            predicted_price = self.best_model.predict([[latest_price, latest_sentiment]])

            print(f"📊 Current price: ${latest_price:.2f}")
            print(f"🤖 Predicted price: ${predicted_price[0]:.2f}")
            change = predicted_price[0] - latest_price
            change_pct = (change / latest_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"{direction} Expected change: ${change:.2f} ({change_pct:+.2f}%)")


            return float(predicted_price[0])

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None



print("Welcome to StockSentry")
print("=" * 30)

# Get user inputs
load_dotenv() # Load variables from .env file
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
if not NEWS_API_KEY:
    print("🔐 No API key found in .env. Using demo mode.")
    NEWS_API_KEY = "your_api_key_here"

TICKER = input("Enter stock ticker (e.g., AAPL, GOOGL, TSLA): ").strip().upper()
if not TICKER:
    TICKER = "AAPL"
    print("Using default ticker: AAPL")

print("\nEnter date range for training data:")
START_DATE = input("Start date (YYYY-MM-DD format, e.g., 2023-01-01): ").strip()
if not START_DATE:
    START_DATE = "2023-01-01"
    print("Using default start date: 2023-01-01")

END_DATE = input("End date (YYYY-MM-DD format, e.g., 2023-06-30): ").strip()
if not END_DATE:
    END_DATE = "2023-06-30"
    print("Using default end date: 2023-06-30")

print("=" * 30)

# Initialize the enhanced StockSentry
stock_sentry = StockSentryML(NEWS_API_KEY)

# Train models and get the best one
try:
    best_model = stock_sentry.train_and_evaluate(TICKER, START_DATE, END_DATE)

    # Get prediction
    if best_model:
        predicted_price = stock_sentry.predict_next_day(TICKER)

        if predicted_price:
            print(f"\n🎯 Final Prediction: ${predicted_price:.2f}")
        else:
            print("\n❌ Prediction failed")
    else:
        print("\n❌ Model training failed")

except Exception as e:
    print(f"\n❌ Error: {e}")


# --- Visualization Section ---
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_closing_price(data, ticker):
    plt.figure(figsize=(12,6))
    sns.lineplot(x=data['Date'], y=data['Close'], marker='o', label='Close Price')
    plt.title(f'{ticker} Closing Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('closing_price.png')
    plt.show()

def plot_sentiment_over_time(data, ticker):
    if 'Sentiment' not in data.columns:
        sentiments = []
        for i in range(len(data)):
            date_str = data.loc[i, 'Date']
            if hasattr(date_str, 'strftime'):
                date_str = date_str.strftime('%Y-%m-%d')
            else:
                date_str = str(date_str)[:10]
            sentiments.append(stock_sentry.get_news_sentiment(ticker, date_str))
        data['Sentiment'] = sentiments
    fig, ax1 = plt.subplots(figsize=(12,6))
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price (USD)', color=color)
    ax1.plot(data['Date'], data['Close'], color=color, label='Close Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sentiment', color=color)
    ax2.plot(data['Date'], data['Sentiment'], color=color, linestyle='--', label='Sentiment')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(f'{ticker} - Close Price & News Sentiment Over Time')
    fig.tight_layout()
    plt.savefig('sentiment_over_time.png')
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred):
    import pandas as pd
    import numpy as np
    # Force to 1D numpy arrays, no matter what
    y_test_flat = np.array(y_test).reshape(-1)
    y_pred_flat = np.array(y_pred).reshape(-1)
    min_len = min(len(y_test_flat), len(y_pred_flat))
    y_test_flat = y_test_flat[:min_len]
    y_pred_flat = y_pred_flat[:min_len]
    results_df = pd.DataFrame({'Actual': y_test_flat, 'Predicted': y_pred_flat})
    fig = px.line(results_df, y=['Actual', 'Predicted'], title='Actual vs. Predicted Closing Price')
    fig.update_layout(xaxis_title='Sample', yaxis_title='Closing Price (USD)')
    fig.write_html('actual_vs_predicted.html')
    fig.show()
print("\n" + "=" * 30)
print("✅ Analysis complete!")

# --- Call visualizations after analysis ---
try:
    plot_closing_price(stock_sentry.data, TICKER)
    plot_sentiment_over_time(stock_sentry.data, TICKER)
    # For actual vs predicted, need y_test and predictions
    # Re-run prediction for test set
    X, y = stock_sentry.prepare_features(TICKER)
    y = np.asarray(y).flatten()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test = np.asarray(y_test).flatten()
    y_pred = stock_sentry.best_model.predict(X_test)
    y_pred = np.asarray(y_pred).flatten()
    plot_actual_vs_predicted(y_test, y_pred)
except Exception as e:
    print(f"Visualization error: {e}")