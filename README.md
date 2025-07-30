<!-- # StockSentry

## Overview

StockSentry is a Python-based project designed to predict the next day's stock price for a given ticker, incorporating both historical price data and news sentiment. It leverages `yfinance` for stock data, `requests` for news API interaction, `TextBlob` for sentiment analysis, and `scikit-learn` for machine learning.

## Features

- **Historical Stock Data Download**: Fetches historical stock data (Open, High, Low, Close, Adj Close, Volume) using `yfinance`.
- **News Sentiment Analysis**: Integrates news headlines from NewsAPI and calculates sentiment polarity using `TextBlob`.
- **Feature Engineering**: Combines historical stock prices with news sentiment to create robust features for prediction.
- **Machine Learning Model**: Utilizes `RandomForestRegressor` from `scikit-learn` for stock price prediction.
- **Data Splitting**: Splits data into training and testing sets for model evaluation.
- **Next Day Price Prediction**: Predicts the next day's closing price based on the latest available data.

## Technical Details

### Libraries Used

- `yfinance`: For downloading historical stock data.
- `requests`: For making HTTP requests to the NewsAPI.
- `pandas`: For data manipulation and analysis.
- `textblob`: For sentiment analysis of news headlines.
- `scikit-learn`: For machine learning functionalities, specifically `RandomForestRegressor` and `train_test_split`.

### Data Flow

1.  **Historical Stock Data**: Downloaded for a specified ticker and date range.
2.  **News Data**: Fetched from NewsAPI for each day, and sentiment is calculated.
3.  **Feature Creation**: Daily closing prices and news sentiments are combined into feature vectors.
4.  **Target Variable**: The next day's closing price is set as the target.
5.  **Model Training**: A `RandomForestRegressor` is trained on the prepared features and targets.
6.  **Prediction**: The trained model predicts the next day's stock price using the latest data.

## Setup and Usage

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone https://github.com/swayum1004/StockSentry.git
    cd stocksentry
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Obtain a NewsAPI Key:**
    - Register at [NewsAPI.org](https://newsapi.org/) to get your API key.
    - In a Google Colab environment, you can store your API key securely using `google.colab.userdata`:

        ```python
        from google.colab import userdata
        NEWS_API_KEY = userdata.get('news-api-key')
        ```
    - For local execution, it is recommended to set your API key as an environment variable:

        ```bash
        export NEWS_API_KEY='YOUR_NEWS_API_KEY'
        ```

### Running the Project

1.  **Open the Jupyter Notebook:**

    ```bash
    jupyter notebook stocksentry.ipynb
    ```

2.  **Execute the cells sequentially:**
    - The notebook will download historical stock data, fetch news, perform sentiment analysis, train the model, and make a prediction.
    - Ensure you have set up your `NEWS_API_KEY` as described above.

## 🔑 API Key Setup

This project uses the [NewsAPI](https://newsapi.org/) to fetch news headlines. To use it, you'll need to securely provide your API key via a `.env` file.

### Steps:

1. **Copy `.env.example` to `.env`**:
   ```bash
   cp .env.example .env
   ```
2. **Edit .env and add your News API key:
```bash
NEWS_API_KEY="your_actual_api_key_here"
```

## Potential Improvements

-   **API Key Management**: Implement more secure and flexible API key management for non-Colab environments (e.g., `.env` files).
-   **Advanced Sentiment Analysis**: Explore more sophisticated NLP models (e.g., VADER, BERT) for improved sentiment accuracy.
-   **Additional Features**: Incorporate technical indicators (e.g., Moving Averages, RSI), trading volume, or broader market indices.
-   **Model Optimization**: Experiment with different machine learning models (e.g., LSTM for time series) and perform hyperparameter tuning.
-   **Error Handling**: Add robust error handling for API calls and data processing.
-   **Real-time Integration**: Develop a system for real-time stock data and news fetching for live predictions.
-   **Deployment**: Refactor the notebook into a production-ready application with a user interface or API endpoint.

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details. -->
<!-- # StockSentry

## Overview

StockSentry is a Python-based project designed to predict the next day's stock price for a given ticker, incorporating both historical price data and news sentiment. It leverages `yfinance` for stock data, `requests` for news API interaction, `TextBlob` for sentiment analysis, and `scikit-learn` for machine learning.

## Features

- **Historical Stock Data Download**: Fetches historical stock data (Open, High, Low, Close, Adj Close, Volume) using `yfinance`.
- **News Sentiment Analysis**: Integrates news headlines from NewsAPI and calculates sentiment polarity using `TextBlob`.
- **Feature Engineering**: Combines historical stock prices with news sentiment to create robust features for prediction.
- **Machine Learning Model**: Utilizes `RandomForestRegressor` from `scikit-learn` for stock price prediction.
- **Data Splitting**: Splits data into training and testing sets for model evaluation.
- **Next Day Price Prediction**: Predicts the next day's closing price based on the latest available data.

## Technical Details

### Libraries Used

- `yfinance`: For downloading historical stock data.
- `requests`: For making HTTP requests to the NewsAPI.
- `pandas`: For data manipulation and analysis.
- `textblob`: For sentiment analysis of news headlines.
- `scikit-learn`: For machine learning functionalities, specifically `RandomForestRegressor` and `train_test_split`.

### Data Flow

1.  **Historical Stock Data**: Downloaded for a specified ticker and date range.
2.  **News Data**: Fetched from NewsAPI for each day, and sentiment is calculated.
3.  **Feature Creation**: Daily closing prices and news sentiments are combined into feature vectors.
4.  **Target Variable**: The next day's closing price is set as the target.
5.  **Model Training**: A `RandomForestRegressor` is trained on the prepared features and targets.
6.  **Prediction**: The trained model predicts the next day's stock price using the latest data.

## Setup and Usage

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone https://github.com/swayum1004/StockSentry.git
    cd stocksentry
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Obtain a NewsAPI Key:**
    - Register at [NewsAPI.org](https://newsapi.org/) to get your API key.
    - In a Google Colab environment, you can store your API key securely using `google.colab.userdata`:

        ```python
        from google.colab import userdata
        NEWS_API_KEY = userdata.get('news-api-key')
        ```
    - For local execution, it is recommended to set your API key as an environment variable:

        ```bash
        export NEWS_API_KEY='YOUR_NEWS_API_KEY'
        ```

### Running the Project

1.  **Open the Jupyter Notebook:**

    ```bash
    jupyter notebook stocksentry.ipynb
    ```

2.  **Execute the cells sequentially:**
    - The notebook will download historical stock data, fetch news, perform sentiment analysis, train the model, and make a prediction.
    - Ensure you have set up your `NEWS_API_KEY` as described above.

## 🔑 API Key Setup

This project uses the [NewsAPI](https://newsapi.org/) to fetch news headlines. To use it, you'll need to securely provide your API key via a `.env` file.

### Steps:

1. **Copy `.env.example` to `.env`**:
   ```bash
   cp .env.example .env
   ```
2. **Edit .env and add your News API key:
```bash
NEWS_API_KEY="your_actual_api_key_here"
```

## Potential Improvements

-   **API Key Management**: Implement more secure and flexible API key management for non-Colab environments (e.g., `.env` files).
-   **Advanced Sentiment Analysis**: Explore more sophisticated NLP models (e.g., VADER, BERT) for improved sentiment accuracy.
-   **Additional Features**: Incorporate technical indicators (e.g., Moving Averages, RSI), trading volume, or broader market indices.
-   **Model Optimization**: Experiment with different machine learning models (e.g., LSTM for time series) and perform hyperparameter tuning.
-   **Error Handling**: Add robust error handling for API calls and data processing.
-   **Real-time Integration**: Develop a system for real-time stock data and news fetching for live predictions.
-   **Deployment**: Refactor the notebook into a production-ready application with a user interface or API endpoint.

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details. -->
<!-- <p align="center">
  <img src="https://github.com/swayum1004/StockSentry/blob/main/docs/logo.png" alt="StockSentry Logo" width="200"/>
</p> -->

<h1 align="center">StockSentry 📈</h1>

<p align="center">
  <strong>An intelligent stock price prediction system that combines historical data with news sentiment analysis.</strong>
  <br />
  <br />
  <!-- <a href="#-getting-started-in-under-5-minutes"><strong>🚀 Get Started</strong></a> -->
  ·
  <a href="https://github.com/swayum1004/StockSentry/issues"><strong>🐛 Report a Bug</strong></a>
  ·
  <a href="https://github.com/swayum1004/StockSentry/issues"><strong>✨ Request a Feature</strong></a>
</p>

<p align="center">
  <a href="https://github.com/swayum1004/StockSentry/stargazers"><img src="https://img.shields.io/github/stars/swayum1004/StockSentry?style=for-the-badge&logo=github&color=FFDD00" alt="Stars"></a>
  <a href="https://github.com/swayum1004/StockSentry/blob/main/LICENSE"><img src="https://img.shields.io/github/license/swayum1004/StockSentry?style=for-the-badge&color=00BFFF" alt="License"></a>
  <a href="https://github.com/swayum1004/StockSentry/network/members"><img src="https://img.shields.io/github/forks/swayum1004/StockSentry?style=for-the-badge&logo=github&color=90EE90" alt="Forks"></a>
</p>

---

## 🌟 The Mission: Democratizing Stock Market Intelligence

The ability to predict stock movements by combining market data with real-world sentiment has traditionally been accessible only to institutional investors with expensive tools and data feeds. 

**StockSentry** changes that.

This project provides a complete, open-source solution for anyone to predict stock prices using both historical market data and news sentiment analysis. Whether you're a trader looking to enhance your strategy, a student learning about financial markets, or a developer exploring machine learning applications in finance, this project is for you.

### 🔥 Core Features

*   **Intelligent Data Fusion:** Combines historical stock prices with real-time news sentiment for more accurate predictions.
*   **100% Open-Source:** No expensive data subscriptions. Uses free APIs and open-source libraries.
*   **Easy-to-Use:** Simple Python notebook that can be run locally or in Google Colab.
*   **Robust ML Pipeline:** Uses RandomForestRegressor with feature engineering for reliable predictions.
*   **Real-time News Integration:** Fetches latest news headlines and analyzes sentiment using TextBlob.

---

## 🏗️ System Architecture: How It Works

StockSentry follows a clear data processing pipeline that transforms raw market data and news into actionable predictions.

<details>
  <summary><strong>Click to expand the detailed prediction workflow</strong></summary>

  ### The Life of a Stock Prediction

  1.  **Historical Data Collection:** 
      *   Uses `yfinance` to download historical stock data (Open, High, Low, Close, Volume)
      *   Fetches data for a specified date range and ticker symbol
  
  2.  **News Data Gathering:**
      *   Connects to NewsAPI to fetch relevant news headlines for each trading day
      *   Filters news by company/ticker relevance
  
  3.  **Sentiment Analysis:**
      *   Uses `TextBlob` to analyze the sentiment polarity of news headlines
      *   Calculates daily sentiment scores ranging from -1 (negative) to +1 (positive)
  
  4.  **Feature Engineering:**
      *   Combines historical price data with sentiment scores
      *   Creates feature vectors that capture both technical and fundamental factors
  
  5.  **Model Training:**
      *   Trains a `RandomForestRegressor` on the engineered features
      *   Uses next-day closing price as the target variable
  
  6.  **Prediction:** 
      *   Makes next-day price predictions using the latest available data
      *   Provides confidence intervals and model performance metrics

</details>

---

## 🚀 The Tech Stack: Built for Accuracy and Simplicity

Every technology was chosen for its reliability, ease of use, and proven effectiveness in financial data analysis.

| Component           | Technology                | Rationale & Key Benefits                                                                                |
| ------------------ | ------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Data Source**    | **yfinance**             | **Reliable & Free.** Yahoo Finance provides accurate historical stock data with a simple Python API. |
| **News API**       | **NewsAPI**              | **Comprehensive Coverage.** Access to thousands of news sources with real-time updates.              |
| **Sentiment Analysis** | **TextBlob**         | **Simple & Effective.** Pre-trained sentiment analysis that works well out-of-the-box.              |
| **Machine Learning** | **scikit-learn**        | **Battle-Tested.** RandomForestRegressor provides robust predictions with minimal overfitting.        |
| **Data Processing** | **pandas**              | **Industry Standard.** The go-to library for financial data manipulation and analysis.               |
| **Environment**    | **Jupyter Notebook** **.py file**    | **Interactive Development.** Perfect for data exploration and model iteration.                        |

---

## 🛠️ Getting Started in Under 5 Minutes

No complex setup required. Just Python and an API key.

### Prerequisites

1.  **Python 3.x:** Make sure you have Python installed. [Get it here](https://www.python.org/downloads/).
2.  **NewsAPI Key:** Free API key from [NewsAPI.org](https://newsapi.org/).

### Installation & Launch

1.  **Clone the Project:**
    ```bash
    git clone https://github.com/swayum1004/StockSentry.git
    cd StockSentry
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **🔑 Set Up Your API Key:**
    
    **Option A: Using .env file (Recommended)**
    ```bash
    cp .env.example .env
    # Edit .env and add your NewsAPI key:
    # NEWS_API_KEY="your_actual_api_key_here"
    ```
    
    **Option B: For Google Colab**
    ```python
    from google.colab import userdata
    NEWS_API_KEY = userdata.get('news-api-key')
    ```

4.  **🎉 Start Predicting:**
    ```bash
    jupyter notebook stocksentry.ipynb
    ```
    
    Or run directly in [Google Colab](https://colab.research.google.com/)!

    Note: You can directly run the file in VScode also 

---

## 📊 Features in Detail

### Historical Stock Data Download
Fetches comprehensive historical stock data including:
- Open, High, Low, Close prices
- Adjusted Close prices
- Trading Volume
- Customizable date ranges

### News Sentiment Analysis
- Integrates news headlines from NewsAPI
- Calculates sentiment polarity using TextBlob
- Handles multiple news sources and relevance filtering
- Daily sentiment aggregation

### Machine Learning Pipeline
- **Feature Engineering**: Combines price data with sentiment scores
- **Model**: RandomForestRegressor for robust predictions
- **Validation**: Train/test split for performance evaluation
- **Prediction**: Next-day closing price forecasting

### Next Day Price Prediction
Predicts tomorrow's closing price based on:
- Latest historical price data
- Recent news sentiment
- Trained model patterns

---

## 🗺️ Roadmap: From Good to Great

*   [ ] **Phase 1: Enhanced Analytics**
    *   [ ] Technical indicators integration (RSI, MACD, Moving Averages)
    *   [ ] Multiple ML model comparison (LSTM, XGBoost, etc.)
    *   [ ] Advanced sentiment analysis with VADER/BERT

*   [ ] **Phase 2: Real-time Features**
    *   [ ] Live data streaming
    *   [ ] Real-time prediction API
    *   [ ] Web dashboard for visualization

*   [ ] **Phase 3: Advanced Intelligence**
    *   [ ] Multi-stock portfolio analysis
    *   [ ] Risk assessment metrics
    *   [ ] Integration with trading platforms

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the Repository**
2. **Create a Feature Branch:** `git checkout -b feature/amazing-feature`
3. **Commit Changes:** `git commit -m 'Add amazing feature'`
4. **Push to Branch:** `git push origin feature/amazing-feature`
5. **Open a Pull Request**

See our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines.

---

## 🌟 Contributors

Thanks to these wonderful people:

<a href="https://github.com/swayum1004/StockSentry/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swayum1004/StockSentry" />
</a>

---

## ⚠️ Disclaimer

**Important:** This project is for educational and research purposes only. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred from using this software.

---

## 📜 License

This project is freely available under the **MIT License**. See the `LICENSE` file for more information.

---

<div align="center">
  <img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
  <p>Built with ❤️ and a passion for democratizing financial intelligence.</p>
  <img src="https://komarev.com/ghpvc/?username=swayum1004-StockSentry&label=Project%20Views&color=00BFFF&style=flat" alt="Project views" />
</div>