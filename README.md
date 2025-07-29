# StockSentry

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

This project is open-source and available under the MIT License. See the `LICENSE` file for more details.