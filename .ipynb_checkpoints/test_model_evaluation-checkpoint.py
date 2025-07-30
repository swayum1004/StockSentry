import unittest
import numpy as np
from Senetry_ML import StockSentryML  # Assuming your class is saved in Senetry_ML.py
import os
from dotenv import load_dotenv

class TestStockSentryEvaluation(unittest.TestCase):

    def setUp(self):
        """Set up the StockSentryML instance and train the model"""
        load_dotenv()
        self.api_key = os.getenv("NEWS_API_KEY", "your_api_key_here")
        self.ticker = "AAPL"
        self.start_date = "2023-01-01"
        self.end_date = "2023-06-30"

        # Initialize model handler
        self.model_handler = StockSentryML(news_api_key=self.api_key)

        # Train the model (this fetches data, prepares features, etc.)
        self.best_model = self.model_handler.train_and_evaluate(
            self.ticker, self.start_date, self.end_date
        )

        # Manually prepare X, y and split to test prediction accuracy
        X, y = self.model_handler.prepare_features(self.ticker)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.predictions = self.model_handler.best_model.predict(self.X_test)

    def test_model_trained(self):
        """Test if best model is trained and available"""
        self.assertIsNotNone(self.best_model, "Best model should not be None")

    def test_prediction_shape(self):
        """Ensure predictions and y_test match in length"""
        self.assertEqual(len(self.predictions), len(self.y_test), "Mismatch in prediction and y_test length")

    def test_prediction_values(self):
        """Check predictions are numeric and finite"""
        self.assertTrue(np.all(np.isfinite(self.predictions)), "Predictions should be finite numbers")

if __name__ == "__main__":
    unittest.main()
