import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Senetry_ML import StockSentryML  # assuming this is your actual model file

class TestModelEvaluationWithDummyData(unittest.TestCase):

    def setUp(self):
        # Step 1: Set up dummy data
        # y = 2x (perfect linear data)
        self.X_train = np.array([[1], [2], [3], [4]])
        self.y_train = np.array([2, 4, 6, 8])

        self.X_test = np.array([[5], [6]])
        self.y_test = np.array([10, 12])

        # Step 2: Create model handler (dummy API key)
        self.model_handler = StockSentryML(news_api_key='dummy_key')

        # Step 3: Use actual model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def test_evaluate_model(self):
        # Step 4: Evaluate using real method
        metrics, predictions = self.model_handler.evaluate_model(
            self.model, self.X_train, self.X_test, self.y_train, self.y_test
        )

        # Step 5: Assert metric keys
        self.assertIn('test_mae', metrics)
        self.assertIn('test_rmse', metrics)
        self.assertIn('test_r2', metrics)
        self.assertIn('directional_accuracy', metrics)

        # Step 6: Check expected results (since it's perfect linear data)
        self.assertAlmostEqual(metrics['test_mae'], 0.0, places=1)
        self.assertAlmostEqual(metrics['test_rmse'], 0.0, places=1)
        self.assertAlmostEqual(metrics['test_r2'], 1.0, places=2)
        self.assertEqual(metrics['directional_accuracy'], 1.0)

        # Step 7: Check predictions match expected [10, 12]
        np.testing.assert_array_almost_equal(predictions, self.y_test, decimal=1)

        # Step 8: Save test data and predictions to CSV
        df = pd.DataFrame({
            'X_test': self.X_test.flatten(),
            'y_test': self.y_test,
            'y_pred': predictions
        })
        df.to_csv('test_predictions_output.csv', index=False)


if __name__ == '__main__':
    unittest.main()
