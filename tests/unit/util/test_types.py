import unittest
from pathlib import Path
import pandas as pd
from src.util.types import MLPrediction, MLPredictionFiles
from enum import Enum


class TestMLPrediction(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = Path('/tmp/ml_prediction_test')
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Sample data for testing
        self.y_pred = pd.Series([1, 2, 3], name='y_pred', dtype=str)
        self.y_test = pd.Series([1, 2, 4], name='y_test', dtype=str)
        self.classes_in_training = {'class1', 'class2'}
        self.outlier_score = pd.Series([0.1, 0.2, 0.3], name='outlier_score')

        # Create an instance of MLPrediction
        self.mlprediction = MLPrediction(
            y_pred=self.y_pred,
            y_test=self.y_test,
            classes_in_training=self.classes_in_training,
            outlier_score=self.outlier_score
        )

    def tearDown(self):
        # Remove the temporary directory and its contents after each test
        for item in self.test_dir.iterdir():
            if item.is_file():
                item.unlink()
        self.test_dir.rmdir()

    def test_save_and_load(self):
        # Save the MLPrediction instance to the test directory
        self.mlprediction.save(self.test_dir)

        # Load the saved data back into a new MLPrediction instance
        loaded_mlprediction = MLPrediction.load(self.test_dir)

        # Verify that the loaded data matches the original data
        pd.testing.assert_series_equal(loaded_mlprediction.y_pred, self.y_pred)
        pd.testing.assert_series_equal(loaded_mlprediction.y_test, self.y_test)
        self.assertEqual(loaded_mlprediction.classes_in_training, self.classes_in_training)

        # Verify outlier scores if they exist
        if self.outlier_score is not None:
            pd.testing.assert_series_equal(loaded_mlprediction.outlier_score, self.outlier_score)

if __name__ == '__main__':
    unittest.main()