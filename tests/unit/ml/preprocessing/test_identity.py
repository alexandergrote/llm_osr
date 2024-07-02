import unittest
import pandas as pd

from src.util.constants import DatasetColumn
from src.ml.preprocessing.identity import IdentityPreprocessor

class TestIdentityPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = IdentityPreprocessor()

    def test__fit(self):
        # Create a sample DataFrame
        data = pd.DataFrame({DatasetColumn.TEXT: [1, 2, 3], 'B': [4, 5, 6]})

        # Apply fit
        result = self.preprocessor._fit(data)

        # Check if the result is None
        self.assertIsNone(result)

    def test__transform(self):
        # Create a sample DataFrame
        data = pd.DataFrame({DatasetColumn.TEXT: [1, 2, 3], 'B': [4, 5, 6]})
        expected_data = data.copy()
        expected_data[DatasetColumn.FEATURES] = data[DatasetColumn.TEXT]

        # Apply transform
        transformed_data = self.preprocessor._transform(data)

        # Check if the transformed data is equal to the original data
        self.assertTrue(expected_data.equals(transformed_data))


    def test__fit_transform(self):
        # Create a sample DataFrame
        data = pd.DataFrame({DatasetColumn.TEXT: [1, 2, 3], 'B': [4, 5, 6]})
        expected_data = data.copy()
        expected_data[DatasetColumn.FEATURES] = data[DatasetColumn.TEXT]

        # Apply fit_transform
        transformed_data = self.preprocessor._fit_transform(data)

        # Check if the transformed data is equal to the original data
        self.assertTrue(transformed_data.equals(expected_data))

if __name__ == '__main__':
    unittest.main()
