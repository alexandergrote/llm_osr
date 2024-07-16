import unittest
import pandas as pd

from src.util.constants import DatasetColumn
from src.ml.preprocessing.embedding import EmbeddingPreprocessor

class TestEmbeddingPreprocessor(unittest.TestCase):

    def setUp(self):

        # test embedding preprocessor
        self.data = pd.DataFrame({DatasetColumn.TEXT: ["I am a cat", "You are a dog", "We are animals"]})
        self.preprocessor = EmbeddingPreprocessor(embedding_model_name="bert-base-nli-mean-tokens")
    
    def test__fit(self):

        # Apply fit
        result = self.preprocessor._fit(self.data)

        # Check if the result is None
        self.assertIsNone(result)

    def test__transform(self):

        # Apply transform
        transformed_data = self.preprocessor._transform(self.data)

        # Check if the transformed data is equal to the original data
        self.assertTrue(transformed_data[[DatasetColumn.TEXT]].equals(self.data))
        self.assertIn(DatasetColumn.FEATURES, transformed_data.columns.to_list())
        self.assertEqual(transformed_data[DatasetColumn.FEATURES].isna().sum(), 0)

    def test__fit_transform(self):
            
        # Apply fit_transform
        transformed_data = self.preprocessor._fit_transform(self.data)

        # Check if the transformed data is equal to the original data
        self.assertTrue(transformed_data[[DatasetColumn.TEXT]].equals(self.data))
        self.assertIn(DatasetColumn.FEATURES, transformed_data.columns.to_list())
        self.assertEqual(transformed_data[DatasetColumn.FEATURES].isna().sum(), 0)

if __name__ == '__main__':
    unittest.main()
