import unittest

import pandas as pd

from src.io.data_import.news import NewsDataset


class TestNewsDataset(unittest.TestCase):

    def setUp(self) -> None:
            
        self.dataset = NewsDataset()
        self.n_texts = 100
    
    def test__load(self):
        
        output = self.dataset._load(random_seed=42)
        self.assertIsInstance(output, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()