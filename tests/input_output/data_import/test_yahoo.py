import unittest

import pandas as pd

from src.io.data_import.yahoo_answers import YahooAnswersDataset

class TestYahooDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = YahooAnswersDataset()
    
    def test_load(self):
        n_samples = 100
        output = self.dataset._load(random_seed=42, n_samples=n_samples)
        self.assertIsInstance(output, pd.DataFrame)

    def test_columns(self):
        n_samples = 100
        output = self.dataset._load(random_seed=42, n_samples=n_samples)
        self.assertIn('text', output.columns)
        self.assertIn('label', output.columns)

    def test_rows(self):
        output = self.dataset._load(random_seed=42, n_samples=None)
        self.assertEqual(len(output), 1460000)

if __name__ == '__main__':
    unittest.main()