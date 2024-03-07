import unittest

import pandas as pd

from src.io.data_import.Banking77 import BankingDataset

class Banking77Dataset(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = BankingDataset()
    
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
        self.assertEqual(len(output), 13083)

if __name__ == '__main__':
    unittest.main()