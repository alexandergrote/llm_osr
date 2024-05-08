import unittest

import pandas as pd

from src.io.data_import.Clinc150 import Clinc150Dataset

class TestClinic150Dataset(unittest.TestCase):

    def setUp(self) -> None:
        self.kind_info = {
            'imbalanced': 19225,
            'small': 16200,
            'plus': 23850
        }
    
    def test_load(self):

        for kind, _ in self.kind_info.items():
            dataset = Clinc150Dataset(kind=kind)
            n_samples = 100
            output = dataset._load(random_seed=42, n_samples=n_samples)
            self.assertIsInstance(output, pd.DataFrame)

    def test_columns(self):
        for kind,_ in self.kind_info.items():
            dataset = Clinc150Dataset(kind=kind)
            n_samples = 100
            output = dataset._load(random_seed=42, n_samples=n_samples)
            self.assertIn('text', output.columns)
            self.assertIn('label', output.columns)

    def test_number_of_rows(self):
        for kind, expected_rows in self.kind_info.items():
            dataset = Clinc150Dataset(kind=kind)
            output = dataset._load(random_seed=42, n_samples=None)
            self.assertEqual(len(output), expected_rows)

    

if __name__ == '__main__':
    unittest.main()