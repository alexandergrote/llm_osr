import unittest
import os
import yaml
import pandas as pd

from src.util.constants import Directory
from src.io.data_import.Clinc150 import Clinc150Dataset

class TestClinic150Dataset(unittest.TestCase):

    def setUp(self) -> None:

        self.kind_info = {
            'imbalanced': 19225,
            'small': 16200,
            'plus': 23850
        }


        filepath = Directory.CONFIG / os.path.join('io__import', 'clinc.yaml')
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)['params']

        domain_mapping = config['domain_mapping']
        integer_mapping = config['integer_mapping']

        self.additional_params = {
            'domain_mapping': domain_mapping,
            'integer_mapping': integer_mapping,
        }
    
    def test_load(self):

        for kind, _ in self.kind_info.items():
            dataset = Clinc150Dataset(kind=kind, **self.additional_params)
            n_samples = 100
            output = dataset._load(random_seed=42, n_samples=n_samples)
            self.assertIsInstance(output, pd.DataFrame)

    def test_columns(self):
        for kind,_ in self.kind_info.items():
            dataset = Clinc150Dataset(kind=kind, **self.additional_params)
            n_samples = 100
            output = dataset._load(random_seed=42, n_samples=n_samples)
            self.assertIn('text', output.columns)
            self.assertIn('label', output.columns)

    def test_number_of_rows(self):
        for kind, expected_rows in self.kind_info.items():
            dataset = Clinc150Dataset(kind=kind, exclude_oos=False, **self.additional_params)
            output = dataset._load(random_seed=42, n_samples=None)
            self.assertEqual(len(output), expected_rows)

    

if __name__ == '__main__':
    unittest.main()