import unittest

import pandas as pd

from src.io.data_import.hwu import HWUDataset

class TestHWUDataset(unittest.TestCase):

    def setUp(self) -> None:

        self.dataset = HWUDataset()
    
    def test_load(self):
        output = self.dataset._load(random_seed=42)
        self.assertIsInstance(output, pd.DataFrame)

    def test_columns(self):
        output = self.dataset._load(random_seed=42)
        self.assertIn('text', output.columns)
        self.assertIn('label', output.columns)

    def test_rows(self):
        output = self.dataset._load(random_seed=42)
        self.assertEqual(len(output), 11109)

if __name__ == '__main__':
    unittest.main()