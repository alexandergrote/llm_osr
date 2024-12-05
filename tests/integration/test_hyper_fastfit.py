import unittest
import sys

from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from src.util.load_hydra import get_hydra_config
from src.main import main


temp_dir = TemporaryDirectory()


class TestHyperFastFit(unittest.TestCase):

    def setUp(self):

        categories = [
            'card_about_to_expire', 
            'apple_pay_or_google_pay',
            'age_limit'
        ]

        categories_str = f"[{', '.join(categories)}]"
        
        self.cfg = get_hydra_config(
            overrides=[
                'io__import=banking',
                f'io__import.params.filter={categories_str}',
                'ml__classifier=hyper_fastfit',
                'ml__classifier.params.n_trials=1',
                'ml__classifier.params.model.params.num_train_epochs=1',
            ]
        )

    @unittest.skipIf(sys.platform.startswith("win"), "Test skipped on Windows")
    @patch("src.io.data_export.mlflow.Exporter.export", return_value=None)
    @patch("src.io.data_import.base.BaseDataset.get_n_rows", return_value=100)
    @patch("src.main.get_hydra_output_dir", return_value=Path(temp_dir.name))
    def test_main(self, mock_export, mock_n_rows, mock_output_dir):
        self.assertIsNone(main(self.cfg))

    def tearDown(self):

        temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()