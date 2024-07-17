import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.integration.util import get_hydra_config
from src.main import main


temp_dir = TemporaryDirectory()


class TestHyperSimpleShot(unittest.TestCase):

    def setUp(self):
        
        self.cfg = get_hydra_config(
            overrides=[
                'ml__preprocessing=embedding',
                'ml__classifier=hyper_simpleshot',
                'ml__classifier.params.n_trials=1'
            ]
        )


    @patch("src.io.data_export.mlflow.Exporter.export", return_value=None)
    @patch("src.io.data_import.base.BaseDataset.get_n_rows", return_value=100)
    @patch("src.main.get_hydra_output_dir", return_value=Path(temp_dir.name))
    def test_main(self, mock_export, mock_n_rows, mock_output_dir):
        self.assertIsNone(main(self.cfg))

    def tearDown(self):

        temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()