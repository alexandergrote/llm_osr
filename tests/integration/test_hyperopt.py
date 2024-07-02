import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from tests.integration.util import get_hydra_config
from src.main import main
from src.util.constants import Directory


temp_dir = TemporaryDirectory()


class TestHyperDoc(unittest.TestCase):

    def setUp(self):
        
        self.cfg = get_hydra_config(
            overrides=[
                'ml__preprocessing=embedding',
                'ml__classifier=hyper_doc',
                'ml__classifier.params.model.params.epochs=1',
                'ml__classifier.params.n_trials=1',
            ]
        )


    @patch("src.io.data_export.mlflow.Exporter.export", return_value=None)
    @patch("src.io.data_import.base.BaseDataset.get_n_rows", return_value=1000)
    @patch("src.main.get_hydra_output_dir", return_value=Path(temp_dir.name))
    def test_main(self, mock_export, mock_n_rows, mock_output_dir):
        self.assertIsNone(main(self.cfg))

    def tearDown(self):

        temp_dir.cleanup()

        # remove generated files
        for filename in ['checkpoint_doc_model.pth', 'loss.png']:
            (Directory.ROOT / filename).unlink(missing_ok=True)

if __name__ == '__main__':
    unittest.main()