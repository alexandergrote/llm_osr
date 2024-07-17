import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from src.util.load_hydra import get_hydra_config
from src.main import main
from src.util.constants import Directory


temp_dir = TemporaryDirectory()


class TestDoc(unittest.TestCase):

    def setUp(self):
        
        self.cfg = get_hydra_config(
            overrides=[
                'ml__preprocessing=embedding',
                'ml__classifier=doc',
                'ml__classifier.params.epochs=1'
            ]
        )


    @patch("src.io.data_export.mlflow.Exporter.export", return_value=None)
    @patch("src.io.data_import.base.BaseDataset.get_n_rows", return_value=100)
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