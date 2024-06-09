import unittest
from unittest.mock import patch

from tests.integration.util import get_hydra_config
from src.main import main
from src.util.constants import Directory


class TestMLP(unittest.TestCase):

    def setUp(self):
        
        self.cfg = get_hydra_config(
            overrides=[
                'ml__preprocessing=embedding',
                'ml__classifier=mlp'
            ]
        )


    @patch("src.io.data_export.mlflow.Exporter.export", return_value=None)
    @patch("src.io.data_import.base.BaseDataset.get_n_rows", return_value=1000)
    def test_main(self, mock_export, mock_n_rows):
        self.assertIsNone(main(self.cfg))

    def tearDown(self):

        # remove generated files
        for filename in ['checkpoint_mlp_model.pth', 'loss_mlp.png']:
            (Directory.ROOT / filename).unlink(missing_ok=True)

if __name__ == '__main__':
    unittest.main()