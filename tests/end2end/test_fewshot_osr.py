import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from src.experiments import Experiment
from src.experiments.run.fewshot import run_experiments

experiments = [Experiment(name='tmp', overrides=[])]

temp_dir = TemporaryDirectory()


class TestFewShotOSR(unittest.TestCase):

    def setUp(self):
        pass       
        
    @patch("src.experiments.run.fewshot.experiments", experiments)
    @patch("src.io.data_export.mlflow.Exporter.export", return_value=None)
    @patch("src.main.get_hydra_output_dir", return_value=Path(temp_dir.name))
    def test_run_exerpiments(self, mock_export, mock_output_dir):
        self.assertIsNone(run_experiments())

    def tearDown(self):
        temp_dir.cleanup()

if __name__ == '__main__':
    unittest.main()