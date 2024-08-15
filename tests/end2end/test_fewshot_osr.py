import unittest
import mlflow
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from src.experiments import Experiment
from src.experiments.main import execute

experiments = [Experiment(name='tmp', overrides=[])]

temp_dir = TemporaryDirectory()


class TestFewShotOSR(unittest.TestCase):

    def setUp(self):
        pass       
        
    @patch("src.experiments.main.Experiment.create_experiments_from_yaml", return_value=experiments)
    @patch("src.main.get_hydra_output_dir", return_value=Path(temp_dir.name))
    def test_main(self, mock_create_experiments_from_yaml, mock_get_hydra_output_dir):
        self.assertIsNone(execute())

    def tearDown(self):

        temp_dir.cleanup()

        for exp in experiments:

            experiment_id = mlflow.get_experiment_by_name(exp.name)
            mlflow.delete_experiment(experiment_id.experiment_id)

if __name__ == '__main__':
    unittest.main()