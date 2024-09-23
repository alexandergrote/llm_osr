import unittest
import mlflow
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from src.experiments.util.types import Experiment
from src.experiments.cli import execute

overrides = [
    "io__import=hwu",
    "ml__classifier=naive",
    "ml__datasplit=fewshot_osr",
    "ml__datasplit.params.percentage_unknown_classes=0",
    "random_seed=0,1",
    "ml__evaluation=osr",
    "io__export=mlflow",
    "io__export.params.experiment_name=tmp"
]

experiments = [Experiment(name='tmp', overrides=overrides)]

temp_dir = TemporaryDirectory()


class TestFewShotOSR(unittest.TestCase):

    def setUp(self):
        pass       
        
    @patch("src.experiments.cli.ExperimentFactory.create_fewshot_experiments", return_value=experiments)
    @patch("src.main.get_hydra_output_dir", return_value=Path(temp_dir.name))
    def test_main(self, mock_create_experiments_from_yaml, mock_get_hydra_output_dir):
        self.assertIsNone(execute("fewshot*", filter_name='tmp'))

    def tearDown(self):

        temp_dir.cleanup()

        for exp in experiments:

            experiment_id = mlflow.get_experiment_by_name(exp.name)
            mlflow.delete_experiment(experiment_id.experiment_id)

if __name__ == '__main__':
    unittest.main()