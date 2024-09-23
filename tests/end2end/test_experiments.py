import unittest
import os

from src.experiments.cli import ExperimentRunner
from src.experiments.util.factory import ExperimentFactory
from src.util.constants import EnvMode


class TestExperimentYamls(unittest.TestCase):

    def setUp(self) -> None:
        self.dryrun_key = 'MODE'
        os.environ[self.dryrun_key] = EnvMode.DRYRUN.value

    
    def test_experiment_yaml(self) -> None:

        experiments = ExperimentFactory.create_fewshot_experiments(
            random_seeds=[0],
            unknown_classes=[0, 0.6],
        )

        for experiment in experiments:
            with self.subTest(msg=f"Testing experiment: {experiment.name}"):
                result = ExperimentRunner.run_process(experiment.command)
                self.assertTrue(result.returncode == 0)

    def tearDown(self) -> None:
        del os.environ[self.dryrun_key]


if __name__ == '__main__':
    unittest.main()
