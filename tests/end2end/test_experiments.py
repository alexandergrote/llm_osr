import concurrent.futures
import unittest
import os

from src.experiments.cli import ExperimentRunner, Experiment
from src.experiments.factory import ExperimentFactory
from src.util.constants import EnvMode


def run_experiment(experiment: Experiment):
    result = ExperimentRunner.run_process(experiment.command)
    return experiment.name, result.returncode == 0


class TestExperimentYamls(unittest.TestCase):

    def setUp(self) -> None:
        self.dryrun_key = 'MODE'
        os.environ[self.dryrun_key] = EnvMode.DRYRUN.value

    
    def test_experiment_yaml(self) -> None:

        experiments = ExperimentFactory.create_benchmark_experiments(
            random_seeds=[0],
            unknown_classes=[0, 0.6],
        )

        experiments += ExperimentFactory.create_llm_fewshot_experiments(
            random_seeds=[0],
            unknown_classes=[0, 0.6],
        )

        with self.subTest(msg="Testing experiments in parallel"):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(run_experiment, experiments))
                for name, success in results:
                    with self.subTest(msg=f"Testing experiment: {name}"):
                        self.assertTrue(success)
                        
    def tearDown(self) -> None:
        del os.environ[self.dryrun_key]


if __name__ == '__main__':
    unittest.main()
