import unittest
import os

from src.experiments.main import get_experiments
from src.util.constants import EnvMode


class TestExperimentYamls(unittest.TestCase):

    def setUp(self) -> None:
        self.dryrun_key = 'MODE'
        os.environ[self.dryrun_key] = EnvMode.DRYRUN.value

    
    def test_experiment_yaml(self) -> None:

        experiments = get_experiments()

        for experiment in experiments:
            with self.subTest(msg=experiment.name):
                self.assertIsNone(experiment.run())

    def tearDown(self) -> None:
        
        del os.environ[self.dryrun_key]
        