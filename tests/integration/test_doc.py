import os
import unittest
import warnings

from hydra import compose, initialize

from src.main import main
from src.util.constants import Directory


class TestDoc(unittest.TestCase):

    def setUp(self):
        
        # get hydra config
        with warnings.catch_warnings():
            
            warnings.simplefilter(action='ignore', category=UserWarning)

            config_path = os.path.join('..', '..', 'config')

            with initialize(config_path=config_path):
                overrides = [
                    'ml__preprocessing=embedding',
                    'ml__classifier=doc'
                ]
                self.cfg = compose(config_name="config", overrides=overrides)

    def test_main(self):
        self.assertIsNone(main(self.cfg))

    def tearDown(self):

        # remove generated files
        for filename in ['checkpoint_doc_model.pth', 'loss.png']:
            (Directory.ROOT / filename).unlink(missing_ok=True)

if __name__ == '__main__':
    unittest.main()