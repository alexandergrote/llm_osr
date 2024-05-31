import unittest

from tests.integration.util import get_hydra_config
from src.main import main
from src.util.constants import Directory


class TestDoc(unittest.TestCase):

    def setUp(self):
        
        self.cfg = get_hydra_config(
            overrides=[
                'ml__preprocessing=embedding',
                'ml__classifier=doc'
            ]
        )

    def test_main(self):
        self.assertIsNone(main(self.cfg))

    def tearDown(self):

        # remove generated files
        for filename in ['checkpoint_doc_model.pth', 'loss.png']:
            (Directory.ROOT / filename).unlink(missing_ok=True)

if __name__ == '__main__':
    unittest.main()