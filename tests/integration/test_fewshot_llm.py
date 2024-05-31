import unittest


from tests.integration.util import get_hydra_config
from src.main import main


class TestFewShotLLM(unittest.TestCase):

    def setUp(self):
        
        self.cfg = get_hydra_config(
            overrides=[
                'io__import=banking',
                'ml__preprocessing=identity',
                'ml__classifier=llm',
            ]
        )

    def test_main(self):
        self.assertIsNone(main(self.cfg))


if __name__ == '__main__':
    unittest.main()