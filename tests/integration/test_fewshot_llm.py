import unittest
from unittest.mock import patch

from tests.integration.util import get_hydra_config
from src.main import main
from src.util.constants import UnknownClassLabel

mock_prediction = f"""
{{
    "reasoning": "The model predicts that the card is about to expire",
    "label": "{UnknownClassLabel.UNKNOWN_STR.value}"
}}
"""

class TestFewShotLLM(unittest.TestCase):

    def setUp(self):

        categories = [
            'card_about_to_expire', 
            'apple_pay_or_google_pay',
            'age_limit'
        ]

        categories_str = f"[{', '.join(categories)}]"
        
        self.cfg = get_hydra_config(
            overrides=[
                'io__import=banking',
                f'io__import.params.filter={categories_str}',
                'ml__datasplit.params.percentage_unknown_classes=0',
                'ml__preprocessing=identity',
                'ml__classifier=llm',
            ]
        )

    @patch("src.io.data_export.mlflow.Exporter.export", return_value=None)
    @patch("src.ml.classifier.util.llm_models.LangchainWrapper._call", return_value=mock_prediction)
    @patch("src.io.data_import.base.BaseDataset.get_n_rows", return_value=100)
    def test_main(self, mock_export, mock_llm_call, mock_n_rows):
        self.assertIsNone(main(self.cfg))


if __name__ == '__main__':
    unittest.main()