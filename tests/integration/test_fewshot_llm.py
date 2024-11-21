import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from src.util.load_hydra import get_hydra_config
from src.main import main
from src.util.types import LogProb
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.util.prediction import Prediction

categories = [
    'card_about_to_expire', 
    'apple_pay_or_google_pay',
    'age_limit'
]

Prediction.valid_labels = categories

mock_rest_llm_call = LogProbScore(
    answer=Prediction(reasoning='hi', label='card_about_to_expire'),
    logprobs=[LogProb(text='hi', logprob=-0.1)]
)    

temp_dir = TemporaryDirectory()

class TestFewShotLLM(unittest.TestCase):

    def setUp(self):

        categories_str = f"[{', '.join(categories)}]"
        
        self.cfg = get_hydra_config(
            overrides=[
                'io__import=banking',
                f'io__import.params.filter={categories_str}',
                'ml__datasplit.params.percentage_unknown_classes=0',
                'ml__preprocessing=identity',
                'ml__classifier=one_stage_llama_8',
            ]
        )

    @patch("src.io.data_export.mlflow.Exporter.export", return_value=None)
    @patch("src.ml.classifier.llm.util.rest_inference.InferenceHandler.__call__", return_value=mock_rest_llm_call)
    @patch("src.io.data_import.base.BaseDataset.get_n_rows", return_value=100)
    @patch("src.main.get_hydra_output_dir", return_value=Path(temp_dir.name))
    def test_main(self, mock_export, mock_llm, mock_n_rows, mock_output_dir):
        self.assertIsNone(main(self.cfg))


if __name__ == '__main__':
    unittest.main()