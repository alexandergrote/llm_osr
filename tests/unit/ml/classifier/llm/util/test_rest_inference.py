import unittest

from unittest.mock import patch

from src.ml.classifier.llm.util.prediction import Prediction, PredictionV1
from src.ml.classifier.llm.util.logprob import LogProbScore, LogProb
from src.ml.classifier.llm.util.rest_inference import InferenceHandler
from src.ml.classifier.llm.util.rate_limit import RateLimitError

class TestRestInference(unittest.TestCase):

    def setUp(self):
        pass

    @patch("src.ml.classifier.llm.util.rest.StructuredRequestLLM.__call__")
    def test_rate_limit_error(self, mock_call):
        
        PredictionV1.valid_labels = ["label"]
        Prediction.valid_labels = ["label"]

        logprob_score = LogProbScore(
            answer=Prediction(reasoning="unittest", label="label"), 
            logprobs=[LogProb(text="reasoning", logprob=0.1)]
        )

        mock_call.side_effect = [
            RateLimitError("first model has rate limit error"),
            logprob_score,
        ]

        llms = [
            "groq-llama-8b.yaml",
            "hf-llama-8b.yaml"
        ]

        inference_handler = InferenceHandler(
            llms=llms
        )

        result = inference_handler(
            text="what is the meaning of life?",
            pydantic_model=PredictionV1,
            use_cache=False
        )

        self.assertEqual(result, logprob_score)        

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
