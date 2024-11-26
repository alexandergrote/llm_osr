import unittest

from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from src.ml.classifier.llm.util.prediction import Prediction, PredictionV1
from src.ml.classifier.llm.util.logprob import LogProbScore, LogProb
from src.ml.classifier.llm.util.rest_inference import InferenceHandler
from src.ml.classifier.llm.util.rate_limit import RateLimitError

from tests.unit.ml.classifier.llm.helper import mock_response


TMP_DIR = TemporaryDirectory()
RL_DIR = Path(TMP_DIR.name) / 'rate_limits'
ERROR_DIR = Path(TMP_DIR.name) / 'errors'


output_correct_groq = {
    'choices': [
        {
            "message": {
                "content": '{"label": "greeting", "reasoning": "just a unittest"}'
            } 
        }
    ],
    "usage": {
        "completion_tokens": 5,
    }
}


class TestRestInference(unittest.TestCase):

    def setUp(self):
        
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        RL_DIR.mkdir(parents=True, exist_ok=True)

        self.patcher = patch.multiple(
            'src.util.caching.JsonCache',
            read=lambda x: None,
            write=lambda x, obj: None
        )
        
        self.mocks = self.patcher.start()

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

    def test_rate_limit_error_from_request(self):

        PredictionV1.valid_labels = ["greeting"]
        Prediction.valid_labels = ["greeting"]

        logprob_score = LogProbScore(
            answer=Prediction(reasoning="just a unittest", label="greeting"), 
            logprobs=[LogProb(text="reasoning", logprob=0.1)]
        )

        llms = [
                "hf-llama-8b.yaml",
                "groq-llama-8b.yaml",
            ]
        
        inference_handler = InferenceHandler(
            llms=llms
        )

        # assert that the rate limit error is raised in first request
        with patch("requests.post") as mock_post:

            mock_post.side_effect = [
                mock_response(status_code=429, json_data={}),
                mock_response(status_code=200, json_data=output_correct_groq)
            ]

            result = inference_handler(
                text="what is the meaning of life?",
                pydantic_model=PredictionV1,
                use_cache=False
            )

            self.assertIsInstance(result, LogProbScore)
            self.assertEqual(result.answer, logprob_score.answer)
            self.assertTrue(mock_post.call_count, 2)

        
        # assert that the rate limit error is raised in both requests
        with patch("requests.post") as mock_post:

            mock_post.side_effect = [
                mock_response(status_code=429, json_data={}),
                mock_response(status_code=429, json_data={}),
            ]

            with self.assertRaises(SystemExit):

                inference_handler(
                    text="what is the meaning of life?",
                    pydantic_model=PredictionV1,
                    use_cache=False
                )


    def tearDown(self):

        # unlink all files in job and log dir
        # cleanup alone does not do the job 
        # todo: find better solution
        directories = [ERROR_DIR, RL_DIR]

        for directory in directories:
            for file in directory.glob("*.json"):
                file.unlink()

        TMP_DIR.cleanup()

        self.patcher.stop()

if __name__ == '__main__':
    unittest.main()
