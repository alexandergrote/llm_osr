import unittest

from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from src.util.constants import LLMModels, RESTAPI_URLS
from src.ml.classifier.llm.util.prediction import Prediction
from src.ml.classifier.llm.util.rest import StructuredRequestLLM
from src.ml.classifier.llm.util.rate_limit import RateLimit, RateLimitError, RateLimitManager
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

output_correct_hf = [
    {
        'generated_text': '{"label": "greeting", "reasoning": "trivial"}',
        'details': {
            'tokens': [
                {'text': '{', 'logprob': 0}
            ]
        }
    }
]



class TestStructuredRestLLM(unittest.TestCase):

    def setUp(self):

        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        RL_DIR.mkdir(parents=True, exist_ok=True)

    @patch("src.ml.classifier.llm.util.rate_limit.get_rate_limit_dir")
    @patch("src.ml.classifier.llm.util.rate_limit.RateLimit._get_timestamp")
    def test_rlm_integration(self, mock_get_timestamp, mock_rate_limit_dir):
        
        mock_rate_limit_dir.return_value = RL_DIR
        mock_get_timestamp.return_value = "a"

        llm = StructuredRequestLLM.create_from_yaml_file(
            "groq-llama-8b.yaml"
        )

        # dummy query
        query = "hello world is wonderful"

        key_value_pairs = [
                ("num_req_minute", 1),
                ("num_token_minute", 30),
                ("num_req_day", 1),
            ]

        # first query
        with patch("requests.post") as mock_post:

            Prediction.valid_labels = ["greeting"]
            mock_post.return_value = mock_response(status_code=200, json_data=output_correct_groq)
            
            llm(text=query, pydantic_model=Prediction, use_cache=False)

            assert isinstance(llm, StructuredRequestLLM)

            # 'num_req_minute', 'num_token_minute', 'num_req_day', 'num_token_day'
            limits = llm.rate_limit_manager.rate_limits
            
            for key, value in key_value_pairs:
                rate_limit = limits[key]
                assert isinstance(rate_limit, RateLimit) 
                
                with self.subTest(msg=key):
                    self.assertEqual(rate_limit.records["a"], value)

        # second query
        with patch("requests.post") as mock_post:

            Prediction.valid_labels = ["greeting"]
            mock_post.return_value = mock_response(status_code=200, json_data=output_correct_groq)
            
            llm(text=query, pydantic_model=Prediction, use_cache=False)

            assert isinstance(llm, StructuredRequestLLM)

            # 'num_req_minute', 'num_token_minute', 'num_req_day', 'num_token_day'
            limits = llm.rate_limit_manager.rate_limits
            
            for key, value in key_value_pairs:
                rate_limit = limits[key]
                assert isinstance(rate_limit, RateLimit) 
                
                with self.subTest(msg=key):
                    self.assertEqual(rate_limit.records["a"], value * 2)

    @patch("src.util.logger.get_log_dir")
    @patch("src.ml.classifier.llm.util.rate_limit.get_rate_limit_dir")
    @patch("src.ml.classifier.llm.util.rate_limit.RateLimit._get_timestamp")
    def test_rlm_failed_integration(self, mock_get_timestamp, mock_rate_limit_dir, mock_error_log_dir):
        
        mock_error_log_dir.return_value = ERROR_DIR
        mock_rate_limit_dir.return_value = RL_DIR
        mock_get_timestamp.return_value = "a"

        rate_limit_manager = RateLimitManager(
            name="my_rlm",
            rate_limits={
                "num_req_minute": RateLimit(
                    limit=0,
                    increment_level="frequency",
                    agg_level="%Y-%m-%d %H:%M",
                    action="raise",
                    waiting_time=60
                )
            }
        )

        Prediction.valid_labels = ["greeting"]

        llm = StructuredRequestLLM(
            name="hf-llama-8b",
            rest_api_model_name="llama-8b",
            url=RESTAPI_URLS[LLMModels.LLAMA_3_8B_Remote_HF],
            request_input_classmethod="create_hf_llama_request_input",
            request_output_classmethod="from_llama_hf_request",
            request_input_data_extraction="get_prompt_from_hf_data",
            rate_limit_manager=rate_limit_manager
        )

        # dummy query
        query = "hello world is wonderful"        

        # first query
        with patch("requests.post") as mock_post:

            Prediction.valid_labels = ["greeting"]
            mock_post.return_value = mock_response(status_code=200, json_data=output_correct_hf)
            
            with self.assertRaises(RateLimitError):
                llm(text=query, pydantic_model=Prediction, use_cache=False)
            

    def tearDown(self):

        # unlink all files in job and log dir
        # cleanup alone does not do the job 
        # todo: find better solution
        for file in RL_DIR.glob("*.json"):
            file.unlink()

        TMP_DIR.cleanup()

if __name__ == '__main__':
    unittest.main()
