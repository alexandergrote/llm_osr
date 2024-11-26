import unittest

from typing import cast
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from tests.unit.ml.classifier.llm.helper import mock_response, data_train, data_valid

from src.ml.classifier.llm.util.prediction import Prediction
from src.ml.classifier.llm.twostage import TwoStageLLM
from src.ml.classifier.llm.util.rest import StructuredRequestLLM
from src.util.load_hydra import get_hydra_config
from src.util.constants import DatasetColumn
from src.util.dynamic_import import DynamicImport


TMP_DIR = TemporaryDirectory()
JOB_DIR = Path(TMP_DIR.name) / 'jobs'
LOG_DIR = Path(TMP_DIR.name) / "logs"


output_binary_correct_true = [
    {
        'generated_text': '{"label": "true", "reasoning": "trivial"}',
        'details': {
            'tokens': [
                {'text': '{', 'logprob': 0},
                {'text': '"', 'logprob': 0},
                {'text': 'label', 'logprob': 0},
                {'text': ':', 'logprob': 0},
                {'text': '"', 'logprob': 0},
                {'text': 'true', 'logprob': 0},
                {'text': '"', 'logprob': 0},
                {'text': '}', 'logprob': 0},
            ]
        }
    }
]

output_binary_correct_false = [
    {
        'generated_text': '{"label": "false", "reasoning": "trivial"}',
        'details': {
            'tokens': [
                {'text': '{', 'logprob': 0},
                {'text': '"', 'logprob': 0},
                {'text': 'label', 'logprob': 0},
                {'text': ':', 'logprob': 0},
                {'text': '"', 'logprob': 0},
                {'text': 'false', 'logprob': 0},
                {'text': '"', 'logprob': 0},
                {'text': '}', 'logprob': 0},
            ]
        }
    }
]

output_binary_wrong = [
    {
        'generated_text': '{"label": "unknown", "reasoning": "trivial"}',
        'details': {
            'tokens': [
                {'text': '{', 'logprob': 0}
            ]
        }
    }
]

output_multiclass_correct = [
    {
        'generated_text': '{"label": "greeting", "reasoning": "trivial"}',
        'details': {
            'tokens': [
                {'text': '{', 'logprob': 0}
            ]
        }
    }
]

output_multiclass_wrong = [
    {
        'generated_text': '{"label": "another greeting", "reasoning": "trivial"}',
        'details': {
            'tokens': [
                {'text': '{', 'logprob': 0}
            ]
        }
    }
]


class TestTwoStage(unittest.TestCase):

    def setUp(self):

        # create directories
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.patcher = patch.multiple(
            'src.util.caching.JsonCache',
            read=lambda x: None,
            write=lambda x, obj: None
        )
        
        self.mocks = self.patcher.start()

    def _get_fitted_llm(self) -> TwoStageLLM:

        key = "ml__classifier"

        config = get_hydra_config(
            overrides=[
                f"{key}=two_stage_llama_8",
                f"{key}.params.unknown_detection_model=[hf-llama-8b.yaml]",
                f"{key}.params.classifier_model=[hf-llama-8b.yaml]",
            ]
        )

        llm = DynamicImport.init_class_from_dict(
            dictionary=config[key],
        )

        # Explicitly cast llm to TwoStageLLM
        llm = cast(TwoStageLLM, llm)

        llm.fit(
            x_train=data_train[DatasetColumn.FEATURES].values,
            y_train=data_train[DatasetColumn.LABEL].values,
            x_valid=data_valid[DatasetColumn.FEATURES].values,
            y_valid=data_valid[DatasetColumn.LABEL].values,
        )

        return llm

    @patch("src.util.logger.get_log_dir")
    @patch("src.ml.util.job_queue.get_job_dir")
    def test_unknown_detection(self, mock_job_dir, mock_log_dir):

        mock_job_dir.return_value = JOB_DIR
        mock_log_dir.return_value = LOG_DIR

        llm = self._get_fitted_llm()

        # case: wrongly formatted request
        # should not be saved
        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response(status_code=200, json_data=output_binary_wrong)
            llm._single_predict(text="Hello friend", use_cache=True)

        job_files = list(JOB_DIR.rglob("*.json"))
        error_files = list(LOG_DIR.rglob("*.json"))

        self.assertEqual(len(job_files), 0)
        self.assertEqual(len(error_files), 1)

        # case: corrected request
        # should be saved
        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response(status_code=200, json_data=output_binary_correct_true)
            result = llm._single_predict(text="Hello friend", use_cache=True)

        self.assertEqual(result, ("unknown", 1.0))

        job_files = list(JOB_DIR.rglob("*.json"))
        error_files = list(LOG_DIR.rglob("*.json"))

        self.assertEqual(len(job_files), 1)
        self.assertEqual(len(error_files), 0)

    
    @patch("src.util.logger.get_log_dir")
    @patch("src.ml.util.job_queue.get_job_dir")
    def test_classification(self, mock_job_dir, mock_log_dir):

        mock_job_dir.return_value = JOB_DIR
        mock_log_dir.return_value = LOG_DIR

        llm = self._get_fitted_llm()

        # case: wrongly formatted request
        # should not be saved
        with patch("requests.post") as mock_post:
            
            mock_post.side_effect = [
                mock_response(status_code=200, json_data=output_binary_correct_false),
                mock_response(status_code=200, json_data=output_multiclass_wrong),
                mock_response(status_code=200, json_data=output_multiclass_wrong),
                mock_response(status_code=200, json_data=output_multiclass_wrong),
                mock_response(status_code=200, json_data=output_multiclass_wrong),
                mock_response(status_code=200, json_data=output_multiclass_wrong)
            ]

            llm._single_predict(text="Hello friend", use_cache=True)

        job_files = list(JOB_DIR.rglob("*.json"))
        error_files = list(LOG_DIR.rglob("*.json"))

        self.assertEqual(len(job_files), 1)
        self.assertEqual(len(error_files), 1)

        rest_llm = llm.classifier_model.llms[0]

        assert isinstance(rest_llm, StructuredRequestLLM)

        prediction = Prediction.from_llm_job(
            filename=job_files[0],
            class_labels=["true", "false"],
            request_output_fun=rest_llm.request_output_classmethod
        )

        self.assertTrue(prediction.label == "false")


        # case: corrected request
        # should be saved
        with patch("requests.post") as mock_post:
            
            mock_post.side_effect = [
                mock_response(status_code=200, json_data=output_binary_correct_false),
                mock_response(status_code=200, json_data=output_multiclass_correct),
            ]

            result = llm._single_predict(text="Hello friend", use_cache=True)

        self.assertEqual(result, ("greeting", 0.0))

        job_files = list(JOB_DIR.rglob("*.json"))
        error_files = list(LOG_DIR.rglob("*.json"))

        self.assertEqual(len(job_files), 2)
        self.assertEqual(len(error_files), 0)


    def tearDown(self):

        # unlink all files in job and log dir
        # cleanup alone does not do the job 
        # todo: find better solution
        for file in JOB_DIR.rglob("*.json"):
            file.unlink()

        for file in LOG_DIR.rglob("*.json"):
            file.unlink()

        TMP_DIR.cleanup()

        self.patcher.stop()
        
        

if __name__ == '__main__':
    unittest.main()

    
    