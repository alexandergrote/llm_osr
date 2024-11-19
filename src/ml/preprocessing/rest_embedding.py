
import pandas as pd
import numpy as np
import requests  # type: ignore
from tqdm import tqdm
from typing import Optional, Any
from pydantic import BaseModel, model_validator
from pydantic.config import ConfigDict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

from src.util.hashing import Hash
from src.ml.preprocessing.base import BasePreprocessor
from src.util.constants import DatasetColumn as dfc
from src.util.environment import PydanticEnvironment
from src.ml.classifier.llm.util.rate_limit import RateLimitManager
from src.ml.util.job_queue import Job, JobStatus

env = PydanticEnvironment.from_environment()


class HFEmbeddingPreprocessor(BaseModel, BasePreprocessor):

    url: str
    tqdm_disable: bool = False
    use_cache: bool = True
    save: bool = True
    rate_limit_manager: Optional[RateLimitManager] = None 
    _rest_model_name: str = "HFEmbeddingPreprocessor"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    @classmethod
    def _init_rate_limits(cls, values: Any):

        if not isinstance(values, dict):
            raise ValueError("Values must be a dictionary")    

        key = "rate_limit_manager"
        rate_limit_manager_str = values.get(key)

        if rate_limit_manager_str is None:
            return values
        
        is_str = isinstance(rate_limit_manager_str, str)
        is_rlm = isinstance(rate_limit_manager_str, RateLimitManager)

        if not (is_str or is_rlm):
            raise ValueError(f"{key} must be a string or RateLimitManager")
        
        if is_str:
        
            obj = RateLimitManager.create_from_config_file(
                rate_limit_manager_str
            )
        else:

            obj = rate_limit_manager_str
            obj.load()
            obj.save()

        assert isinstance(obj, RateLimitManager)
        
        values[key] = obj
        
        return values

    def hash(self, *args, **kwargs) -> str:
        return Hash.hash_recursive(self.__class__.__name__)

    def _fit(self, data: pd.DataFrame, **kwargs):
        """
        No need to fit the model
        """
        pass

    def _check_rate_limit(self):

        if self.rate_limit_manager is None:
            raise ValueError("Rate limit manager not set")

        self.rate_limit_manager.load()

        self.rate_limit_manager.check_execution(
            num_request_tokens=1
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5), reraise=True)
    def _backoff(self, job: Job, save: bool, use_cache: bool, **kwargs):

        """
        This function fails with expoential retries
        """

        if self.rate_limit_manager is None:
            raise ValueError("RateLimitManager not set")

        if job.filepath.exists() and use_cache:

            if job.is_success:
                return job
            
        self._check_rate_limit()

        job = job.execute(save=save, use_cache=use_cache)
        
        if job.status == JobStatus.failed:
            raise Exception(job.error_description)
        
        return job

    def _api_call(self, text: str, **kwargs) -> np.ndarray:

        headers = {"Authorization": f"Bearer {env.hf_token}"}
        paypload = {"inputs": text, "normalize": "true"}

        request_kwargs = {
            "url": self.url,
            "headers": headers,
            "json": paypload
        }

        job_id = f"{self.__class__.__name__}_{Hash.hash_string(text)}"

        job = Job(
            job_id=job_id,
            rest_model_name=self._rest_model_name, 
            function=requests.post, 
            request_dict=request_kwargs
        )

        try:

            job = self._backoff(job=job, save=self.save, use_cache=self.use_cache, **kwargs)

        except Exception as e:
            job.status = JobStatus.failed
            job.error_description = str(e)

        output = job.request_output

        assert isinstance(output, list), f"Expected list, got {type(output)}"
            
        return np.array(output)

    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # work on copy
        data_copy = data.copy()
        data_copy[dfc.FEATURES] = None

        # iterate over data
        embeddings = []

        for _, row in tqdm(data_copy.iterrows(), total=len(data_copy), disable=self.tqdm_disable):

            text = row[dfc.TEXT]

            try:

                embedding = self._api_call(text)

                if not isinstance(embedding, np.ndarray):
                    raise ValueError(f"Expected np.ndarray, got {type(embedding)}")

            except Exception as e:
                tqdm.write(f"Error processing this text: {text} --- error: {e}")
                embedding = None

            # append data
            embeddings.append(embedding)
        
        # update data
        data_copy[dfc.FEATURES] = embeddings

        return data_copy

    def _fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._transform(data=data, **kwargs)
            

if __name__ == '__main__':

    data = pd.DataFrame({
        dfc.TEXT: [
            "This is a test",
            "This is another test"
        ]
    })

    preprocessor = HFEmbeddingPreprocessor(
        url="https://api-inference.huggingface.co/models/mixedbread-ai/mxbai-embed-large-v1",
        rate_limit_manager="hf.yaml",
        save=False,
        use_cache=False,
    )
    processed_data = preprocessor._fit_transform(data)

    print(processed_data)