
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel
from pydantic.config import ConfigDict

from src.ml.util.backoff import BackoffMixin
from src.ml.util.job_queue import RequestFunction
from src.util.hashing import Hash
from src.ml.preprocessing.base import BasePreprocessor
from src.util.constants import DatasetColumn as dfc
from src.util.environment import PydanticEnvironment


env = PydanticEnvironment.from_environment()


class HFEmbeddingPreprocessor(BackoffMixin, BaseModel, BasePreprocessor):

    url: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def hash(self, *args, **kwargs) -> str:
        return Hash.hash_recursive(self.__class__.__name__)


    def _fit(self, data: pd.DataFrame, **kwargs):
        """
        No need to fit the model
        """
        pass

    def _api_call(self, text: str) -> np.ndarray:

        headers = {"Authorization": f"Bearer {env.hf_token}"}
        paypload = {"inputs": text, "normalize": "true"}

        request_kwargs = {
            "url": self.url,
            "headers": headers,
            "json": paypload
        }

        job_id = f"{self.__class__.__name__}_{Hash.hash_string(text)}"

        job = self.completion_with_backoff_and_queue(
            job_id=job_id,
            function=RequestFunction.post,
            **request_kwargs
        )

        output = job.request_output

        assert isinstance(output, list), f"Expected list, got {type(output)}"
            
        return np.array(output)

    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # work on copy
        data_copy = data.copy()
        data_copy[dfc.FEATURES] = None

        # iterate over data
        embeddings = []

        for _, row in tqdm(data_copy.iterrows(), total=len(data_copy)):

            text = row[dfc.TEXT]

            try:

                embedding = self._api_call(text)

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
        url="https://api-inference.huggingface.co/models/mixedbread-ai/mxbai-embed-large-v1"
    )
    processed_data = preprocessor._fit_transform(data)

    print(processed_data)