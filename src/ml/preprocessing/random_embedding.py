
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional
from pydantic import BaseModel
from pydantic.config import ConfigDict


from src.ml.preprocessing.base import BasePreprocessor
from src.util.constants import DatasetColumn as dfc
from src.util.hashing import Hash


class RandomEmbeddingPreprocessor(BaseModel, BasePreprocessor):

    embedding_model_name: str = 'random'
    embedding_model_params: Optional[dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def hash(self, *args, **kwargs) -> str:
        return Hash.hash_recursive(self.embedding_model_name, self.embedding_model_params)


    def _fit(self, data: pd.DataFrame, **kwargs):
        """
        No need to fit the model
        """
        pass

    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # work on copy
        data_copy = data.copy()
        data_copy[dfc.FEATURES] = None

        # iterate over data
        embeddings = []
        embedding_size = 768

        step = kwargs.pop("step", "")
        msg = f"Embedding {step}\t"

        for _, _ in tqdm(data_copy.iterrows(), total=len(data_copy), desc=msg):

            embedding = np.random.randn(embedding_size)

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

    preprocessor = RandomEmbeddingPreprocessor()
    processed_data = preprocessor._fit_transform(data)

    print(processed_data)