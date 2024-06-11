import warnings

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.ml.preprocessing.base import BasePreprocessor
from src.util.constants import DatasetColumn as dfc
from src.util.caching import PickleCacheHandler
from src.util.logging import console
from src.util.hashing import Hash


class EmbeddingPreprocessor(BaseModel, BasePreprocessor):

    embedding_model_name: str = 'paraphrase-MiniLM-L6-v2'
    embedding_model_params: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

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

        # load sentence transformer model
        with warnings.catch_warnings():
            
            warnings.simplefilter(action='ignore', category=FutureWarning)

            embedding_params: dict = self.embedding_model_params or {}
            
            model = SentenceTransformer(
                model_name_or_path=self.embedding_model_name,
                **embedding_params
            )

        # keep track of processed data in dict
        filename = Path(self.__class__.__name__) / f"{self.embedding_model_name}.pkl"

        cache_handler = PickleCacheHandler(
            filepath=filename
        )

        # load dict from file if it exists
        processed_data: Optional[Any] = cache_handler.read()

        if processed_data is None:
            processed_data = {}

        assert isinstance(processed_data, dict), "processed data must be a dictionary"

        # iterate over data
        embeddings = []

        step = kwargs.pop("step", "")
        msg = f"Embedding {step}\t"

        for _, row in tqdm(data_copy.iterrows(), total=len(data_copy), desc=msg):

            text = row[dfc.TEXT]

            # transform text
            try:

                # check if text is already processed
                if text in processed_data:
                    embedding = processed_data[text]
                else:
                
                    embedding = model.encode(text)        

            except Exception as e:

                console.log(f"Error processing text: {text}")
                
                # save processed data to not lose progress
                cache_handler.write(processed_data)
                
                raise e

            # append data
            embeddings.append(embedding)
        
        # update data
        data_copy[dfc.FEATURES] = embeddings

        # save processed data
        cache_handler.write(processed_data)

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

    preprocessor = EmbeddingPreprocessor()
    processed_data = preprocessor._fit_transform(data)

    print(processed_data)