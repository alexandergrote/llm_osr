
import pandas as pd
from tqdm import tqdm
from typing import Optional
from pydantic import BaseModel


from src.ml.util.cached_sentence_encoder import CachedSentenceEncoder
from src.ml.preprocessing.base import BasePreprocessor
from src.util.constants import DatasetColumn as dfc
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

        model = CachedSentenceEncoder(
            embedding_model_name=self.embedding_model_name,
            embedding_model_params=self.embedding_model_params
        )

        # iterate over data
        embeddings = []

        step = kwargs.pop("step", "")
        msg = f"Embedding {step}\t"

        for _, row in tqdm(data_copy.iterrows(), total=len(data_copy), desc=msg):

            text = row[dfc.TEXT]

            embedding = model.encode(text)        

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

    preprocessor = EmbeddingPreprocessor()
    processed_data = preprocessor._fit_transform(data)

    print(processed_data)