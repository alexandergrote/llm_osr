import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from typing import Type

from src.ml.preprocessing.base import BasePreprocessor
from src.ml.preprocessing.rest_embedding import HFEmbeddingPreprocessor
from src.util.constants import DatasetColumn as dfc
from src.util.logger import console


class CosineSelector(BaseModel):

    embedder: Type[BasePreprocessor] = HFEmbeddingPreprocessor(
        url="https://api-inference.huggingface.co/models/mixedbread-ai/mxbai-embed-large-v1",
        tqdm_disable=True
    )
    _key: str = 'cosine_score'

    def get_n_most_similar_datapoints(self, query: str, data: pd.DataFrame, n: int, include_score: bool = False) -> pd.DataFrame:

        # work on copy
        data_copy = data.copy()

        embeddings_df = self.embedder._fit_transform(data=data_copy)
        embeddings = np.stack(embeddings_df[dfc.FEATURES].to_list())

        query_embedding_df = self.embedder._transform(data=pd.DataFrame({dfc.TEXT: [query]}))
        query_embedding = np.stack(query_embedding_df[dfc.FEATURES].to_list())

        # Calculate cosine similarity between the query and each sentence
        cosine_similarities = cosine_similarity(query_embedding, embeddings).flatten()

        data_copy[self._key] = cosine_similarities

        data_copy_sub = data_copy.sort_values(by=self._key, ascending=False).head(n)

        if include_score:
            return data_copy_sub
        
        return data_copy_sub.drop(columns=[self._key])
        
    def get_most_similar_datapoints_for_n_classes(self, query: str, data: pd.DataFrame, n_classes: int, n: int, include_score: bool = False) -> pd.DataFrame:

        # work on copy
        data_copy = data.copy()

        # scored data points
        data_copy = self.get_n_most_similar_datapoints(
            query=query,
            data=data_copy,
            n=len(data_copy),
            include_score=True
        )

        # limit to top classes
        top_n_classes = data_copy.groupby([dfc.LABEL])[self._key].max().sort_values(ascending=False).head(n_classes)
        data_copy_sub = data_copy[data_copy[dfc.LABEL].isin(top_n_classes.index)]
        
        # limit to n data points
        data_copy_sub_sub = data_copy_sub.sort_values([dfc.LABEL, self._key], ascending=[True, False]).groupby('label').head(n)
        
        if include_score:
            return data_copy_sub_sub
        
        return data_copy_sub_sub.drop(columns=[self._key])
        

if __name__ == '__main__':

    data = pd.DataFrame({
        dfc.TEXT: [
            "Physical exercise is good",
            "My dog likes food",
            "Christmas is around the corner",
            "My cat is eating a lot",
            "A lion eats daily"
        ],
        
        dfc.LABEL: [
            "PE", "FOOD", "CHRISTMAS", "FOOD", "FOOD"
        ]
    })

    query = 'My cat eats daily'

    selector = CosineSelector()

    result = selector.get_n_most_similar_datapoints(
        query=query,
        data=data,
        n=2,
        include_score=True
    )

    console.print(result)

    result = selector.get_most_similar_datapoints_for_n_classes(
        query=query,
        data=data,
        n_classes=2,
        n=2
    )

    console.print(result)