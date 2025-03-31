import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from typing import Literal, List, Optional

from src.ml.classifier.llm.util.rate_limit import RateLimitManager
from src.util.constants import DatasetColumn as dfc
from src.ml.classifier.llm.util.prompt import PromptExample
from src.util.logger import console

rlm = RateLimitManager.create_from_config_file(filename="hf.yaml")

class CosineSelector(BaseModel):

    mode: Literal["knn", "random_class", "knn_class"] = "knn"
    n_datapoints: Optional[int] = 5
    n_classes: Optional[int] = 3

    _url: str = "https://api-inference.huggingface.co/models/mixedbread-ai/mxbai-embed-large-v1"
    _key: str = 'cosine_score'
    _rate_limit_manager: RateLimitManager = rlm
    _query: str = 'input'
    _answer: str = 'output'
    

    def get_n_most_similar_datapoints(self, query: str, data: pd.DataFrame, n: int, include_score: bool = False) -> pd.DataFrame:

        # to avoid circular imports
        from src.ml.preprocessing.rest_embedding import HFEmbeddingPreprocessor

        embedder: HFEmbeddingPreprocessor = HFEmbeddingPreprocessor(
            url=self._url,
            tqdm_disable=True,
            rate_limit_manager=self._rate_limit_manager,
            use_cache=True
        )

        # work on copy
        data_copy = data.copy()

        embeddings_df = embedder._fit_transform(data=data_copy)
        embeddings = np.stack(embeddings_df[dfc.FEATURES].to_list())

        query_embedding_df = embedder._transform(data=pd.DataFrame({dfc.TEXT: [query]}))
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

    def _select_random(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> List[PromptExample]:

        # draw a random example from each class
        examples = []

        rng = np.random.default_rng(42)

        classes = np.unique(y_train)

        for selected_class in classes:

            y_mask = y_train == selected_class

            x_sub = x_train[y_mask]

            x_chosen = rng.choice(x_sub)

            examples.append(PromptExample(text=x_chosen, label=selected_class))

        return examples

    def _select_knn(self, text: str, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> List[PromptExample]:

        examples = []

        classes = np.unique(y_train)

        data = pd.DataFrame({
            dfc.LABEL: y_train,
            dfc.TEXT: x_train.reshape(-1,)
        })

        n_classes = len(classes) if self.n_classes is None else self.n_classes
        n_datapoints = 1 if self.n_datapoints is None else self.n_datapoints

        if self.mode == 'knn_class':

            dataframe = self.get_most_similar_datapoints_for_n_classes(
                query=text,
                data=data,
                n_classes=n_classes,
                n=n_datapoints
            )
        else:

            dataframe = self.get_n_most_similar_datapoints(
                query=text,
                data=data,
                n=n_datapoints
            )

        for _, row in dataframe.iterrows():
            examples.append(PromptExample(text=row[dfc.TEXT], label=row[dfc.LABEL]))

        return examples
        
    def get_examples(self, text: str, x_train: np.ndarray, y_train: np.ndarray, shuffle: bool = True, **kwargs) -> List[PromptExample]:

        result = None

        if self.mode in ['knn', 'knn_class']:
            result = self._select_knn(text, x_train, y_train)
        elif self.mode == "random_class":
            result = self._select_random(x_train, y_train, **kwargs)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        
        if not isinstance(result, list):
            raise ValueError(f"Expected result to be a list, got {type(result)}")

        if shuffle:

            rng = np.random.default_rng(
                seed=42
            )
            rng.shuffle(result)

        return result
        

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

    result = selector.get_examples(text=query, x_train=data[dfc.TEXT].values, y_train=data[dfc.LABEL].values)

    console.print(result)

    selector = CosineSelector(
        mode="knn_class",
        n_classes=2
    )

    result = selector.get_examples(text=query, x_train=data[dfc.TEXT].values, y_train=data[dfc.LABEL].values)

    console.print(result)
