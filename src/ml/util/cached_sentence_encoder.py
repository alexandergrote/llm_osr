import numpy as np
import warnings
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import Optional

from src.util.caching import PickleCacheHandler


class CachedSentenceEncoder(BaseModel):

    embedding_model_name: str
    embedding_model_params: Optional[dict] = None

    _model: Optional[SentenceTransformer] = None
    _cache_handler: Optional[PickleCacheHandler] = None
    _memory: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def cache_handler(self) -> PickleCacheHandler:

        if self._cache_handler is None:

            # keep track of processed data in dict
            filename = Path(self.__class__.__name__) / f"{self.embedding_model_name}.pkl"

            cache_handler = PickleCacheHandler(
                filepath=filename
            )

            self._cache_handler = cache_handler

        return self._cache_handler
    
    @property
    def model(self) -> SentenceTransformer:

        if self._model is None:

            # load sentence transformer model
            with warnings.catch_warnings():
                
                warnings.simplefilter(action='ignore', category=FutureWarning)

                embedding_params: dict = self.embedding_model_params or {}
                
                self._model = SentenceTransformer(
                    model_name_or_path=self.embedding_model_name,
                    **embedding_params
                )
            
        return self._model

    @property
    def memory(self) -> dict:

        # try loading from cache if memory is not set
        if self._memory is None:
            self._memory = self.cache_handler.read()

        if self._memory is None:
            self._memory = {}

        assert isinstance(self._memory, dict), "Memory must be a dict"

        return self._memory
    
    @memory.setter
    def memory(self, value):
        self._memory = value

    def encode(self, text: str) -> np.ndarray:


        if text in self.memory:

            cached_result = self.memory[text]

            assert isinstance(cached_result, np.ndarray)

            return cached_result
        
        embedding: np.ndarray = self.model.encode(text)

        # update in memory
        self.memory[text] = embedding
        
        # update cache
        self.cache_handler.write(self.memory)

        return self.memory[text]


if __name__ == '__main__':

    embedding_model_name: str = 'paraphrase-MiniLM-L6-v2'

    model = CachedSentenceEncoder(
        embedding_model_name=embedding_model_name,
        embedding_model_params=None
    )

    query = 'My cat eats daily'

    print(model.encode(query))