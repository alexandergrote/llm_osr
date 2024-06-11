import numpy as np

from abc import abstractmethod
from tqdm import tqdm
from pathlib import Path
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, model_validator
from numpy import ndarray
from typing import Any, Optional

from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.llm.cls.util.prediction import Prediction
from src.ml.classifier.llm.util.mapping import LLM_Mapping
from src.util.constants import UnknownClassLabel, LLMModels
from src.util.caching import PickleCacheHandler
from src.util.logging import console



class BaseLLM(BaseModel, BaseClassifier):

    model: Any
    model_str: str

    # placeholders, values will be set later
    x_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None
    parser: Optional[Any] = None
    
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def _init_model(data: dict):

        data['model'] = LLM_Mapping[LLMModels(data['model_str'])]

        return data

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):

       self.y_train = np.concatenate([
           y_train, y_valid
       ]) 

       self.x_train = np.concatenate([
           x_train, x_valid
       ])

       self.classes = np.unique(self.y_train)

       # Example usage
       valid_labels = [UnknownClassLabel.UNKNOWN_STR.value] + list(self.classes)
       Prediction.valid_labels = valid_labels
       
       self.parser = PydanticOutputParser(pydantic_object=Prediction)

    @abstractmethod
    def _single_predict(self, text: str) -> str:
        raise NotImplementedError("Method must be implemented in subclass")

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:

        result_list = []

        cache_handler = PickleCacheHandler(
            filepath= Path(self.__class__.__name__) / f"{self.model_str}.pkl"
        )

        cache = cache_handler.read()

        if cache is None:
            cache = {}   

        assert isinstance(cache, dict), "cache must be a dictionary"     

        for el in tqdm(x, desc="LLM Prediction"):

            el_str = el[0]
            
            result = cache.get(el_str)

            if result is not None:
                result_list.append(result)
                continue

            try:

                result = self._single_predict(text=el_str)

            except Exception as e:

                console.log(f"Error processing text: {el_str}")

                # in case of an error write cache to not lose progress
                cache_handler.write(cache)

                raise e

        
            cache[el_str] = result

            result_list.append(result)

        cache_handler.write(cache)

        
        return np.array(result_list)
    
    def predict_proba(self, x: ndarray, **kwargs) -> ndarray:
        pass