import numpy as np

from abc import abstractmethod
from tqdm import tqdm
from pydantic import BaseModel
from pydantic.config import ConfigDict
from numpy import ndarray
from typing import Optional, Union, Tuple

from src.util.logger import console
from src.util.constants import ErrorValues
from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.llm.util.logprob import LogProbScore


class AbstractClassifierLLM(BaseModel, BaseClassifier):

    # placeholders, values will be set later
    x_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, **kwargs):
        raise NotImplementedError("Method must be implemented in subclass")

    @abstractmethod
    def _single_predict(self, text: str) -> Optional[LogProbScore]:
        raise NotImplementedError("Method must be implemented in subclass")
    
    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        # assert if array is 2D
        if len(x.shape) != 2:
            raise ValueError("Input should be 2D array")

        result_text_list= []
        result_score_list = []    

        for el in tqdm(x, desc="LLM Prediction"):

            el_str = el[0]

            try:

                result = self._single_predict(text=el_str)

                result_text = ErrorValues.PARSING_STR.value
                result_score = 0.0

                if result is not None:

                    result_text = result.answer.label
                    result_score = result.unknown_score

                result_text_list.append(result_text)
                result_score_list.append(result_score)

            except Exception as e:

                console.log(f"Error processing text: {el_str}")

                raise e
            
        if include_outlierscore:
            return np.array(result_text_list), np.array(result_score_list)
        
        return np.array(result_text_list)
    
    def predict_proba(self, x: ndarray, **kwargs) -> ndarray:
        pass