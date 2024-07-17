import numpy as np
from pydantic import BaseModel
from typing import Optional, Union, Tuple

from src.ml.classifier.base import BaseClassifier

class NaiveClf(BaseClassifier, BaseModel):

    y_train: Optional[np.ndarray] = None  # will be set during fit
    y_valid: Optional[np.ndarray] = None  # will be set during fit

    class Config:
        arbitrary_types_allowed = True

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):

        self.y_train = y_train
        self.y_valid = y_valid

        return None

    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        if any([self.y_train is None, self.y_valid is None]):
            raise ValueError("Model has not been trained yet")

        labels = np.concatenate([self.y_train, self.y_valid])

        # get labels with most counts
        unique, counts = np.unique(labels, return_counts=True)

        # get most frequent label
        most_frequent_label = unique[np.argmax(counts)]

        y_pred = np.full(x.shape[0], most_frequent_label)
        
        if include_outlierscore:
            return y_pred, np.zeros(x.shape[0])

        return y_pred
        

    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise ValueError("Not implemented yet")