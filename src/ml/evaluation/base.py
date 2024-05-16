import numpy as np
from abc import ABC, abstractmethod
from pydantic.v1 import validate_arguments
from typing import Set

from src.util.types import MLPrediction



class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(
        self, y_pred: np.ndarray, y_true: np.ndarray, classes_in_training: Set, **kwargs
    ) -> dict:
        raise NotImplementedError("Method not implemented")

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def execute(
        self, prediction: MLPrediction, **kwargs
    ) -> dict:
        

        # get data
        y_pred: np.ndarray = prediction.y_pred.values
        y_true: np.ndarray = prediction.y_test.values

        unique_classes = prediction.classes_in_training

        # evaluate
        return self.evaluate(y_pred=y_pred, y_true=y_true, classes_in_training=unique_classes, **kwargs)