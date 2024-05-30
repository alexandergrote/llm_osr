import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pydantic.v1 import validate_arguments
from typing import Set

from src.util.types import MLPrediction
from src.util.logging import console
from src.ml.evaluation.util.class_mapping import ClassMapper


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
        
        all_classes = kwargs['all_classes']
        
        mapper = ClassMapper()
        
        mapper.fit(data=pd.Series(all_classes))

        # get data
        y_pred: np.ndarray = mapper.transform(
            data=prediction.y_pred
        ).values

        y_true: np.ndarray = mapper.transform(
            data=prediction.y_test
        ).values

        unique_classes = set(mapper.transform(
            data=pd.Series(list(prediction.classes_in_training))
        ).values)

        # evaluate
        result = self.evaluate(y_pred=y_pred, y_true=y_true, classes_in_training=unique_classes, **kwargs)

        console.log(result['f1_avg'])

        return result