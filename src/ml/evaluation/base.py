import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pydantic.v1 import validate_arguments
from typing import Set, Optional

from src.util.types import MLPrediction
from src.util.logger import console
from src.ml.evaluation.util.class_mapping import ClassMapper


class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(
        self, y_pred: np.ndarray, y_true: np.ndarray, classes_in_training: Set, unknown_scores: Optional[pd.Series], **kwargs
    ) -> dict:
        raise NotImplementedError("Method not implemented")

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def execute(
        self, prediction: MLPrediction, **kwargs
    ) -> dict:
        
        all_classes = kwargs['all_classes']

        # make sure every class is lowercase
        all_classes = [c.lower() for c in all_classes]

        # make sure predictios are lower case as well
        if isinstance(prediction.y_pred, pd.Series):
            prediction.y_pred = prediction.y_pred.str.lower()
        
        # make sure truth is lower case
        if isinstance(prediction.y_test, pd.Series):
            prediction.y_test = prediction.y_test.str.lower()
        
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

        # check if both arrays contain numbers
        are_numbers = all([np.issubdtype(y_pred.dtype, np.number), np.issubdtype(y_true.dtype, np.number)])
        are_str = all([np.issubdtype(y_pred.dtype, np.str_), np.issubdtype(y_true.dtype, np.str_)])

        if not are_numbers and not are_str:

            msg: str = """
            Both, y_pred and y_true, must be consistently typed. 
            They must be either of type number or string
            """.strip()

            raise ValueError(msg)

        # evaluate
        result = self.evaluate(
            y_pred=y_pred, 
            y_true=y_true, 
            classes_in_training=unique_classes, 
            unknown_scores=prediction.outlier_score,
            **kwargs
        )

        console.log(result['metrics']['f1_avg'])

        return result