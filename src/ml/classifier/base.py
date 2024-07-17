import numpy as np
import pandas as pd

from typing import Union, Tuple
from abc import abstractmethod
from pydantic.v1 import validate_arguments

from src.interface.base import BaseExecutionBlock
from src.util.types import MLDataFrame, MLPrediction


class BaseClassifier(BaseExecutionBlock):

    @abstractmethod
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        pass

    @abstractmethod
    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass
    
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def execute(self, data_train: MLDataFrame, data_valid: MLDataFrame, data_test: MLDataFrame, **kwargs) -> dict:
        
        # execute main function
        self.fit(
            x_train=data_train.features(), 
            y_train=data_train.target(), 
            x_valid=data_valid.features(),
            y_valid=data_valid.target(),
            **kwargs
        )
        
        y_pred, outlier_score = self.predict(
            x=data_test.features(),
            include_outlierscore=True,
            **kwargs
        )

        y_test = data_test.target()

        # store results
        kwargs["prediction"] = MLPrediction(
            y_pred=pd.Series(y_pred),
            y_test=pd.Series(y_test),
            classes_in_training=list(set(data_train.target())),
            outlier_score=pd.Series(outlier_score)
        )
        
        return kwargs