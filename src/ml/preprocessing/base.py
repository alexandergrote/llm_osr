from abc import abstractmethod
from pydantic.v1 import validate_arguments

import pandas as pd
from src.interface.base import BaseExecutionBlock
from src.util.types import MLDataFrame

class BasePreprocessor(BaseExecutionBlock):
    
    @abstractmethod
    def _fit(self, data: pd.DataFrame, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
    
    @abstractmethod
    def _fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
    
    @validate_arguments(config={'arbitrary_types_allowed': True})
    def execute(self, data_train: MLDataFrame, data_valid: MLDataFrame, data_test: MLDataFrame, **kwargs) -> dict:

        # execute main function
        data_train = self._fit_transform(data=data_train.data, **kwargs)
        data_valid = self._transform(data=data_valid.data, **kwargs)
        data_test = self._transform(data=data_test.data, **kwargs)

        # update kwargs
        kwargs["data_train"] = MLDataFrame.from_pandas_dataframe(data_train)
        kwargs["data_valid"] = MLDataFrame.from_pandas_dataframe(data_valid)
        kwargs["data_test"] = MLDataFrame.from_pandas_dataframe(data_test)

        return kwargs