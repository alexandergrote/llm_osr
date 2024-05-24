from abc import abstractmethod
from pydantic.v1 import validate_arguments
from pathlib import Path

import pandas as pd
from src.interface.base import BaseExecutionBlock
from src.util.types import MLDataFrame
from src.util.caching import PickleCacheHandler

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

        filename = ' '.join([str(el.hash()) for el in [data_train, data_valid, data_test]]) + '.pkl'
        filepath = Path(self.__class__.__name__) / filename

        cache_handler = PickleCacheHandler(
            filepath=filepath
        )

        # load cache
        cache = cache_handler.read()

        if cache is not None:
            return cache

        # execute main function
        data_train = self._fit_transform(data=data_train.data, step='train', **kwargs)
        data_valid = self._transform(data=data_valid.data, step='valid', **kwargs)
        data_test = self._transform(data=data_test.data, step='test', **kwargs)

        # update kwargs
        kwargs["data_train"] = MLDataFrame.from_pandas_dataframe(data_train)
        kwargs["data_valid"] = MLDataFrame.from_pandas_dataframe(data_valid)
        kwargs["data_test"] = MLDataFrame.from_pandas_dataframe(data_test)

        # store results
        cache_handler.write(kwargs)

        return kwargs