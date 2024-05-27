from abc import abstractmethod
from pydantic.v1 import validate_arguments
from pathlib import Path
from typing import Optional

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

    def _cache_result(self, data: MLDataFrame, step: str, function, **kwargs) -> MLDataFrame:

        filename = data.hash() + '.pkl'
        filepath = Path(self.__class__.__name__) / filename

        cache_handler = PickleCacheHandler(
            filepath=filepath
        )

        # load cache
        cache: Optional[MLDataFrame] = cache_handler.read()

        if cache is not None:
            return cache

        # execute main function
        result = function(data=data.data, step=step, **kwargs)
        result = MLDataFrame.from_pandas_dataframe(result)
        
        # store results
        cache_handler.write(result)

        return result
    
    @validate_arguments(config={'arbitrary_types_allowed': True})
    def execute(self, data_train: MLDataFrame, data_valid: MLDataFrame, data_test: MLDataFrame, **kwargs) -> dict:
        
        steps = [
            ('train', data_train, self._fit_transform),
            ('valid', data_valid, self._transform),
            ('test', data_test, self._transform)
        ]

        for step in steps:

            step_name, data, processing_function = step

            data = self._cache_result(data=data, step=step_name, function=processing_function, **kwargs)
            kwargs[f"data_{step_name}"] = data

        
        return kwargs