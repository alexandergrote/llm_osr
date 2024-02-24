import pandas as pd
from pydantic import BaseModel, validate_call

from src.ml.preprocessing.base import BasePreprocessor
from src.util.constants import DatasetColumn


class IdentityPreprocessor(BaseModel, BasePreprocessor):

    @validate_call(config={"arbitrary_types_allowed": True})
    def _fit(self, data: pd.DataFrame, **kwargs):
        return
        
    @validate_call(config={"arbitrary_types_allowed": True})
    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        
        # work on a copy
        data_copy = data.copy()

        # add features column
        data_copy[DatasetColumn.FEATURES] = data_copy[DatasetColumn.TEXT]
        
        return data_copy


    @validate_call(config={"arbitrary_types_allowed": True})
    def _fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        
        # work on a copy
        data_copy = data.copy()

        # fit
        self._fit(data=data_copy, **kwargs)

        # add features column
        data_copy = self._transform(data=data_copy, **kwargs) 

        return data_copy