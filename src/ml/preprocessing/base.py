from abc import abstractmethod
from pydantic import validate_call

import pandas as pd
from src.interface.base import BaseExecutionBlock
from src.util.validation import DataFrameValidator
from src.util.constants import DatasetColumn

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
    
    @validate_call(config={"arbitrary_types_allowed": True})
    def execute(self, data_train: pd.DataFrame, data_valid: pd.DataFrame, data_test: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # check validity of arguments

        # check validity of output
        options = [
            (data_train, "data_train"), 
            (data_valid, "data_valid"), 
            (data_test, "data_test")
        ]

        for data, name in options:

            DataFrameValidator.assert_non_zero_dataframe(
                data=data, 
                n_rows=None, 
                columns=[DatasetColumn.TEXT, DatasetColumn.LABEL], 
                strict_columns=True, 
                identifier=name
            )
        
        # execute main function
        data_train = self._fit_transform(data=data_train, **kwargs)
        data_valid = self._transform(data=data_valid, **kwargs)
        data_test = self._transform(data=data_test, **kwargs)

        # check validity of output
        DataFrameValidator.assert_non_zero_dataframe(
            data=data, 
            n_rows=None, 
            columns=[DatasetColumn.TEXT, DatasetColumn.LABEL, DatasetColumn.FEATURES], 
            strict_columns=True
        )

        # update kwargs
        kwargs["data_train"] = data_train
        kwargs["data_valid"] = data_valid
        kwargs["data_test"] = data_test

        return kwargs