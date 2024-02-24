from abc import abstractmethod
from typing import Dict
from pydantic import validate_call

import pandas as pd
from src.interface.base import BaseExecutionBlock
from src.util.validation import DataFrameValidator
from src.util.constants import DatasetColumn
from src.util.types import TripleDataFrameTuple

class BaseDatasplit(BaseExecutionBlock):

    @abstractmethod
    def _split_data(self, data: pd.DataFrame, **kwargs) -> TripleDataFrameTuple:
        raise NotImplementedError()
    
    @validate_call(config={"arbitrary_types_allowed": True})
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.DataFrame]:

        # check validity of arguments
        DataFrameValidator.assert_non_zero_dataframe(
            data=data,
            columns=[DatasetColumn.TEXT, DatasetColumn.LABEL]
        )

        # execute main function
        data_train, data_valid, data_test = self._split_data(data=data, **kwargs)

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

        # update kwargs
        kwargs["data_train"] = data_train
        kwargs["data_valid"] = data_valid
        kwargs["data_test"] = data_test

        return kwargs