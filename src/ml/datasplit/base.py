from abc import abstractmethod
from pydantic import validate_call

import pandas as pd
from src.interface.base import BaseExecutionBlock
from src.util.validation import DataFrameValidator
from src.util.constants import DatasetColumn
from src.util.types import TripleDataFrameTuple, MLDataFrame

class BaseDatasplit(BaseExecutionBlock):

    @abstractmethod
    def _split_data(self, data: pd.DataFrame, random_seed: int, **kwargs) -> TripleDataFrameTuple:
        raise NotImplementedError()
    
    @validate_call(config={"arbitrary_types_allowed": True})
    def execute(self, data: pd.DataFrame, random_seed: int, **kwargs) -> dict:

        # check validity of arguments
        DataFrameValidator.assert_non_zero_dataframe(
            data=data,
            columns=[DatasetColumn.TEXT, DatasetColumn.LABEL],
            strict_columns=False
        )

        # execute main function
        triple_mlframe = self._split_data(data=data, random_seed=random_seed, **kwargs)

        for idx, el in enumerate(triple_mlframe):
            assert isinstance(el, MLDataFrame), f"Expected MLDataFrame, got {type(el)} for index {idx}"

        # update kwargs
        kwargs["data_train"], kwargs["data_valid"], kwargs["data_test"] = triple_mlframe

        return kwargs