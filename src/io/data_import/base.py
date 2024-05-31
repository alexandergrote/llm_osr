from abc import abstractmethod

import pandas as pd

from src.interface.base import BaseExecutionBlock
from src.util.constants import DatasetColumn
from src.util.validation import DataFrameValidator
from src.util.environment import PydanticEnvironment


class BaseDataset(BaseExecutionBlock):

    @abstractmethod
    def _load(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
    
    def execute(self, **kwargs) -> dict:

        # check validity of arguments
        # not necessary here because not output has been created yet
        
        # execute main function
        data = self._load(**kwargs)

        # get environment
        env = PydanticEnvironment.create_from_environment()

        if env.is_dev_mode():
            data = data.head(100)

        DataFrameValidator.assert_non_zero_dataframe(
            data=data, 
            n_rows=None, 
            columns=[DatasetColumn.TEXT, DatasetColumn.LABEL], 
            strict_columns=True
        )

        # update kwargs
        kwargs["data"] = data
        kwargs["all_classes"] = data[DatasetColumn.LABEL].unique()

        return kwargs