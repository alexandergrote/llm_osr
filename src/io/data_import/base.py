from abc import abstractmethod
from typing import Optional

import pandas as pd

from src.interface.base import BaseExecutionBlock
from src.util.constants import DatasetColumn
from src.util.validation import DataFrameValidator


class BaseDataset(BaseExecutionBlock):

    @abstractmethod
    def _load(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
    
    def get_n_rows(self) -> Optional[int]:
        return None
    
    def execute(self, **kwargs) -> dict:

        # check validity of arguments
        # not necessary here because not output has been created yet
        
        # execute main function
        data = self._load(**kwargs)

        n_rows = self.get_n_rows()

        if n_rows is not None:
            data = data.head(n_rows)

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
    

class BaseResultDataset(BaseExecutionBlock):

    @abstractmethod
    def _load(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
    
    def execute(self, **kwargs) -> dict:

        assert "experiment_name" in kwargs, "experiment_name must be provided."

        kwargs['result_data'] = self._load(**kwargs)

        print("-"*25)
        print("Experiment:", kwargs["experiment_name"])
        print(kwargs['result_data'])

        return kwargs