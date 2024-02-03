from abc import abstractmethod

import pandas as pd

from src.interface.base import BaseExecutionBlock


class BaseDataset(BaseExecutionBlock):

    @abstractmethod
    def _load(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
    
    def execute(self, **kwargs) -> dict:

        # check validity of arguments
        # not necessary here because not output has been created yet
        
        # execute main function
        data = self._load(**kwargs)

        # check validity of output
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Output is not a pandas Dataframe.")
        
        
        assert isinstance(data, pd.DataFrame), "Data is not a dataframe."
        assert data.shape[0] > 0, "Output data is empty."
        assert len(set(data.columns) - set(["text", "label"])) == 0, "Output data does not contain the correct columns."

        # update kwargs
        kwargs["data"] = data

        return kwargs