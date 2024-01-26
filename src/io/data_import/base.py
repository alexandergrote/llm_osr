from abc import abstractmethod
import pandas as pd

from interface.base import BaseExecutionBlock


class BaseDataset(BaseExecutionBlock):

    @abstractmethod
    def _load(self, *args, **kwargs) -> dict:
        raise NotImplementedError()


class MixinDataset:
    
    def execute(self, **kwargs) -> dict:

        # check validity of arguments
        # not necessary here because not output has been created yet
        
        # execute main function
        output = self._load(**kwargs)

        # check validity of output
        if not isinstance(output, dict):
            raise TypeError("Output is not a dictionary.")
        
        assert "data" in output.keys(), "Output does not contain data."
        
        data = output["data"]
        assert isinstance(data, pd.DataFrame), "Data is not a dataframe."
        assert data.shape[0] > 0, "Output data is empty."
        assert len(set(data.columns) - set(["text", "label"])) == 0, "Output data does not contain the correct columns."

        return output