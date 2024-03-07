import numpy as np
import pandas as pd
from abc import abstractmethod
from src.interface.base import BaseExecutionBlock
from pydantic import validate_call
from src.util.validation import DataFrameValidator
from src.util.constants import DatasetColumn



# gleiche wie bei preprocessing base -> def execute
class BaseClassifier(BaseExecutionBlock):

    @abstractmethod
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
    ):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray):
        pass

    @abstractmethod
    def predict_proba(self, x: np.ndarray):
        pass
    
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
                columns=[DatasetColumn.TEXT, DatasetColumn.LABEL, DatasetColumn.FEATURES], 
                strict_columns=True, 
                identifier=name
            )
        
        # create x_train, x_valid, y_train, y_valid, x_test, y_test
        x_train = data_train[DatasetColumn.FEATURES].values
        y_train = data_train[DatasetColumn.LABEL].values
        x_valid = data_valid[DatasetColumn.FEATURES].values
        y_valid = data_valid[DatasetColumn.LABEL].values
        x_test = data_test[DatasetColumn.FEATURES].values
        y_test = data_test[DatasetColumn.LABEL].values
        
        # execute main function
        self.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
        y_pred = self.predict(x=x_test, **kwargs)



        return y_pred, y_test