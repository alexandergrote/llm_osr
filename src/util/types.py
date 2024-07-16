import pandas as pd
import numpy as np
from typing import Tuple

from typing_extensions import Annotated
from pydantic import Field, BaseModel, model_validator, StrictStr
from typing import Optional, Set

from src.util.constants import DatasetColumn
from src.util.hashing import Hash

# pydantic model fields
Percentage = Annotated[float, Field(ge=0, le=1)]


class LogProb(BaseModel):
    text: str
    logprob: float


# custom dataframe
class MLDataFrame(BaseModel):

    data: pd.DataFrame

    text_column: str
    feature_column: Optional[str]
    target_column: str

    class Config:
        arbitrary_types_allowed = True


    @model_validator(mode='after')
    def _check_init(self):

        columns = list(filter(None,[self.text_column, self.feature_column, self.target_column]))
        
        assert len(self.data) > 0, "Data is empty"

        for col in columns:
            assert col in self.data.columns, f"Column {col} is not in data"

        assert not self.data[self.target_column].isnull().any(), "Target column contains NaN values"
        assert not self.data[self.text_column].isnull().any().any(), "Text column contain NaN values"

        if self.feature_column:
            assert not self.data[self.feature_column].isnull().any().any(), "Feature column contain NaN values"

        # target column must be of type int
        assert pd.api.types.is_object_dtype(self.data[self.target_column].dtype), "Target column must be of type obj"

    def hash(self) -> str:

        hash_list = [str(Hash.hash(v)) for _, v in self.model_dump().items()]

        return Hash.hash(' '.join(hash_list))

    def features(self) -> np.ndarray:

        if self.feature_column is None:
            raise ValueError("No feature column provided")

        array = np.vstack(self.data[self.feature_column].values)

        assert array.shape[0] == len(self.data), "Feature column must be a 2D array"
        assert array.shape[1] > 0, "Feature column must have at least one feature"
        assert len(array.shape) == 2, "Feature column must be a 2D array"
        
        return array
    
    def target(self) -> np.ndarray:

        array = self.data[self.target_column].values

        assert len(array) == len(self.data), "Target column must be a 1D array"
        assert array.shape[0] > 0, "Target column must have at least one target"
        assert len(array.shape) == 1, "Target column must be a 1D array"

        return array
    
    @classmethod
    def from_raw_pandas_dataframe(cls, data: pd.DataFrame) -> "MLDataFrame":
        return cls(text_column=DatasetColumn.TEXT, feature_column=None, target_column=DatasetColumn.LABEL, data=data)

    @classmethod
    def from_pandas_dataframe(cls, data: pd.DataFrame) -> "MLDataFrame":
        return cls(text_column=DatasetColumn.TEXT, feature_column=DatasetColumn.FEATURES, target_column=DatasetColumn.LABEL, data=data)


class MLPrediction(BaseModel):

    y_pred: pd.Series
    y_test: pd.Series

    classes_in_training: Set[StrictStr]  # set of classes in training, needed for evaluation


    class Config:
        arbitrary_types_allowed = True

    
    @model_validator(mode='before')
    def _escape_numpy_dtype(data):

        # convert numpy.int32 in set to int
        key = 'classes_in_training'
        classes = set()

        for el in data[key]:
            classes.add(el)

        data[key] = classes

        return data

    @model_validator(mode='after')
    def _check_init(self):
        
        # check lengths
        assert len(self.y_pred) == len(self.y_test), "Length of prediction and test set do not match"

        # check types
        assert pd.api.types.is_object_dtype(self.y_pred.dtype), "Prediction must be of type int"
        assert pd.api.types.is_object_dtype(self.y_test.dtype), "Test set must be of type int"

        # check for NaN values
        assert not self.y_pred.isnull().any(), "Prediction contains NaN values"
        assert not self.y_test.isnull().any(), "Test set contains NaN values"

        # check for classes
        assert len(self.classes_in_training) > 0, "Classes in training must be provided"
        

# dataframe types
DualDataFrameTuple = Tuple[MLDataFrame, MLDataFrame]
TripleDataFrameTuple = Tuple[MLDataFrame, MLDataFrame, MLDataFrame]
