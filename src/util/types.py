import pandas as pd
import numpy as np
from typing import Tuple
from enum import Enum

from typing_extensions import Annotated
from pydantic import Field, BaseModel, model_validator, StrictStr
from pydantic.config import ConfigDict
from typing import Optional, Set
from pathlib import Path

from src.util.constants import DatasetColumn
from src.util.hashing import Hash

# pydantic model fields
Percentage = Annotated[float, Field(ge=0, le=1)]


class LogProb(BaseModel):
    text: str
    logprob: float

    @classmethod
    def from_prob(cls, text: str, prob: float) -> 'LogProb':
        return cls(text=text, logprob=np.log(prob))


# custom dataframe
class MLDataFrame(BaseModel):

    data: pd.DataFrame

    text_column: str
    feature_column: Optional[str]
    target_column: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


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

        return self

    def hash(self) -> str:

        hash_list = [str(Hash.hash(v)) for _, v in self.model_dump().items()]

        return Hash.hash(' '.join(hash_list))

    def raw_text(self) -> np.ndarray:

        array = self.data[self.text_column].values.reshape(-1)

        assert isinstance(array, np.ndarray), "Text column must be of type ndarray"
        assert isinstance(array[0], str), "Text column must contain strings"
        assert len(array) == len(self.data), "Text column must contain the same number of elements as the data"
        assert not self.data[self.text_column].isnull().any(), "Text column contain NaN values"

        return array

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

    def save(self, filename: str):
        """Save the dataframe to a CSV file."""
        self.data.to_csv(filename, index=False)

    @classmethod
    def load(cls, filename: str) -> "MLDataFrame":
        """Load the dataframe from a CSV file."""
        data = pd.read_csv(filename)

        if DatasetColumn.FEATURES in data.columns:
            return cls.from_pandas_dataframe(data)

        return cls.from_raw_pandas_dataframe(data)
    
    @classmethod
    def from_raw_pandas_dataframe(cls, data: pd.DataFrame) -> "MLDataFrame":
        return cls(text_column=DatasetColumn.TEXT, feature_column=None, target_column=DatasetColumn.LABEL, data=data)

    @classmethod
    def from_pandas_dataframe(cls, data: pd.DataFrame) -> "MLDataFrame":
        return cls(text_column=DatasetColumn.TEXT, feature_column=DatasetColumn.FEATURES, target_column=DatasetColumn.LABEL, data=data)


# Define an Enum for filenames
class MLPredictionFiles(Enum):
    Y_PRED = 'y_pred.csv'
    Y_TEST = 'y_test.csv'
    CLASSES_IN_TRAINING = 'classes_in_training.txt'
    OUTLIER_SCORE = 'outlier_score.csv'


class MLPrediction(BaseModel):

    y_pred: pd.Series
    y_test: pd.Series

    classes_in_training: Set[StrictStr]  # set of classes in training, needed for evaluation
    outlier_score: Optional[pd.Series] = None  # score for being unknown, that is not being reflected in training data

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
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
        assert pd.api.types.is_object_dtype(self.y_pred.dtype), "Prediction must be of type str"
        assert pd.api.types.is_object_dtype(self.y_test.dtype), "Test set must be of type str"

        # check for NaN values
        assert not self.y_pred.isnull().any(), "Prediction contains NaN values"
        assert not self.y_test.isnull().any(), "Test set contains NaN values"

        # check outlier scores if given
        if self.outlier_score is not None:
            assert len(self.outlier_score) == len(self.y_pred), "Length of prediction and outlier scores do not match"
            #assert pd.api.types.is_float_dtype(self.outlier_score.dtype), "Outlier scores must be of type float"
            assert not self.outlier_score.isnull().any(), "Outlier scores contain NaN values"

        # check for classes
        assert len(self.classes_in_training) > 0, "Classes in training must be provided"

        return self

    def error(self) -> pd.Series:
        return self.y_pred == self.y_test

    def save(self, directory: Path):

        if not directory.exists():
            directory.mkdir(parents=True)

        y_pred_path = directory / MLPredictionFiles.Y_PRED.value
        self.y_pred.to_csv(y_pred_path, index=False)

        y_test_path = directory / MLPredictionFiles.Y_TEST.value
        self.y_test.to_csv(y_test_path, index=False)

        classes_in_training_path = directory / MLPredictionFiles.CLASSES_IN_TRAINING.value
        with open(classes_in_training_path, 'w') as f:
            for cls in self.classes_in_training:
                f.write(f'{cls}\n')

        if self.outlier_score is not None:
            outlier_score_path = directory / MLPredictionFiles.OUTLIER_SCORE.value
            self.outlier_score.to_csv(outlier_score_path, index=False)

    @classmethod
    def load(cls, directory: Path) -> "MLPrediction":
        
        y_pred_path = directory / MLPredictionFiles.Y_PRED.value
        y_test_path = directory / MLPredictionFiles.Y_TEST.value
        classes_in_training_path = directory / MLPredictionFiles.CLASSES_IN_TRAINING.value
        outlier_score_path = directory / MLPredictionFiles.OUTLIER_SCORE.value

        # Load y_pred and y_test
        y_pred = pd.read_csv(y_pred_path, header=0).squeeze().astype(str)
        y_test = pd.read_csv(y_test_path, header=0).squeeze().astype(str)

        # Load classes_in_training
        with open(classes_in_training_path, 'r') as f:
            classes_in_training = {line.strip() for line in f}

        # Load outlier_score if it exists
        outlier_score = None
        if outlier_score_path.exists():
            outlier_score = pd.read_csv(outlier_score_path, header=0).squeeze()
            outlier_score = pd.to_numeric(outlier_score)

        return cls(y_pred=y_pred, y_test=y_test, classes_in_training=classes_in_training, outlier_score=outlier_score)


# dataframe types
DualDataFrameTuple = Tuple[MLDataFrame, MLDataFrame]
TripleDataFrameTuple = Tuple[MLDataFrame, MLDataFrame, MLDataFrame]
