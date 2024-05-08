import pandas as pd
from typing import Tuple

from typing_extensions import Annotated
from pydantic import Field, BaseModel, model_validator
from typing import Optional

from src.util.constants import DatasetColumn

# pydantic model fields
Percentage = Annotated[float, Field(ge=0, le=1)]

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
        assert self.data[self.target_column].dtype == int, "Target column must be of type int"


    def features(self) -> pd.DataFrame:
        return self.data[self.feature_columns]
    
    def target(self) -> pd.Series:
        return self.data[self.target_column]
    
    @classmethod
    def from_raw_pandas_dataframe(cls, data: pd.DataFrame) -> "MLDataFrame":
        return cls(text_column=DatasetColumn.TEXT, feature_column=None, target_column=DatasetColumn.LABEL, data=data)

    @classmethod
    def from_pandas_dataframe(cls, data: pd.DataFrame) -> "MLDataFrame":
        return cls(text_column=DatasetColumn.TEXT, feature_column=DatasetColumn.FEATURES, target_column=DatasetColumn.LABEL, data=data)


MLDataFrameDtype = Annotated[MLDataFrame, Field()]
    

# dataframe types
DualDataFrameTuple = Tuple[MLDataFrame, MLDataFrame]
TripleDataFrameTuple = Tuple[MLDataFrame, MLDataFrame, MLDataFrame]
