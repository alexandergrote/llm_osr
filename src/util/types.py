import pandas as pd
from typing import Tuple

from typing_extensions import Annotated
from pydantic import Field, BaseModel, model_validator
from typing import List

# pydantic model fields
Percentage = Annotated[float, Field(ge=0, le=1)]

# custom dataframe
class DataFrame(BaseModel):

    columns: List[str]
    data: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _check_init(self):

        columns = self.columns
        data: pd.DataFrame = self.data

        assert len(columns) == len(data.columns), "Number of columns in columns and data do not match"
        assert all([col in data.columns for col in columns]), "Columns in columns and data do not match"
        assert len(data) > 0, "Data is empty"

    @classmethod
    def from_pandas_dataframe(cls, data: pd.DataFrame) -> "DataFrame":
        return cls(columns=data.columns.tolist(), data=data)
    

# dataframe types
DualDataFrameTuple = Tuple[pd.DataFrame, pd.DataFrame]
TripleDataFrameTuple = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
