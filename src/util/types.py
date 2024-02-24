import pandas as pd
from typing import Tuple

from typing_extensions import Annotated
from pydantic import Field

# pydantic model fields
Percentage = Annotated[float, Field(ge=0, le=1)]

# dataframe types
DualDataFrameTuple = Tuple[pd.DataFrame, pd.DataFrame]
TripleDataFrameTuple = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]