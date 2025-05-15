import pandera as pa
import pandas as pd

from pydantic import BaseModel, ConfigDict
from pandera import Field
from pandera.typing import DataFrame
from typing import Optional


class MetricData(pa.DataFrameModel):
    dataset: str = Field()
    Openness: float = Field(coerce=True)
    Model: str = Field()
    mean: float = Field(ge=0, le=1)
    std: Optional[float] = Field(nullable=True)

    @pa.check(name="unique_except_mean_std", element_wise=False)
    def unique_except_mean_std(cls, df: pd.DataFrame) -> bool:
        # Check uniqueness of dataset + Openness + Model
        return df[["dataset", "Openness", "Model"]].drop_duplicates().shape[0] == df.shape[0]


class F1ScoreTable(BaseModel):
    data: DataFrame[MetricData]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def print(self, **kwargs) -> None:
        data_copy = self.data.copy()
        mean_str = data_copy[MetricData.mean].round(4).astype(str)
        std_str = data_copy[MetricData.std].round(4).astype(str)
        data_copy['metric'] = mean_str + " ± " + std_str
        data_copy_pivot = data_copy.pivot_table(index=[MetricData.Openness, MetricData.Model], columns=[MetricData.dataset], values='metric', aggfunc="first")
        print(data_copy_pivot.to_latex())