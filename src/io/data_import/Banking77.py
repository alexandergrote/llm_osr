from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, field_validator

from src.util.constants import DatasetColumn, Directory

from .base import BaseDataset

# https://huggingface.co/datasets/banking77/tree/main


class BankingDataset(BaseDataset, BaseModel):

    data_home: Path = Directory.INPUT_DIR / "Banking77"
    subset: Literal['train', 'test', 'all'] = 'all'

    @field_validator('data_home')
    def _init_component(cls, value):
        value.mkdir(parents=True, exist_ok=True)
        return value

    def _load(self, **kwargs) -> pd.DataFrame:

        # fetch data
        # Load both train and test data and concatenate them
        train_data = pd.read_parquet(self.data_home / "train.parquet")
        test_data = pd.read_parquet(self.data_home / "test.parquet")
        data = pd.concat([train_data, test_data], ignore_index=True)

        # dataframe columns should be named text and label
        data = pd.DataFrame({
            DatasetColumn.TEXT: data.iloc[:, 0], 
            DatasetColumn.LABEL: data.iloc[:, 1]
        })
  

        return data