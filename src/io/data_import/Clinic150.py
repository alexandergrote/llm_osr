from pathlib import Path
from typing import Literal
import pandas as pd
import os

from pydantic import BaseModel, field_validator, model_validator

from src.util.constants import DatasetColumn, Directory

from .base import BaseDataset
# https://huggingface.co/datasets/clinc_oos/tree/main


class Clinic150Dataset(BaseDataset, BaseModel):
    data_home: Path = Directory.INPUT_DIR / "Clinic150"
    subset: Literal['train', 'test', 'all'] = 'all'
    kind: Literal['imbalanced', 'small', 'plus']


    data_end_train: str = "train.parquet"
    data_end_test: str = "test.parquet"
    data_end_val: str = "validation.parquet"

    @model_validator(mode='after')
    def filepath(self):

        kind = self.kind
        
        self.data_end_train = os.path.join(kind, self.data_end_train)
        self.data_end_test = os.path.join(kind, self.data_end_test)
        self.data_end_val = os.path.join(kind, self.data_end_val)

        
    @classmethod
    @field_validator('data_home')
    def _init_component(cls, value):
        value.mkdir(parents=True, exist_ok=True)
        return value
    
    def _load(self, **kwargs) -> pd.DataFrame:
        
        # fetch data
        # Load both train and test data and concatenate them
        train_data = pd.read_parquet(os.path.join(self.data_home,self.data_end_train))
        test_data = pd.read_parquet(os.path.join(self.data_home,self.data_end_test))
        validation_data = pd.read_parquet(os.path.join(self.data_home,self.data_end_val))
        data = pd.concat([train_data, test_data, validation_data], ignore_index=True)

        data = pd.DataFrame({
            DatasetColumn.TEXT: data.iloc[:,0], 
            DatasetColumn.LABEL: data.iloc[:,1]
        })
  

        return data