from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, field_validator

from src.util.constants import DatasetColumn, Directory
#https://www.kaggle.com/datasets/soumikrakshit/yahoo-answers-dataset?resource=download

# custom code
from .base import BaseDataset

class YahooAnswersDataset(BaseDataset, BaseModel):
    data_home: Path = Directory.INPUT_DIR / "yahoo"
    subset: Literal['train', 'test', 'all'] = 'all'

    @field_validator('data_home')
    def _init_component(cls, value):
        value.mkdir(parents=True, exist_ok=True)
        return value
    
    def _load(self, **kwargs) -> pd.DataFrame:
        # fetch data
        # Load both train and test data and concatenate them
        train_data = pd.read_csv(self.data_home / "train.csv", header=None)
        test_data = pd.read_csv(self.data_home / "test.csv", header=None)
        data = pd.concat([train_data, test_data], ignore_index=True)

        data = pd.DataFrame({
            DatasetColumn.TEXT: data.iloc[:,3], 
            DatasetColumn.LABEL: data.iloc[:,0]
        })
  
        return data