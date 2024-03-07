from datasets import load_dataset
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, model_validator

from src.util.constants import DatasetColumn, Directory

from .base import BaseDataset

# https://huggingface.co/datasets/banking77/tree/main


class BankingDataset(BaseDataset, BaseModel):

    data_home: Path = Directory.INPUT_DIR / "Banking77"
    
    @model_validator(mode='after')
    def init_data_dir(self):
        self.data_home.mkdir(parents=True, exist_ok=True)
    

    def _load(self, **kwargs) -> pd.DataFrame:

        # fetch data
        # Load both train and test data and concatenate them

        filename = self.data_home / "banking77.parquet"

        if filename.exists():
            data = pd.read_parquet(filename)
            return data

        dataset = load_dataset("banking77")
        train_data = dataset["train"].to_pandas()
        test_data = dataset["test"].to_pandas()
        data = pd.concat([train_data, test_data], ignore_index=True)

        # dataframe columns should be named text and label
        data = pd.DataFrame({
            DatasetColumn.TEXT: data['text'], 
            DatasetColumn.LABEL: data['label']
        })

        data.to_parquet(filename)

        return data