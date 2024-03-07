from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, model_validator
from sklearn.datasets import fetch_20newsgroups

from src.util.constants import DatasetColumn, Directory

# custom code
from .base import BaseDataset


class NewsDataset(BaseDataset, BaseModel):

    data_home: Path = Directory.INPUT_DIR / "news"
    subset: Literal['train', 'test', 'all'] = 'all'

    @model_validator(mode='after')
    def init_data_dir(self):
        self.data_home.mkdir(parents=True, exist_ok=True)

    def _load(self, **kwargs) -> pd.DataFrame:

        # fetch data
        X, y = fetch_20newsgroups(
            data_home=self.data_home,
            subset=self.subset,
            return_X_y=True,
            random_state=kwargs['random_seed']
        )

        data = pd.DataFrame({
            DatasetColumn.TEXT: X, 
            DatasetColumn.LABEL: y
        })

        return data