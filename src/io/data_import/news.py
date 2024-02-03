import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Literal

# custom code
from .base import BaseDataset, MixinDataset
from src.util.constants import Directory, DatasetColumn

class NewsDataset(MixinDataset, BaseDataset, BaseModel):

    data_home: Path = Directory.INPUT_DIR / "news"
    subset: Literal['train', 'test', 'all'] = 'all'

    @field_validator('data_home')
    def _init_component(cls, value):
        value.mkdir(parents=True, exist_ok=True)
        return value

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