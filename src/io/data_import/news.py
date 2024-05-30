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
        news_data = fetch_20newsgroups(
            data_home=self.data_home,
            subset=self.subset,
            return_X_y=False,
            random_state=kwargs['random_seed'],
            remove=('headers', 'footers', 'quotes')
        )

        X = news_data.data
        y = [news_data.target_names[idx] for idx in news_data.target]

        data = pd.DataFrame({
            DatasetColumn.TEXT: X, 
            DatasetColumn.LABEL: y,
        })

        return data