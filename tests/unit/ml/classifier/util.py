import os
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_20newsgroups

from src.util.constants import Directory, DatasetColumn
from src.util.caching import PickleCacheHandler

class Data:

    @staticmethod
    def get_data(processed: bool = True, n_rows: int = 100, n_cols: int = 100) -> Tuple[np.ndarray, np.ndarray]:

        cache = PickleCacheHandler(
            filepath=f'test_news_{str(processed)}_{n_rows}_{n_cols}.pkl'
        )

        data = cache.read()

        if data is not None:
            x, y = data
            return x, y
        
        args = {
            'subset': 'test',
            'data_home': Directory.INPUT_DIR
        }

        data_fun = fetch_20newsgroups_vectorized if processed else fetch_20newsgroups

        data = data_fun(**args)

        subset_x, subset_y = data.data[:n_rows], data.target[:n_rows]
        x = subset_x[:, :n_cols].toarray() if processed else subset_x
        y = subset_y.reshape(-1,)

        if not processed:
            x = np.array(x)

        cache.write((x, y))

        return x, y


class BANKING77Dataset:

    @staticmethod
    def get_data(n_rows: int = 100):
        parquet_file = Directory.INPUT_DIR / os.path.join("Banking77", "banking77.parquet")
        data = pd.read_parquet(parquet_file).sample(n_rows, random_state=42)
        return data[DatasetColumn.TEXT].values, data[DatasetColumn.LABEL].values
