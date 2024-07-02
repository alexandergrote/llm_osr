import numpy as np
from typing import Tuple
from sklearn.datasets import fetch_20newsgroups_vectorized

from src.util.constants import Directory
from src.util.caching import PickleCacheHandler

class Data:

    @staticmethod
    def get_data() -> Tuple[np.ndarray, np.ndarray]:

        cache = PickleCacheHandler(
            filepath='test_doc.pkl'
        )

        data = cache.read()

        if data is not None:
            x, y = data
            return x, y

        data = fetch_20newsgroups_vectorized(subset='test', data_home=Directory.INPUT_DIR)
        n_rows, n_cols = 100, 100
        subset_x, subset_y = data.data[:n_rows], data.target[:n_rows]
        x = subset_x[:, :n_cols].toarray()
        y = subset_y.reshape(-1,)

        cache.write((x, y))

        return x, y