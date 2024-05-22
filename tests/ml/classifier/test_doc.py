import unittest
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.datasets import fetch_20newsgroups_vectorized

from src.ml.classifier.doc import DOC, GaussianModels
from src.util.dynamic_import import DynamicImport
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


class TestDOC(unittest.TestCase):

    def setUp(self):
        
        # load yaml config
        config_classifier_dir = Directory.CONFIG / 'ml__classifier'

        self.clf: DOC = DynamicImport.init_class_from_yaml(
            filename=config_classifier_dir / 'doc.yaml'
        )

        self.x, self.y = Data.get_data()
        
    def test_fit(self):
        
        self.clf.fit(x_train=self.x, y_train=self.y, x_valid=self.x, y_valid=self.y)
        
        self.assertIsNotNone(self.clf.model)
        self.assertIsInstance(self.clf.gaussian_models, GaussianModels)

    def test_predict(self):
        
        self.clf.fit(x_train=self.x, y_train=self.y, x_valid=self.x, y_valid=self.y)
        y_pred = self.clf.predict(x=self.x)

        self.assertEqual(len(y_pred), len(self.y))
        self.assertIsInstance(y_pred, np.ndarray)

    def tearDown(self):
        
        for file in [self.clf.filename_checkpoint, self.clf.filename_loss]:
            if Path(file).exists():
                Path(file).unlink()

if __name__ == '__main__':
    unittest.main()
