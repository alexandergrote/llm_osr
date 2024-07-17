import unittest
import numpy as np
from pathlib import Path

from .util import Data

from src.ml.classifier.benchmark.doc import DOC, GaussianModels
from src.util.dynamic_import import DynamicImport
from src.util.constants import Directory


class TestDOC(unittest.TestCase):

    def setUp(self):
        
        # load yaml config
        config_classifier_dir = Directory.CONFIG / 'ml__classifier'

        self.clf: DOC = DynamicImport.init_class_from_yaml(
            filename=config_classifier_dir / 'doc.yaml'
        )

        self.clf.epochs = 1

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
