import unittest
import numpy as np

from .util import Data

from src.ml.classifier.nn.cls.setfit import SetFit
from src.util.dynamic_import import DynamicImport
from src.util.constants import Directory


class TestSetFit(unittest.TestCase):

    def setUp(self):
        
        # load yaml config
        config_classifier_dir = Directory.CONFIG / 'ml__classifier'

        self.clf: SetFit = DynamicImport.init_class_from_yaml(
            filename=config_classifier_dir / 'setfit.yaml'
        )

        self.x, self.y = Data.get_data(processed=False, n_rows=10)

        
    def test_fit(self):
        
        self.clf.fit(x_train=self.x, y_train=self.y, x_valid=self.x, y_valid=self.y)
        
        self.assertIsNotNone(self.clf.model)

    def test_predict(self):
        
        self.clf.fit(x_train=self.x, y_train=self.y, x_valid=self.x, y_valid=self.y)
        y_pred = self.clf.predict(x=self.x)

        self.assertEqual(len(y_pred), len(self.y))
        self.assertIsInstance(y_pred, np.ndarray)


if __name__ == '__main__':
    unittest.main()
