import unittest
import sys

from tests.unit.ml.classifier.llm.helper import data_train, data_valid

from src.util.constants import DatasetColumn as dfc
from src.ml.classifier.benchmark.contrastnetwrapper import ContrastNetWrapper


class TestContrastNetWrapper(unittest.TestCase):
    
    def setUp(self):
        self.clf = ContrastNetWrapper()

        self.x_train = data_train[dfc.FEATURES].values
        self.y_train = data_train[dfc.LABEL].values
        self.x_valid = data_valid[dfc.FEATURES].values
        self.y_valid = data_valid[dfc.LABEL].values
        self.x_test = data_valid[dfc.FEATURES].values

    @unittest.skipIf(sys.platform.startswith("win"), "Test skipped on Windows")
    def test_fit(self):
        self.clf.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)

    @unittest.skipIf(sys.platform.startswith("win"), "Test skipped on Windows")    
    def test_predict(self):
        
        with self.assertRaises(ValueError):
            self.clf.predict(self.x_test)

        self.clf.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)

        y_pred = self.clf.predict(self.x_test)
        self.assertEqual(len(y_pred), len(self.x_test))

if __name__ == '__main__':
    unittest.main()
