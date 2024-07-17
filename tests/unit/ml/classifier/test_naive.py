import unittest
import numpy as np
from src.ml.classifier.benchmark.naive import NaiveClf


class TestNaiveClf(unittest.TestCase):
    
    def setUp(self):
        self.clf = NaiveClf()
        self.x_train = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_train = np.array([0, 1, 0])
        self.x_valid = np.array([[7, 8], [9, 10]])
        self.y_valid = np.array([1, 0])
        self.x_test = np.array([[7, 8], [9, 10]])

    def test_fit(self):
        self.clf.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)
        self.assertTrue(np.array_equal(self.clf.y_train, self.y_train))
        self.assertTrue(np.array_equal(self.clf.y_valid, self.y_valid))

    def test_predict(self):
        
        with self.assertRaises(ValueError):
            self.clf.predict(self.x_test)

        self.clf.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)

        y_pred = self.clf.predict(self.x_test)
        self.assertEqual(len(y_pred), len(self.x_test))
        self.assertTrue(np.array_equal(y_pred, np.array([0, 0])))


if __name__ == '__main__':
    unittest.main()
