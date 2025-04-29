import unittest
import sys

import numpy as np

from sklearn.model_selection import train_test_split

from .util import BANKING77Dataset

from src.ml.classifier.benchmark.contrastnetwrapper import ContrastNetWrapper


class TestContrastNetWrapper(unittest.TestCase):
    
    def setUp(self):
        
        self.clf = ContrastNetWrapper(
            max_iter=1000,
            evaluate_every=10,
            n_test_episodes=20,
            patience=20,
        )

        self.n_train = 800
        self.n_valid = 800
        self.n_test = 500

        x, y = BANKING77Dataset.get_data(n_rows=self.n_train + self.n_test + self.n_valid)

        # convert numpy arrays to strings
        x = np.array([str(i) for i in x])
        y = np.array([str(i) for i in y])

        # split data into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.n_test, random_state=42)
        self.x_train, self.x_valid, self.y_train, self.y_valid  = train_test_split(self.x_train, self.y_train, test_size=self.n_valid, random_state=42)
        

    @unittest.skipIf(sys.platform.startswith("win"), "Test skipped on Windows")
    def test_fit(self):
        self.clf.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)

    @unittest.skipIf(sys.platform.startswith("win"), "Test skipped on Windows")
    def test_predict(self):

        with self.assertRaises(ValueError):
            self.clf.predict(self.x_test)

        self.clf.fit(
            self.x_train, self.y_train, 
            self.x_valid, self.y_valid
        )

        y_pred, y_pred_proba = self.clf._predict(self.x_test)
       
        self.assertEqual(len(y_pred), self.n_test)
        self.assertEqual(len(y_pred_proba), self.n_test)

        # accuracy:
        acc = (y_pred == self.y_test).sum() / len(y_pred)
        print("accuracy:",acc)
        self.assertGreaterEqual(acc, 0.2)

if __name__ == '__main__':
    unittest.main()
