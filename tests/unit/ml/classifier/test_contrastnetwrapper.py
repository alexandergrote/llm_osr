import unittest
import sys
import os

import numpy as np

from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

from .util import BANKING77Dataset

from src.ml.classifier.benchmark.contrastnetwrapper import ContrastNetWrapper


class TestContrastNetWrapper(unittest.TestCase):
    
    def setUp(self):
        
        self.clf = ContrastNetWrapper(
            max_iter=2,
            evaluate_every=10,
            n_test_episodes=20,
            patience=1,
            n_classes=2
        )

        self.n_train = 800
        self.n_valid = 800
        self.n_test = 500

        self.experiment_names = [
            "banking77",
            "clinc__",
            "__hwu"
        ]

        x, y = BANKING77Dataset.get_data(n_rows=self.n_train + self.n_test + self.n_valid)

        # convert numpy arrays to strings
        x = np.array([str(i) for i in x])
        y = np.array([str(i) for i in y])

        # split data into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.n_test, random_state=42)
        self.x_train, self.x_valid, self.y_train, self.y_valid  = train_test_split(self.x_train, self.y_train, test_size=self.n_valid, random_state=42)
    
    def test_check_paraphrase_file_availability(self):

        for name in self.experiment_names:
            with self.subTest(name=name):
                filename = ContrastNetWrapper._get_paraphrase_filename(name)
                self.assertTrue(os.path.exists(filename))

    def test_check_model_availability(self):

        for name in self.experiment_names:
            with self.subTest(name=name):
                config_name_or_path = ContrastNetWrapper._get_tuned_model_name(name)
                tokenizer = AutoTokenizer.from_pretrained(config_name_or_path)
                model = AutoModel.from_pretrained(config_name_or_path).to("cpu")
                
                self.assertIsNotNone(tokenizer)
                self.assertIsNotNone(model)


    @unittest.skipIf(sys.platform.startswith("win"), "Test skipped on Windows")
    def test_fit(self):
        self.clf.fit(self.x_train, self.y_train, self.x_valid, self.y_valid, experiment_name="banking77")

    @unittest.skipIf(sys.platform.startswith("win"), "Test skipped on Windows")
    def test_predict(self):

        with self.assertRaises(ValueError):
            self.clf.predict(self.x_test)

        self.clf.fit(
            self.x_train, self.y_train, 
            self.x_valid, self.y_valid,
            experiment_name="banking77"
        )

        y_pred, y_pred_proba = self.clf._predict(self.x_test)
       
        self.assertEqual(len(y_pred), self.n_test)
        self.assertEqual(len(y_pred_proba), self.n_test)

if __name__ == '__main__':
    unittest.main()
