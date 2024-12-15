import unittest
import numpy as np

from src.ml.evaluation.osr import Evaluator

class TestEvaluator(unittest.TestCase):

    def setUp(self):

        # Set up test data
        self.y_true = np.array([0, 1, 1, 0, 2])
        self.y_pred = np.array([0, 1, 0, 0, -1])
        self.outlier_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.classes_in_training = {0, 1}

        # Set up evaluator
        self.evaluator = Evaluator()

    def test_evaluate_mixed(self):

        y_pred = np.array(["yes", "no", "yes", "no", "yes", "unknown", "unknown"])
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        outlier_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        classes_in_training = {"yes", "no"}

        with self.assertRaises(ValueError):
           
            self.evaluator.evaluate(
                y_pred=y_pred,
                y_true=y_true,
                classes_in_training=classes_in_training,
                unknown_scores=outlier_score
            )
            

    def test_evaluate_str(self):

        y_pred = np.array(["yes", "no", "yes", "no", "yes", "unknown", "unknown"])
        y_true = np.array(["yes", "no", "yes", "yes", "yes", "maybe", "yes"])
        outlier_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        classes_in_training = {"yes", "no"}

        final_result = self.evaluator.evaluate(
            y_pred=y_pred,
            y_true=y_true,
            classes_in_training=classes_in_training,
            unknown_scores=outlier_score
        )['metrics']

        self.assertEqual(final_result['f1_known_class_yes'], 0.75)
        self.assertTrue(final_result['recall_unknown_class_unknown'] == 1)
    
        
    def test_evaluate(self):
        
        final_result = self.evaluator.evaluate(
            y_pred=self.y_pred,
            y_true=self.y_true,
            classes_in_training=self.classes_in_training,
            unknown_scores=self.outlier_score
        )['metrics']

        test_options = [
            ('precision_known_class_0', 2/3),
            ('recall_known_class_0', 1),
            ('precision_known_class_1', 1),
            ('recall_known_class_1', 0.5),
        ]

        for key, value in test_options:

            with self.subTest(msg=key):
                self.assertAlmostEqual(final_result[key], value)
        

if __name__ == "__main__":
    unittest.main()