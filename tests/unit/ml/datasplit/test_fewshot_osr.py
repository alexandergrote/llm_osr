import unittest
import pandas as pd
import numpy as np

from src.ml.datasplit.fewshot_osr import FewShotDataSplitter
from src.util.constants import DatasetColumn
from src.util.types import MLDataFrame

class TestBaseSplit(unittest.TestCase):

    def setUp(self) -> None:
        
        self.data = pd.DataFrame({
            DatasetColumn.LABEL: [
                0, 0, 1, 1, 1
            ]
        })

        self.splitter = FewShotDataSplitter(
            n=5,
            replace=True,
            percentage_unknown_classes=0.2,
            percentage_instances_of_known_classes_in_trainset=0.6,
            percentage_instances_of_known_classes_in_fittingset=0.4,
        )
    
    def test__fewshot_split(self):
        
        # input variables
        params = {
            'random_seed': 42,
            'data': self.data
        }

        ## case: just one observation per class

        # actual outcome
        actual_result = FewShotDataSplitter.fewshot_split(n=1, **params)

        # comparison
        self.assertEqual(len(actual_result), 2)
    
        ## case: two observations per class

        actual_result = FewShotDataSplitter.fewshot_split(n=2, **params)
        self.assertEqual(len(actual_result), 4)

        ## case: three observations per class

        actual_result = FewShotDataSplitter.fewshot_split(n=3, **params)
        self.assertEqual(len(actual_result), 5)


    def test__split_data(self):

        n = 1000

        # Seed and retrieve the values
        rng = np.random.default_rng(seed=42)
        numbers = rng.choice([str(el) for el in range(11)], size=n)

        data = pd.DataFrame({
            DatasetColumn.LABEL: numbers,
            DatasetColumn.TEXT: ['text'] * n
        })

        data_fit, data_valid, data_test = self.splitter._split_data(data, random_seed=42)
        
        self.assertIsInstance(data_test, MLDataFrame)
        self.assertIsInstance(data_fit, MLDataFrame)
        self.assertIsInstance(data_valid, MLDataFrame)
        
        self.assertEqual(len(data_fit.data), len(data_fit.data[DatasetColumn.LABEL].unique()) * self.splitter.n)
        
        self.assertEqual(len(data_fit.data[DatasetColumn.LABEL].unique()), 6)
        self.assertEqual(len(data_valid.data[DatasetColumn.LABEL].unique()), 9)
        self.assertEqual(len(data_test.data[DatasetColumn.LABEL].unique()), 11)

    def tearDown(self) -> None:
        return super().tearDown()