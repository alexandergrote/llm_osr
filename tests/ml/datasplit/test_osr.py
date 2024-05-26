import unittest
import pandas as pd
import numpy as np

from src.ml.datasplit.osr import DataSplitter
from src.util.constants import DatasetColumn
from src.util.types import MLDataFrame


class TestDataSplitter(unittest.TestCase):

    def setUp(self) -> None:

        self.splitter = DataSplitter(
            percentage_unknown_classes=0.2,
            percentage_instances_of_known_classes_in_trainset=0.6,
            percentage_instances_of_known_classes_in_fittingset=0.4,
        )

    def test__determine_known_classes(self):

        y = np.array([1, 1, 2, 2, 3, 4, 5])
        unique_classes = np.unique(y)
        n_known_classes = 3

        known_classes = self.splitter._determine_known_classes(y, n_known_classes)
        self.assertEqual(len(known_classes), n_known_classes)
        self.assertTrue(all([c in unique_classes for c in known_classes]))

    def test__get_subset_mask(self):
        y = np.array([1, 2, 3, 4, 5, 1])
        classes_to_keep = np.array([1, 3, 5])
        mask = self.splitter._get_subset_mask(y, classes_to_keep)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(len(mask), len(y))
        self.assertTrue(all([m in [True, False] for m in mask]))
        self.assertTrue(np.array_equal(mask, np.array([True, False, True, False, True, True])))

    def test__train_test_split_by_known_classes(self):
        
        data = pd.DataFrame({
            DatasetColumn.LABEL: [1, 1, 2, 2, 3, 3, 4, 4, 5],
            DatasetColumn.TEXT: ['text'] * 9
        })
        
        mask_known_classes = np.array([True, True, False, False, True, True, False, False, True])
        
        train_size = 0.6

        data_train, data_test = self.splitter._train_test_split_by_known_classes(
            data, 
            mask_known_classes, 
            train_size,
            random_seed=42
        )

        self.assertIsInstance(data_train, pd.DataFrame)
        self.assertIsInstance(data_test, pd.DataFrame)
        self.assertEqual(len(data_train),3)
        self.assertEqual(len(data_test), 6)
        
    def test__split_into_train_test_data(self):

        perc_known = 0.6
        train_size = 0.6

        n = 1000

        # Seed and retrieve the values
        rng = np.random.default_rng(seed=42)
        numbers = rng.choice(range(5), size=n)

        data = pd.DataFrame({
            DatasetColumn.LABEL: numbers,
            DatasetColumn.TEXT: ['text'] * n
        })

        data_train, data_test = self.splitter._split_into_train_test_data(
            data, 
            perc_known, 
            train_size,
            random_seed=42
        )

        self.assertIsInstance(data_train, MLDataFrame)
        self.assertIsInstance(data_test, MLDataFrame)
        self.assertEqual(len(data_train.data) + len(data_test.data), len(data))
        self.assertEqual(len(data_train.data[DatasetColumn.LABEL].unique()), 3)
        self.assertEqual(len(data_test.data[DatasetColumn.LABEL].unique()), 5)

    def test__split_train_into_fitting_and_validation_data(self):
        perc_known = 0.6
        train_size = 0.6

        n = 1000

        # Seed and retrieve the values
        rng = np.random.default_rng(seed=42)
        numbers = rng.choice(range(11), size=n)

        data = pd.DataFrame({
            DatasetColumn.LABEL: numbers,
            DatasetColumn.TEXT: ['text'] * n
        })

        data_fit, data_valid = self.splitter._split_train_into_fitting_and_validation_data(data, perc_known, train_size, random_seed=42)
        self.assertIsInstance(data_fit, MLDataFrame)
        self.assertIsInstance(data_valid, MLDataFrame)
        self.assertEqual(len(data_fit.data) + len(data_valid.data), n)
        self.assertEqual(len(data_fit.data[DatasetColumn.LABEL].unique()), 8)
        self.assertEqual(len(data_valid.data[DatasetColumn.LABEL].unique()), 11)

    def test__split_train_into_fitting_and_validation_data_zero_openness(self):
        
        perc_known = 1
        train_size = 0.6

        n = 1000

        # Seed and retrieve the values
        rng = np.random.default_rng(seed=42)
        numbers = rng.choice(range(11), size=n)

        data = pd.DataFrame({
            DatasetColumn.LABEL: numbers,
            DatasetColumn.TEXT: ['text'] * n
        })

        data_fit, data_valid = self.splitter._split_train_into_fitting_and_validation_data(data, perc_known, train_size, random_seed=42)
        self.assertIsInstance(data_fit, MLDataFrame)
        self.assertIsInstance(data_valid, MLDataFrame)
        self.assertEqual(len(data_fit.data) + len(data_valid.data), n)
        self.assertEqual(len(data_fit.data[DatasetColumn.LABEL].unique()), 11)
        self.assertEqual(len(data_valid.data[DatasetColumn.LABEL].unique()), 11)

    def test__split_data(self):

        n = 1000

        # Seed and retrieve the values
        rng = np.random.default_rng(seed=42)
        numbers = rng.choice(range(11), size=n)

        data = pd.DataFrame({
            DatasetColumn.LABEL: numbers,
            DatasetColumn.TEXT: ['text'] * n
        })

        data_fit, data_valid, data_test = self.splitter._split_data(data, random_seed=42)
        
        self.assertIsInstance(data_test, MLDataFrame)
        self.assertIsInstance(data_fit, MLDataFrame)
        self.assertIsInstance(data_valid, MLDataFrame)
        
        self.assertEqual(len(data_fit.data) + len(data_test.data) + len(data_valid.data), n)
        
        self.assertEqual(len(data_fit.data[DatasetColumn.LABEL].unique()), 6)
        self.assertEqual(len(data_valid.data[DatasetColumn.LABEL].unique()), 9)
        self.assertEqual(len(data_test.data[DatasetColumn.LABEL].unique()), 11)