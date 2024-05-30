import warnings

import pandas as pd
import numpy as np
from pydantic import BaseModel, validate_call
from sklearn.model_selection import train_test_split

from src.ml.datasplit.base import BaseDatasplit
from src.util.constants import DatasetColumn
from src.util.types import Percentage, DualDataFrameTuple, TripleDataFrameTuple, MLDataFrame


class DataSplitter(BaseModel, BaseDatasplit):
    
    percentage_unknown_classes: Percentage
    percentage_instances_of_known_classes_in_trainset: Percentage
    percentage_instances_of_known_classes_in_fittingset: Percentage

    @validate_call(config={"arbitrary_types_allowed": True})
    def _determine_known_classes(
        self, y: np.ndarray, n_known_classes: int
    ) -> np.ndarray:
        y = y.flatten()
        rng = np.random.default_rng(seed=42) 
        known_classes = rng.choice(y, n_known_classes, replace=False)

        return known_classes

    @validate_call(config={"arbitrary_types_allowed": True})
    def _get_subset_mask(self, y: np.ndarray, classes_to_keep: np.ndarray):
        mask = np.zeros_like(y, dtype=bool)

        for class_to_keep in classes_to_keep:
            mask = mask + (y == class_to_keep)

        return mask

    @validate_call(config={"arbitrary_types_allowed": True})
    def _train_test_split_by_known_classes(self, data: pd.DataFrame, mask_known_classes: np.ndarray, train_size: Percentage, random_seed: int):
        
        # apply mask to get subsets
        data_known_classes = data[mask_known_classes]
        data_unknown_classes = data[~mask_known_classes]

        # split known classes into train and test set
        (
            data_train,
            data_test_known_classes,
        ) = train_test_split(
            data_known_classes, train_size=train_size, random_state=random_seed
        )

        # concatenate test set
        data_test = pd.concat([data_test_known_classes, data_unknown_classes])

        for el in [data_train, data_test]:
            el.reset_index(drop=True, inplace=True)

        return data_train, data_test

    @validate_call(config={"arbitrary_types_allowed": True})
    def _split_into_train_test_data(
        self,
        data: pd.DataFrame,
        perc_known: Percentage,
        train_size: Percentage,
        random_seed: int
    ) -> DualDataFrameTuple:
        
        y = data[DatasetColumn.LABEL].values
        unique_classes = np.unique(y)
        number_of_known_classes = np.ceil(perc_known * len(unique_classes))

        known_classes = self._determine_known_classes(
            y=unique_classes, n_known_classes=number_of_known_classes
        )

        mask_known_classes = self._get_subset_mask(
            y=y, classes_to_keep=known_classes
        )

        data_train, data_test = self._train_test_split_by_known_classes(
            data=data,
            mask_known_classes=mask_known_classes,
            train_size=train_size,
            random_seed=random_seed
        )

        for el in [data_train, data_test]:
            el.reset_index(drop=True, inplace=True)

        data_train = MLDataFrame.from_raw_pandas_dataframe(data_train)
        data_test = MLDataFrame.from_raw_pandas_dataframe(data_test)

        return data_train, data_test
        

    @validate_call(config={"arbitrary_types_allowed": True})
    def _split_train_into_fitting_and_validation_data(
        self, data: pd.DataFrame, perc_known: Percentage, train_size: Percentage, random_seed: int
    ) -> DualDataFrameTuple:
        
        y = data[DatasetColumn.LABEL].values
        unique_classes = np.unique(y)

        number_of_known_classes = np.round(2 / 3 * len(unique_classes) + 0.5)

        # override percentage of known classes when openness_score is zero
        if perc_known == 1:
            number_of_known_classes = len(unique_classes)

        known_classes = self._determine_known_classes(
            y=unique_classes, n_known_classes=number_of_known_classes
        )

        mask_known_classes = self._get_subset_mask(
            y=y, classes_to_keep=known_classes
        )

        data_fit, data_valid = self._train_test_split_by_known_classes(
            data=data,
            mask_known_classes=mask_known_classes,
            train_size=train_size,
            random_seed=random_seed
        )

        for el in [data_fit, data_valid]:
            el.reset_index(drop=True, inplace=True)

        data_fit = MLDataFrame.from_raw_pandas_dataframe(data_fit)
        data_valid = MLDataFrame.from_raw_pandas_dataframe(data_valid)

        return data_fit, data_valid

    
    def _split_data(self, data: pd.DataFrame, random_seed: int, **kwargs) -> TripleDataFrameTuple:
        
        perc_known = 1 - self.percentage_unknown_classes

        with warnings.catch_warnings():

            warnings.simplefilter("ignore", category=UserWarning)

            data_train, data_test = self._split_into_train_test_data(
                data=data,
                perc_known=perc_known,
                train_size=self.percentage_instances_of_known_classes_in_trainset,
                random_seed=random_seed
            )

            data_fit, data_valid = self._split_train_into_fitting_and_validation_data(
                data=data_train.data,
                perc_known=perc_known,
                train_size=self.percentage_instances_of_known_classes_in_fittingset,
                random_seed=random_seed
            )

        for el in [data_fit, data_valid, data_test]:
            el.data.reset_index(drop=True, inplace=True)

        return data_fit, data_valid, data_test