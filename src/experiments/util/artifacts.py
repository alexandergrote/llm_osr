import pandas as pd
import numpy as np

from typing import List, Tuple
from pathlib import Path

from src.util.mlflow_columns import artifact_columns
from src.util.types import MLDataFrame, MLPrediction

def get_artifacts(data_copy: pd.DataFrame) -> Tuple[List[MLPrediction], List[MLDataFrame], List[MLDataFrame], List[MLDataFrame]]:
    
    """Get artifacts from the data copy."""
    # get prediction datasets
    artifact_paths = data_copy[artifact_columns.artifact_path.column_name].to_list()

    prediction_list = []
    data_train_list = []
    data_valid_list = []
    data_test_list = []


    for path in artifact_paths:

        prediction = MLPrediction.load(Path(path)) 
        data_train = MLDataFrame.load(str(Path(path) / 'data_train.csv'))
        data_valid = MLDataFrame.load(str(Path(path) / 'data_valid.csv'))
        data_test = MLDataFrame.load(str(Path(path) / 'data_test.csv'))

        prediction_list.append(prediction)
        data_train_list.append(data_train)
        data_valid_list.append(data_valid)
        data_test_list.append(data_test)

    return prediction_list, data_train_list, data_valid_list, data_test_list

def artifact_sanity_check(data_copy, dataset_col) -> None:

    """Sanity check for artifacts."""

    _, data_train_list, data_valid_list, data_test_list = get_artifacts(data_copy=data_copy)

    # sanity check
    # ensure that same training is used across different models
    datasets = data_copy[dataset_col].unique()

    for dataset in datasets:

        mask = data_copy[dataset_col] == dataset
        
        # sanity check for training data
        data_train_sub = np.array(data_train_list)[mask]
        for data_train in data_train_sub:
            assert data_train.data.equals(data_train_sub[0].data), f"Training data is not the same across different models. Training data for dataset {dataset} is different."

        # sanity check for validation data
        data_valid_sub = np.array(data_valid_list)[mask]
        for data_valid in data_valid_sub:
            assert data_valid.data.equals(data_valid_sub[0].data), f"Validation data is not the same across different models. Validation data for dataset {dataset} is different."

        # sanity check for test data
        data_test_sub = np.array(data_test_list)[mask]
        for data_test in data_test_sub:
            assert data_test.data.equals(data_test_sub[0].data), f"Test data is not the same across different models. Test data for dataset {dataset} is different."
