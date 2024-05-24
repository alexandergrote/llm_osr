import warnings

import numpy as np
import pandas as pd

from pydantic import BaseModel
from typing import List, Dict, Set
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from sklearn.metrics._classification import UndefinedMetricWarning

from src.ml.evaluation.base import BaseEvaluator



class Evaluator(BaseModel, BaseEvaluator):

    def _add_binary_result(
        self,
        data: dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_instance: int,
        name_tag: str,
    ):

        data[f"f1_{name_tag}_class_{class_instance}"] = f1_score(
            y_true=y_true, y_pred=y_pred
        )
        data[f"precision_{name_tag}_class_{class_instance}"] = precision_score(
            y_true=y_true, y_pred=y_pred
        )
        data[f"recall_{name_tag}_class_{class_instance}"] = recall_score(
            y_true=y_true, y_pred=y_pred
        )
        
        data[f"sample_size_{name_tag}_class_{class_instance}"] = sum(y_true)

        return data

    def _add_binary_known_results(
        self,
        data: dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name_tag: str,
        classes_in_training: List[int],
    ):

        # calculate metrics for each class
        for class_instance in classes_in_training:

            # reduce problem to a binary setting
            binary_test = y_true == class_instance
            binary_pred = y_pred == class_instance

            data = self._add_binary_result(
                data=data,
                y_true=binary_test,
                y_pred=binary_pred,
                class_instance=class_instance,
                name_tag=name_tag,
            )

        return data

    def _add_binary_unknown_results(
        self,
        data: dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name_tag: str,
        unknown_classes: List[int],
    ):

        truth = y_true.copy()
        prediction = y_pred.copy()

        # get mask
        mask_uuc = np.isin(
            truth.reshape(
                -1,
            ),
            unknown_classes,
        )

        class_instance = -1
        truth = np.where(mask_uuc, class_instance, truth)

        # reduce problem to a binary setting
        binary_test = truth == class_instance
        binary_pred = prediction == class_instance

        data = self._add_binary_result(
            data=data,
            y_true=binary_test,
            y_pred=binary_pred,
            class_instance=class_instance,
            name_tag=name_tag,
        )

        data["ratio_unknown_pred"] = np.sum(binary_pred) / len(binary_pred)
        data["ratio_unknown_true"] = np.sum(binary_test) / len(binary_test)

        return data

    def _add_overall_results(
        self,
        data: dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name_tag: str,
        classes_in_training: List[int],
    ):

        # get copy of predictions and test
        predictions = y_pred.copy()
        truth = y_true.copy()

        # get mask
        mask_kkc = np.isin(
            truth.reshape(
                -1,
            ),
            classes_in_training,
        )

        truth = np.where(~mask_kkc, -1, truth)

        data[f"f1_{name_tag}"] = f1_score(
            y_true=truth, y_pred=predictions, average="micro"
        )
        data[f"precision_{name_tag}"] = precision_score(
            y_true=truth, y_pred=predictions, average="micro"
        )
        data[f"recall_{name_tag}"] = recall_score(
            y_true=truth, y_pred=predictions, average="micro"
        )
        data[f"accuracy_{name_tag}"] = accuracy_score(
            y_true=truth, y_pred=predictions
        )

        return data

    def evaluate(
        self, y_pred: np.ndarray, y_true: np.ndarray, classes_in_training: Set, **kwargs
    ) -> dict:

        # result placeholder
        results: Dict[str, float] = {}

        classes = list(classes_in_training)

        with warnings.catch_warnings():
            
            warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

            results = self._add_binary_known_results(
                data=results,
                y_true=y_true,
                y_pred=y_pred,
                name_tag="known",
                classes_in_training=classes,
            )

            results = self._add_binary_unknown_results(
                data=results,
                y_true=y_true,
                y_pred=y_pred,
                name_tag="unknown",
                unknown_classes=np.setdiff1d(y_true, classes),
            )

            # add overall results to the results
            results = self._add_overall_results(
                data=results,
                y_true=y_true,
                y_pred=y_pred,
                name_tag="overall",
                classes_in_training=classes,
            )

        # summarize results in dataframe for easier data wrangling
        report = pd.DataFrame(results, index=[0])

        # calculate average values
        for score in ["f1", "precision", "recall"]:
            report[f"{score}_avg"] = report.filter(like=f"{score}_known").mean(
                axis=1
            )[0]

        final_result = report.iloc[0, :].to_dict()

        return final_result