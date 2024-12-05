import warnings

import numpy as np
import pandas as pd

from pydantic import BaseModel
from typing import List, Dict, Set, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_curve, 
    auc
)

from sklearn.metrics._classification import UndefinedMetricWarning

from src.ml.evaluation.base import BaseEvaluator
from src.util.constants import ErrorValues


class Evaluator(BaseModel, BaseEvaluator):

    remove_errors: bool = True

    @staticmethod
    def _add_binary_result(
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
        **kwargs
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

        outlier_score = kwargs.get('unknown_scores', None)

        if outlier_score is not None:
            
            fpr, tpr, _ = roc_curve(binary_test, outlier_score)
            roc_auc = auc(fpr, tpr)

            data['unknown_scores_auc'] = roc_auc

        return data

    @staticmethod
    def _add_overall_results(
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

        if predictions.dtype == "object":
            predictions = predictions.astype(int)


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
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray, 
        classes_in_training: Set, 
        unknown_scores: Optional[pd.Series],
        **kwargs
    ) -> dict:

        # result placeholder
        results: Dict[str, float] = {}

        classes = list(classes_in_training)

        """errors = [
            (f"ratio_{ErrorValues.PARSING_STR.value}", ErrorValues.PARSING_NUM.value),
            (f"ratio_{ErrorValues.LOGPROB_STR.value}", ErrorValues.LOGPROB_NUM.value)
        ]"""

        if self.remove_errors:

            mask_parsing = y_pred != ErrorValues.PARSING_NUM.value
            y_pred = y_pred[mask_parsing]
            y_true = y_true[mask_parsing]

            if unknown_scores is not None:
                unknown_scores = unknown_scores[mask_parsing]

            ratio_parsing_error = 1 - sum(mask_parsing) / len(mask_parsing)

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
                **kwargs
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

        # calculate micro average values
        num_known = report.filter(like="sample_size_known").sum(axis=1)[0]
        scores = ["f1", "precision", "recall"]

        for score in scores:
            report[f"{score}_avg"] = 0

        for class_value in classes_in_training:

            sample_size_known = report[f'sample_size_known_class_{str(class_value)}'][0]
            weighting_factor = sample_size_known / num_known

            for score in scores:
                report[f"{score}_avg"] += weighting_factor * report[f'{score}_known_class_{str(class_value)}'][0]

        # adding error metrics
        report["ratio_parsing_error"] = ratio_parsing_error
        final_result = report.iloc[0, :].to_dict()

        kwargs['metrics'] = final_result
        kwargs['y_pred'] = y_pred
        kwargs['y_true'] = y_true

        return kwargs
    

if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    

    outlier_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    y_true = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1])

    fpr, tpr, thresholds = roc_curve(y_true, outlier_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()