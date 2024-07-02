import torch
import optuna
import numpy as np
from torch.functional import F
from pydantic import BaseModel, validate_call
from pydantic.v1 import validate_arguments
from typing import Optional, Dict, Any

from src.ml.classifier.nn.cls.base import BaseBenchmark
from src.ml.classifier.nn.cls.util.labelling import LabellingUtilities
from src.ml.classifier.nn.cls.util.fewshot import compute_prototypes, compute_outlier_scores
from src.util.constants import UnknownClassLabel

# set random seed for reproducibility
torch.manual_seed(0)


class SimpleShot(BaseModel, BaseBenchmark):

    prototypes: Optional[torch.TensorType] = None
    
    label2idx: dict = {}
    idx2label: dict = {}

    softmax_temperature: float = 1.0
    unknown_threshold: float = -0.05


    class Config:
        arbitrary_types_allowed = True
    

    @staticmethod
    def get_hyperparameters(trial: optuna.Trial) -> Dict[Any, Any]:

        params = {
            'params': {
                'unknown_threshold': trial.suggest_float('unknown_threshold', -1.0, 0.0)
        }}

        return params
    
    def get_logits_from_cosine_distances_to_prototypes(self, samples):
        return (
            self.softmax_temperature
            * F.normalize(samples, dim=1)
            @ F.normalize(self.prototypes, dim=1).T
        )


    @validate_call(config={"arbitrary_types_allowed": True})
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):

        assert len(x_train.shape) == 2, "Input data must be 2D"
        assert len(y_train.shape) == 1, "Labels must be 1D"

        # prepare label mapping
        self.label2idx, self.idx2label = LabellingUtilities.create_label_mapping(y=y_train)

        y_train = LabellingUtilities.map_labels(y=y_train, mapping=self.label2idx, target_dtype='int', unknown_value=UnknownClassLabel.UNKNOWN_NUM.value)
        y_valid = LabellingUtilities.map_labels(y=y_valid, mapping=self.label2idx, target_dtype='int', unknown_value=UnknownClassLabel.UNKNOWN_NUM.value)

        # check labels
        unique_labels = np.unique(y_train)
        assert len(unique_labels) == len(self.label2idx), "Labels are not correctly mapped"
        assert max(unique_labels) == len(unique_labels) - 1, "Labels are not correctly mapped"
        assert min(unique_labels) == 0, "Labels are not correctly mapped"

        self.prototypes = compute_prototypes(support_features=torch.Tensor(x_train), support_labels=torch.Tensor(y_train))

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:

        if self.prototypes is None:
            raise ValueError("Model not fitted")
        
        x_tensor = torch.Tensor(x)

        logits = self.get_logits_from_cosine_distances_to_prototypes(samples=x_tensor)
        y_pred_proba = logits.softmax(-1)

        y_pred = torch.argmax(y_pred_proba, dim=-1)

        outlier_scores = compute_outlier_scores(y_pred_proba)
        y_pred[outlier_scores > self.unknown_threshold] = UnknownClassLabel.UNKNOWN_NUM.value

        y_pred = y_pred.numpy()        
        y_pred = LabellingUtilities.map_labels(y=y_pred, mapping=self.idx2label, target_dtype='str', unknown_value=UnknownClassLabel.UNKNOWN_STR.value)

        return y_pred

    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("Method not implemented")

        


__all__ = ["SimpleShot"]