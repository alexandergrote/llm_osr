import torch
import optuna
import numpy as np
from torch.nn import functional as F
from pydantic import BaseModel, validate_call, confloat
from pydantic.v1 import validate_arguments
from typing import Optional, Dict, Any, Union, Tuple
from sklearn.neighbors import LocalOutlierFactor

from src.ml.classifier.nn.cls.base import BaseBenchmark
from src.ml.classifier.nn.cls.util.labelling import LabellingUtilities
from src.ml.classifier.nn.cls.util.fewshot import compute_prototypes, compute_predictions_from_logits
from src.util.constants import UnknownClassLabel

# set random seed for reproducibility
torch.manual_seed(0)


class Oslo(BaseModel, BaseBenchmark):

    prototypes: Optional[torch.TensorType] = None
    support_features: Optional[torch.TensorType] = None
    support_labels: Optional[torch.TensorType] = None
    
    label2idx: dict = {}
    idx2label: dict = {}

    inference_steps: int = 100
    lambda_s: confloat(ge=0, le=1) = 0.5  # type: ignore
    lambda_z: confloat(ge=0, le=1) = 0.5  # type: ignore
    ema_weight: confloat(ge=0, le=1) = 1  # type: ignore

    unknown_threshold: float = -0.05

    outlier_model: Optional[LocalOutlierFactor] = None 


    class Config:
        arbitrary_types_allowed = True
    

    @staticmethod
    def get_hyperparameters(trial: optuna.Trial) -> Dict[Any, Any]:

        params = {
            'params': {
                'unknown_threshold': trial.suggest_float('unknown_threshold', -1.0, 0.0),
                'inference_steps': trial.suggest_int('inference_steps', 1, 10),
                'lambda_s': trial.suggest_float('lambda_s', 0.0, 1.0),
                'lambda_z': trial.suggest_float('lambda_z', 0.0, 1.0),
                'ema_weight': trial.suggest_float('ema_weight', 0.0, 1.0),
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
        self.support_features = torch.Tensor(x_train)
        self.support_labels = torch.Tensor(y_train)

        self.outlier_model = LocalOutlierFactor(novelty=True)
        self.outlier_model.fit(x_train)

    def cosine(self, X, Y):
        return F.normalize(X, dim=-1) @ F.normalize(Y, dim=-1).T

    def get_logits(self, prototypes: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        return self.cosine(query_features, prototypes)  # [query_size, num_classes]


    @validate_arguments(config={"arbitrary_types_allowed": True})
    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        if self.prototypes is None:
            raise ValueError("Model not fitted")
        
        if self.support_features is None:
            raise ValueError("Support features not provided")
        
        if self.support_labels is None:
            raise ValueError("Support labels not provided")
        
        if self.outlier_model is None:
            raise ValueError("Outlier model not provided")
        
        if include_outlierscore:
            raise ValueError("Outlier score not implemented yet")
        
        x_tensor = torch.Tensor(x)

        num_classes = self.support_labels.unique().size(0)
        one_hot_labels = F.one_hot(self.support_labels.to(torch.int64), num_classes)  # [support_size, num_classes]
        support_size = self.support_features.size(0)

        # init soft assignments
        soft_assignements = (1 / num_classes) * torch.ones(x_tensor.size(0), num_classes)  # [query_size, num_classes]

        # init inlier scores
        inlier_scores = 0.5 * torch.ones((x_tensor.size(0), 1))
        #outliers = self.outlier_model.predict(x_tensor) == -1
        #inliers = ~outliers  # given by outlier detection algorithm

        prototypes = self.prototypes

        for _ in range(self.inference_steps):

            # Compute inlier scores
            logits_q = self.get_logits(prototypes, x_tensor)  # [query_size, num_classes]

            inlier_scores = (
                self.ema_weight
                * (
                    (soft_assignements * logits_q / self.lambda_s)
                    .sum(-1, keepdim=True)
                    .sigmoid()
                )
                + (1 - self.ema_weight) * inlier_scores
            )  # [query_size, 1]

            # Compute new assignements
            soft_assignements = (
                (
                    self.ema_weight
                    * ((inlier_scores * logits_q / self.lambda_z).softmax(-1))
                    + (1 - self.ema_weight) * soft_assignements
                ))  # [query_size, num_classes]

            # compute metrics
            outlier_scores = 1 - inlier_scores 

            # Update prototypes
            all_features = torch.cat([self.support_features, x_tensor], 0)  # [support_size + query_size, feature_dim]
            all_assignements = torch.cat([one_hot_labels, soft_assignements], dim=0)  # [support_size + query_size, num_classes]
            all_inliers_scores = (torch.cat([torch.ones(support_size, 1), inlier_scores], 0))  # [support_size + query_size, 1]

            prototypes = (
                self.ema_weight
                * (
                    (all_inliers_scores * all_assignements).T
                    @ all_features
                    / (all_inliers_scores * all_assignements).sum(0).unsqueeze(1)
                )
                + (1 - self.ema_weight) * prototypes
            )  # [num_classes, feature_dim]

        logits = self.get_logits(prototypes, x_tensor)

        y_pred = compute_predictions_from_logits(logits, self.unknown_threshold, outlier_scores.view(-1,)).numpy()
        y_pred = LabellingUtilities.map_labels(y=y_pred, mapping=self.idx2label, target_dtype='str', unknown_value=UnknownClassLabel.UNKNOWN_STR.value)

        if include_outlierscore:

            outlier_scores = outlier_scores.view(-1,).numpy()

            return y_pred, outlier_scores

        return y_pred

    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("Method not implemented")  


__all__ = ["Oslo"]