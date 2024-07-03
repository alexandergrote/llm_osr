import torch
import optuna
import numpy as np
import pandas as pd
from pydantic import BaseModel, validate_call
from pydantic.v1 import validate_arguments
from typing import Optional, Dict, Any
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset

from src.ml.classifier.nn.cls.base import BaseBenchmark
from src.util.constants import DatasetColumn

# set random seed for reproducibility
torch.manual_seed(0)


class SetFit(BaseModel, BaseBenchmark):

    model: Optional[SetFitModel] = None
    
    label2idx: dict = {}
    idx2label: dict = {}

    # model args
    embedding_name: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    batch_size: int = 16
    num_epochs: int = 1

    # osr threshold
    unknown_threshold: float = -0.05

    class Config:
        arbitrary_types_allowed = True
    

    @staticmethod
    def get_hyperparameters(trial: optuna.Trial) -> Dict[Any, Any]:

        params = {
            'params': {
                'unknown_threshold': trial.suggest_float('unknown_threshold', -1.0, 0.0),
        }}

        return params


    @validate_call(config={"arbitrary_types_allowed": True})
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):

        assert len(x_train.shape) == 1, "Input data must be 1D"
        assert len(y_train.shape) == 1, "Labels must be 1D"

        # check labels
        if issubclass(y_train.dtype.type, np.integer):
            y_train = y_train.astype(str)

        unique_labels = list(map(str, np.unique(y_train)))

        model = SetFitModel.from_pretrained(
            self.embedding_name,
            labels=unique_labels
        )

        args = TrainingArguments(
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            end_to_end=False
        )

        train_df = pd.DataFrame({
            DatasetColumn.TEXT: x_train,
            DatasetColumn.LABEL: y_train
        })

        train_dataset = Dataset.from_pandas(train_df)

        valid_df = pd.DataFrame({
            DatasetColumn.TEXT: x_valid,
            DatasetColumn.LABEL: y_valid
        })

        valid_dataset = Dataset.from_pandas(valid_df)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            metric="accuracy",
            column_mapping={DatasetColumn.TEXT: "text", DatasetColumn.LABEL: "label"}  # Map dataset columns to text/label expected by trainer
        )

        trainer.train()

        self.model = trainer.model

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:

        if self.model is None:
            raise ValueError("Model not fitted")
        
        assert len(x.shape) == 1, "Input data must be 1D"

        x_list = list(x)


        y_pred = self.model.predict(x_list)

        return y_pred

    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("Method not implemented")

        


__all__ = ["SetFit"]