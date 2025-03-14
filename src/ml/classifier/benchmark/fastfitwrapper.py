import numpy as np
import optuna

from pydantic import BaseModel
from pydantic.config import ConfigDict
from typing import Any
from typing import Optional, Union, Tuple
from datasets import Dataset, DatasetDict
from pathlib import Path
from typing import Dict
from types import MethodType
from transformers import AutoTokenizer, pipeline
from transformers.modeling_outputs import SequenceClassifierOutput

from src.ml.classifier.benchmark.fastfit.train import FastFitTrainer
from src.util.hashing import Hash
from src.util.caching import PickleCacheHandler
from src.util.constants import UnknownClassLabel
from src.ml.classifier.benchmark.base import BaseBenchmark


def new_forward(self, input_ids, attention_mask, labels=None, **kwargs):
    return SequenceClassifierOutput(
        logits=self.inference_forward(input_ids, attention_mask),
    )


class FastFitWrapper(BaseModel, BaseBenchmark):

    embedding_model_name: str = "mixedbread-ai/mxbai-embed-large-v1"
    num_train_epochs: int = 40
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    max_text_length: int = 128
    dataloader_drop_last: bool = False
    num_repeats: int = 4
    optim: str = "adafactor"
    clf_loss_factor: float = 0.1
    fp16: bool = False

    unknown_threshold: float = -0.05
    model: Optional[Any] = None  # will be set after fit
    use_cache: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def get_hyperparameters(trial: optuna.Trial) -> Dict[Any, Any]:

        params = {
            'params': {
                'unknown_threshold': trial.suggest_float('unknown_threshold', -1.0, 0.0)
        }}

        return params

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):

        if len(x_train.shape) == 2:
            x_train = x_train.reshape(-1)
        if len(x_valid.shape) == 2:
            x_valid = x_valid.reshape(-1)

        assert len(x_train.shape) == 1, "Input data must be 1D"
        assert len(x_valid.shape) == 1, "Input validation data must be 1D"
        assert len(y_train.shape) == 1, "Labels must be 1D"
        assert len(y_valid.shape) == 1, "Labels from validation set must be 1D"

        # Convert NumPy arrays into dictionaries
        train_data = {"text": x_train.tolist(), "label": y_train.tolist()}
        valid_data = {"text": x_valid.tolist(), "label": y_valid.tolist()}

        # Create Dataset objects
        train_dataset = Dataset.from_dict(train_data)
        valid_dataset = Dataset.from_dict(valid_data)

        # Combine into a DatasetDict
        dataset = DatasetDict({
            "train": train_dataset,
            "test": valid_dataset
        })

        trainer = FastFitTrainer(
            model_name_or_path=self.embedding_model_name,
            label_column_name="label",
            text_column_name="text",
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            max_text_length=self.max_text_length,
            dataloader_drop_last=self.dataloader_drop_last,
            num_repeats=self.num_repeats,
            optim=self.optim,
            clf_loss_factor=self.clf_loss_factor,
            fp16=self.fp16,
            dataset=dataset
        )

        model = trainer.train()
        model.forward = MethodType(new_forward, model)
        self.model = model

        return None
    
    def _predict(self, x: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        classifier = pipeline("text-classification", model=self.model, tokenizer=tokenizer)

        y_pred_raw = classifier(x.tolist())
        y_pred = np.array([el["label"] for el in y_pred_raw])
        y_pred_proba = np.array([el["score"] for el in y_pred_raw])

        return y_pred, y_pred_proba

    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        if len(x.shape) == 2:
            x = x.reshape(-1)

        assert len(x.shape) == 1, "Input data must be 1D" 

        y_pred, y_pred_proba = None, None

        if not self.use_cache:

            y_pred, y_pred_proba = self._predict(x)
            assert isinstance(y_pred, np.ndarray)
            assert isinstance(y_pred_proba, np.ndarray)

        else:


            filename = Hash.hash_list(x.tolist()) + '.pkl'
            filepath = Path(self.__class__.__name__) / filename

            cache_handler = PickleCacheHandler(
                filepath=filepath
            )

            # load cache
            cache: Optional[np.ndarray] = cache_handler.read()

            if cache is not None:
                y_pred, y_pred_proba = cache
            else:
                y_pred, y_pred_proba = self._predict(x)

                # save cache
                cache_handler.write((y_pred, y_pred_proba))

        assert y_pred is not None
        assert y_pred_proba is not None

        outlier_scores = -y_pred_proba
        y_pred[outlier_scores > self.unknown_threshold] = UnknownClassLabel.UNKNOWN_STR.value

        if include_outlierscore:
            return y_pred, outlier_scores

        return y_pred
        

    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise ValueError("Not implemented yet")

    
    