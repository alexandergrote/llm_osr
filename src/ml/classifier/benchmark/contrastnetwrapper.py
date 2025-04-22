import numpy as np
import optuna
import tempfile
import json
import os

from pydantic import BaseModel
from pydantic.config import ConfigDict
from typing import Optional, Union, Tuple, Any
from datasets import Dataset, DatasetDict
from pathlib import Path
from typing import Dict
from types import MethodType
from torch.optim import Adam
from transformers import AutoTokenizer, pipeline
from transformers.modeling_outputs import SequenceClassifierOutput

from src.ml.classifier.benchmark.contrastnet.model import ContrastNet
from src.ml.classifier.benchmark.contrastnet.paraphrase.utils.data import FewShotDataset, FewShotSSLFileDataset
from src.ml.classifier.benchmark.contrastnet.eda import eda

from src.ml.classifier.benchmark.contrastnet.model import ContrastNet
from src.util.hashing import Hash
from src.util.caching import PickleCacheHandler
from src.util.constants import UnknownClassLabel, Directory
from src.ml.classifier.benchmark.base import BaseBenchmark


class ContrastNetWrapper(BaseModel, BaseBenchmark):

    n_classes: int = 1
    n_support: int = 1
    n_query: int = 1
    n_task: int = 1
    n_unlabeled: int = 1
    lr: float = 1e-6
    task_weight: float = 0.1
    max_iter: int = 5


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

    @staticmethod
    def _create_fewshot_dataset(x: np.ndarray, y: np.ndarray, n_class: int, n_support: int, n_query: int, n_task: Optional[int] = None, n_unlabeled: Optional[int] = None) -> Union[FewShotDataset, FewShotSSLFileDataset]:
        
        is_validation_mode = (n_unlabeled is None) and (n_task is None)

        # create a jsonnline tempfile with the training data
        # each line has three keys: 'sentence', 'label' and 'aug_texts'
        data_file = tempfile.NamedTemporaryFile(delete=False).name + ".json"

        # create a text file with all the labels in the training dataset
        y_labels_file = tempfile.NamedTemporaryFile(delete=False).name + ".txt"
        with open(y_labels_file, 'w') as f:
            for label in y:
                f.write(str(label) + '\n')

        if is_validation_mode:

            with open(data_file, 'w') as f:
                for sentence, label in zip(x, y):
                    json.dump({'sentence': sentence, 'label': label}, f)
                    f.write('\n')

            dataset = FewShotDataset(
                data_path=data_file, 
                labels_path=y_labels_file, 
                n_classes=n_class, 
                n_support=n_support, 
                n_query=n_query
            )

            return dataset

        assert n_task is not None, "n_task must be specified for training mode"
        assert n_unlabeled is not None, "n_unlabeled must be specified for training mode"

        aug_texts = [eda(text) for text in x]
        assert len(aug_texts) == len(x), "Number of augmented texts must be equal to number of training sentences"
        assert isinstance(aug_texts[0], list), "Augmented texts must be a list of strings"

        with open(data_file, 'w') as f:
            for sentence, label, aug_text in zip(x, y, aug_texts):
                json.dump({'sentence': sentence, 'label': label, 'aug_texts': aug_text}, f)
                f.write('\n')
    

        augment_data_file = str(Directory.SRC / os.path.join(
            'ml',
            'classifier',
            'benchmark',
            'contrastnet',
            'data',
            'BANKING77',
            'paraphrases',
            'DBS-unigram-flat-1.0',
            'paraphrases.json'
        ))

        train_dataset = FewShotSSLFileDataset(
            data_path=data_file,
            labels_path=y_labels_file,
            n_classes=n_class,
            n_support=n_support,
            n_query=n_query,
            n_unlabeled=n_unlabeled,
            n_task=n_task,
            unlabeled_file_path=augment_data_file,
        )

        return train_dataset


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

        train_dataset = self._create_fewshot_dataset(
            x_train,
            y_train,
            n_class=self.n_classes,
            n_support=self.n_support,
            n_query=self.n_query,
            n_task=self.n_task,
            n_unlabeled=self.n_unlabeled,
        )

        valid_dataset = self._create_fewshot_dataset(
            x_valid,
            y_valid,
            n_class=self.n_classes,
            n_support=self.n_support,
            n_query=self.n_query,
        )

        model = ContrastNet(
            config_name_or_path='bert-base-uncased',
            metric='euclidean'
        )

        optimizer = Adam(model.parameters(), lr=self.lr)
        
        for step in range(self.max_iter):

            print(f'step: {step} / {self.max_iter} step(s)')

            episode = train_dataset.get_episode()

            episode_str = ','.join([f"{k}: {len(episode[k])}" for k in episode])
            print(episode_str)
        
            xs_key = 'xs'
            classes = []
            for i in episode[xs_key]:
                classes.append(i[0]['label'])    

            supervised_loss_share = super_weight*(1. - step/max_iter)
            task_loss_share = task_weight

            loss, loss_dict = model.train_step(optimizer=optimizer, episode=episode, supervised_loss_share=supervised_loss_share, task_loss_share=task_loss_share)

        

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

    
    