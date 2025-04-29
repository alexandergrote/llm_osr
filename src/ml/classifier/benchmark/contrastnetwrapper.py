import numpy as np
import optuna
import tempfile
import json
import os
import torch

from pydantic import BaseModel
from pydantic.config import ConfigDict
from typing import Optional, Union, Tuple, Any
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from torch.optim import Adam
from collections import Counter

from src.ml.classifier.benchmark.contrastnet.model import ContrastNet
from src.ml.classifier.benchmark.contrastnet.paraphrase.utils.data import FewShotDataset, FewShotSSLFileDataset
from src.ml.classifier.benchmark.contrastnet.eda import eda

from src.ml.classifier.benchmark.util.torch_early_stopping import EarlyStopping
from src.ml.classifier.benchmark.util.labelling import LabellingUtilities
from src.util.hashing import Hash
from src.util.caching import PickleCacheHandler
from src.util.constants import UnknownClassLabel, Directory
from src.ml.classifier.benchmark.base import BaseBenchmark

def get_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def convert_to_native_types(obj):
    if isinstance(obj, dict):
        return {convert_to_native_types(k): convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj


class ContrastNetWrapper(BaseModel, BaseBenchmark):

    n_support: int = 1
    n_query: int = 1
    n_classes: int = 5

    super_tau: float = 5
    unsuper_tau: float = 7
    task_tau: float = 7
    max_len: int = 64
    n_task: int = 10
    n_unlabeled: int = 10
    lr: float = 1e-6
    task_weight: float = 0.1
    super_weight: float = 0.95

    
    max_iter: int = 10000
    n_test_episodes: int = 600
    evaluate_every: int = 100

    model_name_or_path: Optional[str] = None 
    metric: str = 'euclidean'

    k_nearest_neighbors: int = 5
    patience: Optional[int] = None

    unknown_threshold: float = -0.05
    model: Optional[Any] = None  # will be set after fit
    use_cache: bool = True

    label2idx: dict = {}
    idx2label: dict = {}

    x_train: Optional[torch.Tensor] = None
    y_train: Optional[torch.Tensor] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def get_hyperparameters(trial: optuna.Trial) -> Dict[Any, Any]:

        params = {
            'params': {
                'unknown_threshold': trial.suggest_float('unknown_threshold', -1.0, 0.0)
        }}

        return params

    @staticmethod
    def _create_fewshot_dataset(x: np.ndarray, y: np.ndarray, n_class: int, n_support: int, n_query: int, experiment_name: str, n_task: Optional[int] = None, n_unlabeled: Optional[int] = None) -> Union[FewShotDataset, FewShotSSLFileDataset]:
        
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
    

        augment_data_file = ContrastNetWrapper._get_paraphrase_filename(
            experiment_name=experiment_name
        )

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

    @staticmethod
    def _get_paraphrase_filename(experiment_name: str) -> str:

        name = None

        if "banking" in experiment_name.lower():
            name = 'BANKING77'

        if "hwu" in experiment_name.lower():
            name = 'HWU64'

        if "clinc" in experiment_name.lower():
            name = 'OOS'

        assert name is not None, f"Experiment {experiment_name} not supported."

        filename = str(Directory.SRC / os.path.join(
            'ml',
            'classifier',
            'benchmark',
            'contrastnet',
            'data',
            name,
            'paraphrases',
            'DBS-unigram-flat-1.0',
            'paraphrases.json'
        ))

        return filename


    @staticmethod
    def _get_tuned_model_name(experiment_name: str) -> str:

        if "banking" in experiment_name.lower():
            return "tdopierre/ProtAugment-LM-BANKING77"

        if "hwu" in experiment_name.lower():
            return "tdopierre/ProtAugment-LM-HWU64"

        if "clinc" in experiment_name.lower():
            return "tdopierre/ProtAugment-LM-Clinic150"
        
        raise ValueError(f"Unknown experiment name: {experiment_name}")

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):

        assert "experiment_name" in kwargs.keys(), "Experiment name not provided."
        exp_name = kwargs["experiment_name"]

        if len(x_train.shape) == 2:
            x_train = x_train.reshape(-1)
        if len(x_valid.shape) == 2:
            x_valid = x_valid.reshape(-1)

        assert self.n_query == self.n_support, "N_query and N_support must be equal. Currently they are {} and {}".format(self.n_query, self.n_support)

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
            experiment_name=exp_name,
        )

        valid_dataset = self._create_fewshot_dataset(
            x_valid,
            y_valid,
            n_class=self.n_classes,
            n_support=self.n_support,
            n_query=self.n_query,
            experiment_name=exp_name,
        )

        model_name_or_path = self.model_name_or_path
        if self.model_name_or_path is None:

            model_name_or_path = self._get_tuned_model_name(
                experiment_name=exp_name
            )

        model = ContrastNet(
            config_name_or_path=model_name_or_path,
            metric='euclidean',
            max_len=self.max_len, 
            super_tau=self.super_tau, 
            unsuper_tau=self.unsuper_tau, 
            task_tau=self.task_tau
        )

        optimizer = Adam(model.parameters(), lr=self.lr)
        pbar = tqdm(range(self.max_iter))

        checkpoint_file = tempfile.NamedTemporaryFile().name + "_contrastnet_checkpoint.pth"

        patience = self.max_iter if self.patience is None else self.patience

        # prepare early stopping
        early_stopping = EarlyStopping(
            patience=patience,
            delta=0,
            path=checkpoint_file,
            verbose=True,
            trace_func=pbar.write
        )
        
        for step in pbar:

            episode = train_dataset.get_episode()

            episode_str = ','.join([f"{k}: {len(episode[k])}" for k in episode])
        
            # why do i need them?
            classes = []
            for i in episode['xs']:
                classes.append(i[0]['label'])

            supervised_loss_share = self.super_weight*(1. - step/self.max_iter)
            task_loss_share = self.task_weight

            loss, _ = model.train_step(optimizer=optimizer, episode=episode, supervised_loss_share=supervised_loss_share, task_loss_share=task_loss_share)
            
            if step % self.evaluate_every != 0:
                continue
            
            loss_validation = model.test_step(dataset=valid_dataset, n_episodes=self.n_test_episodes)
            loss_validation = loss_validation['loss']
            
            # validation has no task loss
            pbar.write(f"Step {step}: {episode_str}, Train Loss: {loss:.4f}, Validation Loss: {loss_validation:.4f}")

            early_stopping(loss_validation, model)

            if early_stopping.early_stop:
                break

        # load the last checkpoint with the best model
        model.load_state_dict(
            torch.load(checkpoint_file)
        )

        self.model = model
        self.x_train = self.model.encode(x_train.tolist())
        self.y_train = y_train
        self.label2idx, self.idx2label = LabellingUtilities.create_label_mapping(y=y_train)
        
        return None

    def _predict_non_episodic(self, x: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:

        if self.model is None:
            raise ValueError("Model has not been fitted yet")

        y_train = LabellingUtilities.map_labels(y=self.y_train, mapping=self.label2idx, target_dtype='int', unknown_value=UnknownClassLabel.UNKNOWN_NUM.value)
        
        assert isinstance(self.y_train, np.ndarray), "Labels must be a numpy array"
        y_train = torch.from_numpy(y_train).to(get_device())
        
        self.model.eval()

        predictions = []
        predictions_proba = []

        with torch.no_grad():
            x_query = self.model.encode(x.tolist())
            distances = torch.cdist(x_query, self.x_train)

            for i in range(len(x_query)):

               k_smallest_distances, k_nearest_indices = torch.topk(-distances[i], k=self.k_nearest_neighbors, largest=True)

               # avoid division by zero by adding a small epsilon to the distances
               epsilon = 1e-6
               k_smallest_distances += epsilon

               # calculate the weights
               weights = 1 / k_smallest_distances

               # normalize the weights
               weights /= torch.sum(weights)

               # Gather labels for the nearest neighbors
               nearest_labels = y_train[k_nearest_indices]
                
               # Accumulate weighted votes for each class
               unique_labels, counts = torch.unique(nearest_labels, return_counts=True)
               weighted_votes = {label.item(): 0.0 for label in unique_labels}
                
               for label, weight in zip(nearest_labels.tolist(), weights.tolist()):
                   weighted_votes[label] += weight

               # obtain the class with the highest weighted vote
               total_weight = sum(weighted_votes.values())
               class_probabilities = {label: weight / total_weight for label, weight in weighted_votes.items()}
               proba_vector = torch.zeros(len(torch.unique(y_train)))
               for label, proba in class_probabilities.items():
                   proba_vector[label] = proba

               predictions_proba.append(proba_vector)

               # Predict the class with the highest weighted vote
               predicted_class = max(weighted_votes, key=lambda x: weighted_votes[x])
               predictions.append(predicted_class)            

        y_pred = np.array([self.idx2label[prediction] for prediction in predictions])
        y_pred_proba = np.array(predictions_proba)
        return y_pred, y_pred_proba

    def get_support_examples(self, x_train: torch.Tensor, y_train: np.ndarray, n_way: int = 5, n_shot: int = 1) -> Tuple[torch.Tensor, np.ndarray]:

        # Get the indices of the nway classes
        class_indices = np.random.choice(np.unique(y_train), n_way, replace=False)

        # Get the indices of the nshot samples per class
        support_indices = []

        for class_index in class_indices:
            class_indices_in_class = np.where(y_train == class_index)[0]
            selected_indices = np.random.choice(class_indices_in_class, n_shot, replace=False)
            support_indices.extend(selected_indices)

        # Get the support examples
        x_support = x_train[support_indices]
        y_support = y_train[support_indices]

        assert isinstance(x_support, torch.Tensor)
        assert isinstance(y_support, np.ndarray)
        assert len(x_support) == len(y_support)
        assert len(x_support) == n_shot * n_way

        return x_support, y_support

    def _predict_episodic(self, x: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        self.model.eval()

        all_predictions = []

        query_embeddings = self.model.encode(x.tolist())

        # Run multiple episodes for more stable predictions                                                                                                                                                                                                                                                                           
        for _ in range(self.n_test_episodes):

            # create support examples from training data
            x_support, y_support = self.get_support_examples(x_train=self.x_train, y_train=self.y_train) 

            # Calculate distances                                                                                                                                                                                                                                                                                                     
            distances = torch.cdist(query_embeddings, x_support)                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                    
            # For each query, find nearest support example                                                                                                                                                                                                                                                                            
            _, nearest_indices = torch.min(distances, dim=1)                                                                                                                                                                                                                                                                          
            predictions = [y_support[i] for i in nearest_indices]                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                    
            all_predictions.append(predictions)

        # Majority vote across episodes                                                                                                                                                                                                                                                                                               
        final_predictions = []
        for i in range(len(x)):
            votes = np.array(all_predictions)[:,i]
            counter = Counter(votes)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
            final_predictions.append(counter.most_common(1)[0][0])
                                                                                                                                                                                                                                                                                                                            
        return np.array(final_predictions), np.array(final_predictions)
    
    def _predict(self, x: np.ndarray, mode: str = 'non-episodic') -> Tuple[np.ndarray, np.ndarray]:

        if mode == 'non-episodic':
            return self._predict_non_episodic(x)

        elif mode == 'episodic':
            return self._predict_episodic(x)

        else:
            raise ValueError("Invalid mode. Choose from 'non-episodic' or 'episodic'")

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

        # if y_pred_proba is proba estimate for each class, choose maximum probability and 
        # invert it to get outlier scores
        if y_pred_proba.ndim == 2:
            y_pred_proba = np.max(y_pred_proba, axis=1)
        elif y_pred_proba.ndim == 1:
            pass
        else:
            raise ValueError(f"Unexpected shape of y_pred_proba: {y_pred_proba.shape}")
        
        assert len(y_pred) == len(y_pred_proba)
        assert len(y_pred) == len(x)

        outlier_scores = -y_pred_proba
        y_pred[outlier_scores > self.unknown_threshold] = UnknownClassLabel.UNKNOWN_STR.value

        if include_outlierscore:
            return y_pred, outlier_scores

        return y_pred
        
    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise ValueError("Not implemented yet")

    
    