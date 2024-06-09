import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pydantic import BaseModel, validator, validate_call
from pydantic.v1 import validate_arguments
from typing import Optional, Any

from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.util.torch_util import TorchMixin
from src.ml.classifier.util.torch_early_stopping import EarlyStopping
from src.util.dynamic_import import DynamicImport 
from src.util.dict_extraction import DictExtraction
from src.util.constants import UnknownClassLabel

# set random seed for reproducibility
torch.manual_seed(0)


class NN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = 128, dropout_prob: Optional[float] = None):
        
        super(NN, self).__init__()

        hidden_dim = (input_dim + output_dim) // 2 if hidden_dim is None else hidden_dim
        dropout_prob = 0.2 if dropout_prob is None else dropout_prob

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class MLP(BaseModel, TorchMixin, BaseClassifier):

    # training related parameters
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int 

    # parameters for NeuralNetwork
    loss_func_clf: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    
    device: torch.device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # placeholders, will be set later on
    model: Optional[torch.nn.Module] = None
    label2idx: dict = {}
    idx2label: dict = {}
    optimizer: Optional[optim.Adam] = None
    classes: Optional[np.ndarray] = None

    # filenames for plots and checkpoints
    filename_loss: str = "loss_mlp.png"
    filename_checkpoint: str = "checkpoint_mlp_model.pth"

    class Config:
        arbitrary_types_allowed = True
    
    @validator("model")
    def _set_model(cls, v):
        name, params = DictExtraction.get_class_obj_and_params(dictionary=v)
        return DynamicImport.init_class(name=name, params=params)

    def _train(
        self,
        model,
        device,
        train_loader,
        validation_loader,
        optimizer,
        epoch,
    ):

        model.train()

        train_loss = []

        for batch_idx, (data, labels) in enumerate(train_loader):

            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            # Vorwärtsdurchlauf
            outputs = model(data)
            labels_long = labels.long()
            loss = self.loss_func_clf(outputs, labels_long)
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                print(
                    "Epoch {} Iteration {}: Loss = {}".format(
                        epoch, batch_idx, loss
                    )
                )

            train_loss.append(loss.cpu().detach().numpy())

        # calculate validation loss
        model.eval()

        valid_loss = []

        with torch.no_grad():

            for _, (data, labels) in enumerate(validation_loader):

                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                labels_long = labels.long()
                loss = self.loss_func_clf(outputs, labels_long)
                valid_loss.append(loss.cpu().detach().numpy())

        return np.mean(np.array(train_loss)), np.mean(np.array(valid_loss))

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def _create_label_mapping(self, y: np.ndarray):

        self.label2idx = {label: idx for idx, label in enumerate(np.unique(y))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def _map_labels(self, y: np.ndarray, mapping: dict, target_dtype: str, unknown_value: Any) -> np.ndarray:
        
        dtype_mapping = {
            'int': int,
            'str': np.object_
        }   

        dtype_final = dtype_mapping[target_dtype]

        y_relabelled = np.copy(y)
        y_relabelled = y_relabelled.astype(np.object_)

        mask_transformed = np.zeros_like(y)

        for key, value in mapping.items():
            mask_treatedformed_sub = y == key
            y_relabelled[mask_treatedformed_sub] = value
            mask_transformed[mask_treatedformed_sub] = 1

        y_relabelled[mask_transformed == 0] = unknown_value

        y_relabelled = y_relabelled.astype(dtype_final)
        
        return y_relabelled


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
        self._create_label_mapping(y=y_train)

        y_train = self._map_labels(y=y_train, mapping=self.label2idx, target_dtype='int', unknown_value=UnknownClassLabel.UNKNOWN_NUM.value)
        y_valid = self._map_labels(y=y_valid, mapping=self.label2idx, target_dtype='int', unknown_value=UnknownClassLabel.UNKNOWN_NUM.value)

        # exclude unknown class
        mask_valid = y_valid != UnknownClassLabel.UNKNOWN_NUM.value
        x_valid = x_valid[mask_valid]
        y_valid = y_valid[mask_valid]

        # record classes
        self.classes = torch.Tensor(np.unique(y_train)).to(self.device)

        # create model
        self.model = NN(input_dim=x_train.shape[1], output_dim=len(self.classes))

        # add new layer to network
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # prepare early stopping
        early_stopping = EarlyStopping(
            patience=self.patience,
            delta=0,
            path=self.filename_checkpoint,
            verbose=True,
        )

        # create data loader
        train_loader = self._get_data_loader(
            x=x_train, y=y_train, batch_size=self.batch_size
        )
        valid_loader = self._get_data_loader(
            x=x_valid, y=y_valid, batch_size=self.batch_size
        )

        # epoch results
        epoch_train_results = []
        epoch_valid_results = []

        for epoch in range(1, self.epochs + 1):

            train_results, valid_results = self._train(
                model=self.model,
                device=self.device,
                train_loader=train_loader,
                validation_loader=valid_loader,
                optimizer=self.optimizer,
                epoch=epoch,
            )

            epoch_train_results.append(train_results)
            epoch_valid_results.append(valid_results)

            early_stopping(valid_results, self.model)

            if early_stopping.early_stop:
                break

        # load the last checkpoint with the best model
        self.model.load_state_dict(
            torch.load(self.filename_checkpoint)
        )

        self._plot_loss(epoch_train_results, epoch_valid_results, filename="loss.png")


    @validate_arguments(config={"arbitrary_types_allowed": True})
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:

        if self.model is None:
            raise ValueError("Model not fitted")

        x = torch.Tensor(x).to("cpu")
        if isinstance(self.model, torch.nn.Module):
            self.model = self.model.to("cpu")
        
        with torch.no_grad():

            self.model.eval()
            clf_output = self.model(x)
            y_pred = torch.argmax(clf_output, dim=1)
            y_pred = y_pred.numpy()
        
        y_pred = self._map_labels(y=y_pred, mapping=self.idx2label, target_dtype='str', unknown_value=UnknownClassLabel.UNKNOWN_STR.value)

        return y_pred

    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("Not implemented yet")


__all__ = ["MLP"]