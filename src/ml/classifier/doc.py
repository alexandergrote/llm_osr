import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pydantic import BaseModel, validator, validate_call
from pydantic.v1 import validate_arguments
from scipy.stats import norm
from typing import Optional, Dict

from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.util.torch_util import TorchMixin
from src.ml.classifier.util.torch_early_stopping import EarlyStopping
from src.util.dynamic_import import DynamicImport 
from src.util.dict_extraction import DictExtraction

# set random seed for reproducibility
torch.manual_seed(0)


class GaussianModel(BaseModel):

    mu: float
    std: float


class GaussianModels(BaseModel):

    models: Optional[Dict[int, GaussianModel]] = None
    classes: np.ndarray = np.empty(shape=(0,))

    class Config:
        arbitrary_types_allowed = True

    def fit_gaussian_models(
        self, y_train: np.ndarray, y_pred_proba: np.ndarray
    ):

        self.classes = np.unique(y_train)
        self.classes = self.classes.astype(np.intp)

        models = {}

        for c in self.classes:

            # get mask for probabilities if prediction is true
            mask = y_train == c

            # get subset of probabilities
            probabilities = list(y_pred_proba[mask])
            probabilities += [2 - prob for prob in probabilities]

            # get params of norm distribution
            mu, std = norm.fit(probabilities)

            # add model
            models[c] = GaussianModel(mu=mu, std=std)

        self.models = models

    def predict(self, y_pred_proba: np.ndarray) -> np.ndarray:

        if self.models is None:
            raise ValueError("Gaussian models not fitted")

        scale: float = 0.2
        y_pred = []

        for row in y_pred_proba:

            max_score = -1
            max_class = -1

            for class_idx in self.classes:

                threshold = max(0.5, 1 - scale * self.models[class_idx].std)
                class_proba = row[class_idx]

                # update max_score if class_proba is bigger than
                # threshold and previous max_score
                if class_proba > max(threshold, max_score):
                    max_score = class_proba
                    max_class = class_idx

            y_pred.append(max_class)

        return np.array(y_pred)


class NN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int = 128):
        super(NN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class DOC(BaseModel, TorchMixin, BaseClassifier):

    # training related parameters
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int 

    # parameters for NeuralNetwork
    loss_func_clf: nn.BCELoss = nn.BCELoss()
    sigmoid: nn.Sigmoid = nn.Sigmoid()
    device: torch.device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # placeholders, will be set later on
    model: Optional[torch.nn.Module] = None
    label2idx: dict = {}
    idx2label: dict = {}
    gaussian_models: Optional[GaussianModels] = None
    optimizer: Optional[optim.Adam] = None
    classes: Optional[np.ndarray] = None

    # filenames for plots and checkpoints
    filename_loss: str = "loss.png"
    filename_checkpoint: str = "checkpoint_doc_model.pth"

    class Config:
        arbitrary_types_allowed = True
    
    @validator("model")
    def _set_epochs(cls, v):
        name, params = DictExtraction.get_class_obj_and_params(dictionary=v)
        return DynamicImport.init_class(name=name, params=params)

    def _adding_layer_to_model(self, n_classes):

        incoming_number_features = self.model.model[-1].out_features

        self.model.model.add_module(
            "feature_layer", nn.Linear(incoming_number_features, n_classes)
        )

    def _calculate_loss(self, model, data, labels):

        # calculate classification loss
        clf_output = model(data)

        classification_loss = 0
        sigmoid = nn.Sigmoid()

        for c in np.arange(len(self.classes)):

            tgt_class_label = torch.eq(labels, c).float()

            class_loss = self.loss_func_clf(
                sigmoid(clf_output[:, c]), tgt_class_label
            )
            classification_loss += class_loss

        return classification_loss

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
            loss = self._calculate_loss(model=model, data=data, labels=labels)
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
                loss = self._calculate_loss(
                    model=model, data=data, labels=labels
                )
                valid_loss.append(loss.cpu().detach().numpy())

        return np.mean(np.array(train_loss)), np.mean(np.array(valid_loss))

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def _create_label_mapping(self, y: np.ndarray):

        self.label2idx = {label: idx for idx, label in enumerate(np.unique(y))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def _map_labels(self, y: np.ndarray, mapping: dict) -> np.ndarray:

        y_relabelled = np.copy(y)

        all_classes = np.unique(y)

        uuc = [u for u in all_classes if u not in mapping.keys()]

        for u in uuc:
            y_relabelled[y == u] = -1

        for label, index in mapping.items():
            y_relabelled[y == label] = index

        return y_relabelled

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def _fit_gaussians(self, x_train: np.ndarray, y_train: np.ndarray):

        if self.model is None:
            raise ValueError("Model not fitted")

        x_train = torch.Tensor(x_train).to(self.device)
        y_train = torch.Tensor(y_train).to(self.device)

        with torch.no_grad():

            self.model.eval()

            forward_pass = self.model(x_train)
            y_pred_proba = self.sigmoid(forward_pass)

        self.gaussian_models = GaussianModels()

        self.gaussian_models.fit_gaussian_models(
            y_train=y_train.cpu().numpy(),
            y_pred_proba=y_pred_proba.cpu().numpy(),
        )

        return self.gaussian_models

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

        # record classes
        self.classes = torch.Tensor(np.unique(y_train)).to(self.device)

        # create model
        self.model = NN(input_dim=x_train.shape[1])

        # add new layer to network
        self._adding_layer_to_model(n_classes=len(self.classes))
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # prepare label mapping
        self._create_label_mapping(y=y_train)

        y_train = self._map_labels(y=y_train, mapping=self.label2idx)
        y_valid = self._map_labels(y=y_valid, mapping=self.label2idx)

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

        self._fit_gaussians(x_train, y_train)

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:

        if self.model is None:
            raise ValueError("Model not fitted")

        if self.gaussian_models is None:
            raise ValueError("Gaussian models not fitted")

        x = torch.Tensor(x).to("cpu")
        if isinstance(self.model, torch.nn.Module):
            self.model = self.model.to("cpu")
        
        with torch.no_grad():

            self.model.eval()
            clf_output = self.model(x)
            clf_output = self.sigmoid(clf_output).numpy()

        y_pred = self.gaussian_models.predict(y_pred_proba=clf_output)
        y_pred = self._map_labels(y=y_pred, mapping=self.idx2label)

        return y_pred

    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("Not implemented yet")


__all__ = ["DOC"]