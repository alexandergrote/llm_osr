import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from pydantic import validate_call

from sklearn.model_selection import StratifiedKFold

class StratifiedBatchSampler:
    """
    Stratified batch sampling
    Provides equal representation of target classes in each batch
    Taken from https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/7
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = max(2, int(len(y) / batch_size))

        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle, random_state=42)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for _, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


class TorchMixin:
    @validate_call(config={"arbitrary_types_allowed": True})
    def _get_dataset(self, x: np.ndarray, y: np.ndarray) -> TensorDataset:

        x = np.array([el for el in x])

        tensor_x = torch.Tensor(x)
        tensor_y = torch.Tensor(y)
        dataset = TensorDataset(tensor_x, tensor_y)

        return dataset

    @validate_call(config={"arbitrary_types_allowed": True})
    def _get_data_loader(
        self, x: np.ndarray, y: np.ndarray, batch_size: int
    ) -> DataLoader:

        dataset = self._get_dataset(x=x, y=y)

        dataset_loader = DataLoader(
            dataset,
            batch_sampler=StratifiedBatchSampler(
                y, batch_size=batch_size
            )
        )

        return dataset_loader

    def _plot_loss(self, train_results, valid_results, filename: str):
        epochs = np.array(range(len(train_results))) + 1
        plt.figure()
        plt.plot(epochs, train_results, label="train loss")
        plt.plot(epochs, valid_results, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.xticks(epochs)
        plt.savefig(filename)

