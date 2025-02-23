"""Base class and concrete classifiers for tsai deep learning models."""

__author__ = ["joseph-c-mcguire"]
__all__ = ["BaseTsaiClassifier", "InceptionTimePlus",
           "TST", "XceptionTime", "PatchTST"]

from typing import Optional

import numpy as np

from sktime.classification._delegate import _DelegatedClassifier
from sktime.classification.base import BaseClassifier

# Lazy imports of heavy dependencies
tsai = None


class _TsaiWrapperClassifier(BaseClassifier):
    """_TsaiWrapperClassifier is a wrapper for the tsai library's TSClassifier,
     providing a scikit-learn compatible interface for time series classification.

     arch : callable
         The architecture to be used by the TSClassifier.
     batch_size : int, default=64
         The number of samples per gradient update.
     max_epochs : int, default=100
         The maximum number of epochs to train the model.
     learning_rate : float, default=1e-3
         The learning rate for the optimizer.
     optimizer : str, default="Adam"
         The optimizer to be used during training.
     loss : str, default="CrossEntropyLoss"
         The loss function to be used during training.
     device : str or None, default=None
         The device to be used for training (e.g., 'cpu', 'cuda').
     num_workers : int, default=0
         The number of subprocesses to use for data loading.
     Methods
     _fit(X, y)
         Fit the classifier on training data.
     _predict(X)
         Predict class labels for the input data.
     _predict_proba(X)
         Predict class probabilities for the input data.
    """

    def __init__(
        self,
        arch: callable,
        batch_size: int = 64,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
        loss: str = "CrossEntropyLoss",
        device: Optional[str] = None,
        num_workers: int = 0,
    ):
        """
        Initialize the TsaiClassifier.

        Parameters
        ----------
        arch : callable
            The architecture of the neural network.
        batch_size : int, optional (default=64)
            The number of samples per batch.
        max_epochs : int, optional (default=100)
            The maximum number of epochs for training.
        learning_rate : float, optional (default=1e-3)
            The learning rate for the optimizer.
        optimizer : str, optional (default="Adam")
            The optimizer to use for training.
        loss : str, optional (default="CrossEntropyLoss")
            The loss function to use for training.
        device : str or torch.device, optional
            The device to use for training (e.g., 'cpu' or 'cuda').
        num_workers : int, optional (default=0)
            The number of worker threads to use for data loading.
        """
        # Store hyperparameters
        self.arch = arch
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.num_workers = num_workers
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> "_TsaiWrapperClassifier":
        """Fit the classifier on training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape [n_instances, n_channels, series_length]
            Input training data
        y : 1D np.array
            Target values
        """
        global tsai
        if tsai is None:
            try:
                from tsai.all import TSClassifier, TSStandardize, TSClassification  # noqa
            except ImportError as e:
                raise ModuleNotFoundError(
                    "tsai is required for tsai classifiers. "
                    "Install tsai using: pip install tsai"
                ) from e

        # Setup tsai model, transforms, learner
        tfms = [None, "TSClassification"]
        batch_tfms = "TSStandardize"

        # Create TSClassifier with provided architecture
        from tsai.learner import TSClassifier

        self.model_ = TSClassifier(
            X,
            y,
            arch=self.arch,
            batch_size=self.batch_size,
            tfms=tfms,
            batch_tfms=batch_tfms,
        )

        # Train the model
        self.model_.fit_one_cycle(self.max_epochs, self.learning_rate)
        return self

    def _predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : 3D np.ndarray [n_instances, n_channels, series_length]
            Input data to predict

        Returns
        -------
        y_pred : 1D np.array
            Predicted class labels
        """
        _, _, preds = self.model_.get_X_preds(X)
        return preds

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : 3D np.ndarray [n_instances, n_channels, series_length]
            Input data

        Returns
        -------
        y_proba : 2D np.array [n_instances, n_classes]
            Predicted probabilities using classes_ ordering
        """
        probas, _, _ = self.model_.get_X_preds(X)
        return probas


class BaseTsaiClassifier(_DelegatedClassifier):
    """
    BaseTsaiClassifier is a wrapper for the Tsai deep learning library's classifier.

    Parameters
    ----------
    arch : callable
        The architecture of the neural network to be used.
    batch_size : int, default=64
        The number of samples per gradient update.
    max_epochs : int, default=100
        The maximum number of epochs to train the model.
    learning_rate : float, default=1e-3
        The learning rate for the optimizer.
    optimizer : str, default="Adam"
        The optimizer to be used for training the model.
    loss : str, default="CrossEntropyLoss"
        The loss function to be used for training the model.
    device : str or torch.device, optional
        The device on which to train the model. If None, defaults to the best available device.
    num_workers : int, default=0
        The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.

    Attributes
    ----------
    estimator_ : _TsaiWrapperClassifier
        The underlying Tsai classifier instance.
    """
    _delegate_name = "estimator_"

    def __init__(
        self,
        arch: callable,
        batch_size: int = 64,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
        loss: str = "CrossEntropyLoss",
        device: Optional[str] = None,
        num_workers: int = 0,
    ):
        """
        Initialize the TsaiClassifier.

        Parameters
        ----------
        arch : callable
            The architecture of the model to be used.
        batch_size : int, optional (default=64)
            The number of samples per batch.
        max_epochs : int, optional (default=100)
            The maximum number of epochs for training.
        learning_rate : float, optional (default=1e-3)
            The learning rate for the optimizer.
        optimizer : str, optional (default="Adam")
            The optimizer to be used for training.
        loss : str, optional (default="CrossEntropyLoss")
            The loss function to be used for training.
        device : Optional[str], optional
            The device to be used for training (e.g., 'cpu', 'cuda').
        num_workers : int, optional (default=0)
            The number of worker threads to use for data loading.
        """
        self.estimator_ = _TsaiWrapperClassifier(
            arch=arch,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss=loss,
            device=device,
            num_workers=num_workers,
        )
        super().__init__()
        self._set_delegated_tags(self.estimator_)


class InceptionTimePlus(BaseTsaiClassifier):
    """InceptionTime classifier from tsai."""

    _tags = {
        "authors": ["timeseriesAI", "Fawaz2019"]
    }

    def __init__(self, batch_size: int = 64, max_epochs: int = 100, learning_rate: float = 1e-3, **kwargs):
        """
        Initialize the TsaiClassifier.

        Parameters
        ----------
        batch_size : int, optional (default=64)
            The number of samples per batch.
        max_epochs : int, optional (default=100)
            The maximum number of epochs for training.
        learning_rate : float, optional (default=1e-3)
            The learning rate for the optimizer.
        **kwargs : dict
            Additional keyword arguments to pass to the parent class initializer.
        """
        super().__init__(
            arch="InceptionTimePlus",
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            **kwargs
        )


class TST(BaseTsaiClassifier):
    """Time Series Transformer (TST) classifier from tsai."""

    _tags = {
        "authors": ["timeseriesAI", "Zerveas2020", "commitUser2"]
    }

    def __init__(self, batch_size=64, max_epochs=100, learning_rate=1e-3, **kwargs):
        super().__init__(
            arch="TST",
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            **kwargs
        )


class XceptionTime(BaseTsaiClassifier):
    """XceptionTime classifier from tsai."""

    _tags = {
        "authors": ["timeseriesAI", "Rahimian2019", "commitUser3"]
    }

    def __init__(self, batch_size=64, max_epochs=100, learning_rate=1e-3, **kwargs):
        super().__init__(
            arch="XceptionTime",
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            **kwargs
        )


class PatchTST(BaseTsaiClassifier):
    """PatchTST classifier from tsai."""

    _tags = {
        "authors": ["timeseriesAI", "Nie2022", "commitUser4"]
    }

    def __init__(self, batch_size=64, max_epochs=100, learning_rate=1e-3, **kwargs):
        super().__init__(
            arch="PatchTST",
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            **kwargs
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return {
            "max_epochs": 1,
            "batch_size": 4,
        }
