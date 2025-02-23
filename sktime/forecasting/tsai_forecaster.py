# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class and concrete forecasters for tsai deep learning models."""

__author__ = ["joseph-c-mcguire"]
__all__ = ["BaseTsaiForecaster", "PatchTSTForecaster", "RocketForecaster"]

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._delegate import _DelegatedForecaster

# Lazy imports of heavy dependencies
tsai = None


class _TsaiWrapperForecaster(BaseForecaster):
    """Internal wrapper that directly uses tsai code."""

    def __init__(
        self,
        arch,
        batch_size=64,
        max_epochs=100,
        learning_rate=1e-3,
        optimizer="Adam",
        loss="MSELoss",
        device=None,
        num_workers=0,
        model_type=None,
        scorer=None,
    ):
        # Store hyperparameters
        self.arch = arch
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.num_workers = num_workers
        self.model_type = model_type
        self.scorer = scorer
        super().__init__()

    def _fit(self, y, X, fh):
        self._initialize_model()

        # Setup tsai model, transforms, learner
        tfms = [None, "TSRegression"]
        batch_tfms = "TSStandardize"

        # Create TSRegressor with provided architecture
        from tsai.learner import TSRegressor

        self.learner_ = TSRegressor(
            X if X is not None else y,  # Input features
            y,  # Target values
            arch=self.arch,
            batch_size=self.batch_size,
            tfms=tfms,
            batch_tfms=batch_tfms,
        )

        # Train the model
        self.learner_.fit_one_cycle(self.max_epochs, self.learning_rate)
        return self

    def _predict(self, fh, X):
        preds, *_ = self.learner_.get_X_preds(X if X is not None else self._y)
        return preds

    def _initialize_model(self):
        """Initialize the tsai model."""
        global tsai
        if tsai is None:
            try:
                from tsai.all import TSRegressor, TSStandardize, TSRegression  # noqa
            except ImportError as e:
                raise ModuleNotFoundError(
                    "tsai is required for tsai forecasters. "
                    "Install tsai using: pip install tsai"
                ) from e


class BaseTsaiForecaster(_DelegatedForecaster):
    """Base forecaster that delegates to an internal _TsaiWrapperForecaster."""
    _delegate_name = "estimator_"

    def __init__(
        self,
        arch,
        batch_size=64,
        max_epochs=100,
        learning_rate=1e-3,
        optimizer="Adam",
        loss="MSELoss",
        device=None,
        num_workers=0,
        model_type=None,
        scorer=None,
    ):
        self.estimator_ = _TsaiWrapperForecaster(
            arch=arch,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss=loss,
            device=device,
            num_workers=num_workers,
            model_type=model_type,
            scorer=scorer,
        )
        super().__init__()
        self._set_delegated_tags(self.estimator_)


class PatchTSTForecaster(BaseTsaiForecaster):
    """PatchTST forecaster from tsai, delegated via BaseTsaiForecaster."""

    _tags = {
        "authors": ["timeseriesAI", "Nie2022"],
        "capability:pred_int": True,
    }

    def __init__(self, batch_size=64, max_epochs=100, learning_rate=1e-3, **kwargs):
        super().__init__(
            arch="PatchTSTRegressor",
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


class RocketForecaster(BaseTsaiForecaster):
    """ROCKET/MiniROCKET forecaster from tsai, delegated via BaseTsaiForecaster."""

    _tags = {
        "authors": ["timeseriesAI", "Dempster2020"],
        "capability:pred_int": False,
    }

    def __init__(self, model_type="minirocket", batch_size=64, max_epochs=100, learning_rate=1e-3, **kwargs):
        scorer = None  # or any necessary tsai scorer
        super().__init__(
            arch=None,  # actual arch can be handled in the wrapper by model_type
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            model_type=model_type,
            scorer=scorer,
            **kwargs
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return {
            "model_type": "minirocket",
        }
