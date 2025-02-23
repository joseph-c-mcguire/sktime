import numpy as np
import torch
from sktime.classification._delegate import _DelegatedClassifier


class _TsaiModelClassifier(_DelegatedClassifier):
    """Base class for TSAI model classifiers.

    Parameters
    ----------
    arch : str
        TSAI model architecture name
    batch_size : int, default=64
        Batch size for training
    max_epochs : int, default=100
        Number of epochs
    learning_rate : float, default=1e-3
        Learning rate
    device : str or None, optional
        Device to train on, e.g. "cuda" or "cpu"
    num_workers : int, default=0
        Number of workers for data loading
    """

    _tags = {
        "authors": ["oguiza"],
        "python_dependencies": ["tsai", "torch"],
    }


class MiniRocketClassifier(_TsaiModelClassifier):
    """Time series classification using MINIROCKET features and a linear classifier.

    Recommended for datasets up to 10k time series. For larger datasets, 
    consider using the PyTorch MINIROCKET implementation.

    Parameters
    ----------
    num_features : int, default=10000
        Number of features for MINIROCKET transform
    max_dilations_per_kernel : int, default=32
        Maximum number of dilations per kernel
    random_state : int, optional (default=None)
        Random state for reproducibility
    alphas : array-like, default=np.logspace(-3, 3, 7)
        Array of alpha values to try for linear classifier
    normalize_features : bool, default=True
        Whether to normalize features
    memory : str or None, default=None
        Memory parameter
    verbose : bool, default=False
        Whether to print progress messages
    scoring : str or None, default=None
        Scoring metric (defaults to accuracy if None)
    class_weight : dict or 'balanced' or None, default=None
        Class weights for linear classifier
    batch_size : int, default=64
        Batch size for training
    max_epochs : int, default=100
        Number of epochs
    learning_rate : float, default=1e-3
        Learning rate
    device : str or None, optional
        Device to train on, e.g. "cuda" or "cpu"
    num_workers : int, default=0
        Number of workers for data loading
    """

    def __init__(
        self,
        num_features=10000,
        max_dilations_per_kernel=32,
        random_state=None,
        alphas=np.logspace(-3, 3, 7),
        normalize_features=True,
        memory=None,
        verbose=False,
        scoring=None,
        class_weight=None,
        **kwargs
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.alphas = alphas
        self.normalize_features = normalize_features
        self.memory = memory
        self.verbose = verbose
        self.scoring = scoring
        self.class_weight = class_weight
        # Lazy import to avoid hard dependency on tsai
        import tsai
        self.arch = arch
        # Check if the architecture is valid
        if not hasattr(tsai.models, self.arch):
            raise ValueError(f"Invalid TSAI model architecture: {self.arch}")
        # Initialize the classifier
        super().__init__()
        # Gets the classifier from TSAI
        self.estimator_ = getattr(tsai.models, self.arch)
        self.estimator_ = self.estimator_(**kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class
        """
        params1 = {
            "num_features": 100,  # Small number for testing
            "max_epochs": 1,
            "batch_size": 4,
        }

        params2 = {
            "num_features": 50,
            "max_dilations_per_kernel": 4,
            "random_state": 42,
            "normalize_features": False,
            "max_epochs": 1,
            "batch_size": 4,
        }

        return [params1, params2]


class InceptionTimePlus(_TsaiModelClassifier):
    """InceptionTimePlus classifier using tsai."""

    def __init__(self, **kwargs):
        super().__init__(arch="InceptionTimePlus", **kwargs)


class TST(_TsaiModelClassifier):
    """TST (Time Series Transformer) classifier using tsai."""

    def __init__(self, **kwargs):
        super().__init__(arch="TST", **kwargs)


class XceptionTime(_TsaiModelClassifier):
    """XceptionTime classifier using tsai."""

    def __init__(self, **kwargs):
        super().__init__(arch="XceptionTime", **kwargs)


class PatchTST(_TsaiModelClassifier):
    """PatchTST classifier using tsai."""

    def __init__(self, **kwargs):
        super().__init__(arch="PatchTST", **kwargs)
