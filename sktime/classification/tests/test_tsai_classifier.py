import pytest
import numpy as np
from sktime.classification.tsai_classifier import _TsaiModelClassifier, InceptionTimePlus, TST, XceptionTime, PatchTST


@pytest.fixture
def sample_data():
    X = np.random.rand(10, 50, 1)  # 10 samples, 50 time points, 1 feature
    y = np.random.randint(0, 2, 10)  # 10 binary labels
    return X, y


@pytest.fixture
def sample_data():
    X = np.random.rand(10, 50, 1)  # 10 samples, 50 time points, 1 feature
    y = np.random.randint(0, 2, 10)  # 10 binary labels
    return X, y


def test_tsai_network_initialization():
    network = _TsaiModelClassifier(arch="InceptionTimePlus")
    assert network.arch == "InceptionTimePlus"
    assert network.batch_size == 64
    assert network.max_epochs == 100
    assert network.learning_rate == 1e-3
    assert network.device is None
    assert network.num_workers == 0


def test_tsai_network_build_network():
    network = _TsaiModelClassifier(arch="InceptionTimePlus")
    model, history = network.build_network((50, 1))
    assert model is None
    assert history is None


def test_tsai_network_fit(sample_data):
    X, y = sample_data
    network = _TsaiModelClassifier(arch="InceptionTimePlus", max_epochs=1)
    network.fit(X, y)
    assert hasattr(network, 'model_')


def test_tsai_network_predict(sample_data):
    X, y = sample_data
    network = _TsaiModelClassifier(arch="InceptionTimePlus", max_epochs=1)
    network.fit(X, y)
    preds = network.predict(X)
    assert preds.shape == (10,)


def test_tsai_network_predict_proba(sample_data):
    X, y = sample_data
    network = _TsaiModelClassifier(arch="InceptionTimePlus", max_epochs=1)
    network.fit(X, y)
    probas = network.predict_proba(X)
    assert probas.shape == (10, 2)


def test_inception_time_plus_initialization():
    classifier = InceptionTimePlus()
    assert classifier.arch == "InceptionTimePlus"


def test_tst_initialization():
    classifier = TST()
    assert classifier.arch == "TST"


def test_xception_time_initialization():
    classifier = XceptionTime()
    assert classifier.arch == "XceptionTime"


def test_patch_tst_initialization():
    classifier = PatchTST()
    assert classifier.arch == "PatchTST"
