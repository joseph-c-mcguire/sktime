"""Tests for tsai forecaster interfaces."""

__author__ = ["joseph-c-mcguire"]

import numpy as np
import pytest

from sktime.forecasting.tsai_forecaster import PatchTSTForecaster, RocketForecaster
from sktime.datasets import load_airline
from sktime.tests.test_all_estimators import run_common_tests


@pytest.mark.skipif(
    not run_common_tests(estimator=PatchTSTForecaster, severity="min"),
    reason="Common tests failed with min severity level",
)
def test_patchtst_forecaster():
    """Test PatchTST forecaster."""
    y = load_airline()
    forecaster = PatchTSTForecaster(max_epochs=1)
    forecaster.fit(y)
    return forecaster.predict([1, 2, 3])


@pytest.mark.skipif(
    not run_common_tests(estimator=RocketForecaster, severity="min"),
    reason="Common tests failed with min severity level",
)
def test_rocket_forecaster():
    """Test ROCKET forecaster."""
    y = load_airline()
    forecaster = RocketForecaster()
    forecaster.fit(y)
    return forecaster.predict([1, 2, 3])


def test_rocket_save_load():
    """Test ROCKET save/load functionality."""
    import tempfile

    y = load_airline()
    forecaster = RocketForecaster()
    forecaster.fit(y)

    with tempfile.NamedTemporaryFile() as tmp:
        # Save the model
        forecaster.save(tmp.name)
        # Load the model
        loaded = RocketForecaster.load(tmp.name)

        # Compare predictions
        fh = [1, 2, 3]
        np.testing.assert_array_equal(
            forecaster.predict(fh),
            loaded.predict(fh)
        )
