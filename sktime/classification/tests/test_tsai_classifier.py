"""Tests for tsai classifier interfaces."""

__author__ = ["joseph-c-mcguire"]

import numpy as np
import pytest

from sktime.classification.tsai_classifier import (
    InceptionTimePlus,
    PatchTST,
    TST,
    XceptionTime,
)
from sktime.datasets import load_unit_test
from sktime.tests.test_all_estimators import run_common_tests


@pytest.mark.skipif(
    not run_common_tests(estimator=InceptionTimePlus, severity="min"),
    reason="Common tests failed with min severity level",
)
def test_tsai_inception():
    """Test InceptionTimePlus classifier."""
    X, y = load_unit_test(split="train", return_X_y=True)
    clf = InceptionTimePlus(max_epochs=1)
    clf.fit(X, y)
    return clf.predict(X)


@pytest.mark.skipif(
    not run_common_tests(estimator=TST, severity="min"),
    reason="Common tests failed with min severity level",
)
def test_tsai_tst():
    """Test TST classifier."""
    X, y = load_unit_test(split="train", return_X_y=True)
    clf = TST(max_epochs=1)
    clf.fit(X, y)
    return clf.predict(X)


@pytest.mark.skipif(
    not run_common_tests(estimator=XceptionTime, severity="min"),
    reason="Common tests failed with min severity level",
)
def test_tsai_xception():
    """Test XceptionTime classifier."""
    X, y = load_unit_test(split="train", return_X_y=True)
    clf = XceptionTime(max_epochs=1)
    clf.fit(X, y)
    return clf.predict(X)


@pytest.mark.skipif(
    not run_common_tests(estimator=PatchTST, severity="min"),
    reason="Common tests failed with min severity level",
)
def test_tsai_patchtst():
    """Test PatchTST classifier."""
    X, y = load_unit_test(split="train", return_X_y=True)
    clf = PatchTST(max_epochs=1)
    clf.fit(X, y)
    return clf.predict(X)
