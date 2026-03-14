"""
tests/test_lightgbm_model.py
-----------------------------
Unit tests for models/lightgbm_model.py
"""

import pytest
import numpy as np
from models.lightgbm_model import LightGBMModel


def make_data(n=300):
    """Create simple synthetic regression data."""
    np.random.seed(42)
    X = np.random.randn(n, 15).astype(np.float32)
    Y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n) * 0.1
    return X, Y


def test_lightgbm_fit_and_predict():
    """Model should fit and return predictions of correct shape."""
    X, Y = make_data()
    model = LightGBMModel()
    model.fit(X, Y)
    preds = model.predict(X)
    assert preds.shape == (300,)


def test_lightgbm_predict_before_fit_raises():
    """Calling predict before fit should raise RuntimeError."""
    model = LightGBMModel()
    X, _ = make_data()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)


def test_lightgbm_tune_sets_best_params():
    """After tuning, best_params should be set."""
    X, Y = make_data()
    model = LightGBMModel()
    model.tune(X, Y, n_iter=3, n_splits=2)
    assert model.best_params is not None
    assert "n_estimators" in model.best_params


def test_lightgbm_tune_allows_predict():
    """After tuning, predict should work without calling fit separately."""
    X, Y = make_data()
    model = LightGBMModel()
    model.tune(X, Y, n_iter=3, n_splits=2)
    preds = model.predict(X)
    assert preds.shape == (300,)


def test_lightgbm_default_params():
    """Default parameters should match MSc notebook baseline."""
    model = LightGBMModel()
    assert model.n_estimators == 100
    assert model.learning_rate == 0.1
    assert model.num_leaves == 31


def test_lightgbm_predictions_are_finite():
    """Predictions should not contain NaN or infinity."""
    X, Y = make_data()
    model = LightGBMModel()
    model.fit(X, Y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
