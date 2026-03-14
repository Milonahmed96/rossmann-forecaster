"""
tests/test_ridge_model.py
-------------------------
Unit tests for models/ridge_model.py
"""

import pytest
import numpy as np
from models.ridge_model import RidgeModel


def make_data(n=200):
    """Create simple synthetic regression data."""
    np.random.seed(42)
    X = np.random.randn(n, 10)
    Y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n) * 0.1
    return X, Y


def test_ridge_fit_and_predict():
    """Model should fit and return predictions of correct shape."""
    X, Y = make_data()
    model = RidgeModel()
    model.fit(X, Y)
    preds = model.predict(X)
    assert preds.shape == (200,)


def test_ridge_predict_before_fit_raises():
    """Calling predict before fit should raise RuntimeError."""
    model = RidgeModel()
    X, _ = make_data()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)


def test_ridge_tune_sets_best_alpha():
    """After tuning, best_alpha should be set."""
    X, Y = make_data()
    model = RidgeModel()
    model.tune(X, Y, n_iter=5, n_splits=2)
    assert model.best_alpha is not None
    assert model.best_alpha > 0


def test_ridge_tune_allows_predict():
    """After tuning, predict should work without calling fit separately."""
    X, Y = make_data()
    model = RidgeModel()
    model.tune(X, Y, n_iter=5, n_splits=2)
    preds = model.predict(X)
    assert preds.shape == (200,)


def test_ridge_default_alpha():
    """Default alpha should be 0.1 matching MSc notebook."""
    model = RidgeModel()
    assert model.alpha == 0.1


def test_ridge_predictions_are_finite():
    """Predictions should not contain NaN or infinity."""
    X, Y = make_data()
    model = RidgeModel()
    model.fit(X, Y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))