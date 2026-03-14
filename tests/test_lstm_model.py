"""
tests/test_lstm_model.py
------------------------
Unit tests for models/lstm_model.py

NOTE: Tests that require TensorFlow are skipped automatically
if TensorFlow is not installed (e.g. on Python 3.13).
Run these tests on Python 3.10 or Google Colab.
"""

import pytest
import numpy as np

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Skip all TF-dependent tests if TensorFlow not installed
requires_tf = pytest.mark.skipif(
    not TF_AVAILABLE,
    reason="TensorFlow not installed — run on Python 3.10 or Google Colab"
)

from models.lstm_model import LSTMModel


def make_data(n=200, n_features=10):
    """Create simple synthetic scaled data (values between 0 and 1)."""
    np.random.seed(42)
    X = np.random.rand(n, n_features).astype(np.float32)
    Y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n) * 0.05
    return X, Y


def test_lstm_model_instantiates():
    """LSTMModel should instantiate without TensorFlow."""
    model = LSTMModel()
    assert model.timesteps == 1
    assert model.units == 100
    assert model.dropout == 0.2
    assert model.epochs == 10
    assert model.batch_size == 256
    assert not model.is_fitted


def test_lstm_predict_before_fit_raises():
    """Calling predict before fit should raise RuntimeError."""
    model = LSTMModel()
    X, _ = make_data()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)


def test_lstm_reshape():
    """_reshape should convert (n, f) to (n, 1, f)."""
    model = LSTMModel(timesteps=1)
    X = np.random.rand(100, 15).astype(np.float32)
    reshaped = model._reshape(X)
    assert reshaped.shape == (100, 1, 15)


@requires_tf
def test_lstm_fit_and_predict():
    """Model should fit and return predictions of correct shape."""
    X, Y = make_data()
    model = LSTMModel(epochs=2)
    model.fit(X, Y)
    preds = model.predict(X)
    assert preds.shape == (200,)


@requires_tf
def test_lstm_predictions_are_finite():
    """Predictions should not contain NaN or infinity."""
    X, Y = make_data()
    model = LSTMModel(epochs=2)
    model.fit(X, Y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


@requires_tf
def test_lstm_fit_optimised():
    """Optimised model should fit and predict correctly."""
    X, Y = make_data()
    model = LSTMModel(epochs=2)
    model.fit_optimised(X, Y, patience=1)
    preds = model.predict(X)
    assert preds.shape == (200,)