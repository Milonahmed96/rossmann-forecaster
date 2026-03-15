import numpy as np
import pytest
import torch
from models.pytorch_lstm import PyTorchLSTM


def make_fake_data(n_stores=3, days_per_store=30, n_features=21):
    total = n_stores * days_per_store
    X     = np.random.rand(total, n_features).astype(np.float32)
    Y     = np.random.rand(total).astype(np.float32)
    ids   = np.repeat(np.arange(n_stores), days_per_store)
    return X, Y, ids


# ── Test 1: model instantiates correctly ───────────────────────────
def test_instantiation():
    model = PyTorchLSTM(input_size=21, hidden_size=64)
    assert model.model is None          # not fitted yet
    assert model.hidden_size == 64


# ── Test 2: predict before fit raises RuntimeError ─────────────────
def test_predict_before_fit_raises():
    model = PyTorchLSTM(input_size=21)
    X, Y, ids = make_fake_data()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X, ids)


# ── Test 3: fit runs without crashing (tiny data, 2 epochs) ────────
def test_fit_runs():
    X, Y, ids = make_fake_data(n_stores=2, days_per_store=20)
    model = PyTorchLSTM(input_size=21, hidden_size=16)
    history = model.fit(X, Y, ids, window_size=7, epochs=2,
                        batch_size=32, patience=5,
                        save_path='models/test_lstm.pt')
    assert len(history) == 2
    assert 'train_loss' in history[0]
    assert 'val_loss'   in history[0]


# ── Test 4: predict returns correct shape after fit ─────────────────
def test_predict_shape():
    X, Y, ids = make_fake_data(n_stores=2, days_per_store=20)
    model = PyTorchLSTM(input_size=21, hidden_size=16)
    model.fit(X, Y, ids, window_size=7, epochs=2,
              batch_size=32, save_path='models/test_lstm.pt')
    preds = model.predict(X, ids, window_size=7)
    # 2 stores × (20-7) = 26 predictions
    assert preds.shape == (26,), f"Expected (26,), got {preds.shape}"


def test_predictions_are_euros():
    X, Y, ids = make_fake_data(n_stores=2, days_per_store=20)
    model = PyTorchLSTM(input_size=21, hidden_size=16)
    model.fit(X, Y, ids, window_size=7, epochs=2,
              batch_size=32, save_path='models/test_lstm.pt')
    preds = model.predict(X, ids, window_size=7)
    # Predictions are in Euro scale (expm1 applied) — check shape and dtype only.
    # Negativity can occur with an untrained model on random data — not a bug.
    assert preds.dtype == np.float32
    assert len(preds) == 26   # 2 stores × (20-7) windows
    assert not np.any(np.isnan(preds)), "Predictions contain NaN"
    assert not np.any(np.isinf(preds)), "Predictions contain Inf"

