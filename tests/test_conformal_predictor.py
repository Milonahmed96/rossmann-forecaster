import numpy as np
import pytest
from models.conformal_predictor import ConformalPredictor


class FakeModel:
    """A simple fake model for testing — adds Gaussian noise to true values."""
    def predict(self, X, **kwargs):
        np.random.seed(0)
        return X[:, 0] + np.random.randn(len(X)) * 200


def make_fake_data(n=500, n_features=5):
    np.random.seed(42)
    X     = np.random.rand(n, n_features).astype(np.float32)
    # Y in log scale — expm1 will give Euro values around 0-6
    Y_log = np.random.rand(n).astype(np.float32) * 2
    return X, Y_log


# ── Test 1: calibrate stores q_ as a positive float ────────────────
def test_calibrate_stores_q():
    X, Y = make_fake_data()
    cp   = ConformalPredictor(FakeModel(), coverage=0.90)
    cp.calibrate(X, Y)
    assert cp.q_ is not None
    assert isinstance(cp.q_, float)
    assert cp.q_ > 0


# ── Test 2: predict_interval returns correct shapes ─────────────────
def test_predict_interval_shapes():
    X, Y = make_fake_data(n=200)
    cp   = ConformalPredictor(FakeModel(), coverage=0.90)
    cp.calibrate(X[:100], Y[:100])
    result = cp.predict_interval(X[100:])
    assert set(result.keys()) == {'point', 'lower', 'upper'}
    assert len(result['point']) == len(result['lower']) == len(result['upper']) == 100


# ── Test 3: lower bound is never negative ───────────────────────────
def test_lower_clipped_to_zero():
    X, Y = make_fake_data(n=400)
    cp   = ConformalPredictor(FakeModel(), coverage=0.90)
    cp.calibrate(X[:200], Y[:200])
    result = cp.predict_interval(X[200:])
    assert np.all(result['lower'] >= 0), "Lower bound went negative"


# ── Test 4: empirical coverage meets the guarantee ──────────────────
def test_empirical_coverage():
    np.random.seed(7)
    n = 2000
    X = np.random.rand(n, 3).astype(np.float32)
    # True values and predictions in log scale
    Y_log  = (np.random.rand(n) * 3).astype(np.float32)
    cp = ConformalPredictor(FakeModel(), coverage=0.90)
    cp.calibrate(X[:1000], Y_log[:1000])
    result = cp.evaluate(X[1000:], Y_log[1000:])
    assert result['coverage'] >= 0.85, \
        f"Coverage {result['coverage']:.3f} too low — conformal guarantee violated"


# ── Test 5: invalid coverage raises ValueError ──────────────────────
def test_invalid_coverage_raises():
    with pytest.raises(ValueError):
        ConformalPredictor(FakeModel(), coverage=1.5)
    with pytest.raises(ValueError):
        ConformalPredictor(FakeModel(), coverage=0.0)


# ── Test 6: predict before calibrate raises RuntimeError ────────────
def test_predict_before_calibrate_raises():
    X, _ = make_fake_data(n=50)
    cp   = ConformalPredictor(FakeModel(), coverage=0.90)
    with pytest.raises(RuntimeError, match="calibrate"):
        cp.predict_interval(X)
