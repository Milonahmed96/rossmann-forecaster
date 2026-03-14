"""
tests/test_metrics.py
---------------------
Unit tests for evaluation/metrics.py
"""

import pytest
import numpy as np
from evaluation.metrics import rmspe, rmse, r2, evaluate_model, print_results


def make_perfect_predictions():
    """Y_pred == Y_true — perfect predictions."""
    y = np.log1p(np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0]))
    return y, y.copy()


def make_realistic_predictions():
    """Realistic predictions close to but not equal to true values."""
    y_true = np.log1p(np.array([5263.0, 6064.0, 4987.0, 5800.0, 5100.0]))
    y_pred = np.log1p(np.array([5100.0, 6200.0, 4800.0, 5900.0, 5050.0]))
    return y_true, y_pred


def test_rmspe_perfect_predictions():
    """RMSPE should be 0 for perfect predictions."""
    y_true, y_pred = make_perfect_predictions()
    assert rmspe(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)


def test_rmspe_positive_value():
    """RMSPE should be positive for imperfect predictions."""
    y_true, y_pred = make_realistic_predictions()
    assert rmspe(y_true, y_pred) > 0


def test_rmspe_handles_zero_actual():
    """RMSPE should not raise when actual sales contains zeros."""
    y_true = np.log1p(np.array([0.0, 1000.0, 2000.0]))
    y_pred = np.log1p(np.array([500.0, 1100.0, 1900.0]))
    result = rmspe(y_true, y_pred)
    assert np.isfinite(result)


def test_rmse_perfect_predictions():
    """RMSE should be 0 for perfect predictions."""
    y_true, y_pred = make_perfect_predictions()
    assert rmse(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)


def test_rmse_positive_value():
    """RMSE should be positive for imperfect predictions."""
    y_true, y_pred = make_realistic_predictions()
    assert rmse(y_true, y_pred) > 0


def test_rmse_in_euro_scale():
    """RMSE should be in original Euro scale, not log scale."""
    y_true = np.log1p(np.array([5000.0, 5000.0]))
    y_pred = np.log1p(np.array([4000.0, 4000.0]))
    result = rmse(y_true, y_pred)
    # Error should be ~1000 EUR, not ~0.18 (log scale)
    assert result > 500


def test_r2_perfect_predictions():
    """R² should be 1.0 for perfect predictions."""
    y_true, y_pred = make_perfect_predictions()
    assert r2(y_true, y_pred) == pytest.approx(1.0, abs=1e-6)


def test_r2_realistic_range():
    """R² should be between 0 and 1 for reasonable predictions."""
    y_true, y_pred = make_realistic_predictions()
    result = r2(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_evaluate_model_returns_dict():
    """evaluate_model should return dict with rmspe, rmse, r2 keys."""
    y_true, y_pred = make_realistic_predictions()
    results = evaluate_model(y_true, y_pred)
    assert "rmspe" in results
    assert "rmse" in results
    assert "r2" in results


def test_evaluate_model_values_consistent():
    """Values in evaluate_model dict should match individual functions."""
    y_true, y_pred = make_realistic_predictions()
    results = evaluate_model(y_true, y_pred)
    assert results["rmspe"] == pytest.approx(rmspe(y_true, y_pred))
    assert results["rmse"]  == pytest.approx(rmse(y_true, y_pred))
    assert results["r2"]    == pytest.approx(r2(y_true, y_pred))


def test_print_results_runs_without_error(capsys):
    """print_results should execute without raising."""
    train = {"rmspe": 0.34, "rmse": 1200.0, "r2": 0.93}
    test  = {"rmspe": 0.35, "rmse": 1058.0, "r2": 0.87}
    print_results("LightGBM", train, test)
    captured = capsys.readouterr()
    assert "LightGBM" in captured.out
    assert "RMSPE" in captured.out