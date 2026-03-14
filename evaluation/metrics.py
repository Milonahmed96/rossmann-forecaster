"""
evaluation/metrics.py
---------------------
Performance metrics for the Rossmann forecasting models.

All metrics operate on log-transformed predictions and targets,
applying np.expm1() to reverse the transformation before calculating
errors in the original sales scale (Euro).

Primary metric: RMSPE — scale-free, treats all stores equally regardless
of their sales volume. A store selling 1,000 EUR/day and one selling
10,000 EUR/day are weighted equally.

Secondary metrics:
    RMSE — absolute error in Euro, useful for business stakeholders
    R²   — goodness of fit, calculated on log-transformed values
"""

import numpy as np
from sklearn.metrics import r2_score


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Percentage Error.

    Primary metric for the Rossmann forecasting task. Scale-free —
    treats small and large stores equally. Lower is better.

    Parameters
    ----------
    y_true : np.ndarray
        True log-transformed sales values.
    y_pred : np.ndarray
        Predicted log-transformed sales values.

    Returns
    -------
    float
        RMSPE value. Perfect score = 0.0.

    Examples
    --------
    >>> rmspe(Y_test, Y_pred)
    0.3464
    """
    # Reverse log transformation to get original sales values
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)

    # Only calculate where actual sales are positive — avoids division by zero
    mask = y_true_orig > 0

    percentage_error = np.zeros_like(y_true_orig, dtype=float)
    percentage_error[mask] = (
        (y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask]
    )

    return float(np.sqrt(np.mean(np.square(percentage_error))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error in original Euro scale.

    Provides a tangible sense of prediction error in currency —
    useful for communicating results to non-technical stakeholders.

    Parameters
    ----------
    y_true : np.ndarray
        True log-transformed sales values.
    y_pred : np.ndarray
        Predicted log-transformed sales values.

    Returns
    -------
    float
        RMSE in Euro. Perfect score = 0.0.

    Examples
    --------
    >>> rmse(Y_test, Y_pred)
    1057.70
    """
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    return float(np.sqrt(np.mean(np.square(y_true_orig - y_pred_orig))))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (Coefficient of Determination).

    Calculated on log-transformed values. Values closer to 1.0
    indicate a better fit. Bundles overall model quality into
    a single interpretable number.

    Parameters
    ----------
    y_true : np.ndarray
        True log-transformed sales values.
    y_pred : np.ndarray
        Predicted log-transformed sales values.

    Returns
    -------
    float
        R² score. Perfect score = 1.0.

    Examples
    --------
    >>> r2(Y_test, Y_pred)
    0.8679
    """
    return float(r2_score(y_true, y_pred))


def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray) -> dict:
    """
    Calculate all three metrics and return as a dictionary.

    Parameters
    ----------
    y_true : np.ndarray
        True log-transformed sales values.
    y_pred : np.ndarray
        Predicted log-transformed sales values.

    Returns
    -------
    dict with keys: rmspe, rmse, r2

    Examples
    --------
    >>> results = evaluate_model(Y_test, Y_pred)
    >>> print(results)
    {'rmspe': 0.3464, 'rmse': 1057.70, 'r2': 0.8679}
    """
    return {
        "rmspe": rmspe(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "r2":    r2(y_true, y_pred),
    }


def print_results(model_name: str,
                  train_results: dict,
                  test_results: dict) -> None:
    """
    Print formatted model evaluation results.

    Parameters
    ----------
    model_name : str
        Name of the model for display.
    train_results : dict
        Results from evaluate_model() on training set.
    test_results : dict
        Results from evaluate_model() on test set.

    Examples
    --------
    >>> print_results("LightGBM", train_results, test_results)
    """
    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  {'Metric':<12} {'Train':>10} {'Test':>10}")
    print(f"  {'-'*34}")
    print(f"  {'RMSPE':<12} {train_results['rmspe']:>10.4f} "
          f"{test_results['rmspe']:>10.4f}")
    print(f"  {'RMSE (€)':<12} {train_results['rmse']:>10.2f} "
          f"{test_results['rmse']:>10.2f}")
    print(f"  {'R²':<12} {train_results['r2']:>10.4f} "
          f"{test_results['r2']:>10.4f}")
    print(f"{'='*50}\n")