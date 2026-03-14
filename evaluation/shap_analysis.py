"""
evaluation/shap_analysis.py
---------------------------
SHAP feature importance analysis for the LightGBM model.

SHAP (SHapley Additive exPlanations) reveals which features drive
predictions. In the MSc project, SHAP confirmed that promotional
features are the primary sales drivers — validating the research focus.

Top features from MSc project:
    1. Sales(Rolling_Mean_7)  — recent sales history
    2. Promo                  — short-term promotion flag
    3. DayOfWeek              — weekly seasonality
    4. Sales(Lag_7)           — 7-day lagged sales
    5. CompetitionDistance    — proximity to competition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_shap_values(model, X_sample: pd.DataFrame,
                        sample_size: int = 5000,
                        random_state: int = 42):
    """
    Compute SHAP values for a fitted LightGBM model.

    Parameters
    ----------
    model : LightGBMModel
        A fitted LightGBMModel instance.
    X_sample : pd.DataFrame
        Feature DataFrame to compute SHAP values on.
    sample_size : int
        Number of rows to sample for efficiency.
        Default 5000 matches MSc notebook.
    random_state : int
        Random seed for reproducible sampling.

    Returns
    -------
    tuple: (shap_values, X_sampled)
        shap_values : np.ndarray of SHAP values
        X_sampled   : pd.DataFrame of sampled features

    Raises
    ------
    ImportError
        If shap package is not installed.
    RuntimeError
        If model is not fitted.
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap package is required. "
            "Install with: pip install shap"
        )

    if not model.is_fitted:
        raise RuntimeError(
            "Model must be fitted before computing SHAP values. "
            "Call model.fit() or model.tune() first."
        )

    # Sample for efficiency — full dataset is too slow for SHAP
    if len(X_sample) > sample_size:
        X_sampled = X_sample.sample(n=sample_size, random_state=random_state)
    else:
        X_sampled = X_sample.copy()

    print(f"Computing SHAP values for {len(X_sampled):,} samples ...")

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_sampled)

    print("  SHAP values computed.")
    return shap_values, X_sampled


def get_feature_importance(shap_values: np.ndarray,
                           X_sampled: pd.DataFrame) -> pd.Series:
    """
    Calculate mean absolute SHAP values per feature.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values from compute_shap_values().
    X_sampled : pd.DataFrame
        Sampled features used to compute SHAP values.

    Returns
    -------
    pd.Series
        Feature importances sorted descending.
        Index = feature names, values = mean |SHAP|.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(
        mean_abs_shap,
        index=X_sampled.columns
    ).sort_values(ascending=False)
    return importance


def plot_shap_summary(shap_values: np.ndarray,
                      X_sampled: pd.DataFrame,
                      save_path: str = None) -> None:
    """
    Plot SHAP summary beeswarm plot.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values from compute_shap_values().
    X_sampled : pd.DataFrame
        Sampled features used to compute SHAP values.
    save_path : str, optional
        If provided, saves the plot to this path.
        Example: 'outputs/shap_summary.png'
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap package is required.")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sampled, cmap="coolwarm", show=False)
    plt.title("SHAP Feature Importance — Rossmann LightGBM", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  SHAP plot saved to: {save_path}")

    plt.show()


def print_top_features(importance: pd.Series, n: int = 10) -> None:
    """
    Print top N features by mean absolute SHAP value.

    Parameters
    ----------
    importance : pd.Series
        Feature importances from get_feature_importance().
    n : int
        Number of top features to display. Default 10.
    """
    print(f"\nTop {n} Features by Mean |SHAP| Value:")
    print(f"{'='*40}")
    for i, (feature, value) in enumerate(importance.head(n).items(), 1):
        bar = "█" * int(value * 50 / importance.iloc[0])
        print(f"  {i:>2}. {feature:<30} {value:.4f}  {bar}")
    print(f"{'='*40}\n") 
