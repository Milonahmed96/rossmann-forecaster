"""
models/ridge_model.py
---------------------
Ridge Regression model — baseline and hyperparameter-tuned versions.

Wraps sklearn Ridge with a clean interface matching the other models
in this package. Uses One-Hot Encoded features (X_OHE).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error


class RidgeModel:
    """
    Ridge Regression forecasting model.

    Parameters
    ----------
    alpha : float
        Regularisation strength. Higher = more regularisation.
        Default 0.1 matches the MSc notebook baseline.

    Examples
    --------
    >>> model = RidgeModel()
    >>> model.fit(X_train_OHE, Y_train)
    >>> predictions = model.predict(X_test_OHE)

    >>> tuned = RidgeModel()
    >>> tuned.tune(X_train_OHE, Y_train)
    >>> predictions = tuned.predict(X_test_OHE)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=42)
        self.best_alpha = None
        self.is_fitted = False

    def fit(self, X_train, Y_train) -> "RidgeModel":
        """
        Fit the Ridge model on training data.

        Parameters
        ----------
        X_train : array-like
            One-hot encoded training features.
        Y_train : array-like
            Log-transformed target values.

        Returns
        -------
        self
        """
        print(f"Fitting Ridge model (alpha={self.alpha}) ...")
        self.model.fit(X_train, Y_train)
        self.is_fitted = True
        print("  Ridge model fitted.")
        return self

    def predict(self, X) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like
            One-hot encoded features.

        Returns
        -------
        np.ndarray
            Log-scale predictions.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted yet. Call fit() or tune() first."
            )
        return self.model.predict(X)

    def tune(self, X_train, Y_train,
             n_iter: int = 10,
             n_splits: int = 3) -> "RidgeModel":
        """
        Tune alpha using RandomizedSearchCV with TimeSeriesSplit.

        Parameters
        ----------
        X_train : array-like
            One-hot encoded training features.
        Y_train : array-like
            Log-transformed target values.
        n_iter : int
            Number of random parameter combinations to try.
        n_splits : int
            Number of TimeSeriesSplit folds.

        Returns
        -------
        self
        """
        print(f"Tuning Ridge model ({n_iter} iterations, {n_splits} folds) ...")

        param_dist = {"alpha": np.logspace(-4, 2, 100)}

        scorer = make_scorer(
            lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=True
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)

        search = RandomizedSearchCV(
            estimator=Ridge(random_state=42),
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scorer,
            cv=tscv,
            verbose=0,
            random_state=42,
            n_jobs=1,
        )

        search.fit(X_train, Y_train)

        self.best_alpha = search.best_params_["alpha"]
        self.alpha = self.best_alpha
        self.model = search.best_estimator_
        self.is_fitted = True

        print(f"  Best alpha: {self.best_alpha:.6f}")
        return self