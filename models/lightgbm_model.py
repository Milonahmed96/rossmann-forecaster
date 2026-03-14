"""
models/lightgbm_model.py
------------------------
LightGBM forecasting model — baseline and hyperparameter-tuned versions.

Best performing model in the MSc project:
    R² = 0.8679  |  RMSPE = 0.3464  |  RMSE = £1,057.70/store

Uses Label Encoded features (X_LE).
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error


class LightGBMModel:
    """
    LightGBM forecasting model.

    Parameters
    ----------
    n_estimators : int
        Number of boosting trees. Default 100 for baseline.
    learning_rate : float
        Step size shrinkage. Default 0.1.
    num_leaves : int
        Maximum number of leaves per tree. Default 31.

    Examples
    --------
    >>> model = LightGBMModel()
    >>> model.fit(X_train_LE, Y_train)
    >>> predictions = model.predict(X_test_LE)

    >>> tuned = LightGBMModel()
    >>> tuned.tune(X_train_LE, Y_train)
    >>> predictions = tuned.predict(X_test_LE)
    """

    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 num_leaves: int = 31):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.model = self._build_model(n_estimators, learning_rate, num_leaves)
        self.best_params = None
        self.is_fitted = False

    def _build_model(self, n_estimators, learning_rate, num_leaves):
        """Construct a LGBMRegressor with standard settings."""
        return lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X_train, Y_train) -> "LightGBMModel":
        """
        Fit the LightGBM model on training data.

        Parameters
        ----------
        X_train : array-like
            Label encoded training features.
        Y_train : array-like
            Log-transformed target values.

        Returns
        -------
        self
        """
        print(f"Fitting LightGBM model "
              f"(n_estimators={self.n_estimators}, "
              f"lr={self.learning_rate}, "
              f"num_leaves={self.num_leaves}) ...")
        self.model.fit(X_train, Y_train)
        self.is_fitted = True
        print("  LightGBM model fitted.")
        return self

    def predict(self, X) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like
            Label encoded features.

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
             n_splits: int = 3) -> "LightGBMModel":
        """
        Tune hyperparameters using RandomizedSearchCV with TimeSeriesSplit.

        Search space matches the MSc notebook exactly:
            n_estimators: [500, 1000, 1500]
            learning_rate: [0.01, 0.05, 0.1]
            num_leaves: [31, 70, 100]
            max_depth: [-1, 15, 20]
            min_child_samples: [20, 50, 100]
            colsample_bytree: [0.7, 0.9]
            subsample: [0.7, 0.9]

        Parameters
        ----------
        X_train : array-like
            Label encoded training features.
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
        print(f"Tuning LightGBM model ({n_iter} iterations, {n_splits} folds) ...")

        param_dist = {
            "n_estimators":     [500, 1000, 1500],
            "learning_rate":    [0.01, 0.05, 0.1],
            "num_leaves":       [31, 70, 100],
            "max_depth":        [-1, 15, 20],
            "min_child_samples":[20, 50, 100],
            "colsample_bytree": [0.7, 0.9],
            "subsample":        [0.7, 0.9],
        }

        scorer = make_scorer(
            lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=True
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)

        base = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scorer,
            cv=tscv,
            verbose=0,
            random_state=42,
            n_jobs=1,
        )

        search.fit(X_train, Y_train)

        self.best_params = search.best_params_
        self.model = search.best_estimator_
        self.is_fitted = True

        print(f"  Best params: {self.best_params}")
        return self