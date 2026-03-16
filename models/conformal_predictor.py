import numpy as np


class ConformalPredictor:
    """
    Model-agnostic split conformal predictor.

    Wraps any model with a .predict() method and produces
    calibrated prediction intervals with a coverage guarantee.

    Args:
        model       : any object with a .predict() method
        coverage    : desired coverage level, e.g. 0.90 = 90%
        predict_log : True if model.predict() returns log-scale values
                      (e.g. LightGBM). False if it returns Euro-scale
                      values (e.g. PyTorchLSTM which applies expm1 internally).
    """

    def __init__(self, model, coverage=0.90, predict_log=False):
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be between 0 and 1, got {coverage}")
        self.model       = model
        self.coverage    = coverage
        self.predict_log = predict_log
        self.q_          = None

    def _predict_euros(self, X, **predict_kwargs):
        """Get predictions in Euro scale regardless of model output scale."""
        y_hat = self.model.predict(X, **predict_kwargs)
        if self.predict_log:
            return np.expm1(y_hat)
        return y_hat

    def calibrate(self, X_cal, Y_cal_log, **predict_kwargs):
        """
        Compute the nonconformity quantile from the calibration set.

        Args:
            X_cal        : calibration features
            Y_cal_log    : true log-scale sales for calibration rows
            **predict_kwargs : passed to model.predict()
        """
        y_hat  = self._predict_euros(X_cal, **predict_kwargs)
        n_pred = len(y_hat)
        y_true = np.expm1(Y_cal_log)

        if len(y_true) != n_pred:
            y_true = y_true[-n_pred:]

        scores = np.abs(y_true - y_hat)

        n     = len(scores)
        alpha = 1.0 - self.coverage
        level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        self.q_ = float(np.quantile(scores, level))

    def predict_interval(self, X, **predict_kwargs):
        """
        Produce prediction intervals for new data.

        Returns dict with keys: point, lower, upper (all Euro scale).
        """
        if self.q_ is None:
            raise RuntimeError("Call calibrate() before predict_interval().")

        y_hat = self._predict_euros(X, **predict_kwargs)
        lower = np.maximum(y_hat - self.q_, 0.0)
        upper = y_hat + self.q_

        return {'point': y_hat, 'lower': lower, 'upper': upper}

    def evaluate(self, X, Y_true_log, **predict_kwargs):
        """
        Evaluate coverage and interval width on a labelled set.

        Returns dict with keys: coverage, mean_width, median_width, q.
        """
        if self.q_ is None:
            raise RuntimeError("Call calibrate() before evaluate().")

        intervals = self.predict_interval(X, **predict_kwargs)
        n_pred    = len(intervals['point'])

        y_true = np.expm1(Y_true_log)
        if len(y_true) != n_pred:
            y_true = y_true[-n_pred:]

        covered = (y_true >= intervals['lower']) & (y_true <= intervals['upper'])
        widths  = intervals['upper'] - intervals['lower']

        return {
            'coverage'     : float(covered.mean()),
            'mean_width'   : float(widths.mean()),
            'median_width' : float(np.median(widths)),
            'q'            : self.q_,
        }