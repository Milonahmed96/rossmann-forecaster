import numpy as np


class ConformalPredictor:
    """
    Model-agnostic split conformal predictor.

    Wraps any model with a .predict() method and produces
    calibrated prediction intervals with a coverage guarantee.

    Usage:
        cp = ConformalPredictor(model, coverage=0.90)
        cp.calibrate(X_cal, Y_cal_log)
        intervals = cp.predict_interval(X_test)
        results   = cp.evaluate(X_test, Y_test_log)
    """

    def __init__(self, model, coverage=0.90):
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be between 0 and 1, got {coverage}")
        self.model    = model
        self.coverage = coverage
        self.q_       = None   # set by calibrate()

    def calibrate(self, X_cal, Y_cal_log, **predict_kwargs):
        """
        Compute the nonconformity quantile from the calibration set.

        Args:
            X_cal       : calibration features
            Y_cal_log   : true log-scale sales for calibration rows
            **predict_kwargs : passed to model.predict() — use for
                               store_ids=..., window_size=... with PyTorch LSTM
        """
        # Get point predictions in Euro scale
        y_hat = self.model.predict(X_cal, **predict_kwargs)   # Euros

        # Align true values to the same rows as predictions
        # (PyTorch LSTM predicts fewer rows than X_cal has)
        n_pred = len(y_hat)
        y_true = np.expm1(Y_cal_log)   # log → Euros

        # If model predicts fewer rows, take the last n_pred true values
        # This matches the SequenceDataset target alignment from Project 2
        if len(y_true) != n_pred:
            y_true = y_true[-n_pred:]

        # Nonconformity scores: absolute prediction error in Euro scale
        scores = np.abs(y_true - y_hat)

        # Finite-sample corrected quantile
        n     = len(scores)
        alpha = 1.0 - self.coverage
        level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        self.q_ = float(np.quantile(scores, level))

    def predict_interval(self, X, **predict_kwargs):
        """
        Produce prediction intervals for new data.

        Returns:
            dict with keys:
                'point' : point predictions in Euros
                'lower' : lower bounds in Euros (clipped to 0)
                'upper' : upper bounds in Euros
        """
        if self.q_ is None:
            raise RuntimeError("Call calibrate() before predict_interval().")

        y_hat = self.model.predict(X, **predict_kwargs)
        lower = np.maximum(y_hat - self.q_, 0.0)
        upper = y_hat + self.q_

        return {'point': y_hat, 'lower': lower, 'upper': upper}

    def evaluate(self, X, Y_true_log, **predict_kwargs):
        """
        Evaluate coverage and interval width on a labelled set.

        Returns:
            dict with keys:
                'coverage'     : empirical coverage (fraction of true values inside interval)
                'mean_width'   : mean interval width in Euros
                'median_width' : median interval width in Euros
                'q'            : the nonconformity quantile in Euros
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

