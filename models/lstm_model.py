"""
models/lstm_model.py
--------------------
LSTM forecasting model — baseline and optimised versions.

Requires TensorFlow 2.x. Not compatible with Python 3.13 at time of writing.
Run on Google Colab (Python 3.10) or in a Python 3.10 virtual environment.

MSc project results:
    Baseline LSTM:  RMSPE = 0.3480  |  R² = 0.8232
    Optimised LSTM: RMSPE = 0.3592  |  R² = 0.8232
    (Stacked LSTM degraded — tabular data favours tree-based models)

Uses MinMax scaled Label Encoded features (X_train_scaled, X_test_scaled).
"""

import numpy as np

# TensorFlow is imported inside methods to allow the module to be imported
# even when TensorFlow is not installed. Tests can then skip gracefully.


class LSTMModel:
    """
    LSTM forecasting model with baseline and optimised architectures.

    Parameters
    ----------
    timesteps : int
        Sequence length for LSTM input. Default 1 matches MSc notebook.
        Note: timesteps=1 makes the LSTM effectively a dense network.
        A proper sequential window (e.g. 7 or 14) will be implemented
        in Project 2 (PyTorch rebuild).
    units : int
        Number of LSTM units in the first layer. Default 100.
    dropout : float
        Dropout rate after LSTM layers. Default 0.2.
    epochs : int
        Maximum training epochs. Default 10 for baseline.
    batch_size : int
        Training batch size. Default 256.

    Examples
    --------
    >>> model = LSTMModel()
    >>> model.fit(X_train_scaled, Y_train, n_features=X_train_scaled.shape[1])
    >>> predictions = model.predict(X_test_scaled)

    >>> optimised = LSTMModel(units=128, epochs=50)
    >>> optimised.fit_optimised(X_train_scaled, Y_train,
    ...                         n_features=X_train_scaled.shape[1])
    >>> predictions = optimised.predict(X_test_scaled)
    """

    def __init__(self,
                 timesteps: int = 1,
                 units: int = 100,
                 dropout: float = 0.2,
                 epochs: int = 10,
                 batch_size: int = 256):
        self.timesteps = timesteps
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.n_features = None
        self.is_fitted = False
        self.history = None

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape flat 2D array into 3D LSTM input.

        LSTM expects: (samples, timesteps, features)
        We use timesteps=1 to match the MSc notebook.
        """
        return X.reshape((X.shape[0], self.timesteps, X.shape[1]))

    def _build_baseline(self, n_features: int):
        """
        Build baseline LSTM architecture from MSc notebook.

        Architecture:
            LSTM(100, activation=relu) → Dropout(0.2) → Dense(1)
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            import tensorflow as tf
            tf.random.set_seed(42)
            np.random.seed(42)
        except ImportError:
            raise ImportError(
                "TensorFlow is required for LSTMModel. "
                "Install it with: pip install tensorflow\n"
                "Note: TensorFlow does not support Python 3.13. "
                "Use Python 3.10 or Google Colab."
            )

        from tensorflow.keras.layers import Input
        model = Sequential([
            Input(shape=(self.timesteps, n_features)),
            LSTM(self.units,
                 activation="relu",
                 return_sequences=False),
            Dropout(self.dropout),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def _build_optimised(self, n_features: int):
        """
        Build optimised stacked LSTM architecture from MSc notebook.

        Architecture:
            LSTM(128, return_sequences=True) → Dropout(0.2)
            → LSTM(64) → Dropout(0.2) → Dense(1)

        Note: This architecture degraded in the MSc project (RMSPE 0.3480
        → 0.3592). Stacked LSTMs overfit on this tabular dataset because
        the lag features already capture the temporal patterns explicitly.
        Included here for completeness and comparison.
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            import tensorflow as tf
            tf.random.set_seed(42)
            np.random.seed(42)
        except ImportError:
            raise ImportError(
                "TensorFlow is required for LSTMModel. "
                "Install it with: pip install tensorflow\n"
                "Note: TensorFlow does not support Python 3.13. "
                "Use Python 3.10 or Google Colab."
            )

        from tensorflow.keras.layers import Input
        model = Sequential([
            Input(shape=(self.timesteps, n_features)),
            LSTM(128,
                 activation="relu",
                 return_sequences=True),
            Dropout(self.dropout),
            LSTM(64, activation="relu", return_sequences=False),
            Dropout(self.dropout),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X_train: np.ndarray, Y_train,
            n_features: int = None) -> "LSTMModel":
        """
        Fit the baseline LSTM model.

        Parameters
        ----------
        X_train : np.ndarray
            MinMax scaled training features (2D: samples x features).
        Y_train : array-like
            Log-transformed target values.
        n_features : int, optional
            Number of input features. Inferred from X_train if not provided.

        Returns
        -------
        self
        """
        if n_features is None:
            n_features = X_train.shape[1]
        self.n_features = n_features

        print(f"Fitting baseline LSTM "
              f"(units={self.units}, epochs={self.epochs}, "
              f"batch_size={self.batch_size}) ...")

        self.model = self._build_baseline(n_features)
        X_reshaped = self._reshape(X_train)

        self.history = self.model.fit(
            X_reshaped, Y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=1,
        )

        self.is_fitted = True
        print("  Baseline LSTM fitted.")
        return self

    def fit_optimised(self, X_train: np.ndarray, Y_train,
                      n_features: int = None,
                      patience: int = 5) -> "LSTMModel":
        """
        Fit the optimised LSTM with early stopping and model checkpointing.

        Parameters
        ----------
        X_train : np.ndarray
            MinMax scaled training features.
        Y_train : array-like
            Log-transformed target values.
        n_features : int, optional
            Number of input features. Inferred from X_train if not provided.
        patience : int
            EarlyStopping patience — stops if val_loss does not improve
            for this many epochs.

        Returns
        -------
        self
        """
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        except ImportError:
            raise ImportError(
                "TensorFlow is required for LSTMModel. "
                "Install it with: pip install tensorflow"
            )

        if n_features is None:
            n_features = X_train.shape[1]
        self.n_features = n_features

        print(f"Fitting optimised LSTM "
              f"(units=128/64, max_epochs=50, patience={patience}) ...")

        self.model = self._build_optimised(n_features)
        X_reshaped = self._reshape(X_train)

        callbacks = [
            EarlyStopping(
                patience=patience,
                monitor="val_loss",
                mode="min",
                verbose=1,
                restore_best_weights=True,
            ),
        ]

        self.history = self.model.fit(
            X_reshaped, Y_train,
            epochs=50,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=1,
            callbacks=callbacks,
        )

        self.is_fitted = True
        print("  Optimised LSTM fitted.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : np.ndarray
            MinMax scaled features (2D: samples x features).

        Returns
        -------
        np.ndarray
            Log-scale predictions (flattened 1D array).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted yet. Call fit() or fit_optimised() first."
            )
        X_reshaped = self._reshape(X)
        return self.model.predict(X_reshaped).flatten()