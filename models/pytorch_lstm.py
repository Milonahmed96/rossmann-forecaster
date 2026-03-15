import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.sequence_dataset import SequenceDataset


class _LSTMNet(nn.Module):
    """The actual neural network — kept private, used only by PyTorchLSTM."""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True        # input shape: (batch, seq_len, features)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)           # out: (batch, window_size, hidden_size)
        last = out[:, -1, :]            # take only the last time step
        dropped = self.dropout(last)
        return self.fc(dropped).squeeze(-1)  # shape: (batch,)


class PyTorchLSTM:
    """
    PyTorch LSTM with the same interface as RidgeModel and LightGBMModel.
    Methods: fit(), predict(), evaluate()
    """

    def __init__(self, input_size=21, hidden_size=128, num_layers=1, dropout=0.2):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model       = None
        print(f"PyTorchLSTM — device: {self.device}")

    def fit(self, X_scaled, Y_train, store_ids_train,
            window_size=7, epochs=50, batch_size=512,
            learning_rate=0.001, patience=5,
            save_path='models/best_pytorch_lstm.pt'):
        """
        Train the LSTM on sequence windows.

        Args:
            X_scaled        : numpy array (n_samples, n_features) — MinMax scaled
            Y_train         : numpy array (n_samples,)            — log-transformed sales
            store_ids_train : numpy array (n_samples,)            — store number per row
            window_size     : int — days per sequence (7 recommended)
            epochs          : int — maximum training epochs
            batch_size      : int — samples per gradient step
            learning_rate   : float — Adam learning rate
            patience        : int — early stopping patience
            save_path       : str — where to save the best model weights
        """
        # ── Split off validation rows (at least window_size + 1 rows) ──
        n = len(X_scaled)
        min_val_rows = window_size + 1
        val_rows = max(min_val_rows, int(n * 0.10))
        val_start = n - val_rows

        X_tr, Y_tr, ids_tr = X_scaled[:val_start], Y_train[:val_start], store_ids_train[:val_start]
        X_val, Y_val, ids_val = X_scaled[val_start:], Y_train[val_start:], store_ids_train[val_start:]

        # ── Build datasets and loaders ──
        train_ds = SequenceDataset(X_tr, Y_tr, ids_tr, window_size)
        val_ds   = SequenceDataset(X_val, Y_val, ids_val, window_size)

        print(f"Training windows : {len(train_ds):,}")
        print(f"Validation windows: {len(val_ds):,}")

        # num_workers=0 required on Windows
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=batch_size,
                                  shuffle=False, num_workers=0)

        # ── Build model ──
        self.model = _LSTMNet(self.input_size, self.hidden_size,
                              self.num_layers, self.dropout).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        best_val_loss    = float('inf')
        patience_counter = 0
        history          = []

        print(f"\nTraining for up to {epochs} epochs (patience={patience})...")
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Status':>10}")
        print("-" * 46)

        for epoch in range(1, epochs + 1):
            # ── Training pass ──
            self.model.train()
            train_loss = 0.0
            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss  = criterion(preds, Y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(Y_batch)

            train_loss /= len(train_ds)

            # ── Validation pass ──
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)
                    preds    = self.model(X_batch)
                    loss     = criterion(preds, Y_batch)
                    val_loss += loss.item() * len(Y_batch)
            val_loss /= len(val_ds)

            history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

            # ── Early stopping ──
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                status = "saved"
            else:
                patience_counter += 1
                status = f"patience {patience_counter}/{patience}"

            print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>12.6f} {status:>10}")

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        # ── Restore best weights ──
        self.model.load_state_dict(
            torch.load(save_path, map_location=self.device)
        )
        print(f"\nBest val loss: {best_val_loss:.6f}")
        return history

    def predict(self, X_scaled, store_ids, window_size=7, batch_size=512):
        """
        Generate predictions on a scaled feature matrix.

        Returns a numpy array of predictions in EURO scale (log transform reversed).
        Note: returns (n - window_size) predictions per store, not n.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        ds     = SequenceDataset(X_scaled, np.zeros(len(X_scaled)), store_ids, window_size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                preds   = self.model(X_batch)
                all_preds.append(preds.cpu().numpy())

        log_preds = np.concatenate(all_preds)
        return np.expm1(log_preds)      # reverse log transform → Euros

    def evaluate(self, X_scaled, Y_true, store_ids, window_size=7):
        """
        Evaluate on a test set. Returns dict with rmspe, rmse, r2.
        Aligns predictions with the correct target rows automatically.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Build dataset to get the index (tells us which rows are targets)
        ds = SequenceDataset(X_scaled, Y_true, store_ids, window_size)

        # Get target row indices
        target_indices = np.array([t for _, t in ds.index])

        # Get actual sales values at those target rows (reverse log transform)
        Y_actual = np.expm1(Y_true[target_indices])

        # Get predictions
        Y_pred = self.predict(X_scaled, store_ids, window_size)

        from evaluation.metrics import evaluate_model
        return evaluate_model(Y_actual, Y_pred)
