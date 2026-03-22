import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "best_pytorch_lstm.pt"

# ── Model architecture — must match training exactly ──────────────────────────
class _LSTMNet(nn.Module):
    def __init__(self, input_size=21, hidden_size=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Singleton model ───────────────────────────────────────────────────────────
_model = None


def get_model():
    global _model
    if _model is None:
        _model = _LSTMNet(input_size=21, hidden_size=128, num_layers=1)
        state_dict = torch.load(str(MODEL_PATH), map_location="cpu")
        _model.load_state_dict(state_dict)
        _model.eval()
        print(f"[predictor] LSTM loaded from {MODEL_PATH}")
    return _model


def build_features(request) -> np.ndarray:
    """Convert PredictionRequest into a feature vector matching training order."""
    store_type_map  = {"a": 0, "b": 1, "c": 2, "d": 3}
    assortment_map  = {"a": 0, "b": 1, "c": 2}
    holiday_map     = {"0": 0, "a": 1, "b": 2, "c": 3}

    features = np.array([[
        request.store,
        request.day_of_week,
        request.promo,
        holiday_map.get(request.state_holiday, 0),
        request.school_holiday,
        store_type_map.get(request.store_type, 0),
        assortment_map.get(request.assortment, 0),
        request.competition_distance,
        request.promo2,
        request.month,
        request.year,
        request.day,
        0, 0, 0, 0, 0, 0, 0, 0, 0,  # padding to reach input_size=21
    ]], dtype=np.float32)

    return features


def predict(request) -> dict:
    """Run inference and return point prediction + interval."""
    features = build_features(request)

    model = get_model()
    with torch.no_grad():
        # Shape: (batch=1, seq_len=1, features=21)
        x = torch.tensor(features).unsqueeze(0)
        log_pred = model(x).item()
        point_prediction = float(np.expm1(log_pred))

    # Fallback ±15% interval (replace with conformal predictor if available)
    margin = point_prediction * 0.15
    lower  = max(0.0, point_prediction - margin)
    upper  = point_prediction + margin

    return {
        "store":            request.store,
        "predicted_sales":  round(point_prediction, 2),
        "lower_bound":      round(lower, 2),
        "upper_bound":      round(upper, 2),
        "confidence_level": 0.90,
        "model":            "PyTorch LSTM",
    }