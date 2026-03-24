from monitoring.mlflow_logger import log_model_version

# Log all three models with their results from P1, P2, P3
log_model_version(
    model_name="Ridge",
    rmspe=0.4821,
    r2=0.6234,
    model_path="models/ridge_model.py",
    params={"alpha": 1.0}
)

log_model_version(
    model_name="LightGBM",
    rmspe=0.3464,
    r2=0.8679,
    model_path="models/lightgbm_model.py",
    params={"n_estimators": 1000, "learning_rate": 0.05}
)

log_model_version(
    model_name="PyTorch-LSTM",
    rmspe=0.2871,
    r2=0.9131,
    model_path="models/best_pytorch_lstm.pt",
    params={"hidden_size": 128, "num_layers": 1, "dropout": 0.2, "input_size": 21}
)

print("\nAll models logged. Run: mlflow ui")