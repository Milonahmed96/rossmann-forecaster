import mlflow
import mlflow.pytorch
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
MLFLOW_TRACKING_URI = "file:///" + str(ROOT / "mlruns").replace("\\", "/")


def setup_mlflow():
    """Initialise MLflow with local tracking URI."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("rossmann-forecasting")
    print(f"[mlflow] Tracking URI: {MLFLOW_TRACKING_URI}")


def log_model_version(
    model_name: str,
    rmspe: float,
    r2: float,
    model_path: str,
    params: dict = None,
):
    """Log a model version with its metrics to MLflow."""
    setup_mlflow()

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_path", model_path)
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metric("rmspe", rmspe)
        mlflow.log_metric("r2", r2)

        # Log model file as artifact
        mlflow.log_artifact(model_path)

        print(f"[mlflow] Logged {model_name} — RMSPE: {rmspe}, R²: {r2}")


def log_prediction(store: int, prediction: float, lower: float, upper: float):
    """Log a single prediction for monitoring."""
    setup_mlflow()
    mlflow.set_experiment("rossmann-predictions")

    with mlflow.start_run(run_name=f"prediction-store-{store}"):
        mlflow.log_param("store", store)
        mlflow.log_metric("predicted_sales", prediction)
        mlflow.log_metric("lower_bound", lower)
        mlflow.log_metric("upper_bound", upper)
        mlflow.log_metric("interval_width", upper - lower)
