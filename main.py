"""
main.py
-------
Entry point for the Rossmann Store Sales Forecaster.

Usage:
    python main.py                         # full pipeline, all models
    python main.py --model lightgbm        # one model only
    python main.py --model pytorch_lstm    # PyTorch LSTM only
    python main.py --skip-lstm --no-shap   # fast dev mode
"""

import argparse
import time
import numpy as np

from data.loader import load_data
from data.preprocessor import clean_data
from data.feature_engineer import engineer_features
from data.splitter import prepare_data
from models.ridge_model import RidgeModel
from models.lightgbm_model import LightGBMModel
from models.lstm_model import LSTMModel
from models.pytorch_lstm import PyTorchLSTM
from evaluation.metrics import evaluate_model, print_results
from evaluation.shap_analysis import (
    compute_shap_values,
    get_feature_importance,
    print_top_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Rossmann Store Sales Forecaster")
    parser.add_argument(
        "--model",
        choices=["ridge", "lightgbm", "lstm", "pytorch_lstm", "all"],
        default="all",
        help="Which model to run (default: all)"
    )
    parser.add_argument("--skip-lstm",   action="store_true",
                        help="Skip Keras LSTM")
    parser.add_argument("--skip-ridge",  action="store_true",
                        help="Skip Ridge Regression")
    parser.add_argument("--skip-lgbm",   action="store_true",
                        help="Skip LightGBM")
    parser.add_argument("--no-tune",     action="store_true",
                        help="Skip hyperparameter tuning")
    parser.add_argument("--no-shap",     action="store_true",
                        help="Skip SHAP analysis")
    return parser.parse_args()


def run_pipeline(args):
    start_time = time.time()

    print("\n" + "="*60)
    print("  ROSSMANN STORE SALES FORECASTER")
    print("  MSc Data Science Project — Production Package")
    print("="*60 + "\n")

    # ── Stage 1: Data Pipeline ────────────────────────────────────
    print("STAGE 1: Data Pipeline")
    print("-" * 40)

    raw      = load_data()
    clean    = clean_data(raw)
    features = engineer_features(clean)
    splits   = prepare_data(features)

    X_train_LE     = splits["X_train_LE"]
    X_test_LE      = splits["X_test_LE"]
    X_train_OHE    = splits["X_train_OHE"]
    X_test_OHE     = splits["X_test_OHE"]
    Y_train        = splits["Y_train"]
    Y_test         = splits["Y_test"]
    X_train_scaled = splits["X_train_scaled"]
    X_test_scaled  = splits["X_test_scaled"]
    store_ids_train = splits["store_ids_train"]
    store_ids_test  = splits["store_ids_test"]

    print(f"\n  Training set: {len(X_train_LE):,} records")
    print(f"  Test set:     {len(X_test_LE):,} records")
    print(f"  Features:     {X_train_LE.shape[1]}")

    # ── Stage 2: Model Training ───────────────────────────────────
    print("\nSTAGE 2: Model Training and Evaluation")
    print("-" * 40)

    all_results = {}
    lgbm = None

    # Ridge
    run_ridge = (args.model in ["ridge", "all"]) and not args.skip_ridge
    if run_ridge:
        print("\n[Ridge Regression]")
        ridge = RidgeModel()
        if args.no_tune:
            ridge.fit(X_train_OHE, Y_train)
        else:
            ridge.tune(X_train_OHE, Y_train)
        train_r = evaluate_model(Y_train, ridge.predict(X_train_OHE))
        test_r  = evaluate_model(Y_test,  ridge.predict(X_test_OHE))
        print_results("Ridge Regression", train_r, test_r)
        all_results["Ridge"] = test_r

    # LightGBM
    run_lgbm = (args.model in ["lightgbm", "all"]) and not args.skip_lgbm
    if run_lgbm:
        print("\n[LightGBM]")
        lgbm = LightGBMModel()
        if args.no_tune:
            lgbm.fit(X_train_LE, Y_train)
        else:
            lgbm.tune(X_train_LE, Y_train)
        train_r = evaluate_model(Y_train, lgbm.predict(X_train_LE))
        test_r  = evaluate_model(Y_test,  lgbm.predict(X_test_LE))
        print_results("LightGBM", train_r, test_r)
        all_results["LightGBM"] = test_r

    # Keras LSTM
    run_keras = (args.model in ["lstm", "all"]) and not args.skip_lstm
    if run_keras:
        print("\n[LSTM (Keras)]")
        lstm = LSTMModel()
        if args.no_tune:
            lstm.fit(X_train_scaled, Y_train)
        else:
            lstm.fit_optimised(X_train_scaled, Y_train)
        train_r = evaluate_model(Y_train, lstm.predict(X_train_scaled))
        test_r  = evaluate_model(Y_test,  lstm.predict(X_test_scaled))
        print_results("LSTM (Keras)", train_r, test_r)
        all_results["LSTM (Keras)"] = test_r

    # PyTorch LSTM
    run_pytorch = args.model in ["pytorch_lstm", "all"]
    if run_pytorch:
        print("\n[PyTorch LSTM — window=7, hidden=128]")
        pt = PyTorchLSTM(input_size=X_train_scaled.shape[1])
        pt.fit(
            X_train_scaled,
            Y_train.values,
            store_ids_train,
            window_size=7,
            epochs=50,
            batch_size=512,
            patience=5,
            save_path='models/best_pytorch_lstm.pt'
        )
        test_r = pt.evaluate(
            X_test_scaled,
            Y_test.values,
            store_ids_test,
            window_size=7
        )
        # Train evaluation on a subset for speed (5% of training windows)
        mask = np.isin(store_ids_train, np.unique(store_ids_train)[:56])
        train_r = pt.evaluate(
            X_train_scaled[mask],
            Y_train.values[mask],
            store_ids_train[mask],
            window_size=7
        )
        print_results("PyTorch LSTM", train_r, test_r)
        all_results["PyTorch LSTM"] = test_r

    # ── Stage 3: Comparison ───────────────────────────────────────
    if len(all_results) > 1:
        print("\nSTAGE 3: Final Model Comparison")
        print("-" * 40)
        print(f"\n  {'Model':<18} {'RMSPE':>8} {'RMSE (€)':>10} {'R²':>8}")
        print(f"  {'-'*46}")
        for name, r in sorted(all_results.items(), key=lambda x: x[1]["rmspe"]):
            print(f"  {name:<18} {r['rmspe']:>8.4f} {r['rmse']:>10.2f} {r['r2']:>8.4f}")
        print()

    # ── Stage 4: SHAP ─────────────────────────────────────────────
    if not args.no_shap and lgbm is not None:
        print("\nSTAGE 4: SHAP Feature Importance Analysis")
        print("-" * 40)
        shap_values, X_sampled = compute_shap_values(lgbm, X_test_LE)
        importance = get_feature_importance(shap_values, X_sampled)
        print_top_features(importance)

    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.1f} seconds.")
    print("="*60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
