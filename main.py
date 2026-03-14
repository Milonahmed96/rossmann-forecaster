"""
main.py
-------
Entry point for the Rossmann Store Sales Forecaster.

Reproduces the full MSc project pipeline as a clean Python package.

Usage:
    python main.py                    # run full pipeline
    python main.py --model lightgbm   # run one model only
    python main.py --skip-lstm        # skip LSTM (faster)

Expected results (from MSc project):
    Ridge    — RMSPE: 0.4056  RMSE: £1,926  R²: 0.5910
    LightGBM — RMSPE: 0.3464  RMSE: £1,058  R²: 0.8679
    LSTM     — RMSPE: 0.3592  RMSE: £1,163  R²: 0.8232
"""

import argparse
import time

from data.loader import load_data
from data.preprocessor import clean_data
from data.feature_engineer import engineer_features
from data.splitter import prepare_data
from models.ridge_model import RidgeModel
from models.lightgbm_model import LightGBMModel
from models.lstm_model import LSTMModel
from evaluation.metrics import evaluate_model, print_results
from evaluation.shap_analysis import (
    compute_shap_values,
    get_feature_importance,
    print_top_features,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rossmann Store Sales Forecaster"
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "lightgbm", "lstm", "all"],
        default="all",
        help="Which model to run (default: all)"
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM training (faster for development)"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Use baseline models without hyperparameter tuning"
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP analysis"
    )
    return parser.parse_args()


def run_pipeline(args):
    start_time = time.time()

    print("\n" + "="*60)
    print("  ROSSMANN STORE SALES FORECASTER")
    print("  MSc Data Science Project — Production Package")
    print("="*60 + "\n")

    # ── Stage 1: Data Pipeline ────────────────────────────────────────────────
    print("STAGE 1: Data Pipeline")
    print("-" * 40)

    raw = load_data()
    clean = clean_data(raw)
    features = engineer_features(clean)
    splits = prepare_data(features)

    X_train_LE  = splits["X_train_LE"]
    X_test_LE   = splits["X_test_LE"]
    X_train_OHE = splits["X_train_OHE"]
    X_test_OHE  = splits["X_test_OHE"]
    Y_train     = splits["Y_train"]
    Y_test      = splits["Y_test"]
    X_train_scaled = splits["X_train_scaled"]
    X_test_scaled  = splits["X_test_scaled"]

    print(f"\n  Training set: {len(X_train_LE):,} records")
    print(f"  Test set:     {len(X_test_LE):,} records")
    print(f"  Features:     {X_train_LE.shape[1]}")

    # ── Stage 2: Model Training and Evaluation ────────────────────────────────
    print("\nSTAGE 2: Model Training and Evaluation")
    print("-" * 40)

    all_results = {}

    # Ridge Regression
    if args.model in ["ridge", "all"]:
        print("\n[1/3] Ridge Regression")
        ridge = RidgeModel()
        if args.no_tune:
            ridge.fit(X_train_OHE, Y_train)
        else:
            ridge.tune(X_train_OHE, Y_train)

        train_results = evaluate_model(
            Y_train, ridge.predict(X_train_OHE)
        )
        test_results = evaluate_model(
            Y_test, ridge.predict(X_test_OHE)
        )
        print_results("Ridge Regression", train_results, test_results)
        all_results["Ridge"] = test_results

    # LightGBM
    if args.model in ["lightgbm", "all"]:
        print("\n[2/3] LightGBM")
        lgbm = LightGBMModel()
        if args.no_tune:
            lgbm.fit(X_train_LE, Y_train)
        else:
            lgbm.tune(X_train_LE, Y_train)

        train_results = evaluate_model(
            Y_train, lgbm.predict(X_train_LE)
        )
        test_results = evaluate_model(
            Y_test, lgbm.predict(X_test_LE)
        )
        print_results("LightGBM", train_results, test_results)
        all_results["LightGBM"] = test_results

    # LSTM
    skip_lstm = args.skip_lstm or args.model not in ["lstm", "all"]
    if not skip_lstm:
        print("\n[3/3] LSTM")
        lstm = LSTMModel()
        if args.no_tune:
            lstm.fit(X_train_scaled, Y_train)
        else:
            lstm.fit_optimised(X_train_scaled, Y_train)

        train_results = evaluate_model(
            Y_train,
            lstm.predict(X_train_scaled)
        )
        test_results = evaluate_model(
            Y_test,
            lstm.predict(X_test_scaled)
        )
        print_results("LSTM", train_results, test_results)
        all_results["LSTM"] = test_results

    # ── Stage 3: Final Comparison ─────────────────────────────────────────────
    if len(all_results) > 1:
        print("\nSTAGE 3: Final Model Comparison")
        print("-" * 40)
        print(f"\n  {'Model':<15} {'RMSPE':>8} {'RMSE (€)':>10} {'R²':>8}")
        print(f"  {'-'*43}")
        for model_name, results in sorted(
            all_results.items(), key=lambda x: x[1]["rmspe"]
        ):
            print(f"  {model_name:<15} "
                  f"{results['rmspe']:>8.4f} "
                  f"{results['rmse']:>10.2f} "
                  f"{results['r2']:>8.4f}")
        print()

    # ── Stage 4: SHAP Analysis ────────────────────────────────────────────────
    if not args.no_shap and "lgbm" in dir():
        print("\nSTAGE 4: SHAP Feature Importance Analysis")
        print("-" * 40)
        shap_values, X_sampled = compute_shap_values(lgbm, X_test_LE)
        importance = get_feature_importance(shap_values, X_sampled)
        print_top_features(importance)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.1f} seconds.")
    print("="*60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)