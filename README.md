# Rossmann Store Sales Forecaster

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://github.com/Milonahmed96/rossmann-forecaster/workflows/Run%20Tests/badge.svg)

A production-grade machine learning package for daily sales forecasting across 1,115 Rossmann drug stores in Germany. The system covers the full modelling lifecycle — data ingestion, feature engineering, model training, evaluation, and calibrated uncertainty quantification.

## Versions

| Version | Capability | Key Result |
|---|---|---|
| v1.0 | Ridge Regression, LightGBM, Keras LSTM | LightGBM RMSPE 0.3409 |
| v2.0 | PyTorch LSTM with 7-day sequence windows | RMSPE 0.2871 — 15.8% improvement over LightGBM |
| v3.0 | Calibrated prediction intervals via conformal prediction | LightGBM 90% coverage, mean width €3,049 |

---

## Results

### Point Predictions

| Model | R² | RMSPE | RMSE (€) | Notes |
|---|---|---|---|---|
| Ridge Regression | 0.4722 | 0.4321 | 2,275 | Baseline |
| Keras LSTM (TIMESTEPS=1) | 0.8308 | 0.3650 | 1,158 | Architectural flaw — no temporal memory |
| LightGBM | 0.8696 | 0.3409 | 1,043 | Tuned, 1,500 trees |
| **PyTorch LSTM (window=7)** | **0.9131** | **0.2871** | **825** | **Best model** |

The PyTorch LSTM with 7-day sequence windows achieves RMSPE 0.2871 — a 15.8% improvement over LightGBM. The Keras LSTM used `TIMESTEPS=1`, making it functionally a dense network with no temporal memory. Feeding 7 consecutive days as a proper sequence allows the LSTM to learn weekly sales rhythms that LightGBM can only approximate through engineered lag features.

### Prediction Intervals (v3.0)

Calibrated 90% prediction intervals using split conformal prediction. Every interval is `[prediction - q, prediction + q]`, clipped to zero on the lower bound.

| Model | Point RMSPE | q (€) | Empirical Coverage | Mean Width |
|---|---|---|---|---|
| LightGBM | 0.3409 | 1,525 | 0.900 | €3,049 |
| PyTorch LSTM | 0.2871 | 4,682 | 0.877 | €9,191 |

LightGBM produces better-calibrated intervals despite weaker point predictions. The PyTorch LSTM's calibration period (weeks 49–52) has lower errors than the test period (weeks 53–60, which includes Christmas trading). Conformal prediction correctly exposes this non-stationarity rather than hiding it.

### SHAP Feature Importance (LightGBM)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Sales(Rolling\_Mean\_7) | 0.2245 |
| 2 | Promo | 0.1542 |
| 3 | DayOfWeek | 0.0516 |
| 4 | Day | 0.0509 |
| 5 | WeekOfYear | 0.0209 |
| 6 | DayOfYear | 0.0168 |
| 7 | Sales(Lag\_7) | 0.0166 |
| 8 | StoreType | 0.0151 |
| 9 | Assortment | 0.0082 |
| 10 | CompetitionDistance | 0.0070 |

---

## Project Structure
```
rossmann-forecaster/
├── data/
│   ├── loader.py              # Load and merge train.csv + store.csv
│   ├── preprocessor.py        # Clean data, handle nulls, convert dtypes
│   ├── feature_engineer.py    # Lag features, promo features, temporal features
│   ├── splitter.py            # Encode, scale, and split into train/test
│   └── sequence_dataset.py    # PyTorch Dataset with per-store sliding windows
├── models/
│   ├── ridge_model.py         # Ridge Regression with tuning
│   ├── lightgbm_model.py      # LightGBM with tuning
│   ├── lstm_model.py          # Keras LSTM (TIMESTEPS=1)
│   ├── pytorch_lstm.py        # PyTorch LSTM with 7-day sequence windows
│   └── conformal_predictor.py # Model-agnostic conformal prediction wrapper
├── evaluation/
│   ├── metrics.py             # RMSPE, RMSE, R²
│   └── shap_analysis.py       # SHAP feature importance
├── tests/                     # 17 unit tests — all passing
├── configs/                   # Model hyperparameters
└── main.py                    # Entry point — runs full pipeline
```

---

## Installation
```bash
git clone git@github.com:Milonahmed96/rossmann-forecaster.git
cd rossmann-forecaster
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set DATA_DIR to your local folder containing train.csv and store.csv
```

---

## Usage
```bash
# Full pipeline — all models with tuning and SHAP
python main.py

# Single model
python main.py --model lightgbm
python main.py --model pytorch_lstm --no-shap

# Development mode — fast
python main.py --skip-lstm --no-tune --no-shap

# Run tests
pytest tests/ -v
```

---

## Dataset

Rossmann Store Sales — [Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales)

Place `train.csv` and `store.csv` in the directory specified by `DATA_DIR` in your `.env` file.

| Dataset | Rows | Description |
|---|---|---|
| train.csv | 1,017,209 | Daily sales per store (Jan 2013 – Jul 2015) |
| store.csv | 1,115 | Store metadata |
| After preprocessing | 844,338 | Open days only |
| Training set | 781,898 | First ~92.5% by date |
| Test set | 62,440 | Final 8 weeks |

---

## Methodology

**Data preparation** — Removed closed store days (172,871 rows), log-transformed target `np.log1p(Sales)`, filled `CompetitionDistance` NaNs with median (2,320m), time-based train/test split with no data leakage.

**Feature engineering** — Temporal features (Year, Month, Day, DayOfWeek, WeekOfYear, DayOfYear), competition features (CompetitionOpen), holiday flag (Is_Holiday), promotional lifecycle features (Condition_of_Promo2, Promo2Status), and lag features (Sales\_Lag\_7, Sales\_Rolling\_Mean\_7) grouped by store.

**Evaluation** — Primary metric: RMSPE (scale-free, treats all 1,115 stores equally). Secondary: RMSE in Euros, R² on log scale.

---

## Licence

MIT
```
