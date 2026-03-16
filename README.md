# Rossmann Store Sales Forecaster

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://github.com/Milonahmed96/rossmann-forecaster/workflows/Run%20Tests/badge.svg)

Predicting daily promotional sales for 1,115 Rossmann drug stores across
Germany using LightGBM, LSTM, and Ridge Regression. Converted from an
MSc Data Science final project notebook into a clean, tested Python package.

## Results

| Model | R² | RMSPE | RMSE (€/store/day) | Notes |
|---|---|---|---|---|
| Ridge Regression | 0.4722 | 0.4321 | 2,275 | Baseline — L2 regularisation |
| Keras LSTM (TIMESTEPS=1) | 0.8308 | 0.3650 | 1,158 | Architectural flaw — no temporal memory |
| LightGBM | 0.8696 | 0.3409 | 1,043 | Previous best — tuned, 1,500 trees |
| **PyTorch LSTM (window=7)** | **0.9131** | **0.2871** | **825** | **Best model — proper sequential input** |

### Key finding

The PyTorch LSTM with 7-day sequence windows achieves RMSPE 0.2871 — a **15.8% improvement over LightGBM** (0.3409) and a **21.3% improvement over the Keras LSTM** (0.3650).

The Keras LSTM used `TIMESTEPS=1`, making it functionally a dense network with no temporal memory. Fixing this architectural flaw — feeding 7 consecutive days as a proper sequence — allowed the LSTM to learn weekly sales rhythms and promotional dynamics that LightGBM can only approximate through engineered lag features.

## Prediction Intervals — Conformal Prediction (v3.0)

Calibrated 90% prediction intervals using split conformal prediction,
wrapping both models with a model-agnostic `ConformalPredictor` class.

| Model | Point RMSPE | q (€) | Coverage | Mean Width |
|---|---|---|---|---|
| LightGBM | 0.3409 | 1,525 | 0.900 | €3,049 |
| PyTorch LSTM | 0.2871 | 4,682 | 0.877 | €9,191 |

**q** is the interval half-width: every prediction interval is
`[prediction - q, prediction + q]`, clipped to zero on the lower bound.

**Key finding:** LightGBM produces better-calibrated intervals despite
weaker point predictions. The PyTorch LSTM's calibration period
(weeks 49–52) has lower errors than the test period (weeks 53–60,
Christmas trading) — conformal prediction correctly exposes this
non-stationarity rather than hiding it.

## SHAP Feature Importance (Top 10)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | Sales(Rolling_Mean_7) | 0.2245 |
| 2 | Promo | 0.1542 |
| 3 | DayOfWeek | 0.0516 |
| 4 | Day | 0.0509 |
| 5 | WeekOfYear | 0.0209 |
| 6 | DayOfYear | 0.0168 |
| 7 | Sales(Lag_7) | 0.0166 |
| 8 | StoreType | 0.0151 |
| 9 | Assortment | 0.0082 |
| 10 | CompetitionDistance | 0.0070 |

Promotions confirmed as the second strongest sales driver after recent
sales history — validating the MSc research focus on promotional influence.

## Project Structure
```
rossmann-forecaster/
├── data/
│   ├── loader.py            # Load and merge train.csv + store.csv
│   ├── preprocessor.py      # Clean data, handle nulls, convert dtypes
│   ├── feature_engineer.py  # Lag features, promo features, temporal features
│   └── splitter.py          # Encode, scale, and split into train/test
├── models/
│   ├── ridge_model.py       # Ridge Regression with tuning
│   ├── lightgbm_model.py    # LightGBM with tuning (best model)
│   └── lstm_model.py        # LSTM baseline and optimised
├── evaluation/
│   ├── metrics.py           # RMSPE, RMSE, R² implementations
│   └── shap_analysis.py     # SHAP feature importance
├── tests/                   # 59 unit tests — all passing
├── configs/                 # Model hyperparameters
├── notebooks/               # Exploratory data analysis
└── main.py                  # Entry point — runs full pipeline
```

## Installation
```bash
git clone git@github.com:Milonahmed96/rossmann-forecaster.git
cd rossmann-forecaster
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set DATA_DIR to your local folder containing train.csv and store.csv
```

## Usage
```bash
# Run full pipeline (all three models with tuning + SHAP)
python main.py

# Run LightGBM only with tuning
python main.py --model lightgbm

# Run without hyperparameter tuning (development mode — fast)
python main.py --no-tune --skip-lstm

# Skip SHAP analysis
python main.py --no-shap
```

## Dataset

Rossmann Store Sales — [Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales)

Place `train.csv` and `store.csv` in the directory specified by `DATA_DIR`
in your `.env` file.

| Dataset | Rows | Description |
|---------|------|-------------|
| train.csv | 1,017,209 | Daily sales per store (Jan 2013 – Jul 2015) |
| store.csv | 1,115 | Store metadata |
| After preprocessing | 844,338 | Open days with non-zero sales only |
| Training set | 781,898 | ~92.5% of data |
| Test set | 62,440 | Final 8 weeks — ~7.5% of data |

## Methodology

### Data Preparation
- Removed closed store days — 172,871 rows dropped
- Log-transformed target `np.log1p(Sales)` to reduce right skew
- Filled `CompetitionDistance` NaNs with median (2,320m)
- Time-based train/test split — final 8 weeks as test set (no data leakage)

### Feature Engineering
- **Temporal:** Year, Month, Day, DayOfWeek, WeekOfYear, DayOfYear
- **Competition:** CompetitionOpen — months since competition opened
- **Holiday:** Is_Holiday — combined StateHoliday and SchoolHoliday flag
- **Promo2:** Condition_of_Promo2, Promo2Status — promotional lifecycle
- **Lag:** Sales(Lag_7), Sales(Rolling_Mean_7) — grouped by store, no leakage

### Models and Tuning

**Ridge Regression**
- One-hot encoded features, StandardScaler
- RandomizedSearchCV over alpha (logspace -4 to 2, 10 iterations)
- Best alpha: 10.72

**LightGBM** ← Best Model
- Label encoded features
- RandomizedSearchCV over 7 parameters, 10 iterations, 3-fold TimeSeriesSplit
- Best params: subsample=0.7, num_leaves=70, n_estimators=1500,
  min_child_samples=100, max_depth=20, learning_rate=0.01, colsample_bytree=0.9

**LSTM**
- MinMax scaled features, timesteps=1
- Architecture: LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(1)
- EarlyStopping(patience=5) — stopped at epoch 44, best weights from epoch 39

### Evaluation
- Primary: RMSPE — scale-free, treats all 1,115 stores equally
- Secondary: RMSE (Euro), R² (goodness of fit on log scale)
- SHAP analysis confirms promotional features as primary sales drivers

## Key Findings

- `Sales(Rolling_Mean_7)` is the strongest predictor (SHAP=0.2245)
- `Promo` is the second strongest driver (SHAP=0.1542) — validates research focus
- LightGBM outperforms LSTM — engineered lag features already capture
  temporal patterns that LSTM tries to learn implicitly
- Full pipeline runs in ~25 minutes on a standard laptop

## Tests
```bash
pytest tests/ -v   # 59 tests, all passing
```

## Licence

MIT
