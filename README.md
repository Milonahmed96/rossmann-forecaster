# Rossmann Store Sales Forecaster

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://github.com/Milonahmed96/rossmann-forecaster/workflows/Run%20Tests/badge.svg)

Predicting daily promotional sales for 1,115 Rossmann drug stores across
Germany using LightGBM, LSTM, and Ridge Regression. Converted from an
MSc Data Science final project notebook into a clean, tested Python package.

## Results

| Model            | R²         | RMSPE      | RMSE (€/store/day) |
|------------------|------------|------------|---------------------|
| Ridge (baseline) | 0.3532     | 0.4360     | 2,479               |
| LSTM (optimised) | 0.8232     | 0.3592     | 1,163               |
| **LightGBM**     | **0.8696** | **0.3409** | **1,043**           |

LightGBM reduces per-store prediction error by 58% versus the Ridge baseline.

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
# Run full pipeline (all three models with tuning)
python main.py

# Run LightGBM only (fastest)
python main.py --model lightgbm

# Run without hyperparameter tuning (development mode)
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

### Models
- **Ridge:** OHE features, StandardScaler, RandomizedSearchCV over alpha
- **LightGBM:** Label encoded features, RandomizedSearchCV over 7 parameters
- **LSTM:** MinMax scaled features, timesteps=1, EarlyStopping(patience=5)

### Evaluation
- Primary: RMSPE — scale-free, treats all 1,115 stores equally
- Secondary: RMSE (Euro), R² (goodness of fit on log scale)
- SHAP analysis confirms promotional features as primary sales drivers

## Key Findings

- Promotions increase average daily sales by ~20% across all store types
- `Sales(Rolling_Mean_7)` is the strongest predictor (confirmed by SHAP)
- LightGBM outperforms LSTM on this tabular dataset — engineered lag features
  already capture temporal patterns that LSTM tries to learn implicitly
- Store type B shows the highest promotional lift (~25%)

## Tests
```bash
pytest tests/ -v   # 59 tests, all passing
```

## Licence

MIT

