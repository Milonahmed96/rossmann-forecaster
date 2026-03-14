# Rossmann Store Sales Forecaster

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://github.com/Milonahmed96/rossmann-forecaster/workflows/Run%20Tests/badge.svg)

Predicting daily promotional sales for 1,115 Rossmann drug stores across
Germany using LightGBM, LSTM, and Ridge Regression.

## Results

| Model        | R²         | RMSPE      | RMSE (€/store) |
|--------------|------------|------------|----------------|
| Ridge        | 0.5910     | 0.4056     | 1,926          |
| LSTM         | 0.8232     | 0.3592     | 1,163          |
| **LightGBM** | **0.8679** | **0.3464** | **1,058**      |

LightGBM reduces per-store prediction error by 45% versus the Ridge baseline.

## Project Structure
```
rossmann-forecaster/
├── data/               # Data loading, cleaning, feature engineering
├── models/             # Ridge, LightGBM, LSTM implementations
├── evaluation/         # Metrics and SHAP analysis
├── tests/              # Unit tests for all modules
├── notebooks/          # Exploratory data analysis
├── configs/            # Model hyperparameters
└── main.py             # Entry point — runs full pipeline
```

## Installation
```bash
git clone git@github.com:Milonahmed96/rossmann-forecaster.git
cd rossmann-forecaster
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set DATA_DIR to your local data folder
```

## Usage
```bash
python main.py
```

## Dataset

Rossmann Store Sales dataset from Kaggle. Place `train.csv` and `store.csv`
in the directory specified by `DATA_DIR` in your `.env` file.

## Methodology

- **Data:** 844,338 records across 1,115 stores after removing closed days
- **Target:** Log-transformed daily sales (np.log1p) to reduce right skew
- **Validation:** TimeSeriesSplit — final 8 weeks as test set, no data leakage
- **Key features:** 7-day sales lag, 7-day rolling mean, promotional lifecycle
  features, competition duration, combined holiday flag
- **Best model:** LightGBM with RandomizedSearchCV hyperparameter tuning

## Licence

MIT

