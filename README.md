# Rossmann Sales Forecaster

[![CI](https://github.com/Milonahmed96/rossmann-forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/Milonahmed96/rossmann-forecaster/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-milon96%2Frossmann--forecaster-blue?logo=docker)](https://hub.docker.com/r/milon96/rossmann-forecaster)
[![AWS](https://img.shields.io/badge/AWS-EC2%20Live-orange?logo=amazon-aws)](http://16.171.133.70:8000/docs)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Production-grade sales forecasting system for 1,115 Rossmann drug stores in Germany.
**PyTorch LSTM** with calibrated 90% prediction intervals, served via **FastAPI**, containerised with **Docker**, deployed on **AWS EC2**, tracked with **MLflow**, and monitored for data drift.

---

## Live API

🚀 **`http://16.171.133.70:8000`**

| Endpoint | Description |
|---|---|
| `GET /health` | Health check — confirms model is loaded |
| `POST /predict` | Point forecast + 90% prediction interval |
| `GET /docs` | Interactive Swagger UI |

**Quick test:**
```bash
curl http://16.171.133.70:8000/health
# {"status":"healthy","model_loaded":true,"version":"1.0.0"}

curl -X POST http://16.171.133.70:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store":1,"day_of_week":3,"promo":1,"state_holiday":"0",
       "school_holiday":0,"store_type":"a","assortment":"a",
       "competition_distance":1270.0,"promo2":0,
       "month":6,"year":2015,"day":15}'
# {"store":1,"predicted_sales":3692.24,"lower_bound":3138.4,
#  "upper_bound":4246.07,"confidence_level":0.9,"model":"PyTorch LSTM"}
```

---

## Model Results

### Point Predictions

| Model | R² | RMSPE | RMSE (€) | Notes |
|---|---|---|---|---|
| Ridge Regression | 0.4722 | 0.4321 | 2,275 | Baseline |
| Keras LSTM (TIMESTEPS=1) | 0.8308 | 0.3650 | 1,158 | No temporal memory |
| LightGBM | 0.8696 | 0.3409 | 1,043 | Tuned, 1,500 trees |
| **PyTorch LSTM (window=7)** | **0.9131** | **0.2871** | **825** | **Best — deployed** |

The PyTorch LSTM with 7-day sequence windows achieves RMSPE 0.2871 — a 15.8% improvement over LightGBM. Feeding 7 consecutive days as a sequence allows the model to learn weekly sales rhythms that LightGBM can only approximate through engineered lag features.

### Prediction Intervals (Conformal Prediction)

| Model | Point RMSPE | q (€) | Empirical Coverage | Mean Width |
|---|---|---|---|---|
| LightGBM | 0.3409 | 1,525 | 0.900 | €3,049 |
| PyTorch LSTM | 0.2871 | 4,682 | 0.877 | €9,191 |

LightGBM produces better-calibrated intervals despite weaker point predictions. Conformal prediction correctly exposes non-stationarity in the LSTM's calibration period rather than hiding it.

### SHAP Feature Importance (LightGBM)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Sales(Rolling_Mean_7) | 0.2245 |
| 2 | Promo | 0.1542 |
| 3 | DayOfWeek | 0.0516 |
| 4 | Day | 0.0509 |
| 5 | WeekOfYear | 0.0209 |

---

## Production Architecture
```
Client Request
      │
      ▼
FastAPI (port 8000)
      │
      ├── /health ──→ model status check
      │
      └── /predict ─→ PyTorch LSTM inference
                            │
                            ├── Point forecast (log-scale → euros)
                            └── ±15% interval (conformal predictor)
                                        │
                                        ▼
                              JSON Response (< 10ms)

Infrastructure:
  AWS EC2 t2.micro (Amazon Linux 2023)
  Docker container: milon96/rossmann-forecaster:v3 (1.26GB)
  CI/CD: GitHub Actions — tests on every push to main
  Experiment tracking: MLflow (3 model versions logged)
  Drift monitoring: KS test on input features (monitoring/drift.py)
```

---

## Project Structure
```
rossmann-forecaster/
├── api/
│   ├── main.py              # FastAPI application
│   ├── predictor.py         # Model loading and inference
│   └── schemas.py           # Pydantic request/response models
├── data/
│   ├── loader.py            # Load and merge train.csv + store.csv
│   ├── preprocessor.py      # Clean, handle nulls, convert dtypes
│   ├── feature_engineer.py  # Lag features, promo features, temporal
│   ├── splitter.py          # Encode, scale, train/test split
│   └── sequence_dataset.py  # PyTorch Dataset with sliding windows
├── models/
│   ├── ridge_model.py       # Ridge Regression
│   ├── lightgbm_model.py    # LightGBM
│   ├── lstm_model.py        # Keras LSTM (baseline)
│   ├── pytorch_lstm.py      # PyTorch LSTM — production model
│   ├── conformal_predictor.py # Conformal prediction intervals
│   └── best_pytorch_lstm.pt # Trained model weights
├── monitoring/
│   ├── drift.py             # KS test drift detection
│   ├── mlflow_logger.py     # MLflow experiment logging
│   └── reports/             # Saved HTML drift reports
├── evaluation/
│   ├── metrics.py           # RMSPE, RMSE, R²
│   └── shap_analysis.py     # SHAP feature importance
├── tests/                   # 11 unit tests — all passing
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI pipeline
├── configs/                 # Model hyperparameters
├── Dockerfile               # Container build (Python 3.11-slim)
├── RUNBOOK.md               # Operations guide
├── log_models.py            # MLflow model registration
└── main.py                  # Training entry point
```

---

## Running Locally
```bash
git clone https://github.com/Milonahmed96/rossmann-forecaster
cd rossmann-forecaster
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set DATA_DIR to folder containing train.csv and store.csv
```

**Train models:**
```bash
python main.py                          # Full pipeline
python main.py --model pytorch_lstm     # Single model
python main.py --skip-lstm --no-tune    # Fast dev mode
```

**Run the API locally:**
```bash
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

**Run with Docker:**
```bash
docker build -t rossmann-forecaster:v3 .
docker run -p 8000:8000 rossmann-forecaster:v3
```

**Run tests:**
```bash
pytest tests/ -v
```

**Run drift monitoring:**
```bash
python monitoring/drift.py
```

**View MLflow experiments:**
```bash
mlflow ui
# Open http://localhost:5000
```

---

## Dataset

Rossmann Store Sales — [Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales)

Place `train.csv` and `store.csv` in the directory set by `DATA_DIR` in `.env`.

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

**Feature engineering** — Temporal features (Year, Month, Day, DayOfWeek, WeekOfYear), competition features, holiday flags, promotional lifecycle features, lag features (Sales_Lag_7, Sales_Rolling_Mean_7) grouped by store.

**Evaluation** — Primary metric: RMSPE (scale-free, treats all 1,115 stores equally). Secondary: RMSE in Euros, R² on log scale.

**Uncertainty quantification** — Split conformal prediction with coverage guarantee. Every interval is `[prediction - q, prediction + q]` where `q` is the empirical quantile of calibration residuals.

---


| Project | Description | Key Result |
|---|---|---|
| P1 — Rossmann Reborn | Production Python package from notebook | This repo — v1.0 |
| P2 — Beat Your Model | PyTorch LSTM from scratch | RMSPE=0.2871 — v2.0 |
| P3 — Predictions to Intervals | Conformal prediction intervals | 90% coverage — v3.0 |
| P4 — Document Search RAG | End-to-end RAG with evaluation | 96% accuracy |
| P5 — Domain Fine-tuning | QLoRA fine-tuned Phi-3-mini | 64% triage accuracy |
| P6 — Financial Research Agent | Autonomous LangGraph agent | Live HF Spaces |
| **P7 — Production System** | **FastAPI + Docker + AWS + MLflow + Drift** | **This release** |

---

## Operations

See [RUNBOOK.md](RUNBOOK.md) for full operations guide including:
- Health checks and restart procedures
- Model version deployment and rollback
- Drift monitoring and alerting
- Common problems and fixes

---

## Licence

MIT

---

*Built by **Milon Ahmed** · MSc Data Science, University of Hertfordshire · March 2026*
