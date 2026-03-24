# Rossmann Forecaster — Production Runbook

**System:** Rossmann Sales Forecasting API  
**Version:** 1.0.0  
**Model:** PyTorch LSTM (RMSPE=0.2871, R²=0.9131)  
**Live URL:** http://16.171.133.70:8000  
**Last Updated:** March 2026

---

## 1. System Overview

The Rossmann Forecaster is a containerised FastAPI service that predicts daily
sales for 1,115 Rossmann drug stores with calibrated 90% prediction intervals.
```
Request → FastAPI (port 8000) → PyTorch LSTM → Point Forecast + Interval → Response
```

**Infrastructure:**
- Server: AWS EC2 t2.micro (Amazon Linux 2023)
- Container: Docker — milon96/rossmann-forecaster:v3
- CI/CD: GitHub Actions — runs tests on every push to main
- Experiment tracking: MLflow (local mlruns/)
- Drift monitoring: KS test on input features (monitoring/drift.py)

---

## 2. How to Check if the System is Running

**From anywhere:**
```bash
curl http://16.171.133.70:8000/health
```

Expected response:
```json
{"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

**On the EC2 server:**
```bash
ssh -i "rossmann-key.pem" ec2-user@16.171.133.70
docker ps
```

Expected: one container named `rossmann-api` with status `Up`.

---

## 3. How to Make a Prediction
```bash
curl -X POST http://16.171.133.70:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store": 1,
    "day_of_week": 3,
    "promo": 1,
    "state_holiday": "0",
    "school_holiday": 0,
    "store_type": "a",
    "assortment": "a",
    "competition_distance": 1270.0,
    "promo2": 0,
    "month": 6,
    "year": 2015,
    "day": 15
  }'
```

Expected response:
```json
{
  "store": 1,
  "predicted_sales": 3692.24,
  "lower_bound": 3138.40,
  "upper_bound": 4246.07,
  "confidence_level": 0.9,
  "model": "PyTorch LSTM"
}
```

---

## 4. How to Restart the API

SSH into the server:
```bash
ssh -i "rossmann-key.pem" ec2-user@16.171.133.70
```

Restart the container:
```bash
docker stop rossmann-api
docker rm rossmann-api
docker run -d -p 8000:8000 --name rossmann-api milon96/rossmann-forecaster:v3
```

Verify it started:
```bash
curl http://localhost:8000/health
```

---

## 5. How to Deploy a New Model Version

**Step 1 — Build and push new image locally:**
```bash
docker build -t rossmann-forecaster:v4 .
docker tag rossmann-forecaster:v4 milon96/rossmann-forecaster:v4
docker push milon96/rossmann-forecaster:v4
```

**Step 2 — Deploy on EC2:**
```bash
ssh -i "rossmann-key.pem" ec2-user@16.171.133.70
docker pull milon96/rossmann-forecaster:v4
docker stop rossmann-api
docker rm rossmann-api
docker run -d -p 8000:8000 --name rossmann-api milon96/rossmann-forecaster:v4
curl http://localhost:8000/health
```

**Step 3 — Log new model in MLflow:**
```bash
python log_models.py
```

---

## 6. How to Roll Back to Previous Version

If the new deployment breaks, roll back immediately:
```bash
ssh -i "rossmann-key.pem" ec2-user@16.171.133.70
docker stop rossmann-api
docker rm rossmann-api
docker run -d -p 8000:8000 --name rossmann-api milon96/rossmann-forecaster:v3
curl http://localhost:8000/health
```

Rollback takes under 60 seconds.

---

## 7. How to Run Drift Monitoring

Run weekly to detect input feature distribution shifts:
```bash
python monitoring/drift.py
```

Reports saved to `monitoring/reports/` as HTML files.

**What to do if drift is detected:**
1. Open the HTML report and identify which features drifted
2. Check if the drift is real (new stores, promotions policy change) or a data issue
3. If real drift: retrain the model on more recent data
4. If data issue: fix the data pipeline and re-run

**Alert threshold:** 30% of features drifted = action required.

---

## 8. How to Run Tests
```bash
pytest tests/ -v
```

All 11 tests must pass before any deployment. GitHub Actions runs this
automatically on every push to main.

---

## 9. How to View Experiment History
```bash
mlflow ui
```

Open http://localhost:5000 to compare all model versions and metrics.

---

## 10. Common Problems and Fixes

| Problem | Likely Cause | Fix |
|---|---|---|
| `/health` returns 503 | Container stopped | Restart container (Section 4) |
| Predictions look wrong | Model not loaded | Check `/health` for model_loaded=true |
| Container won't start | Port 8000 in use | `docker ps` then `docker stop` old container |
| EC2 unreachable | Instance stopped | Start instance in AWS Console |
| Drift alert firing | Distribution shift | Run drift report, investigate (Section 7) |
| Tests failing on CI | Dependency issue | Check GitHub Actions log, update requirements.txt |

---

## 11. Key File Locations

| File | Purpose |
|---|---|
| `api/main.py` | FastAPI application entry point |
| `api/predictor.py` | Model loading and inference |
| `models/best_pytorch_lstm.pt` | Trained PyTorch LSTM weights |
| `monitoring/drift.py` | Drift detection script |
| `monitoring/reports/` | Saved drift HTML reports |
| `log_models.py` | MLflow model logging |
| `mlruns/` | MLflow experiment data |
| `.github/workflows/ci.yml` | GitHub Actions CI pipeline |
| `Dockerfile` | Container build instructions |

---

## 12. Contacts

| Role | Name |
|---|---|
| Model Owner | Milon Ahmed |
| GitHub | github.com/Milonahmed96/rossmann-forecaster |
| Docker Hub | hub.docker.com/r/milon96/rossmann-forecaster |

---

*Rossmann Forecaster Runbook · Milon Ahmed · March 2026*
