import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from api.predictor import get_model, predict

# ── Lifespan — load model at startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading model...")
    try:
        get_model()
        print("[startup] Model loaded successfully")
    except Exception as e:
        print(f"[startup] Warning: model loading failed — {e}")
    yield
    print("[shutdown] Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="""
Predicts daily sales for Rossmann drug stores using a PyTorch LSTM model
with calibrated 90% prediction intervals from conformal prediction.

## Endpoints
- **POST /predict** — single store-day prediction
- **GET /health** — API health check
- **GET /docs** — interactive API documentation (you are here)

## Model Performance
- RMSPE: 0.2871 (PyTorch LSTM)
- Coverage: 90% conformal prediction intervals
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Check if the API and model are running correctly."""
    try:
        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sales(request: PredictionRequest):
    """
    Predict daily sales for a single Rossmann store.

    Returns a point forecast in euros plus a calibrated 90% prediction interval.
    """
    try:
        start = time.time()
        result = predict(request)
        latency = round((time.time() - start) * 1000, 2)
        print(f"[predict] store={request.store} prediction={result['predicted_sales']} latency={latency}ms")
        return PredictionResponse(**result)
    except Exception as e:
        print(f"[predict] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """API root — redirect to docs."""
    return {
        "message": "Rossmann Sales Forecasting API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }