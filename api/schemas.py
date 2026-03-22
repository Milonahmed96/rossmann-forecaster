from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Input features for a single store-day prediction."""
    store: int = Field(..., description="Store ID (1-1115)", example=1)
    day_of_week: int = Field(..., description="Day of week (1=Monday, 7=Sunday)", example=3)
    promo: int = Field(..., description="Whether store is running a promo (0 or 1)", example=1)
    state_holiday: str = Field("0", description="State holiday code (0, a, b, c)", example="0")
    school_holiday: int = Field(0, description="Whether school holiday (0 or 1)", example=0)
    store_type: str = Field("a", description="Store type (a, b, c, d)", example="a")
    assortment: str = Field("a", description="Assortment level (a, b, c)", example="a")
    competition_distance: float = Field(1000.0, description="Distance to nearest competitor in metres", example=1000.0)
    promo2: int = Field(0, description="Whether store participates in Promo2 (0 or 1)", example=0)
    month: int = Field(..., description="Month (1-12)", example=6)
    year: int = Field(..., description="Year", example=2015)
    day: int = Field(..., description="Day of month (1-31)", example=15)


class PredictionResponse(BaseModel):
    """Forecast response with point prediction and uncertainty interval."""
    store: int
    predicted_sales: float = Field(..., description="Point forecast in euros")
    lower_bound: float = Field(..., description="Lower bound of 90% prediction interval")
    upper_bound: float = Field(..., description="Upper bound of 90% prediction interval")
    confidence_level: float = Field(0.90, description="Confidence level of the interval")
    model: str = Field(..., description="Model used for prediction")


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    version: str