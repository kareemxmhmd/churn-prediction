import os
from pathlib import Path
import sys
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Ensure custom transformers are accessible to unpickler
from src.build_features import DomainFeatureEngineer

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML API for early identification and risk tiering of customer churn.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
metrics = None
feature_columns = None
load_error = None

try:
    model = joblib.load(MODEL_PATH)
    metrics = joblib.load(METRICS_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    print("Model, metrics, and feature metadata loaded successfully.", flush=True)
except Exception as e:
    import traceback
    load_error = traceback.format_exc()
    print("ERROR loading model artifacts:\n", load_error, flush=True)


class Customer(BaseModel):
    Tenure: Optional[float] = Field(None, ge=0.0, description="Account lifespan in months")
    PreferredLoginDevice: Optional[str] = Field(None, description="Device used to log in (e.g. Phone, Computer)")
    CityTier: Optional[int] = Field(None, ge=1, le=3, description="City tier category (1, 2, or 3)")
    WarehouseToHome: Optional[float] = Field(None, ge=0.0, description="Distance from warehouse to home")
    PreferredPaymentMode: Optional[str] = Field(None, description="Payment method (e.g. Debit Card, Credit Card, UPI)")
    Gender: Optional[str] = Field(None, description="Customer gender (Male, Female)")
    HourSpendOnApp: Optional[float] = Field(None, ge=0.0, le=24.0, description="Hours spent per day on mobile app")
    NumberOfDeviceRegistered: Optional[int] = Field(None, ge=1, le=20, description="Total devices registered to account")
    PreferedOrderCat: Optional[str] = Field(None, description="Preferred product category (e.g. Mobile, Laptop & Accessory)")
    SatisfactionScore: Optional[int] = Field(None, ge=1, le=5, description="Customer satisfaction rating (1 to 5)")
    MaritalStatus: Optional[str] = Field(None, description="Marital status (Single, Married, Divorced)")
    NumberOfAddress: Optional[int] = Field(None, ge=1, description="Number of registered delivery addresses")
    Complain: Optional[int] = Field(None, ge=0, le=1, description="Whether a complaint was recorded (1) or not (0)")
    OrderAmountHikeFromlastYear: Optional[float] = Field(None, description="Percentage increase in spend compared to last year")
    CouponUsed: Optional[float] = Field(None, ge=0.0, description="Number of coupons redeemed")
    OrderCount: Optional[float] = Field(None, ge=0.0, description="Total order count")
    DaySinceLastOrder: Optional[float] = Field(None, ge=0.0, description="Days elapsed since the most recent order")
    CashbackAmount: Optional[float] = Field(None, ge=0.0, description="Average cashback amount earned")


def assign_risk_tier(probability: float) -> str:
    """Translates predicted churn probability into actionable retention risk tiers."""
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    else:
        return "High"


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Customer Churn Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if load_error or model is None:
        return {"status": "degraded", "error": load_error}
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": metrics.get("model_name", "calibrated_model") if metrics else "unknown",
    }


@app.get("/model/metadata")
def model_metadata():
    if metrics:
        return metrics
    raise HTTPException(status_code=503, detail=f"Model metrics not available: {load_error}")


@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    df_input = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(df_input)[0, 1])
    prediction = bool(probability >= 0.50)
    risk_tier = assign_risk_tier(probability)

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": prediction,
        "risk_tier": risk_tier,
    }


@app.post("/predict-batch")
def predict_batch(customers: List[Customer]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    if not customers:
        return {"total": 0, "results": []}

    df_input = pd.DataFrame([c.model_dump() for c in customers])
    probabilities = model.predict_proba(df_input)[:, 1]

    results = []
    for prob in probabilities:
        prob_val = round(float(prob), 4)
        prediction = bool(prob_val >= 0.50)
        risk_tier = assign_risk_tier(prob_val)
        results.append({
            "churn_probability": prob_val,
            "churn_prediction": prediction,
            "risk_tier": risk_tier,
        })

    return {
        "total": len(results),
        "results": results,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
