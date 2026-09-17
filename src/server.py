import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.pkl"

from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI(title="Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Churn Prediction API is running", "docs": "/docs"}
model = None
metrics = None
load_error = None

try:
    model = joblib.load(MODEL_PATH)
    metrics = joblib.load(METRICS_PATH)
    print("Model and metrics loaded successfully.", flush=True)
except Exception as e:
    import traceback
    load_error = traceback.format_exc()
    print("ERROR loading model/metrics:\n", load_error, flush=True)


class Customer(BaseModel):
    Tenure: float
    PreferredLoginDevice: str
    CityTier: int
    WarehouseToHome: float
    PreferredPaymentMode: str
    Gender: str
    HourSpendOnApp: float
    NumberOfDeviceRegistered: int
    PreferedOrderCat: str
    SatisfactionScore: int
    MaritalStatus: str
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: float
    OrderCount: float
    DaySinceLastOrder: float
    CashbackAmount: float


def prepare_input(customer: Customer):
    return pd.DataFrame([customer.model_dump()])


@app.get("/health")
def health():
    if load_error:
        return {"status": "degraded", "error": load_error}
    return {"status": "ok"}


@app.get("/model/metadata")
def model_metadata():
    if metrics:
        return metrics
    return {"error": "Metrics not loaded", "detail": load_error}


@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        return {"error": "Model not loaded", "detail": load_error}

    X = prepare_input(customer)

    probability = model.predict_proba(X)[0][1]
    prediction = bool(probability >= 0.5)

    if probability < 0.3:
        risk_tier = "Low"
    elif probability < 0.6:
        risk_tier = "Medium"
    else:
        risk_tier = "High"

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction,
        "risk_tier": risk_tier,
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
