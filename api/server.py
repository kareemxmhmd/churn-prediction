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


from typing import List, Optional

class Customer(BaseModel):
    Tenure: Optional[float] = 12.0
    PreferredLoginDevice: Optional[str] = "Mobile Phone"
    CityTier: Optional[int] = 1
    WarehouseToHome: Optional[float] = 10.0
    PreferredPaymentMode: Optional[str] = "Debit Card"
    Gender: Optional[str] = "Male"
    HourSpendOnApp: Optional[float] = 2.0
    NumberOfDeviceRegistered: Optional[int] = 1
    PreferedOrderCat: Optional[str] = "Mobile"
    SatisfactionScore: Optional[int] = 3
    MaritalStatus: Optional[str] = "Single"
    NumberOfAddress: Optional[int] = 1
    Complain: Optional[int] = 0
    OrderAmountHikeFromlastYear: Optional[float] = 10.0
    CouponUsed: Optional[float] = 1.0
    OrderCount: Optional[float] = 1.0
    DaySinceLastOrder: Optional[float] = 5.0
    CashbackAmount: Optional[float] = 50.0


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


@app.post("/predict-batch")
def predict_batch(customers: List[Customer]):
    if model is None:
        return {"error": "Model not loaded", "detail": load_error}

    if not customers:
        return {"total": 0, "results": []}

    records = [c.model_dump() for c in customers]
    X = pd.DataFrame(records)

    # In case numeric features have None, fill with defaults
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype == 'object':
                X[col] = X[col].fillna("Unknown")
            else:
                X[col] = X[col].fillna(0.0)

    probabilities = model.predict_proba(X)[:, 1]

    results = []
    for prob in probabilities:
        prob_val = round(float(prob), 4)
        prediction = bool(prob_val >= 0.5)
        if prob_val < 0.3:
            risk_tier = "Low"
        elif prob_val < 0.6:
            risk_tier = "Medium"
        else:
            risk_tier = "High"

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
