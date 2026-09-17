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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load(MODEL_PATH)
metrics = joblib.load(METRICS_PATH)


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
    return {"status": "ok"}


@app.get("/model/metadata")
def model_metadata():
    return metrics


@app.post("/predict")
def predict(customer: Customer):
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
