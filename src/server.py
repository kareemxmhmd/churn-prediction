from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from build_features import TARGET_COL

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
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
feature_columns = joblib.load(COLUMNS_PATH)
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
    df = pd.DataFrame([customer.model_dump()])

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols)

    # add any missing columns from training (one-hot categories not present in this input)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # keep only columns the model was trained on, in the same order
    df = df[feature_columns]

    return df


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
    # If run via `python src/server.py`, the module name is just `server`
    uvicorn.run("server:app", host="0.0.0.0", port=port)