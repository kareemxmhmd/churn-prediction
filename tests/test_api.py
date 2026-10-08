from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from api.server import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_metadata_endpoint():
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "f1" in data
    assert "roc_auc" in data
    assert "pr_auc" in data
    assert data["f1"] > 0.70


def test_predict_valid_customer():
    payload = {
        "Tenure": 18.0,
        "CityTier": 2,
        "WarehouseToHome": 12.0,
        "HourSpendOnApp": 3.0,
        "NumberOfDeviceRegistered": 4,
        "SatisfactionScore": 4,
        "NumberOfAddress": 3,
        "Complain": 0,
        "OrderAmountHikeFromlastYear": 14.0,
        "CouponUsed": 1.0,
        "OrderCount": 2.0,
        "DaySinceLastOrder": 5.0,
        "CashbackAmount": 160.0,
        "PreferredLoginDevice": "Phone",
        "PreferredPaymentMode": "Credit Card",
        "Gender": "Female",
        "PreferedOrderCat": "Laptop & Accessory",
        "MaritalStatus": "Married",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_tier" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)
    assert data["risk_tier"] in ["Low", "Medium", "High"]


def test_predict_with_missing_fields():
    """Verify that partial payloads and null fields are imputed cleanly without crashing."""
    payload = {
        "Tenure": None,
        "SatisfactionScore": 2,
        "Complain": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert data["risk_tier"] in ["Low", "Medium", "High"]


def test_predict_batch_regular():
    batch = [
        {"Tenure": 24.0, "SatisfactionScore": 5, "Complain": 0},
        {"Tenure": 1.0, "SatisfactionScore": 1, "Complain": 1},
    ]
    response = client.post("/predict-batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    for res in data["results"]:
        assert 0.0 <= res["churn_probability"] <= 1.0
        assert res["risk_tier"] in ["Low", "Medium", "High"]


def test_predict_batch_with_null_columns():
    """Verify the fix for the fatal batch crash where all records contain None in a numeric column."""
    batch = [
        {"Tenure": None, "WarehouseToHome": None},
        {"Tenure": None, "WarehouseToHome": None},
    ]
    response = client.post("/predict-batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


def test_predict_batch_empty():
    response = client.post("/predict-batch", json=[])
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["results"] == []


def test_validation_errors():
    """Verify strict Pydantic range constraints reject invalid inputs with 422."""
    # Negative tenure
    r1 = client.post("/predict", json={"Tenure": -10.0})
    assert r1.status_code == 422

    # Invalid satisfaction score (> 5)
    r2 = client.post("/predict", json={"SatisfactionScore": 6})
    assert r2.status_code == 422

    # Invalid satisfaction score (< 1)
    r3 = client.post("/predict", json={"SatisfactionScore": 0})
    assert r3.status_code == 422

    # Invalid city tier (> 3)
    r4 = client.post("/predict", json={"CityTier": 4})
    assert r4.status_code == 422
