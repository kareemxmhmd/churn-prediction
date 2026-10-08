from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_data
from src.build_features import (
    clean_categories,
    engineer_domain_features,
    DomainFeatureEngineer,
    split_features_target,
    RAW_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"


def test_data_loader():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Churn" in df.columns
    assert "CustomerID" in df.columns
    assert len(df) > 5000


def test_clean_categories():
    df = pd.DataFrame({
        "PreferredLoginDevice": ["Mobile Phone", "Phone", "Computer"],
        "PreferredPaymentMode": ["CC", "Credit Card", "COD"],
        "PreferedOrderCat": ["Mobile Phone", "Mobile", "Grocery"],
    })
    cleaned = clean_categories(df)
    assert cleaned["PreferredLoginDevice"].tolist() == ["Phone", "Phone", "Computer"]
    assert cleaned["PreferredPaymentMode"].tolist() == ["Credit Card", "Credit Card", "Cash on Delivery"]
    assert cleaned["PreferedOrderCat"].tolist() == ["Mobile", "Mobile", "Grocery"]


def test_engineer_domain_features():
    df = pd.DataFrame({
        "Tenure": [10.0, 0.0, None],
        "OrderCount": [20.0, 5.0, 1.0],
        "DaySinceLastOrder": [5.0, 10.0, None],
        "CouponUsed": [4.0, 0.0, None],
        "Complain": [1, 0, 1],
        "SatisfactionScore": [1, 4, 2],
    })
    feat = engineer_domain_features(df)
    
    assert "OrderVelocity" in feat.columns
    assert "InactivityRatio" in feat.columns
    assert "PromotionDependency" in feat.columns
    assert "FrictionIndicator" in feat.columns

    # OrderVelocity: 20/10 = 2.0; 5/max(0,1) = 5.0; NaN handles gracefully
    assert feat["OrderVelocity"].iloc[0] == 2.0
    assert feat["OrderVelocity"].iloc[1] == 5.0
    assert np.isnan(feat["OrderVelocity"].iloc[2])

    # FrictionIndicator: (Complain==1 & Satisfaction<=2) -> 1.0, 0.0, 1.0
    assert feat["FrictionIndicator"].tolist() == [1.0, 0.0, 1.0]


def test_domain_feature_engineer_transformer():
    transformer = DomainFeatureEngineer()
    df = pd.DataFrame({
        "Tenure": [12.0],
        "OrderCount": [6.0],
        "DaySinceLastOrder": [3.0],
        "CouponUsed": [1.0],
        "Complain": [0],
        "SatisfactionScore": [4],
        "PreferredLoginDevice": ["Mobile Phone"],
    })
    transformed = transformer.transform(df)
    assert transformed["PreferredLoginDevice"].iloc[0] == "Phone"
    assert "OrderVelocity" in transformed.columns
    assert transformed["OrderVelocity"].iloc[0] == 0.5


def test_split_features_target():
    df = pd.DataFrame({
        "CustomerID": [1, 2],
        "Tenure": [5.0, 10.0],
        "Churn": [0, 1],
    })
    X, y = split_features_target(df)
    assert "CustomerID" not in X.columns
    assert "Churn" not in X.columns
    assert list(y) == [0, 1]


def test_model_artifact_predictions():
    assert MODEL_PATH.exists(), f"Model artifact not found at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLUMNS_PATH)

    df_raw = load_data()
    X, _ = split_features_target(df_raw)
    sample = X.head(5)

    probs = model.predict_proba(sample)
    assert probs.shape == (5, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_model_robustness_to_missing_and_unseen():
    """Verify that model handles NaNs and unseen categories without crashing."""
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLUMNS_PATH)

    edge_case = pd.DataFrame([{c: None for c in cols}])
    edge_case["PreferredLoginDevice"] = "BrandNewDevice"
    edge_case["PreferredPaymentMode"] = "CryptoCurrency"

    probs = model.predict_proba(edge_case)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
