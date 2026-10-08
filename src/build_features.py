import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COL = "Churn"

# Cleaned category synonym mappings
CATEGORY_MAPPINGS = {
    "PreferredLoginDevice": {"Mobile Phone": "Phone"},
    "PreferredPaymentMode": {
        "CC": "Credit Card",
        "COD": "Cash on Delivery",
    },
    "PreferedOrderCat": {"Mobile Phone": "Mobile"},
}

RAW_NUMERIC_FEATURES = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

ENGINEERED_NUMERIC_FEATURES = [
    "OrderVelocity",
    "InactivityRatio",
    "PromotionDependency",
    "FrictionIndicator",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes redundant categorical values to prevent feature fragmentation."""
    df = df.copy()
    for col, mapping in CATEGORY_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates domain-driven behavioral features reflecting e-commerce customer habits.
    
    1. Order Velocity: OrderCount / max(Tenure, 1)
    2. Inactivity Ratio: DaySinceLastOrder / max(Tenure * 30, 1)
    3. Promotion Dependency: CouponUsed / max(OrderCount, 1)
    4. Friction Indicator: Interaction between Complain == 1 and SatisfactionScore <= 2
    """
    df = df.copy()

    # Safely convert any numeric column with Python None to float64 NaN
    for col in RAW_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Order Velocity: orders per month of tenure
    if "OrderCount" in df.columns and "Tenure" in df.columns:
        tenure_safe = df["Tenure"].clip(lower=1.0)
        df["OrderVelocity"] = df["OrderCount"] / tenure_safe
    else:
        df["OrderVelocity"] = np.nan

    # 2. Inactivity Ratio: days since last order relative to customer lifetime days
    if "DaySinceLastOrder" in df.columns and "Tenure" in df.columns:
        tenure_days_safe = (df["Tenure"] * 30.0).clip(lower=1.0)
        df["InactivityRatio"] = df["DaySinceLastOrder"] / tenure_days_safe
    else:
        df["InactivityRatio"] = np.nan

    # 3. Promotion Dependency: proportion of orders relying on coupons
    if "CouponUsed" in df.columns and "OrderCount" in df.columns:
        orders_safe = df["OrderCount"].clip(lower=1.0)
        df["PromotionDependency"] = df["CouponUsed"] / orders_safe
    else:
        df["PromotionDependency"] = np.nan

    # 4. Friction Indicator: registered complaint AND low satisfaction
    if "Complain" in df.columns and "SatisfactionScore" in df.columns:
        is_complaint = df["Complain"] == 1
        is_low_sat = df["SatisfactionScore"] <= 2
        df["FrictionIndicator"] = (is_complaint & is_low_sat).astype(float)
    else:
        df["FrictionIndicator"] = 0.0

    return df


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn transformer that cleans categories and engineers
    domain features consistently across training and inference pipelines.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = clean_categories(X)
        X = engineer_domain_features(X)
        return X


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw DataFrame by cleaning categories and generating domain features.
    No global imputation or row deletion is performed here, preventing data leakage.
    """
    df = clean_categories(df)
    df = engineer_domain_features(df)
    return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Separates features X and target y, dropping CustomerID if present."""
    cols_to_drop = [c for c in ["CustomerID", target_col] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col] if target_col in df.columns else None
    return X, y
