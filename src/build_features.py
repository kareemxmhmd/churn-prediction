import pandas as pd
from src.data_loader import load_data

TARGET_COL = "Churn"

def build_features(df):
    df = df.drop(columns=["CustomerID"])

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    missing_before = df[numeric_cols].isna().sum().sum()
    print(f"Missing values before fillna: {missing_before}")
 
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
 
    missing_after = df[numeric_cols].isna().sum().sum()
    print(f"Missing values after fillna: {missing_after}")

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows ({before} -> {after})")

    return df

def split_features_target(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def main():
    df = load_data()
    df = build_features(df)
    X, y = split_features_target(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(X.columns.tolist())

if __name__ == "__main__":
    main()
