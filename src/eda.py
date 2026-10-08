import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_data
from src.build_features import build_features

FIGURES_DIR = BASE_DIR / "reports" / "figures"
REPORTS_DIR = BASE_DIR / "reports"


def run_eda():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    df_raw = load_data()
    print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 1. Target Distribution & Class Imbalance
    churn_counts = df_raw["Churn"].value_counts()
    churn_pct = df_raw["Churn"].value_counts(normalize=True) * 100
    print("\n--- Target Distribution (Churn) ---")
    for val in [0, 1]:
        label = "Retained (0)" if val == 0 else "Churned (1)"
        print(f"  {label}: {churn_counts[val]:,} ({churn_pct[val]:.2f}%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["Retained (0)", "Churned (1)"], y=churn_counts.values, ax=ax, palette=["#2ecc71", "#e74c3c"])
    ax.set_title("Customer Churn Class Distribution (5:1 Imbalance)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Customer Count")
    for i, v in enumerate(churn_counts.values):
        ax.text(i, v + 50, f"{v:,} ({churn_pct.values[i]:.1f}%)", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_imbalance.png", dpi=300)
    plt.close()

    # 2. Missing Values Analysis
    missing = df_raw.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_df = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Pct": (missing / len(df_raw)) * 100
    })
    print("\n--- Missing Values ---")
    print(missing_df.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing_df["Missing_Pct"], y=missing_df.index, ax=ax, palette="Blues_r")
    ax.set_title("Missing Value Percentage by Feature", fontsize=13, fontweight="bold")
    ax.set_xlabel("% Missing")
    for i, v in enumerate(missing_df["Missing_Pct"]):
        ax.text(v + 0.1, i, f"{v:.2f}% ({missing_df['Missing_Count'].iloc[i]})", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png", dpi=300)
    plt.close()

    # 3. Categorical Values & Synonym Inconsistencies
    print("\n--- Categorical Feature Value Counts (Raw) ---")
    cat_cols = df_raw.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        print(f"  {col}: {dict(df_raw[col].value_counts())}")

    # 4. Feature Engineering Inspection
    df_feat = build_features(df_raw)
    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    if "CustomerID" in num_cols:
        num_cols.remove("CustomerID")

    # 5. Correlation with Churn
    corr_churn = df_feat[num_cols].corr()["Churn"].sort_values()
    print("\n--- Correlation with Churn ---")
    print(corr_churn.round(4).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=corr_churn.values, y=corr_churn.index, ax=ax, palette="coolwarm")
    ax.set_title("Feature Correlations with Target (Churn)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=300)
    plt.close()

    # 6. Behavioral Driver: Tenure vs Churn
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df_feat, x="Tenure", hue="Churn", bins=30, kde=True, ax=ax, palette=["#2ecc71", "#e74c3c"])
    ax.set_title("Tenure Distribution by Churn Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (Months)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tenure_vs_churn.png", dpi=300)
    plt.close()

    # 7. Domain Feature: Friction Indicator vs Churn
    friction_churn = df_feat.groupby("FrictionIndicator")["Churn"].agg(["count", "mean"])
    print("\n--- Friction Indicator Impact ---")
    print(friction_churn)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["No High Friction (0)", "High Friction (1)"], y=friction_churn["mean"] * 100, ax=ax, palette="Reds")
    ax.set_title("Churn Rate by Friction Indicator (Complain=1 & Sat<=2)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Empirical Churn Rate (%)")
    for i, v in enumerate(friction_churn["mean"] * 100):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "friction_vs_churn.png", dpi=300)
    plt.close()

    # Write summary report
    summary_md = f"""# Exploratory Data Analysis Summary

## 1. Dataset Overview
- **Total Records:** {len(df_raw):,} customers
- **Total Raw Columns:** {df_raw.shape[1]}
- **Target Variable:** `Churn` (Binary: 0 = Retained, 1 = Churned)
- **Class Balance:** Retained: {churn_counts[0]:,} ({churn_pct[0]:.2f}%), Churned: {churn_counts[1]:,} ({churn_pct[1]:.2f}%) (~5:1 class imbalance).

## 2. Missing Value Analysis
Missing values are present in 7 numeric features, ranging between 4.46% and 5.45%:
- `DaySinceLastOrder`: 307 missing (5.45%)
- `OrderAmountHikeFromlastYear`: 265 missing (4.71%)
- `Tenure`: 264 missing (4.69%)
- `OrderCount`: 258 missing (4.58%)
- `CouponUsed`: 256 missing (4.55%)
- `HourSpendOnApp`: 255 missing (4.53%)
- `WarehouseToHome`: 251 missing (4.46%)

**Implication for Pipeline:** A median imputer (`SimpleImputer(strategy='median')`) must be integrated into the Scikit-Learn pipeline to handle missing values consistently at training and inference time without data leakage.

## 3. Categorical Anomalies Identified & Addressed
Redundant synonyms identified:
- `PreferredLoginDevice`: 'Mobile Phone' vs 'Phone' (mapped to 'Phone')
- `PreferredPaymentMode`: 'CC' vs 'Credit Card', 'COD' vs 'Cash on Delivery' (mapped to standard forms)
- `PreferedOrderCat`: 'Mobile Phone' vs 'Mobile' (mapped to 'Mobile')

## 4. Key Behavioral Drivers of Churn
1. **Tenure (r = {corr_churn.get('Tenure', 0):.4f}):** Strongest negative predictor. New customers (< 2 months) experience the highest churn rate; customers with > 12 months show significantly lower attrition.
2. **Complain & Friction Indicator (r = {corr_churn.get('FrictionIndicator', 0):.4f}):** Customers with recorded complaints and low satisfaction have a churn rate of {friction_churn['mean'].iloc[-1]*100:.1f}%, far exceeding the 16.8% baseline.
3. **CashbackAmount (r = {corr_churn.get('CashbackAmount', 0):.4f}):** Higher cashback is strongly correlated with customer retention.
4. **Order Velocity & Inactivity:** Short tenure combined with low order velocity signals acute early drop-off risk.

Generated figures are saved in `reports/figures/`.
"""
    (REPORTS_DIR / "eda_summary.md").write_text(summary_md, encoding="utf-8")
    print("\nEDA completed! Figures saved to reports/figures/ and summary written to reports/eda_summary.md")


if __name__ == "__main__":
    run_eda()
