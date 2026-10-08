import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.build_features import (
    CATEGORICAL_FEATURES,
    DomainFeatureEngineer,
    NUMERIC_FEATURES,
    split_features_target,
)
from src.data_loader import load_data
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.pkl"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def get_preprocessor():
    """Builds a leakage-free Scikit-Learn ColumnTransformer for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def create_pipeline(classifier):
    """Combines category cleaning, domain feature engineering, preprocessing, and classifier."""
    return Pipeline(
        steps=[
            ("feature_engineer", DomainFeatureEngineer()),
            ("preprocessor", get_preprocessor()),
            ("classifier", classifier),
        ]
    )


def evaluate_cross_validation(X_train, y_train):
    """Performs 5-Fold Stratified Cross-Validation on training data across candidate models."""
    print("\n" + "=" * 60)
    print("5-FOLD STRATIFIED CROSS-VALIDATION (80% TRAINING SET)")
    print("=" * 60)

    # Class weight ratio: 4682 non-churners / 948 churners = ~4.94
    candidate_models = {
        "Dummy (Baseline)": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=4.94,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    cv_results = {}
    for name, clf in candidate_models.items():
        pipe = create_pipeline(clf)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "pr_auc_mean": float(scores["test_pr_auc"].mean()),
            "pr_auc_std": float(scores["test_pr_auc"].std()),
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "f1_mean": float(scores["test_f1"].mean()),
            "f1_std": float(scores["test_f1"].std()),
            "precision_mean": float(scores["test_precision"].mean()),
            "precision_std": float(scores["test_precision"].std()),
            "recall_mean": float(scores["test_recall"].mean()),
            "recall_std": float(scores["test_recall"].std()),
            "raw_clf": clf,
        }
        print(f"\n{name}:")
        print(f"  PR-AUC:    {cv_results[name]['pr_auc_mean']:.4f} (+/- {cv_results[name]['pr_auc_std']:.4f})")
        print(f"  ROC-AUC:   {cv_results[name]['roc_auc_mean']:.4f} (+/- {cv_results[name]['roc_auc_std']:.4f})")
        print(f"  F1-Score:  {cv_results[name]['f1_mean']:.4f} (+/- {cv_results[name]['f1_std']:.4f})")
        print(f"  Precision: {cv_results[name]['precision_mean']:.4f} (+/- {cv_results[name]['precision_std']:.4f})")
        print(f"  Recall:    {cv_results[name]['recall_mean']:.4f} (+/- {cv_results[name]['recall_std']:.4f})")

    # Select best model based on PR-AUC (key metric for imbalanced classification)
    best_model_name = max(
        [m for m in cv_results if m != "Dummy (Baseline)"],
        key=lambda m: cv_results[m]["pr_auc_mean"],
    )
    print(f"\nBest Model by 5-Fold CV PR-AUC: {best_model_name}")
    return best_model_name, cv_results


def perform_error_analysis(y_test, y_pred, y_prob, X_test):
    """Analyzes False Positives and False Negatives across key customer cohorts."""
    analysis_df = X_test.copy()
    analysis_df["Actual"] = y_test.values
    analysis_df["Predicted"] = y_pred
    analysis_df["Probability"] = y_prob

    analysis_df["Error_Type"] = "Correct"
    analysis_df.loc[(analysis_df["Actual"] == 0) & (analysis_df["Predicted"] == 1), "Error_Type"] = "False Positive"
    analysis_df.loc[(analysis_df["Actual"] == 1) & (analysis_df["Predicted"] == 0), "Error_Type"] = "False Negative"

    fp_count = (analysis_df["Error_Type"] == "False Positive").sum()
    fn_count = (analysis_df["Error_Type"] == "False Negative").sum()

    # Tenure cohort analysis
    def get_tenure_bracket(t):
        if pd.isna(t):
            return "Missing"
        if t <= 3:
            return "0-3 Months"
        if t <= 12:
            return "4-12 Months"
        if t <= 24:
            return "13-24 Months"
        return "25+ Months"

    analysis_df["Tenure_Bracket"] = analysis_df["Tenure"].apply(get_tenure_bracket)
    cohort_breakdown = (
        analysis_df.groupby("Tenure_Bracket")["Error_Type"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict()
    )

    error_summary = {
        "total_test_samples": len(y_test),
        "false_positives": int(fp_count),
        "false_negatives": int(fn_count),
        "cohort_breakdown": cohort_breakdown,
    }

    with open(REPORTS_DIR / "error_analysis.json", "w") as f:
        json.dump(error_summary, f, indent=2)

    print("\n--- Error Analysis Summary ---")
    print(f"False Positives: {fp_count} (Retention offer wasted on loyal customer)")
    print(f"False Negatives: {fn_count} (At-risk churner missed)")
    return error_summary


def explain_model_shap(best_pipeline, X_train, X_test):
    """Generates SHAP feature importance plot and exports feature importances."""
    print("\nGenerating SHAP Summary & Feature Importances...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Transform raw data through pipeline steps prior to classifier
    engineer = best_pipeline.named_steps["feature_engineer"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    clf = best_pipeline.named_steps["classifier"]

    X_train_eng = engineer.transform(X_train)
    X_test_eng = engineer.transform(X_test)

    X_train_trans = preprocessor.transform(X_train_eng)
    X_test_trans = preprocessor.transform(X_test_eng)

    feature_names = preprocessor.get_feature_names_out()

    # Tree SHAP Explainer
    try:
        sample_size = min(300, X_test_trans.shape[0])
        sample_indices = np.random.choice(X_test_trans.shape[0], sample_size, replace=False)
        X_sample = X_test_trans[sample_indices]

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification output shape
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals_class1 = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_vals_class1 = shap_values[:, :, 1]
        else:
            shap_vals_class1 = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals_class1, X_sample, feature_names=feature_names, show=False, max_display=15)
        plt.title("SHAP Feature Importance (Impact on Churn Prediction)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Notice during SHAP plot generation: {e}")

    # Global feature importances
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
        imp_dict = importances.head(20).to_dict()
        with open(REPORTS_DIR / "feature_importance.json", "w") as f:
            json.dump(imp_dict, f, indent=2)
        print("Top 5 Most Important Features:")
        for feat, val in list(imp_dict.items())[:5]:
            print(f"  {feat}: {val:.4f}")


def main():
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - PRODUCTION TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    df = load_data()
    X, y = split_features_target(df)
    raw_feature_cols = X.columns.tolist()

    # 2. Pristine 80/20 Holdout Split (Held out untouched until final evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training Samples: {len(X_train)} | Holdout Test Samples: {len(X_test)}")

    # 3. 5-Fold Stratified Cross-Validation on 80% Training Set
    best_model_name, cv_results = evaluate_cross_validation(X_train, y_train)
    best_raw_clf = cv_results[best_model_name]["raw_clf"]

    # 4. Construct Best Pipeline
    best_pipe = create_pipeline(best_raw_clf)
    best_pipe.fit(X_train, y_train)

    # 5. Model Probability Calibration (using 5-fold CV on train data)
    print(f"\nCalibrating {best_model_name} probabilities with CalibratedClassifierCV (Sigmoid)...")
    calibrated_model = CalibratedClassifierCV(estimator=best_pipe, cv=5, method="sigmoid")
    calibrated_model.fit(X_train, y_train)

    # 6. Final Evaluation on Pristine Holdout Test Set (Touched ONLY Once)
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON PRISTINE HOLDOUT TEST SET (20%)")
    print("=" * 60)

    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:         {acc:.4f}")
    print(f"Precision:        {prec:.4f}")
    print(f"Recall:           {rec:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"PR-AUC:           {pr_auc:.4f}")
    print(f"Brier Score:      {brier:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # Risk Tier Breakdown on Test Set
    low_risk = (y_prob < 0.3)
    med_risk = (y_prob >= 0.3) & (y_prob < 0.6)
    high_risk = (y_prob >= 0.6)

    def get_tier_stats(mask):
        total = int(mask.sum())
        churners = int(y_test[mask].sum()) if total > 0 else 0
        rate = (churners / total) if total > 0 else 0.0
        return {"total_customers": total, "actual_churners": churners, "empirical_churn_rate": round(rate, 4)}

    risk_tier_stats = {
        "Low_Risk (<30%)": get_tier_stats(low_risk),
        "Medium_Risk (30-60%)": get_tier_stats(med_risk),
        "High_Risk (>=60%)": get_tier_stats(high_risk),
    }
    print("\nEmpirical Risk Tier Validation on Holdout:")
    for tier, s in risk_tier_stats.items():
        print(f"  {tier}: {s['total_customers']} customers, {s['actual_churners']} churners ({s['empirical_churn_rate']*100:.1f}%)")

    # 7. Error Analysis
    error_summary = perform_error_analysis(y_test, y_pred, y_prob, X_test)

    # 8. Model Interpretability (SHAP & Feature Importance)
    explain_model_shap(best_pipe, X_train, X_test)

    # 9. Model Persistence
    joblib.dump(calibrated_model, MODEL_PATH)
    joblib.dump(raw_feature_cols, COLUMNS_PATH)

    metrics_payload = {
        "model_name": best_model_name,
        "is_calibrated": True,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "confusion_matrix": cm.tolist(),
        "risk_tier_stats": risk_tier_stats,
        "cv_results": {k: {m: v for m, v in val.items() if m != "raw_clf"} for k, val in cv_results.items()},
    }
    joblib.dump(metrics_payload, METRICS_PATH)

    print(f"\nArtifacts saved successfully:")
    print(f"  Calibrated Model: {MODEL_PATH}")
    print(f"  Feature Columns:  {COLUMNS_PATH}")
    print(f"  Metrics & Report: {METRICS_PATH}")


if __name__ == "__main__":
    main()
