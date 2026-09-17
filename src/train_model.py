import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.build_features import build_features, split_features_target

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.pkl"

def get_preprocessor(X):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    return ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    results = {}

    for name, clf in MODELS.items():
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        print(f"{name} -> accuracy: {acc:.4f}, f1: {f1:.4f}, roc_auc: {auc:.4f}")
        results[name] = {"model": model, "acc": acc, "f1": f1, "auc": auc}

    return results

def get_best_model(results):
    best_name = max(results, key=lambda name: results[name]["auc"])
    best = results[best_name]
    print(f"\nBest model: {best_name}")
    return best_name, best

def main():
    df = load_data()
    df = build_features(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor(X)
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    best_name, best = get_best_model(results)

    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(X.columns.tolist(), COLUMNS_PATH)
    joblib.dump(
        {"model_name": best_name, "accuracy": best["acc"], "f1": best["f1"], "roc_auc": best["auc"]},
        METRICS_PATH,
    )

    print(f"Saved best model to {MODEL_PATH}")
    print(f"Saved feature columns to {COLUMNS_PATH}")

if __name__ == "__main__":
    main()
