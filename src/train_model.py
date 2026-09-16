import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
 
from data_loader import load_data
from build_features import build_features, split_features_target

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
 
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
 
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
 
        print(f"{name} -> accuracy: {acc:.4f}, f1: {f1:.4f}, roc_auc: {auc:.4f}")
        results[name] = {"model": model, "auc": auc}
 
    return results

def get_best_model(results):
    best_name = max(results, key=lambda name: results[name]["auc"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}")
    return best_model

def main():
    df = load_data()
    df = build_features(df)
    X, y = split_features_target(df)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_model = get_best_model(results)
 
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

main()

