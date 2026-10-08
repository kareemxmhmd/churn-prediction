# E-Commerce Customer Churn Prediction & Retention Microservice

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/pytest-16%20passed-brightgreen.svg)](https://docs.pytest.org/)

An end-to-end, production-ready Machine Learning system that predicts e-commerce customer churn in advance based on behavioral, transaction, and engagement patterns. The service translates calibrated probabilities into actionable retention risk tiers and serves real-time and batch predictions through a high-performance FastAPI microservice.

---

## 1. Business Problem & Value

In e-commerce, acquiring new customers costs 5x to 7x more than retaining existing ones. Most platforms discover that a customer has churned only after they stop ordering, when win-back marketing is costly and low-yield. 

This project shifts customer retention from **reactive damage control** to **proactive prevention** by predicting churn risk well before departure.

### Actionable Risk Tiers
Predictions are translated into three distinct business action tiers:
- **High Risk ($\ge 60\%$ probability):** Immediate retention interventions (direct account manager outreach, personalized loyalty credits, VIP support).
- **Medium Risk ($30\%\text{--}60\%$ probability):** Proactive engagement (automated push notifications, targeted category discounts, tailored product recommendations).
- **Low Risk ($< 30\%$ probability):** Standard organic customer journey, preventing unnecessary discount spend on loyal customers.

---

## 2. Dataset & Behavioral Signals

The model is trained on customer transactional and behavioral records (`data/E_Commerce_Dataset.csv`) covering 5,630 customer accounts across 20 features:
- **Tenure & Loyalty:** Account age (`Tenure`), total orders (`OrderCount`), days elapsed since last purchase (`DaySinceLastOrder`).
- **Platform Engagement:** Hours spent on app (`HourSpendOnApp`), registered devices (`NumberOfDeviceRegistered`), login preferences.
- **Spending & Value:** Spend increase year-over-year (`OrderAmountHikeFromlastYear`), cashback received (`CashbackAmount`), coupon usage (`CouponUsed`).
- **Customer Friction & Satisfaction:** Survey satisfaction rating (`SatisfactionScore`), recorded complaints (`Complain`), distance from fulfillment center (`WarehouseToHome`).
- **Demographics & Account Preferences:** Preferred payment mode, preferred product category, marital status, city tier.

**Target Variable:** `Churn` (Binary: `0` = Retained, `1` = Churned).  
Class distribution: 83.16% retained, 16.84% churned (~5:1 class imbalance).

---

## 3. Exploratory Data Analysis & Domain Features

Reproducible EDA (`python src/eda.py` or `notebooks/eda.ipynb`) revealed critical behavioral patterns:
1. **Tenure is the strongest negative churn indicator ($r = -0.35$):** Customers in their first 2 months experience peak defection risk. Retention past month 12 drops churn significantly.
2. **Customer Friction Amplifies Churn ($r = +0.25$):** Filing a complaint while recording a low satisfaction score ($\le 2$) raises empirical churn probability to over 22%.
3. **Cashback Drives Loyalty ($r = -0.15$):** Customers with above-average cashback rewards exhibit substantially higher retention.

### Domain-Engineered Behavioral Features
To capture non-linear customer dynamics without data leakage:
- **Order Velocity:** $\text{OrderCount} / \max(\text{Tenure}, 1)$ — Measures ordering density per month of account lifespan ($r = +0.41$).
- **Inactivity Ratio:** $\text{DaySinceLastOrder} / \max(\text{Tenure} \times 30, 1)$ — Measures recency of inactivity relative to account lifetime ($r = +0.23$).
- **Promotion Dependency:** $\text{CouponUsed} / \max(\text{OrderCount}, 1)$ — Ratio of discounted orders.
- **Friction Indicator:** Boolean interaction of $\text{Complain} == 1$ and $\text{SatisfactionScore} \le 2$.

---

## 4. Machine Learning Architecture

```
Raw Customer Features (18)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ DomainFeatureEngineer                                  │
│ - Category synonym standardization                     │
│ - OrderVelocity, InactivityRatio, PromotionDependency   │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ ColumnTransformer (Scikit-Learn)                       │
│ - Numeric (17 cols): SimpleImputer(median) + Scaler    │
│ - Categorical (5 cols): Imputer(mode) + OneHotEncoder  │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ Tuned XGBoost Classifier (scale_pos_weight=4.94)       │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ CalibratedClassifierCV (Sigmoid / Platt Scaling, 5-CV) │
└────────────────────────────────────────────────────────┘
          │
          ▼
Calibrated Churn Probability & Risk Tier
```

### Validation Methodology
- **Strict 80/20 Holdout Split:** A pristine 20% test set ($n = 1,126$) was isolated at the very start and untouched until final reporting.
- **5-Fold Stratified Cross-Validation:** Candidate models were tuned and compared across 5 stratified folds on the 80% training set ($n = 4,504$):

| Candidate Model | CV PR-AUC | CV ROC-AUC | CV F1-Score | CV Precision | CV Recall |
|---|---:|---:|---:|---:|---:|
| **Dummy (Majority Baseline)** | 0.1683 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression (Balanced)** | 0.7151 | 0.8986 | 0.6050 | 0.4800 | 0.8206 |
| **Random Forest (Tuned)** | 0.8514 | 0.9589 | 0.7730 | 0.7559 | 0.7916 |
| **XGBoost (Tuned & Weighted)** | **0.8945** | **0.9670** | **0.8249** | **0.7818** | **0.8734** |

---

## 5. Holdout Evaluation Results

Final evaluation on the unseen holdout test set ($n = 1,126$):

| Metric | Score | Interpretation |
|---|---:|---|
| **PR-AUC (Average Precision)** | **0.9356** | Primary metric for imbalanced data; demonstrates high precision across all recall thresholds. |
| **ROC-AUC** | **0.9880** | Strong ranking discrimination between churners and retained customers. |
| **F1-Score** | **0.8756** | Optimal harmonic mean between precision and recall. |
| **Recall (Sensitivity)** | **88.95%** | Successfully flags ~89 out of every 100 true churners. |
| **Precision** | **86.22%** | High intervention accuracy; minimal wasted retention spend on false alarms. |
| **Accuracy** | **95.74%** | Overall correct classification rate. |
| **Brier Score** | **0.0332** | Indicates probability calibration error close to zero. |

### Confusion Matrix
```
                 Predicted Retained    Predicted Churned
Actual Retained         909 (TN)             27 (FP)
Actual Churned           21 (FN)            169 (TP)
```

### Empirical Risk Tier Validation
Predicted probability thresholds validate against real churn outcomes on the holdout:
- **Low Risk ($< 30\%$ prob):** 896 customers $\rightarrow$ **0.8%** empirical churn rate.
- **Medium Risk ($30\%\text{--}60\%$ prob):** 61 customers $\rightarrow$ **52.5%** empirical churn rate.
- **High Risk ($\ge 60\%$ prob):** 169 customers $\rightarrow$ **89.3%** empirical churn rate.

---

## 6. Model Interpretability & SHAP Analysis

Global feature importances from tree-based SHAP analysis (`reports/figures/shap_summary.png`):
1. **Tenure:** Primary protective factor. High tenure sharply decreases churn likelihood.
2. **Complain:** Primary risk driver. Recorded complaints shift probability directly into High Risk.
3. **PreferedOrderCat (Mobile / Laptop):** Product categories with high replacement cycles show distinct churn hazard curves.
4. **OrderVelocity & InactivityRatio:** Customers with declining order velocity relative to tenure indicate disengagement.

---

## 7. Project Structure

```
churn-prediction/
├── api/
│   └── server.py              # FastAPI application with input validation and batch scoring
├── data/
│   └── E_Commerce_Dataset.csv # Raw customer transactional and survey dataset
├── models/
│   ├── churn_model.pkl        # Calibrated end-to-end Scikit-Learn Pipeline
│   ├── feature_columns.pkl    # Serialized input feature schema
│   └── metrics.pkl            # Holdout metrics, confusion matrix & CV results
├── notebooks/
│   └── eda.ipynb              # Interactive Exploratory Data Analysis notebook
├── reports/
│   ├── figures/               # Generated figures (SHAP, correlation, distributions)
│   ├── eda_summary.md         # Statistical EDA summary
│   ├── error_analysis.json    # Cohort breakdown of False Positives & Negatives
│   └── feature_importance.json# Top feature importance rankings
├── src/
│   ├── __init__.py
│   ├── build_features.py      # Category cleaning, domain feature engineering & transformers
│   ├── data_loader.py         # Data loading utility
│   ├── eda.py                 # Reproducible EDA script
│   └── train_model.py         # Leakage-free training, CV, calibration & evaluation pipeline
├── tests/
│   ├── test_api.py            # API integration tests & validation checks
│   └── test_pipeline.py       # Preprocessing, imputer & inference robustness tests
├── .dockerignore
├── .gitignore
├── docker-compose.yml         # Container orchestration configuration
├── Dockerfile                 # Production Docker container specification
├── Procfile                   # Cloud PaaS deployment entrypoint
├── README.md                  # Comprehensive documentation
└── requirements.txt           # Pinned production dependencies
```

---

## 8. Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### Setup Virtual Environment
```bash
# Clone the repository
git clone <repo-url>
cd churn-prediction

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 9. Running the Pipeline & Tests

### 1. Run Exploratory Data Analysis
Generates all visualization figures in `reports/figures/`:
```bash
python src/eda.py
```

### 2. Run Model Training & Evaluation
Executes cross-validation, tunes hyperparameters, calibrates probabilities, evaluates holdout set, conducts error analysis, and serializes artifacts:
```bash
python src/train_model.py
```

### 3. Run Automated Tests
Executes the comprehensive test suite (16 tests across pipeline and API):
```bash
pytest tests/ -v
```

---

## 10. Running the Production API

### Option A: Local Uvicorn Server
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Option B: Docker Container
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Option C: Docker Compose
```bash
docker compose up --build
```

---

## 11. API Reference & Examples

### Health Check
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "XGBoost"
}
```

### Model Metadata
```bash
curl -X GET http://localhost:8000/model/metadata
```
**Response:** Returns holdout metrics (`pr_auc`, `roc_auc`, `f1`, `precision`, `recall`, `brier_score`, `confusion_matrix`, etc.).

---

### Single Customer Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure": 2.0,
    "PreferredLoginDevice": "Phone",
    "CityTier": 1,
    "WarehouseToHome": 15.0,
    "PreferredPaymentMode": "Credit Card",
    "Gender": "Male",
    "HourSpendOnApp": 3.0,
    "NumberOfDeviceRegistered": 4,
    "PreferedOrderCat": "Mobile",
    "SatisfactionScore": 2,
    "MaritalStatus": "Single",
    "NumberOfAddress": 2,
    "Complain": 1,
    "OrderAmountHikeFromlastYear": 12.0,
    "CouponUsed": 1.0,
    "OrderCount": 2.0,
    "DaySinceLastOrder": 7.0,
    "CashbackAmount": 140.0
  }'
```
**Response:**
```json
{
  "churn_probability": 0.8842,
  "churn_prediction": true,
  "risk_tier": "High"
}
```

---

### Batch Prediction (`POST /predict-batch`)
```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "Tenure": 24.0,
      "SatisfactionScore": 5,
      "Complain": 0,
      "CashbackAmount": 210.0
    },
    {
      "Tenure": 1.0,
      "SatisfactionScore": 1,
      "Complain": 1,
      "CashbackAmount": 120.0
    }
  ]'
```
**Response:**
```json
{
  "total": 2,
  "results": [
    {
      "churn_probability": 0.0112,
      "churn_prediction": false,
      "risk_tier": "Low"
    },
    {
      "churn_probability": 0.9148,
      "churn_prediction": true,
      "risk_tier": "High"
    }
  ]
}
```

---
