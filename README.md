# E-Commerce Customer Churn Prediction

**Live App:** [Churn-Prediction.app](https://churn-prediction-app-ta.vercel.app/)

## 1. The Problem & The Solution

E-commerce businesses usually find out a customer has churned only after they've already stopped ordering, too late to do anything about it. This project predicts churn *before* it happens: given a customer's order history, engagement, and complaint record, the model outputs a churn probability.

That probability is mapped into three tiers, so the business knows exactly who to act on:

| Tier | Probability | Action |
|---|---|---|
| High Risk | ≥ 0.60 | Call, retention offer, shipping fee waiver |
| Medium Risk | 0.30 to 0.60 | Push notification, product recommendations |
| Low Risk | < 0.30 | No action, save the budget |

This way, retention spend goes to customers actually at risk, instead of being wasted on everyone.

---

## 2. How the Data Was Handled

Starting from 5,630 customer records:

1. Dropped `CustomerID` (just an identifier, not useful for prediction).
2. Filled missing values (~1,856 across the dataset) with the column median.
3. Removed 558 duplicate customer rows, so the same customer can't end up in both training and testing.
4. Encoded categorical fields (device, payment method, gender, etc.) automatically as part of the model itself, so the same encoding is always used at prediction time.

Result: 5,072 clean rows used for training.

---

## 3. Why These Models

Three models were trained and compared on the same data split:

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.882 | 0.621 | 0.884 |
| XGBoost | 0.958 | 0.865 | 0.979 |
| **Random Forest (chosen)** | **0.965** | **0.884** | **0.985** |

Random Forest was picked because it had the best balance of catching real churners while not raising too many false alarms (F1-Score), not just the highest accuracy, accuracy alone is misleading here since most customers don't churn.

---

## 4. Output Quality

- **ROC-AUC of 0.985**, the model separates churners from non-churners almost perfectly. This was double-checked for leakage (no ID or "cheating" columns feed the model), the top signal is `Tenure`, which makes real business sense (new customers churn more).
- **F1-Score of 0.884**, confirms the model is also good at catching the customers who actually churn, not just the easy majority.
- Numbers were reproduced independently and matched exactly, confirming they're stable.

---

## 5. How the Model Was Trained

```
python src/train_model.py
```

This one command runs the full pipeline:
1. Loads and cleans the data (`src/data_loader.py` + `src/build_features.py`).
2. Splits it 80/20 into train/test.
3. Trains all three models (Logistic Regression, XGBoost, Random Forest).
4. Automatically picks the best one by ROC-AUC.
5. Saves the trained model, its feature list, and its metrics into `models/`.

To retrain on new data, just replace `data/E_Commerce_Dataset.csv` and re-run the same command.

---

## 6. How to Use It

### Live App
 **https://churn-prediction-app-ta.vercel.app/**

No installation needed, open the link and start predicting. The app has two modes:

**Single Customer Form**
Fill in one customer's profile, organized into three sections:
- **Demographics**: gender, marital status, city tier, number of addresses, distance from warehouse.
- **Preferences**: login device, payment method, preferred order category, app usage hours, devices registered.
- **Engagement**: tenure, satisfaction score, order count, days since last order, order amount change, coupons used, cashback amount, recent complaint.

Click **Predict Customer Churn** to instantly get that customer's churn probability and risk tier (Low / Medium / High).

**Batch CSV Prediction**
Switch to this tab to upload a CSV file with many customers at once (e.g. your full active customer list) and get churn predictions for all of them in one go, useful for a weekly or monthly retention review instead of checking customers one by one.

**Run the API yourself:**
```
pip install -r requirements.txt
python src/server.py
```

- `POST /predict`: send one customer's data, get back their churn probability and risk tier.
- `POST /predict-batch`: send a list of customers, get predictions for all of them at once.
- `GET /health`: check the API is running.
- `GET /model/metadata`: see the model's saved performance metrics.