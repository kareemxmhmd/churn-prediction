# Exploratory Data Analysis Summary

## 1. Dataset Overview
- **Total Records:** 5,630 customers
- **Total Raw Columns:** 20
- **Target Variable:** `Churn` (Binary: 0 = Retained, 1 = Churned)
- **Class Balance:** Retained: 4,682 (83.16%), Churned: 948 (16.84%) (~5:1 class imbalance).

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
1. **Tenure (r = -0.3494):** Strongest negative predictor. New customers (< 2 months) experience the highest churn rate; customers with > 12 months show significantly lower attrition.
2. **Complain & Friction Indicator (r = 0.0462):** Customers with recorded complaints and low satisfaction have a churn rate of 22.1%, far exceeding the 16.8% baseline.
3. **CashbackAmount (r = -0.1541):** Higher cashback is strongly correlated with customer retention.
4. **Order Velocity & Inactivity:** Short tenure combined with low order velocity signals acute early drop-off risk.

Generated figures are saved in `reports/figures/`.
