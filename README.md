# E-Commerce Customer Churn Prediction

## Business Problem
In e-commerce, acquiring new customers costs significantly more than retaining existing ones. Most companies only discover that a customer has churned after they have stopped ordering, when win-back efforts are expensive and often unsuccessful. Identifying churn risk early is critical to protecting recurring revenue and customer lifetime value.

## Solution
This project provides a machine learning solution that predicts customer churn risk in advance based on customer behavioral and transaction patterns.

Predictions are translated into three actionable risk tiers:
- **High Risk (≥ 60% probability):** Immediate retention actions (e.g., direct outreach, special retention offers, loyalty incentives).
- **Medium Risk (30%–60% probability):** Proactive engagement (e.g., targeted notifications, personalized recommendations).
- **Low Risk (< 30% probability):** Standard customer experience, avoiding unnecessary discount spend.

This allows retention teams to intervene proactively and focus their budget where it delivers the highest return.

## Data
The model is trained on e-commerce customer transaction and behavioral data (`E_Commerce_Dataset.csv`) covering over 5,000 customer records:
- **Tenure & Loyalty:** Account lifespan, order count, and days since last purchase.
- **Engagement & Spending:** App usage hours, year-over-year spend changes, cashback earned, and coupon usage.
- **Customer Experience & Demographics:** Customer satisfaction scores, recorded complaints, distance from warehouse, and delivery/payment preferences.

These signals capture shifting customer habits and emerging dissatisfaction well before a customer leaves.

## Business Value
- **Identifying Risks Early:** Flags at-risk customers before they disengage, turning retention from reactive damage control into proactive prevention.
- **Reducing Retention Costs:** Segments customers by risk tier so marketing budgets and discounts are allocated only where needed, avoiding wasted spend on already loyal buyers.
- **Improving Decisions:** Gives marketing and customer success teams clear, objective risk scores to prioritize weekly outreach and retention campaigns.
- **Protecting Revenue & LTV:** Helps curb avoidable customer defection, preserving recurring order value and long-term customer lifetime value.
- **Optimizing Operations:** Enables automated scoring for individual customer inquiries as well as bulk list reviews across the entire customer base.

## Machine Learning Approach
- **Problem Type:** Supervised Binary Classification (predicting whether an active customer will churn).
- **Model:** Random Forest Classifier (selected over Logistic Regression and XGBoost for its superior balance in catching true churners while minimizing false alarms).

## Result
- **Performance:** Achieved an **ROC-AUC of 0.985**, an **F1-Score of 0.884**, and **96.5% Accuracy** on unseen test data.
- **Business Impact:** The model reliably distinguishes churners from non-churners with high precision and recall, allowing retention teams to act decisively without overburdening staff with false positives.