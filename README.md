# Telco Customer Churn Prediction

> **Business-focused ML case study**
> Predicting customer churn and translating model outputs
> into actionable retention strategies via threshold tuning
> and model interpretability.


End-to-end applied machine learning project for customer churn prediction
using tabular data.
The project covers the full ML lifecycle: EDA, feature engineering,
baseline models, CatBoost, interpretability (SHAP),
and business-oriented threshold tuning.

---

## 📌 Project Overview

Customer churn prediction is a classic business problem where the goal
is to identify customers who are likely to leave a service in the near future.

In this project we:

- analyze customer behavior using exploratory data analysis (EDA),
- engineer business-driven features,
- train and compare baseline models,
- build a stronger CatBoost model,
- interpret model predictions,
- translate model outputs into actionable business scenarios.

The focus is on **applied ML**:
building models that can be realistically deployed
and used for decision-making, not academic experimentation.


---

## 🎯 Business Goal

Predict the probability of customer churn and provide
a decision framework that allows the business to:

- prioritize customers for retention campaigns,
- choose different operating modes depending on budget and cost of contact,
- understand the key drivers of churn.

---

## 📊 Dataset

- Source: Telco customer churn dataset (IBM / Kaggle)
- Size: ~7,000 customers
- Target: `churn_value` (binary)

### Feature groups

- customer demographics,
- contract and tenure information,
- service usage and add-ons,
- billing and payment behavior.

Target leakage features (e.g. churn score, churn reason, CLTV)
were explicitly removed before modeling.

---

## 🧱 Project Structure

```text
ml_churn_telco/
├─ data/
│  ├─ raw/
│  │  └─ telco_churn_raw.csv
│  └─ processed/
│     ├─ telco_churn_features_full.csv
│     ├─ train.csv
│     ├─ valid.csv
│     └─ test.csv
├─ notebooks/
│  ├─ 01_eda_telco_churn.ipynb
│  ├─ 02_feature_engineering.ipynb
│  ├─ 03_modeling_baselines.ipynb
│  ├─ 04_modeling_catboost_shap.ipynb
│  └─ 05_threshold_tuning_and_business_impact.ipynb
├─ logs/
│  ├─ notebook_04_catboost/
│  └─ notebook_05_thresholds/
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## 🧪 Notebooks Description

**01 — Exploratory Data Analysis**

- Churn distribution and class imbalance
- Analysis of tenure, contracts, pricing, and services
- Initial business hypotheses

**02 — Feature Engineering**

- Creation of business-driven features
- Removal of leakage and identifiers
- Train / validation / test split

**03 — Baseline Modeling**

- Logistic Regression (interpretable baseline)
- Random Forest (non-linear baseline)
- Evaluation using ROC-AUC and PR-AUC

**04 — CatBoost & Interpretability**

- CatBoost model with native categorical features
- Early stopping and validation monitoring
- Feature importance analysis
- SHAP-based interpretability

**05 — Threshold Tuning & Business Impact**

- Precision–recall trade-off analysis
- Threshold selection for different business scenarios
- Aggressive / balanced / conservative retention strategies
- Evaluation of campaign coverage and churn capture

---

## 📈 Model Performance (Test Set)

| Model               | ROC-AUC | PR-AUC |
|---------------------|:-------:|:------:|
| Logistic Regression | ~0.854  | ~0.665 |
| Random Forest       | ~0.853  | ~0.662 |
| CatBoost            | ~0.854  | ~0.678 |

CatBoost slightly improves PR-AUC, which is especially important for identifying churners in an imbalanced setting.

Main results and business scenarios are demonstrated in:
`05_threshold_tuning_and_business_impact.ipynb`

---

## 🧠 Key Insights

- Short-term contracts and low tenure are strong churn indicators.
- Customers with fewer add-on services are more likely to churn.
- Billing and payment behavior has a measurable impact on churn risk.
- No single feature dominates: churn is driven by a combination of contract, usage, and pricing factors.

---

## 💼 Business Scenarios (Threshold Tuning)

Three operational modes were evaluated:

- **Aggressive**
  High recall, large campaign size, suitable for low-cost communication.

- **Balanced**
  Best F1-score, reasonable coverage and precision.

- **Conservative**
  High precision, small campaign size, suitable for expensive retention actions.

The model provides ranked churn probabilities, allowing the business to adapt thresholds to budget and strategy.

---

## ⚙️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- CatBoost
- SHAP
- matplotlib

---

## ✅ Key Takeaway

This project shows how churn prediction can be used
not only as a classification task,
but as a flexible decision-support system
adaptable to different business budgets and strategies.

---

## 🚀 Conclusion

This project demonstrates how a churn prediction model can be developed and transformed into a practical decision-support tool.
The emphasis is on:

- realistic evaluation,
- interpretability,
- business applicability.

---

## 👤 Author

Project developed as part of a machine learning portfolio.
If you have questions or suggestions — feel free to reach out.

**Albert Nubaryan**
📧 Email: [alnubwork@gmail.com](mailto:alnubwork@gmail.com)
