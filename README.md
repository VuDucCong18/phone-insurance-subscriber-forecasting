# Phone Insurance Subscriber Forecasting

Forecasting monthly active phone insurance subscribers (**Closing Subs Monthly**) per device model, market, and colour — using a leakage-free ML pipeline with lag-1 features, walk-forward cross-validation, and a champion/challenger model evaluation framework.

> Full interactive report: **[report.html](report.html)** — open in any browser for charts, model cards, and recommendations.

---

## About Asurion

Asurion is a leading provider of device insurance, warranty, and support services for cell phones, consumer electronics, and home appliances. With over 300 million devices protected globally, accurate subscriber forecasting is critical to operational and financial planning.

**Closing Subs Monthly** is the number of active phone insurance subscribers at the end of each month for a given (device model, colour, market) combination. It is not units sold — it is the live insured base at a point in time.

Forecasting this figure one month ahead drives three operational decisions:

| Decision | Why it needs a forecast |
|---|---|
| Claims reserve sizing | Expected claims scale directly with the active subscriber base |
| Device parts inventory | Warranty repairs require stocked components per model — wrong forecasts cause stockouts or overstock |
| Renewal campaign targeting | Models approaching lifecycle decay need proactive retention to slow subscriber loss |

---

## Dataset

| Attribute | Detail |
|---|---|
| Period | Jan 2019 – Nov 2023 |
| Markets | California, Nevada, Texas |
| Brands | Apple, Samsung, Oppo |
| Example models | iPhone XS Max, Samsung Galaxy S21, Oppo A15 |
| Raw size | 13,381 rows × 25 columns |
| After cleaning | 10,859 rows × 56 features |

---

## Data Leakage & How It Was Fixed

Two columns in the raw data are highly correlated with the target but **cannot be used as concurrent features**:

- **Churn** — subscribers who cancelled in the same month. Only known after the month closes, the same moment as the target.
- **Filed Claims** — warranty claims submitted in the same month. Same problem.
- **IR Rate** (Swap, Replacement, Monthly) — incidence rate calculated as `Filed Claims / Closing Subs Monthly`. Contains the target in its denominator — direct leakage even after lagging.

Using any of these directly means the model is trained on information that would not exist at real forecast time. The fix is **lag-1 features**: shift each value back one month so the model only sees information available before the forecast period begins.

| Feature | Status | Action |
|---|---|---|
| Churn | Concurrent — leaked | Shifted 1 month → `lag1_churn` |
| Filed Claims | Concurrent — leaked | Shifted 1 month → `lag1_filed_claims` |
| Closing Subs Monthly | Target — leaked | Shifted 1 month → `lag1_closing_subs` |
| IR Rate, Claims Swap/Replacement | Concurrent + formula leakage | Dropped entirely |
| Model Age (Days), Size | Known in advance | Retained as-is |

The high correlation of Churn and Claims with the target also partly reflects a **size effect** — both scale with the installed base, so larger subscriber bases naturally produce more of both. This is not a causal relationship.

---

## Data Cleaning

- **Pre-order rows dropped (2,128 rows / 16.3%)** — retailers and carriers pre-register subscriber slots before a device launches, creating rows where `Model Age (Days) < 0`. These produce corrupt lag values (the "prior month" is a pre-launch placeholder). Only ~4.2% of total subscriber volume is lost — the data integrity gain outweighs the row loss.
- **NaN lag rows dropped (91 rows)** — the first observed month of each (country, ModelFamily) group has no prior month to look back at. Dropped rather than zero-filled, as zero falsely implies zero prior subscribers.
- **Horizon filter** — rows limited to model families present in Nov 2023, ensuring all SKUs have a valid forecast base.

---

## Methodology

### Feature set

| Feature | Type | Business meaning |
|---|---|---|
| `lag1_closing_subs` | Lag · numeric | Prior month subscribers — demand is sticky (r = 0.67) |
| `lag1_churn` | Lag · numeric | Prior month cancellations — base shrinking signal (r = 0.46) |
| `lag1_filed_claims` | Lag · numeric | Prior month claims — base activity level (r = 0.46) |
| `Model Age (Days)` | Numeric | Lifecycle stage — fully known before forecast period |
| `Size` | Numeric | Storage capacity (GB) — proxy for price tier |
| `country`, `ModelFamily`, `Colour` | One-hot encoded (drop_first=True) | Market and SKU identity — 51 dummy columns |

One-hot encoding converts categorical columns into binary (1/0) columns — one per unique value minus one (the dropped reference category eliminates multicollinearity).

### Train / Validation / Test Split

Chronological split, no shuffling:

```
Train  60%  →  Jan 2019 – ~Apr 2022   (6,515 rows)   ← model training
Val    15%  →  ~May 2022 – ~Dec 2022  (1,629 rows)   ← hyperparameter tuning only
Test   25%  →  ~Jan 2023 – Nov 2023   (2,715 rows)   ← held-out, final eval only
```

The validation set is reserved exclusively for hyperparameter tuning (alpha for Ridge/Lasso, tree depth for RF/XGBoost). The test set is never touched until final evaluation, ensuring reported metrics are a true unbiased estimate of real-world performance.

### Cross-Validation

**TimeSeriesSplit (5 folds, walk-forward)** is used instead of standard K-fold. K-fold randomly shuffles months across folds, so future months can appear in training — leaking the future into the past. TimeSeriesSplit enforces strict chronological ordering: each fold trains only on data before its validation window, mirroring real production conditions.

`StandardScaler` is fitted on the training fold only and applied to validation/test to prevent scale leakage.

---

## Models

A **champion/challenger** framework compares linear baselines against non-linear tree models.

| Role | Model | Concept |
|---|---|---|
| Baseline | **Ridge** (RidgeCV) | Linear with L2 penalty — shrinks all coefficients toward zero. Interpretable, suitable for stakeholder reporting. |
| Baseline | **Lasso** (LassoCV) | Linear with L1 penalty — drives irrelevant coefficients to exactly zero. Automatic feature selection. |
| Challenger | **Random Forest** | 200 trees averaging predictions. Does not assume linearity. Learns generalizable demand rules — best for new device launches. |
| Champion | **XGBoost** | Gradient boosting — trees built sequentially, each correcting prior residuals. Best accuracy on known SKUs. |

---

## Results

### Time-series CV mean RMSE (5 walk-forward folds)

| Model | Mean RMSE | Std |
|---|---|---|
| Ridge | 1,551 | 271 |
| Lasso | 1,576 | 268 |
| XGBoost | 899 | 240 |

### Held-out test set

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Ridge | 869 | 1,309 | 0.44 | 620% |
| Lasso | 946 | 1,362 | 0.39 | 651% |
| Random Forest | 667 | 1,140 | 0.57 | 866% |
| **XGBoost (champion)** | **552** | **1,006** | **0.67** | 882% |

**Non-linear gain**: 23.2% RMSE reduction vs best linear model.

**On MAE**: an MAE of 552 means the forecast is off by ±552 subscribers per SKU per month on average. For a flagship like iPhone XS Max in California (~18,000 active subscribers), that is a 3% error — operationally acceptable for reserve sizing and inventory planning.

**On MAPE**: values are inflated by low-volume SKUs where actual subscriber counts approach zero — even a small absolute miss becomes a large percentage. MAE and RMSE are the primary metrics for this dataset.

### Feature importance finding

XGBoost's top features are specific model names and colour variants (memorized which SKUs are popular). Random Forest's top features are generalizable demand drivers — prior month subscriber count (0.36), storage size (0.11), prior month churn (0.11), prior month claims (0.09), device age (0.08). This means:

- **XGBoost** is more accurate for known SKUs in existing markets
- **Random Forest** generalizes better to new device launches with no prior history

---

## Recommendations

| Priority | Action |
|---|---|
| Deploy now | XGBoost for monthly operational forecasts on existing fleet |
| Deploy now | Random Forest for pre-launch forecasts on new device models |
| Deploy now | Flag SKUs with < 500 subscribers for manual review — model error can exceed actual count |
| Short-term | Log-transform the target variable to reduce skew sensitivity across all models |
| Medium-term | Integrate SHAP values for per-SKU explainability to business stakeholders |

---

## Pipeline

```
demand_forecast.py
│
├── load_and_clean()          — load Excel, drop pre-orders, filter horizon
├── build_features()          — lag-1 features, drop concurrent columns, one-hot encode
├── split_data()              — 60/15/25 chronological split
├── scale_features()          — StandardScaler fitted on train only
├── ts_cross_validate()       — TimeSeriesSplit 5-fold RMSE comparison
├── train_models()            — Ridge, Lasso, Random Forest, XGBoost
├── evaluate_all()            — MAE/RMSE/MAPE/R² on val + test; champion/challenger summary
├── plot_model_comparison()   — 4-model bar chart across MAE, RMSE, MAPE
├── plot_lasso_selection()    — non-zero Lasso coefficients + kept/zeroed donut chart
├── plot_feature_importance() — XGBoost + RF top-15 importances side by side
├── plot_residuals()          — predicted vs actual + residuals vs predicted (XGBoost)
└── forecast_future()         — Dec 2023 + Jan 2024 forecast using Nov 2023 lag-1 inputs
```

### Output files

| File | Description |
|---|---|
| `report.html` | Full interactive model assessment report — open in browser |
| `model_comparison.png` | MAE / RMSE / MAPE bar chart across all 4 models |
| `lasso_feature_selection.png` | Non-zero Lasso coefficients and feature retention summary |
| `feature_importance.png` | XGBoost and Random Forest top-15 feature importances side by side |
| `residual_analysis.png` | XGBoost predicted vs actual and residual distribution |
| `forecast_202312_202401.csv` | Dec 2023 and Jan 2024 subscriber forecasts (XGBoost + RF) |

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl

# 2. Place the data file at the path configured in demand_forecast.py
#    DATA_FILE = "Worksheet in ML_Assignment_2023.xlsx"

# 3. Run EDA (generates 8 exploratory charts)
python eda.py

# 4. Run forecasting pipeline
python demand_forecast.py
```

> The raw data file (`.xlsx`) is excluded from the repository via `.gitignore`.

---

## Project Structure

```
.
├── demand_forecast.py            # Main forecasting pipeline
├── eda.py                        # Exploratory data analysis (8 charts)
├── report.html                   # Interactive model assessment report
├── model_comparison.png
├── lasso_feature_selection.png
├── feature_importance.png
├── residual_analysis.png
├── forecast_202312_202401.csv
└── .gitignore
```
