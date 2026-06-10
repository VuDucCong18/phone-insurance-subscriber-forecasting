# Phone Insurance Subscriber Forecasting

Forecasting monthly active phone insurance subscribers (Closing Subs Monthly) across device models, markets, and colours — using a leakage-free ML pipeline with lag-1 features, walk-forward cross-validation, and a champion/challenger model evaluation framework.

---

## Business Context

A phone insurance provider tracks the number of active subscribers per device model at the end of each month (**Closing Subs Monthly**). This is not units sold — it is the size of the insured base for a given device variant in a given market at a point in time.

Accurate subscriber forecasts drive three operational decisions:

| Decision | Why it needs a forecast |
|---|---|
| Claims reserve sizing | Expected claims scale with the active subscriber base |
| Device parts inventory | Warranty repairs require stocked components per model |
| Renewal campaign targeting | Models approaching lifecycle decay need proactive retention |

The dataset covers **Jan 2019 – Nov 2023**, three US markets (California, Nevada, Texas), three brands (Apple, Samsung, Oppo), and 40 device model families — totalling 13,381 rows across 25 columns.

---

## Key Insight: Data Leakage Risk

Two features in the raw data are highly correlated with the target but **cannot be used as concurrent features**:

- **Churn** — subscribers who cancelled *in the same month*. This is known only after the month closes, the same moment as the target.
- **Filed Claims** — warranty claims submitted *in the same month*. Same problem.

Using them directly as features would constitute data leakage: the model would be trained on information that would not exist at forecast time. Both are instead **lagged by 1 month** — prior month values are legitimate signals available before the forecast period begins.

The high correlation also partly reflects a **size effect**: churn and claims both scale with the installed base, so they correlate with the target simply because larger subscriber bases produce more of both. This is not a causal relationship.

---

## Methodology

### Target Variable

```
Closing Subs Monthly  =  active subscribers at month-end for a given
                         (country, ModelFamily, Colour, YearMonth) combination
```

Right-skewed distribution (median ~700, mean ~1,400, max ~25,000) — justifies tree-based models over plain linear regression.

### Feature Engineering

| Feature | Source | Notes |
|---|---|---|
| `lag1_closing_subs` | Closing Subs Monthly, shifted 1 month | Strongest honest predictor (r = 0.67) |
| `lag1_churn` | Churn, shifted 1 month | Retention pressure signal (r = 0.46) |
| `lag1_filed_claims` | Filed Claims, shifted 1 month | Base activity signal (r = 0.46) |
| `Model Age (Days)` | Days since device launch | Captures lifecycle stage |
| `Size` | Storage capacity (GB) | Proxy for price tier |
| `country`, `ModelFamily`, `Colour` | One-hot encoded (drop_first=True) | Market and SKU identity |

Lag shifts are applied **within each (country, ModelFamily) group** to avoid cross-group contamination.

**Dropped features** — concurrent with the target (leakage): Churn, Filed Claims, Claims, Claims Swap, Claims Replacement, IR Rate Swap, IR Rate Replacement, IR Rate Monthly, Churn Rate, Model Age (Months).

### Data Cleaning

- **Pre-order rows dropped**: 2,128 rows where `Model Age (Days) < 0` (16.3% of rows, ~4.2% of subscriber volume). These corrupt lag calculations for the first real month of each model.
- **NaN lag rows dropped**: 91 rows representing the first observed month of each (country, ModelFamily) group — no prior month exists to look back at.
- **Horizon filter**: rows limited to model families present in the final month (Nov 2023), ensuring forecast continuity.
- Final dataset: **10,859 rows × 56 features**.

### Train / Validation / Test Split

Chronological split, no shuffling:

```
Train  60%  →  Jan 2019 – ~Apr 2022   (6,515 rows)
Val    15%  →  ~May 2022 – ~Dec 2022  (1,629 rows)  ← hyperparameter tuning
Test   25%  →  ~Jan 2023 – Nov 2023   (2,715 rows)  ← held-out, final eval only
```

### Cross-Validation

**TimeSeriesSplit (5 folds, walk-forward)** — each fold trains only on data before its validation window. Standard K-fold is not used as it would allow future data to leak into training.

`StandardScaler` is fitted on the training fold only and applied to validation/test, preventing scale leakage.

---

## Models

A **champion/challenger** framework is used to compare linear baselines against non-linear models.

| Role | Model | Notes |
|---|---|---|
| Linear baseline | **Ridge** (RidgeCV) | L2 regularisation, alpha tuned via TimeSeriesSplit |
| Linear + selection | **Lasso** (LassoCV) | L1 regularisation; drives irrelevant coefficients to zero |
| Challenger | **Random Forest** | 200 trees, max_depth=15, max_features=0.5 |
| Champion | **XGBoost** | 300 estimators, lr=0.05, subsample=0.8, colsample_bytree=0.8 |

---

## Results

### Time-Series CV (mean RMSE across 5 walk-forward folds)

| Model | Mean RMSE | Std |
|---|---|---|
| Ridge | 1,551 | 271 |
| Lasso | 1,576 | 268 |
| XGBoost | 899 | 240 |

### Held-Out Test Set

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Ridge | 869 | 1,309 | 0.44 | 620% |
| Lasso | 946 | 1,362 | 0.39 | 651% |
| Random Forest | 667 | 1,140 | 0.57 | 866% |
| **XGBoost (champion)** | **552** | **1,006** | **0.67** | 882% |

**Non-linear gain**: 23.2% RMSE reduction vs best linear model (Ridge).

> **Note on MAPE**: values are inflated by low-volume SKUs where the denominator (actual subscribers) approaches zero. MAE and RMSE are the more reliable metrics for this dataset.

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

### Output Files

| File | Description |
|---|---|
| `model_comparison.png` | MAE / RMSE / MAPE bar chart across all 4 models |
| `lasso_feature_selection.png` | Non-zero Lasso coefficients and feature retention summary |
| `feature_importance.png` | XGBoost and Random Forest top-15 feature importances |
| `residual_analysis.png` | XGBoost predicted vs actual and residual distribution |
| `forecast_202312_202401.csv` | Dec 2023 and Jan 2024 subscriber forecasts (XGBoost + RF) |

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl

# 2. Place the data file at the path configured in demand_forecast.py
#    DATA_FILE = "Worksheet in ML_Assignment_2023.xlsx"

# 3. Run EDA
python eda.py

# 4. Run forecasting pipeline
python demand_forecast.py
```

> The raw data file (`.xlsx`) is excluded from the repository via `.gitignore`.

---

## Project Structure

```
.
├── demand_forecast.py          # Main forecasting pipeline
├── eda.py                      # Exploratory data analysis (8 charts)
├── model_comparison.png
├── lasso_feature_selection.png
├── feature_importance.png
├── residual_analysis.png
├── forecast_202312_202401.csv
└── .gitignore
```
