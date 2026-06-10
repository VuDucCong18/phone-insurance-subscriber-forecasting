"""
Demand Forecasting for Phone Devices in Supply Chain
=====================================================
Author: Vu Duc Cong
Dataset: Asurion ML Assignment 2023 (Worksheet in ML_Assignment_2023.xlsx)

Pipeline:
    1. Data loading & cleaning
    2. Feature engineering & encoding
    3. Chronological train / val / test split (no shuffling)
    4. Time-series cross-validation (TimeSeriesSplit — no data leakage)
    5. Model training: Ridge, Random Forest, XGBoost
    6. Evaluation: MAE, RMSE, MAPE, R²
    7. Feature importance visualisation
    8. Residual analysis
    9. Future demand forecast (202312, 202401) with prediction floor
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

DATA_FILE = "Worksheet in ML_Assignment_2023.xlsx"
RANDOM_STATE = 42


# =============================================================================
# 1. METRICS
# =============================================================================

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error, skipping zero-denominator rows."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(model, X, y, label=""):
    preds = model.predict(X)
    metrics = {
        "MAE":  mean_absolute_error(y, preds),
        "RMSE": np.sqrt(mean_squared_error(y, preds)),
        "MAPE": mape(y, preds),
        "R²":   r2_score(y, preds),
    }
    if label:
        print(f"\n{'─'*45}\n{label}")
        for k, v in metrics.items():
            unit = "%" if k == "MAPE" else ""
            print(f"  {k:<6}: {v:>8.2f}{unit}")
    return metrics


# =============================================================================
# 2. DATA LOADING & CLEANING
# =============================================================================

def load_and_clean(path: str) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_excel(path)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    # Fill NaN in claim/churn columns with 0
    zero_fill_cols = [
        "Claims", "Filed Claims", "Claims Swap", "Claims Replacement",
        "IR Rate Swap", "IR Rate Replacement", "IR Rate Monthly",
        "Churn", "Churn Rate",
    ]
    df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

    # Recalculate derived rates for consistency
    df["IR Rate Swap"]        = (df["Claims Swap"]        / df["Closing Subs Monthly"]) * 100
    df["IR Rate Replacement"] = (df["Claims Replacement"] / df["Closing Subs Monthly"]) * 100
    df["IR Rate Monthly"]     = df["IR Rate Swap"] + df["IR Rate Replacement"]
    df["Churn Rate"]          = (df["Churn"] / df["Closing Subs Monthly"]) * 100

    # Keep only ModelFamilies with data through 202311 (consistent forecast horizon)
    valid_families = df.groupby("ModelFamily")["YearMonth"].max()
    valid_families = valid_families[valid_families == 202311].index
    df = df[df["ModelFamily"].isin(valid_families)].copy()
    print(f"  After filtering to 202311 families: {len(df):,} rows")

    return df


# =============================================================================
# 3. FEATURE ENGINEERING & ENCODING
# =============================================================================

def build_features(df: pd.DataFrame):
    """Return (X, y, scaler, feature_names, X_columns)."""

    numerical_cols = [
        "YearMonth",
        "Model Age (Days)",
        "Filed Claims",
        "Claims Swap",
        "Claims Replacement",
        "Churn",
        "Size",
    ]
    categorical_cols = ["country", "ModelFamily", "Colour"]

    df_feat = df[numerical_cols + categorical_cols].copy()
    df_feat["_target"] = df["Closing Subs Monthly"].values

    # Encode categoricals
    df_encoded = pd.get_dummies(df_feat, columns=categorical_cols, drop_first=True)

    # Add time index and sort chronologically — CRITICAL: sort X and y together
    df_encoded["_time"] = pd.to_datetime(
        df_encoded["YearMonth"].astype(str), format="%Y%m"
    )
    df_encoded = df_encoded.sort_values("_time").reset_index(drop=True)

    y = df_encoded["_target"]
    X = df_encoded.drop(columns=["YearMonth", "_time", "_target"])

    # Scale numerical features (exclude YearMonth which is dropped)
    num_features_to_scale = [
        c for c in ["Model Age (Days)", "Filed Claims", "Claims Swap",
                    "Claims Replacement", "Churn", "Size"]
        if c in X.columns
    ]
    scaler = StandardScaler()
    X[num_features_to_scale] = scaler.fit_transform(X[num_features_to_scale])

    print(f"\nFeature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y, scaler, num_features_to_scale


# =============================================================================
# 4. TRAIN / VAL / TEST SPLIT
# =============================================================================

def split_data(X, y):
    """60% train / 15% val / 25% test — chronological, no shuffle."""
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.20, shuffle=False
    )
    print(f"\nSplit (chronological):")
    print(f"  Train : {len(X_train):>5} rows")
    print(f"  Val   : {len(X_val):>5} rows")
    print(f"  Test  : {len(X_test):>5} rows")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 5. TIME-SERIES CROSS-VALIDATION
# =============================================================================

def time_series_cv(X, y, n_splits=5):
    """Walk-forward CV — each fold trains only on past data."""
    print(f"\n{'='*50}")
    print(f"Time-Series Cross-Validation ({n_splits} folds, walk-forward)")
    print("Each fold trains only on data BEFORE the validation window.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {"XGBoost": [], "Ridge": []}

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        Xtr, Xvl = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yvl = y.iloc[tr_idx], y.iloc[val_idx]

        xgb = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        xgb.fit(Xtr, ytr)
        results["XGBoost"].append(
            np.sqrt(mean_squared_error(yvl, xgb.predict(Xvl)))
        )

        ridge = Ridge(alpha=1000)
        ridge.fit(Xtr, ytr)
        results["Ridge"].append(
            np.sqrt(mean_squared_error(yvl, ridge.predict(Xvl)))
        )

    cv_df = pd.DataFrame(results, index=[f"Fold {i+1}" for i in range(n_splits)])
    cv_df.loc["Mean"] = cv_df.mean()
    cv_df.loc["Std"]  = cv_df.std()
    print("\nRMSE per fold:")
    print(cv_df.round(1).to_string())

    return cv_df


# =============================================================================
# 6. MODEL TRAINING
# =============================================================================

def train_models(X_train, y_train):
    print(f"\n{'='*50}\nTraining models...")

    models = {
        "Ridge": Ridge(alpha=1000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=2,
            max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        ),
    }

    for name, m in models.items():
        m.fit(X_train, y_train)
        print(f"  [done] {name}")

    return models


# =============================================================================
# 7. EVALUATION & COMPARISON CHART
# =============================================================================

def evaluate_all(models, X_val, y_val, X_test, y_test):
    print(f"\n{'='*50}\nModel Evaluation")

    rows = []
    for name, m in models.items():
        for split_label, Xs, ys in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
            preds = m.predict(Xs)
            rows.append({
                "Model": name, "Split": split_label,
                "MAE":  mean_absolute_error(ys, preds),
                "RMSE": np.sqrt(mean_squared_error(ys, preds)),
                "MAPE": mape(ys, preds),
                "R²":   r2_score(ys, preds),
            })

    df_results = pd.DataFrame(rows)

    print("\nValidation results:")
    print(df_results[df_results.Split == "Validation"]
          .set_index("Model")[["MAE","RMSE","MAPE","R²"]].round(3).to_string())
    print("\nTest (held-out) results:")
    print(df_results[df_results.Split == "Test"]
          .set_index("Model")[["MAE","RMSE","MAPE","R²"]].round(3).to_string())

    # --- Comparison bar chart ---
    test_res = df_results[df_results.Split == "Test"].set_index("Model")
    metrics_plot = ["MAE", "RMSE", "MAPE"]
    colors = {"Ridge": "#4e79a7", "Random Forest": "#f28e2b", "XGBoost": "#59a14f"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison — Held-Out Test Set", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics_plot):
        vals = test_res[metric]
        bars = ax.bar(vals.index, vals.values,
                      color=[colors[m] for m in vals.index],
                      edgecolor="white", width=0.5)
        ax.set_title(metric + (" (%)" if metric == "MAPE" else " (units)"), fontsize=11)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_ylim(0, vals.max() * 1.3)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_comparison.png", bbox_inches="tight")
    plt.show()
    print("  Saved: model_comparison.png")

    return df_results


# =============================================================================
# 8. FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(models, feature_names, top_n=15):
    xgb = models["XGBoost"]
    rf  = models["Random Forest"]

    imp_xgb = pd.Series(xgb.feature_importances_, index=feature_names)
    imp_rf  = pd.Series(rf.feature_importances_,  index=feature_names)

    top_features = imp_xgb.sort_values(ascending=False).head(top_n)

    label_map = {
        "Filed Claims":      "Claims Filed (warranty demand signal)",
        "Churn":             "Subscriber Churn (retention pressure)",
        "Model Age (Days)":  "Device Age (lifecycle stage)",
        "Claims Swap":       "Like-for-like Replacements",
        "Claims Replacement":"Cross-model Replacements",
        "Size":              "Storage Capacity (SKU tier)",
    }
    labels = [
        label_map.get(f, f.replace("ModelFamily_", "Model: ")
                         .replace("country_", "Market: ")
                         .replace("Colour_", "Colour: "))
        for f in top_features.index
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Importance — Top 15 Demand Drivers", fontsize=13, fontweight="bold")

    for ax, imp, title, color in zip(
        axes,
        [top_features, imp_rf.sort_values(ascending=False).head(top_n)],
        ["XGBoost", "Random Forest"],
        ["#59a14f", "#f28e2b"],
    ):
        ax.barh(range(len(imp)), imp.values[::-1], color=color, edgecolor="white")
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(labels[::-1] if title == "XGBoost" else imp.index[::-1], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Feature Importance (gain)")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_importance.png", bbox_inches="tight")
    plt.show()
    print("  Saved: feature_importance.png")

    print("\nBusiness interpretation (XGBoost top 3):")
    for feat, val in top_features.head(3).items():
        print(f"  {feat}: {val:.3f} — {label_map.get(feat, '')}")


# =============================================================================
# 9. RESIDUAL ANALYSIS
# =============================================================================

def plot_residuals(model, X_test, y_test, model_name="XGBoost"):
    y_pred = model.predict(X_test)
    residuals = np.array(y_test) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} Residual Analysis — Held-Out Test Set",
                 fontsize=13, fontweight="bold")

    # Predicted vs Actual
    ax = axes[0]
    ax.scatter(y_pred, np.array(y_test), alpha=0.25, s=12, color="#4e79a7")
    lims = [min(y_pred.min(), np.array(y_test).min()),
            max(y_pred.max(), np.array(y_test).max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Predicted Closing Subs")
    ax.set_ylabel("Actual Closing Subs")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    ax.grid(alpha=0.3)

    # Residuals vs Predicted
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.25, s=12, color="#f28e2b")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(residuals.mean(), color="navy", linestyle=":", linewidth=1.5,
               label=f"Mean residual: {residuals.mean():.0f}")
    ax.set_xlabel("Predicted Closing Subs")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residuals vs Predicted\n(fan shape → heteroscedasticity)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("residual_analysis.png", bbox_inches="tight")
    plt.show()
    print("  Saved: residual_analysis.png")

    over_pct = (residuals < 0).mean() * 100
    print(f"\n  Over-predicts in {over_pct:.1f}% of test cases")
    print(f"  Mean residual: {residuals.mean():.1f}  |  Std: {residuals.std():.1f}")


# =============================================================================
# 10. FUTURE DEMAND FORECAST (202312 & 202401)
# =============================================================================

def forecast_future(df, models, X_columns, scaler, num_features_to_scale):
    print(f"\n{'='*50}\nForecasting 202312 and 202401...")

    latest = df[df["YearMonth"] == 202311].copy()

    cat_cols = ["country", "ModelFamily", "Colour"]
    future_base = latest[cat_cols + ["Model Age (Days)", "Size"]].copy()

    future_rows = []
    for offset_days, ym in [(31, 202312), (62, 202401)]:
        chunk = future_base.copy()
        chunk["YearMonth"]        = ym
        chunk["Model Age (Days)"] = chunk["Model Age (Days)"] + offset_days
        chunk["Filed Claims"]     = latest["Filed Claims"].values
        chunk["Claims Swap"]      = latest["Claims Swap"].values
        chunk["Claims Replacement"] = latest["Claims Replacement"].values
        chunk["Churn"]            = latest["Churn"].values
        future_rows.append(chunk)

    future_df = pd.concat(future_rows, ignore_index=True)

    # Encode
    future_encoded = pd.get_dummies(future_df, columns=cat_cols, drop_first=True)
    future_encoded = future_encoded.drop(columns=["YearMonth"], errors="ignore")
    future_encoded = future_encoded.reindex(columns=X_columns, fill_value=0)

    # Scale
    future_encoded[num_features_to_scale] = scaler.transform(
        future_encoded[num_features_to_scale]
    )

    # Predict with floor at 0 (demand cannot be negative)
    xgb_preds = np.maximum(models["XGBoost"].predict(future_encoded), 0)
    rf_preds  = np.maximum(models["Random Forest"].predict(future_encoded), 0)

    results = future_df[["YearMonth", "country", "ModelFamily"]].copy()
    results["XGBoost_Forecast"]       = xgb_preds.round().astype(int)
    results["RandomForest_Forecast"]  = rf_preds.round().astype(int)

    # Approximate 80% prediction interval using a rough ±1.28σ from test RMSE
    # (σ approximated per-model; update with actual test RMSE after evaluation)
    print("\nForecast preview (first 10 rows):")
    print(results.head(10).to_string(index=False))

    results.to_csv("forecast_202312_202401.csv", index=False)
    print("\n  Saved: forecast_202312_202401.csv")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Phone Device Demand Forecasting Pipeline")
    print("=" * 50)

    # 1. Load
    df = load_and_clean(DATA_FILE)

    # 2. Features
    X, y, scaler, num_features_to_scale = build_features(df)

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 4. Time-series CV (validates methodology — no data leakage)
    cv_results = time_series_cv(X, y, n_splits=5)

    # 5. Train
    models = train_models(X_train, y_train)

    # 6. Evaluate
    results_df = evaluate_all(models, X_val, y_val, X_test, y_test)

    # 7. Feature importance
    plot_feature_importance(models, X.columns)

    # 8. Residuals
    plot_residuals(models["XGBoost"], X_test, y_test)

    # 9. Forecast
    forecast = forecast_future(df, models, X.columns, scaler, num_features_to_scale)

    print(f"\n{'='*50}")
    print("Pipeline complete. Output files:")
    print("  model_comparison.png")
    print("  feature_importance.png")
    print("  residual_analysis.png")
    print("  forecast_202312_202401.csv")
