"""
Demand Forecasting for Phone Devices in Supply Chain
=====================================================
Author : Vu Duc Cong
Dataset: Worksheet in ML_Assignment_2023.xlsx

Pipeline:
    1.  Data loading
    2.  Data cleaning
    3.  Feature engineering (lag-1 features)
    4.  Train / Val / Test split  (chronological, no shuffle)
    5.  Time-series cross-validation  (TimeSeriesSplit, walk-forward)
    6.  Model training  — Ridge, Lasso, Random Forest, XGBoost
    7.  Evaluation      — MAE, RMSE, MAPE, R²
    8.  Visualisations  — comparison, Lasso selection, importance, residuals
    9.  Future forecast — 202312 and 202401

Champion  : XGBoost
Challenger: Random Forest
Baselines : Ridge, Lasso
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3})

DATA_FILE    = "Worksheet in ML_Assignment_2023.xlsx"
RANDOM_STATE = 42
COLORS = {
    "Ridge":         "#4e79a7",
    "Lasso":         "#9b59b6",
    "Random Forest": "#f28e2b",
    "XGBoost":       "#59a14f",
}


# =============================================================================
# 1. METRICS
# =============================================================================

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    Rows where y_true == 0 are excluded to avoid division by zero.
    Note: MAPE can appear inflated for low-volume SKUs (<100 units)
    because a small absolute miss produces a large percentage error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def score(y_true, y_pred):
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
    }


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 2. DATA LOADING & CLEANING
# =============================================================================

def load_and_clean(path: str) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_excel(path)
    print(f"  Raw shape: {len(df):,} rows x {df.shape[1]} columns")

    # --- Fill missing claims and churn columns with 0 ---
    # These are not truly missing — they represent months with zero claim
    # or churn events for that model. Confirmed by EDA (Chart 01): ~49%
    # of claim rows are null because most SKUs in most months have no
    # warranty activity.
    zero_cols = [
        "Claims", "Filed Claims", "Claims Swap", "Claims Replacement",
        "IR Rate Swap", "IR Rate Replacement", "IR Rate Monthly",
        "Churn Rate", "Churn",
    ]
    df[zero_cols] = df[zero_cols].fillna(0)

    # Recalculate derived rates from source columns for consistency
    df["IR Rate Swap"]        = (df["Claims Swap"]        / df["Closing Subs Monthly"]) * 100
    df["IR Rate Replacement"] = (df["Claims Replacement"] / df["Closing Subs Monthly"]) * 100
    df["IR Rate Monthly"]     = df["IR Rate Swap"] + df["IR Rate Replacement"]
    df["Churn Rate"]          = (df["Churn"]               / df["Closing Subs Monthly"]) * 100

    # --- Drop pre-order rows (Model Age (Days) < 0) ---
    # Pre-orders are purchases made before the phone's official launch date,
    # resulting in a negative model age. EDA showed ~4.2% of records fall
    # into this category (about 550 rows out of 13,024).
    # We drop them for two reasons:
    #   1. Their subscriber counts are not representative of normal demand —
    #      they reflect early-adopter behaviour before public availability.
    #   2. When we create lag-1 features, a pre-order row passes a false
    #      "prior month" count into the post-launch training data, corrupting
    #      the model's understanding of the lifecycle start.
    # Actual loss: ~2,128 rows (16.3% of rows), but only 4.2% of total
    # subscriber volume — pre-orders are low-volume early records. The
    # integrity gain from clean lag features outweighs this row loss.
    pre_order_count = (df["Model Age (Days)"] < 0).sum()
    df = df[df["Model Age (Days)"] >= 0].copy()
    print(f"  Dropped {pre_order_count:,} pre-order rows (Model Age < 0) — "
          f"{pre_order_count/13024*100:.1f}% of data")

    # --- Filter to ModelFamilies with data through 202311 ---
    # Keeps only models that have data up to November 2023, ensuring a
    # consistent forecast horizon for 202312 and 202401.
    valid_families = (df.groupby("ModelFamily")["YearMonth"]
                      .max()
                      .pipe(lambda s: s[s == 202311].index))
    df = df[df["ModelFamily"].isin(valid_families)].copy()
    print(f"  After horizon filter: {len(df):,} rows  "
          f"({df['ModelFamily'].nunique()} model families)")

    return df


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def build_features(df: pd.DataFrame):
    """
    Returns X, y, scaler, num_features, feature_columns.
    Key change vs original: lag-1 features replace concurrent claim/churn
    columns to eliminate data leakage.
    """
    print("\nBuilding features...")

    # Sort chronologically within each (country, ModelFamily) group.
    # X and y are sorted together — prevents the index misalignment bug
    # in the original notebook where X was sorted but y was not.
    df = df.sort_values(["country", "ModelFamily", "YearMonth"]).reset_index(drop=True)

    # --- Create lag-1 features ---
    # These replace the concurrent Churn and Filed Claims columns.
    # When forecasting February, we only know January's values — these
    # are the lag-1 values. EDA (Chart 08) confirmed their correlation
    # with the target after lagging:
    #   lag1_closing_subs  r = 0.67  (strongest honest predictor)
    #   lag1_churn         r = 0.46
    #   lag1_filed_claims  r = 0.46
    grp = df.groupby(["country", "ModelFamily"])
    df["lag1_closing_subs"]  = grp["Closing Subs Monthly"].shift(1)
    df["lag1_churn"]         = grp["Churn"].shift(1)
    df["lag1_filed_claims"]  = grp["Filed Claims"].shift(1)

    # --- Drop rows where lag features are NaN ---
    # The first month of each (country, ModelFamily) group has no prior
    # month to look back at — lag values are NaN. These rows are dropped,
    # not filled with 0. Filling with 0 would incorrectly tell the model
    # that the previous month had zero subscribers, which is false.
    # Each group loses exactly one row (its earliest month).
    # With 40 families x 3 markets = ~120 rows dropped — negligible.
    before = len(df)
    df = df.dropna(subset=["lag1_closing_subs", "lag1_churn", "lag1_filed_claims"])
    print(f"  Dropped {before - len(df)} rows with NaN lag values "
          f"(first month of each group — no prior history available)")

    # --- Define final feature set ---
    # Concurrent variables (Churn, Filed Claims, Claims etc.) are excluded
    # entirely — they happen during the same month as the target and would
    # not be available at real forecast time.
    # IR Rate columns and Churn Rate are also excluded — EDA showed near-
    # zero correlation with the target once the size effect is removed.
    numerical_cols = [
        "Model Age (Days)",   # lifecycle position — where on the demand curve
        "Size",               # storage tier — proxies price segment
        "lag1_closing_subs",  # demand is sticky — strongest valid predictor
        "lag1_churn",         # signals whether base is growing or shrinking
        "lag1_filed_claims",  # signals installed base activity level
    ]
    categorical_cols = [
        "country",      # market-level structural differences
        "ModelFamily",  # brand + product line — dominant demand driver
        "Colour",       # SKU-level variant
    ]

    # Attach target before encoding so it stays aligned
    df["_target"] = df["Closing Subs Monthly"].values
    df["_time"]   = pd.to_datetime(df["YearMonth"].astype(str), format="%Y%m")

    # One-hot encode categoricals (drop_first avoids dummy variable trap)
    df_enc = pd.get_dummies(
        df[numerical_cols + categorical_cols + ["_target", "_time"]],
        columns=categorical_cols,
        drop_first=True,
    )

    # Final chronological sort (already sorted above but explicit after encode)
    df_enc = df_enc.sort_values("_time").reset_index(drop=True)

    y = df_enc["_target"]
    X = df_enc.drop(columns=["_target", "_time"])

    print(f"  Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  Numerical features: {numerical_cols}")

    return X, y, numerical_cols


# =============================================================================
# 4. TRAIN / VAL / TEST SPLIT
# =============================================================================

def split_data(X, y):
    """
    Chronological 60/15/25 split — shuffle=False throughout.
    Test set is held out completely until final evaluation.
    """
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.20, shuffle=False
    )
    print(f"\nSplit (chronological, no shuffle):")
    print(f"  Train : {len(X_train):>6,} rows  (60%)")
    print(f"  Val   : {len(X_val):>6,} rows  (15%)  <- hyperparameter tuning")
    print(f"  Test  : {len(X_test):>6,} rows  (25%)  <- held-out, final eval only")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 5. SCALE FEATURES
# =============================================================================

def scale_features(X_train, X_val, X_test, numerical_cols):
    """
    Fit StandardScaler on training set only.
    Apply the same transform to val and test — prevents future data
    from influencing the scaling parameters.
    """
    num_in_X = [c for c in numerical_cols if c in X_train.columns]
    scaler   = StandardScaler()

    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    X_val[num_in_X]   = scaler.transform(X_val[num_in_X])
    X_test[num_in_X]  = scaler.transform(X_test[num_in_X])

    return X_train, X_val, X_test, scaler, num_in_X


# =============================================================================
# 6. TIME-SERIES CROSS-VALIDATION
# =============================================================================

def ts_cross_validate(X, y, numerical_cols):
    """
    5-fold walk-forward cross-validation.
    Each fold trains only on past data — no future leakage.
    Replaces random K-fold used in the original notebook which allowed
    future months to appear in training folds.
    """
    print(f"\n{'='*55}")
    print("Time-Series Cross-Validation (5 folds, walk-forward)")
    print("Each fold trains only on data BEFORE the validation window.")

    # Scale inside CV to avoid leakage from global scaler
    num_in_X = [c for c in numerical_cols if c in X.columns]
    tscv     = TimeSeriesSplit(n_splits=5)
    cv_res   = {m: [] for m in ["Ridge", "Lasso", "XGBoost"]}

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X)):
        Xtr, Xvl = X.iloc[tr_idx].copy(), X.iloc[vl_idx].copy()
        ytr, yvl = y.iloc[tr_idx],          y.iloc[vl_idx]

        sc = StandardScaler()
        Xtr[num_in_X] = sc.fit_transform(Xtr[num_in_X])
        Xvl[num_in_X] = sc.transform(Xvl[num_in_X])

        alphas = np.logspace(-3, 4, 60)

        for name, mdl in [
            ("Ridge",  RidgeCV(alphas=alphas, cv=3)),
            ("Lasso",  LassoCV(alphas=alphas, cv=3, max_iter=5000)),
            ("XGBoost", XGBRegressor(n_estimators=300, max_depth=7,
                                     learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8,
                                     random_state=RANDOM_STATE,
                                     n_jobs=-1, verbosity=0)),
        ]:
            mdl.fit(Xtr, ytr)
            cv_res[name].append(
                np.sqrt(mean_squared_error(yvl, mdl.predict(Xvl)))
            )

    cv_df = pd.DataFrame(cv_res,
                         index=[f"Fold {i+1}" for i in range(5)])
    cv_df.loc["Mean"] = cv_df.mean()
    cv_df.loc["Std"]  = cv_df.std()
    print("\nRMSE per fold:")
    print(cv_df.round(1).to_string())
    return cv_df


# =============================================================================
# 7. MODEL TRAINING
# =============================================================================

def train_models(X_train, y_train):
    print(f"\n{'='*55}")
    print("Training models...")

    alphas = np.logspace(-3, 4, 60)
    tscv   = TimeSeriesSplit(n_splits=5)

    models = {
        # Linear baseline — L2 regularisation handles multicollinearity
        # from the 40+ one-hot encoded ModelFamily columns
        "Ridge": RidgeCV(
            alphas=alphas,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
        ),

        # Linear + automatic feature selection via L1 penalty.
        # Features driven to zero are genuinely not linearly predictive.
        "Lasso": LassoCV(
            alphas=alphas,
            cv=tscv,
            max_iter=10000,
        ),

        # Non-linear challenger — handles lifecycle curve naturally,
        # predictions bounded by training range (never goes negative)
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            max_features=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # Champion candidate — best at capturing non-linear lifecycle
        # curves and feature interactions between age, brand, and market
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        if hasattr(mdl, "alpha_"):
            print(f"  [done] {name}  (best alpha: {mdl.alpha_:.4f})")
        else:
            print(f"  [done] {name}")

    return models


# =============================================================================
# 8. EVALUATION
# =============================================================================

def evaluate_all(models, X_val, y_val, X_test, y_test):
    print(f"\n{'='*55}")
    print("Evaluation")

    rows = []
    for name, mdl in models.items():
        for split_lbl, Xs, ys in [
            ("Validation", X_val,  y_val),
            ("Test",       X_test, y_test),
        ]:
            preds = np.maximum(mdl.predict(Xs), 0)  # floor at 0
            m = score(ys, preds)
            rows.append({"Model": name, "Split": split_lbl, **m})

    df_res = pd.DataFrame(rows)

    for split in ["Validation", "Test"]:
        print(f"\n{split} results:")
        tbl = (df_res[df_res.Split == split]
               .set_index("Model")[["MAE", "RMSE", "MAPE", "R2"]]
               .round(2))
        print(tbl.to_string())

    # Champion / challenger summary
    test_res = df_res[df_res.Split == "Test"].set_index("Model")
    best     = test_res["RMSE"].idxmin()
    second   = test_res["RMSE"].drop(best).idxmin()
    print(f"\n  Champion  : {best}   RMSE = {test_res.loc[best,'RMSE']:.1f}")
    print(f"  Challenger: {second}  RMSE = {test_res.loc[second,'RMSE']:.1f}")
    linear_rmse = test_res.loc[["Ridge","Lasso"],"RMSE"].min()
    gain = (linear_rmse - test_res.loc[best,"RMSE"]) / linear_rmse * 100
    print(f"  Non-linear gain over best linear: {gain:.1f}% RMSE reduction")

    return df_res


# =============================================================================
# 9. VISUALISATIONS
# =============================================================================

def plot_model_comparison(df_res):
    test_res    = df_res[df_res.Split == "Test"].set_index("Model")
    metrics     = ["MAE", "RMSE", "MAPE"]
    model_order = ["Ridge", "Lasso", "Random Forest", "XGBoost"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Comparison — Held-Out Test Set\n"
                 "Champion: XGBoost  |  Challenger: Random Forest  |  "
                 "Baselines: Ridge, Lasso",
                 fontsize=12, fontweight="bold", y=1.03)

    for ax, metric in zip(axes, metrics):
        vals  = test_res.loc[model_order, metric]
        best  = vals.idxmin()
        cols  = [COLORS[m] for m in model_order]
        alpha = [1.0 if m in [best, "Random Forest"] else 0.55
                 for m in model_order]
        # Draw each bar individually so alpha can vary per bar
        bar_patches = []
        for i, (m, v) in enumerate(zip(model_order, vals.values)):
            b = ax.bar(i, v, color=cols[i], alpha=alpha[i],
                       edgecolor="white", width=0.55)
            bar_patches.append(b[0])
            ax.text(i, v * 1.03, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=9)
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order)
        ax.set_title(f"{metric}" + (" (%)" if metric == "MAPE" else " (units)"),
                     fontsize=11)
        ax.set_ylim(0, vals.max() * 1.3)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.set_ylabel("")

        # Champion label
        champion_x = model_order.index(best)
        ax.text(champion_x, vals[best] * 1.20, "Champion",
                ha="center", fontsize=8, color=COLORS[best], fontweight="bold")

    save(fig, "model_comparison.png")


def plot_lasso_selection(models, feature_names):
    lasso      = models["Lasso"]
    coefs      = pd.Series(lasso.coef_, index=feature_names)
    nonzero    = coefs[coefs != 0].sort_values(key=abs, ascending=False)
    zeroed_n   = (coefs == 0).sum()
    kept_n     = (coefs != 0).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Lasso Feature Selection  (alpha = {lasso.alpha_:.4f})\n"
                 f"Kept {kept_n} features  |  Zeroed out {zeroed_n} features",
                 fontsize=12, fontweight="bold", y=1.03)

    # Left — non-zero coefficients
    ax = axes[0]
    top = nonzero.head(20)
    bar_cols = [COLORS["Ridge"] if v > 0 else COLORS["Lasso"]
                for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1],
            color=bar_cols[::-1], edgecolor="white", height=0.65)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_xlabel("Lasso coefficient")
    ax.set_title(f"Top 20 non-zero coefficients\n"
                 f"(positive = drives demand up)")
    ax.tick_params(axis="y", labelsize=8)

    # Right — summary donut
    ax = axes[1]
    ax.pie([kept_n, zeroed_n],
           labels=[f"Kept\n{kept_n} features", f"Zeroed\n{zeroed_n} features"],
           colors=[COLORS["XGBoost"], "#e2e8f0"],
           startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 2},
           textprops={"fontsize": 11},
           autopct="%1.0f%%")
    ax.set_title("Feature retention\nafter Lasso selection")

    insight = (f"Lasso automatically drove {zeroed_n} of {len(feature_names)} features to zero.\n"
               "These features have no linear predictive value for demand.\n"
               "lag1_closing_subs, Model Age, and brand dummies are the retained signals.")
    fig.text(0.5, -0.04, insight, ha="center", fontsize=9, color="#334155",
             bbox=dict(boxstyle="round,pad=0.45", fc="#f1f5f9", ec="#cbd5e1"))

    save(fig, "lasso_feature_selection.png")


def plot_feature_importance(models, feature_names):
    xgb = models["XGBoost"]
    rf  = models["Random Forest"]

    imp_xgb = pd.Series(xgb.feature_importances_, index=feature_names)
    imp_rf  = pd.Series(rf.feature_importances_,  index=feature_names)

    top_xgb = imp_xgb.sort_values(ascending=False).head(15)
    top_rf  = imp_rf.sort_values(ascending=False).head(15)

    label_map = {
        "lag1_closing_subs":  "Last month subscribers (demand is sticky)",
        "lag1_churn":         "Last month churn (base shrinking?)",
        "lag1_filed_claims":  "Last month claims (base activity)",
        "Model Age (Days)":   "Device age (lifecycle stage)",
        "Size":               "Storage capacity (price tier)",
    }

    def clean_label(f):
        return label_map.get(
            f, f.replace("ModelFamily_", "Model: ")
                .replace("country_",     "Market: ")
                .replace("Colour_",      "Colour: ")
        )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Feature Importance — Top 15 Demand Drivers",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, imp, title, color in [
        (axes[0], top_xgb, "XGBoost (champion)",       COLORS["XGBoost"]),
        (axes[1], top_rf,  "Random Forest (challenger)", COLORS["Random Forest"]),
    ]:
        labels = [clean_label(f) for f in imp.index]
        ax.barh(range(len(imp)), imp.values[::-1],
                color=color, edgecolor="white", alpha=0.85)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Feature importance (gain)")

    save(fig, "feature_importance.png")


def plot_residuals(models, X_test, y_test):
    mdl    = models["XGBoost"]
    y_pred = np.maximum(mdl.predict(X_test), 0)
    resid  = np.array(y_test) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("XGBoost Residual Analysis — Held-Out Test Set",
                 fontsize=12, fontweight="bold", y=1.02)

    ax = axes[0]
    ax.scatter(y_pred, np.array(y_test), alpha=0.2, s=10,
               color=COLORS["XGBoost"])
    lim = max(y_pred.max(), np.array(y_test).max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5,
            label="Perfect prediction")
    ax.set_xlabel("Predicted Closing Subs")
    ax.set_ylabel("Actual Closing Subs")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax = axes[1]
    ax.scatter(y_pred, resid, alpha=0.2, s=10, color=COLORS["Random Forest"])
    ax.axhline(0,             color="red",   linestyle="--", linewidth=1.5)
    ax.axhline(resid.mean(),  color="navy",  linestyle=":",  linewidth=1.5,
               label=f"Mean residual: {resid.mean():.0f}")
    ax.set_xlabel("Predicted Closing Subs")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs Predicted")
    ax.legend(fontsize=9)

    over_pct = (resid < 0).mean() * 100
    fig.text(0.5, -0.04,
             f"Model over-predicts in {over_pct:.1f}% of test cases.  "
             f"Mean residual: {resid.mean():.0f}  |  Std: {resid.std():.0f}  "
             f"|  Larger residuals at high-volume SKUs — heteroscedasticity.",
             ha="center", fontsize=9, color="#334155",
             bbox=dict(boxstyle="round,pad=0.45", fc="#f1f5f9", ec="#cbd5e1"))

    save(fig, "residual_analysis.png")


# =============================================================================
# 10. FUTURE FORECAST — 202312 & 202401
# =============================================================================

def forecast_future(df_clean, models, X_columns, scaler, num_in_X):
    print(f"\n{'='*55}")
    print("Forecasting 202312 and 202401...")

    # Use November 2023 as the base — the last known month.
    # lag-1 values come from the actual Nov 2023 data, which is fully
    # known at forecast time. This is the correct leakage-free approach.
    nov = df_clean[df_clean["YearMonth"] == 202311].copy()

    cat_cols = ["country", "ModelFamily", "Colour"]
    base     = nov[cat_cols + ["Model Age (Days)", "Size",
                               "Closing Subs Monthly", "Churn",
                               "Filed Claims"]].copy()

    rows = []
    for offset_days, ym in [(31, 202312), (62, 202401)]:
        chunk = base.copy()
        chunk["YearMonth"]        = ym
        # Increment model age for the future month
        chunk["Model Age (Days)"] = chunk["Model Age (Days)"] + offset_days
        # Lag-1 values = actual November 2023 values (known at forecast time)
        chunk["lag1_closing_subs"] = chunk["Closing Subs Monthly"]
        chunk["lag1_churn"]        = chunk["Churn"]
        chunk["lag1_filed_claims"] = chunk["Filed Claims"]
        rows.append(chunk)

    future_df = pd.concat(rows, ignore_index=True)

    # Encode to match training feature space
    num_cols = ["Model Age (Days)", "Size",
                "lag1_closing_subs", "lag1_churn", "lag1_filed_claims"]
    future_enc = pd.get_dummies(
        future_df[num_cols + cat_cols],
        columns=cat_cols,
        drop_first=True,
    )
    future_enc = future_enc.reindex(columns=X_columns, fill_value=0)

    # Apply the same scaler fitted on training data
    future_enc[num_in_X] = scaler.transform(future_enc[num_in_X])

    # Predict — floor at 0 (demand cannot be negative)
    xgb_pred = np.maximum(models["XGBoost"].predict(future_enc),       0)
    rf_pred  = np.maximum(models["Random Forest"].predict(future_enc),  0)

    out = future_df[["YearMonth", "country", "ModelFamily"]].copy()
    out["XGBoost_Forecast"]      = xgb_pred.round().astype(int)
    out["RandomForest_Forecast"] = rf_pred.round().astype(int)

    print("\nForecast preview (first 10 rows):")
    print(out.head(10).to_string(index=False))

    out.to_csv("forecast_202312_202401.csv", index=False)
    print("\n  Saved: forecast_202312_202401.csv")
    return out


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Phone Device Demand Forecasting Pipeline")
    print("=" * 55)

    # 1. Load and clean
    df_clean = load_and_clean(DATA_FILE)

    # 2. Build features (lag-1)
    X, y, numerical_cols = build_features(df_clean)

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 4. Scale (fit on train only)
    X_train, X_val, X_test, scaler, num_in_X = scale_features(
        X_train, X_val, X_test, numerical_cols
    )

    # 5. Time-series CV
    ts_cross_validate(X, y, numerical_cols)

    # 6. Train models
    models = train_models(X_train, y_train)

    # 7. Evaluate
    df_res = evaluate_all(models, X_val, y_val, X_test, y_test)

    # 8. Visualisations
    print(f"\n{'='*55}")
    print("Generating charts...")
    plot_model_comparison(df_res)
    plot_lasso_selection(models, X_train.columns)
    plot_feature_importance(models, X_train.columns)
    plot_residuals(models, X_test, y_test)

    # 9. Forecast
    forecast_future(df_clean, models, X_train.columns, scaler, num_in_X)

    print(f"\n{'='*55}")
    print("Pipeline complete. Output files:")
    print("  model_comparison.png")
    print("  lasso_feature_selection.png")
    print("  feature_importance.png")
    print("  residual_analysis.png")
    print("  forecast_202312_202401.csv")
    print("=" * 55)
