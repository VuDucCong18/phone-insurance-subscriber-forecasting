"""
Exploratory Data Analysis — Phone Device Demand Forecasting
===========================================================
Author : Vu Duc Cong
Dataset: Worksheet in ML_Assignment_2023.xlsx

Story arc (4 questions):
    Q1. What does the data look like?
        Chart 1 — Dataset overview + missing values

    Q2. What does demand look like?
        Chart 2 — Target distribution (right skew)
        Chart 3 — Demand over time by brand
        Chart 4 — Demand by market + top 10 models

    Q3. What drives the lifecycle of a single model?
        Chart 5 — Lifecycle curve (demand vs model age, binned)
        Chart 6 — Single model story: iPhone 12 Pro Max California
                  (Closing Subs + Churn over 58 months)

    Q4. What are the relationships between features and target?
        Chart 7 — Correlation heatmap framed around the size-effect problem
        Chart 8 — Lag-1 relationships (why lagging fixes the leakage)

Outputs:
    eda_01_overview.png
    eda_02_distribution.png
    eda_03_demand_over_time.png
    eda_04_market_and_top_models.png
    eda_05_lifecycle_curve.png
    eda_06_single_model_story.png
    eda_07_correlations.png
    eda_08_lag_relationships.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

PALETTE  = {"Apple": "#4e79a7", "Samsung": "#f28e2b", "Oppo": "#59a14f"}
MARKET_C = {"California": "#9b59b6", "Nevada": "#e67e22", "Texas": "#1abc9c"}
BLUE     = "#4e79a7"
RED      = "#e05c5c"
GREEN    = "#59a14f"
AMBER    = "#f28e2b"

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

DATA_FILE = "Worksheet in ML_Assignment_2023.xlsx"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw():
    return pd.read_excel(DATA_FILE)


def load_clean():
    df = load_raw()
    zero_cols = [
        "Claims", "Filed Claims", "Claims Swap", "Claims Replacement",
        "IR Rate Swap", "IR Rate Replacement", "IR Rate Monthly",
        "Churn Rate", "Churn",
    ]
    df[zero_cols] = df[zero_cols].fillna(0)
    df["IR Rate Swap"]        = (df["Claims Swap"]        / df["Closing Subs Monthly"]) * 100
    df["IR Rate Replacement"] = (df["Claims Replacement"] / df["Closing Subs Monthly"]) * 100
    df["IR Rate Monthly"]     = df["IR Rate Swap"] + df["IR Rate Replacement"]
    df["Churn Rate"]          = (df["Churn"] / df["Closing Subs Monthly"]) * 100
    df["Date"]  = pd.to_datetime(df["YearMonth"].astype(str), format="%Y%m")
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  Saved: {filename}")


def note(fig, text, y=-0.03):
    fig.text(0.5, y, text, ha="center", fontsize=9, color="#334155",
             bbox=dict(boxstyle="round,pad=0.45", fc="#f1f5f9", ec="#cbd5e1"))


# =============================================================================
# CHART 1 — DATASET OVERVIEW + MISSING VALUES
# Question: What does the data look like?
# =============================================================================

def chart_01_overview(df_raw):
    missing  = df_raw.isnull().sum()
    missing  = missing[missing > 0].sort_values(ascending=False)
    pct      = (missing / len(df_raw) * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q1 — What does the data look like?", fontsize=15,
                 fontweight="bold", y=1.03)

    # Left — missing values bar chart
    ax = axes[0]
    colors = ["#ef4444" if p > 40 else "#f97316" if p > 10 else "#facc15"
              for p in pct.values]
    bars = ax.barh(missing.index[::-1], pct.values[::-1],
                   color=colors[::-1], edgecolor="white", height=0.6)
    ax.bar_label(bars, labels=[f"{p}%" for p in pct.values[::-1]],
                 padding=4, fontsize=9)
    ax.set_xlabel("% of rows missing")
    ax.set_title("Missing Values by Column", fontsize=12)
    ax.set_xlim(0, pct.max() * 1.3)
    ax.axvline(0, color="#334155", linewidth=0.8)

    insight = ("Claims columns: ~49% missing\n"
               "→ Not truly missing. These are months\n"
               "   with zero claim events. Fill with 0.")
    ax.text(0.55, 0.12, insight, transform=ax.transAxes, fontsize=9,
            color="#1e40af",
            bbox=dict(boxstyle="round,pad=0.4", fc="#dbeafe", ec="#93c5fd"))

    # Right — dataset card
    ax = axes[1]
    ax.axis("off")

    stats = [
        ("Total rows",             f"{len(df_raw):,}"),
        ("Total columns",          f"{df_raw.shape[1]}"),
        ("Date range",             f"Jan 2019 – Nov 2023  (58 months)"),
        ("Brands",                 "Apple, Samsung, Oppo"),
        ("Markets",                "California, Nevada, Texas"),
        ("Unique model families",  f"{df_raw['ModelFamily'].nunique()}"),
        ("Total subscriber-months",f"{df_raw['Closing Subs Monthly'].sum():,.0f}"),
        ("Target variable",        "Closing Subs Monthly"),
    ]

    ax.text(0.0, 0.98, "Dataset at a Glance", fontsize=13,
            fontweight="bold", transform=ax.transAxes, va="top",
            color="#0f172a")

    y = 0.85
    for label, value in stats:
        ax.text(0.0, y, f"{label}:", fontsize=10.5, color="#64748b",
                transform=ax.transAxes, va="top")
        ax.text(0.52, y, value, fontsize=10.5, fontweight="500",
                color="#0f172a", transform=ax.transAxes, va="top")
        y -= 0.115

    decision = ("Decision: fill missing claims/churn with 0 (no-event months),\n"
                "then recalculate all rates from source columns.")
    ax.text(0.0, y - 0.02, decision, fontsize=9, color="#166534",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", fc="#dcfce7", ec="#86efac"))

    save(fig, "eda_01_overview.png")


# =============================================================================
# CHART 2 — TARGET DISTRIBUTION
# Question: What does demand look like? (shape of the target)
# =============================================================================

def chart_02_distribution(df):
    y = df["Closing Subs Monthly"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Q2 — What does demand look like?  Distribution of Closing Subs Monthly",
                 fontsize=13, fontweight="bold", y=1.03)

    # Histogram raw
    ax = axes[0]
    ax.hist(y, bins=70, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(y.median(), color=RED,   linestyle="--", linewidth=1.8,
               label=f"Median: {y.median():,.0f}")
    ax.axvline(y.mean(),   color=AMBER, linestyle=":",  linewidth=1.8,
               label=f"Mean:   {y.mean():,.0f}")
    ax.set_xlabel("Closing Subs Monthly")
    ax.set_ylabel("Number of records")
    ax.set_title("Raw distribution\n(heavily right-skewed)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))

    insight_text = ("Mean (1,860) >> Median (928)\n"
                    "Long tail up to 41,031\n"
                    "→ A few flagship models\n"
                    "   dominate total volume")
    ax.text(0.55, 0.65, insight_text, transform=ax.transAxes, fontsize=8.5,
            color="#1e40af",
            bbox=dict(boxstyle="round,pad=0.35", fc="#dbeafe", ec="#93c5fd"))

    # Histogram log scale
    ax = axes[1]
    ax.hist(np.log1p(y), bins=70, color=GREEN, edgecolor="white", alpha=0.85)
    ax.set_xlabel("log(1 + Closing Subs Monthly)")
    ax.set_ylabel("Number of records")
    ax.set_title("Log-transformed\n(approximately normal)")
    ax.text(0.05, 0.88,
            "After log transform the\ndistribution is symmetric\n"
            "→ tree models handle raw\n   skew better than linear",
            transform=ax.transAxes, fontsize=8.5, color="#166534",
            bbox=dict(boxstyle="round,pad=0.35", fc="#dcfce7", ec="#86efac"))

    # Box plot by brand
    ax = axes[2]
    order = (df.groupby("Make")["Closing Subs Monthly"]
               .median().sort_values(ascending=False).index)
    sns.boxplot(data=df, x="Make", y="Closing Subs Monthly",
                order=order, palette=PALETTE, ax=ax,
                flierprops={"marker": ".", "alpha": 0.25, "markersize": 3})
    ax.set_yscale("log")
    ax.set_title("Spread by brand\n(log y-axis)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.text(0.05, 0.88,
            "All brands show wide\nspread — low-volume niche\nSKUs alongside flagships",
            transform=ax.transAxes, fontsize=8.5, color="#334155",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f1f5f9", ec="#cbd5e1"))

    note(fig,
         "Right skew justifies using tree-based models (Random Forest, XGBoost) over linear regression.  "
         "Linear models assume normally distributed errors — this data violates that assumption.")
    save(fig, "eda_02_distribution.png")


# =============================================================================
# CHART 3 — DEMAND OVER TIME BY BRAND
# Question: How has total demand evolved?
# =============================================================================

def chart_03_demand_over_time(df):
    monthly      = df.groupby("Date")["Closing Subs Monthly"].sum().reset_index()
    monthly_make = (df.groupby(["Date", "Make"])["Closing Subs Monthly"]
                    .sum().reset_index())

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Q2 — How has demand evolved over time?",
                 fontsize=13, fontweight="bold", y=1.02)

    # Total trend
    ax = axes[0]
    ax.fill_between(monthly["Date"], monthly["Closing Subs Monthly"],
                    alpha=0.12, color=BLUE)
    ax.plot(monthly["Date"], monthly["Closing Subs Monthly"],
            color=BLUE, linewidth=2.2)
    ax.set_title("Total active subscriber-months across all brands & markets",
                 fontsize=11)
    ax.set_ylabel("Total Closing Subs")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

    peak = monthly.loc[monthly["Closing Subs Monthly"].idxmax()]
    ax.annotate(
        f"Peak: {peak['Closing Subs Monthly']:,.0f}\n({peak['Date'].strftime('%b %Y')})",
        xy=(peak["Date"], peak["Closing Subs Monthly"]),
        xytext=(-80, -45), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2),
        fontsize=9, color="#334155",
    )
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-09-01"),
               alpha=0.07, color=RED, label="COVID-19 period")
    ax.axvline(pd.Timestamp("2021-06-01"), color="#94a3b8",
               linestyle="--", linewidth=1, label="Plateau begins (~Jun 2021)")
    ax.legend(fontsize=9)

    # By brand
    ax = axes[1]
    for make, grp in monthly_make.groupby("Make"):
        grp = grp.sort_values("Date")
        ax.fill_between(grp["Date"], grp["Closing Subs Monthly"],
                        alpha=0.08, color=PALETTE[make])
        ax.plot(grp["Date"], grp["Closing Subs Monthly"],
                label=make, color=PALETTE[make], linewidth=2.2)
    ax.set_title("Monthly demand split by brand", fontsize=11)
    ax.set_ylabel("Total Closing Subs")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
    ax.legend(fontsize=10)

    note(fig,
         "Growth from 2019 → mid-2021 reflects a maturing subscriber base.  "
         "Plateau from mid-2021 onward signals market saturation.  "
         "→ YearMonth and Model Age are needed as time signals in the model.",
         y=-0.02)
    save(fig, "eda_03_demand_over_time.png")


# =============================================================================
# CHART 4 — DEMAND BY MARKET + TOP 10 MODELS
# Question: Who and what drives demand?
# =============================================================================

def chart_04_market_and_top_models(df):
    top10 = (df.groupby(["ModelFamily", "Make"])["Closing Subs Monthly"]
               .sum().reset_index()
               .sort_values("Closing Subs Monthly", ascending=False)
               .head(10))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Q2 — Who drives demand?  Markets & top model families",
                 fontsize=13, fontweight="bold", y=1.03)

    # Market breakdown — grouped bar (total + mean)
    ax = axes[0]
    market_stats = (df.groupby("country")["Closing Subs Monthly"]
                    .agg(["sum", "mean"])
                    .sort_values("sum", ascending=False)
                    .reset_index())
    x     = np.arange(len(market_stats))
    width = 0.38
    bars1 = ax.bar(x - width/2, market_stats["sum"] / 1e6,
                   width, label="Total (M)", edgecolor="white",
                   color=[MARKET_C[c] for c in market_stats["country"]], alpha=0.9)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, market_stats["mean"],
                    width, label="Avg per SKU/month", edgecolor="white",
                    color=[MARKET_C[c] for c in market_stats["country"]], alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(market_stats["country"])
    ax.set_ylabel("Total subscriber-months (Millions)")
    ax2.set_ylabel("Average Closing Subs per record")
    ax.set_title("Demand by market\n(total volume vs average per SKU)")
    ax.bar_label(bars1, labels=[f"{v:.1f}M" for v in market_stats["sum"]/1e6],
                 padding=3, fontsize=9)
    ax2.bar_label(bars2, labels=[f"{int(v):,}" for v in market_stats["mean"]],
                  padding=3, fontsize=9)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.text(0.04, 0.78,
            "Nevada: highest avg per SKU\nCalifornia: most records",
            transform=ax.transAxes, fontsize=8.5, color="#334155",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f1f5f9", ec="#cbd5e1"))

    # Top 10 models horizontal bar
    ax = axes[1]
    colors = [PALETTE[m] for m in top10["Make"]]
    bars = ax.barh(top10["ModelFamily"][::-1],
                   top10["Closing Subs Monthly"][::-1] / 1e6,
                   color=colors[::-1], edgecolor="white", height=0.65)
    ax.set_xlabel("Total subscriber-months (Millions)")
    ax.set_title("Top 10 model families\nby total demand")
    ax.bar_label(bars, labels=[f"{v:.2f}M" for v in top10["Closing Subs Monthly"][::-1]/1e6],
                 padding=3, fontsize=8.5)
    ax.set_xlim(0, top10["Closing Subs Monthly"].max()/1e6 * 1.28)
    handles = [Patch(fc=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    note(fig,
         "Nevada leads in average per-SKU volume despite similar record count to other markets.  "
         "iPhone 12 Pro Max is the single largest model (2.3M subscriber-months).  "
         "→ Both country and ModelFamily are essential categorical features.")
    save(fig, "eda_04_market_and_top_models.png")


# =============================================================================
# CHART 5 — LIFECYCLE CURVE
# Question: How does demand evolve over a model's life?
# =============================================================================

def chart_05_lifecycle_curve(df):
    # Bin model age into 90-day buckets, compute median demand
    df_pos = df[df["Model Age (Days)"] >= 0].copy()
    df_pos["Age Bucket"] = (df_pos["Model Age (Days)"] // 90) * 90
    lifecycle = (df_pos.groupby("Age Bucket")["Closing Subs Monthly"]
                 .agg(["median", "mean", "count"])
                 .reset_index()
                 .query("`count` >= 20"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Q3 — What drives the lifecycle of a model?  Demand vs device age",
                 fontsize=13, fontweight="bold", y=1.03)

    # Lifecycle curve
    ax = axes[0]
    ax.fill_between(lifecycle["Age Bucket"] / 365,
                    lifecycle["median"],
                    alpha=0.15, color=BLUE)
    ax.plot(lifecycle["Age Bucket"] / 365, lifecycle["median"],
            color=BLUE, linewidth=2.2, label="Median demand")
    ax.plot(lifecycle["Age Bucket"] / 365, lifecycle["mean"],
            color=AMBER, linewidth=1.5, linestyle="--", label="Mean demand")

    peak_idx = lifecycle["median"].idxmax()
    peak_age = lifecycle.loc[peak_idx, "Age Bucket"] / 365
    peak_val = lifecycle.loc[peak_idx, "median"]
    ax.annotate(f"Peak ~{peak_age:.1f} years\nafter launch",
                xy=(peak_age, peak_val),
                xytext=(30, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2),
                fontsize=9, color="#334155")

    ax.axvspan(0, 1.5, alpha=0.05, color=GREEN, label="Growth phase (0–18 months)")
    ax.axvspan(1.5, lifecycle["Age Bucket"].max()/365,
               alpha=0.05, color=RED, label="Decay phase (18+ months)")

    ax.set_xlabel("Device age (years since launch)")
    ax.set_ylabel("Median Closing Subs Monthly")
    ax.set_title("Lifecycle curve — all models combined\n(binned into 90-day windows)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9)

    # Scatter sample (individual records)
    ax = axes[1]
    sample = df_pos.sample(min(5000, len(df_pos)), random_state=42)
    colors_s = [PALETTE[m] for m in sample["Make"]]
    ax.scatter(sample["Model Age (Days)"] / 365,
               sample["Closing Subs Monthly"],
               c=colors_s, alpha=0.18, s=8)
    ax.set_xlabel("Device age (years since launch)")
    ax.set_ylabel("Closing Subs Monthly (log scale)")
    ax.set_yscale("log")
    ax.set_title("Individual SKU records\n(each dot = one model, one market, one month)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    handles = [Patch(fc=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=handles, fontsize=9)

    note(fig,
         "Demand follows a predictable arc: rapid growth in the first 6–18 months after launch, "
         "then gradual decline as successors arrive.  "
         "→ Model Age (Days) is a critical feature — it tells the model where in the lifecycle a device currently sits.")
    save(fig, "eda_05_lifecycle_curve.png")


# =============================================================================
# CHART 6 — SINGLE MODEL STORY: iPhone 12 Pro Max California
# Question: How do Closing Subs and Churn relate over time for one real model?
# =============================================================================

def chart_06_single_model_story(df):
    model = (df[(df["ModelFamily"] == "APPLE IPHONE 12 PRO MAX") &
                (df["country"] == "California")]
             .groupby("Date")
             .agg(Subs=("Closing Subs Monthly", "sum"),
                  Churn=("Churn", "sum"),
                  Claims=("Filed Claims", "sum"))
             .reset_index()
             .sort_values("Date"))

    # Mark lifecycle phases
    launch  = pd.Timestamp("2020-10-01")
    peak    = model.loc[model["Subs"].idxmax(), "Date"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Q3 — Single model story: Apple iPhone 12 Pro Max · California\n"
                 "58 months of real data showing the full subscriber lifecycle",
                 fontsize=13, fontweight="bold", y=1.02)

    # Top — Closing Subs
    ax = axes[0]
    ax.fill_between(model["Date"], model["Subs"], alpha=0.12, color=BLUE)
    ax.plot(model["Date"], model["Subs"], color=BLUE, linewidth=2.2,
            label="Closing Subs Monthly")
    ax.axvline(launch, color=GREEN, linestyle="--", linewidth=1.5,
               label=f"Launch ({launch.strftime('%b %Y')})")
    ax.axvline(peak, color=RED, linestyle="--", linewidth=1.5,
               label=f"Peak: {model.loc[model['Subs'].idxmax(),'Subs']:,} subs  "
                     f"({peak.strftime('%b %Y')})")

    ax.axvspan(pd.Timestamp("2019-01-01"), launch,
               alpha=0.05, color=AMBER, label="Pre-launch (pre-orders exist)")
    ax.set_ylabel("Active subscribers")
    ax.set_title("Active subscriber base — grows fast at launch, peaks, then decays steadily")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9, loc="upper left")

    # Bottom — Churn overlaid on Subs scaled
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(model["Date"], model["Churn"], width=20,
           color=RED, alpha=0.55, label="Churn (monthly, left axis)")
    ax2.plot(model["Date"], model["Subs"], color=BLUE,
             linewidth=1.8, linestyle=":", alpha=0.6, label="Closing Subs (right axis)")
    ax.axvline(launch, color=GREEN, linestyle="--", linewidth=1.2)
    ax.axvline(peak,   color=RED,   linestyle="--", linewidth=1.2)

    ax.set_ylabel("Churn (subscribers lost)", color=RED)
    ax2.set_ylabel("Closing Subs Monthly", color=BLUE)
    ax.tick_params(axis="y", labelcolor=RED)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax.set_title("Churn rises as the base grows — then stays elevated during decay phase\n"
                 "Key insight: churn is concurrent (happens in same month as subs count)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))

    note(fig,
         "Churn and Closing Subs move together because both scale with installed base size — "
         "this is the size effect, not a causal relationship.  "
         "→ To use Churn as a feature, we must lag it by 1 month so it becomes a known input at forecast time.",
         y=-0.02)
    save(fig, "eda_06_single_model_story.png")


# =============================================================================
# CHART 7 — CORRELATIONS (framed around the size-effect problem)
# Question: What correlates with demand — and why should we be suspicious?
# =============================================================================

def chart_07_correlations(df):
    num_cols = [
        "Model Age (Days)", "Size",
        "Filed Claims", "Churn",
        "Claims Swap", "Claims Replacement",
        "IR Rate Monthly", "Churn Rate",
        "Closing Subs Monthly",
    ]
    corr = df[num_cols].corr()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Q4 — What are the relationships between features and target?",
                 fontsize=13, fontweight="bold", y=1.03)

    # Heatmap
    ax = axes[0]
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", linewidths=0.4,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.75})
    ax.set_title("Correlation matrix\n(all numerical features)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    # Ranked bar — correlation with target
    ax = axes[1]
    corr_target = (corr["Closing Subs Monthly"]
                   .drop("Closing Subs Monthly")
                   .sort_values(ascending=True))
    bar_colors = [RED if v < 0 else BLUE for v in corr_target.values]
    bars = ax.barh(corr_target.index, corr_target.values,
                   color=bar_colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_title("Correlation with\nClosing Subs Monthly (r)")
    ax.set_xlabel("Pearson r")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
    ax.set_xlim(-0.25, 1.05)

    for label in ["Filed Claims", "Churn"]:
        idx = list(corr_target.index).index(label)
        ax.get_yticklabels()[idx].set_color(RED)
        ax.get_yticklabels()[idx].set_fontweight("bold")

    ax.text(0.35, 0.08,
            "Red labels = concurrent\nvariables (leakage risk)",
            transform=ax.transAxes, fontsize=8.5, color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fee2e2", ec="#fca5a5"))

    # Scatter: Filed Claims vs Closing Subs — coloured by brand
    ax = axes[2]
    sample = df[df["Filed Claims"] > 0].sample(min(4000, len(df)), random_state=42)
    c_scatter = [PALETTE[m] for m in sample["Make"]]
    ax.scatter(sample["Filed Claims"], sample["Closing Subs Monthly"],
               c=c_scatter, alpha=0.22, s=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Filed Claims (log scale)")
    ax.set_ylabel("Closing Subs Monthly (log scale)")
    ax.set_title(f"Filed Claims vs Demand\nr = 0.75 — but why?")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    handles = [Patch(fc=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=handles, fontsize=8)
    ax.text(0.05, 0.78,
            "High r is a SIZE EFFECT:\n"
            "10,000 subs → more claims\n"
            "than 100 subs, always.\n"
            "Not causal — both driven\nby installed base.",
            transform=ax.transAxes, fontsize=8.5, color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fee2e2", ec="#fca5a5"))

    note(fig,
         "Churn (r=0.76) and Filed Claims (r=0.75) look like strong predictors "
         "but both are concurrent — they happen DURING the month we are forecasting.  "
         "Rates (Churn Rate, IR Rate) have near-zero correlation once the size effect is removed.",
         y=-0.03)
    save(fig, "eda_07_correlations.png")


# =============================================================================
# CHART 8 — LAG-1 RELATIONSHIPS
# Question: Does lagging fix the problem? What do the honest relationships look like?
# =============================================================================

def chart_08_lag_relationships(df):
    df_s = df.sort_values(["country", "ModelFamily", "Date"]).copy()
    grp  = df_s.groupby(["country", "ModelFamily"])

    df_s["lag1_subs"]   = grp["Closing Subs Monthly"].shift(1)
    df_s["lag1_churn"]  = grp["Churn"].shift(1)
    df_s["lag1_claims"] = grp["Filed Claims"].shift(1)
    df_s["mom_change"]  = grp["Closing Subs Monthly"].diff()

    df_lag = df_s.dropna(subset=["lag1_subs", "lag1_churn", "lag1_claims"])

    # Correlations before vs after lagging
    raw_corr = {
        "Churn (same month)":        df_s["Churn"].corr(df_s["Closing Subs Monthly"]),
        "Filed Claims (same month)": df_s["Filed Claims"].corr(df_s["Closing Subs Monthly"]),
    }
    lag_corr = {
        "Churn (lag-1)":        df_lag["lag1_churn"].corr(df_lag["Closing Subs Monthly"]),
        "Filed Claims (lag-1)": df_lag["lag1_claims"].corr(df_lag["Closing Subs Monthly"]),
        "Closing Subs (lag-1)": df_lag["lag1_subs"].corr(df_lag["Closing Subs Monthly"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Q4 — Does lagging fix the leakage?  Honest feature relationships",
                 fontsize=13, fontweight="bold", y=1.03)

    # Before vs after correlation comparison
    ax = axes[0]
    labels = ["Churn\n(same month)", "Churn\n(lag-1)",
              "Filed Claims\n(same month)", "Filed Claims\n(lag-1)",
              "Closing Subs\n(lag-1)"]
    values = [raw_corr["Churn (same month)"],
              lag_corr["Churn (lag-1)"],
              raw_corr["Filed Claims (same month)"],
              lag_corr["Filed Claims (lag-1)"],
              lag_corr["Closing Subs (lag-1)"]]
    bar_c = [RED, GREEN, RED, GREEN, BLUE]
    bars = ax.barh(labels[::-1], values[::-1],
                   color=bar_c[::-1], edgecolor="white", height=0.55)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_xlabel("Pearson r with Closing Subs Monthly")
    ax.set_title("Correlation before vs after\nlagging by 1 month")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xlim(-0.1, 1.0)
    ax.text(0.52, 0.06, "Red = concurrent\nGreen/Blue = lagged (valid)",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f1f5f9", ec="#cbd5e1"))

    # Scatter: lag-1 subs vs current subs
    ax = axes[1]
    sample = df_lag.sample(min(4000, len(df_lag)), random_state=42)
    ax.scatter(sample["lag1_subs"], sample["Closing Subs Monthly"],
               alpha=0.2, s=9, color=BLUE)
    lim = max(sample["lag1_subs"].max(), sample["Closing Subs Monthly"].max())
    ax.plot([0, lim], [0, lim], color=RED, linestyle="--",
            linewidth=1.5, label="Perfect persistence")
    ax.set_xlabel("Last month's Closing Subs (lag-1)")
    ax.set_ylabel("This month's Closing Subs")
    ax.set_title(f"Lag-1 Closing Subs vs Current\nr = {lag_corr['Closing Subs (lag-1)']:.2f}  "
                 f"(strongest feature)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9)
    ax.text(0.05, 0.82,
            "Points hug the diagonal:\nlast month's subs is the\nbest single predictor\nof this month's subs",
            transform=ax.transAxes, fontsize=8.5, color="#1e40af",
            bbox=dict(boxstyle="round,pad=0.35", fc="#dbeafe", ec="#93c5fd"))

    # Distribution of month-over-month change
    ax = axes[2]
    mom = df_lag["mom_change"].dropna()
    ax.hist(mom.clip(-3000, 3000), bins=80, color=AMBER,
            edgecolor="white", alpha=0.85)
    ax.axvline(0, color=RED, linewidth=2, linestyle="--", label="No change")
    ax.axvline(mom.median(), color=BLUE, linewidth=1.5, linestyle=":",
               label=f"Median: {mom.median():.0f}")
    ax.set_xlabel("Month-over-month change in Closing Subs\n(clipped to ±3,000 for readability)")
    ax.set_ylabel("Count")
    ax.set_title("How much does demand shift\nmonth to month?")
    ax.legend(fontsize=9)
    ax.text(0.05, 0.80,
            f"Most months: small change\n"
            f"25th pct: {mom.quantile(0.25):,.0f}\n"
            f"75th pct: +{mom.quantile(0.75):,.0f}\n"
            f"Demand is sticky — lag-1\nis a strong baseline",
            transform=ax.transAxes, fontsize=8.5, color="#334155",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f1f5f9", ec="#cbd5e1"))

    note(fig,
         "Lag-1 Closing Subs (r=0.67) is the single strongest valid feature — last month's base predicts this month's base.  "
         "Lagged Churn (r=0.46) and lagged Claims (r=0.46) add genuine signal without leakage.  "
         "→ These three lagged features replace the concurrent versions in the pipeline.",
         y=-0.03)
    save(fig, "eda_08_lag_relationships.png")


# =============================================================================
# MAIN
# =============================================================================

def run_eda():
    print("=" * 60)
    print("EDA — Phone Device Demand Forecasting")
    print("=" * 60)

    df_raw = load_raw()
    df     = load_clean()
    print(f"  Loaded: {len(df):,} rows × {df.shape[1]} columns\n")

    print("[1/8] Q1 — Dataset overview + missing values...")
    chart_01_overview(df_raw)

    print("[2/8] Q2 — Target distribution...")
    chart_02_distribution(df)

    print("[3/8] Q2 — Demand over time...")
    chart_03_demand_over_time(df)

    print("[4/8] Q2 — Demand by market + top models...")
    chart_04_market_and_top_models(df)

    print("[5/8] Q3 — Lifecycle curve...")
    chart_05_lifecycle_curve(df)

    print("[6/8] Q3 — Single model story (iPhone 12 Pro Max)...")
    chart_06_single_model_story(df)

    print("[7/8] Q4 — Correlations + size-effect problem...")
    chart_07_correlations(df)

    print("[8/8] Q4 — Lag-1 relationships...")
    chart_08_lag_relationships(df)

    print("\n" + "=" * 60)
    print("EDA complete. Key decisions justified:")
    print("  1. Fill missing claims with 0  (no-event months, not gaps)")
    print("  2. Use tree models             (right-skewed target)")
    print("  3. Include YearMonth + Age     (growth plateau from mid-2021)")
    print("  4. Include country + ModelFamily (structural level differences)")
    print("  5. Lag Churn + Claims by 1 month (concurrent = data leakage)")
    print("  6. Lag-1 Closing Subs as feature (r=0.67, strongest valid signal)")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
