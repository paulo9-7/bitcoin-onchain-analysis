"""
Correlation heatmaps for the Bitcoin feature table.

READ THIS BEFORE TRUSTING THE NUMBERS
-------------------------------------
Correlating the *smoothed* columns (gaussian_* / old moving_avg_*) pushes
correlations artificially toward +/-1: smoothing removes the noise that
distinguishes two series and leaves their shared trend, and it destroys the
independence of observations so any significance is meaningless. Correlating
trending *levels* is also spurious (difficulty and price both rise over the
window, so they correlate by construction).

More honest views, all one-line switches below:
  - correlate the RAW daily series, not the smoothed ones;
  - correlate day-over-day CHANGES (use_diff=True) instead of levels;
  - use method="spearman" (rank) to reduce sensitivity to trend/outliers.

The smoothed matrix is kept only to reproduce the original analysis.

Includes the recommended candidate predictors:
  cdd_total       - coin-days destroyed (older-coin movement / holder activity)
  generation_usd  - block subsidy alone (reward_usd = generation_usd + fee_total_usd)
NOTE: these columns only exist if 02_build_features.py includes them in METRICS.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/Users/pauloconnor/Desktop/py.scripts/Bitcoin tsv/Bitcoin_Redo")
DATA_FILE = BASE / "BTC-USD_df2.tsv"   # output of 02_build_features.py

# Raw daily block metrics + price (the trustworthy inputs).
RAW_LABELS = {
    "reward_usd_sum": "Total Reward/Day USD",
    "reward_usd_mean": "Avg Reward/Day USD",
    "transaction_count_sum": "Total Trans Count/Day",
    "transaction_count_mean": "Avg Trans Count/Day",
    "fee_total_usd_sum": "Total Fee/Day",
    "fee_total_usd_mean": "Avg Fee/Day",
    "difficulty_sum": "Total Difficulty/Day",
    "difficulty_mean": "Avg Difficulty/Day",
    "reward_usd_per_transaction": "Avg Reward/Trans",
    "fee_total_usd_per_transaction": "Avg Fee/Trans",
    "difficulty_per_transaction": "Avg Difficulty/Trans",
    # Added candidate predictors:
    "cdd_total_sum": "Total CDD/Day",
    "cdd_total_mean": "Avg CDD/Block",
    "generation_usd_sum": "Total Subsidy/Day",
    "generation_usd_mean": "Avg Subsidy/Block",
    "Adj Close": "Bitcoin USD/Day",
}

# Smoothed columns may be prefixed gaussian_ (new) or moving_avg_ (old).
SMOOTH_PREFIXES = ["gaussian_", "moving_avg_"]


def smoothed_label_map(df):
    """Find whatever smoothed columns actually exist and label them.

    Returns {} if none are present (e.g. the feature table predates smoothing
    or wasn't regenerated) so the caller can warn instead of drawing a blank.
    """
    for prefix in SMOOTH_PREFIXES:
        smoothed = [c for c in df.columns if c.startswith(prefix)]
        if not smoothed:
            continue
        labels = {}
        for col in smoothed:
            base = col[len(prefix):]
            if base.replace(" ", "_").upper().startswith("BTC"):
                labels[col] = "Bitcoin USD"
            else:
                labels[col] = RAW_LABELS.get(base, base)
        return labels
    return {}


def correlation_heatmap(df, col_map, title, method="pearson",
                        use_diff=False, mask_upper=True):
    cols = [c for c in col_map if c in df.columns]
    missing = [c for c in col_map if c not in df.columns]
    if missing:
        print(f"[warn] {title}: skipping missing columns: {missing}")

    data = df[cols]
    if use_diff:
        data = data.diff()  # correlate day-over-day changes, not levels
    corr = data.corr(method=method).rename(columns=col_map, index=col_map)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1) if mask_upper else None
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5,
                cbar_kws={"shrink": .5}, ax=ax)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig, ax


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")

    # 1. Reproduction of the original: correlations of the smoothed series.
    gauss_labels = smoothed_label_map(df)
    if gauss_labels:
        correlation_heatmap(df, gauss_labels,
                            "Correlation Matrix — Smoothed (inflated; see note)")
    else:
        print(
            "[warn] No smoothed columns (gaussian_* / moving_avg_*) found in "
            f"{DATA_FILE.name}. Re-run 02_build_features.py, or point DATA_FILE "
            "at the file that contains the smoothed columns. Skipping matrix 1."
        )

    # 2. Raw daily levels (Pearson).
    correlation_heatmap(df, RAW_LABELS,
                        "Correlation Matrix — Raw Daily Values (Pearson)")

    # 3. Recommended: day-over-day changes (Spearman), far less trend-driven.
    correlation_heatmap(df, RAW_LABELS,
                        "Correlation Matrix — Daily Changes (Spearman)",
                        method="spearman", use_diff=True)

    plt.show()


if __name__ == "__main__":
    main()