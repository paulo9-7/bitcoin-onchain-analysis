"""
Build the merged daily feature table: Bitcoin block metrics + BTC-USD price.

Smoothing: centered Gaussian (scipy.ndimage.gaussian_filter1d) instead of a
trailing moving average. Chosen because this is descriptive/insight work, not
prediction — a centered filter has no lag and gives a cleaner trend line.
NOTE: it is non-causal (each point uses days on both sides), so do NOT reuse
these smoothed columns as inputs to a predictive model without switching to a
causal filter first.

Fixes / changes over the moving-average version:
  - Gaussian smoothing via a NaN-aware helper (raw gaussian_filter1d spreads
    NaNs across a whole neighbourhood; the helper masks and renormalises).
  - Sorts frames before smoothing (correctness, not luck).
  - Snapshots columns before the smoothing loop (no mutate-while-iterate).
  - Validates that expected columns exist, with a clear error otherwise.

Flagged but intentionally unchanged (methodology decisions):
  - `difficulty` is summed as well as averaged; only the mean is meaningful.
  - Left-merging onto BTC drops non-trading days; a centered filter spanning a
    weekend gap behaves slightly differently on the two date grids.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

BASE = Path("PATH/Bitcoin_Redo")

BLOCKS_FILE = BASE / "tsv_files/Bitcoin_Combined_20191111_20201109.tsv"
BTC_FILE = BASE / "BTC-USD.tsv"
OUTPUT_FILE = BASE / "BTC-USD_df2.tsv"   # read by 03/04/05

# Gaussian standard deviation in days. Larger = smoother.
# Roughly comparable to the old 20-day boxcar: a width-N moving average has
# std ~ N/sqrt(12), so window 20 ~ sigma 5-6. Tune to taste.
SIGMA = 5.0
EDGE_MODE = "nearest"  # how the filter treats the ends of the series

# Added candidate predictors:
#   cdd_total       - coin-days destroyed (older-coin movement / holder activity)
#   generation_usd  - block subsidy alone (reward_usd = generation_usd + fee_total_usd)
METRICS = ["reward_usd", "transaction_count", "fee_total_usd", "difficulty",
           "cdd_total", "generation_usd"]


def gaussian_smooth(series, sigma=SIGMA, mode=EDGE_MODE):
    """Centered Gaussian smoothing that tolerates NaNs.

    Plain gaussian_filter1d turns any NaN into a hole that contaminates every
    point within ~sigma of it. Here we smooth the data (NaNs->0) and a validity
    mask separately, then divide, so NaNs are ignored rather than spread.
    """
    values = series.to_numpy(dtype=float)
    nan_mask = np.isnan(values)
    if not nan_mask.any():
        return gaussian_filter1d(values, sigma=sigma, mode=mode)

    filled = np.where(nan_mask, 0.0, values)
    weight = (~nan_mask).astype(float)
    smoothed = gaussian_filter1d(filled, sigma=sigma, mode=mode)
    norm = gaussian_filter1d(weight, sigma=sigma, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smoothed / norm
    out[norm == 0] = np.nan
    return out


def load_blocks(path):
    df = pd.read_csv(path, sep="\t")
    missing = {"time", *METRICS} - set(df.columns)
    if missing:
        raise KeyError(f"Blocks file missing expected columns: {sorted(missing)}")
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    return df


def build_daily_metrics(blocks):
    # groupby sorts by date, so the smoothing below sees ascending order.
    daily = blocks.groupby("date")[METRICS].agg(["sum", "mean"])
    daily.columns = [f"{col}_{stat}" for col, stat in daily.columns]

    for metric in METRICS:
        if metric != "transaction_count":
            daily[f"{metric}_per_transaction"] = (
                daily[f"{metric}_sum"] / daily["transaction_count_sum"]
            )

    # Snapshot columns first: we are adding new columns inside the loop.
    for col in list(daily.columns):
        daily[f"gaussian_{col}"] = gaussian_smooth(daily[col])

    return daily.reset_index()


def load_btc(path):
    df = pd.read_csv(path, sep="\t", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if "Adj Close" not in df.columns:
        raise KeyError("BTC file missing 'Adj Close' column")
    df["date"] = df["Date"].dt.date
    df["gaussian_BTC_USD"] = gaussian_smooth(df["Adj Close"])
    return df


def main():
    blocks = load_blocks(BLOCKS_FILE)
    daily_metrics = build_daily_metrics(blocks)
    btc = load_btc(BTC_FILE)

    final_data = btc.merge(daily_metrics, on="date", how="left")
    final_data = final_data.drop(columns=["date"])
    final_data.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Wrote {OUTPUT_FILE} ({len(final_data):,} rows, {final_data.shape[1]} cols)")


if __name__ == "__main__":
    main()
