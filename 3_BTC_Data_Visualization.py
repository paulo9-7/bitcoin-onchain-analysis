"""
Exploratory plots: BTC price, per-block metrics, and per-transaction ratios
over time, each with a centered-Gaussian trend line and the 2020 halving marked.

Improvements over the original:
  - Blocks file is read and grouped ONCE, serving every plot (was read/grouped
    per metric, across two separate scripts).
  - Smoothing uses the same centered Gaussian as the feature build (consistency).
  - Shared plotting/formatting helper (no duplicated boilerplate).
  - Halving line is labelled and defined as a named constant.
  - Clean date axis via AutoDateLocator + ConciseDateFormatter.
  - Explicit fig/ax; figures can be saved for the write-up.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

BASE = Path("PATH/Bitcoin_Redo")
MERGED_FILE = BASE / "BTC-USD_df2.tsv"
BLOCKS_FILE = Path(
    "PATH"
    "tsv_files/Bitcoin_Combined_20191111_20201109.tsv"
)
FIG_DIR = BASE / "figures"          # set SAVE_FIGS=True to write PNGs here
SAVE_FIGS = False

SIGMA = 5.0                          # Gaussian smoothing strength, in days
EDGE_MODE = "nearest"
HALVING_DATE = pd.Timestamp("2020-05-11")  # 3rd Bitcoin halving

# Added candidate predictors from the blocks schema:
#   cdd_total       - coin-days destroyed (older-coin movement / holder activity)
#   generation_usd  - block subsidy alone (reward_usd = generation_usd + fee_total_usd),
#                     so it isolates the halving step from fee revenue.
BLOCK_COLS = ["reward_usd", "transaction_count", "difficulty", "fee_total_usd",
              "cdd_total", "generation_usd"]

# Per-block daily averages (metric mean across blocks in a day).
PER_BLOCK = [
    ("reward_usd", "Average Reward per Block vs. Time", "Reward (USD)"),
    ("transaction_count", "Average Transactions per Block vs. Time", "Transaction Count"),
    ("difficulty", "Average Difficulty per Block vs. Time", "Difficulty"),
    ("fee_total_usd", "Average Fee per Block vs. Time", "Fee (USD)"),
    ("cdd_total", "Average Coin-Days Destroyed per Block vs. Time", "CDD"),
    ("generation_usd", "Average Block Subsidy vs. Time", "Subsidy (USD)"),
]

# Per-transaction ratios (daily metric sum / daily transaction_count sum).
# NOTE: difficulty-per-transaction is not a physically meaningful quantity
# (difficulty is a network level, not something spent per tx) — kept only to
# reproduce the original analysis.
PER_TX = [
    ("reward_usd", "Average Reward per Transaction vs. Time", "Reward per Tx (USD)"),
    ("fee_total_usd", "Average Fee per Transaction vs. Time", "Fee per Tx (USD)"),
    ("difficulty", "Average Difficulty per Transaction vs. Time", "Difficulty per Tx"),
]

# Daily totals (metric summed across all blocks in a day).
# NOTE: summing `difficulty` is not meaningful (it is a level, not a flow) —
# kept only to reproduce the original analysis.
TOTALS = [
    ("reward_usd", "Total Daily Rewards vs. Time", "Rewards (USD)"),
    ("transaction_count", "Total Daily Transactions vs. Time", "Transaction Count"),
    ("difficulty", "Total Daily Difficulty vs. Time", "Difficulty"),
    ("fee_total_usd", "Total Daily Fee vs. Time", "Total Daily Fee (USD)"),
    ("cdd_total", "Total Daily Coin-Days Destroyed vs. Time", "CDD"),
    ("generation_usd", "Total Daily Block Subsidy vs. Time", "Subsidy (USD)"),
]


def gaussian_smooth(series, sigma=SIGMA, mode=EDGE_MODE):
    """Centered Gaussian smoothing that ignores (rather than spreads) NaNs."""
    values = np.asarray(series, dtype=float)
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


def plot_series(dates, raw, title, y_label, raw_label):
    """Plot a raw daily series plus its Gaussian trend, halving marked."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, raw, color="lightblue", lw=1, label=raw_label)
    ax.plot(dates, gaussian_smooth(raw), color="orange", lw=2,
            label=f"Gaussian trend (σ={SIGMA:g}d)")
    ax.axvline(HALVING_DATE, color="red", alpha=0.4, ls="--", lw=1.5,
               label="Halving (2020-05-11)")

    ax.set(title=title, xlabel="Date", ylabel=y_label)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        safe = title.split(" vs.")[0].strip().replace(" ", "_").lower()
        fig.savefig(FIG_DIR / f"{safe}.png", dpi=150)
    return fig, ax


def plot_btc_price():
    df = pd.read_csv(MERGED_FILE, sep="\t")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    plot_series(df["Date"], df["Adj Close"].to_numpy(),
                "BTC Value (USD) vs. Time", "BTC Value (USD)",
                "Bitcoin Value (USD)")


def plot_block_metrics():
    # Read the large blocks file ONCE; derive both means and sums from one groupby.
    blocks = pd.read_csv(BLOCKS_FILE, sep="\t")
    blocks["date"] = pd.to_datetime(blocks["time"]).dt.normalize()
    grouped = blocks.groupby("date")[BLOCK_COLS]
    daily_mean = grouped.mean()
    daily_sum = grouped.sum()

    # Per-block daily averages.
    for metric, title, y_label in PER_BLOCK:
        plot_series(daily_mean.index, daily_mean[metric].to_numpy(),
                    title, y_label, f"Average Daily {y_label}")

    # Per-transaction ratios (share the single daily_sum groupby).
    tx_total = daily_sum["transaction_count"]
    for metric, title, y_label in PER_TX:
        per_tx = (daily_sum[metric] / tx_total).to_numpy()
        plot_series(daily_sum.index, per_tx, title, y_label, f"Average {y_label}")

    # Daily totals (share the same daily_sum groupby).
    for metric, title, y_label in TOTALS:
        plot_series(daily_sum.index, daily_sum[metric].to_numpy(),
                    title, y_label, f"Total Daily {y_label}")


def main():
    plot_btc_price()
    plot_block_metrics()
    plt.show()


if __name__ == "__main__":
    main()
