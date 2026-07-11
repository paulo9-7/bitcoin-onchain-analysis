"""
Lead/lag screen: does any on-chain metric carry information BEFORE price moves?

This is the bridge from the (contemporaneous) VIP screening to a predictive
study. It answers a different question than the correlation matrix or PLS:
not "does X move WITH price?" but "does X move BEFORE price?".

Method
------
- Everything is made STATIONARY first: price -> log return, predictors ->
  first difference. This is essential — cross-correlating trending levels
  produces high correlations at every lag purely from shared trend, which
  looks like lead/lag but is an artifact.
- Cross-correlation function (CCF): for each lag k we correlate
  predictor_change(t) with price_return(t + k), i.e. corr(pred.shift(k), ret).
  Sign convention:
      k > 0  ->  predictor's PAST vs price's PRESENT  ->  PREDICTOR LEADS price
                 (this is the interesting, potentially predictive case)
      k < 0  ->  price leads the predictor (predictor reacts to price)
      k = 0  ->  contemporaneous co-movement
- An approximate 95% significance band of +/- 2/sqrt(N) flags CCF values
  unlikely to be noise.
- Granger causality: tests whether past predictor values improve a forecast of
  the price return BEYOND price's own past. Complements the CCF.

CAVEATS
-------
- Multiple testing: several predictors x many lags = many chances for a false
  positive. Treat a lone "significant" lag skeptically; look for a coherent
  run of significant lags with a sensible sign, not an isolated spike.
- One year, one halving: limited power. This screens candidates, it does not
  prove causation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

# --- Configuration -----------------------------------------------------------

BASE = Path("/Users/pauloconnor/Desktop/py.scripts/Bitcoin tsv/Bitcoin_Redo")
DATA_FILE = BASE / "BTC-USD_df2.tsv"
PRICE_COL = "Adj Close"

MAX_LAG = 10          # cross-correlation range, +/- days
GRANGER_MAXLAG = 5    # Granger test up to this many lags

# Raw daily candidate predictors (NOT the smoothed ones, NOT generation_usd).
PREDICTORS = [
    "transaction_count_sum",
    "fee_total_usd_sum",
    "fee_total_usd_per_transaction",
    "difficulty_mean",
    "cdd_total_sum",
    "cdd_total_mean",
]

LABELS = {
    "transaction_count_sum": "Total Tx Count",
    "fee_total_usd_sum": "Total Fees",
    "fee_total_usd_per_transaction": "Fee / Tx",
    "difficulty_mean": "Difficulty",
    "cdd_total_sum": "Total CDD",
    "cdd_total_mean": "CDD / Block",
}


def load():
    df = pd.read_csv(DATA_FILE, sep="\t")
    missing = [c for c in PREDICTORS + [PRICE_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {DATA_FILE.name}: {missing}")

    ret = np.log(df[PRICE_COL].astype(float)).diff()   # stationary target
    preds = df[PREDICTORS].astype(float).diff()         # stationary predictors
    data = pd.concat([preds, ret.rename("ret")], axis=1).dropna().reset_index(drop=True)
    return data


def ccf_row(pred, ret, max_lag):
    """corr(pred.shift(k), ret) for k in [-max_lag, max_lag]; k>0 => pred leads."""
    return {k: ret.corr(pred.shift(k)) for k in range(-max_lag, max_lag + 1)}


def main():
    data = load()
    ret = data["ret"]
    n = len(ret)
    band = 2.0 / np.sqrt(n)   # ~95% white-noise significance band
    print(f"Rows (after differencing): {n} | significance band: +/-{band:.3f}")

    if HAVE_SM:
        p_adf = adfuller(ret)[1]
        print(f"ADF on price log-return: p={p_adf:.4g} "
              f"({'stationary' if p_adf < 0.05 else 'NOT stationary — interpret with care'})")

    # Build the CCF matrix (predictors x lags).
    lags = list(range(-MAX_LAG, MAX_LAG + 1))
    ccf = pd.DataFrame(
        {name: ccf_row(data[name], ret, MAX_LAG) for name in PREDICTORS}
    ).T[lags]
    ccf.index = [LABELS[c] for c in PREDICTORS]

    # Per-predictor summary: strongest LEAD (k>0) correlation and Granger.
    print("\nLead/lag summary (positive lag = predictor leads price):")
    header = f"  {'predictor':16s} {'peak|CCF|@lag':>16s} {'sig?':>5s}"
    if HAVE_SM:
        header += f" {'Granger minp@lag':>18s}"
    print(header)

    for name in PREDICTORS:
        row = ccf.loc[LABELS[name]]
        peak_lag = row.abs().idxmax()
        peak_val = row[peak_lag]
        sig = "yes" if abs(peak_val) > band else "no"
        line = f"  {LABELS[name]:16s} {peak_val:+.3f} @ {peak_lag:+d}    {sig:>3s}"

        if HAVE_SM:
            # Test: does the predictor Granger-cause the return?
            # statsmodels: 2nd column is the hypothesised cause.
            gdata = data[["ret", name]].values
            res = grangercausalitytests(gdata, maxlag=GRANGER_MAXLAG, verbose=False)
            pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
            g_lag = min(pvals, key=pvals.get)
            line += f"   p={pvals[g_lag]:.3f} @ {g_lag}"
        print(line)

    if not HAVE_SM:
        print("\n[note] statsmodels not installed — Granger test skipped. "
              "Install with: pip install statsmodels")

    # --- Heatmap of the CCF --------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(ccf, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                center=0, vmin=-max(abs(ccf.values.min()), abs(ccf.values.max())),
                vmax=max(abs(ccf.values.min()), abs(ccf.values.max())),
                annot=True, fmt=".2f", linewidths=.5, cbar_kws={"label": "correlation"},
                ax=ax)
    ax.axvline(MAX_LAG + 0.5, color="black", lw=1.5)  # lag = 0 divider
    ax.set(xlabel="lag (days)   —   left: price leads   |   right: predictor leads",
           ylabel="", title="Cross-correlation of on-chain changes vs. BTC log-return")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()