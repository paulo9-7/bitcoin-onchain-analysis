"""
PLS regression: relating Bitcoin block metrics to BTC price.

This consolidates the original five repeated blocks into one script and fixes
several correctness/leakage issues. Read the NOTES before trusting any number.

KEY FIXES vs. the original
--------------------------
1. TIME ORDER IS RESPECTED. The original used KFold(shuffle=True) and
   train_test_split (shuffle by default). On an autocorrelated — and here
   *smoothed* — time series that leaks the future into the past and produces
   fantasy scores. We use TimeSeriesSplit (forward chaining) and a chronological
   train/test split.
2. NO SCALING LEAKAGE. StandardScaler was fit on the whole dataset before CV,
   so every fold's scaler had seen the test rows. Scaling now lives inside a
   Pipeline, refit within each fold.
3. WARM-UP HANDLED PROPERLY. `df.iloc[19:]` assumed a 20-day trailing MA; it is
   wrong for Gaussian smoothing. We drop rows with NaNs in the used columns.
4. Q^2 BUG REMOVED. The original computed `q2 = 1 - (1 - cv_r2)`, i.e. exactly
   cv_r2 — a no-op. Cross-validated R^2 already *is* Q^2.
5. A NAIVE BASELINE is reported, because a model that can't beat "tomorrow =
   today" on a trending price series isn't adding value.
6. De-duplicated: data loaded once, columns defined once, no repeated prints.

NOTES ON WHAT IS BEING MODELLED (see the chat suggestions for detail)
--------------------------------------------------------------------
- Predicting the *smoothed* target from *smoothed* predictors inflates fit and,
  with a centered Gaussian, leaks future information. Set USE_SMOOTHED = False
  to model the raw series instead — strongly recommended for any real claim.
- Features are contemporaneous (day t predicts day t price): this is a
  nowcast/association, not a forecast. Set LAG_DAYS > 0 to lag predictors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration -----------------------------------------------------------

BASE = Path("/Users/pauloconnor/Desktop/py.scripts/Bitcoin tsv/Bitcoin_Redo")
DATA_FILE = BASE / "BTC-USD_df2.tsv"

USE_SMOOTHED = True      # True reproduces the original; False models raw series
LAG_DAYS = 0             # >0 lags predictors so day t-LAG predicts day t price
N_SPLITS = 5
MAX_COMPONENTS = 10      # up to the number of predictors
RUN_SUBSET_SEARCH = False # exhaustive 2^k-1 search; biased selection — see notes

# Base predictor names.
# Excluded on purpose (target leakage): reward_usd and generation_usd. The
# block subsidy generation_usd is ~ (fixed BTC per block) x price, i.e. nearly
# a constant multiple of the target, and reward_usd = generation_usd +
# fee_total_usd. Using either to "predict" price is circular. cdd_total (coin-
# days destroyed) is added because it carries independent information about
# older-coin movement and is not derived from price.
BASE_PREDICTORS = [
    "transaction_count_sum", "transaction_count_mean",
    "fee_total_usd_sum", "fee_total_usd_mean",
    "difficulty_sum", "difficulty_mean",
    "fee_total_usd_per_transaction", "difficulty_per_transaction",
    "cdd_total_sum", "cdd_total_mean",
]

# Leakage-prone; only enable to demonstrate the effect, never for a real claim:
# BASE_PREDICTORS += ["generation_usd_sum", "generation_usd_mean"]


def resolve_columns(df):
    """Pick smoothed or raw column names, matching whatever prefix exists."""
    if not USE_SMOOTHED:
        return BASE_PREDICTORS, "Adj Close"
    for prefix, btc in (("gaussian_", "gaussian_BTC_USD"),
                        ("moving_avg_", "moving_avg_BTC USD")):
        preds = [f"{prefix}{c}" for c in BASE_PREDICTORS]
        if all(c in df.columns for c in preds) and btc in df.columns:
            return preds, btc
    raise KeyError("Could not find smoothed predictor/target columns in the file.")


def make_model(n_components):
    """Scaling + PLS as one estimator so scaling is refit inside each CV fold."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pls", PLSRegression(n_components=n_components)),
    ])


def load_xy():
    df = pd.read_csv(DATA_FILE, sep="\t")
    predictors, target = resolve_columns(df)

    data = df[predictors + [target]].copy()
    if LAG_DAYS > 0:
        data[predictors] = data[predictors].shift(LAG_DAYS)  # past -> present
    data = data.dropna().reset_index(drop=True)

    X = data[predictors]
    y = data[target]
    return X, y, predictors, target


# --- 1. Component-selection curve (time-aware CV) ----------------------------

def component_curve(X, y):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []
    for n in range(1, min(X.shape[1], MAX_COMPONENTS) + 1):
        scores = cross_val_score(make_model(n), X, y,
                                 scoring="neg_root_mean_squared_error", cv=tscv)
        rows.append({"Components": n, "CV_RMSE": -scores.mean(),
                     "CV_RMSE_std": scores.std()})
    table = pd.DataFrame(rows)
    print("\nComponent selection (TimeSeriesSplit):")
    print(table.to_string(index=False))

    best_n = int(table.loc[table["CV_RMSE"].idxmin(), "Components"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(table["Components"], table["CV_RMSE"], yerr=table["CV_RMSE_std"],
                marker="o", capsize=3)
    ax.axvline(best_n, color="red", ls="--", alpha=0.5, label=f"best = {best_n}")
    ax.set(xlabel="PLS components", ylabel="CV RMSE",
           title="Cross-validated RMSE vs. number of components")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return best_n


# --- 2. Optional exhaustive subset search (biased; use with care) ------------

def subset_search(X, y, predictors):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    best = {"rmse": np.inf, "cols": None, "n": None}
    for r in range(1, len(predictors) + 1):
        for subset in combinations(predictors, r):
            Xs = X[list(subset)]
            for n in range(1, min(len(subset), MAX_COMPONENTS) + 1):
                s = cross_val_score(make_model(n), Xs, y,
                                    scoring="neg_root_mean_squared_error", cv=tscv)
                rmse = -s.mean()
                if rmse < best["rmse"]:
                    best = {"rmse": rmse, "cols": subset, "n": n}
    print("\n[!] Exhaustive search min-RMSE is optimistically biased "
          "(hundreds of configs, one winner). Treat as exploratory.")
    print(f"Best subset RMSE: {best['rmse']:.2f} | n_components={best['n']}")
    print(f"Best subset: {best['cols']}")
    return best


# --- 3. Final model, chronological hold-out + baseline ------------------------

def final_model(X, y, n_components, predictors, target):
    split = int(len(X) * 0.8)               # last 20% is the future (no shuffle)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = make_model(n_components).fit(X_train, y_train)
    pred_train = model.predict(X_train).ravel()
    pred_test = model.predict(X_test).ravel()

    def report(name, yt, yp):
        print(f"{name:6s} RMSE={np.sqrt(mean_squared_error(yt, yp)):.2f} "
              f"MAE={mean_absolute_error(yt, yp):.2f} R2={r2_score(yt, yp):.3f}")

    print(f"\nFinal PLS ({n_components} comp), chronological 80/20 split:")
    report("train", y_train, pred_train)
    report("test", y_test, pred_test)

    # Naive persistence baseline on the test window: yhat_t = y_{t-1}.
    naive = y_test.shift(1).bfill()
    print(f"naive  RMSE={np.sqrt(mean_squared_error(y_test, naive)):.2f}  "
          "(persistence: tomorrow = today)")

    # Standardized coefficients (units of std, because inputs are scaled).
    pls = model.named_steps["pls"]
    coef = pls.coef_.reshape(-1)
    print("\nStandardized PLS coefficients:")
    for name, c in sorted(zip(predictors, coef), key=lambda t: -abs(t[1])):
        print(f"  {name:38s} {c:+.3f}")

    print("\nVIP scores (variable importance in projection):")
    for name, v in sorted(zip(predictors, vip(pls)), key=lambda t: -t[1]):
        print(f"  {name:38s} {v:.3f}")

    return model


def vip(pls):
    """Variable Importance in Projection for a fitted PLSRegression."""
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    ssy = np.array([(q[0, j] ** 2) * (t[:, j] @ t[:, j]) for j in range(h)])
    total = ssy.sum()
    vips = np.zeros(p)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (ssy @ weight) / total)
    return vips


# --- 4. Time-aware cross-validated predictions (Q^2) -------------------------

def cv_report(X, y, n_components):
    # TimeSeriesSplit is not a partition (early rows never appear in a test
    # fold), so cross_val_predict can't be used. Collect out-of-fold
    # predictions manually and score only the rows that received one.
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    y_cv = pd.Series(np.nan, index=y.index)
    for train_idx, test_idx in tscv.split(X):
        model = make_model(n_components).fit(X.iloc[train_idx], y.iloc[train_idx])
        y_cv.iloc[test_idx] = model.predict(X.iloc[test_idx]).ravel()

    mask = y_cv.notna()
    print("\nTime-series cross-validation "
          f"(scored on {int(mask.sum())} of {len(y)} rows):")
    print(f"  CV RMSE={np.sqrt(mean_squared_error(y[mask], y_cv[mask])):.2f}  "
          f"MAE={mean_absolute_error(y[mask], y_cv[mask]):.2f}  "
          f"Q2 (CV R2)={r2_score(y[mask], y_cv[mask]):.3f}")
    return y_cv


# --- 5. Plots ----------------------------------------------------------------

def plot_fit(y, y_pred, target):
    # y_pred may carry NaNs for early rows with no out-of-fold prediction.
    y_pred = pd.Series(np.asarray(y_pred).ravel(), index=y.index)
    mask = y_pred.notna()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(y.index, y, color="lightblue", lw=2, label=f"Actual {target}")
    ax.plot(y.index[mask], y_pred[mask], color="salmon", lw=2,
            label="PLS predicted (CV)")
    ax.set(xlabel="Day", ylabel=target, title="Actual vs. PLS-predicted")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()

    ya, yp = y[mask], y_pred[mask]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ya, yp, color="lightblue", edgecolor="k", alpha=0.6)
    lim = [min(ya.min(), yp.min()), max(ya.max(), yp.max())]
    ax.plot(lim, lim, color="salmon", lw=2)
    ax.set(xlabel="Actual", ylabel="Predicted", title="Parity plot")
    ax.grid(True, alpha=0.3); fig.tight_layout()


def main():
    X, y, predictors, target = load_xy()
    print(f"Rows: {len(X)} | Predictors: {len(predictors)} | Target: {target}")

    best_n = component_curve(X, y)
    if RUN_SUBSET_SEARCH:
        subset_search(X, y, predictors)

    final_model(X, y, best_n, predictors, target)
    y_cv = cv_report(X, y, best_n)
    plot_fit(y, y_cv, target)
    plt.show()


if __name__ == "__main__":
    main()