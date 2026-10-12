"""
Homework 07 - VaR backtesting for a constant $1 position in the S&P 500 index.

Follows the lecture's backtesting example (slides 25-37):
  Task 1: download daily S&P 500 closes, 2016-05-23 to 2026-05-22.
  Task 2: one-day VaR from a rolling empirical distribution (M = 1010 returns),
          full loss operator l(x) = -(e^x - 1), and the resulting exceedances.
  Task 3: one-day VaR from normal log returns with EWMA updating
          (lambda = theta = 0.97), VaR = 1 - exp(mu + sigma * Phi^{-1}(1 - alpha)).
  Task 4: 1 - beta confidence interval for the number of exceedances.

Run:  python hw07_var_backtest.py              (downloads from Yahoo Finance)
      python hw07_var_backtest.py --csv FILE   (uses a saved CSV with Date, Close)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
alpha = 0.95
beta = 0.05
lam = theta = 0.97
M = 1010                       # rolling window (number of log returns)
START, END = "2016-05-23", "2026-05-22"

HERE = os.path.dirname(os.path.abspath(__file__))


def loss(x):
    """Full loss operator for a constant $1 position: l(x) = -(e^x - 1)."""
    return -(np.exp(x) - 1.0)


# ---------------------------------------------------------------------------
# Task 1: data
# ---------------------------------------------------------------------------
def load_prices():
    if "--csv" in sys.argv:
        path = sys.argv[sys.argv.index("--csv") + 1]
        S = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf
        # yfinance treats `end` as exclusive, so add one day to include END
        end_excl = (pd.Timestamp(END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download("^GSPC", start=START, end=end_excl,
                          auto_adjust=False, progress=False)
        S = raw["Close"]
        if isinstance(S, pd.DataFrame):          # newer yfinance returns a DataFrame
            S = S.iloc[:, 0]
        S.to_frame("Close").to_csv(os.path.join(HERE, "sp500_prices.csv"))
    S = S.dropna()
    S.name = "Close"
    S.index.name = "Date"
    return S


# ---------------------------------------------------------------------------
# Task 2: rolling empirical-distribution VaR
# ---------------------------------------------------------------------------
def empirical_var(x):
    """VaR_t for each evaluation day, from the previous M returns.
    x has length n; VaR for day k (k = M, ..., n-1, 0-based) uses x[k-M:k]."""
    n = len(x)
    k_idx = int(np.ceil(M * alpha)) - 1           # ceil(M * alpha)-th smallest
    var = np.empty(n - M)
    for j, k in enumerate(range(M, n)):
        var[j] = np.sort(loss(x[k - M:k]))[k_idx]
    return var


# ---------------------------------------------------------------------------
# Task 3: normal log returns with EWMA updating
# ---------------------------------------------------------------------------
def ewma_var(x):
    """VaR_t for each evaluation day. Initial mu, sigma^2 = sample mean and
    sample variance of the first M returns; afterwards
        mu_{t+1}      = lam   * mu_t      + (1 - lam)   * x_t
        sigma^2_{t+1} = theta * sigma^2_t + (1 - theta) * (x_t - mu_t)^2
    and VaR_t = 1 - exp(mu + sigma * Phi^{-1}(1 - alpha))."""
    n = len(x)
    mu = x[:M].mean()
    s2 = x[:M].var(ddof=1)
    z = norm.ppf(1 - alpha)
    var, mus, sigs = np.empty(n - M), np.empty(n - M), np.empty(n - M)
    for j, k in enumerate(range(M, n)):
        var[j] = 1.0 - np.exp(mu + np.sqrt(s2) * z)
        mus[j], sigs[j] = mu, np.sqrt(s2)
        xk = x[k]                                 # realized return, then update
        s2 = theta * s2 + (1 - theta) * (xk - mu) ** 2
        mu = lam * mu + (1 - lam) * xk
    return var, mus, sigs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pd.set_option("display.float_format", "{:,.2f}".format)

    # Task 1
    S = load_prices()
    N = len(S)
    print("Task 1. S&P 500 daily closing prices (^GSPC, Yahoo Finance)")
    print("  First five observations:")
    print(S.head().to_string().replace("\n", "\n    ").join(["    ", ""]))
    print("  Last five observations:")
    print(S.tail().to_string().replace("\n", "\n    ").join(["    ", ""]))
    print(f"  Total number of observations N = {N}  "
          f"({S.index[0]:%Y-%m-%d} to {S.index[-1]:%Y-%m-%d})")

    x = np.diff(np.log(S.values))                 # n = N - 1 log returns
    n = len(x)
    J = n - M
    dates = S.index[1:][M:]                       # date of each evaluated return
    L = loss(x[M:])                               # realized losses
    print(f"  Log returns: {n};  window M = {M};  evaluation periods J = {n} - {M} = {J}")
    print()

    # Task 2
    var_emp = empirical_var(x)
    exc_emp = L > var_emp
    print("Task 2. Rolling empirical-distribution VaR")
    print(f"  First VaR estimate (for {dates[0]:%Y-%m-%d}): {var_emp[0]:.6f}")
    print(f"  Average VaR: {var_emp.mean():.6f}   range [{var_emp.min():.6f}, {var_emp.max():.6f}]")
    print(f"  Number of exceedances: {exc_emp.sum()}  ({exc_emp.mean():.2%} of days)")
    print()

    # Task 3
    var_ewma, mus, sigs = ewma_var(x)
    exc_ewma = L > var_ewma
    print("Task 3. Normal log returns with EWMA updating")
    print(f"  Initial mu = {x[:M].mean():.6e},  sigma = {x[:M].std(ddof=1):.6f}  "
          f"(sample mean / sd of the first {M} returns)")
    print(f"  First VaR estimate (for {dates[0]:%Y-%m-%d}): {var_ewma[0]:.6f}")
    print(f"  Average VaR: {var_ewma.mean():.6f}   range [{var_ewma.min():.6f}, {var_ewma.max():.6f}]")
    print(f"  Number of exceedances: {exc_ewma.sum()}  ({exc_ewma.mean():.2%} of days)")
    print()

    # Task 4
    z = norm.ppf(1 - beta / 2)
    mean_exc = J * (1 - alpha)
    half = z * np.sqrt(J * alpha * (1 - alpha))
    lo, hi = mean_exc - half, mean_exc + half
    print(f"Task 4. {1 - beta:.0%} confidence interval for the number of exceedances")
    print(f"  Expected J(1 - alpha) = {mean_exc:.2f},  z = {z:.4f},  "
          f"half-width = {half:.2f}")
    print(f"  CI = [{lo:.2f}, {hi:.2f}]")
    for name, e in [("Empirical", exc_emp), ("EWMA", exc_ewma)]:
        k = e.sum()
        if k > hi:
            verdict = "above the CI -> reject; VaR too low (model too optimistic)"
        elif k < lo:
            verdict = "below the CI -> reject; VaR too high (model too pessimistic)"
        else:
            verdict = "inside the CI -> do not reject"
        zstat = (k - mean_exc) / np.sqrt(J * alpha * (1 - alpha))
        print(f"  {name:9s}: {k:4d} exceedances, Z = {zstat:6.2f}  ->  {verdict}")
    print()

    # Diagnostic: clustering of exceedances (independence)
    print("  Clustering check: P(exceedance tomorrow | exceedance today)")
    print(f"    (should be about {1 - alpha:.2f} if exceedances are independent)")
    for name, e in [("Empirical", exc_emp), ("EWMA", exc_ewma)]:
        prev = e[:-1]
        p = e[1:][prev].mean() if prev.sum() > 0 else float("nan")
        print(f"    {name:9s}: {p:.3f}  ({int((e[1:] & prev).sum())} back-to-back pairs)")
    by_year = pd.DataFrame({"Empirical": exc_emp, "EWMA": exc_ewma}, index=dates)
    by_year = by_year.groupby(by_year.index.year).agg(["sum", "count"])
    print("  Exceedances by calendar year (sum / trading days):")
    for yr, row in by_year.iterrows():
        print(f"    {yr}: Empirical {int(row[('Empirical', 'sum')]):3d}, "
              f"EWMA {int(row[('EWMA', 'sum')]):3d}  of {int(row[('Empirical', 'count')])} days")

    # Save the series
    out = pd.DataFrame({"loss": L, "VaR_empirical": var_emp, "exc_empirical": exc_emp,
                        "VaR_ewma": var_ewma, "exc_ewma": exc_ewma,
                        "mu_ewma": mus, "sigma_ewma": sigs}, index=dates)
    out.to_csv(os.path.join(HERE, "hw07_var_series.csv"))

    # Figure
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(dates, L, color=GREY, lw=0.5, label="Realized loss")
    ax1.plot(dates, var_emp, color=BLUE, lw=1.4, label="Empirical VaR")
    ax1.plot(dates, var_ewma, color=ORANGE, lw=1.0, label="EWMA VaR")
    ax1.scatter(dates[exc_emp], L[exc_emp], s=14, color=BLUE, zorder=3,
                label=f"Empirical exceedance ({exc_emp.sum()})")
    ax1.scatter(dates[exc_ewma], L[exc_ewma], s=14, facecolors="none",
                edgecolors=ORANGE, lw=1, zorder=4, label=f"EWMA exceedance ({exc_ewma.sum()})")
    ax1.set_ylabel("Loss per $1")
    ax1.set_title("One-day 95% VaR and realized losses, S&P 500", pad=28)
    ax1.legend(frameon=False, fontsize=8, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.4, columnspacing=1.2)
    # gains are not of interest here; keep the loss side readable
    ax1.set_ylim(-0.04, max(L.max(), var_emp.max(), var_ewma.max()) * 1.05)

    k = np.arange(1, J + 1)
    band = z * np.sqrt(k * alpha * (1 - alpha))
    ax2.fill_between(dates, k * (1 - alpha) - band, k * (1 - alpha) + band,
                     color=GREY, alpha=0.2, lw=0, label=f"{1 - beta:.0%} band")
    ax2.plot(dates, k * (1 - alpha), color=GREY, lw=1, ls="--", label="Expected")
    ax2.plot(dates, np.cumsum(exc_emp), color=BLUE, lw=1.6, label="Empirical")
    ax2.plot(dates, np.cumsum(exc_ewma), color=ORANGE, lw=1.6, label="EWMA")
    ax2.set_ylabel("Cumulative exceedances")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
    plt.tight_layout()
    fig_path = os.path.join(HERE, "hw07_backtest.png")
    plt.savefig(fig_path, dpi=200)
    print()
    print(f"Figure saved to {fig_path}")
    print(f"VaR series saved to {os.path.join(HERE, 'hw07_var_series.csv')}")


if __name__ == "__main__":
    main()
