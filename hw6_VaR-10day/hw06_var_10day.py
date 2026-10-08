"""
Homework 06 - 10-day VaR of a delta-hedged short put position.

Short 100 puts (strike kappa, maturity T), delta-hedged with the stock and a
money-market account, following the lecture slides "MC Example: Hedged Call
Option" (slides 22-24), with the put in place of the call.

  Q1: one-day VaR over [t, t+Delta] by simulation, then sqrt(10) x one-day VaR.
  Q2: 10-day VaR over [t, t+10 Delta] by full simulation with daily re-hedging
      (self-financing strategy).
  Q3: comparison (printed numbers + figure).

Run:  python hw06_var_10day.py
"""

import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
alpha  = 0.95
t0     = 0.0
Delta  = 1 / 252
T      = 0.292          # maturity (time to maturity at t = 0)
S0     = 152.51
kappa  = 170.0
r      = 0.00119        # 0.119%
sigma  = 0.4907         # 49.07%
mu     = 0.1691         # 16.91%  (physical drift, used only to simulate S)
J      = 10             # days
n_opt  = 100            # number of puts sold
M      = 40_000         # simulation runs
SEED   = 2026


# ---------------------------------------------------------------------------
# Black-Scholes price and delta
# ---------------------------------------------------------------------------
def _d1(t, s, k, mat, r, sig):
    tau = mat - t
    return (np.log(s / k) + (r + 0.5 * sig**2) * tau) / (sig * np.sqrt(tau))

def bs_put(t, s, k=kappa, mat=T, r=r, sig=sigma):
    tau = mat - t
    d1 = _d1(t, s, k, mat, r, sig)
    d2 = d1 - sig * np.sqrt(tau)
    return k * np.exp(-r * tau) * norm.cdf(-d2) - s * norm.cdf(-d1)

def bs_put_delta(t, s, k=kappa, mat=T, r=r, sig=sigma):
    return norm.cdf(_d1(t, s, k, mat, r, sig)) - 1.0

def bs_call(t, s, k=kappa, mat=T, r=r, sig=sigma):
    tau = mat - t
    d1 = _d1(t, s, k, mat, r, sig)
    d2 = d1 - sig * np.sqrt(tau)
    return s * norm.cdf(d1) - k * np.exp(-r * tau) * norm.cdf(d2)

def bs_call_delta(t, s, k=kappa, mat=T, r=r, sig=sigma):
    return norm.cdf(_d1(t, s, k, mat, r, sig))


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
def simulate_paths(n_days, n_paths, rng):
    """Stock paths under the physical measure P (GBM, exact discretization).
    Returns an array of shape (n_paths, n_days + 1); column 0 is S0."""
    Z = rng.standard_normal((n_paths, n_days))
    X = (mu - 0.5 * sigma**2) * Delta + sigma * np.sqrt(Delta) * Z   # daily log returns
    S = S0 * np.exp(np.cumsum(X, axis=1))
    return np.hstack([np.full((n_paths, 1), S0), S])

def hedged_values(S, price, delta, rehedge=True):
    """Value path V_{T_0}, ..., V_{T_J} of a short option (one unit) that is
    delta-hedged with the stock and a money-market account.

    Follows the lecture's self-financing strategy (slides 22-23):
      day 1:  h_{T_0} = delta(T_0, S_{T_0}),  Y^new_{T_0} = Y_{T_0} - h_{T_0} S_{T_0},  Y_{T_0} = 0
      close of day j:
              V_{T_j} = h_{T_{j-1}} S_{T_j} + Y^new_{T_{j-1}} e^{r Delta} - price(T_j, S_{T_j})
      open of day j+1 (re-hedge):
              h_{T_j} = delta(T_j, S_{T_j})
              Y^new_{T_j} = Y_{T_j} + (h_{T_{j-1}} - h_{T_j}) S_{T_j}
    With rehedge=False the initial hedge h_{T_0} is kept for all days.
    S has shape (n_paths, n_days + 1); the result has the same shape."""
    n_days = S.shape[1] - 1
    V = np.empty_like(S)
    h = delta(t0, S[:, 0])
    Y_new = 0.0 - h * S[:, 0]
    V[:, 0] = Y_new + h * S[:, 0] - price(t0, S[:, 0])
    for j in range(1, n_days + 1):
        Tj = t0 + j * Delta
        Y = Y_new * np.exp(r * Delta)               # bank account at the close of day j
        V[:, j] = h * S[:, j] + Y - price(Tj, S[:, j])
        if j < n_days:
            h_new = delta(Tj, S[:, j]) if rehedge else h
            Y_new = Y + (h - h_new) * S[:, j]
            h = h_new
    return V

def hedged_loss(S, price, delta, rehedge=True):
    """Total loss L = -(V_{T_J} - V_{T_0}) of one hedged short option."""
    V = hedged_values(S, price, delta, rehedge)
    return -(V[:, -1] - V[:, 0])

def emp_var(L, a=alpha):
    """Empirical VaR: the ceil(M * a)-th smallest loss."""
    Ls = np.sort(L)
    return Ls[int(np.ceil(len(Ls) * a)) - 1]


# ---------------------------------------------------------------------------
# Check against the lecture (100 short calls, slides 11 and 24)
# ---------------------------------------------------------------------------
def replicate(price, delta, n_rep=20):
    """Repeat the simulation n_rep times (M paths each) to measure Monte Carlo
    variability. Returns arrays of sqrt(10) x one-day VaR, 10-day VaR (re-hedged)
    and 10-day VaR (no re-hedging)."""
    out = []
    for k in range(n_rep):
        rng = np.random.default_rng(1000 + k)
        S = simulate_paths(J, M, rng)
        L1 = n_opt * hedged_loss(S[:, :2], price, delta)
        L10 = n_opt * hedged_loss(S, price, delta, rehedge=True)
        L10s = n_opt * hedged_loss(S, price, delta, rehedge=False)
        out.append((np.sqrt(J) * emp_var(L1), emp_var(L10), emp_var(L10s)))
    return np.array(out).T

def lecture_check():
    sq, agg, stat = replicate(bs_call, bs_call_delta)
    print("Check against the lecture (100 short calls, delta-hedged; mean +/- sd over 20 runs of 40,000):")
    print(f"  10-day VaR, daily re-hedging : {agg.mean():7.2f} +/- {agg.std():.2f}   (lecture: 86.76)")
    print(f"  sqrt(10) x one-day VaR       : {sq.mean():7.2f} +/- {sq.std():.2f}   (lecture: 92.97)")
    print(f"  10-day VaR, no re-hedging    : {stat.mean():7.2f} +/- {stat.std():.2f}   (lecture: 300.6)")
    print()


# ---------------------------------------------------------------------------
# Homework: 100 short puts
# ---------------------------------------------------------------------------
def main():
    lecture_check()

    P0, h0 = bs_put(t0, S0), bs_put_delta(t0, S0)
    print("Homework 06: short 100 puts, delta-hedged")
    print(f"  Put price P^BS(0, S0) = {P0:.4f},  put delta h_0 = {h0:.4f}")
    print(f"  Initial hedge: long {n_opt * h0:.2f} shares (i.e. short {-n_opt * h0:.2f}), "
          f"money market {-n_opt * h0 * S0:,.2f}")
    print()

    # Q1: one-day VaR and square-root-of-time rule
    rng1 = np.random.default_rng(SEED)
    S_1day = simulate_paths(1, M, rng1)
    L1 = n_opt * hedged_loss(S_1day, bs_put, bs_put_delta)
    var1 = emp_var(L1)
    var10_sqrt = np.sqrt(J) * var1
    print("Q1. One-day VaR and square-root-of-time rule")
    print(f"  One-day VaR_0.95            = {var1:8.4f}")
    print(f"  sqrt(10) x one-day VaR_0.95 = {var10_sqrt:8.4f}")
    print()

    # Q2: 10-day VaR by time aggregation (daily re-hedging)
    rng2 = np.random.default_rng(SEED + 1)
    S_10day = simulate_paths(J, M, rng2)
    L10 = n_opt * hedged_loss(S_10day, bs_put, bs_put_delta, rehedge=True)
    var10 = emp_var(L10)
    print("Q2. 10-day VaR by time aggregation (daily re-hedging)")
    print(f"  10-day VaR_0.95             = {var10:8.4f}")
    print()

    # Q3: comparison and diagnostics
    L10_static = n_opt * hedged_loss(S_10day, bs_put, bs_put_delta, rehedge=False)
    var10_static = emp_var(L10_static)

    # daily losses along the re-hedged paths, to check the i.i.d. assumption
    V10 = n_opt * hedged_values(S_10day, bs_put, bs_put_delta, rehedge=True)
    daily = -np.diff(V10, axis=1)
    sum_daily = daily.sum(axis=1)

    def skew(x):  return np.mean((x - x.mean())**3) / x.std()**3
    def ekurt(x): return np.mean((x - x.mean())**4) / x.std()**4 - 3

    print("Q3. Comparison")
    print(f"  sqrt(10) rule  : {var10_sqrt:8.4f}")
    print(f"  Time aggregation: {var10:8.4f}")
    print(f"  Ratio sqrt(10) rule / time aggregation = {var10_sqrt / var10:.4f}")
    print(f"  (For reference: 10-day VaR without re-hedging = {var10_static:.4f})")
    print()
    sq, agg, _ = replicate(bs_put, bs_put_delta)
    print("  Monte Carlo variability (20 independent runs of 40,000 paths):")
    print(f"    sqrt(10) x one-day VaR : {sq.mean():8.4f} +/- {sq.std():.4f}")
    print(f"    10-day VaR (aggregated): {agg.mean():8.4f} +/- {agg.std():.4f}")
    print(f"    Ratio of the means     : {sq.mean() / agg.mean():.4f}")
    print()
    print("  Diagnostics")
    print(f"    One-day loss   : mean {L1.mean():8.4f}, std {L1.std():8.4f}, "
          f"skew {skew(L1):6.3f}, excess kurtosis {ekurt(L1):7.3f}")
    print(f"    10-day loss    : mean {L10.mean():8.4f}, std {L10.std():8.4f}, "
          f"skew {skew(L10):6.3f}, excess kurtosis {ekurt(L10):7.3f}")
    print(f"    sqrt(10) x one-day std = {np.sqrt(J) * L1.std():.4f}")
    print(f"    Check: sum of daily losses equals 10-day loss "
          f"(max abs diff {np.abs(sum_daily - L10).max():.2e})")
    print(f"    Std of daily loss, day 1 vs day 10: {daily[:, 0].std():.4f} vs {daily[:, -1].std():.4f}")
    corr = np.corrcoef(daily.T)
    off = corr[np.triu_indices(J, 1)]
    print(f"    Correlation between daily losses: mean {off.mean():.4f}, "
          f"range [{off.min():.4f}, {off.max():.4f}]")
    print(f"    Normal approx. 10-day VaR (mean + 1.645 std): "
          f"{L10.mean() + norm.ppf(alpha) * L10.std():.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lo, hi = np.percentile(np.r_[np.sqrt(J) * L1, L10], [0.05, 99.7])
    bins = np.linspace(lo, hi, 80)
    ax.hist(L10, bins=bins, density=True, color="#2a78d6", alpha=0.55,
            label="10-day loss, daily re-hedging")
    ax.hist(np.sqrt(J) * L1, bins=bins, density=True, histtype="step", lw=2,
            color="#eb6834", label=r"$\sqrt{10}\times$ one-day loss")
    ax.axvline(var10, color="#2a78d6", ls="--", lw=1.5, label=f"Time-aggregation VaR = {var10:.2f}")
    ax.axvline(var10_sqrt, color="#eb6834", ls="--", lw=1.5, label=rf"$\sqrt{{10}}$-rule VaR = {var10_sqrt:.2f}")
    ax.set_xlabel("Loss of 100 delta-hedged short puts ($)")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(lo, hi)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("10-day loss distribution: time aggregation vs. square-root-of-time scaling")
    plt.tight_layout()
    # save next to this script, whatever the current working directory is
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw06_var_comparison.png")
    plt.savefig(fig_path, dpi=200)
    print()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
