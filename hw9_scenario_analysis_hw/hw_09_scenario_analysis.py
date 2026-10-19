import itertools
import numpy as np

# Parameters
r = 0.0132
sig1_0, sig2_0, rho_0 = 0.2214, 0.3036, 0.237
T = 0.25
S1_0, S2_0 = 158.12, 170.33
dt = 1 / 252
h = 5 * dt  # 5-day horizon


def price(tau, s1, s2, sig1, sig2, rho):
    """Option price from part (a)."""
    return np.exp(r * tau) * (
        s1**2 * np.exp(sig1**2 * tau)
        - 2 * s1 * s2 * np.exp(sig1 * sig2 * rho * tau)
        + s2**2 * np.exp(sig2**2 * tau)
    )


def deltas(tau, s1, s2, sig1, sig2, rho):
    """Partial derivatives of the price w.r.t. s1 and s2."""
    c = np.exp(sig1 * sig2 * rho * tau)
    h1 = np.exp(r * tau) * (2 * s1 * np.exp(sig1**2 * tau) - 2 * s2 * c)
    h2 = np.exp(r * tau) * (2 * s2 * np.exp(sig2**2 * tau) - 2 * s1 * c)
    return h1, h2


# Portfolio at t = 0: long 1 option, short h1, h2 shares
V_opt0 = price(T, S1_0, S2_0, sig1_0, sig2_0, rho_0)
h1, h2 = deltas(T, S1_0, S2_0, sig1_0, sig2_0, rho_0)
V0 = V_opt0 - h1 * S1_0 - h2 * S2_0

# Scenarios and weights
X_vals = {0.20: 0.5, -0.20: 0.5, 0.10: 0.75, -0.10: 0.75, 0.05: 1.0, -0.05: 1.0}
sig_vals = {0.5: 0.5, 2.0: 0.5, 0.75: 1.25, 1.75: 1.25, 1.25: 0.75, 1.5: 0.75}
drho_vals = {0.0: 0.5, 0.5: 1.25, -0.5: 1.25}

rows = []
for x1, x2, k1, k2, dr in itertools.product(X_vals, X_vals, sig_vals, sig_vals, drho_vals):
    S1 = S1_0 * np.exp(x1)
    S2 = S2_0 * np.exp(x2)
    V_opt = price(T - h, S1, S2, k1 * sig1_0, k2 * sig2_0, rho_0 + dr)
    V = V_opt - h1 * S1 - h2 * S2
    loss = -(V - V0)
    w = X_vals[x1] * X_vals[x2] * sig_vals[k1] * sig_vals[k2] * drho_vals[dr]
    rows.append((x1, x2, k1, k2, dr, loss, w, w * loss))

assert len(rows) == 3888

print(f"Initial option value V(0)      = {V_opt0:.4f}")
print(f"Deltas h1, h2                  = {h1:.4f}, {h2:.4f}")
print(f"Initial portfolio value V0     = {V0:.4f}\n")


def show(label, row):
    x1, x2, k1, k2, dr, loss, w, wl = row
    print(label)
    print(f"  X1 = {x1:+.2f}, X2 = {x2:+.2f}")
    print(f"  sigma1 ratio = {k1}, sigma2 ratio = {k2}  "
          f"(sigma1 = {k1*sig1_0:.4f}, sigma2 = {k2*sig2_0:.4f})")
    print(f"  delta rho = {dr:+.1f}  (rho = {rho_0+dr:.3f})")
    print(f"  loss = {loss:.4f}, weight = {w:.6f}, weighted loss = {wl:.4f}\n")


show("(b) Worst-case scenario (max loss):", max(rows, key=lambda r_: r_[5]))
show("(c) Max weighted loss scenario:", max(rows, key=lambda r_: r_[7]))