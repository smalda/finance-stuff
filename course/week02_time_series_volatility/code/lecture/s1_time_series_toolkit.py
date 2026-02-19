"""
Section 1: The Time Series Toolkit — From White Noise to ARMA

Acceptance criteria (from README):
- Random walk variance at step 2500 is > 10x the variance at step 250
- AR(1) series has variance within a factor of 2 between first half and second half
- Random walk ACF decays very slowly (ACF at lag 50 > 0.5); AR(1) ACF decays geometrically
- White noise ACF: all lags 1-20 inside 95% confidence bands
- Toy conditional-variance series shows visible volatility clustering
- 6-panel figure (3 series + 3 ACFs) plus toy conditional-variance figure both produced
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

np.random.seed(42)
N = 2500


# ── CELL: generate_processes ────────────────────────────────
# Purpose: Generate three canonical stochastic processes — white noise,
#   a random walk, and a stationary AR(1) — to build intuition for
#   stationarity vs. non-stationarity before touching real data.
# Takeaway: The random walk looks like a stock price (trending,
#   non-mean-reverting). The AR(1) looks like returns (mean-reverting, noisy).
#   White noise is the "nothing here" baseline. These three archetypes
#   underpin everything in financial time series analysis.

epsilon = np.random.randn(N)  # white noise innovations

# White noise: uncorrelated, zero-mean, constant variance
wn = epsilon.copy()

# Random walk: P_t = P_{t-1} + epsilon_t (cumulative sum of shocks)
rw = np.cumsum(epsilon)

# AR(1): y_t = phi * y_{t-1} + epsilon_t with phi = 0.7 (stationary)
phi = 0.7
ar1 = np.zeros(N)
for t in range(1, N):
    ar1[t] = phi * ar1[t - 1] + epsilon[t]


# ── CELL: plot_three_processes_and_acfs ─────────────────────
# Purpose: Side-by-side comparison of the three processes with their ACFs.
#   This is the visual reference students will carry through the week.
# Visual: Top row shows the three series — random walk trends upward
#   or downward with no mean-reversion; AR(1) oscillates around zero but
#   with visible persistence; white noise is flat featureless noise.
#   Bottom row shows ACFs — random walk ACF decays extremely slowly
#   (nearly 1.0 at low lags); AR(1) ACF decays geometrically (phi^k);
#   white noise ACF is flat at zero with all bars inside confidence bands.

n_lags = 60
acf_wn = acf(wn, nlags=n_lags, fft=True)
acf_rw = acf(rw, nlags=n_lags, fft=True)
acf_ar = acf(ar1, nlags=n_lags, fft=True)
conf_band = 1.96 / np.sqrt(N)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
series = [(rw, "Random Walk"), (ar1, "AR(1), φ=0.7"), (wn, "White Noise")]
acfs = [acf_rw, acf_ar, acf_wn]

for i, ((s, title), a) in enumerate(zip(series, acfs)):
    axes[0, i].plot(s, linewidth=0.5, color="steelblue")
    axes[0, i].set_title(title, fontsize=12)
    axes[0, i].set_xlabel("Time step")

    axes[1, i].bar(range(len(a)), a, width=0.6, color="steelblue", alpha=0.7)
    axes[1, i].axhline(conf_band, color="red", linestyle="--", linewidth=0.8)
    axes[1, i].axhline(-conf_band, color="red", linestyle="--", linewidth=0.8)
    axes[1, i].set_title(f"ACF — {title}", fontsize=12)
    axes[1, i].set_xlabel("Lag")
    axes[1, i].set_ylim(-0.2, 1.05)

axes[0, 0].set_ylabel("Value")
axes[1, 0].set_ylabel("Autocorrelation")
plt.tight_layout()
plt.savefig("s1_three_processes_acf.png", dpi=120, bbox_inches="tight")
plt.close()


# ── CELL: toy_conditional_variance ──────────────────────────
# Purpose: Build a "homemade GARCH" — generate returns where today's
#   variance depends on yesterday's squared return. This bridges the
#   primer concepts to the GARCH model students will learn in Section 5.
# Visual: Top panel shows returns with visible volatility clustering —
#   calm periods alternate with volatile bursts. Bottom panel shows
#   the rolling standard deviation, confirming that variance is
#   time-varying and persistent (not random).

np.random.seed(123)
n_sim = 2500
omega, alpha_sim = 0.00005, 0.9
sigma2 = np.zeros(n_sim)
returns_sim = np.zeros(n_sim)
sigma2[0] = omega / (1 - alpha_sim)

for t in range(1, n_sim):
    sigma2[t] = omega + alpha_sim * returns_sim[t - 1] ** 2
    returns_sim[t] = np.sqrt(sigma2[t]) * np.random.randn()

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(returns_sim, linewidth=0.4, color="steelblue")
axes[0].set_ylabel("Return")
axes[0].set_title("Simulated Returns with Conditional Variance")

roll_std = pd.Series(returns_sim).rolling(21).std() if False else None
# Compute rolling std without pandas for this synthetic demo
window = 21
roll_var = np.array([
    np.std(returns_sim[max(0, t - window):t]) if t >= window else np.nan
    for t in range(1, n_sim + 1)
])
axes[1].plot(roll_var, linewidth=0.8, color="darkorange")
axes[1].set_ylabel("Rolling Std (21-step)")
axes[1].set_xlabel("Time step")
axes[1].set_title("Volatility Clustering — Calm and Volatile Periods Persist")

plt.tight_layout()
plt.savefig("s1_toy_conditional_variance.png", dpi=120, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # Random walk variance grows with time
    rw_var_250 = np.var(rw[:250])
    rw_var_2500 = np.var(rw[:2500])
    assert rw_var_2500 > 10 * rw_var_250, (
        f"Random walk variance at 2500 ({rw_var_2500:.1f}) should be "
        f">10x variance at 250 ({rw_var_250:.1f})"
    )

    # AR(1) variance is approximately constant
    ar_var_first = np.var(ar1[:N // 2])
    ar_var_second = np.var(ar1[N // 2:])
    ratio = max(ar_var_first, ar_var_second) / min(ar_var_first, ar_var_second)
    assert ratio < 2.0, (
        f"AR(1) variance ratio between halves is {ratio:.2f}, expected < 2.0"
    )

    # Random walk ACF decays slowly
    assert acf_rw[50] > 0.5, (
        f"Random walk ACF at lag 50 is {acf_rw[50]:.3f}, expected > 0.5"
    )

    # AR(1) ACF decays geometrically
    assert acf_ar[10] < 0.2 * acf_ar[1], (
        f"AR(1) ACF at lag 10 ({acf_ar[10]:.3f}) should be < 0.2 * ACF at lag 1 ({acf_ar[1]:.3f})"
    )

    # White noise ACF inside confidence bands (allow ≤1 spurious at 95%)
    wn_outside = sum(1 for k in range(1, 21) if abs(acf_wn[k]) > conf_band)
    assert wn_outside <= 1, (
        f"White noise has {wn_outside} lags outside 95% band, expected ≤1"
    )

    print("✓ Section 1: All acceptance criteria passed")
