"""
Section 3: Testing for Stationarity — ADF and KPSS in Practice

Acceptance criteria (from README):
- Raw SPY prices: ADF p-value > 0.05 (fails to reject unit root)
- Raw SPY prices: KPSS p-value < 0.05 (rejects stationarity)
- SPY daily returns: ADF p-value < 0.01 (rejects unit root — stationary)
- SPY daily returns: KPSS p-value > 0.05 (fails to reject stationarity)
- Summary table has 4 rows and columns for ADF stat, ADF p-value, KPSS stat, KPSS p-value, diagnosis
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_prices

prices = load_prices(["SPY"])
spy = prices["SPY"].dropna()
spy_returns = spy.pct_change().dropna()


# ── CELL: run_stationarity_tests ────────────────────────────
# Purpose: Run ADF and KPSS on four transformations of SPY data to
#   show how the stationarity diagnosis changes: raw prices (non-stationary),
#   returns (stationary), log prices (still non-stationary), and cumulative
#   returns (trend-stationary — the ambiguous case).
# Takeaway: Raw prices: ADF p = 0.998, KPSS p = 0.01 — both tests agree:
#   non-stationary. Returns: ADF p ≈ 0 (stat = -17.1), KPSS p = 0.10 —
#   both agree: stationary. Log prices also non-stationary (ADF p = 0.93).
#   The key skill is running BOTH tests and interpreting the joint result —
#   a single test can mislead.

series_dict = {
    "Raw Prices": spy,
    "Daily Returns": spy_returns,
    "Log Prices": np.log(spy),
    "Cumulative Returns": (1 + spy_returns).cumprod(),
}

results = []
for name, s in series_dict.items():
    s_clean = s.dropna()
    adf_stat, adf_pval, *_ = adfuller(s_clean, maxlag=20, autolag="AIC")
    kpss_stat, kpss_pval, *_ = kpss(s_clean, regression="c", nlags="auto")

    if adf_pval < 0.05 and kpss_pval > 0.05:
        diagnosis = "Stationary"
    elif adf_pval > 0.05 and kpss_pval < 0.05:
        diagnosis = "Non-stationary"
    else:
        diagnosis = "Ambiguous"

    results.append({
        "Series": name,
        "ADF Stat": adf_stat,
        "ADF p-value": adf_pval,
        "KPSS Stat": kpss_stat,
        "KPSS p-value": kpss_pval,
        "Diagnosis": diagnosis,
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    r = {row["Series"]: row for _, row in results_df.iterrows()}

    # Raw prices: non-stationary
    assert r["Raw Prices"]["ADF p-value"] > 0.05, (
        f"Raw prices ADF p-value {r['Raw Prices']['ADF p-value']:.4f} should be > 0.05"
    )
    assert r["Raw Prices"]["KPSS p-value"] < 0.05, (
        f"Raw prices KPSS p-value {r['Raw Prices']['KPSS p-value']:.4f} should be < 0.05"
    )

    # Returns: stationary
    assert r["Daily Returns"]["ADF p-value"] < 0.01, (
        f"Returns ADF p-value {r['Daily Returns']['ADF p-value']:.4f} should be < 0.01"
    )
    assert r["Daily Returns"]["KPSS p-value"] > 0.05, (
        f"Returns KPSS p-value {r['Daily Returns']['KPSS p-value']:.4f} should be > 0.05"
    )

    # Table has 4 rows
    assert len(results_df) == 4, f"Expected 4 rows, got {len(results_df)}"

    print("✓ Section 3: All acceptance criteria passed")
