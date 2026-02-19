"""
Section 6: Corporate Actions — The Silent Data Corruptors

Acceptance criteria (from README):
- Reconstructed nominal AAPL price on 2014-06-06 > $600 (pre-split)
- Reconstructed nominal AAPL price on 2014-06-09 < $100 (post-split)
- Nominal daily return on split date < -80%
- Adjusted daily return on split date within +/-5% (normal trading day)
- Plot shows visually dramatic discontinuity in nominal line vs. smooth adjusted line

NOTE: yfinance's Close with auto_adjust=False is ALREADY split-adjusted.
To show truly unadjusted (nominal) prices, we reverse split adjustments
using yf.Ticker().splits data.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_or_download, load_adjusted, CACHE_DIR


# ── CELL: download_splits ───────────────────────────────
# Purpose: Fetch AAPL's split history from yfinance. We need this to
#   reverse-engineer nominal (truly unadjusted) prices from the
#   split-adjusted Close that yfinance provides.
# Takeaway: yfinance's "Close" with auto_adjust=False is already
#   split-adjusted — it divides all historical prices by the cumulative
#   split factor. To reconstruct what the stock ACTUALLY traded at on
#   any given day, we must reverse this adjustment using the split history.

cache_file = CACHE_DIR / "aapl_splits.parquet"
if cache_file.exists():
    splits = pd.read_parquet(cache_file).squeeze()
else:
    aapl = yf.Ticker("AAPL")
    splits = aapl.splits
    splits.to_frame().to_parquet(cache_file)

splits.index = splits.index.tz_localize(None)
print("AAPL split history:")
print(splits[splits > 0])


# ── CELL: reconstruct_nominal ───────────────────────────
# Purpose: Build the cumulative split factor and multiply it back
#   into the split-adjusted Close to recover nominal prices.
# Takeaway: The split-adjusted close is nominal_price / cumulative_factor.
#   So nominal_price = close * cumulative_factor. The factor changes at
#   each split date and is constant between splits.

raw = load_or_download()
close_aapl = raw["Close"]["AAPL"].dropna().copy()

# Build cumulative split factor from the END backwards
# At the end of the series the factor is 1.0 (current prices are nominal).
# Going back in time, each split MULTIPLIES the factor.
cumulative_factor = pd.Series(1.0, index=close_aapl.index)
for split_date, ratio in splits.items():
    if ratio > 0 and split_date >= close_aapl.index[0]:
        cumulative_factor.loc[:split_date - pd.Timedelta(days=1)] *= ratio

nominal_close = close_aapl * cumulative_factor


# ── CELL: inspect_split_date ────────────────────────────
# Purpose: Zoom in on the AAPL 7:1 split date (June 9, 2014).
#   Show nominal vs. adjusted prices and returns around the event.
# Takeaway: In nominal terms, AAPL "crashed" 85% in one day. In
#   adjusted terms, it was a normal trading day. If your model trains
#   on nominal prices, it learns a phantom catastrophe.

adj = load_adjusted()
adj_close_aapl = adj["Close"]["AAPL"].dropna()

split_date = "2014-06-09"
pre_date = "2014-06-06"

pre_nominal = nominal_close.loc[pre_date]
post_nominal = nominal_close.loc[split_date]
nominal_return = (post_nominal - pre_nominal) / pre_nominal

pre_adj = adj_close_aapl.loc[pre_date]
post_adj = adj_close_aapl.loc[split_date]
adj_return = (post_adj - pre_adj) / pre_adj

print(f"Nominal price 2014-06-06: ${pre_nominal:.2f}")
print(f"Nominal price 2014-06-09: ${post_nominal:.2f}")
print(f"Nominal daily return on split: {nominal_return:.2%}")
print(f"Adjusted daily return on split: {adj_return:.2%}")


# ── CELL: plot_nominal_vs_adjusted ──────────────────────
# Purpose: Plot nominal vs. adjusted AAPL close prices from 2010 to present.
# Visual: The adjusted line is smooth and continuously rising. The nominal
#   line shows dramatic discontinuities — a cliff in June 2014 (7:1 split)
#   and another in August 2020 (4:1 split). The contrast is striking and
#   makes the data corruption immediately obvious.

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(nominal_close.index, nominal_close.values,
        label="Nominal (unadjusted)", color="#C62828", linewidth=0.8)
ax.plot(adj_close_aapl.index, adj_close_aapl.values,
        label="Adjusted close", color="#1565C0", linewidth=0.8)

ax.axvline(pd.Timestamp("2014-06-09"), color="gray", linestyle="--",
           alpha=0.5, label="7:1 split (Jun 2014)")
ax.axvline(pd.Timestamp("2020-08-31"), color="gray", linestyle=":",
           alpha=0.5, label="4:1 split (Aug 2020)")

ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.set_title("AAPL: Nominal vs. Adjusted Close — The Split Illusion")
ax.legend()
plt.tight_layout()
plt.show()


# ── CELL: return_comparison_table ────────────────────────
# Purpose: Build a table comparing nominal and adjusted returns
#   around both AAPL split dates.
# Takeaway: On split dates, nominal returns are catastrophically wrong.
#   Between splits, they're identical. This means the damage is concentrated
#   on specific dates — but those dates create training signals that a model
#   will weight heavily because they look like extreme events.

split_dates = ["2014-06-09", "2020-08-31"]
rows = []
for sd in split_dates:
    prev = nominal_close.index[nominal_close.index.get_loc(sd) - 1]
    nom_ret = (nominal_close.loc[sd] - nominal_close.loc[prev]) / nominal_close.loc[prev]
    adj_ret = (adj_close_aapl.loc[sd] - adj_close_aapl.loc[prev]) / adj_close_aapl.loc[prev]
    rows.append({
        "Split Date": sd,
        "Nominal Return": f"{nom_ret:.2%}",
        "Adjusted Return": f"{adj_ret:.2%}",
        "Error": f"{abs(nom_ret - adj_ret):.2%}",
    })

comparison = pd.DataFrame(rows)
print(comparison.to_string(index=False))


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    assert pre_nominal > 600, f"Pre-split nominal should be > $600, got {pre_nominal:.2f}"
    assert post_nominal < 100, f"Post-split nominal should be < $100, got {post_nominal:.2f}"
    assert nominal_return < -0.80, f"Nominal return should be < -80%, got {nominal_return:.2%}"
    assert abs(adj_return) < 0.05, f"Adjusted return should be within +/-5%, got {adj_return:.2%}"

    # Verify the plot would show dramatic discontinuity: nominal range >> adjusted range
    nom_range = nominal_close.max() - nominal_close.min()
    adj_range = adj_close_aapl.max() - adj_close_aapl.min()
    assert nom_range > adj_range * 2, (
        f"Nominal range ({nom_range:.0f}) should be >> adjusted range ({adj_range:.0f})"
    )

    print("Section 6: All acceptance criteria passed")
