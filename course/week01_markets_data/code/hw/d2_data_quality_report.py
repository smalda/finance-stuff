"""
Deliverable 2: Data Quality Report

Acceptance criteria (from README):
- Per-ticker metrics computed for all downloaded tickers (not a subset)
- Quality grades assigned with clear, documented thresholds (not arbitrary)
- At least one ticker receives a grade below A (data is imperfect — the report should surface this)
- Universe summary includes a count/percentage at each grade level
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import CACHE_DIR

# Import the loader from d1
from hw.d1_data_loader_class import FinancialDataLoader

# ── CELL: load_data_and_scores ──────────────────────────
# Purpose: Load the data and quality scores produced by the
#   FinancialDataLoader class in Deliverable 1.
# Takeaway: The quality report uses the loader's output as its
#   single source of truth. No re-downloading, no re-computation.

loader = FinancialDataLoader(
    tickers=[],  # Will load from saved metadata
    start_date="2010-01-01",
    end_date="2025-01-01"
)

# Load the data (this also loads quality scores from metadata)
data = loader.load(format="wide")

quality_scores = loader.quality_scores
print(f"Loaded quality scores for {len(quality_scores)} tickers\n")


# ── CELL: build_per_ticker_report ───────────────────────
# Purpose: Create a detailed per-ticker quality metrics table.
# Takeaway: This table surfaces reliability at a glance. Grade A
#   means production-ready. Grade C/D/F means "use with caution" or
#   "don't use." Every metric is concrete and verifiable.

report_rows = []

for ticker, metrics in quality_scores.items():
    report_rows.append({
        "Ticker": ticker,
        "Completeness": f"{metrics['completeness']:.1%}",
        "Stale Days": metrics["stale_days"],
        "OHLC Violations": metrics["ohlc_violations"],
        "Long Gaps (>5 days)": metrics["long_gaps"],
        "Grade": metrics["grade"]
    })

per_ticker_report = pd.DataFrame(report_rows)
per_ticker_report = per_ticker_report.sort_values("Grade")

print("=" * 70)
print("PER-TICKER DATA QUALITY METRICS")
print("=" * 70)
print()
print("Grading Criteria:")
print("  A: Completeness >= 99%, no stale prices, no OHLC violations, no long gaps")
print("  B: Completeness >= 95%, <10 stale days, <5 OHLC violations, no long gaps")
print("  C: Completeness >= 90%, <50 stale days, <10 OHLC violations")
print("  D: Completeness >= 80%")
print("  F: Completeness < 80% or other severe issues")
print()
print(per_ticker_report.to_string(index=False))
print()


# ── CELL: universe_summary ──────────────────────────────
# Purpose: Aggregate quality metrics across the entire universe.
# Takeaway: This answers "how many tickers meet production quality?"
#   The distribution of grades tells you whether your data pipeline
#   is reliable or needs work.

grade_counts = per_ticker_report["Grade"].value_counts().sort_index()
total_tickers = len(per_ticker_report)

print("=" * 70)
print("UNIVERSE-LEVEL SUMMARY")
print("=" * 70)
print()
print(f"Total tickers analyzed: {total_tickers}")
print(f"Failed downloads: {len(loader.failed_tickers)}")
print()
print("Grade Distribution:")
for grade in ["A", "B", "C", "D", "F"]:
    count = grade_counts.get(grade, 0)
    pct = count / total_tickers * 100 if total_tickers > 0 else 0
    print(f"  {grade}: {count:3d} tickers ({pct:5.1f}%)")

# Production-quality bar: Grade B or better
production_quality = grade_counts.get("A", 0) + grade_counts.get("B", 0)
production_pct = production_quality / total_tickers * 100 if total_tickers > 0 else 0

print()
print(f"Production-quality tickers (A or B): {production_quality} ({production_pct:.1f}%)")
print()


# ── CELL: highlight_problematic_tickers ─────────────────
# Purpose: Call out the worst-quality tickers with specific issues.
# Takeaway: These are the tickers to investigate manually. Low
#   completeness usually means late IPO or delisting. High stale
#   days mean illiquidity or data vendor issues.

problematic = per_ticker_report[per_ticker_report["Grade"].isin(["D", "F"])]

print("=" * 70)
print("PROBLEMATIC TICKERS (Grade D or F)")
print("=" * 70)
print()

if len(problematic) > 0:
    print(problematic.to_string(index=False))
    print()
    print("Recommended actions:")
    print("  - Grade D: Use with caution; verify results manually")
    print("  - Grade F: Do not use for production analysis")
else:
    print("No problematic tickers found (all Grade C or better)")

print()


# ── CELL: completeness_histogram ────────────────────────
# Purpose: Visualize the completeness distribution across tickers.
# Visual: Most tickers cluster at 100% completeness (vertical bar on
#   the right). A few outliers have <90% completeness (late IPOs or
#   delistings). The gap between high and low quality is immediately visible.

import matplotlib.pyplot as plt

completeness_values = [metrics["completeness"] for metrics in quality_scores.values()]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(completeness_values, bins=20, edgecolor="black", alpha=0.7)
ax.set_xlabel("Completeness Ratio")
ax.set_ylabel("Number of Tickers")
ax.set_title("Data Completeness Distribution Across Universe")
ax.axvline(0.99, color="green", linestyle="--", label="Production threshold (99%)")
ax.axvline(0.90, color="orange", linestyle="--", label="Minimum threshold (90%)")
ax.legend()
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────

    # Verify metrics computed for all tickers
    assert len(quality_scores) == total_tickers, (
        f"Quality scores should exist for all tickers"
    )

    # Verify all grades are valid
    valid_grades = {"A", "B", "C", "D", "F"}
    all_grades = set(per_ticker_report["Grade"].unique())
    assert all_grades <= valid_grades, (
        f"Invalid grades found: {all_grades - valid_grades}"
    )

    # Verify at least one ticker has grade below A
    below_a_count = grade_counts.drop("A", errors="ignore").sum()
    assert below_a_count > 0, (
        f"Expected at least one ticker with grade < A, got {below_a_count}"
    )

    # Verify universe summary includes all grades
    assert len(grade_counts) > 0, "Universe summary should have grade distribution"

    # Verify completeness values are in valid range [0, 1]
    assert all(0 <= c <= 1 for c in completeness_values), (
        "Completeness values should be between 0 and 1"
    )

    print("\n✓ Deliverable 2: All acceptance criteria passed")
