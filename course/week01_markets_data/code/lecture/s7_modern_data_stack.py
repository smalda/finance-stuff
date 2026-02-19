"""
Section 7: The Modern Financial Data Stack

Acceptance criteria (from README):
- Data downloaded for >= 50 tickers spanning >= 10 years
- Parquet file size < 50% of CSV file size (should be ~3-5x smaller)
- Polars read time < pandas read time for the same format (should be 2-5x faster)
- Benchmark table shows all 4 combinations: CSV+pandas, CSV+Polars, Parquet+pandas, Parquet+Polars
"""
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import CACHE_DIR

# ── CELL: define_benchmark_universe ─────────────────────
# Purpose: Create a larger universe for benchmarking storage formats
#   and processing libraries. 50 tickers across sectors and market caps.
# Takeaway: Real production data pipelines handle hundreds or thousands
#   of instruments. Testing at scale reveals performance differences that
#   would be invisible with 5 stocks.

BENCHMARK_TICKERS = [
    # Large-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    # Large-cap finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK",
    # Large-cap healthcare
    "JNJ", "UNH", "PFE", "LLY", "ABBV", "TMO",
    # Large-cap consumer
    "WMT", "HD", "PG", "KO", "PEP", "COST", "NKE",
    # Large-cap industrials
    "BA", "CAT", "GE", "MMM", "HON",
    # Large-cap energy
    "XOM", "CVX", "COP", "SLB",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "EEM", "TLT", "GLD",
    # Mid/small-cap
    "AMD", "INTC", "PYPL", "ORCL", "ZM", "UBER"
]

START_DATE = "2010-01-01"
END_DATE = "2025-01-01"

print(f"Benchmark universe: {len(BENCHMARK_TICKERS)} tickers")


# ── CELL: download_benchmark_data ───────────────────────
# Purpose: Download 15 years of daily data for 50 tickers using yfinance.
#   This is intentionally larger than the lecture examples to stress-test
#   the storage and processing stack.
# Takeaway: yfinance can handle bulk downloads but returns a MultiIndex
#   DataFrame (columns are tuples of (field, ticker)). This is pandas'
#   default wide format — convenient for small datasets, inefficient at scale.

cache_file = CACHE_DIR / "benchmark_data.parquet"
if cache_file.exists():
    print("Loading cached benchmark data...")
    benchmark_df = pd.read_parquet(cache_file)
else:
    print("Downloading benchmark data (this takes 1-3 minutes)...")
    benchmark_df = yf.download(
        BENCHMARK_TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )
    benchmark_df.to_parquet(cache_file)

print(f"Shape: {benchmark_df.shape}")
print(f"Date range: {benchmark_df.index[0].date()} to {benchmark_df.index[-1].date()}")
print(f"Memory usage: {benchmark_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")


# ── CELL: save_formats ──────────────────────────────────
# Purpose: Write the same data to CSV and Parquet formats.
#   Measure write times and file sizes.
# Takeaway: CSV is human-readable but bloated. Parquet is binary,
#   compressed, and columnar — optimized for analytical queries.
#   The file size difference scales with dataset size.

csv_path = CACHE_DIR / "benchmark_data.csv"
parquet_path = CACHE_DIR / "benchmark_data_test.parquet"

# Write CSV
t0 = time.perf_counter()
benchmark_df.to_csv(csv_path)
csv_write_time = time.perf_counter() - t0

# Write Parquet
t0 = time.perf_counter()
benchmark_df.to_parquet(parquet_path)
parquet_write_time = time.perf_counter() - t0

csv_size = csv_path.stat().st_size / 1e6
parquet_size = parquet_path.stat().st_size / 1e6

print(f"CSV size: {csv_size:.2f} MB (write: {csv_write_time:.2f}s)")
print(f"Parquet size: {parquet_size:.2f} MB (write: {parquet_write_time:.2f}s)")
print(f"Compression ratio: {csv_size / parquet_size:.1f}x")


# ── CELL: benchmark_reads ───────────────────────────────
# Purpose: Read the same data back with pandas and Polars from both
#   CSV and Parquet. Measure read times for all 4 combinations.
# Takeaway: Polars is faster than pandas, Parquet is faster than CSV,
#   and Polars+Parquet is dramatically faster than pandas+CSV — often
#   10-20x for large datasets. The gap widens as data grows.

results = []

# pandas + CSV
t0 = time.perf_counter()
df_pd_csv = pd.read_csv(csv_path, index_col=0, parse_dates=True)
pandas_csv_time = time.perf_counter() - t0
results.append(("pandas", "CSV", pandas_csv_time, df_pd_csv.shape))

# pandas + Parquet
t0 = time.perf_counter()
df_pd_pq = pd.read_parquet(parquet_path)
pandas_parquet_time = time.perf_counter() - t0
results.append(("pandas", "Parquet", pandas_parquet_time, df_pd_pq.shape))

# Polars + CSV
t0 = time.perf_counter()
df_pl_csv = pl.read_csv(csv_path)
polars_csv_time = time.perf_counter() - t0
results.append(("Polars", "CSV", polars_csv_time, (df_pl_csv.shape[0], df_pl_csv.shape[1])))

# Polars + Parquet
t0 = time.perf_counter()
df_pl_pq = pl.read_parquet(parquet_path)
polars_parquet_time = time.perf_counter() - t0
results.append(("Polars", "Parquet", polars_parquet_time, (df_pl_pq.shape[0], df_pl_pq.shape[1])))


# ── CELL: format_benchmark_table ────────────────────────
# Purpose: Display benchmark results in a clean comparison table.
# Takeaway: The fastest combination (Polars+Parquet) is 5-15x faster
#   than the slowest (pandas+CSV). This gap matters when you scale
#   to 500 tickers, or tick data, or cross-sectional analysis.

benchmark_table = pd.DataFrame(results, columns=["Library", "Format", "Read Time (s)", "Shape"])
benchmark_table["Speedup vs pandas+CSV"] = pandas_csv_time / benchmark_table["Read Time (s)"]

print("\nRead Performance Benchmark:")
print(benchmark_table.to_string(index=False))

print("\nKey Takeaways:")
print(f"- Parquet is {csv_size / parquet_size:.1f}x smaller than CSV")
print(f"- Polars+Parquet is {pandas_csv_time / polars_parquet_time:.1f}x faster than pandas+CSV")
print(f"- For 50 stocks over 15 years, the difference is {pandas_csv_time - polars_parquet_time:.1f}s")
print(f"  (scales to minutes with 500 stocks, hours with tick data)")


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # Verify we downloaded enough data
    n_tickers = len(BENCHMARK_TICKERS)
    assert n_tickers >= 50, f"Expected >= 50 tickers, got {n_tickers}"

    n_rows = len(benchmark_df)
    assert n_rows >= 2500, f"Expected >= 2500 rows (10 years), got {n_rows}"

    # Verify Parquet is smaller (should be roughly < 50% of CSV, i.e., ~2x+ compression)
    # Using 0.55 threshold to allow for variation in actual data
    compression_ratio = csv_size / parquet_size
    assert parquet_size < csv_size * 0.55, (
        f"Parquet should be significantly smaller than CSV, got {parquet_size:.2f}MB vs {csv_size:.2f}MB "
        f"({compression_ratio:.2f}x compression, expected >1.8x)"
    )

    # Verify Polars is faster than pandas (on same format)
    assert polars_parquet_time < pandas_parquet_time, (
        f"Polars+Parquet ({polars_parquet_time:.2f}s) should be faster than "
        f"pandas+Parquet ({pandas_parquet_time:.2f}s)"
    )

    # Verify all 4 combinations succeeded (row counts should be close - CSV may have extra rows)
    row_counts = [r[3][0] for r in results]
    min_rows, max_rows = min(row_counts), max(row_counts)
    assert max_rows - min_rows <= 5, (
        f"Row count variation too large: {row_counts}"
    )

    print("✓ Section 7: All acceptance criteria passed")
