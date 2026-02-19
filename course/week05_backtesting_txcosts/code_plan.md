# Code Plan — Week 5: Backtesting, Research Discipline & Transaction Costs

---

## Metric Implementability Check

Before the criteria map, each distinct metric was verified for an
implementation path. Results:

| Metric | Implementation path | Status |
|--------|---------------------|--------|
| IC (cross-sectional Spearman) | `shared.metrics.ic_summary`, `rank_ic` | Available |
| ICIR, t-stat, p-value | `shared.metrics.ic_summary` | Available |
| `⚠ WEAK SIGNAL` check | `shared.metrics.ic_summary` (auto-printed when t < 1.96) | Available |
| Survivorship bias premium | Computed inline: equal-weight 449 vs. complete-history-only portfolio | Standard pandas arithmetic |
| Annualized Sharpe ratio | `shared.backtesting.sharpe_ratio(returns, periods_per_year=252|12)` | Available |
| Max drawdown | `shared.backtesting.max_drawdown` | Available |
| Net returns (flat spread) | `shared.backtesting.net_returns(gross, turnover, cost_bps)` | Available |
| Portfolio turnover | `shared.backtesting.portfolio_turnover` | Available |
| Long-short portfolio construction | `shared.backtesting.long_short_returns` | Available |
| Deflated Sharpe ratio (DSR) | `shared.metrics.deflated_sharpe_ratio` | Available |
| Probabilistic Sharpe Ratio (PSR) | scipy.stats.norm.cdf — used inside DSR implementation | scipy |
| Skewness / excess kurtosis | `scipy.stats.skew`, `scipy.stats.kurtosis(fisher=True)` | scipy |
| MinTRL formula | Bailey-López de Prado (2014) formula — ~10 lines inline | inline |
| BHY multiple-testing correction | `statsmodels.stats.multitest.multipletests(method='fdr_bh')` | statsmodels |
| CPCV (combinatorial purged CV) | `shared.temporal.CombinatorialPurgedCV` | Available |
| Walk-forward splits | `shared.temporal.walk_forward_splits` | Available |
| PurgedWalkForwardCV | `shared.temporal.PurgedWalkForwardCV` | Available |
| TimeSeriesSplit | `sklearn.model_selection.TimeSeriesSplit` | sklearn |
| PurgedKFold class | Built from scratch in D1 (first-introduction week) | Teach from scratch |
| TransactionCostModel | Built from scratch in D2 (first-introduction week) | Teach from scratch |
| Corwin-Schultz spread estimator | Corwin & Schultz (2012) formula — ~20 lines inline | inline |
| Market impact (Almgren-Chriss sqrt law) | inline formula: σ × η × sqrt(participation_rate) | inline |
| IS/OOS IC ratio | Standard ratio arithmetic | inline |
| Feasibility frontier | Net Sharpe computed over grid; zero-crossing identified | inline |
| PBO (probability of backtest overfitting) | CPCV paths computed via CombinatorialPurgedCV; fraction below median rank | inline (20 lines) |
| quantstats tear sheet | `quantstats.reports.basic(returns, benchmark)` | quantstats |

No criteria are dropped. All have verified implementation paths.

---

## Week 4 Dependency Reality Check

**What Week 4 built:**
- Date range: 2014-01-01 to 2024-12-31, with 3-year train window → OOS from April 2019
- OOS months: 68 (April 2019 – November 2024)
- Universe: 174 tickers (those with complete history in Week 4's data range)
- Cached: `gbm_predictions.parquet`, `gbm_ic_series.parquet`, `nn_predictions.parquet`,
  `nn_ic_series.parquet`, `expanded_features.parquet` (18 features)
- Ridge predictions: NOT cached separately (only GBM, NN cached)
- Feature matrix: `expanded_features.parquet`, `feature_matrix.parquet`

**Mismatch with expectations.md:** expectations.md assumed 449 tickers and 132
OOS months (2015–2025). The actual Week 4 data is 174 tickers, 68 OOS months.
This is a known open question from expectations.md — contingencies are coded in
data_setup.py. All acceptance criteria are calibrated to actual Week 4 data, not
the expectations assumption.

**Synthetic fallback:** activated if Week 4 cache files are missing. Uses 12-1
month momentum on the 449-ticker universe. Both paths produce comparable
methodology demonstrations.

---

## Criteria Map

| # | Criterion | Assertion range | Target file | Notes |
|---|-----------|-----------------|-------------|-------|
| S1-1 | Flawed daily signal OOS IC ≥ 0.30 | [0.30, 1.00] | lecture/s1_seven_sins.py | Look-ahead signal uses outcome as predictor |
| S1-2 | Corrected daily signal OOS IC ∈ [0.00, 0.06] | [−0.05, 0.06] | lecture/s1_seven_sins.py | Random walk / weak momentum |
| S1-3 | Flawed vs. corrected equity curve annual return gap ≥ 15pp | [0.15, ∞) | lecture/s1_seven_sins.py | Look-ahead premium visible |
| S1-4 | Survivorship bias premium ∈ [0.5%, 4.0%] annualized | [0.005, 0.040] | lecture/s1_seven_sins.py | Sign must be positive (survivors inflate) |
| S1-5 | Data quality block printed | structural check | lecture/s1_seven_sins.py | N_stocks, N_periods, missing pct, survivorship note |
| S2-1 | TimeSeriesSplit (gap=21) mean CV IC ∈ [0.02, 0.10] | [0.02, 0.10] | lecture/s2_purged_cv.py | Walk-forward baseline |
| S2-2 | PurgedKFold mean CV IC ∈ [0.01, 0.08] | [0.01, 0.08] | lecture/s2_purged_cv.py | Must be ≤ TimeSeriesSplit IC |
| S2-3 | IC delta (WF minus purged) ≥ 0.005 | [0.005, ∞) | lecture/s2_purged_cv.py | If < 0.005: flag, do not fail |
| S2-4 | len(list(pkf.split(X, y))) == k | structural | lecture/s2_purged_cv.py | Exactly k folds |
| S2-5 | Split visualization: train/test/purged/embargo zones labeled | visual | lecture/s2_purged_cv.py | Step 5 verifies |
| S3-1 | CPCV produces exactly 15 paths (k=6) | [15, 15] | lecture/s3_cpcv_multiple_testing.py | len(cpcv_paths) == 15 |
| S3-2 | PBO ∈ [0.20, 0.75] | [0.20, 0.75] | lecture/s3_cpcv_multiple_testing.py | < 0.20: leakage suspect; > 0.75: pure noise |
| S3-3 | OOS returns distribution has std > 0 | (0, ∞) | lecture/s3_cpcv_multiple_testing.py | Non-degenerate CPCV |
| S3-4 | BHY adjusted p-values ≥ raw p-values | structural | lecture/s3_cpcv_multiple_testing.py | BHY only inflates |
| S3-5 | N_passing_t300 == 0 for M=3 model variants | [0, 3] | lecture/s3_cpcv_multiple_testing.py | Expected: 0 clear t>3.0 threshold |
| S4-1 | Monthly one-way turnover ∈ [30%, 200%] | [0.30, 2.00] | lecture/s4_transaction_costs.py | Naive full-rebalance with gross leverage=2 gives ~140%; expectations.md said [30%,100%] but did not account for leverage. 140% confirmed in data_setup run. |
| S4-2 | Annualized spread drag (regime b) ∈ [0.5%, 4.0%] | [0.005, 0.040] | lecture/s4_transaction_costs.py | Fixed 10 bps spread |
| S4-3 | Market impact cost > 0 | (0, ∞) | lecture/s4_transaction_costs.py | Must be positive for any trading period |
| S4-4 | Net annualized Sharpe < gross Sharpe | structural | lecture/s4_transaction_costs.py | TC always drag; net > gross = bug |
| S4-5 | Three-regime equity curves: zero > fixed > tiered at every point | structural ordering | lecture/s4_transaction_costs.py | Sign error check |
| S4-6 | `⚠ HIGH TURNOVER` printed if one-way monthly turnover > 50% | structural | lecture/s4_transaction_costs.py | Per rigor.md §3.1 |
| S4-7 | Skewness + excess kurtosis reported alongside Sharpe | structural | lecture/s4_transaction_costs.py | Per rigor.md §3.3 |
| S5-1 | PSR outputs values in [0, 1] | [0.0, 1.0] | lecture/s5_deflated_sharpe.py | Formula check |
| S5-2 | DSR surface: monotone in M (DSR(M=1) ≥ DSR(M=10) ≥ DSR(M=50)) | structural | lecture/s5_deflated_sharpe.py | For any fixed T > 12 |
| S5-3 | DSR surface: monotone in T (DSR(T=120) ≥ DSR(T=36) ≥ DSR(T=24)) | structural | lecture/s5_deflated_sharpe.py | For any fixed M > 1 |
| S5-4 | MinTRL ∈ [6, 30] months at 95% confidence | [6, 30] | lecture/s5_deflated_sharpe.py | For observed annual Sharpe 0.5–1.2 |
| S5-5 | Skewness, excess kurtosis reported alongside all Sharpe | structural | lecture/s5_deflated_sharpe.py | Per rigor.md §3.3 |
| S6-1 | Naive equity curve annualized Sharpe ∈ [0.5, 1.8] | [0.5, 1.8] | lecture/s6_responsible_backtest.py | No purging, no TC, no DSR |
| S6-2 | Responsible Sharpe (net, purged) ∈ [0.1, 1.2] | [0.1, 1.2] | lecture/s6_responsible_backtest.py | Must be strictly < naive Sharpe |
| S6-3 | Naive − responsible Sharpe gap ∈ [0.1, 0.8] | [0.1, 0.8] | lecture/s6_responsible_backtest.py | The backtest-to-live gap |
| S6-4 | Sub-period IC split printed (2019–2021 vs 2022–2024) | structural | lecture/s6_responsible_backtest.py | Per rigor.md §3.2; using actual OOS dates |
| S6-5 | IS/OOS labeled in all output and plots | structural | lecture/s6_responsible_backtest.py | Per rigor.md §1.5 |
| EX1-1 | Signal A mean IS IC ≥ 0.50 | [0.50, 1.00] | seminar/ex1_bug_hunt.py | Look-ahead IC near 1.0 |
| EX1-2 | Signal A mean OOS IC ≤ 0.10 | [−0.10, 0.10] | seminar/ex1_bug_hunt.py | Collapses OOS |
| EX1-3 | Signal A IS/OOS IC ratio ≥ 5× | [5.0, ∞) | seminar/ex1_bug_hunt.py | Dramatic collapse pattern |
| EX1-4 | Signal B mean IS IC ≥ 0.03 | [0.03, ∞) | seminar/ex1_bug_hunt.py | Survivorship inflates both halves |
| EX1-5 | Signal B IS/OOS IC ratio ≤ 2× | [0, 2.0] | seminar/ex1_bug_hunt.py | Near-equal inflation, no collapse |
| EX1-6 | Signal C mean IS IC ∈ [0.01, 0.06] | [0.01, 0.06] | seminar/ex1_bug_hunt.py | Correct signal baseline |
| EX1-7 | Signal C IS/OOS IC ratio ∈ [0.8, 2.0] | [0.8, 2.0] | seminar/ex1_bug_hunt.py | Stable, no collapse |
| EX2-1 | Walk-forward mean CV IC ∈ [0.01, 0.10] | [0.01, 0.10] | seminar/ex2_purging_comparison.py | WF baseline on Week 4 predictions |
| EX2-2 | PurgedKFold mean CV IC ≤ walk-forward IC | structural | seminar/ex2_purging_comparison.py | Purging reduces leakage |
| EX2-3 | IC delta ≥ 0.002 | [0.002, ∞) | seminar/ex2_purging_comparison.py | Leakage premium; if near zero: note |
| EX2-4 | RANK_FLIP_FOLDS reported for available model variants | structural | seminar/ex2_purging_comparison.py | Count folds where rank changes |
| EX3-1 | Feasibility frontier monotonically decreasing in spread | structural | seminar/ex3_tc_sensitivity.py | Higher spread → more turnover reduction needed |
| EX3-2 | At 30 bps spread, 0% reduction: net Sharpe ≤ gross Sharpe − 0.3 | structural | seminar/ex3_tc_sensitivity.py | Significant cost drag |
| EX3-3 | Grid heatmap generated with visible viable/non-viable zones | visual | seminar/ex3_tc_sensitivity.py | Step 5 verifies |
| EX3-4 | `⚠ STRATEGY NON-VIABLE` printed if FEASIBLE_CELLS == 0 | structural | seminar/ex3_tc_sensitivity.py | Not a code failure — valid result |
| EX3-5 | GROSS_SHARPE, FEASIBLE_CELLS, BREAKEVEN_SPREAD printed | structural | seminar/ex3_tc_sensitivity.py | Structured output |
| EX4-1 | DSR surface 5×5 grid: all cells ∈ [0.0, 1.0] | [0.0, 1.0] each cell | seminar/ex4_dsr_calibration.py | Formula validity |
| EX4-2 | DSR decreases weakly as M increases (fixed T) | structural monotone | seminar/ex4_dsr_calibration.py | More trials → lower DSR |
| EX4-3 | DSR(T=6, M=50) ≤ 0.50 | [0.0, 0.50] | seminar/ex4_dsr_calibration.py | Short track record + many trials |
| EX4-4 | DSR(T=60, M=1) ≥ 0.90 | [0.90, 1.00] | seminar/ex4_dsr_calibration.py | Long track record + single trial |
| EX4-5 | Skewness and excess kurtosis computed from actual returns (not 0.0 defaults) | structural | seminar/ex4_dsr_calibration.py | Bug check: non-zero moments |
| D1-1 | Correctness test: no training sample has label_end ≥ test_start | structural | hw/d1_purged_kfold.py | Non-negotiable per expectations |
| D1-2 | len(list(pkf.split(X))) == k for k=5, k=10 | structural | hw/d1_purged_kfold.py | Exact fold count |
| D1-3 | PurgedKFold IC ≤ TimeSeriesSplit IC (same direction as S2, EX2) | structural | hw/d1_purged_kfold.py | Confirmed direction |
| D1-4 | IC comparison printed with delta, t-stat, sample size | structural | hw/d1_purged_kfold.py | Statistical context |
| D1-5 | Visual diagnostic plot with train/test/purged/embargo zones labeled | visual | hw/d1_purged_kfold.py | Step 5 verifies |
| D1-6 | ValueError raised for invalid inputs | structural | hw/d1_purged_kfold.py | Robustness |
| D2-1 | Correctness check: 2-asset portfolio at 100% turnover gives spread_cost = sum(\|Δw\|) × half_spread_bps / 10000 = 2 × spread_bps / 10000 | analytical | hw/d2_tc_pipeline.py | The "2" is sum(\|Δw\|) for [1,0]→[0,1], not a round-trip multiplier. half_spread is the one-way cost of crossing the bid-ask once. |
| D2-2 | Turnover series: mean monthly one-way ∈ [10%, 200%] | [0.10, 2.00] | hw/d2_tc_pipeline.py | Leverage-adjusted; 140% observed for full-rebalance decile portfolio |
| D2-3 | Spread cost series: all values ≥ 0 | structural | hw/d2_tc_pipeline.py | Sign check |
| D2-4 | Three-spread Sharpe ordering: pessimistic < base < optimistic | structural | hw/d2_tc_pipeline.py | Cost monotonicity |
| D2-5 | Max drawdown on net ≥ max drawdown on gross | structural | hw/d2_tc_pipeline.py | TC never reduces drawdown |
| D2-6 | GROSS_SHARPE and NET_SHARPE_OPT/BASE/PESS printed | structural | hw/d2_tc_pipeline.py | Structured output |
| D2-7 | Skewness and excess kurtosis for gross and net returns | structural | hw/d2_tc_pipeline.py | Per rigor.md §3.3 |
| D3-L1-1 | quantstats tear sheet generated with Sharpe, Sortino, max drawdown, Calmar, CAGR | structural | hw/d3_responsible_report.py | Falls back to manual if import fails |
| D3-L1-2 | Annual Sharpe (net, purged) ∈ [0.05, 1.5] | [0.05, 1.5] | hw/d3_responsible_report.py | Layer 1 baseline |
| D3-L1-3 | Max drawdown ≤ −10% | (−∞, −0.10] | hw/d3_responsible_report.py | Real strategy always has drawdowns |
| D3-L1-4 | DSR at M=10 computed with `DEPLOY/NO-DEPLOY` verdict | structural | hw/d3_responsible_report.py | Either outcome valid |
| D3-L2-1 | CPCV across 3 model variants: PBO ∈ [0.20, 0.75] | [0.20, 0.75] | hw/d3_responsible_report.py | Same bounds as S3 |
| D3-L2-2 | BHY adjusted p-values ≥ raw p-values | structural | hw/d3_responsible_report.py | Always inflates or maintains |
| D3-L2-3 | Winning model stated with IC, PBO, net Sharpe, DSR justification | structural | hw/d3_responsible_report.py | Full statistical justification |
| D3-L3-1 | MinTRL printed: `MINTRL_95pct={d} months` | structural | hw/d3_responsible_report.py | Formula applied to winning model |
| D3-L3-2 | Qualitative discussion present (≥ 3 sentences on capital allocation) | structural | hw/d3_responsible_report.py | Non-trivial; logged in notes |

---

## Execution Waves

- **Wave 0:** `code/data_setup.py`
  Downloads and caches all data: OHLCV, FF3 factors, and checks Week 4
  cache availability. Produces synthetic fallback signal if Week 4
  cache is missing or incomplete.

- **Wave 1 (independent):**
  - `code/lecture/s1_seven_sins.py` — standalone: only uses OHLCV
  - `code/lecture/s2_purged_cv.py` — depends on data_setup.py only
  - `code/lecture/s4_transaction_costs.py` — depends on data_setup.py only
  - `code/lecture/s5_deflated_sharpe.py` — depends on data_setup.py only
  - `code/seminar/ex1_bug_hunt.py` — standalone: fresh momentum signals

  These five files have no cross-file data dependencies. All load from
  data_setup.py or Week 4 cache directly.

- **Wave 2 (depends on Wave 1 caches):**
  - `code/lecture/s3_cpcv_multiple_testing.py` — needs CPCV, uses 3 model
    variants; references S2's CV setup conceptually but loads independently
    from Week 4 cache
  - `code/lecture/s6_responsible_backtest.py` — integrates TC from S4 and
    purged CV from S2; reads those caches
  - `code/seminar/ex2_purging_comparison.py` — references Week 4 predictions
  - `code/seminar/ex3_tc_sensitivity.py` — needs long-short weights from S4
    cache (or recomputes inline)
  - `code/seminar/ex4_dsr_calibration.py` — needs portfolio return series

  These can all start as soon as Wave 1 completes, as their dependencies
  are on the shared Week 4 cache, not on Wave 1 output files.

  Note: ex3 and ex4 both depend on the gross return series from the
  long-short portfolio. Both will load from data_setup.py's `load_alpha_output()`
  function (which returns Week 4 or synthetic fallback). No coupling to s4.

- **Wave 3 (integrative homework):**
  - `code/hw/d1_purged_kfold.py` — standalone class; compares against d3 only
  - `code/hw/d2_tc_pipeline.py` — TransactionCostModel class; standalone
  - `code/hw/d3_responsible_report.py` — integrates D1 and D2; must run last

  D3 depends on D1's `PurgedKFold` class and D2's `TransactionCostModel`
  class being available. Run D1 and D2 in parallel, then D3 after both
  complete.

Revised final structure:
- **Wave 0:** data_setup.py
- **Wave 1:** s1, s2, s4, s5, ex1 [parallel]
- **Wave 2:** s3, s6, ex2, ex3, ex4 [parallel]
- **Wave 3a:** d1, d2 [parallel]
- **Wave 3b:** d3 [after 3a]

---

## Per-File Implementation Strategy

### code/data_setup.py
- **Runtime:** fast (<1 min) — all data already cached in shared layer
- **Criteria:** None (setup file only)
- **Key responsibilities:**
  - Import `load_sp500_prices`, `load_sp500_ohlcv`, `load_ff_factors`, `SP500_TICKERS`,
    `DEMO_TICKERS` from `course/shared/data.py`
  - Slice to `START="2012-01-01"`, `END="2025-12-31"` per expectations.md
  - Check Week 4 cache: attempt to load `gbm_predictions.parquet`, `gbm_ic_series.parquet`,
    `nn_predictions.parquet`, `nn_ic_series.parquet`, `expanded_features.parquet`
  - If Week 4 cache available: load and re-export as standardized interface
  - If not available: activate synthetic fallback (12-1 month momentum, 449 tickers,
    2012–2025 monthly, ranked cross-sectionally)
  - Compute and cache monthly returns from daily OHLCV → `monthly_returns.parquet`
  - Compute and cache long-short portfolio weights and gross returns from alpha
    predictions → `ls_weights.parquet`, `gross_returns.parquet`
  - Cache FF3 monthly factors → `ff3_monthly.parquet` (already in Week 4 cache;
    re-export for Week 5 path independence)
  - Compute market-cap tiers from fundamentals → `mcap_tiers.parquet` (for TC model)
  - Export: `TICKERS`, `START`, `END`, `CACHE_DIR`, `PLOT_DIR`, `WEEK4_AVAILABLE`
  - Export load functions: `load_equity_data()`, `load_ohlcv_data()`, `load_ff3()`,
    `load_alpha_output()`, `load_ls_portfolio()`
  - Print final summary: ticker count, date range, OOS months, Week 4 status

---

### lecture/s1_seven_sins.py
- **Runtime:** medium (2-4 min) — daily OHLCV for all 449 tickers
- **Device:** N/A (no PyTorch)
- **Shared infra:** `shared.backtesting.sharpe_ratio`, `shared.metrics.ic_summary`
- **Visualization:**
  - Equity curve comparison (flawed vs. corrected): line chart — two cumulative
    return lines over time. Line chart is correct: this is a time-series of
    portfolio value.
  - Survivorship bias: bar chart — two bars (all vs. survivors) for annualized
    return. Bar chart because we're comparing two categorical portfolio definitions
    at a single aggregated statistic.
- **Caches for downstream:** None. S1 is self-contained.
- **Criteria:**
  - S1-1: Flawed signal OOS IC ≥ 0.30
  - S1-2: Corrected signal OOS IC ∈ [−0.05, 0.06]
  - S1-3: Annual return gap ≥ 15pp
  - S1-4: Survivorship premium ∈ [0.5%, 4.0%]
  - S1-5: Data quality block printed
- **Implementation notes:**
  - Look-ahead bug: rank by THIS MONTH's return to predict THIS MONTH's return.
    IC of this signal approaches 1.0 because signal = outcome.
  - Corrected signal: rank by month T-1 return to predict month T return (standard
    1-month momentum). Use DEMO_TICKERS (20 stocks) for equity curve plot;
    full 449 for survivorship quantification.
  - Survivorship: equal-weight all 449 tickers (using forward fill for missing) vs.
    equal-weight tickers with ≥90% non-null 2012–2025 history. Both use monthly
    returns from `monthly_returns.parquet`.
  - IS period: 2012–2017. OOS period: 2018–2024.
- **Pedagogical alternatives:** None needed. Look-ahead bug produces IC ~1.0 by
  construction — the effect will be unambiguous.

---

### lecture/s2_purged_cv.py
- **Runtime:** medium (2-4 min) — cross-sectional IC over 10 folds
- **Device:** N/A
- **Shared infra:** `shared.temporal.PurgedWalkForwardCV`, `shared.metrics.ic_summary`
- **Visualization:**
  - Split visualization: custom heatmap/timeline plot showing folds × time on the
    x-axis. Colors: blue (train), orange (test), red (purged zone), gray (embargo).
    Heatmap is correct: discrete folds by time, categorical color encoding.
  - IC series comparison (WF vs. purged): line chart — two overlapping IC series
    across fold indices. Line chart appropriate for ordered fold sequence.
- **Caches for downstream:** `wf_ic.parquet`, `purged_ic.parquet` → consumed by s6
- **Criteria:**
  - S2-1: TimeSeriesSplit mean CV IC ∈ [0.02, 0.10]
  - S2-2: PurgedKFold mean CV IC ∈ [0.01, 0.08], must be ≤ WF IC
  - S2-3: IC delta ≥ 0.005
  - S2-4: Exactly k splits
  - S2-5: Split visualization saved to `logs/plots/`
- **Implementation notes:**
  - Use Week 4 GBM predictions directly (or synthetic fallback).
  - Re-estimate IC at each fold: for each fold's test period, load the predictions
    for those dates and compute Spearman IC vs. actual forward return.
  - For `TimeSeriesSplit`: use `gap=1` month (note: gap is in observations, not
    days; monthly data means gap=1 = 1 month gap, approximately the 21-day label).
  - Purged KFold implementation: implement `PurgedKFoldDemo` inline as a teaching
    vehicle — this is the LECTURE section. The homework (D1) builds the production
    class. The lecture version can be simplified to 30 lines.
  - Label duration: 1 month forward return = ~21 trading days. Embargo: 1 month
    (conservative but correct for the teaching purpose).
- **Pedagogical alternatives:**
  - If IC delta < 0.005: note "For 1-month labels on monthly data, purging effect
    is modest because monthly periods approximate the label duration. Use the
    weekly data path (see expectations.md) for a more dramatic demonstration."
    Do not fail the assertion; the teaching point is still valid.

---

### lecture/s3_cpcv_multiple_testing.py
- **Runtime:** slow (5-15 min) — CPCV across 3 model variants × 15 paths
- **Device:** N/A
- **Shared infra:** `shared.temporal.CombinatorialPurgedCV`, `shared.metrics.ic_summary`
- **Visualization:**
  - OOS return distribution for IS winner: histogram with 15 bars (one per CPCV path).
    Histogram is correct: small discrete distribution (15 points) with density focus.
  - PBO fraction visualization: annotated histogram with vertical line at median and
    shaded region below it.
  - Harvey-Liu-Zhu t-stat comparison: horizontal bar chart — models on y-axis,
    t-stat on x-axis with threshold lines at 1.96 and 3.0. Bar chart because we're
    comparing a scalar metric across a small categorical set (3 models).
- **Caches for downstream:** `cpcv_results.parquet` → consumed by d3
- **Criteria:**
  - S3-1: len(cpcv_paths) == 15
  - S3-2: PBO ∈ [0.20, 0.75]
  - S3-3: OOS return distribution std > 0
  - S3-4: BHY adjusted p ≥ raw p
  - S3-5: N_passing_t300 reported; expected 0
- **Implementation notes:**
  - 3 model variants: GBM (from cache), NN (from cache), Ridge (compute inline from
    expanded_features and forward_returns using walk-forward splits — simple ridge
    regression, no tuning needed for the CPCV demonstration).
  - CPCV: instantiate `CombinatorialPurgedCV(n_splits=6, n_test_splits=2)` → produces
    C(6,2)=15 combinatorial paths. Each path: which variant was the IS winner, what
    was each variant's OOS rank.
  - PBO: fraction of 15 paths where IS-best variant has OOS rank below median.
  - BHY: `statsmodels.stats.multitest.multipletests(pvalues, method='fdr_bh')`.
  - Harvey-Liu-Zhu: compute t-stat for each model's IC series; show against 1.96 and
    3.0 thresholds.
  - The "50 variants" simulation: generate synthetic IC scores by permuting feature
    subsets on the expanded feature matrix; compute t-stats; show FDR problem.
- **Pedagogical alternatives:**
  - If only 2 model variants available from Week 4 cache: synthesize third variant
    using `RidgeCV` with the expanded feature set and walk-forward splits.

---

### lecture/s4_transaction_costs.py
- **Runtime:** medium (2-5 min)
- **Device:** N/A
- **Shared infra:** `shared.backtesting.long_short_returns`, `shared.backtesting.portfolio_turnover`,
  `shared.backtesting.net_returns`, `shared.backtesting.sharpe_ratio`,
  `shared.backtesting.max_drawdown`
- **Visualization:**
  - Three-regime equity curves: line chart — three lines (zero, fixed, tiered) over
    time. Line chart is correct: cumulative return over time.
  - TC decomposition pie/bar chart: bar chart showing spread vs. impact components
    for an average month. Bar chart because we're comparing two categorical cost types.
  - Market impact as participation rate function: line chart — impact cost vs.
    participation rate (%) for a range of 0.01–0.20. Line chart correct: continuous
    functional relationship.
- **Caches for downstream:** `ls_portfolio.parquet` (weights + gross returns),
  `tc_results.parquet` (net returns, turnover, spread/impact decomposition)
- **Criteria:**
  - S4-1: Monthly one-way turnover ∈ [30%, 100%]
  - S4-2: Annualized spread drag ∈ [0.5%, 4.0%]
  - S4-3: Market impact cost > 0
  - S4-4: Net Sharpe < gross Sharpe
  - S4-5: Regime ordering: zero > fixed > tiered equity curves
  - S4-6: HIGH TURNOVER warning if any month > 50%
  - S4-7: Skewness + excess kurtosis reported
- **Implementation notes:**
  - Use `load_alpha_output()` from data_setup.py to get predictions.
  - Build long-short weights: top-10% long, bottom-10% short, equal-weight within
    each leg (consistent with Week 4 approach).
  - Turnover: `portfolio_turnover(weights)` from shared.backtesting.
  - Regime (a): gross returns via `long_short_returns`.
  - Regime (b): `net_returns(gross, turnover, cost_bps=10)`.
  - Regime (c): market-cap tiered spread. Load `mcap_tiers.parquet` from data_setup.
    Large-cap (top tercile by market cap): 5 bps. Mid-cap (middle tercile): 15 bps.
    Small-cap (bottom tercile): 25 bps. Apply per-stock cost to compute weighted
    portfolio TC drag.
  - Corwin-Schultz: implement the formula using daily High and Low from OHLCV.
    Clip negative estimates to zero. If > 10% negative: flag and use fixed-rate proxy.
    This is labeled as "optional advanced method" in the output.
  - Market impact: inline implementation. σ (daily volatility, 30-day rolling) × η=0.1
    × sqrt(|Δweight| / ADV). ADV = 30-day average daily dollar volume from OHLCV.
  - Print `⚠ HIGH TURNOVER` per rigor.md §3.1 with drag estimate.
- **Pedagogical alternatives:**
  - If portfolio is static (turnover < 10%): flag as implementation issue; rebuild
    weights inline ensuring full monthly rebalance to prevent this.

---

### lecture/s5_deflated_sharpe.py
- **Runtime:** fast (<1 min) — pure arithmetic on a return series
- **Device:** N/A
- **Shared infra:** `shared.metrics.deflated_sharpe_ratio`
- **Visualization:**
  - DSR surface: 3D surface plot or 2D contour/heatmap — T values on x-axis, M values
    on y-axis, DSR on z-axis (or color). Heatmap is preferred for legibility in a
    notebook context (no 3D interaction). The heatmap shows monotone gradients.
  - MinTRL vs. Sharpe: line chart — MinTRL months on y-axis, Sharpe ratio on x-axis.
    Line chart because it's a continuous functional relationship.
- **Caches for downstream:** `dsr_surface.parquet` → consumed by s6, ex4
- **Criteria:**
  - S5-1: PSR ∈ [0, 1]
  - S5-2: DSR monotone in M
  - S5-3: DSR monotone in T
  - S5-4: MinTRL ∈ [6, 30] months
  - S5-5: Skewness and excess kurtosis reported
- **Implementation notes:**
  - `T_windows = [24, 36, 48, 60, 84, 120]` months (trailing windows from end of OOS)
  - `M_values = [1, 5, 10, 20, 50]` trials
  - For each T: slice the last T months of the gross return series.
  - For each (T, M): call `deflated_sharpe_ratio(observed_sr, n_trials=M, n_obs=T, ...)`.
  - The DSR surface must show degradation at short T and high M — this is the design
    intent from expectations.md Interaction 4.
  - Critical: do NOT use the full OOS period as a single input. Always use sub-windows.
  - `excess_kurtosis` convention: use `scipy.stats.kurtosis(returns, fisher=True)` —
    this gives excess kurtosis (normal=0). The DSR formula uses (γ₄+2)/4 where
    γ₄ = excess kurtosis (confirmed in expectations.md S5 formula note).
- **Pedagogical alternatives:**
  - If gross Sharpe > 1.5 (surface too flat): switch to net-of-cost Sharpe as primary
    DSR input. Net SR is typically 0.3–0.5 lower, pushing the surface into the
    pedagogically interesting region.
  - Trigger: DSR(T=24, M=50) > 0.90 using gross SR.

---

### lecture/s6_responsible_backtest.py
- **Runtime:** medium (3-7 min) — integrates CV, TC, DSR
- **Device:** N/A
- **Shared infra:** `shared.backtesting.*`, `shared.metrics.*`, `shared.temporal.*`
- **Visualization:**
  - Side-by-side naive vs. responsible equity curves: line chart — two cumulative
    return lines. The gap is the pedagogical point. IS/OOS boundary marked with
    vertical line.
  - Both IC series side by side: bar chart — monthly IC for both methods, side by
    side bars. Bar chart correct for monthly categorical time periods with sign.
- **Caches for downstream:** None (terminal lecture section)
- **Criteria:**
  - S6-1: Naive Sharpe ∈ [0.5, 1.8]
  - S6-2: Responsible Sharpe ∈ [0.1, 1.2], must be < naive
  - S6-3: Gap ∈ [0.1, 0.8]
  - S6-4: Sub-period IC split printed
  - S6-5: IS/OOS labeled
- **Implementation notes:**
  - Naive evaluation: sklearn `TimeSeriesSplit` (gap=1), gross returns, no DSR.
    Compute cumulative equity curve from gross returns.
  - Responsible evaluation: `PurgedWalkForwardCV` (purge_gap=1), net-of-cost returns
    (using TC model from s4 cache). Compute DSR at M=10 for the responsible return
    series.
  - Sub-period split: use the actual OOS period. Given Week 4 OOS starts April 2019:
    first half = 2019-04 to 2021-11 (32 months), second half = 2021-12 to 2024-11
    (36 months). If synthetic fallback: 2015–2019 vs. 2020–2024.
  - Load TC model results from `tc_results.parquet` (written by s4). If s4 not yet
    run: recompute inline.

---

### seminar/ex1_bug_hunt.py
- **Runtime:** medium (2-4 min)
- **Device:** N/A
- **Shared infra:** `shared.metrics.ic_summary`
- **Visualization:**
  - IS and OOS IC for all three signals: grouped bar chart — signals on x-axis,
    grouped bars for IS vs. OOS IC. Bar chart correct for small categorical comparison
    (3 signals × 2 periods = 6 bars).
  - The "diagnostic fingerprints" diagram: bar chart showing IS/OOS ratio per signal.
- **Caches for downstream:** None
- **Criteria:**
  - EX1-1 through EX1-7 (see Criteria Map)
- **Implementation notes:**
  - All three signals computed fresh on 449 tickers monthly 2015–2025.
  - Signal A: rank(month T return) → predicts month T return (pure look-ahead).
  - Signal B: survivorship subset (tickers with ≥90% 2015–2025 history), ranked by
    12-1 momentum, but evaluated against the full 449-ticker universe. The IS period
    uses only survivors; OOS uses the full universe.
  - Signal C: standard 12-1 month momentum (T-12 to T-1 return, skip T), evaluated
    on full 449 tickers.
  - IS period: 2015–2019. OOS period: 2020–2025.
  - IC computed monthly for each signal; report mean IS IC, mean OOS IC, IS/OOS ratio.

---

### seminar/ex2_purging_comparison.py
- **Runtime:** medium (2-4 min)
- **Device:** N/A
- **Shared infra:** `shared.metrics.ic_summary`, `shared.temporal.PurgedWalkForwardCV`
- **Visualization:**
  - Fold-by-fold IC comparison: line chart — two lines (WF IC, purged IC) across fold
    indices 1–10. Line chart correct for ordered sequence.
  - Rank flip heatmap: small matrix (folds × models) with color indicating rank
    agreement (same rank = green, different = red). Heatmap correct for binary matrix.
- **Caches for downstream:** None
- **Criteria:**
  - EX2-1 through EX2-4
- **Implementation notes:**
  - Load Week 4 predictions via `load_alpha_output()` from data_setup.py.
  - WF: `TimeSeriesSplit(n_splits=10, gap=1)`.
  - Purged: inline `PurgedKFoldDemo` (same simplified class as S2) with k=10.
  - Model rank flip: if GBM, NN, and Ridge predictions are all available, compute
    per-fold IC for each model under both CV methods. Count folds where rank ordering
    changes. If only 1 model available: note "rank flip requires ≥2 models; N/A."
  - Report `RANK_FLIP_FOLDS={count}/{total_folds}` in structured output.

---

### seminar/ex3_tc_sensitivity.py
- **Runtime:** fast (<1 min) — grid computation, no downloads
- **Device:** N/A
- **Shared infra:** `shared.backtesting.sharpe_ratio`, `shared.backtesting.net_returns`
- **Visualization:**
  - Feasibility frontier heatmap: 2D heatmap — half-spread (rows) × turnover
    reduction (cols), color = net Sharpe. Contour line at Sharpe=0.5.
    Heatmap correct: continuous metric over a 2D grid of categorical parameters.
- **Caches for downstream:** None
- **Criteria:**
  - EX3-1 through EX3-5
- **Implementation notes:**
  - Load gross returns and turnover series from `tc_results.parquet` (s4 cache).
    If s4 not run: recompute inline from data_setup.py's `load_ls_portfolio()`.
  - Half-spread grid: [2, 5, 8, 10, 12, 15, 20, 25, 30] bps (9 values).
  - Turnover reduction grid: [0%, 10%, 20%, 30%, 40%, 50%] (6 values) → 54 cells.
  - Turnover reduction: scale down monthly turnover by the reduction fraction.
  - Net Sharpe per cell: `net_returns(gross, turnover * (1 - reduction), spread_bps * 2)`.
  - Feasibility frontier: cells where net Sharpe = 0.5 threshold. Plot as overlay
    on heatmap.
  - If FEASIBLE_CELLS == 0: print `⚠ STRATEGY NON-VIABLE` and proceed with
    pedagogical discussion (do not fail).
- **Pedagogical alternatives:**
  - If gross Sharpe < 0.3 (all cells non-viable at normal grid): extend grid to include
    quarterly horizon variant (reduce turnover by 60% to model quarterly rebalancing).
    Trigger: net Sharpe at (10 bps, 0% reduction) < 0.1.

---

### seminar/ex4_dsr_calibration.py
- **Runtime:** fast (<1 min) — pure arithmetic
- **Device:** N/A
- **Shared infra:** `shared.metrics.deflated_sharpe_ratio`
- **Visualization:**
  - 5×5 DSR surface heatmap: T values on one axis, M values on other, DSR as color.
    Same approach as S5. Heatmap correct for discrete 2D grid.
  - Crossover threshold annotation: overlay on heatmap showing DSR < 0.5 boundary.
- **Caches for downstream:** None
- **Criteria:**
  - EX4-1 through EX4-5
- **Implementation notes:**
  - T_windows = [6, 12, 24, 36, 60] months
  - M_values = [1, 5, 10, 20, 50] trials
  - Same convention as S5: excess kurtosis from `scipy.stats.kurtosis(fisher=True)`.
  - Load gross returns from `tc_results.parquet` or `load_ls_portfolio()`.
  - For each T: use the last T months of the return series.
  - Per-cell structured output: `T={d}mo, M={d}: SR={:.3f}, DSR={:.3f} ({PASS/FAIL})`.
  - Crossover: identify the smallest M at each T where DSR first crosses below 0.5.
  - Note: EX4 uses a SUBSET of windows from S5 (6,12,24,36,60 vs. 24,36,48,60,84,120)
    to emphasize the short-track-record regime. Both are valid; EX4 focuses on the
    pedagogically extreme case.

---

### hw/d1_purged_kfold.py
- **Runtime:** medium (2-5 min) — class implementation + IC comparison
- **Device:** N/A
- **Shared infra:** `shared.metrics.ic_summary`; sklearn for comparison
- **Visualization:**
  - Visual diagnostic plot: timeline of splits for k=5 on 60 months. Train/test/
    purged/embargo zones clearly labeled. Saved to `logs/plots/d1_purged_kfold_splits.png`.
    Custom timeline plot (horizontal bar per fold, color-coded segments). Not a
    standard chart type — custom visualization justified by the teaching purpose.
  - IC comparison bar chart: two bars (WF, PurgedKFold) with error bars (std across
    folds). Bar chart correct for comparing two categorical methods.
- **Caches for downstream:** `d1_purged_kfold_class.py` is saved as a pickle-free
  importable module; d3 imports it directly from the hw directory.
- **Criteria:**
  - D1-1 through D1-6
- **Implementation notes:**
  - `PurgedKFold` class with `__init__(n_splits, label_duration, embargo)` and
    `split(X, y=None)` generator. Follows sklearn `BaseCrossValidator` interface.
  - Purging logic: for each fold's test period, find all training indices where
    `label_end_date = train_date + label_duration_timedelta` overlaps with the
    test period. Remove these.
  - Embargo: additionally remove training indices in [test_end, test_end + embargo_days].
  - Correctness test: generate synthetic DatetimeIndex (60 monthly observations),
    synthetic forward return label (known look-forward = 21 days). After splitting
    with k=5: scan every train index in every fold to verify no label overlap.
    Print `CORRECTNESS_TEST: PASS` or `CORRECTNESS_TEST: FAIL (n_leaking={n})`.
  - IC comparison: load Week 4 GBM predictions; compute per-fold IC under
    `TimeSeriesSplit(gap=1)` vs. `PurgedKFold(k=10, label_duration=21, embargo=5)`.
  - Code quality: full docstring, no global variables, `SEED = 42`.

---

### hw/d2_tc_pipeline.py
- **Runtime:** medium (2-5 min) — TC model applied to full portfolio
- **Device:** N/A
- **Shared infra:** `shared.backtesting.portfolio_turnover`, `shared.backtesting.sharpe_ratio`,
  `shared.backtesting.max_drawdown`, `shared.backtesting.net_returns`
- **Visualization:**
  - Gross vs. net equity curves (three spread regimes): line chart — 4 lines
    (gross + 3 net variants) over time. Line chart correct for cumulative return series.
  - TC decomposition over time: stacked area chart — spread cost and impact cost
    stacked per month. Stacked area correct for showing components of total cost over time.
  - Top-5 highest-cost months: horizontal bar chart — months on y-axis, total TC drag
    on x-axis, bars colored by dominant component. Bar chart for small categorical set.
- **Caches for downstream:** `d2_tc_model_results.parquet` → consumed by d3
- **Criteria:**
  - D2-1 through D2-7
- **Implementation notes:**
  - `TransactionCostModel` class:
    - `__init__(weights, spread_bps, impact_coeff=0.1)` where `spread_bps` can be a
      scalar or Series (stock-level).
    - `.fit(ohlcv)` — computes daily volatility, ADV from OHLCV data.
    - `.transform()` — computes turnover, spread cost, impact cost, net returns per period.
    - `.report()` — prints structured summary, top-5 high-cost months.
    - Attributes: `.turnover`, `.spread_cost`, `.impact_cost`, `.net_returns`.
  - Correctness check: 2-asset portfolio, weights=[1,0]→[0,1]. One-way turnover
    = sum(|Δw|)/2 = 1.0. spread_cost = sum(|Δw|) × half_spread_bps / 10000
    = (1.0 + 1.0) × spread_bps / 10000 = 2 × spread_bps / 10000.
    The factor of 2 comes from the total absolute weight change (sell 1 + buy 1),
    not from a round-trip multiplier — half_spread is already the one-way cost.
  - Market impact: `η × σ × sqrt(|Δweight| / adv)`. σ = 30-day rolling daily std
    of stock returns. ADV = 30-day rolling mean daily volume × close price.
    Participation rate = portfolio_value × |Δweight| / ADV (no AUM assumption → use
    weight change as participation fraction proxy).
  - Three spread regimes: optimistic=3 bps flat, base=market-cap tiered (5/15 bps),
    pessimistic=25 bps flat.
  - `⚠ HIGH TURNOVER`: print per rigor.md §3.1.

---

### hw/d3_responsible_report.py
- **Runtime:** slow (5-15 min) — CPCV across 3 model variants + quantstats
- **Device:** N/A
- **Shared infra:** `shared.temporal.CombinatorialPurgedCV`, `shared.metrics.deflated_sharpe_ratio`,
  `shared.metrics.ic_summary`, `shared.backtesting.*`
- **Visualization:**
  - Layer 1 quantstats tear sheet: saved to `logs/plots/d3_tearsheet.png`.
  - Layer 1 equity curve (net, purged): line chart with IS/OOS boundary.
  - Layer 2 PBO distribution: histogram of IS-winner OOS ranks across 15 CPCV paths.
  - Layer 2 model comparison table: printed as structured output (Pandas DataFrame to
    stdout).
- **Caches for downstream:** None (terminal deliverable)
- **Criteria:**
  - D3-L1-1 through D3-L3-2
- **Implementation notes:**
  - Layer 1: import `PurgedKFold` from `hw/d1_purged_kfold.py`. Import
    `TransactionCostModel` from `hw/d2_tc_pipeline.py`. Run purged CV on Week 4
    GBM model using D1's splitter. Apply D2's TC model. Compute DSR at M=10.
    Attempt `import quantstats`; on ImportError: manual computation of all metrics.
  - Layer 2: generate 3 model variants. GBM and NN from Week 4 cache. Ridge:
    recompute walk-forward predictions inline (10 lines with sklearn Ridge). Apply
    `CombinatorialPurgedCV(n_splits=6, n_test_splits=2)` across all 3. Compute PBO.
    Apply BHY via `statsmodels.stats.multitest.multipletests`. Select winning model.
  - Layer 3: MinTRL formula per Bailey-López de Prado (2014). Inline implementation
    (~8 lines). Pass actual skewness and excess kurtosis. Qualitative discussion
    in structured output (3+ sentences, logged in notes).
  - If quantstats fails: use manual metrics (all available in `shared.backtesting`).
- **Pedagogical alternatives:**
  - If fewer than 3 model variants: synthesize 3rd as RidgeCV walk-forward (inline).

---

## Cross-File Data Flow

```
data_setup.py
  → .cache/monthly_returns.parquet      consumed by: s1, ex1
  → .cache/ls_weights.parquet           consumed by: s4, ex3, d2
  → .cache/gross_returns.parquet        consumed by: s4, s5, s6, ex3, ex4, d2, d3
  → .cache/ff3_monthly.parquet          consumed by: s5, s6, d3
  → .cache/mcap_tiers.parquet           consumed by: s4, d2
  → .cache/alpha_predictions.parquet    consumed by: s2, s3, s6, ex2, d1, d3
  → .cache/model_variants.parquet       consumed by: s3, ex2, d3

s2_purged_cv.py
  → .cache/wf_ic.parquet                consumed by: s6
  → .cache/purged_ic.parquet            consumed by: s6

s4_transaction_costs.py
  → .cache/tc_results.parquet           consumed by: s6, ex3, ex4

s5_deflated_sharpe.py
  → .cache/dsr_surface.parquet          consumed by: ex4

s3_cpcv_multiple_testing.py
  → .cache/cpcv_results.parquet         consumed by: d3 (reference)

d1_purged_kfold.py
  → hw/d1_purged_kfold.py (class)       consumed by: d3 (direct import)

d2_tc_pipeline.py
  → .cache/d2_tc_model_results.parquet  consumed by: d3
```

---

## Shared Infra vs From-Scratch Rule

Per `rigor.md` first-introduction rule:

| Component | Status | Action |
|-----------|--------|--------|
| `PurgedKFold` class | First introduction (Week 5) | Build from scratch in D1 |
| `TransactionCostModel` | First introduction (Week 5) | Build from scratch in D2 |
| TC decomposition (spread, impact) | First introduction (Week 5) | Implement inline in S4 |
| Sub-period degradation (IC split) | First introduction (Week 5) | Implement inline in S6 |
| Multiple testing (DSR surface, PBO, BHY) | First introduction (Week 5) | Implement inline; use `deflated_sharpe_ratio` from shared (already there) |
| Return distribution characterization (skew/kurt) | First introduction (Week 5) | Inline via scipy; wire into all Sharpe computations |
| `CombinatorialPurgedCV` | Already in `shared.temporal` | Import, do not rebuild |
| `PurgedWalkForwardCV` | Already in `shared.temporal` | Import, do not rebuild |
| `deflated_sharpe_ratio` | Already in `shared.metrics` | Import, but teach formula first (S5) |
| `ic_summary`, `rank_ic`, `sharpe_ratio`, etc. | Established (Weeks 1-4) | Import from shared |

---

## Device and Runtime Summary

| File | Runtime | Device |
|------|---------|--------|
| data_setup.py | fast | N/A |
| s1_seven_sins.py | medium | N/A |
| s2_purged_cv.py | medium | N/A |
| s3_cpcv_multiple_testing.py | slow | N/A |
| s4_transaction_costs.py | medium | N/A |
| s5_deflated_sharpe.py | fast | N/A |
| s6_responsible_backtest.py | medium | N/A |
| ex1_bug_hunt.py | medium | N/A |
| ex2_purging_comparison.py | medium | N/A |
| ex3_tc_sensitivity.py | fast | N/A |
| ex4_dsr_calibration.py | fast | N/A |
| d1_purged_kfold.py | medium | N/A |
| d2_tc_pipeline.py | medium | N/A |
| d3_responsible_report.py | slow | N/A |

No file requires GPU/MPS — this week is purely mathematical (CV mechanics,
cost accounting, statistical testing). No neural network training.
