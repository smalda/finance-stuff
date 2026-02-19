# Week 5: Backtesting, Research Discipline & Transaction Costs — Expectations

**Band:** 1 — Essential Core
**Implementability:** HIGH

**Rigor:** Tier 3.
- Teaches: transaction costs (decomposition into spread, impact, slippage; net-of-cost reporting — implement from scratch).
- Teaches: sub-period performance degradation (split-half IC decay reporting — implement from scratch).
- Teaches: multiple testing awareness (DSR surface, PBO from CPCV, BHY correction on IC estimates — implement from scratch).
- Teaches: return distribution characterization (skewness, excess kurtosis alongside Sharpe — implement from scratch).

---

## Gap Scan

### Search gaps filled via web

**S1 — Look-ahead premium magnitude.** Research notes state that a momentum signal coded with a look-ahead bug (using current-period returns to form the signal) produces an inflated equity curve, but do not give a quantified IC premium. Web search did not surface a peer-reviewed point estimate for the look-ahead IC inflation of a simple momentum bug (using close-at-T to rank for period T rather than close-at-T-1). The magnitude is context-dependent (signal type, sample period, universe). No benchmark found. Range estimated from structure of the bug: the signal becomes a perfect one-period predictor, producing IC approaching 1.0 in-sample for the contaminated period; the corrected signal should show IC in the 0.02–0.05 range per GKX. The IC premium is therefore large and visually obvious — the equity curve comparison will speak clearly without needing a published exact number.

**S3 — PBO for a well-specified model.** The CPCV literature (Bailey et al. 2015; Arian et al. 2024) does not publish a canonical "expected PBO" for a correctly-specified equity alpha model. The theoretical neutral point is 0.5 (random), and PBO < 0.5 means the in-sample winner tends to perform at or above median out-of-sample. For a well-specified model with genuine alpha (IC ≈ 0.03–0.05), PBO should be below 0.5 but how far depends on alpha persistence across sub-periods. Realistic range: 0.25–0.50 for a real signal, 0.50–0.75 for an overfit signal. No tighter benchmark found.

**S5 — DSR calibration.** Code probe (see below) confirms that over 132 OOS months with annual Sharpe ≈ 0.7, DSR remains near 1.0 even at 50 trials. This is because the OOS period is long enough to have high statistical power. The interesting teaching regime is the DSR *surface*: across shortened sub-windows (24–48 months) and moderate trial counts (10–50), DSR degrades meaningfully. At SR=0.7, T=24 months, M=50 trials: DSR=0.75 (still passing). At SR=0.3, T=24 months, M=50 trials: DSR=0.22 (fails). The blueprint correctly designs Section 5 to vary T and M — this is where the teaching point lives.

**TC magnitudes (production).** Multiple practitioner sources (frec.com 2023 direct indexing analysis; Anderson UCLA bid-ask spread study) confirm: large-cap S&P 500 one-way spread cost ≈ 2–5 bps in current markets; mid-cap ≈ 12–25 bps. The blueprint's cost tier ranges (large-cap ~5–8 bps, mid-cap ~12–20 bps, small-cap ~25–50 bps) are reasonable one-way half-spread assumptions for research-grade modeling. Production firms use NYSE TAQ (NYSE Trade and Quote database) for precise tick-level spread estimation.

### Probe gaps filled via code

**Probe 1 — Data shape.** `load_sp500_prices('2012-01-01', '2025-12-31', SP500_TICKERS)` returns 449 stocks with monthly frequency, 192 monthly observations. OOS period starting 2015 gives 132 months. (Verified via code probe: 449 stocks × 132 OOS months at consistent coverage.)

**Probe 2 — CPCV feasibility.** With k=6 folds over 120 OOS months: 20 months/fold, 15 CPCV paths, each with 20 OOS months. k=8: 28 paths, 15 months/fold. The CPCV in Section 3 operates over 3 model variants from Week 4 across the full OOS period, not 50 independent backtests from scratch — so the combinatorial load is manageable. (Verified via code probe.)

**Probe 3 — OHLCV availability.** `load_sp500_ohlcv()` returns High and Low fields at 100% non-null rate for the sample tested (30 tickers, 2018–2025). Corwin-Schultz spread estimator is feasible. (Verified via code probe: High/Low confirmed for 30 tickers, 2011 daily rows, 100% coverage.)

**Probe 4 — DSR surface.** Computed DSR surface via code probe using Bailey-López de Prado formula. Key findings:
- At SR=0.7, T=132 months: DSR ≥ 0.99 for all M ≤ 50. Full OOS period is too long for DSR to fail even at high trial counts.
- At SR=0.7, T=24 months, M=50: DSR=0.75 (teaching point visible but weak).
- At SR=0.3 (weak/net-of-cost signal), T=24 months, M=50: DSR=0.22 (clear fail — pedagogically powerful).
- MinTRL at annual SR=0.5, normal returns: ~17 months (1.4 years). At SR=0.7: ~10 months.
- The Section 5 demonstration must use shortened OOS windows (24–36 months) to show DSR degradation. The full 132-month window hides the effect.

---

## Data Plan

### Dataset A: Equity Prices (OHLCV)

Used for: Section 1 (look-ahead bug demonstration), Section 4 (TC decomposition), Section 6 (responsible backtest integration), Ex 1 (bug hunt), Ex 3 (TC sensitivity), D2 (TransactionCostModel).

#### Universe

- **Option A: Full available universe — 449 tickers (SP500_TICKERS, 2012–2025).** Maximum cross-sectional power. 449 stocks × 132 OOS months = ~59,268 stock-months. Strongly supports Section 6's responsible backtest and D3's full pipeline. The TC model (D2) benefits from a wide universe showing spread-tier variation (large-cap vs. mid-cap tickers). Requires the shared data layer only — no additional downloads.
- **Option B: Restricted to Week 4 model universe — ~200 tickers.** Matches Week 4's AlphaModelPipeline output exactly, avoiding any mismatch in ticker coverage between the alpha signal and the TC model. Simpler to cross-reference. Reduces cross-sectional power slightly but still well above the statistical minimum.
- **Option C: DEMO_TICKERS (20 tickers) for lecture demonstrations only.** Appropriate for Section 1's look-ahead bug equity curve plot and Section 4's illustrative TC decomposition, where the teaching point is visual and does not require cross-sectional statistical power.

**Recommendation:**
- **Primary (Sections 1, 6, D2, D3):** The full 449-ticker universe for all statistical and pipeline exercises. Week 4 produced predictions on ~449 stocks (verified: 449 with data throughout 2012–2025). There is no mismatch to worry about.
- **Lecture demonstrations (S1 illustration, S4 TC example):** Use DEMO_TICKERS (20 tickers) for the visual demonstration plots; the full universe for the quantitative comparisons in Sections 4 and 6.

**Risk:** The survivorship bias in the 449-ticker universe inflates backtest returns by 1–4% annually. This is a teaching moment (Week 1 established this), not a reason to reduce the universe. All backtest results must note survivorship bias in structured output.

---

#### Date Range

- **Option A: 2010–2025 (15 years).** Provides the longest possible training history for the walk-forward CV sections. With a 3-year initial training window (2010–2013), OOS starts 2013 and runs ~144 months. Covers: GFC recovery, post-2012 bull market, 2015–2016 correction, 2018 volatility, COVID crash and recovery, 2022 rate shock. Full business cycle coverage.
- **Option B: 2012–2025 (13 years).** Still covers all major market regimes of the ML era. 3-year initial train window puts OOS at 2015–2025 (132 months). This is the period that aligns with Week 4's verified cross-sectional coverage (449 stocks from 2012 onwards per probe). Slightly shorter than Option A but avoids the thin coverage period of 2010–2011 for some tickers.
- **Option C: 2015–2025 (10 years).** Shorter but entirely within the ML-era strategy landscape. Adequate for demonstrating purged CV mechanics but provides fewer sub-periods for sub-period degradation analysis (rigor.md §3.2). Insufficient for a meaningful CPCV distribution.

**Recommendation: Option B — 2012–2025.**
- Aligns with verified data coverage (449 stocks from 2012). Provides 132 OOS months from a 2015 OOS start point (with 3-year initial train window), which is long enough for sub-period degradation analysis (66 months per half), CPCV at k=6 (22 months per fold), and a meaningful MinTRL demonstration. The full OOS length ensures the DSR surface demonstration requires the code to deliberately shorten sub-windows to show DSR degradation — consistent with the blueprint's design for Section 5 ("vary backtest length from 6 months to 5 years").
- API calls: `load_sp500_prices('2012-01-01', '2025-12-31', SP500_TICKERS)` and `load_sp500_ohlcv('2012-01-01', '2025-12-31', SP500_TICKERS)` via `course/shared/data.py`.

**Risk:** The post-2022 regime (aggressive rate tightening, tech multiple compression) may produce sub-period IC degradation in the second half of the OOS window (2020–2025). This is a teaching moment, not a failure — it illustrates the point of sub-period analysis (rigor.md §3.2).

---

#### Frequency

- **Primary:** Monthly for all cross-sectional alpha signal work (purged CV, CPCV, IC evaluation, TC sensitivity, DSR). Monthly is the correct frequency for the long-short portfolio inherited from Week 4 and eliminates daily microstructure noise from the core learning exercises.
- **Daily:** Required for Section 1 (the look-ahead bug demonstration plots a daily momentum equity curve), Section 4 (daily OHLCV for Corwin-Schultz spread estimation and dollar-volume-based market impact parameters), and D2 (portfolio weight matrix computed at daily frequency for fine-grained TC accounting).
- **Risk:** The Section 1 look-ahead bug demonstration uses daily momentum — a 1-day close-to-close momentum signal. Using daily data at this frequency introduces bid-ask bounce noise, which is a real effect but does not materially affect the demonstration's teaching point (the look-ahead premium is large and dominates).

---

### Dataset B: Week 4 AlphaModelPipeline Output

Used for: Sections 2–6 (the alpha signal being evaluated), Ex 2 (purging vs. walk-forward CV comparison), Ex 3 (TC sensitivity grid), Ex 4 (DSR calibration), D1 (PurgedKFold correctness test on real data), D2 (TransactionCostModel applied to the long-short portfolio), D3 (full responsible backtest report).

**Source:** Week 4 `AlphaModelPipeline` output — cross-sectional monthly alpha scores (model predictions) and the derived long-short portfolio weights (the top-minus-bottom decile portfolio). These are stored in the Week 4 `.cache/` directory.

**Format (confirmed from curriculum_state.md):** `AlphaModelPipeline` produces:
- Cross-sectional predictions: DataFrame (stocks × months), values are predicted forward returns (or equivalent score)
- Long-short portfolio weights: DataFrame (stocks × months), +1/0/-1 or continuous weights
- Gross return series: monthly time-indexed Series
- IC/rank IC series: monthly time-indexed Series

**Dependency risk:** If Week 4 has not been built or its cache is unavailable, Week 5 cannot run Sections 2–6 or any homework deliverable. This is a hard dependency. The code agent must check for Week 4 output at startup and fail clearly with a diagnostic message if unavailable. A synthetic fallback (simulated cross-sectional predictions from a simple momentum signal on the 449-ticker universe) should be implemented in `data_setup.py` as a contingency — this allows the week to be taught without Week 4's specific output while still demonstrating the same methodology.

**Synthetic fallback design:** 12-1 month momentum signal (11-month return, skipping the most recent month to avoid short-term reversal) computed cross-sectionally on 449 stocks monthly, rank-normalized, sorted into long-short decile portfolios. This replicates the Week 4 portfolio construction without requiring the ML model output. Mark clearly in code with `# SYNTHETIC FALLBACK — replace with AlphaModelPipeline output when available`.

**API calls:** No new downloads for this dataset. Read Week 4 cache: `pd.read_parquet('../week04_ml_alpha/code/.cache/alpha_predictions.parquet')` (path confirmed from course directory structure convention).

**Risk:** The Week 4 alpha signal may be weaker than expected (IC ~ 0.02 rather than 0.03–0.05), which reduces the gross Sharpe of the long-short portfolio and makes it harder to demonstrate meaningful DSR variation. The synthetic fallback momentum signal is likely to have similar IC characteristics (momentum IC ≈ 0.03–0.04 on S&P 500 monthly in 2015–2025 based on GKX and Week 3's momentum factor findings), so this risk applies to both the real and fallback signal.

---

### Dataset C: Fama-French Factors

Used for: Section 6 (benchmark comparison — the responsible backtest uses FF3 as the risk-free benchmark and market return as the passive benchmark); D3 (quantstats tear sheet benchmark input).

**Source:** `load_ff_factors(model='3', frequency='M')` — Mkt-RF, SMB, HML, RF. Monthly, 2012–2025.

**Access:** `load_ff_factors(model='3', frequency='M')` via `course/shared/data.py`.

**Risk:** None significant. Ken French data is PIT-clean and available through 2025. Risk-free rate (RF column) is needed for excess return computation in DSR formula. Not the primary data source for this week.

---

### Interaction Analysis

**Interaction 1 — Week 4 cache dependency × Section 1 independence.** Section 1 is explicitly designed to be independent of Week 4 output (it demonstrates the look-ahead bug using a fresh simple momentum signal on the full 449-ticker universe). All other sections and homework depend on Week 4 output. This creates a structural risk: if the Week 4 dependency fails, only Section 1 and parts of Section 4 can run. The synthetic fallback in `data_setup.py` must handle this gracefully and switch all downstream sections to the synthetic signal without requiring separate code changes.

**Interaction 2 — Monthly frequency × CPCV structure.** The lecture's Section 3 CPCV uses the monthly prediction series. With 132 OOS months and k=6 CPCV folds, each fold is 22 months. The 3 model variants (linear, LightGBM, NN from Week 4) produce 3 IC series of 132 points each. CPCV generates 15 combinatorial train-test paths. The resulting PBO distribution has 15 data points — enough to visualize the distribution and compute the fraction above/below median. This is feasible but produces a coarse PBO distribution. If only 2 model variants are available, the CPCV still works but the distribution is trivial (15 paths across 2 models). The code agent should verify that at least 3 model variants are available from Week 4 before running CPCV; if not, generate synthetic variants by varying the model's feature set.

**Interaction 3 — Daily OHLCV × TC model.** Section 4 and D2 use daily OHLCV for spread estimation (via the Corwin-Schultz estimator or fixed-rate proxy). The TC model then converts daily TC costs to monthly totals (via monthly turnover × daily cost assumptions). This interaction is clean: daily OHLCV is already available at 100% coverage (verified probe), and the aggregation to monthly TC drag is straightforward. Risk: the Corwin-Schultz estimator can produce negative spread estimates during low-volatility periods. The code agent must clip these to zero.

**Interaction 4 — DSR surface × OOS window length.** As established in the probe, the full 132-month OOS period produces near-unity DSR even at 50 trials for a reasonable signal. The Section 5 DSR surface demonstration must explicitly subsample the OOS period to show the pedagogically interesting region. The code should implement DSR computation over rolling sub-windows of 24, 36, 48, 60, 84, and 120 months (using the most recent K months from each OOS period). This creates the surface the blueprint describes. The code agent must understand this design intent and not use the full 132 months as the single input.

**Interaction 5 — Survivorship bias × DSR demonstration.** The survivorship-biased universe inflates gross returns, which inflates the observed Sharpe, which inflates the DSR. When Section 5 demonstrates that DSR is sensitive to the number of trials, the starting Sharpe already has a 1–4% survivorship premium baked in. The code agent must note this in structured output: `Universe: current S&P 500 — survivorship inflates Sharpe by approximately 0.1–0.3 (annual) before DSR correction.` This is documented as a constraint, not a reason to shrink the universe.

**Interaction 6 — TC sensitivity grid × turnover.** Exercise 3 computes net Sharpe across a grid of half-spread (2–30 bps) and turnover reduction (0–50%) assumptions. The Week 4 long-short portfolio has estimated monthly one-way turnover of 20–40% (from Week 4 curriculum state: long-short decile rebalancing monthly). At 40% one-way turnover and 10 bps half-spread, round-trip TC drag = 2 × 0.0010 × 0.40 = 0.08% per month = ~1.0% per year — meaningful but not fatal for a strategy with 0.7 annual Sharpe. At 30 bps half-spread (mid-cap tier), TC drag = ~3.0% per year — potentially strategy-killing. This interaction produces the feasibility frontier the exercise targets. (Verified via formula: net_return = gross_return − turnover × 2 × half_spread.)

---

### Failure Modes & Contingencies

| Risk | Likelihood | Affected Sections | Contingency |
|------|-----------|-------------------|-------------|
| Week 4 cache unavailable (AlphaModelPipeline output missing) | Medium | S2–S6, Ex2–Ex4, D1–D3 | Activate synthetic fallback in `data_setup.py`: 12-1 momentum signal on 449-ticker universe. Mark clearly in code. All methodology demonstrations work identically on the synthetic signal. |
| Week 4 has only 1 model variant (not 3) | Low | S3 (CPCV), D3 L2 | Synthesize 2 additional variants by using feature subsets (momentum-only vs. value-only vs. combined) from the shared data cache. These are pedagogically valid "model variants" for the CPCV demonstration. |
| DSR demo is pedagogically flat (DSR near 1.0 for all displayed T/M combinations) | High (certain if full OOS used) | S5 | As designed: Section 5 must use rolling sub-windows (24–36 months) within the OOS period. Ensure code uses `T_window` parameter that truncates to recent K months when computing DSR surface. The 24-month window at M=50, SR=0.3 gives DSR=0.22 — clearly pedagogical. |
| Corwin-Schultz estimator produces negative spread estimates | Medium (known behavior during low-vol periods) | S4, D2 | Clip to zero: `spread_bps = max(0, cs_spread_bps)`. The fixed-rate proxy (5 bps large-cap, 15 bps mid-cap) serves as the primary method; Corwin-Schultz is offered as an "intermediate option." |
| quantstats incompatibility (import errors or API changes) | Low (v0.0.81 confirmed Jan 2026, Production/Stable) | D3 (tear sheet) | If import fails, replace with manual quantstats-style metrics (Sharpe, Sortino, max drawdown, calmar) computed from scratch. All formulas are in `course/shared/backtesting.py` or directly implementable. |
| pypbo GitHub-only install fails in the course environment | Medium | S5, D3 L2 (PBO) | Implement PBO from scratch using the CPCV paths already computed — count the fraction of paths where the in-sample winner underperforms. The code is ~20 lines and does not require pypbo. Research notes recommend this approach as primary anyway. |
| CPCV produces < 10 OOS paths (insufficient for PBO distribution) | Low (k=6 gives 15 paths, k=8 gives 28) | S3, D3 L2 | Use k=8 instead of k=6 to get 28 paths with 15 months per fold. Always validate n_paths ≥ 10 before running PBO computation; fall back to k=8 if k=6 gives too few paths. |
| Monthly one-way turnover > 50% (triggering ⚠ HIGH TURNOVER warning from rigor.md §3.1) | Medium-High (Week 4 long-short decile rebalances 100% each month in naive implementation) | D2, D3 | This is expected behavior for a monthly full-rebalance long-short strategy and is itself the teaching point — the warning IS the lesson. Do not suppress it. Frame as: the unmodified strategy has high turnover; Exercise 3 explores turnover-reduction techniques. |
| Section 1 look-ahead premium too small to see visually (IC contamination smaller than expected) | Low | S1 | If the look-ahead IC premium is < 0.10 (unlikely given the signal uses the exact forward return), use a 5-day holding period to amplify the effect. The bug is structural (IC → 1.0 for 1-period forward return prediction), so the effect size is determined by the label definition, not randomness. |
| mlfinpy install fails (Section 2 reference implementation) | Low-Medium (MIT, PyPI accessible) | S2 (sidebar reference only) | mlfinpy is a sidebar reference, not primary code. The primary purged CV implementation is manual (homework D1 and lecture S2). If mlfinpy is unavailable, skip the reference and note in execution_log.md. |

---

### Feasibility Concerns

No blueprint sections are infeasible with the available data. All exercises and deliverables can be implemented with the shared data layer plus Week 4 outputs (or the synthetic fallback). The main structural concern — CPCV infeasibility — was resolved in Phase 3: 15–28 CPCV paths are achievable at k=6–8, which is sufficient for the demonstration.

One design note that is not an infeasibility but warrants user awareness: **Section 5's DSR surface will be pedagogically flat if the code naively uses the full 132-month OOS period.** The code agent must explicitly implement the rolling-subwindow design (see Interaction 4 above). This is coded behavior, not a data limitation, and should be called out in the code plan (Step 4A). This is flagged here so it appears at the data plan checkpoint before Step 4 begins.

---

### Statistical Implications

**Primary OOS window (2015–2025, 132 months):**
- Cross-sectional IC computation: 449 stocks × 132 months = 59,268 stock-months. IC t-statistic has degrees of freedom ≈ 131 (monthly IC observations). At IC ≈ 0.03: t ≈ 0.03 × sqrt(132) ≈ 0.34 (weak). At IC ≈ 0.05: t ≈ 0.05 × sqrt(132) ≈ 0.57 (still weak). Signal significance requires IC ≈ 0.17 for t > 1.96 at n=132 months. This is above the realistic IC range for free data, confirming that monthly IC t-stats will likely be below 1.96. Expected result: ⚠ WEAK SIGNAL print in run_log.txt. This is anticipated and planned for — it does NOT indicate a code failure. The teaching angle is that this is the honest result of an underpowered cross-sectional signal on a small universe (449 vs. CRSP's 3,000+).
- **Annual Sharpe of long-short portfolio:** Cross-sectional IC of 0.03–0.05 translates (via Fundamental Law of Active Management with BR ≈ 449 × 12 ≈ 5,388 bets/year) to IR ≈ IC × sqrt(BR) ≈ 0.03 × 73 ≈ 2.2 gross. In practice, long-short decile Sharpe is lower due to concentration (only 10% long, 10% short), transaction costs, and non-GKX conditions. Expected gross annual Sharpe: 0.6–1.2. Net Sharpe (after TC): 0.2–0.8 depending on turnover.
- **CPCV PBO:** At 15 combinatorial paths across 3 model variants, PBO resolution is coarse. Each path ranks 3 models on OOS performance; PBO = fraction of paths where best IS model underperforms median OOS rank. Statistically weak due to small n, but pedagogically clear.
- **Sub-period analysis (rigor.md §3.2):** First half (2015–2019, 60 months) vs. second half (2020–2025, 72 months). The 2020–2025 period includes COVID (massive regime shift) and 2022 rate shock — cross-sectional factor premia may have behaved differently. Signal degradation across halves is likely and expected. This is a teaching moment about non-stationarity, not a failure.

---

## Known Constraints

- **Survivorship bias (Tier 1 established, Week 1):** The 449-ticker universe is today's S&P 500 + SP400 projected backward. Companies that went bankrupt, were delisted, or were removed from the index are excluded. This inflates backtest returns by an estimated 1–4% annually and inflates measured IC by an unknown but non-trivial amount. All backtest results must include: `Universe: current S&P 500 — subject to survivorship bias. Returns inflated by estimated 1–4% annually.` The survivorship bias in IC elevation is likely smaller than in raw returns but not zero. Production comparison: CRSP's full universe includes ~3,000 NYSE/AMEX/NASDAQ stocks including delistings; Ken French's portfolios are the gold standard. Affects: all sections that compute IC or portfolio returns (S2, S3, S4, S5, S6, Ex1, Ex2, Ex3, Ex4, D1, D2, D3).

- **mlfinlab inaccessibility:** The Hudson & Thames `mlfinlab` library, widely referenced in 2019–2021 educational materials, is no longer available on public PyPI (404 error confirmed per research notes). Any references to `mlfinlab` in student resources must be replaced with `mlfinpy` (MIT license, PyPI-accessible, readthedocs.io confirmed) or manual implementation. The primary vehicle for purged CV in this week is manual implementation (Homework D1). `mlfinpy` is a reference/verification option only. Affects: S2, D1.

- **No TAQ data for spread estimation:** The NYSE Trade and Quote (TAQ) database, which provides tick-level bid-ask quotes, requires an institutional subscription (WRDS). The course uses a proxy approach: fixed-rate spread tiers (large-cap ~5–8 bps, mid-cap ~12–20 bps, small-cap ~25–50 bps one-way half-spread) as the primary TC model, with the Corwin-Schultz (2012) OHLC-based estimator as an optional data-driven alternative. Production TC modeling uses TAQ fills for TCA (Transaction Cost Analysis). This limitation is the primary teaching point of Section 4's "Sidebar: TAQ and institutional TC data." Affects: S4, Ex3, D2.

- **Week 4 dependency for Sections 2–6 and all homework:** Sections 2–6 and all three homework deliverables require the `AlphaModelPipeline` output from Week 4 (cross-sectional predictions, long-short portfolio weights, gross return series). If Week 4 has not been completed, the synthetic fallback (12-1 momentum signal on 449 tickers) must be activated. The synthetic fallback produces qualitatively similar results for all methodology demonstrations but will not match any specific IC or Sharpe number from Week 4's model comparison. Affects: S2, S3, S4, S5, S6, Ex1, Ex2, Ex3, Ex4, D1, D2, D3.

- **Monthly IC statistical power:** As computed in the Statistical Implications section, monthly cross-sectional IC on 449 stocks will not reach t > 1.96 at the realistic IC range of 0.02–0.05 (would require IC ≈ 0.17). The `ic_summary()` function from `course/shared/metrics.py` will print `⚠ WEAK SIGNAL` for IC t-stats below 1.96. This is anticipated behavior, not a code failure. Week 4 established this pattern; Week 5 continues it. Teaching angle: honest reporting of signal weakness is itself a core backtesting discipline. Affects: all sections computing rolling IC (S2, S3, S6, Ex1, Ex2, D1, D3).

- **DSR surface requires sub-window implementation:** The DSR over the full 132-month OOS period will be near 1.0 for all realistic trial counts (as verified by code probe). The Section 5 demonstration must explicitly compute DSR over rolling sub-windows of 24, 36, 48, 60, 84, and 120 months to produce a pedagogically informative surface. This is a design constraint on the code implementation, not a data limitation. Affects: S5, Ex4.

- **Corwin-Schultz estimator produces negative values:** The Corwin-Schultz (2012) spread estimator using daily High-Low ratios can produce negative values during low-volatility periods (a known limitation of the estimator). The code must clip these to zero before computing TC. When negative estimates appear in > 10% of stock-days, this suggests the estimator is unreliable for the applicable period and the fixed-rate proxy should be used instead. Affects: S4, D2 (optional Corwin-Schultz path).

- **Week 4 turnover estimate:** From curriculum_state.md, the Week 4 AlphaModelPipeline likely implements naive monthly full rebalancing of the long-short decile portfolio. This implies one-way monthly turnover ≈ 40–100% (naive long-short turns over most positions each month as the alpha scores re-rank). This is high enough to trigger the `⚠ HIGH TURNOVER` warning from rigor.md §3.1 at any spread assumption > 5 bps. The high-turnover warning is the teaching point of Section 4 and Exercise 3 — do not suppress it. Affects: S4, Ex3, D2.

- **No intraday data:** The OHLCV cache provides daily bars only (no tick, minute, or order book data). Market impact estimation uses the Almgren-Chriss square-root model with daily volume as the participation rate denominator. The course cannot model intraday execution or VWAP slippage from actual order routing. Production TC models use intraday data for implementation shortfall measurement. This limitation is documented in the Section 4 sidebar: "TAQ and institutional TC data." Affects: S4, D2.

- **quantstats API dependency:** D3 requires `quantstats` (v0.0.81, verified January 2026, Production/Stable) for tear sheet generation. If import fails or API changes, the code agent must fall back to manual metric computation. Key metrics needed: Sharpe, Sortino, max drawdown, Calmar, CAGR, monthly returns heatmap. All available in `course/shared/backtesting.py` or directly in pandas/numpy. Affects: D3 L1.

- **Non-stationarity and regime shifts:** The OOS period (2015–2025) spans multiple distinct market regimes: 2015–2019 (low-vol, momentum-driven), 2020 (COVID crash + recovery), 2021 (meme-stock/retail-driven), 2022 (rate shock, value resurgence), 2023–2025 (AI-driven concentration). Cross-sectional factor premia are highly regime-dependent. The sub-period degradation analysis (rigor.md §3.2) is likely to find significant IC variation across halves — expected behavior. The `⚠ SIGNAL DECAY` warning should appear in run_log.txt. Affects: S3, S5, S6, D3.

---

## Data Allocation Summary

| Section / Exercise | Data Source | Universe / Subset | Specifics |
|-------------------|-------------|------------------|-----------|
| S1: Seven Sins / Look-Ahead Bug Demo | yfinance (shared OHLCV) | 449 tickers | Daily close 2012–2025; simple 1-month momentum signal, flawed vs. corrected equity curve |
| S2: Purged Cross-Validation | Week 4 AlphaModelPipeline output (or synthetic fallback) | 449 tickers | Monthly predictions 2015–2025 (132 OOS months); sklearn TimeSeriesSplit comparison; manual PurgedKFold |
| S3: CPCV & Multiple Testing | Week 4 alpha model variants (3 models) + CPCV paths | 449 tickers | Monthly IC series per model; 15–28 CPCV paths; BHY correction on IC t-stats |
| S4: Transaction Cost Decomposition | Week 4 long-short portfolio + OHLCV | 449 tickers | Monthly weights → daily TC via turnover; fixed-rate spread tiers; square-root market impact |
| S5: Deflated Sharpe Ratio | Week 4 long-short gross returns | 449 tickers (aggregated to portfolio) | Monthly return series; rolling sub-windows 24–120 months; M=1–50 trials; MinTRL |
| S6: Responsible Backtest | Week 4 pipeline output + FF3 factors | 449 tickers | Full OOS 2015–2025; naive vs. responsible equity curve comparison; TC-adjusted; DSR verdict |
| Ex 1: Bug Hunt | Fresh momentum signal (independent of Week 4) | 449 tickers | Daily/monthly 2015–2025; three signal variants (look-ahead, survivorship, correct) |
| Ex 2: Purging vs. Walk-Forward | Week 4 alpha predictions | 449 tickers | Monthly 2015–2025; sklearn gap-CV vs. PurgedKFold; IC delta |
| Ex 3: TC Sensitivity Grid | Week 4 long-short portfolio weights | 449 tickers | Monthly turnover; half-spread 2–30 bps × turnover reduction 0–50%; feasibility frontier |
| Ex 4: DSR Calibration | Week 4 gross Sharpe + return distribution | Portfolio (aggregated) | Rolling sub-windows 6/12/24/36/60 months; M=1/5/10/20/50 trials; DSR surface |
| D1: PurgedKFold Engine | Week 4 features + returns (or synthetic) | 449 tickers | Monthly 2015–2025; PurgedKFold class; correctness test; visual diagnostic; sklearn comparison |
| D2: TransactionCostModel | Week 4 long-short portfolio weights + OHLCV | 449 tickers | Monthly weights → daily TC; spread tiers; market impact; net return series; decomposition report |
| D3 L1: Responsible Backtest Report (Baseline) | D1 + D2 outputs + FF3 | Portfolio-level | quantstats tear sheet; DSR at 10 trials; go/no-go verdict |
| D3 L2: Smarter Evaluation (CPCV + BHY) | Week 4 model variants (3) | Portfolio-level | CPCV across 3 models; PBO; BHY correction; winning model selection |
| D3 L3: MinTRL (Frontier) | Surviving model from L2 | Portfolio-level | MinTRL formula applied; qualitative capital-allocation discussion |

---

## Per-Section Expectations

### Lecture Sections

---

### Section 1: The Seven Sins — A Taxonomy of Backtest Failure

**Data:** Daily OHLCV, 2012–2025, 449 tickers via `load_sp500_ohlcv('2012-01-01', '2025-12-31', SP500_TICKERS)`. The key demonstration plots use DEMO_TICKERS (20 tickers) for the equity curve visualization. The quantitative survivorship bias computation uses the full 449-ticker universe. The look-ahead bug demonstration uses a simple 1-day momentum signal (rank by prior day's return, long top 10%, short bottom 10%, rebalance daily); the corrected version shifts signal by one day. No ML model training — purely a data demonstration.

**Acceptance criteria:**
- Flawed momentum strategy (look-ahead) daily equity curve: OOS IC across flawed daily signal ≥ 0.30 (t >> 1.96 — signal is near-trivially predictive because it uses the outcome itself). Corrected signal OOS IC: 0.00–0.06 (consistent with random walk with weak short-term momentum).
- The flawed equity curve must have visibly higher cumulative return than the corrected curve over any 12-month sub-window. Expected annualized return difference ≥ 15 percentage points.
- Survivorship bias quantification: equal-weight portfolio of all 449 tickers (2012–2025 full history) vs. same portfolio restricted to tickers with complete 13-year history. Annualized return difference: 0.5–4.0% per year. If less than 0.5%, the survivorship effect is below noise — flag in execution_log.md. The sign must be positive (survivors inflate returns).
- Data quality block printed before the demonstration: N_stocks, N_periods, missing value pct, survivorship note.
- A structured output summary of all seven sins with a one-line each "how to detect" note. No assertion on this — it is informational.

**Production benchmark:**
- Look-ahead bias return inflation: Luo et al. (2014), Deutsche Bank Markets Research, document look-ahead bugs as the most common single source of backtest failure in systematic strategies, with realistic inflation of 20–50%+ annualized for strategies where the label leaks into the signal. Our daily momentum example is an extreme case (the signal IS the label), so the equity curve divergence will be larger than typical realistic bugs.
- Survivorship bias return inflation: Elton, Gruber & Blake (1996, Journal of Finance) estimated 0.9% per year for mutual fund survivorship; Campbell Harvey (various, CRSP data) estimates 1–4% per year for equity universe survivorship depending on rebalancing methodology. Our 449-ticker universe inflation should fall in the 0.5–3.0% range.

**Pedagogical alternatives:**
- **Primary:** 1-day daily momentum with close-to-close return as both signal and label (classic look-ahead bug).
- **Alternative A:** If the daily equity curve divergence is visually ambiguous (< 10% annualized gap), switch to a monthly signal where the look-ahead uses T+1 month's return to rank at month T — amplifies the IC premium at monthly frequency and makes the divergence unambiguous.
  Trigger: Flawed-vs-corrected annual return difference < 10%.
  Expected effect: Monthly look-ahead IC approaches 0.9+ (nearly perfect predictor), making the curve separation dramatic.

---

### Section 2: Purged Cross-Validation — Closing the Leakage Gap

**Data:** Week 4 `AlphaModelPipeline` predictions (or synthetic fallback: 12-1 month momentum signal), 449 tickers monthly, 2012–2025. Label: 1-month forward return. Label look-forward window: 21 trading days. Purge window: 21 days. Embargo: 5 days after each test fold. Walk-forward (sklearn `TimeSeriesSplit` with gap=21) runs 10 folds on the full 2012–2025 monthly series. `PurgedKFold` runs 10 folds with the same fold boundaries but with purging and embargo applied. The side-by-side split visualization uses 12 months of data for clarity (so the purged vs. contaminated regions are legible).

**ML methodology:**
- Validation: Walk-forward (`TimeSeriesSplit`, gap=21) vs. `PurgedKFold` (k=10, label_duration=21 trading days, embargo=5 trading days) — side-by-side comparison is the section's purpose.
- Train window: Expanding from 2012 (each fold adds previous test period to training).
- Hyperparameter search: Not applicable — uses Week 4's pre-trained model; this section evaluates CV methodology, not model quality.
- Prediction horizon: 1-month forward return (inherited from Week 4).

**Acceptance criteria:**
- `TimeSeriesSplit` (with gap=21) mean CV IC: [0.02, 0.10]. `PurgedKFold` mean CV IC: [0.01, 0.08]. Purged CV IC must be ≤ walk-forward IC (purging removes leakage, reducing estimated signal strength). If purged IC > walk-forward IC in any fold, flag as unexpected — may indicate implementation error in purging logic.
- IC delta (walk-forward minus purged): ≥ 0.005. If the delta is < 0.005, the label look-forward window is too short to create meaningful leakage — flag but do not fail. Expected delta given 21-day labels: 0.005–0.030.
- The split visualization plot must clearly show: (a) the training fold in blue, (b) the test fold in orange, (c) the purged zone in red (overlapping training observations removed), (d) the embargo zone in gray. Verified visually in Step 5.
- `PurgedKFold` produces exactly k train/test split index pairs, each with non-overlapping test folds and with all training observations whose label interval overlaps the test fold removed. Verified: `len(list(pkf.split(X, y))) == k`.
- Structured output prints both IC series (walk-forward and purged) with t-stats.

**Signal viability:** Low — 132 OOS monthly observations. IC t-stat < 1.96 expected (as established in Statistical Implications). Teaching angle: the section demonstrates correct methodology, not signal strength. The point is the IC gap between methods, not the absolute IC level. IC sign consistency (pct_positive across folds) is the secondary indicator.

**Production benchmark:**
- The gap between standard time-series CV and purged CV has been quantified in the machine learning finance literature. López de Prado (2018, Chapters 7–8) demonstrates that for typical financial labels with 20-day forward returns, walk-forward CV overestimates IC by 20–50% relative to purged CV. Our expected delta of 0.005–0.030 on a baseline IC of 0.02–0.05 is consistent with this 20–50% overestimation range.
- Arian, Norouzi & Seco (2024, Knowledge-Based Systems Vol. 305) show in a synthetic controlled environment that purged k-fold produces lower variance IC estimates than walk-forward — at the cost of slightly higher bias. The lower variance is a feature: it means the purged IC estimate is more stable across sub-periods.

---

### Section 3: Combinatorial Purged CV & the Multiple-Testing Problem

**Data:** Same as Section 2 — Week 4 predictions for 3 model variants (linear/Ridge, LightGBM, feedforward NN from Week 4's model comparison deliverable D3), monthly, 2015–2025 (132 OOS months). CPCV parameters: k=6 folds, producing C(6,2)=15 combinatorial train-test paths. For the "50 variants" simulation that motivates the PBO discussion: generate 50 synthetic variants by permuting feature subsets on the Week 4 feature matrix (or via synthetic signal, if fallback active) — each variant is a different random feature subset of size 3–7 drawn from the full feature set. The BHY correction applies to the 3 IC t-statistics from the 3 model variants.

**ML methodology:**
- Validation: Combinatorial purged CV (CPCV, k=6, label_duration=21 trading days, embargo=5 trading days) — generates 15 combinatorial train-test paths per model variant.
- Train window: Expanding within each CPCV combinatorial partition (consistent with Section 2 and López de Prado Chapter 8).
- Hyperparameter search: Not applicable — uses Week 4's pre-trained model variants; this section evaluates overfitting risk, not model quality.
- Prediction horizon: 1-month forward return (inherited from Week 4).

**Acceptance criteria:**
- CPCV produces exactly 15 combinatorial paths for k=6 (verified: `len(cpcv_paths) == 15`).
- PBO ∈ [0.20, 0.75]. If Week 4 models have genuine alpha: PBO ∈ [0.25, 0.55] (the in-sample winner beats median OOS rank in at least 45–75% of paths). If synthetic fallback: PBO may be higher (0.40–0.70) because momentum variants are less stable across sub-periods. PBO outside [0.20, 0.75] warrants investigation: PBO < 0.20 suggests the models are suspiciously consistent (possible leakage); PBO > 0.75 suggests all variants are fitting noise.
- The distribution plot of OOS returns for the IS winner shows visible spread. The OOS returns of the in-sample winner across 15 CPCV paths must have standard deviation > 0 — if all 15 OOS returns are identical, the CPCV was implemented incorrectly.
- BHY correction on 3 IC t-statistics: the corrected p-values must be ≥ the raw p-values (correction always inflates or maintains p-values). If all 3 raw p-values > 0.05, BHY correction changes nothing practically. If any raw p-value < 0.05, the BHY-adjusted p-value for the best model may or may not remain < 0.05 — either result is pedagogically valid.
- Harvey-Liu-Zhu t-stat hurdle demonstration: compute observed t-stats for all 3 models. Report how many clear t > 3.0 vs. t > 1.96. Expected: 0 models clear t=3.0 (IC signal is weak with n=132 months). This is the teaching point: most signals that "look significant" at t=1.96 do not meet the production hurdle.
- Structured output: `N_variants_tested=50, N_passing_t196=[count], N_passing_t300=[count]` (for the 50-variant simulation).

**Signal viability:** Low — same 132-month series as Section 2. CPCV does not change the signal; it changes the evaluation framework. Interpret PBO with awareness that low n_paths (15) means each PBO estimate has high variance.

**Production benchmark:**
- Bailey, Borwein, López de Prado & Zhu (2015, Journal of Computational Finance) established PBO as the fraction of CPCV paths where the in-sample best strategy underperforms the median OOS strategy. In their simulations, pure noise strategies produce PBO near 0.5; well-specified strategies with genuine alpha produce PBO near 0.25–0.35. Our expected range (0.25–0.55) spans both regimes.
- Harvey, Liu & Zhu (2016, Journal of Finance) established that with ~316 factors published through 2012, any new single-factor claim should require t > 3.0 for the claim to have a 5% false discovery rate. As of 2024 the hurdle is arguably higher. Our 3-model evaluation is not factor discovery per se, but the t=3.0 framing illustrates why the traditional p<0.05 threshold is insufficient for financial prediction.

---

### Section 4: Transaction Cost Decomposition

**Data:** Week 4 long-short portfolio weight matrix (stocks × months), 449 tickers, 2015–2025. Daily OHLCV (all 449 tickers, 2015–2025) for Corwin-Schultz spread estimation and volume-based impact parameters. Three cost regimes: (a) zero cost, (b) fixed 10 bps one-way half-spread for all stocks, (c) market-cap-tiered spread (large-cap ~5–8 bps, mid-cap ~12–20 bps, derived from `fundamentals_mcap.parquet`). Market impact: Almgren-Chriss square-root model using daily volume from OHLCV. Impact coefficient η = 0.1 (standard research default per Almgren & Chriss 2000). Participation rate = portfolio weight change / 30-day ADV.

**Acceptance criteria:**
- Turnover series computed correctly: one-way monthly turnover = sum of absolute weight changes / 2, averaged across months. Expected range given Week 4 naive full-rebalance: 30–100% one-way per month. If turnover < 10%, the portfolio is being held nearly static — flag as implementation issue.
- Spread cost (regime b, fixed 10 bps): monthly spread drag = turnover × 2 × 0.001. At 50% one-way turnover: drag ≈ 0.10% per month = 1.2% per year. At 100% one-way: drag ≈ 2.4% per year. Final spread drag ∈ [0.5%, 4.0%] annualized.
- Market impact cost: must be > 0 for any month where the portfolio trades. Impact ∈ [0.0%, 2.0%] annualized for a strategy of this size (implementation shortfall via square-root law). Impact > gross spread at realistic participation rates (>5% of daily volume) — flag with `⚠ HIGH IMPACT` in structured output.
- Net return series: R_net = R_gross − (spread_cost + impact_cost) per period. Net annualized Sharpe ratio must be lower than gross Sharpe ratio (TC always drag; net > gross is a sign of bug). Expected net Sharpe: 0.1–0.7 (wide range due to Week 4 signal uncertainty and turnover).
- Three-regime equity curve plot: zero-cost curve must be above the fixed-spread curve, which must be above the tiered-spread curve at every point. If the ordering is violated, the cost subtraction has a sign error.
- Structured output: `GROSS_SHARPE={:.3f}, NET_SHARPE={:.3f}, SPREAD_DRAG={:.3f}, IMPACT_DRAG={:.3f}, TOTAL_TC_DRAG={:.3f}, ONE_WAY_TURNOVER={:.1%}`.
- `⚠ HIGH TURNOVER` warning printed if one-way monthly turnover > 50% (per rigor.md §3.1).
- Skewness and excess kurtosis reported alongside Sharpe for all three regime return series (per rigor.md §3.3).

**Production benchmark:**
- Almgren & Chriss (2000, Journal of Risk) establish that for a $100M portfolio trading S&P 500 stocks, market impact via the square-root model typically adds 2–5 bps per trade (one-way) at 5% daily participation. At institutional AUM ($1B+), impact can exceed spread cost. Our research-grade model (no AUM assumption) will produce lower impact estimates — that's correct; note in structured output that production impact scales with AUM.
- Practitioner consensus (López de Prado GARP 2017 "10 Reasons ML Funds Fail"; QuantStart practitioner blog): the most common gap between backtest and live performance is transaction cost underestimation. The typical pattern is gross return 25–35% → net return 12–20% after realistic costs at institutional AUM. Our sandbox model operates at lower AUM assumptions, so the gap will be smaller but still directionally meaningful.
- Corwin & Schultz (2012, Journal of Finance): Corwin-Schultz spread estimator calibrated on NYSE data produces median spread estimates of 5–15 bps for large-caps, 20–50 bps for small-caps. Our fixed-rate proxy (5–8 bps large-cap, 12–20 bps mid-cap) is conservative relative to NYSE-observed spreads from 2005–2012 but approximately in line with current post-decimalization large-cap spreads.

---

### Section 5: The Deflated Sharpe Ratio

**Data:** Week 4 long-short portfolio monthly gross return series, 2015–2025 (132 months). FF3 risk-free rate (RF column) via `load_ff_factors(model='3', frequency='M')` for excess return computation. The DSR surface uses rolling sub-windows of 24, 36, 48, 60, 84, and 120 months (trailing windows ending at the most recent period). The MinTRL computation uses the full 132-month series. Trials parameter M swept from 1 to 50.

**ML methodology:** Not applicable — this section evaluates a pre-existing return series. No model training.

**Acceptance criteria:**
- Probabilistic Sharpe Ratio (PSR): `PSR = Phi[(SR_hat - SR*) × sqrt(T-1) / sqrt(1 - gamma3×SR_hat + (gamma4+2)/4×SR_hat²)]`, where gamma3 = skewness, gamma4 = excess kurtosis (normal = 0; use `scipy.stats.kurtosis(returns, fisher=True)`). Note: the original Bailey-López de Prado (2014) paper uses kurtosis (normal=3) with coefficient (κ-1)/4; with excess kurtosis γ₄ = κ-3, this becomes (γ₄+2)/4. At T=132, SR=0.7, M=1: PSR ≈ 1.0. At T=24, SR=0.3, M=50: PSR ≈ 0.10–0.30. The PSR function must output values in [0, 1].
- DSR surface: the surface must show visible degradation as M increases at fixed T, and as T decreases at fixed M. Monotonicity: DSR(M=1) ≥ DSR(M=10) ≥ DSR(M=50) for any fixed T > 12. DSR(T=120) ≥ DSR(T=36) ≥ DSR(T=24) for any fixed M > 1. Violations indicate a bug in the SR*(M) hurdle computation.
- MinTRL: for the observed annual Sharpe ratio (expected 0.5–1.2), MinTRL ∈ [6, 30] months at 95% confidence. MinTRL increases as Sharpe decreases: MinTRL(SR=0.5) > MinTRL(SR=0.8) > MinTRL(SR=1.2). Verify: computed T* = ((z_alpha / SR)² × (1 − gamma3×SR + (gamma4+2)/4×SR²)) + 1, where gamma4 = excess kurtosis (same convention as PSR formula above).
- At full 132-month OOS: DSR at M=10 trials ≥ 0.95 for gross SR ≥ 0.5 (as confirmed by code probe). If DSR at M=10 < 0.50 on the full OOS period, the gross Sharpe is below 0.2 — flag as signal unusually weak, not a formula error.
- Skewness and excess kurtosis of the return series reported alongside every Sharpe computation. If |skewness| > 1 or excess_kurtosis > 3: note that Sharpe understates tail risk (per rigor.md §3.3).
- Structured output: `SR_HAT={:.3f}, SKEW={:.3f}, EX_KURT={:.3f}, T_MONTHS={d}, MINTRL_MONTHS={d}, DSR_M1={:.3f}, DSR_M10={:.3f}, DSR_M50={:.3f}`.

**Production benchmark:**
- Bailey & López de Prado (2014, Journal of Portfolio Management): the DSR formula is defined in this paper. Their worked example shows a strategy with observed SR=2.5, T=1250 trading days, M=100 trials. DSR adjusts the effective hurdle upward, requiring the raw SR to be substantially higher than zero to remain statistically significant. For typical ML fund performance (annual SR 0.5–1.5 before costs, 3–5 year track records, 10–50 strategy variants tested), DSR frequently falls below 0.5, meaning most fund backtests would fail this test. This is the paper's central empirical claim and the section's "so what?".
- Arian, Norouzi & Seco (2024) confirm in simulation that DSR-based evaluation reduces false positive rates in strategy selection by 30–50% relative to naive Sharpe comparison. This supports framing DSR as a default evaluation tool, not an exotic correction.

**Pedagogical alternatives:**
- **Primary:** DSR surface computed over the full range of (T, M) combinations as described above.
- **Alternative A:** If the gross Sharpe is so high (> 1.5) that the DSR surface is flat near 1.0 across all pedagogically interesting (T, M) combinations, use the net-of-cost Sharpe (typically 0.3–0.5 lower) as the input to the DSR surface. The net Sharpe will produce more interesting degradation.
  Trigger: DSR at T=24, M=50 > 0.90 (surface too flat to teach from).
  Expected effect: Using net SR ≈ 0.3–0.5 instead of gross SR ≈ 0.7–1.0 pushes DSR into the pedagogically interesting [0.10, 0.70] range at short track records.

---

### Section 6: The Responsible Backtest — Putting It Together

**Data:** Week 4 AlphaModelPipeline predictions + long-short portfolio weights, 449 tickers monthly, 2015–2025. TC model from Section 4 (net return series). FF3 factors (Mkt-RF, SMB, HML, RF), monthly, 2015–2025. Naive evaluation: sklearn walk-forward CV, no TC, no DSR correction. Responsible evaluation: PurgedKFold CV, net-of-cost returns, DSR at M=10. Both pipelines run on the same underlying data and produce comparable equity curves and metrics.

**Acceptance criteria:**
- Naive equity curve (no purging, no TC, no correction): annualized Sharpe ∈ [0.5, 1.8]. Higher Sharpe than responsible evaluation — required by construction (the two corrections always reduce metrics).
- Responsible equity curve (purged CV + net TC): annualized Sharpe ∈ [0.1, 1.2]. Must be strictly lower than naive Sharpe. If responsible Sharpe ≥ naive Sharpe, there is a sign error in the TC subtraction or a bug in the purging logic.
- The difference (naive Sharpe − responsible Sharpe) ∈ [0.1, 0.8]. This gap is the "backtest-to-live performance gap" and is the section's central quantitative result. If the gap < 0.05, the corrections are negligibly small — flag and investigate.
- DSR at M=10 trials, using the responsible (net, purged) return series over the full 132-month OOS window: ≥ 0.50 is a "pass" verdict; < 0.50 is a "no-deploy" verdict. Either outcome is acceptable — the verdict is the teaching point, not the direction.
- Sub-period degradation (rigor.md §3.2): first-half IC (2015–2019, 60 months) and second-half IC (2020–2025, 72 months) reported separately. If second-half IC < 50% of first-half IC: `⚠ SIGNAL DECAY` printed.
- Structured output covers: naive Sharpe (gross, OOS), responsible Sharpe (net, purged CV), delta, DSR verdict at M=10, sub-period IC split, `Universe: current S&P 500 — subject to survivorship bias` note.
- IS/OOS explicitly labeled in all plots and structured output.
- Both IS and OOS Sharpe ratios reported (IS should be higher — if IS < OOS, flag for investigation).

**Production benchmark:**
- López de Prado (2017, "10 Reasons ML Funds Fail", GARP): documents that the average ML fund's live Sharpe ratio is approximately 40–60% of the backtest Sharpe ratio, primarily due to transaction costs, regime non-stationarity, and overfitting to backtest period. Our expected gap (naive − responsible Sharpe of 0.1–0.8) is consistent with this range.
- Gu, Kelly & Xiu (2020, Review of Financial Studies): report OOS R² of ~0.35% monthly for gradient boosting on CRSP data, translating to IC ≈ 0.04–0.05. With a survivorship-biased S&P 500 universe and fewer features, our responsible IC should be in the 0.01–0.04 range (lower due to smaller universe, fewer features, and no PIT fundamental data).

---

### Seminar Exercises

---

### Exercise 1: The Look-Ahead Bug Hunt

**Data:** Three independent momentum signals computed fresh (not from Week 4 AlphaModelPipeline), all on 449 tickers monthly, 2015–2025 (132 OOS months):
- **Signal A (look-ahead bug):** Ranks stocks by month T's return to predict month T's return (signal formation uses the outcome itself). Monthly IC approaches 1.0 in-sample.
- **Signal B (survivorship bias):** Uses only tickers with complete 2015–2025 history (survivorship-clean subset ≈ 420–440 tickers) AND the full universe for evaluation — the training signal is computed on a different (cleaner) universe than the test universe. Both in-sample and out-of-sample IC are inflated relative to Signal C.
- **Signal C (correct):** Standard 12-1 month momentum signal, using return from month T-12 to T-1 (skipping T) to rank at month T-1, evaluated on month T return. Walk-forward split, no leakage.

The student receives three pre-computed IC series (one per signal) and must diagnose which fingerprint belongs to which bug type. The IS and OOS splits are 2015–2019 (in-sample) and 2020–2025 (out-of-sample).

**Acceptance criteria:**
- Signal A IC series: mean IS IC ≥ 0.50 (close to 1.0 for the contaminated months), mean OOS IC ≤ 0.10 (collapses out-of-sample — the signal was using data not available at prediction time). IS/OOS IC ratio ≥ 5×.
- Signal B IC series: mean IS IC ≥ 0.03, mean OOS IC ≥ 0.02. IS/OOS IC ratio ≤ 2× (survivorship bias inflates both halves nearly equally, so the degradation is gradual, not a collapse). IS IC modestly above OOS IC.
- Signal C IC series: mean IS IC ∈ [0.01, 0.06], mean OOS IC ∈ [0.01, 0.05]. IS/OOS IC ratio: 0.8–2.0 (small degradation, no collapse — the corrected signal behaves consistently across regimes). IS IC approximately equal to or modestly above OOS IC.
- Student (code) must correctly identify the fingerprint patterns. This is a labeled comparison — acceptance criterion is that the IC series have the described properties so the "discovery" is unambiguous. If Signal A's IS/OOS IC ratio is < 3×, the look-ahead bug is not producing the intended effect — revisit signal construction.
- IS/OOS clearly labeled in all output.

**Production benchmark:**
- Luo et al. (2014, Deutsche Bank Markets Research): describe survivorship bias as producing "deceptively smooth equity curves with modest but consistent outperformance" — consistent with Signal B's near-equal IS/OOS inflation. Look-ahead bugs, by contrast, produce "explosively profitable in-sample with immediate collapse" — consistent with Signal A's fingerprint.

---

### Exercise 2: Purging vs. Walk-Forward — The CV Comparison

**Data:** Week 4 AlphaModelPipeline predictions (or synthetic fallback), 449 tickers monthly, 2012–2025. CV configuration: 10 folds for both methods. Walk-forward: sklearn `TimeSeriesSplit(n_splits=10, gap=21)`. Purged k-fold: manual implementation (or using the student's D1 PurgedKFold class if D1 was completed first, otherwise a provided reference implementation), k=10, label_duration=21 days, embargo=5 days.

**Acceptance criteria:**
- Walk-forward mean CV IC: [0.01, 0.10]. Purged k-fold mean CV IC: [0.01, 0.09]. PurgedKFold IC ≤ walk-forward IC (purging always removes some positively-biased observations). If reversed, flag implementation.
- IC delta (walk-forward minus purged): ≥ 0.002 on absolute mean IC. Ideally 0.005–0.030. If the delta is near zero, the label duration is short enough that overlap contamination is minimal — note this and interpret: "For a 21-day label on monthly frequency, the purging adjustment is modest because monthly periods already approximate the label duration."
- Model rank flip check: for each fold, rank the IC of all available model variants (linear, LightGBM, NN). Count the number of folds where the ranking under walk-forward ≠ the ranking under purged CV. Report: `RANK_FLIP_FOLDS={count} / {total_folds}`. Expected: 2–6 out of 10 folds show rank flips, illustrating that methodology affects which model looks best. If zero rank flips occur, the models' relative performance is robust to the CV method — note this as a nuanced finding rather than a failure.
- Structured output: both IC series with mean ± std, t-stats, and delta.
- IC sign consistency (pct_positive across folds) reported for both methods alongside IC level.

**Signal viability:** Low — same signal as Section 2. The exercise measures the CV method gap, not absolute signal strength.

**Production benchmark:**
- López de Prado (2018, Chapters 7–8): demonstrates on real data that the difference in CV score between walk-forward and purged k-fold is typically 10–30% of the raw IC value. For IC ≈ 0.03, expect delta ≈ 0.003–0.009. Our expected delta (0.005–0.030) is consistent with this range.

---

### Exercise 3: Transaction Cost Sensitivity Analysis

**Data:** Week 4 long-short portfolio weight matrix, 449 tickers monthly, 2015–2025. Gross return series from Section 4 or computed inline. TC sensitivity grid: half-spread from 2 to 30 bps (steps: 2, 5, 8, 10, 12, 15, 20, 25, 30 bps) × one-way turnover reduction from 0% to 50% (steps: 0%, 10%, 20%, 30%, 40%, 50%). Turnover reduction implemented by zeroing out weight changes below a conviction threshold (e.g., only trade if |Δweight| > threshold). Net Sharpe computed for each (spread, turnover_reduction) grid cell.

**Acceptance criteria:**
- Feasibility frontier: the boundary in the (spread, turnover) grid where net Sharpe = 0.5 (typical institutional hurdle rate for a new strategy). The frontier must be a monotonically decreasing curve: higher spreads require more turnover reduction to remain feasible. If the frontier is not monotonic, there is a sign error in the TC calculation.
- At zero spread (column 1), net Sharpe = gross Sharpe (no cost drag). At 30 bps spread with 0% turnover reduction: net Sharpe must be substantially lower than gross Sharpe. Expected: net Sharpe ≤ gross Sharpe − 0.3 at (30 bps, 0% reduction).
- The grid heatmap must be generated with net Sharpe values displayed per cell. Cells above the feasibility frontier are green; cells below are red (or equivalent visual encoding). The heatmap must show a visible transition from viable to non-viable.
- Turnover reduction effect: at 50% turnover reduction and 10 bps spread, the strategy must be viable (net Sharpe ≥ 0.3) if gross Sharpe ≥ 0.5. If the strategy remains non-viable even at 50% turnover reduction and low spread, the gross Sharpe is too weak for any TC level — flag and note.
- Grid viability check: if FEASIBLE_CELLS = 0 (no cell has net Sharpe ≥ 0.5), print `⚠ STRATEGY NON-VIABLE: no (spread, turnover_reduction) combination produces net Sharpe ≥ 0.5. Gross Sharpe too weak for any realistic TC regime.` This is not a code failure — it is a valid and pedagogically powerful result showing the strategy cannot survive implementation costs. Continue to the pedagogical alternative below.
- Structured output: `GROSS_SHARPE={:.3f}, FEASIBLE_CELLS={d}/{total}, BREAKEVEN_SPREAD_AT_0PCT_REDUCTION={:.0f}bps`.

**Production benchmark:**
- (No published benchmark found — range estimated from formula.) At gross IC ≈ 0.03–0.05 for a monthly long-short decile portfolio with ~50% monthly one-way turnover, the breakeven half-spread (where net return = 0) is approximately 10–20 bps. Reducing turnover by 20–30% can push the breakeven to 15–25 bps. This is consistent with large-cap S&P 500 strategies (5–8 bps actual spread) being viable but mid-cap strategies (12–25 bps) being marginal.

**Pedagogical alternatives:**
- **Primary:** Full (spread, turnover_reduction) grid as described.
- **Alternative A:** If gross Sharpe < 0.3, the feasibility frontier falls entirely in the low-spread region and the heatmap shows mostly red. In this case, extend the grid to include an additional dimension: prediction horizon (monthly vs. quarterly). Quarterly signals have lower turnover by construction, which shifts the frontier dramatically.
  Trigger: Net Sharpe at (10 bps, 0% reduction) < 0.1.
  Expected effect: Quarterly horizon reduces one-way turnover by ~60%, making the strategy viable at a wider spread range.

---

### Exercise 4: DSR Calibration — How Many Strategies Did You Actually Try?

**Data:** Week 4 long-short portfolio monthly return series, 2015–2025, aggregated to portfolio level (no per-stock data needed). The return series is truncated to rolling sub-windows of length T ∈ {6, 12, 24, 36, 60} months (trailing windows ending at 2025-12-31). Trials M ∈ {1, 5, 10, 20, 50}. DSR computed for all 5 × 5 = 25 (T, M) combinations using the Bailey-López de Prado formula with skewness and excess kurtosis from the return series (same convention as Section 5: excess kurtosis with (γ₄+2)/4 coefficient).

**Acceptance criteria:**
- DSR surface: for each (T, M) cell, DSR ∈ [0.0, 1.0]. Monotonicity: DSR decreases weakly as M increases (more trials = lower DSR). DSR decreases as T decreases (shorter track record = lower DSR). The 5×5 grid should show a clear gradient from high-DSR (short M, long T) to low-DSR (large M, short T).
- Crossover point: identify the (T, M) cells where DSR first crosses below 0.5 (strategy fails the test). Expected: at net SR ≈ 0.3–0.5, DSR < 0.5 appears at T ≤ 24 months with M ≥ 10 trials. If no cell falls below 0.5, the gross Sharpe is unusually high — note and investigate survivorship inflation.
- The 6-month window at M=50 trials must show DSR substantially lower than the 60-month window at M=1 trial. Expected: DSR(T=6, M=50) ≤ 0.50, DSR(T=60, M=1) ≥ 0.90. If this ordering fails, check the MinTRL formula.
- Skewness and excess kurtosis extracted from the return series and passed correctly to the PSR formula. If skewness = 0.0 and excess_kurtosis = 0.0 (defaults, not actual values), flag as potential bug — these should be computed from the actual return distribution.
- Structured output per (T, M) cell: `T={d}mo, M={d}: SR={:.3f}, DSR={:.3f} ({PASS/FAIL})`.

**Production benchmark:** Not applicable. This exercise applies the Bailey-López de Prado (2014) DSR formula to a pre-computed return series; the formula is the benchmark. The relevant production reference (expected DSR behavior for realistic fund track records) was established in Section 5 and is not separately cited here to avoid redundancy.

---

### Homework Deliverables

---

### Deliverable 1: Build a Purged CV Engine

**Data:** Week 4 AlphaModelPipeline features and returns (or synthetic fallback), 449 tickers monthly, 2012–2025. For the correctness test: a synthetic DatetimeIndex DataFrame where the "label" for row T is the forward return from T to T+21 days (known by construction). The correctness test verifies that after purging, no training sample in fold k has a label window that overlaps with fold k's test period. The sklearn comparison uses the full feature matrix from Week 4.

**ML methodology:**
- Validation: `PurgedKFold` (k=10, label_duration=21 trading days, embargo=5 trading days).
- Train window: Expanding (each fold adds the preceding fold's test period to training, consistent with López de Prado Chapter 7).
- Hyperparameter search: Not applicable — this deliverable builds the CV splitter, not a model. The IC comparison uses the same model as Week 4 (LightGBM with Week 4's hyperparameters).
- Prediction horizon: 1 month forward return (same as Week 4 and Sections 2–3).

**Acceptance criteria:**
- **Correctness test (non-negotiable):** Given a synthetic DataFrame with DatetimeIndex, label window = 21 days, and k=5 folds, `PurgedKFold.split()` must return train/test index pairs where no training index i satisfies `label_end[i] ≥ test_start`. Verified by scanning every returned train index. If this test fails, the purging logic is incorrect — the deliverable fails regardless of other results.
- `len(list(pkf.split(X)))` == k for any valid k. Verify k=5, 10.
- The `PurgedKFold` class must accept: `DatetimeIndex` input, `label_duration` in trading days, `embargo` in trading days, `n_splits` (k). Must raise `ValueError` for invalid inputs (negative duration, k > n_samples).
- IC comparison: `PurgedKFold` mean IC ≤ `TimeSeriesSplit` mean IC (same as Section 2 and Exercise 2 — confirmed direction). The deliverable must report this delta with t-stat and sample size.
- Visual diagnostic plot: for a sample DatetimeIndex with 60 months of data, plot the train/test/purged/embargo splits for k=5. Purged zone must be visible (non-zero width). Plot must be labeled with legend identifying each zone.
- Code quality: `PurgedKFold` is a class with `split(X, y=None)` method and `__repr__`. Docstring explains the purging logic. No global variables. SEED = 42 set at top if randomness is used.

**Production benchmark:**
- The López de Prado PurgedKFold algorithm is defined in Chapter 7 of *Advances in Financial Machine Learning* (Wiley, 2018). The `mlfinpy` library (MIT license) provides a reference implementation for verification. Students may compare their `PurgedKFold.split()` output against `mlfinpy.cross_validation.PurgedKFold` on the same dataset. Index-level agreement with `mlfinpy` is not required (implementations may differ in edge handling), but the directional result (IC reduction vs. sklearn `TimeSeriesSplit`) must match.

---

### Deliverable 2: Transaction Cost Accounting Pipeline

**Data:** Week 4 long-short portfolio weight matrix (stocks × months), 449 tickers, 2015–2025. Daily OHLCV for spread estimation and volume-based impact (same dataset as Section 4). Three spread regimes: optimistic (3 bps flat), base (market-cap tiered: large-cap 5 bps, mid-cap 15 bps), pessimistic (flat 25 bps). Market impact coefficient η = 0.1 (Almgren-Chriss default).

**ML methodology:** Not applicable — no model training. This deliverable builds a cost accounting layer.

**Acceptance criteria:**
- `TransactionCostModel` class accepts: weight matrix (DataFrame, stocks × dates), spread assumption (float or Series), impact coefficient (float). Returns an object with attributes: `.turnover` (Series, monthly one-way), `.spread_cost` (Series, monthly), `.impact_cost` (Series, monthly), `.net_returns` (Series, monthly), `.report()` method.
- **Correctness check:** For a synthetic 2-asset portfolio where weights shift from [1.0, 0.0] to [0.0, 1.0] in one period (100% one-way turnover), spread cost must equal: `spread_bps × 2 × 1.0 / 10000`. Verify this analytically. If the formula produces a different result, the TC accounting has a sign or scaling error.
- Turnover series: mean monthly one-way turnover ∈ [10%, 100%]. `⚠ HIGH TURNOVER` warning printed if any month > 50%.
- Spread cost series: all values ≥ 0 (costs are always a drag). If any value < 0, there is a sign error.
- Net return series: R_net = R_gross − spread_cost − impact_cost. Max drawdown on net returns ≥ max drawdown on gross returns (costs never reduce drawdown). Expected exception: in months where TC costs are very large and the strategy would have had a loss, net loss > gross loss. This is correct behavior.
- Three-spread sensitivity table: annualized Sharpe for optimistic/base/pessimistic regimes. Pessimistic Sharpe < base Sharpe < optimistic Sharpe. All three values printed in structured output.
- Highest-cost period identification: `report()` prints the 5 months with largest total TC drag and their dominant cost component (spread-dominated vs. impact-dominated).
- Gross vs. net Sharpe in structured output: `GROSS_SHARPE={:.3f} | NET_SHARPE_OPT={:.3f} | NET_SHARPE_BASE={:.3f} | NET_SHARPE_PESS={:.3f}`.
- Skewness and excess kurtosis reported for gross and net return series alongside Sharpe (per rigor.md §3.3).

**Production benchmark:**
- Almgren & Chriss (2000, Journal of Risk): the square-root market impact model predicts temporary impact = σ × η × sqrt(Q / V), where Q = trade size, V = average daily volume, σ = daily volatility, η ≈ 0.1–0.2. For large-cap S&P 500 stocks trading at ~5% ADV participation rate, this produces 3–8 bps temporary impact per trade. Our model with η=0.1 is at the lower end of this range — appropriate for a research-grade approximation without AUM assumptions.

---

### Deliverable 3: The Responsible Backtest Report

**Data:** D1 `PurgedKFold` output (or Section 2 purged CV results if D1 not yet complete), D2 `TransactionCostModel` output (or Section 4 net returns if D2 not yet complete), Week 4 model variants (linear, LightGBM, NN) for CPCV in Layer 2, FF3 factors monthly for quantstats benchmark.

**ML methodology:**
- Validation: `PurgedKFold` (k=10) for all OOS evaluation in Layers 1 and 2.
- Train window: Expanding from 2012, OOS from 2015 (3-year initial window, consistent with Week 4 setup).
- Hyperparameter search: Use Week 4's hyperparameters for all models (no re-tuning in this deliverable — the strategy is "locked" before the final backtest, per López de Prado's "backtesting is not a research tool" principle).
- Prediction horizon: 1 month forward return (same as Week 4).

**Signal viability (Layer 1):** Low — same 132-month signal as established throughout the week. IC t-stat expected below 1.96. DSR at M=10 is the primary go/no-go signal at the portfolio level, not IC significance.

**Acceptance criteria:**

**Layer 1 (Baseline — non-negotiable):**
- `quantstats` tear sheet generated and saved. Must include: Sharpe, Sortino, max drawdown, Calmar ratio, CAGR, monthly returns heatmap. If quantstats fails: manual computation of all five metrics with equivalent formatting.
- Annual Sharpe (net, purged CV) ∈ [0.05, 1.5]. Lower bound verifies the strategy has positive expected return. Upper bound verifies survivorship/TC accounting has not been omitted.
- Max drawdown ≤ −10% (any real equity long-short strategy experiences meaningful drawdowns; a max drawdown > −3% suggests TC or returns are computed incorrectly).
- DSR at M=10 computed and printed. Verdict: `DSR={:.3f} → {'DEPLOY' if DSR >= 0.5 else 'NO-DEPLOY'}`. Either verdict is acceptable. If DSR < 0.5 → Layer 2 proceeds anyway (D3 explicitly continues to Layer 2 regardless of Layer 1 verdict per blueprint design).

**Layer 2 (Smarter Evaluation — required):**
- CPCV across 3 model variants (k=6, 15 paths): PBO ∈ [0.20, 0.75] (same bounds as S3 — outside this range warrants investigation). PBO computed as fraction of 15 paths where the IS winner's OOS rank < median OOS rank. Printed: `PBO={:.3f} → {'PROCEED' if PBO <= 0.5 else 'OVERFIT'}`.
- Winning model selection: the model with (a) PBO ≤ 0.5 AND (b) highest net-of-cost Sharpe under purged CV. If no model satisfies both conditions, select by highest net Sharpe alone and note the overfit concern.
- BHY correction applied to 3 IC t-statistics. Printed: for each model, `raw_p={:.4f}, bhy_adjusted_p={:.4f}`. The adjusted p-values must be ≥ raw p-values (BHY only inflates p-values, never deflates).
- Final report states the winning model with justification covering: IC (raw and BHY-corrected), PBO result, net Sharpe under purged CV, DSR at M=10.

**Layer 3 (Frontier — qualitative + formula):**
- MinTRL formula applied to the winning model's observed Sharpe (net, purged). MinTRL printed: `MINTRL_95pct={d} months ({:.1f} years) of live trading to confirm observed Sharpe with 95% confidence.`
- Qualitative discussion (minimum 3 sentences in the execution_log.md developer notes): how a fund researcher would use the MinTRL number in a capital allocation decision. Must address: (1) what happens during the MinTRL monitoring period (continue paper trading or small live allocation), (2) the relationship between MinTRL and strategy decay risk, (3) why MinTRL is a floor, not a guarantee.
- No acceptance criterion on the qualitative discussion content — but it must be present and non-trivial (not "the researcher uses the number to decide").

**Production benchmark:**
- López de Prado (2017, GARP "10 Reasons ML Funds Fail"): Fund researchers at institutional shops produce a formal evaluation report for every new strategy before capital allocation. The report structure maps directly to D3's three layers: Layer 1 is the standard performance report, Layer 2 is the overfitting assessment, Layer 3 is the production readiness assessment (MinTRL + capital allocation protocol). D3 teaches the structure of this report, not just the metrics.
- Gu, Kelly & Xiu (2020, Review of Financial Studies): the industry-academic benchmark for ML equity alpha on free data. Their best-performing model (gradient boosting, full CRSP, 94 features, 1957–2016) achieves OOS R² ≈ 0.35% monthly. Our sandbox model should produce substantially lower OOS R²; if it exceeds 0.5%, survivorship bias is likely inflating results beyond what the signal quality justifies.

---

## Open Questions

1. **Whether the Week 4 long-short portfolio produces a gross Sharpe above or below 0.5 over the 2015–2025 OOS window.**
   **Why unknowable:** The Week 4 pipeline has not yet been run against this expectations document. The Sharpe depends on the specific hyperparameters, feature set, and model architecture choices made by the Week 4 code agent.
   **Affects:** S4 (TC drag magnitude), S5 (DSR surface pedagogical content), S6 (responsible backtest verdict), D3 L1 (go/no-go decision).
   **If Sharpe ≥ 0.5 (gross):** The TC and CV corrections produce a meaningful but not fatal reduction. Section 6's "responsible vs. naive" comparison will show a visible gap with the responsible strategy remaining viable. DSR at M=10 likely passes over the full OOS window.
   **If Sharpe < 0.5 (gross):** The strategy may not survive TC + CV corrections. Net Sharpe could be near zero or negative. DSR will fail at most (T, M) combinations except very short track records with few trials. This is pedagogically equally valuable — the student sees a real example of a strategy that fails responsible evaluation.

2. **Whether all three Week 4 model variants (linear, LightGBM, NN) are available in the cache for CPCV in Section 3 and D3 L2.**
   **Why unknowable:** Week 4's homework D3 is the source of the model comparison. If only one or two models were successfully trained and cached, the CPCV across variants cannot proceed as designed.
   **Affects:** S3 (CPCV demonstration), D3 L2 (smarter evaluation with PBO).
   **If all 3 available:** CPCV runs as designed with 15 paths across 3 variants. PBO is interpretable.
   **If fewer than 3:** Synthesize additional variants by varying the feature set (momentum-only, value-only, combined) using the shared data layer. These are valid "model variants" for the CPCV demonstration and produce a comparable distribution of OOS performance.

