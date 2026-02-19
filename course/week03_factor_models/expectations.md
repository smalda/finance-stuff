# Week 3: Factor Models & Cross-Sectional Analysis — Expectations

**Band:** 1 — Essential Core
**Implementability:** HIGH

---

## Data Plan

### Dataset 1: Equity Prices

#### Universe

- **Option A: 200 S&P 500 stocks** — Sufficient cross-sectional breadth for double-sort portfolio construction (6 portfolios with ~33 stocks per bucket), Fama-MacBeth regressions (200 observations per cross-section), and Barra-style regressions. Download time ~15s via `yf.download()`. 192 of 201 tested tickers had >95% completeness over 2014-2024 (verified via code probe). Five tickers showed delisting warnings (`ABC`, `LM`, `SQ`, `WRK`, `WBA`), easily replaced.
- **Option B: 100 S&P 500 stocks** — Faster downloads, simpler debugging, but double-sort portfolios have only ~16 stocks per bucket (marginal for value-weighted returns). Fama-MacBeth cross-sections become thin for 5-7 regressors.
- **Option C: Full S&P 500 (~500 stocks)** — Maximum cross-sectional power. Price download ~30s. Fundamental download ~3.3 minutes. Diminishing pedagogical returns vs. 200 — same large-cap bias problem (see Risk), more download failures to handle.
- **Recommendation: Option A (200 stocks).** Best balance of statistical power, download reliability, and pedagogical tractability.
- **Risk: Large-cap-only universe kills the size effect (SMB).** All S&P 500 stocks have market caps >$10B. The "small" bucket in our 2x3 double sorts contains stocks with market caps of $150B-$340B (verified via code probe). Self-constructed SMB from this universe correlates only ~0.36 with official Ken French SMB (verified via code probe). HML is similarly degraded at ~0.38 correlation. This is not a sample-size problem — going to 500 stocks does not fix it because the entire S&P 500 is large-cap. The fundamental issue is universe coverage: Ken French uses all NYSE/AMEX/NASDAQ (~4,000-6,000 stocks including genuine small-caps with market caps of $100M-$2B).

#### Date Range

- **Option A: 2014-01-01 to 2024-12-31 (11 years)** — Covers a full business cycle (2014-2019 bull, COVID crash March 2020, recovery, 2022 rate hiking). Provides ~2,760 trading days and ~131 monthly observations. Enough for time-series factor regressions with 5 regressors.
- **Option B: 2019-01-01 to 2024-12-31 (6 years)** — Better alignment with yfinance fundamental data window (~2021-2025 balance sheet data). Only ~72 monthly observations — marginal for time-series regressions with 5+ factors and problematic for Fama-MacBeth (the time-series average of cross-sectional slopes needs T > 60 for reliable Newey-West standard errors).
- **Option C: 2010-01-01 to 2024-12-31 (15 years)** — More data for pure price-based analyses, but widens the gap between the price window and the fundamental window even further.
- **Recommendation: Option A (2014-2024).** 131 monthly observations provides adequate power for all time-series and cross-sectional analyses. The fundamental data gap (see Risk) is managed by scoping which sections use which window.
- **Risk: Fundamental data window mismatch.** yfinance returns only 4-5 annual balance sheet periods per ticker, covering approximately 2021-2025 (verified via code probe: AAPL has 2021-09 to 2025-09, JPM has 2022-12 to 2025-12). This means:
  - **Sections using only price data + official factors (S1, S2, S4, S5-time-series-step):** Can use the full 2014-2024 window with no constraint.
  - **Sections requiring fundamentals for factor construction or feature computation (S3, S6, S7, Ex1, Ex3, Ex4, D1, D2, D3):** Fundamental-derived characteristics (book-to-market, profitability, investment, P/E, ROE, etc.) can only be computed for the ~2021-2025 period. Factor construction via double sorts on fundamentals produces returns for ~3-4 years of monthly data (~36-48 monthly factor returns).
  - **Implication for Fama-MacBeth (Ex2, D3):** The cross-sectional step requires characteristics for each stock in each period. If characteristics come from fundamentals, the cross-sectional regressions are limited to the fundamental window. However, for a standard Fama-MacBeth test, the first step (time-series beta estimation) uses the full price window, and the second step (cross-sectional regression of returns on betas) also uses the full window since the betas are estimated once. The constraint only binds when characteristics themselves (not betas) are used as regressors — which is the modern Fama-MacBeth approach. For sections that use characteristics directly, the effective window is ~3-4 years.

### Dataset 2: Fundamental Data

**Source:** yfinance `Ticker.balance_sheet`, `Ticker.income_stmt`, `Ticker.quarterly_balance_sheet`, `Ticker.info` (for sector, current market cap, shares outstanding)

**Fields required per ticker:**
- Balance sheet: `Stockholders Equity`, `Total Assets`, `Ordinary Shares Number` — 100% availability across 30 tested tickers (verified via code probe)
- Income statement: `Operating Income` — available for ~96% of tickers; missing for banks/financials (JPM, GS, etc.) which use different income structure (verified via code probe). Fallback: use `Net Income` or `Pretax Income` for profitability.
- Market cap: historical approximation via `Ordinary Shares Number` from balance sheet x price at that date. Current market cap available via `Ticker.info['marketCap']`.
- Sector: available via `Ticker.info['sector']` — 100% success across 50 tested tickers (verified via code probe). 11 GICS sectors represented.

**Coverage:** 4-5 annual periods per ticker, earliest ~2021, latest ~2025 (verified via code probe). Quarterly data: 5-7 quarters available.

**Download time:** ~0.4s per ticker for fundamentals (verified via code probe). ~0.22s per ticker for `.info` (verified via code probe). Estimated total for 200 tickers: ~80s fundamentals + ~44s info = ~2 minutes.

- **Risk: Banks and financials have different income statement structure.** ~4% of tickers lack `Operating Income`. The fallback (`Net Income`) is adequate for profitability ratios but introduces inconsistency. This affects RMW factor construction (S3, D1) and profitability features (S7, D2).

### Dataset 3: Official Factor Returns (benchmark data)

**Source:** `getfactormodels` library (verified via code probe)

**Available models:**

| Model | API Call | Columns | Date Range |
|-------|----------|---------|------------|
| FF3 | `FamaFrenchFactors(model='3', frequency='M')` | `date, Mkt-RF, SMB, HML, RF` | 1963-07 to 2025-12 |
| FF5 | `FamaFrenchFactors(model='5', frequency='M')` | `+ RMW, CMA` | 1963-07 to 2025-12 |
| FF6 | `FamaFrenchFactors(model='6', frequency='M')` | `+ UMD` | 1963-07 to 2025-12 |
| Carhart | `CarhartFactors(frequency='M')` | `date, Mkt-RF, SMB, HML, MOM, RF` | 1926-07 to 2025-12 |

**Format:** Returns PyArrow Table; convert via `.to_pandas()`. Returns in decimal form (e.g., 0.0196 = 1.96%).

**Download time:** <2s.

**Risk-free rate:** `RF` column = 1-month T-bill rate (monthly), from Ken French library.

- **Risk: None material.** This is the most reliable data source in the week. Minor concern: `getfactormodels` is a wrapper around Ken French's data library — if the upstream data is temporarily unavailable, the download fails. Low-probability and retriable.

### Dataset 4: Risk-Free Rate

Included in Dataset 3 as the `RF` column. No separate download needed.

---

### Interaction Analysis

**Interaction 1: Large-cap universe x fundamental-based factor construction = structurally weak self-built factors.**

This is the most important interaction in the data plan. The S&P 500 universe means our "small" stocks are $150B+ megacaps. The fundamental data window means we can only construct factor returns for ~3-4 years. These compound: not only will our SMB and HML have low correlation with official factors (~0.36), but the short fundamental window means we have very few monthly observations of these poorly-replicated factors. For Exercise 1 ("Can You Replicate Fama-French?"), this means the student sees a low correlation number, but from only ~36-48 data points, making it hard to tell whether the divergence is structural (universe) or noise (short window).

However, this interaction is pedagogically productive if framed correctly. The student *should* discover that free data + large-cap universe cannot replicate Ken French factors well, and the *reasons why* (universe coverage, survivorship bias, data depth) are precisely the lesson. The risk is that with only ~36-48 months of self-built factor returns, the noise dominates the signal, and the student concludes "factor construction doesn't work" rather than "universe matters."

**Mitigation:** For factor construction exercises that compare self-built to official, use the *full price window* (2014-2024) for price-based factors (momentum) and restrict fundamental-based factors (HML, RMW, CMA) to the fundamental window. Price-based factor construction (momentum) will have 131 monthly observations and should show reasonable correlation with Ken French MOM. This provides a calibration anchor: "momentum replication works well (price-only, long window); value replication doesn't (fundamental-dependent, short window, universe mismatch)."

**Interaction 2: Fundamental download (200 tickers x sequential API calls) x sector info download (200 tickers x `.info` calls) = ~2-3 minutes of API calls with rate-limit exposure.**

The combined fundamental + info download for 200 tickers takes ~2-3 minutes. This is manageable in a single run, but if yfinance rate-limits mid-download, partial data results. The code agent needs retry logic with exponential backoff. At 50 tickers (verified probe), there were zero failures. But 200 tickers quadruples the exposure window to rate limiting.

**Mitigation:** The `data_setup.py` script should implement: (1) batch price download first (single call, ~15s, most reliable), (2) sequential fundamental downloads with per-ticker try/except and retry, (3) `.info` calls bundled with fundamentals (same ticker object), (4) caching to disk so partial runs can resume.

**Interaction 3: Full price window (2014-2024) used in some sections x fundamental window (~2021-2025) used in others = potential student confusion about inconsistent date ranges.**

Sections 1-2, 4 use 11 years of monthly data; Sections 3, 6, 7 and most exercises use only ~3-4 years of fundamentals. If not clearly documented, students may wonder why time-series R-squareds in Section 1 (131 observations) differ from Section 6 analysis periods (~36-48 observations). The data allocation table below is explicit about which window each section uses.

**Interaction 4: 200 stocks x monthly returns x 5-7 characteristics = panel data for Fama-MacBeth that must be properly indexed.**

The `linearmodels.FamaMacBeth` class requires a MultiIndex panel DataFrame (verified: `fit()` accepts `cov_type`, `bandwidth`, `kernel`). With 200 stocks x 131 months, the panel has ~26,200 rows. This is a manageable size, but missing fundamentals for earlier periods mean the panel is unbalanced (stocks enter the panel only when they have characteristic data). An unbalanced panel is acceptable for Fama-MacBeth (each cross-section is independent), but the code agent needs to handle NaN characteristics cleanly.

---

### Failure Modes & Contingencies

| Risk | Likelihood | Affected Sections | Contingency |
|------|-----------|-------------------|-------------|
| Self-built SMB has very low correlation (~0.3-0.4) with Ken French SMB due to large-cap-only universe | High (verified via probe: r=0.36) | S3, Ex1, D1 | Accept as a pedagogical feature, not a bug. Set acceptance criteria around *correlation with official*, not *correlation > 0.9*. Criteria: SMB correlation [0.15, 0.55], HML correlation [0.20, 0.60]. The divergence IS the lesson. Use momentum as the calibration anchor (price-only, expected r > 0.60). If correlation is <0.15, flag as potential code bug. |
| Self-built HML also has low correlation (~0.3-0.5) with Ken French HML | High (verified via probe: r=0.38) | S3, Ex1, D1 | Same mitigation as SMB. HML may perform slightly better because value dispersion exists within large-caps. Accept correlation range [0.20, 0.60]. |
| yfinance fundamental data covers only ~4-5 years (~2021-2025), not the full 2014-2024 price window | High (verified via probe) | S3, S6, S7, Ex1, Ex3, Ex4, D1, D2, D3 | Split analyses into two windows: price-only analyses use full 2014-2024; fundamental-dependent analyses use ~2021-2024 or most recent annual data. For Fama-MacBeth with characteristics, effective T is ~36-48 months. Widen t-statistic acceptance ranges accordingly. |
| yfinance rate-limits or throttles during 200-ticker fundamental download | Medium | All sections using fundamentals | Implement per-ticker retry with exponential backoff (3 retries, 2s/4s/8s delays). Cache to Parquet. Resume from cache on re-run. If >20% of tickers fail after retries, fall back to 150-ticker universe. |
| Banks/financials missing `Operating Income` (~4% of tickers) | High (verified) | S3 (RMW), S7 (profitability features), D1, D2 | Use `Net Income` as fallback for profitability. Flag tickers in quality report. ~4% inconsistency does not materially affect cross-sectional results with 200 stocks. |
| `getfactormodels` upstream data temporarily unavailable | Low | S2, S4, Ex1, D1, D3 | Cache official factor data to Parquet on first download. Load from cache if API fails. Data changes only monthly. |
| yfinance `.info` returns incomplete sector data for some tickers | Low (0/50 failures in probe) | S6, Ex3, D2 | If sector missing for <5% of tickers, assign to "Other." If >5% missing, use a static sector mapping file. |
| Panel data for Fama-MacBeth is unbalanced (stocks enter when fundamental data becomes available) | High (structural) | Ex2, Ex4, D3 | Acceptable for Fama-MacBeth (each cross-section is independent). Drop NaN rows per cross-section, not impute. Document effective sample size per month in run_log. |
| Fama-MacBeth gamma estimates noisy with only ~36-48 monthly cross-sections for fundamental-based analyses | Medium-High | Ex2, D3 | Widen acceptance criteria for t-statistics. With T~40, weak factors (size, value) may not reach significance at 5% level even if the premium exists. Accept t-statistic ranges that include insignificance for weak factors. |

---

### Statistical Implications

- **Price-only analyses (S1, S2, S4):** 200 stocks x 131 monthly observations = 26,200 stock-months for time-series regressions. Individual stock regressions have 131 observations each — adequate for CAPM (2 parameters), FF3 (4 parameters), and FF5 (6 parameters). Portfolio-level R-squared will be high (>0.90 for diversified portfolios, verified via probe).

- **Individual stock time-series R-squared (verified via code probe on 30 stocks, 2014-2024 monthly):** CAPM R-squared ranges 0.10-0.56 (median 0.34), FF3 ranges 0.18-0.73 (median 0.40), FF5 ranges 0.21-0.77 (median 0.47). Median R-squared improvement CAPM-to-FF3: +0.07, FF3-to-FF5: +0.03.

- **Factor construction (S3, D1):** 200 stocks sorted into 2x3 = 6 portfolios gives ~33 stocks per bucket. Value-weighted returns are dominated by the 5-10 largest stocks in each bucket. With ~36-48 monthly factor returns, correlations with official factors have wide confidence intervals. The 0.36 SMB correlation (probe) has a 95% CI of roughly [0.08, 0.59] with 60 observations.

- **Fama-MacBeth cross-sectional regressions (Ex2, D3):** 200 stocks per cross-section provides adequate degrees of freedom for 5-7 regressors (~28-40 observations per regressor). The time-series of cross-sectional slopes (gammas) has T=131 for price-based characteristics and T~36-48 for fundamental-based characteristics. Standard errors of the time-series mean scale as sigma/sqrt(T), so the shorter window inflates standard errors by a factor of ~sqrt(131/40) = ~1.8.

- **Multiple testing (Ex4):** Testing 8-10 characteristics via Fama-MacBeth, then applying Bonferroni correction with 10 tests raises the effective significance threshold from p=0.05 to p=0.005 (t > ~2.8). With 200 stocks per cross-section, well-known factors (momentum, market beta) should still survive. Noise factors should not.

- **Barra-style cross-sectional regression (S6, Ex3):** Monthly cross-sectional regressions on 200 stocks with ~15 regressors (5-7 style factors + ~10 sector dummies) gives a monthly cross-section with 200 observations and ~15 parameters — adequate (13 observations per parameter).

---

## Known Constraints

- **Survivorship bias:** yfinance returns only current listings. Our "historical S&P 500" is actually today's S&P 500, which inflates historical backtest returns by an estimated 1-3% annually (research_notes.md). Stocks that were delisted, went bankrupt, or were removed from the index are absent from our universe. This affects all sections but is most impactful for factor construction (S3, D1) and long-run return analyses (S1, S2, S4). The survivorship bias IS a teaching point — it partially explains divergence between self-built and official factors.

- **Universe bias (large-cap only):** S&P 500 "small-caps" have market caps >$10B. True small-cap effects (the SMB factor) require the full NYSE/AMEX/NASDAQ universe, which includes stocks with market caps of $100M-$2B. Self-constructed SMB will reflect relative size differences within large-caps (mega-cap vs. large-cap), not the true small-cap premium. This is a structural limitation, not a data quality issue, and cannot be solved by adding more S&P 500 tickers (research_notes.md). Affects S3, Ex1, D1.

- **Point-in-time violation:** yfinance fundamental data is "as-reported" but not point-in-time — we cannot verify exact reporting dates, introducing potential look-ahead bias. In production (CRSP/Compustat), point-in-time databases track when each data point was actually available to investors. Our factor construction may use fundamentals before they were publicly reported. This affects S3, S6, S7, Ex1, D1, D2 (research_notes.md).

- **Fundamental data depth:** yfinance provides only ~4-5 annual periods of balance sheet and income statement data per ticker (~2021-2025), not the full historical range needed for deep backtests (research_notes.md, verified via code probe). This constrains all fundamental-dependent analyses to a ~3-4 year effective window.

- **Financial sector accounting differences:** Banks and financial companies (JPM, GS, AXP, etc.) use different income statement structure and do not report `Operating Income` via yfinance (verified via code probe). This affects ~4% of the 200-ticker universe and requires a `Net Income` fallback for profitability ratios, introducing minor cross-sectional inconsistency in RMW construction and profitability features.

- **No delisted stock data:** yfinance cannot retrieve data for delisted tickers. Factor construction implicitly conditions on survival, which biases value and small-cap factor returns upward (the worst performers are removed). This compounds with survivorship bias above.

- **`getfactormodels` returns PyArrow format:** Factor data from `getfactormodels` returns as PyArrow Tables, not pandas DataFrames. Code must call `.to_pandas()` before use (verified via code probe).

---

## Data Allocation Summary

| Section / Exercise | Data Source | Universe Subset | Date Window | Specifics |
|--------------------|------------|-----------------|-------------|-----------|
| S1: CAPM | yfinance + getfactormodels | 15-20 diverse tickers | 2014-2024 (price window) | Monthly returns, FF3 factors for Mkt-RF and RF |
| S2: FF3 Model | getfactormodels + yfinance | 10-15 tickers + official FF3 | 2014-2024 (price window) | Monthly returns, official FF3 factors (1963-2025 for cumulative plots) |
| S3: Factor Construction | yfinance | Full 200-ticker universe | Fundamental window (~2021-2024) | Annual balance sheet + income stmt + monthly prices |
| S4: FF5 + Momentum | getfactormodels + yfinance | 10-15 tickers + official FF5/FF6 | 2014-2024 (price window) | Monthly returns, official FF5+MOM factors |
| S5: Fama-MacBeth | yfinance + getfactormodels | Full 200-ticker universe | 2014-2024 (betas); fundamental window (characteristics) | Monthly returns, FF3 betas estimated over full window |
| S6: Barra Risk Models | yfinance | Full 200-ticker universe | Fundamental window (~2021-2024) | Monthly returns + characteristics + sector dummies |
| S7: Feature Engineering | yfinance | Full 200-ticker universe | Fundamental window (~2021-2024) | Balance sheet, income stmt, prices for ratio computation |
| Ex1: Replicate FF | yfinance + getfactormodels | Full 200-ticker universe | Fundamental window (HML/SMB); 2014-2024 (momentum) | Fundamentals for sorts, official factors for comparison |
| Ex2: Factor Premia | yfinance + getfactormodels | Full 200-ticker universe | 2014-2024 (betas); fundamental window (characteristics) | Monthly panel, Fama-MacBeth cross-sections |
| Ex3: Portfolio Risk | yfinance | 20-stock portfolio subset + full universe for regression | Fundamental window (~2021-2024) | Monthly returns + characteristics + sector |
| Ex4: Factor Zoo | yfinance + getfactormodels | Full 200-ticker universe | Fundamental window (~2021-2024) | 8-10 characteristics, monthly Fama-MacBeth |
| D1: Factor Factory | yfinance + getfactormodels | Full 200-ticker universe | Fundamental window (fundamental factors); 2014-2024 (momentum) | All 6 factors, validation against official |
| D2: Feature Matrix | yfinance | Full 200-ticker universe | Fundamental window (~2021-2024) | Fundamentals + prices for all features |
| D3: Horse Race | yfinance + getfactormodels | Full 200-ticker universe | 2014-2024 (betas); fundamental window (characteristics) | Fama-MacBeth for CAPM/FF3/FF5, Newey-West |

---

## Lecture Sections

### Section 1: The One-Factor World — `s1_capm.py`

**Data:** 15-20 diverse stocks spanning market caps and sectors (e.g., AAPL, NVDA, JPM, XOM, JNJ, KO, TSLA, PG, GS, AMD, NEE, PFE, CAT, COST, NFLX, WMT, BA, MO, LLY, T). Monthly adjusted close prices from yfinance, 2014-01-01 to 2024-12-31. Official FF3 factors via `FamaFrenchFactors(model='3', frequency='M')` for `Mkt-RF` and `RF`. Compute monthly excess returns as stock return minus RF.

**Acceptance criteria:**
- Estimated betas range from ~0.3 to ~2.1 across the 15-20 stock sample (verified via code probe: beta range 0.38-2.01 for 30 S&P 500 stocks). Defensive stocks (KO, JNJ, PG) should have beta < 0.8; high-growth/cyclical stocks (NVDA, TSLA, AMD) should have beta > 1.3.
- Individual stock CAPM time-series R-squared ranges from ~0.05 to ~0.60 (verified via code probe: 0.10-0.56, median 0.34). Median R-squared across the sample should be in [0.20, 0.45].
- Jensen's alpha for the sample: most stocks show alpha != 0 at 5% significance (CAPM fails to fully explain individual stock returns). Annualized alphas should range from approximately -10% to +25% across the sample, reflecting CAPM's poor cross-sectional fit for individual stocks.
- Security market line (SML) cross-sectional regression (average excess return on estimated beta): R-squared should be very low, in [0.00, 0.20]. The slope should be positive but much flatter than the theoretical prediction (theoretical slope = average market excess return ~7-10% annualized; empirical slope typically 0-5%). SML R-squared with only 15-20 stocks is very noisy — the key visual is the scatter's poor fit, not the exact R-squared.
- The SML plot should visually show that high-beta stocks do not earn proportionally higher returns — this is the "beta is flat" empirical failure of CAPM.

**Production benchmark:** Fama & French (1992), using CRSP data 1963-1990, found that beta has essentially no explanatory power for the cross-section of average returns once size is controlled for. The SML is "too flat" — a finding replicated across decades and datasets. With institutional data (CRSP, full NYSE/AMEX/NASDAQ), cross-sectional R-squared of average return on beta is typically <0.05 for individual stocks. Our S&P 500 sample may show slightly higher R-squared due to the narrower universe, but the flat-SML pattern should hold.

### Section 2: When One Factor Fails — The Fama-French Three-Factor Model — `s2_ff3_model.py`

**Data:** 10-15 stocks from Section 1 sample (reuse downloaded data). Official FF3 monthly factors via `FamaFrenchFactors(model='3', frequency='M')` — `Mkt-RF`, `SMB`, `HML`, `RF`. Full factor history (1963-2025) for cumulative return plots. Stock excess returns computed from 2014-2024 monthly data.

**Acceptance criteria:**
- FF3 time-series R-squared for individual stocks: range [0.15, 0.75], median in [0.30, 0.50] (verified via code probe: median 0.40 for 30 stocks). This should be a visible improvement over CAPM R-squared (median ~0.34).
- Median R-squared improvement from CAPM to FF3: in [0.03, 0.15] (verified via code probe: median improvement +0.07). The improvement should be consistent across most stocks (positive for >70% of the sample).
- FF3 alpha (Jensen's alpha under the three-factor model): should be smaller in absolute value than CAPM alpha for the majority of stocks (>60%). This demonstrates "alpha shrinkage" — what looked like stock-picking skill under CAPM is partially explained by size and value exposure.
- Cumulative factor return plots (1963-2025): Mkt-RF should show strong long-run positive growth with drawdowns (including 2008-2009, 2020). SMB should show positive long-run returns but with periods of underperformance (especially post-2000). HML should show positive long-run returns but with notable post-2007 weakness ("death of value"). The sign of cumulative HML return over 2014-2024 is uncertain — it may be negative for this recent period, which is expected and pedagogically valuable.
- Factor loadings: stocks known for small-cap characteristics within S&P 500 (lower market cap tickers) should have positive SMB loadings; value stocks (banks, energy) should have positive HML loadings; growth stocks (tech) should have negative HML loadings. Loadings should be statistically significant for at least some stocks.

**Production benchmark:** Fama & French (1993), using CRSP data 1963-1991, report that the three-factor model captures most of the cross-sectional variation in average returns. Typical individual stock FF3 R-squared with CRSP monthly data is 0.20-0.60 (Davis, Fama & French, 2000). Our S&P 500 sample should fall within this range. Tidy Finance (Scheuch, Voigt & Weiss, 2023), using CRSP/Compustat, achieves SMB replication R-squared of ~0.99 and HML replication R-squared of ~0.96 when using the same universe as Ken French. Our results will be substantially lower due to the S&P 500 universe constraint.

### Section 3: Building Factors from Scratch — Portfolio Sorts — `s3_factor_construction.py`

**Data:** Full 200-ticker universe. Annual balance sheet data from yfinance for `Stockholders Equity`, `Total Assets`, `Ordinary Shares Number`. Annual income statement data for `Operating Income` (with `Net Income` fallback for financials). Monthly adjusted close prices for return computation. Fundamental data covers ~2021-2025 annual periods (verified via code probe: 4-5 periods per ticker). Monthly price data covers 2014-2024 but factor returns from fundamental-based sorts are limited to the fundamental window. Official FF3 factors via `getfactormodels` for validation comparison.

**Acceptance criteria:**
- Book-to-market (B/M) ratios computed for >180 of 200 tickers (missing data acceptable for <10%). B/M values should range from ~0.01 (high-growth tech) to ~0.80 (value/financial stocks), with median in [0.10, 0.40] for an S&P 500 universe.
- Double-sort produces 6 non-empty portfolios (2 size x 3 B/M). Each portfolio should contain at least 15 stocks. If any portfolio has <10 stocks, the breakpoints are miscalibrated.
- Self-constructed SMB monthly returns: mean in [-0.02, +0.01], standard deviation in [0.01, 0.05]. Correlation with official Ken French SMB: [0.15, 0.55] (verified via code probe: r=0.36). The low correlation is expected and structural (see Known Constraints: universe bias).
- Self-constructed HML monthly returns: mean in [-0.02, +0.02], standard deviation in [0.01, 0.06]. Correlation with official Ken French HML: [0.20, 0.60] (verified via code probe: r=0.38).
- Annualized tracking error (self-built vs. official) for both SMB and HML: in [5%, 20%] (verified via code probe: SMB tracking error ~10.6%, HML ~15.1%).
- The demonstration should make explicit that the low correlations are due to the S&P 500 large-cap universe, not a code error. A side-by-side comparison of the "small" bucket market caps ($150B+) vs. true Ken French small-caps ($100M-$2B) should make the cause clear.

**Production benchmark:** Tidy Finance (Scheuch, Voigt & Weiss, 2023), using CRSP/Compustat with the full NYSE/AMEX/NASDAQ universe, achieves SMB replication R-squared ~0.99 and HML replication R-squared ~0.96 vs. Ken French official factors. Our S&P 500-only replication is expected to achieve R-squared (= correlation squared) of ~0.10-0.30 for SMB and ~0.05-0.35 for HML. The gap demonstrates the value of institutional data.

### Section 4: The Five-Factor Model and the Momentum Orphan — `s4_ff5_momentum.py`

**Data:** 10-15 stocks from Section 1 sample. Official FF5 factors via `FamaFrenchFactors(model='5', frequency='M')` — `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF`. Official momentum via `FamaFrenchFactors(model='6', frequency='M')` — `UMD` column. Full factor history (1963-2025) for cumulative return plots. Stock monthly returns 2014-2024 for regressions.

**Acceptance criteria:**
- FF5 time-series R-squared for individual stocks: range [0.15, 0.80], median in [0.35, 0.55] (verified via code probe: median 0.47). Should be higher than FF3 median for the majority of stocks (>60%).
- Median R-squared improvement from FF3 to FF5: in [0.01, 0.08] (verified via code probe: median improvement +0.03). Smaller than the CAPM-to-FF3 improvement — diminishing returns of adding factors.
- Momentum (UMD) cumulative return plot (1963-2025): should show strong long-run positive performance with at least one catastrophic drawdown visible (2009 momentum crash: ~-40% to -60% in a few months).
- Alpha under FF5 should be smaller than alpha under FF3 for the majority of stocks (>55%). The marginal alpha reduction from FF3 to FF5 should be smaller than from CAPM to FF3.
- Factor correlation matrix: RMW and CMA should show low-to-moderate correlation with Mkt-RF, SMB, HML (|r| < 0.40 for most pairs). UMD should show near-zero or slightly negative correlation with HML (the value-momentum negative correlation is well-documented).

**Production benchmark:** Fama & French (2015), using CRSP data 1963-2013, report that FF5 substantially improves on FF3 for explaining the cross-section of returns. The incremental R-squared from adding RMW and CMA varies by portfolio but is typically 0.01-0.10 for individual stocks. Momentum (UMD) has a long-run Sharpe ratio of ~0.5-0.7 annualized using Ken French data (1927-2024), making it one of the strongest anomalies, but with extreme left-tail risk.

### Section 5: Are Factors Priced? — The Fama-MacBeth Methodology — `s5_fama_macbeth.py`

**Data:** Full 200-ticker universe. Monthly returns 2014-2024. FF3 factor data from `getfactormodels` for the time-series beta estimation step. For the manual Fama-MacBeth demonstration: first-pass time-series regression of each stock's excess returns on Mkt-RF, SMB, HML to estimate betas (full 2014-2024 window, 131 months). Second-pass cross-sectional regression of monthly excess returns on estimated betas (each of 131 months). Time-series average of cross-sectional slopes = risk premium estimates. For the `linearmodels.FamaMacBeth` production version: panel DataFrame with MultiIndex (ticker, date), dependent variable = excess returns, independent variables = estimated betas. `fit(cov_type='kernel')` for Newey-West standard errors.

**Acceptance criteria:**
- Manual Fama-MacBeth and `linearmodels.FamaMacBeth` should produce risk premium estimates (gamma coefficients) that agree within +-0.001 monthly. If they disagree by more, either the manual implementation or the panel setup has a bug.
- Market risk premium (gamma_MKT): monthly estimate in [-0.005, +0.015]. The cross-sectional market risk premium is notoriously flat or even negative ("beta anomaly"). A negative or insignificant gamma_MKT is expected and is a key teaching point. Sensitivity note: the sign and magnitude of gamma_MKT depend heavily on the sample period and universe.
- SMB risk premium (gamma_SMB): monthly estimate in [-0.010, +0.005]. Likely insignificant with Newey-West t-statistic |t| < 2.0 in recent samples. The size premium has weakened substantially since the 1990s.
- HML risk premium (gamma_HML): monthly estimate in [-0.010, +0.010]. Direction uncertain in post-2007 samples ("death of value"). Sensitivity note: HML premium is positive long-run (Fama & French 1993 report ~0.40% monthly using 1963-1991 CRSP data), but may be negative in 2014-2024.
- Newey-West t-statistics: should differ from OLS t-statistics for at least some factors, demonstrating the serial correlation correction. Newey-West t-statistics are typically 10-30% smaller in absolute value than naive OLS t-statistics.
- Cross-sectional R-squared (average across months): in [0.01, 0.15] for the three-factor specification. Cross-sectional R-squared for individual-stock Fama-MacBeth is typically very low because idiosyncratic returns dominate.

**Production benchmark:** Fama & MacBeth (1973), using NYSE data 1926-1968, found a positive and significant market risk premium and a positive relationship between beta and average return. However, later studies (Fama & French, 1992, using CRSP 1963-1990) showed the cross-sectional relationship between beta and average return is flat once size is controlled. The "beta anomaly" is now well-established. Typical Fama-MacBeth market risk premium estimates with CRSP monthly data range from -0.2% to +0.8% monthly depending on the sample period.

### Section 6: The Practitioner's Lens — Barra-Style Risk Models — `s6_barra_risk_models.py`

**Data:** Full 200-ticker universe. Monthly returns over the fundamental window (~2021-2024, ~36-48 months). Cross-sectional characteristics computed from most recent yfinance fundamentals: log market cap (size), book-to-market (value), 12-1 month momentum, trailing volatility (60-day), ROE or operating profitability. Sector dummies from `yf.Ticker().info['sector']` (11 GICS sectors, verified via code probe: 100% availability). All characteristics cross-sectionally standardized (z-score or rank-transform) each month.

**Acceptance criteria:**
- Monthly cross-sectional regression runs for each month in the sample without errors. Each regression has ~200 observations and ~15-17 regressors (5-7 style factors + ~10 sector dummies + intercept). Degrees of freedom: ~183-185.
- Cross-sectional R-squared per month: range [0.05, 0.50], median in [0.10, 0.30]. Cross-sectional R-squared in Barra-style regressions is typically higher than in Fama-MacBeth because characteristics (not estimated betas) are used directly and industry dummies absorb sector-level variation.
- Factor returns (regression coefficients) for market-cap (size) factor: monthly mean in [-0.005, +0.005] with standard deviation in [0.005, 0.030]. Size factor returns should be noisy and near zero for a large-cap-only universe.
- Factor returns for momentum: monthly mean in [0.000, +0.010]. Momentum typically carries a positive premium even in cross-sectional regressions.
- Risk decomposition for a sample equal-weight portfolio: common factor risk should explain 40-80% of total portfolio return variance. The exact fraction depends on portfolio concentration and sector mix. A diversified 20-stock cross-sector portfolio should show factor risk explaining >50% of variance. A concentrated all-tech portfolio should show higher factor risk (>60%) due to shared sector exposure.
- Specific (idiosyncratic) risk should explain 20-60% of total portfolio return variance, inversely related to diversification.

**Production benchmark:** MSCI Barra USE4 model (MSCI, 2011) uses 55-70 industry classifications and ~10 style factors with daily cross-sectional regressions on ~3,000+ stocks. Typical cross-sectional R-squared in production Barra models is 0.20-0.40 per period. Our simplified version (11 sectors, 5-7 style factors, 200 stocks, monthly) should produce lower R-squared. For risk decomposition, institutional results typically show common factors explaining 30-60% of individual stock variance and 60-90% of diversified portfolio variance (Menchero, 2010, MSCI Research).

### Section 7: From Factors to Features — Cross-Sectional Feature Engineering — `s7_feature_engineering.py`

**Data:** Full 200-ticker universe. yfinance balance sheet (`Stockholders Equity`, `Total Assets`, `Ordinary Shares Number`), income statement (`Operating Income`, `Net Income`, `Total Revenue`), and monthly adjusted close prices. Fundamental data from the ~2021-2025 window. Prices from 2013-2024 (extra year for 12-month momentum lookback). Features to compute: P/E (price/earnings), P/B (price/book = 1/B-M), ROE (net income / equity), asset growth (YoY total assets change), earnings yield (earnings/price), momentum (12-1 month return), short-term reversal (1-month return), trailing volatility (60-day daily return std).

**Acceptance criteria:**
- Feature matrix shape: (N_months x N_stocks) panel with ~8-12 feature columns. For the fundamental window (~36-48 months x 200 stocks), expect ~7,200-9,600 rows before dropping NaN.
- Missing data rate per feature: fundamental features (P/E, P/B, ROE, asset growth, earnings yield) should have 0-15% missing across the panel. Price-based features (momentum, reversal, volatility) should have <5% missing.
- After winsorizing at [1st, 99th] percentiles: no feature should have values beyond the winsorization bounds. Feature distributions should be less extreme than pre-winsorized versions (max |z-score| should drop from potentially >10 to <3 after winsorizing + standardization).
- Cross-sectional z-scores: within each month, each feature should have mean ~0 and std ~1 (by construction). Verify round-trip: mean in [-0.05, +0.05], std in [0.90, 1.10] after standardization.
- Rank-transformed features: within each month, rank-transformed features should have a uniform distribution. Range should be [0, 1] or [0, N-1]. Spearman correlation between z-score and rank-transform of the same feature should be >0.95 (both capture the same cross-sectional ordering).
- Feature correlation matrix: momentum and reversal should have negative correlation (r in [-0.40, -0.05]). P/E and earnings yield should have strong negative correlation (r < -0.70, since earnings yield ~ 1/P*E). Size (log market cap) and P/B may show moderate correlation.
- The feature matrix should be output as a panel DataFrame with MultiIndex (date, ticker) suitable for direct consumption by Week 4 ML models.

**Production benchmark:** Gu, Kelly & Xiu (2020), using CRSP/Compustat, construct 94 firm characteristics as features for ML-based cross-sectional prediction. Their feature matrix covers ~30,000 stock-months per year over 1957-2016. Our matrix is much smaller (200 stocks, ~3-4 years, 8-12 features) but follows the same cross-sectional standardization methodology. The key benchmark is methodological correctness (proper cross-sectional standardization, no look-ahead bias in feature computation), not scale.

---

## Seminar Exercises

### Exercise 1: Can You Replicate Fama-French?

**Data:** Full 200-ticker universe. Monthly returns from yfinance. Annual fundamentals (`Stockholders Equity`, `Ordinary Shares Number`) for book-to-market computation. Monthly prices for momentum computation. Official Ken French factors via `getfactormodels` — `FamaFrenchFactors(model='3', frequency='M')` for SMB and HML benchmark; `CarhartFactors(frequency='M')` for MOM benchmark. Fundamental-based factors (SMB, HML) use the fundamental window (~2021-2024 annual data). Price-based factor (momentum) uses the full price window (2014-2024).

**Acceptance criteria:**
- Self-constructed SMB correlation with Ken French SMB: [0.15, 0.55] (verified via code probe: r=0.36). If r < 0.15, likely a code bug (wrong sort methodology, inverted sign, etc.).
- Self-constructed HML correlation with Ken French HML: [0.20, 0.60] (verified via code probe: r=0.38). If r < 0.15, likely a code bug.
- Self-constructed momentum correlation with Ken French MOM/UMD: [0.50, 0.85]. Momentum is price-only with no fundamental data dependency and should replicate substantially better than SMB or HML. This serves as the calibration anchor — if momentum correlation is also low (<0.40), the issue is likely in return computation or portfolio weighting, not the universe.
- The student discovers that SMB/HML correlations are much lower than momentum correlation, and can articulate at least two reasons: (1) universe coverage (S&P 500 = large-cap only vs. Ken French's full NYSE/AMEX/NASDAQ), (2) survivorship bias (our universe excludes delisted stocks).
- Tracking error (annualized, self-built minus official): SMB in [5%, 25%], HML in [5%, 25%], momentum in [2%, 15%].

**Production benchmark:** Tidy Finance (Scheuch, Voigt & Weiss, 2023), using CRSP/Compustat with the full universe, achieves SMB replication R-squared ~0.99 and HML R-squared ~0.96. Our R-squared (correlation squared) targets: SMB ~0.02-0.30, HML ~0.04-0.36, momentum ~0.25-0.72. The gap between our results and Tidy Finance's results IS the insight.

### Exercise 2: Which Factors Carry a Risk Premium?

**Data:** Full 200-ticker universe. Monthly excess returns 2014-2024. Five characteristics per stock per month: (1) market beta (estimated from rolling 60-month window of excess returns on Mkt-RF, using official factor data), (2) log market cap (size — from yfinance `.info['marketCap']` as of most recent available, or price x shares from balance sheet), (3) book-to-market (from fundamentals, fundamental window), (4) operating profitability (operating income / book equity, from fundamentals, fundamental window), (5) 12-1 month momentum (price-based, full window). For the Fama-MacBeth procedure: cross-sectional regression of monthly excess returns on the five characteristics, each month separately. Time-series average of cross-sectional slopes with Newey-West standard errors. Use `linearmodels.FamaMacBeth` with `cov_type='kernel'`.

**Acceptance criteria:**
- Fama-MacBeth produces monthly risk premium estimates (gammas) for all 5 characteristics. The panel should have ~200 stocks x T months (T=131 for price-based characteristics; T~36-48 for months where all 5 characteristics including fundamentals are available).
- Market beta gamma: monthly estimate in [-0.010, +0.015]. Newey-West |t| likely < 2.0 (the "beta anomaly" — flat or negative cross-sectional relationship). This is the key insight for this characteristic.
- Size gamma (log market cap): monthly estimate in [-0.005, +0.005]. Sign ambiguous in recent large-cap samples. Newey-West |t| likely < 2.0.
- Value gamma (book-to-market): monthly estimate in [-0.008, +0.008]. Direction uncertain in post-2007 data. Newey-West |t| in [0.0, 2.5]. Sensitivity note: positive long-run (Fama & French, 1992), but may be insignificant or negative in 2014-2024.
- Profitability gamma: monthly estimate in [-0.005, +0.010]. Tends to be positive and more robust than size/value in recent samples. Newey-West |t| in [0.5, 3.0].
- Momentum gamma: monthly estimate in [0.000, +0.015]. Tends to be positive and significant in most samples. Newey-West |t| in [1.0, 4.0]. Most likely to survive significance testing.
- The student discovers that factor significance varies by period and that the "are factors risk or anomaly?" debate is empirically messy. At least one factor (momentum or profitability) should be significant at 5%. At least one factor (beta or size) should be insignificant.

**Production benchmark:** Fama & French (1992), using CRSP 1963-1990, found size and B/M significant but beta insignificant in cross-sectional regressions. More recent studies (e.g., Hou, Xue & Zhang, 2015, using CRSP 1967-2012) find profitability and investment factors significant while size weakens. Our S&P 500 results will differ from full-universe CRSP results due to the large-cap bias.

### Exercise 3: Decompose Your Portfolio's Risk

**Data:** A portfolio of ~20 stocks selected from the 200-ticker universe (equal-weighted). Two portfolios for comparison: (1) a diversified portfolio spanning sectors (e.g., 2 stocks from each of 10 sectors), (2) a concentrated portfolio (e.g., all tech or all financials). Full 200-ticker universe used for the cross-sectional Barra-style regression that produces factor returns. Monthly returns over the fundamental window (~2021-2024). Cross-sectional characteristics: log market cap, book-to-market, momentum, volatility, sector dummies. All standardized cross-sectionally per month.

**Acceptance criteria:**
- Cross-sectional regressions run for each month without errors. Each has ~200 observations and ~15 regressors.
- For the diversified portfolio: common factor risk explains 40-75% of total portfolio return variance. Specific risk explains 25-60%. Factor risk share should be >40%.
- For the concentrated (single-sector) portfolio: common factor risk explains 55-90% of total return variance. Factor risk share should be higher than the diversified portfolio by at least 10 percentage points. The dominant factor exposure should be the sector dummy corresponding to the concentrated sector.
- The difference in factor risk share between diversified and concentrated portfolios should be visible and in [5, 40] percentage points. If the difference is <5 pp, the two portfolios are not sufficiently different in concentration.
- Risk decomposition components should sum to approximately 100% of total variance (within rounding tolerance of +-5 pp due to cross-terms).

**Production benchmark:** MSCI Barra research (Menchero, 2010) reports that for diversified institutional portfolios, common factors typically explain 60-90% of portfolio variance, with specific risk contributing 10-40%. For individual stocks, common factors explain 30-60% of variance. Our simplified model (fewer factors, fewer industry classifications) should show lower factor explanatory power.

### Exercise 4: The Factor Zoo Safari

**Data:** Full 200-ticker universe. Monthly returns over the fundamental window (~2021-2024). 8-10 candidate characteristics: (1) momentum (12-1 month), (2) book-to-market, (3) operating profitability, (4) asset growth, (5) short-term reversal (1-month return), (6) earnings yield, (7-10) 2-4 noise characteristics constructed from random permutations of real characteristics or random numbers (to serve as known-false signals). All characteristics cross-sectionally standardized. Fama-MacBeth regression for each characteristic individually (univariate), then joint. Apply Bonferroni and Benjamini-Hochberg corrections.

**Acceptance criteria:**
- Without multiple testing correction: at least 3-5 of 8-10 characteristics appear significant at p < 0.05 (naive t > 2.0). The noise characteristics may occasionally appear significant by chance (~5% false positive rate per test, so with 3-4 noise characteristics, 0-1 false positives expected).
- After Bonferroni correction (p < 0.05/10 = 0.005, equivalent to |t| > ~2.8): only 1-3 characteristics survive. The noise characteristics should not survive. Momentum and possibly profitability are the most likely survivors.
- After Benjamini-Hochberg correction at FDR = 0.05: 1-4 characteristics survive. BH is less conservative than Bonferroni, so more signals may survive, but the noise characteristics should still be filtered out.
- The student discovers the magnitude of the multiple testing problem: the number of "significant" factors drops by 50-80% after correction. This is the "factor zoo" problem in miniature.
- Harvey, Liu & Zhu (2016) threshold (t > 3.0): 0-2 characteristics survive, consistent with their finding that the majority of published factors are false discoveries.

**Production benchmark:** Harvey, Liu & Zhu (2016), analyzing 296 published significant factors, estimate that ~158 (53%) are false discoveries under multiple testing correction. Their recommended threshold of t > 3.0 (instead of 2.0) eliminates the majority of spurious factors. Our exercise uses 8-10 characteristics (not 296), so the effect is less dramatic, but the principle should be clearly visible: the drop from pre-correction to post-correction significance is the key pedagogical outcome.

---

## Homework Deliverables

### Deliverable 1: The Factor Factory

**Data:** Full 200-ticker universe. Monthly adjusted close prices from yfinance (2014-2024 for momentum; fundamental window for fundamental-based factors). Annual balance sheet data (`Stockholders Equity`, `Total Assets`, `Ordinary Shares Number`) and income statement data (`Operating Income` with `Net Income` fallback) from yfinance. Official factor returns from `getfactormodels`: `FamaFrenchFactors(model='5', frequency='M')` for SMB, HML, RMW, CMA; `FamaFrenchFactors(model='6', frequency='M')` for UMD; `RF` for risk-free rate. Sector info via `yf.Ticker().info['sector']` for quality reporting.

**Acceptance criteria:**
- `FactorBuilder` class instantiates and runs end-to-end without errors on the full 200-ticker universe. Handles download, fundamental ratio computation, double-sort portfolio formation, value-weighted return computation, and factor return aggregation.
- Produces monthly return series for all 6 factors: SMB, HML, RMW, CMA, MOM, and Mkt-RF (market factor = value-weighted universe excess return).
- Factor return series lengths: momentum has ~120-130 monthly returns (price-based, full window minus lookback); fundamental-based factors (SMB, HML, RMW, CMA) have ~36-48 monthly returns (fundamental window).
- Validation against official Ken French data — correlation and tracking error reported for each factor:
  - MOM (calibration anchor): correlation in [0.50, 0.85], tracking error (annualized) in [2%, 15%]. If MOM correlation < 0.40, flag as potential code bug in return computation or portfolio weighting.
  - SMB: correlation in [0.15, 0.55], tracking error in [5%, 25%].
  - HML: correlation in [0.20, 0.60], tracking error in [5%, 25%].
  - RMW: correlation in [0.10, 0.55]. Wider range because operating profitability computation from yfinance may diverge from Compustat's definition.
  - CMA: correlation in [0.10, 0.55]. Wider range because asset growth computation depends on balance sheet coverage depth.
- Double-sort portfolios: each of the 6 portfolios (2 size x 3 characteristic) has at least 15 stocks. No empty portfolios.
- Handles at least 180 of 200 tickers without crash. Graceful failure for tickers with missing fundamental data (logged, not fatal).
- Quality report produced: lists tickers with missing data, tickers using fallback profitability metric, portfolio composition per sort period.

**Production benchmark:** Tidy Finance (Scheuch, Voigt & Weiss, 2023), using CRSP/Compustat with full NYSE/AMEX/NASDAQ universe (~4,000+ stocks), achieves SMB replication R-squared ~0.99 and HML R-squared ~0.96 vs. Ken French. Our S&P 500 version with free data is expected to produce R-squared of ~0.02-0.30 for fundamental-based factors and ~0.25-0.72 for momentum. The gap is structural (universe, survivorship, data depth) and is documented in the validation report.

### Deliverable 2: The Cross-Sectional Feature Matrix

**Data:** Full 200-ticker universe. yfinance balance sheet, income statement, and monthly adjusted close prices. Fundamental window (~2021-2025) for fundamental features. Price data from 2013-2024 (extra year for momentum lookback). Requires Deliverable 1 outputs for validation (factor returns to verify that feature-based portfolio sorts produce returns consistent with factor returns).

**Acceptance criteria:**
- `FeatureEngineer` class instantiates and runs end-to-end without errors on the full 200-ticker universe.
- Output feature matrix: panel DataFrame with MultiIndex (date, ticker). Shape should be approximately (T_months x N_stocks_with_data) rows x (8-12 feature columns). Expected: ~7,200-9,600 rows for the fundamental window, ~24,000-26,000 rows if price-based features extend to the full window.
- Fundamental features computed: P/E, P/B (or B/M), ROE, asset growth, earnings yield. At least 4 of 5 fundamental features successfully computed for >85% of stock-month observations.
- Technical features computed: momentum (12-1 month), short-term reversal (1-month), trailing volatility (60-day std of daily returns). Price-based features should have <5% missing.
- Cross-sectional standardization verified: within each month, each z-scored feature has mean in [-0.05, +0.05] and std in [0.90, 1.10]. Rank-transformed features have range [0, 1].
- Winsorizing verified: after winsorizing at [1st, 99th] percentiles, no raw feature values beyond winsorization bounds. Max |z-score| after standardization < 4.0 for all features (extreme outliers removed).
- Missing data handling: NaN rate documented per feature per month. No feature has >25% missing in any single month (if so, that feature should be flagged or dropped for that month).
- Handles at least 180 of 200 tickers without crash. Graceful degradation for tickers with partial data.
- Output DataFrame is directly consumable by `pd.read_parquet()` and compatible with `linearmodels` panel format for Week 4.

**Production benchmark:** Gu, Kelly & Xiu (2020) construct 94 firm characteristics from CRSP/Compustat covering ~30,000 stock-months per year. Their preprocessing includes cross-sectional rank-transformation and adaptive imputation. Our feature matrix is smaller (8-12 features, 200 stocks) but follows the same standardization methodology. The key quality benchmark is methodological correctness, not scale.

### Deliverable 3: The Factor Model Horse Race

**Data:** Full 200-ticker universe. Monthly excess returns 2014-2024. Three nested sets of characteristics for Fama-MacBeth: (1) CAPM: market beta only (estimated from rolling 60-month window using official Mkt-RF), (2) FF3: market beta + size (log market cap) + value (book-to-market), (3) FF5: FF3 + profitability + investment (asset growth). Requires Deliverable 1 (factor returns) and Deliverable 2 (feature matrix for characteristics). Use `linearmodels.FamaMacBeth` with `cov_type='kernel'` for Newey-West standard errors.

**Acceptance criteria:**
- Fama-MacBeth regressions run for all three model specifications without errors.
- **Layer 1 (CAPM):** Single-characteristic Fama-MacBeth using market beta only. Market beta gamma: monthly estimate in [-0.010, +0.015]. Newey-West |t| likely < 2.0. Average cross-sectional R-squared: [0.00, 0.08].
- **Layer 2 (FF3):** Three-characteristic Fama-MacBeth. Average cross-sectional R-squared should be higher than CAPM: [0.02, 0.15]. The improvement from CAPM to FF3 should be in [0.01, 0.10]. Size and/or value gamma may or may not be individually significant, but the joint model should improve cross-sectional fit.
- **Layer 2 (FF5):** Five-characteristic Fama-MacBeth. Average cross-sectional R-squared should be highest: [0.03, 0.20]. The improvement from FF3 to FF5 should be in [0.00, 0.08]. Diminishing marginal improvement expected.
- R-squared progression: R-squared(CAPM) < R-squared(FF3) <= R-squared(FF5). If this monotonicity is violated, either a code bug exists or the additional factors are truly adding noise for this specific sample.
- Residual alpha analysis: compute average absolute residual (unexplained return) for each model. Average absolute residual should decrease from CAPM to FF3 to FF5, but substantial residual should remain (absolute alpha > 0.5% monthly for the majority of stocks even under FF5). This residual is what Week 4's ML models attempt to capture.
- Newey-West t-statistics for at least one factor in the FF5 specification should exceed 2.0 (at least one factor is priced). Momentum (if included as a 6th characteristic) is the most likely candidate.
- Written interpretation addresses: (1) which model "wins" and by how much, (2) which individual factors are priced (significant gamma), (3) how much alpha remains unexplained, (4) what this implies for ML-based approaches (Week 4 bridge).

**Production benchmark:** Fama & French (2020), using CRSP data, directly compare time-series and cross-sectional factor models. They find that cross-sectional models (the approach used here) generally provide a more complete picture of factor pricing. Typical cross-sectional R-squared for individual stocks is 0.02-0.10 for CAPM, 0.05-0.15 for FF3, and 0.05-0.20 for FF5 with CRSP data. Our S&P 500 results should fall in similar but potentially narrower ranges due to the more homogeneous universe.

---

## Open Questions

1. **Will self-built fundamental-based factors show any meaningful signal above noise given the combination of short fundamental window (~36-48 months) and large-cap-only universe?**
   **Why unknowable:** The compound interaction of two degrading effects (short window + wrong universe) has no published benchmark. Each effect individually is documented, but their joint impact on factor correlation and risk premia significance for exactly our setup (200 S&P 500 stocks, ~2021-2024 fundamentals via yfinance) cannot be predicted with confidence. The probe used current market caps (not historical) and equal weights (not value-weighted), so the actual result depends on the exact implementation.
   **Affects:** S3, Ex1, D1, D3
   **If correlations are in the expected range [0.15-0.55]:** The exercise works as designed — the divergence is the lesson.
   **If correlations are below 0.15 (essentially zero):** The exercise becomes confusing rather than educational. The consolidation agent (Step 6) should frame this as a demonstration that free data has hard limits, and redirect the pedagogical focus to the momentum factor (which should replicate well) as the primary validation target.

2. **Will any factor carry a statistically significant Fama-MacBeth risk premium with only ~36-48 monthly cross-sections and Newey-West standard errors?**
   **Why unknowable:** Statistical power depends on the true effect size relative to noise, and with T~40, even a real premium of 0.5% monthly may not reach significance at 5% with Newey-West correction. The result depends on the specific stocks in the universe, the exact fundamental window, and the realization of returns during that period. Probe results on betas (full 131-month window) showed some significance, but the fundamental-dependent characteristics use a much shorter window.
   **Affects:** Ex2, D3
   **If 1-2 factors are significant:** The exercise demonstrates which factors are robust — the core pedagogical goal.
   **If zero factors are significant:** The exercise still works but with a different lesson: "with limited data, even real factors can appear insignificant." The consolidation agent should contrast with published results (Fama & French, 1992, using CRSP 1963-1990 with T=336 months) to show that power matters.

3. **How will the Barra-style cross-sectional R-squared compare between monthly and daily frequency regressions?**
   **Why unknowable:** The blueprint specifies "daily" Barra regressions, but daily returns have higher noise-to-signal ratios than monthly. With 200 stocks and ~15 regressors, daily cross-sectional R-squared could be much lower than monthly (potentially <0.05), or it could be comparable if the characteristics capture systematic daily patterns. The result depends on the specific characteristics used and the period's market microstructure.
   **Affects:** S6, Ex3
   **If daily R-squared is very low (<0.03):** Use monthly frequency as the primary analysis and show daily as a brief comparison. The consolidation agent should note that production Barra models use daily data but with far more stocks and factors.
   **If daily R-squared is comparable to monthly (>0.05):** Both frequencies work; daily provides more observations for factor return estimation but noisier individual estimates.
