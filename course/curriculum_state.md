# Curriculum State

## Shared Data Infrastructure

All weeks draw from a central cache at `course/shared/.data_cache/`. Code lives in `course/shared/data.py`. Each week's `data_setup.py` imports from `shared.data` and slices to its needed date range / ticker subset. Notebooks (Step 7) never import from shared — they inline their own downloads. Full data quality analysis (bias, limitations, PIT concerns) lives in `course/guides/data.md`.

### Equity Prices — OHLCV

| File | What | Coverage |
|---|---|---|
| `all_ohlcv.parquet` (~110 MB) | Full OHLCV (Open, High, Low, Close, Volume). MultiIndex columns (field, ticker). | 2000–2025, daily |
| `all_prices.parquet` (~19 MB) | Close-only fast path for `load_sp500_prices()`. | Same |

- **Universe constants:**
  - `SP500_TICKERS` (503) — current S&P 500 constituents (Feb 2026)
  - `SP400_TICKERS` (~150) — curated mid-cap stocks
  - `ALL_EQUITY_TICKERS` = SP500 + SP400 (deduplicated)
  - `ETF_TICKERS` (~70) — organized by category: core (SPY, QQQ, IWM…), sector (XL*), factor (MTUM, QUAL…), fixed income (AGG, TLT…), international (EFA, EEM…), commodity (GLD, USO…), volatility (VXX…), REIT (VNQ…), thematic (ARKK…)
  - `DEMO_TICKERS` (20) — quick-loading subset for demos
- **After completeness filter:** 456 of 511 attempted tickers survived (>50% non-null days over 2000–2025). Dropped tickers are mostly post-2010 IPOs.
- **Access:** `load_sp500_prices(start, end, tickers)` → Close only. `load_sp500_ohlcv(start, end, tickers, fields)` → full OHLCV with MultiIndex.
- **Limitations:** Survivorship bias — current constituents projected backward, inflates returns 1–4%/yr. Daily bars only (no intraday). 2–5% missing days normal for mid-caps.

### Fama-French & Carhart Factor Returns

| File | Factors | Coverage |
|---|---|---|
| `ff3_{M,D}_factors.parquet` | Mkt-RF, SMB, HML, RF | 1926-07 to 2025-12 |
| `ff5_{M,D}_factors.parquet` | Mkt-RF, SMB, HML, RMW, CMA, RF | 1963-07 to 2025-12 |
| `ff6_{M,D}_factors.parquet` | FF5 + UMD (momentum) | 1963-07 to 2025-12 |
| `carhart_{M,D}_factors.parquet` | Mkt-RF, SMB, HML, MOM | 1926-11 to 2025-06 |

- **Source:** Ken French Data Library via `getfactormodels`
- **Access:** `load_ff_factors(model='3'|'5'|'6', frequency='M'|'D')`, `load_carhart_factors(frequency='M'|'D')`
- **Limitations:** Aggregate portfolio-level returns only (not stock-level). Value-weighted. Returns in percent (1.5 = 1.5%) — divide by 100 when mixing with yfinance decimal returns.

### Ken French Sorted Portfolios

17 datasets, each cached in monthly and daily frequency (34 parquet files total):

| Key | Description | Portfolios |
|---|---|---|
| `25_size_bm` | 5×5 double sort: size × book-to-market | 25 |
| `25_size_mom` | 5×5 double sort: size × prior 12-2 return | 25 |
| `25_size_inv` | 5×5 double sort: size × investment | 25 |
| `25_size_op` | 5×5 double sort: size × oper. profitability | 25 |
| `6_size_bm` | 2×3 sort: size × B/M (FF factor construction) | 6 |
| `17_industry` | 17 industry portfolios | 17 |
| `49_industry` | 49 industry portfolios | 49 |
| `10_momentum` | Deciles on prior 12-2 month returns | 10 |
| `10_size` | Deciles on market equity | 10 |
| `10_bm` | Deciles on book-to-market | 10 |
| `10_ep` | Deciles on earnings/price | 10 |
| `10_cfp` | Deciles on cash flow/price | 10 |
| `10_dp` | Deciles on dividend/price | 10 |
| `10_inv` | Deciles on investment (asset growth) | 10 |
| `10_op` | Deciles on operating profitability | 10 |
| `10_st_reversal` | Deciles on short-term reversal | 10 |
| `10_lt_reversal` | Deciles on long-term reversal | 10 |

- **Source:** Ken French Data Library (direct ZIP download, CSV parsed). Coverage: 1926–2025 (monthly + daily).
- **Access:** `load_ff_portfolios(name, frequency='M'|'D')`. Returns in percent.
- **Limitations:** Portfolio-level only (no stock-level membership or weights). Value-weighted. US equities only. Returns in percent.
- **Use cases:** Factor model validation (W3), industry analysis (W3-4), momentum benchmarking (W3-4), PIT-clean benchmarks for validating self-built fundamental sorts (compare yfinance sorts to KF deciles), causal factor tests (W10).

### FRED Economic & Market Data

59 series across 12 categories, each cached as `fred_{SERIES_ID}.parquet`. Coverage: 2000–2025 (or closest available).

| Category | Const | Series | Freq |
|---|---|---|---|
| Treasury yields (11) | `FRED_TREASURY_YIELDS` | DGS1MO through DGS30 — full nominal curve | Daily |
| TIPS real yields (4) | `FRED_TIPS_YIELDS` | DFII5, DFII10, DFII20, DFII30 | Daily |
| Fed funds (2) | `FRED_RATES` | FEDFUNDS (monthly), DFF (daily) | Mixed |
| VIX (1) | `FRED_VOLATILITY` | VIXCLS | Daily |
| Credit spreads (5) | `FRED_CREDIT` | BAMLH0A0HYM2 (HY OAS), BAMLC0A4CBBB (BBB), BAMLC0A0CM (IG), TEDRATE, DPRIME | Daily |
| FX rates (5) | `FRED_DOLLAR_FX` | DTWEXBGS (dollar index), DEXJPUS, DEXUSEU, DEXCHUS, DEXUSUK | Daily |
| Commodity prices (5) | `FRED_COMMODITIES_PRICES` | DCOILWTICO, DCOILBRENTEU, GOLDAMGBD228NLBM, DHHNGSP, PCOPPUSDM | Daily/Monthly |
| Inflation expectations (4) | `FRED_INFLATION_EXPECTATIONS` | T5YIE, T10YIE (breakevens), T5YIFR (forward), MICH (survey) | Daily/Monthly |
| Financial conditions (3) | `FRED_FINANCIAL_CONDITIONS` | NFCI, STLFSI2, DRTSCILM | Weekly/Quarterly |
| Macro (12) | `FRED_MACRO` | GDP, GDPC1, CPIAUCSL, CPILFESL, PCEPI, UNRATE, UMCSENT, INDPRO, T10Y2Y, T10Y3M, USREC, M2SL, WALCL | Mixed |
| Housing (3) | `FRED_HOUSING` | HOUST, CSUSHPISA, MORTGAGE30US | Monthly/Weekly |
| Labor (4) | `FRED_LABOR` | PAYEMS, ICSA, JTSJOL, AWHAETP | Monthly/Weekly |

- **Source:** FRED public CSV endpoint (`fredgraph.csv`), no API key.
- **Access:** `load_fred_series(series_ids, start, end)` — also accepts ad-hoc IDs not pre-cached; they download and cache on first call.
- **Limitations:** Macro series (GDP, payrolls, industrial production) carry revision risk — FRED stores the latest revision, not the first print. GDP worst case: ±1pp. Apply 1-month publication lag for macro features. USREC is backdated (NBER declares recessions 6–18 months after they begin) — use only as evaluation label, never as predictive feature. Market-observed series (yields, VIX, credit spreads, FX, commodities) have no revision issues.
- **Use cases:** Risk-free rate (W3-6), yield curve (W12), recession regimes (W14), credit spreads (W12), macro features (W4, W10), cross-asset analysis (W6, W14).

### Fundamentals

| File | What |
|---|---|
| `fundamentals_bs.parquet` | Balance sheet: Stockholders Equity, Total Assets, Shares, Total Debt, Net Debt, Cash, Current/Non-Current Assets & Liabilities, Invested Capital, Goodwill, Working Capital (14 fields). Multi-index (ticker, date). |
| `fundamentals_inc.parquet` | Income statement: Revenue, Gross Profit, Operating Income, EBITDA, Net Income, Pretax Income, R&D, Cost of Revenue, SG&A, EPS (12 fields). Multi-index (ticker, date). |
| `fundamentals_cf.parquet` | Cash flow: Operating CF, CapEx, FCF, Dividends, Buybacks, Debt Issuance/Repayment, Changes in Cash (8 fields). Multi-index (ticker, date). |
| `fundamentals_sector.parquet` | GICS sector per ticker (current snapshot). |
| `fundamentals_industry.parquet` | GICS industry per ticker (current snapshot). |
| `fundamentals_mcap.parquet` | Market capitalization per ticker (current snapshot). |
| `fundamentals_shares.parquet` | Shares outstanding over time. Multi-index (ticker, date). |
| `fundamentals_ratios.parquet` | Valuation ratios (P/E, P/B, EV/EBITDA, EV/Revenue, ROE, ROA, D/E, current ratio, margins, growth rates, short interest, institutional holdings — 28 fields). **Current snapshot only, not historical.** |

- **Source:** yfinance `.balance_sheet`, `.income_stmt`, `.cashflow`, `.info` (~455 tickers)
- **Access:** `load_sp500_fundamentals(tickers)` → dict with keys: `balance_sheet`, `income_stmt`, `cashflow`, `sector`, `industry`, `market_cap`, `shares`, `ratios`
- **Limitations:** Point-in-time contaminated — yfinance indexes by fiscal period end but 10-K filings arrive ~55 days later (reporting-lag bias), and restated values silently overwrite originals (restatement bias). Default mitigation: +90 day PIT lag. Snapshot-only ratios (P/E, P/B, etc.) reflect today's values, not historical — never use for backtesting. ~4 years of history only. 15–20% coverage gaps on some balance sheet items.
- **Use cases:** Feature engineering (W3-4), sector/industry groupings (W3, W10, W17), market-cap weighting (W6).

### Options Chains

- **Source:** yfinance (current snapshot). Cached with 7-day freshness check.
- **Access:** `load_options_chain(tickers, near_expiries=4, max_age_days=7)`.
- **Fields:** ticker, expiry, strike, type (call/put), lastPrice, bid, ask, volume, openInterest, impliedVolatility, inTheMoney, spot price, moneyness (S/K), time-to-expiry.
- **Limitations:** Current snapshot only — no historical option data, cannot backtest option strategies. Not reproducible (chains change on every download) — exercises must reference relationships (put-call parity, vol smile shape, Greeks sensitivity), not specific contracts. Stale quotes on low-volume strikes.
- **Use cases:** Week 12 (Derivatives — Black-Scholes pricing, Greeks, vol surface fitting).

### SEC EDGAR Filings

- **Source:** SEC EDGAR submissions API. Cached per ticker per filing type.
- **Access:** `load_sec_filings(tickers, filing_type='10-K', max_filings=5)`.
- **What:** Full text of 10-K/10-Q filings with real filing dates (actual SEC submission timestamps). Filing dates are when the market could actually see the document.
- **Limitations:** 5 filings/ticker default (SEC rate-limits to 10 req/sec). HTML artifacts may remain after stripping. Text truncated at ~500K chars. Some tickers may not map to a CIK. Amendments (10-K/A) not flagged.
- **Use cases:** Week 15 (NLP — sentiment extraction, readability scoring, MD&A tone analysis).

### Crypto / DeFi Prices

| File | What | Coverage |
|---|---|---|
| `crypto_prices.parquet` | Daily close for 8–21 tokens (Layer 1 + DeFi governance + L2) | 2014–2025 |

- **Tokens:** BTC, ETH, SOL, BNB, ADA, AVAX, DOT, MATIC, ATOM, NEAR (Layer 1); UNI, AAVE, LINK, CRV, MKR, COMP, SNX, SUSHI, LDO (DeFi); ARB, OP (L2). After completeness filter, ~8–21 survive.
- **Source:** yfinance. **Access:** `load_crypto_prices(start, end, tickers)`.
- **Limitations:** Short histories (most DeFi tokens post-2020). 24/7 markets (daily close = midnight UTC, not a market close) — affects correlation studies with equities. No on-chain data (no TVL, gas fees, or protocol metrics).
- **Use cases:** Week 18 (DeFi, AMM dynamics, impermanent loss analysis).

### Synthetic Limit Order Book

- **Source:** Generated by `generate_synthetic_lob()` — not downloaded.
- **Access:** `generate_synthetic_lob(n_levels=10, n_snapshots=10_000, mid_price=100.0, tick_size=0.01, avg_spread_ticks=3.0, seed=42)`.
- **What:** LOB snapshots with mean-reverting mid-price, bid-ask spread dynamics, exponential depth decay, autocorrelated order flow, simulated trades.
- **Limitations:** Synthetic — no real market dynamics. Minimal placeholder (no strategic agents, single day/asset, random flow only). Will likely need reimplementation for Week 13 with more sophisticated dynamics (configurable market impact, multi-day sessions, realistic queue dynamics).
- **Use cases:** Week 13 (LOB visualization, OFI computation, Almgren-Chriss execution simulation).

---

## After Week 1: How Markets Work & Financial Data

### Concepts Taught
- Trade execution mechanics: order submission → broker → exchange/dark pool → matching engine → settlement (COVERED)
- Limit order book: bid/ask structure, L1/L2 data, spread as market maker compensation, mid-price (COVERED)
- Market makers: liquidity provision, adverse selection risk, maker-taker model, PFOF (COVERED)
- OHLCV bars: tick data aggregation into Open/High/Low/Close/Volume at fixed intervals (COVERED)
- Survivorship bias: current-day universe projected backward inflates backtest returns 1-4%/yr (COVERED)
- Look-ahead bias: using information at time T not available until T+k (COVERED)
- Corporate actions: splits, dividends, adjusted vs. unadjusted prices, adjustment factor computation (COVERED)
- Modern data stack: yfinance, CRSP/WRDS awareness, Parquet vs. CSV, Polars vs. pandas (COVERED)
- Commodities, FX, crypto market structure differences from equities (MENTIONED ABSTRACTLY)

### Skills Acquired
- Download and handle OHLCV data via yfinance for arbitrary tickers and date ranges
- Compute daily returns from adjusted close prices (`pct_change()`)
- Detect survivorship bias by comparing current vs. historical universes
- Detect corporate actions programmatically from adjustment factor jumps
- Run automated data quality checks: missing days, stale prices, OHLC consistency, outlier returns
- Store and load data in Parquet format using Polars; benchmark against CSV/pandas alternatives

### Finance Vocabulary Established
order book, bid, ask, spread, mid-price, market order, limit order, stop order, matching engine, dark pool, market maker, adverse selection, maker-taker, PFOF, OHLCV, adjusted close, unadjusted price, stock split, reverse split, dividend, adjustment factor, survivorship bias, look-ahead bias, point-in-time database, CRSP, Compustat, WRDS, Parquet

### Artifacts Built
- `FinancialDataLoader` class: downloads, cleans, validates, stores multi-asset daily data in Parquet (homework)
- Data quality report with per-ticker grades A/B/C/D (homework)
- Storage format benchmark: CSV vs. Parquet × pandas vs. Polars (homework)
- Return statistics summary: mean, volatility, skewness, kurtosis for 10 tickers (homework — bridge to Week 2)

### Key Caveats Established
- yfinance is educational-grade, not production-reliable (rate limits, schema changes, no point-in-time support)
- Always use adjusted prices for return computation, unadjusted for level analysis (e.g., strike prices, dollar volume)
- Data quality varies by liquidity: large-cap near-perfect, smaller names have real issues

### Reusable Downstream
- Can assume: OHLCV data handling, return computation, adjusted vs. unadjusted distinction, yfinance usage, Polars/pandas familiarity, Parquet I/O
- Can reference by name: `FinancialDataLoader` class, data quality checks pattern
- Don't re-teach: trade execution, order book mechanics, survivorship bias definition, corporate actions, CSV vs. Parquet motivation

---

## After Week 2: Financial Time Series & Volatility

### Concepts Taught
- Time series primer: white noise, random walk, strict/weak stationarity, AR/MA/ARMA models, lag operator, ACF/PACF interpretation, conditional variance concept (PREREQ EXPLAINED)
- Stylized facts of returns: uncorrelated returns, correlated squared returns (volatility clustering), fat tails (kurtosis >> 3), negative skewness / leverage effect, volatility persistence (COVERED)
- Stationarity testing: ADF (null: unit root) and KPSS (null: stationary), joint interpretation, disagreement as signal of fractional integration (COVERED)
- GARCH(1,1): conditional variance model — omega, alpha (reaction), beta (persistence), long-run volatility; 40-year industry standard (COVERED)
- Asymmetric GARCH: EGARCH (log-vol, gamma < 0), GJR-GARCH (bad-news indicator), leverage effect, news impact curves, BIC model selection (COVERED)
- Realized volatility: model-free measurement at 5/21/63-day horizons, rolling window tradeoff (TOUCHED)
- Fractional differentiation: d ∈ (0,1), stationarity-memory tradeoff, minimum d search, Lopez de Prado Ch. 5 (COVERED)
- Volatility forecasting: GARCH one-step-ahead, QLIKE loss, Mincer-Zarnowitz regression, comparison with rolling-window baseline (TOUCHED)

### Skills Acquired
- Run and jointly interpret ADF + KPSS stationarity tests on any financial series
- Read ACF/PACF plots and identify AR, MA, and ARCH structure
- Fit GARCH(1,1), EGARCH, GJR-GARCH using the `arch` library and interpret all parameters financially
- Select best volatility model by BIC; handle convergence failures gracefully
- Compute rolling realized volatility at multiple horizons (annualized)
- Apply fractional differentiation and find minimum d for stationarity via grid search
- Evaluate volatility forecasts with QLIKE and Mincer-Zarnowitz R²

### Finance Vocabulary Established
stationarity, unit root, ADF test, KPSS test, autocorrelation, partial autocorrelation, ACF, PACF, ARMA, ARCH effect, volatility clustering, fat tails, kurtosis, skewness, leverage effect, GARCH, EGARCH, GJR-GARCH, conditional variance, unconditional variance, persistence (alpha + beta), long-run volatility, realized volatility, news impact curve, fractional differentiation, Jarque-Bera test, Ljung-Box test, BIC, VaR, Expected Shortfall, QLIKE, Mincer-Zarnowitz regression

### Artifacts Built
- `VolatilityAnalyzer` class: stationarity diagnostics, stylized fact verification, multi-model GARCH fitting, volatility forecasting (homework)
- Multi-asset volatility comparison report across equities, ETFs, bonds, commodities (homework)
- Fractional differentiation feature builder: finds optimal d per ticker, returns stationary + memory-rich series (homework)
- GARCH forecast evaluation pipeline: out-of-sample QLIKE, MSE, Mincer-Zarnowitz vs. rolling-window baseline (homework)

### Key Caveats Established
- Returns are approximately stationary but NOT independent — squared returns show strong autocorrelation (ARCH effects)
- GARCH(1,1) is the production standard; asymmetric models (EGARCH/GJR) usually win by BIC for equities due to the leverage effect
- Volatility is forecastable (Mincer-Zarnowitz R² typically 0.15-0.40); return direction is not — this asymmetry is fundamental
- Fractional differentiation preserves 30-70% more memory than integer differencing (d=1); optimal d varies by asset (typically 0.3-0.7 for equities)
- ADF/KPSS disagreement signals possible fractional integration, not a test failure

### Reusable Downstream
- Can assume: stationarity testing, ACF/PACF reading, GARCH fitting and parameter interpretation, realized vol computation, fractional differentiation, the six stylized facts as established baseline
- Can reference by name: `VolatilityAnalyzer` class, stylized facts, GARCH conditional volatility, fractional differentiation feature builder
- Don't re-teach: what stationarity means, why returns are fat-tailed, GARCH(1,1) specification, the leverage effect, the stationarity-memory tradeoff

---

## After Week 3: Factor Models & Cross-Sectional Analysis

### Concepts Taught
- CAPM: beta estimation (OLS of excess returns on market excess return), security market line, Jensen's alpha, R-squared as explanatory power, sample-dependence of the SML (steep in 2014-2024, flat in 1963-1990 per Fama & French 1992) (COVERED)
- Fama-French three-factor model: SMB, HML, time-series regression, R-squared improvement over CAPM, alpha shrinkage (60% of stocks), cumulative factor returns 1926-2025, "death of value" post-2007 (COVERED)
- Factor construction via double sorts: breakpoints (median size, 30/70 B/M), 2x3 portfolio formation, value-weighted returns, SMB and HML from raw data (COVERED)
- Factor replication and validation: correlation and tracking error against Ken French official data, universe bias as driver of replication quality (HML r=0.82 vs. SMB r=0.19); static-sort methodology caveat (look-ahead bias from applying ~2025 fundamentals backward over 2014-2024 returns) (COVERED)
- FF5 model: RMW (profitability) and CMA (investment) factors, diminishing R-squared improvement (+0.025 median), HML-CMA redundancy (r=0.65), alpha reallocation vs. alpha shrinkage (COVERED)
- Momentum: strong long-run returns, catastrophic crash risk (2009: 60% drawdown), excluded from FF5 (no risk-based theory), UMD-HML negative correlation (-0.31) (COVERED)
- Fama-MacBeth two-step procedure: time-series beta estimation, cross-sectional regression per period, time-series average of slopes as risk premium estimate, Newey-West standard errors, manual vs. linearmodels implementation comparison (COVERED)
- Barra-style risk model: cross-sectional regression of returns on characteristics + sector dummies, factor risk vs. specific risk decomposition, diversified portfolio R-squared ~ 0.89 (COVERED)
- Factor premia sample dependence: beta significantly priced in 2014-2024 but not 1963-1990; momentum insignificant in recent sample; factor significance varies by period and universe (COVERED)
- Factor zoo and multiple testing: 50% of naive significant factors eliminated by Bonferroni correction, noise factor false discovery at p < 0.01 (noise_1 t=2.69), Harvey-Liu-Zhu t > 3.0 threshold (COVERED)
- Cross-sectional feature engineering: why P/E is pathological (sign-flip for loss-makers, non-monotonic), why E/P is preferred, two-stage outlier control (percentile clip + z-cap), feature correlation verification (COVERED)

### Skills Acquired
- Estimate CAPM betas and plot the security market line for arbitrary stock universes
- Run FF3 and FF5 time-series regressions and compare explanatory power across model specifications
- Construct factors from raw data via double-sort portfolio methodology (implemented in FactorBuilder class)
- Validate self-built factors against benchmark data using correlation and tracking error metrics
- Execute the Fama-MacBeth two-step procedure (manual and via linearmodels library)
- Apply Newey-West standard errors to correct for serial correlation in cross-sectional slope estimates
- Build a simplified Barra-style cross-sectional regression and decompose portfolio risk into factor and specific components
- Apply Bonferroni and Benjamini-Hochberg multiple testing corrections to Fama-MacBeth t-statistics
- Construct a standardized cross-sectional feature matrix with two-stage outlier control (FeatureEngineer class)

### Finance Vocabulary Established
beta, alpha (Jensen's alpha), security market line (SML), systematic risk, idiosyncratic risk, excess return, risk-free rate, SMB, HML, RMW, CMA, UMD/MOM, double sort, breakpoint, value-weighted return, book-to-market, operating profitability, asset growth, earnings yield, Fama-MacBeth regression, cross-sectional regression, risk premium, Newey-West standard error, factor loading, factor risk, specific risk, Barra risk model, factor zoo, multiple testing correction, Bonferroni, Benjamini-Hochberg, false discovery rate, Harvey-Liu-Zhu threshold, winsorization, z-score standardization

### Artifacts Built
- FactorBuilder class: downloads raw data, computes fundamentals, forms double-sort portfolios, produces 6 factor return series, validates against Ken French (homework D1)
- FeatureEngineer class: computes 7 cross-sectional features (4 fundamental + 3 technical), two-stage outlier control (percentile clip + z-cap at ±3), rank-transform, output to Parquet for Week 4 (homework D2)
- Factor model horse race: Fama-MacBeth regressions for CAPM, FF3, FF5 with R-squared progression and residual alpha analysis (homework D3)

### Key Caveats Established
- S&P 500 universe cannot produce a real size factor (SMB r=0.19 with Ken French); value factor replicates well (HML r=0.82), but the HML correlation is flattered by the static-sort methodology (look-ahead bias from applying ~2025 fundamentals backward over the full 2014-2024 window; a proper annual-rebalancing approach would likely produce lower correlations)
- Factor premia significance is sample-period dependent — the "beta anomaly" and momentum strength vary across decades
- Self-built factor returns should always be validated against benchmark data; never trust unchecked factor construction; and always understand the methodology behind any benchmark comparison (static sort vs. annual rebalancing produces different correlation numbers)
- Cross-sectional R-squared from factor models is modest (8-15%); ~85% of return variation remains unexplained — this is the opportunity space for ML
- Diversified portfolios can have HIGHER factor risk share than concentrated portfolios if the concentrated sector has high idiosyncratic risk
- P/E ratio is pathological for cross-sectional ML (sign-flip at E<0, non-monotonic); use earnings yield (E/P) instead
- Winsorization at [1,99] percentiles is insufficient for small universes (~179 stocks); always add a z-score hard cap
- yFinance fundamental data is not point-in-time: reporting dates are unknown, introducing potential look-ahead bias beyond the static-sort issue

### Reusable Downstream
- Can assume: CAPM/FF3/FF5 factor regression, beta estimation, factor loading interpretation, Fama-MacBeth procedure, cross-sectional standardization, multiple testing correction concepts, factor construction methodology, two-stage outlier control for feature matrices
- Can reference by name: FactorBuilder class, FeatureEngineer class, factor model horse race results, the factor replication hierarchy (HML > MOM > RMW > SMB > CMA), feature_matrix_ml.parquet
- Don't re-teach: what factors are, why CAPM fails, how double sorts work, why survivorship bias and universe coverage matter (established in Week 1 and reinforced here)

---

## After Week 4: ML for Alpha — From Features to Signals

### Concepts Taught
- Cross-sectional return prediction: the Gu-Kelly-Xiu framework — features at time t predict returns at t+1, repeated monthly across a stock universe. 174 S&P 500 stocks, 7 features, 130 months (2014-03 to 2024-12), 68 OOS months (COVERED)
- Signal evaluation metrics: IC, rank IC, ICIR. Mean Pearson IC = 0.047 (t = 2.54, p = 0.013), ICIR = 0.234. IC of 0.02–0.05 is realistic; IC > 0.10 warrants suspicion (COVERED)
- The fundamental law of active management: IR = IC × sqrt(BR). Predicted IR = 0.615 vs. actual ICIR = 0.234 — the law overestimates by 2.6x because it assumes stock-level independence (COVERED)
- Gradient boosting for cross-sectional alpha: LightGBM OOS IC = 0.046 (t = 2.15, p = 0.035). Train/OOS ratio = 6.73 (overfitting). Does not significantly outperform naive momentum (p = 0.57). HP grid shifted from expectations (lr grid [0.005, 0.01, 0.05], n_est=10 via aggressive early stopping) (COVERED)
- Neural networks vs. trees: NN IC = 0.046 (t = 1.74, p = 0.087 — borderline, not significant at 5%). GBM-NN paired p = 0.99 (indistinguishable). Scale dependence of "virtue of complexity" (Kelly-Malamud-Zhou 2023) (COVERED)
- Feature engineering for alpha: expanding 7→12 features yields IC change +0.004 (not significant). 7→18 features degrades IC by 0.020 (p = 0.13). Near-multicollinearity (r = 0.923 between momentum_z and mom_12m_1m). Substitution effect ratio = 1.68 (COVERED)
- SHAP / TreeSHAP: volatility_z dominates (1.6x next feature), momentum variants and reversal fill top spots. Grouped SHAP-permutation rank correlation = 0.54. Consistent with GKX finding (COVERED)
- Signal-to-portfolio: decile sort → long-short gross Sharpe = 0.77, net (10 bps) = 0.69. Turnover = 79% monthly (9.5x annual). Max drawdown = -23.5%. Excess kurtosis = 3.67 (Sharpe understates tail risk). Breakeven cost = 61 bps (COVERED)
- Alternative data landscape: 7 taxonomy categories, $1.6M/year median hedge fund spend, NLP as most accessible entry point, bridge to Week 7 (TOUCHED)
- Sandbox vs. production gap: S&P 500 / 7 features / yfinance vs. CRSP / 94 features / PIT data — systematically established across all sections (COVERED)

### Skills Acquired
- Frame cross-sectional prediction in the GKX structure (features → target → temporal split → IC evaluation)
- Compute IC, rank IC, and ICIR; interpret t-statistics and assess signal significance
- Train LightGBM with walk-forward temporal splitting (60-month window, 1-month purge gap, no shuffle)
- Train a feedforward neural network and compare head-to-head via paired t-test
- Engineer interaction features (mom_x_vol), non-linear transforms (momentum_sq, abs_reversal), and measure marginal IC contribution
- Use TreeSHAP for feature importance with grouped attribution for correlated features
- Construct long-short decile portfolios from model predictions and compute Sharpe, turnover, net Sharpe
- Apply transaction cost haircuts at multiple cost levels; compute breakeven cost
- Run feature knockout experiments and measure substitution effects in correlated features

### Finance Vocabulary Established
information coefficient (IC), rank IC, IC information ratio (ICIR), fundamental law of active management, breadth (BR), cross-sectional prediction, forward return, excess cross-sectional return, alpha signal, signal quality, decile portfolio, quintile portfolio, long-short portfolio, turnover, transaction cost drag, breakeven cost, alternative data, alpha decay, signal-to-portfolio gap, Gu-Kelly-Xiu framework, TreeSHAP (financial application), feature knockout, substitution effect, implementation sensitivity, deflated Sharpe ratio

### Artifacts Built
- `AlphaModelPipeline` class: temporal walk-forward, IC/rank IC/ICIR evaluation, long-short portfolio construction; default GBM IC = 0.055 (t = 2.51), gross Sharpe = 0.976 (homework D1)
- Extended feature matrix: 18 features (7 original + 5 price-derived + 2 fundamental + 2 interaction + 2 non-linear), with SHAP importance ranking. IC degraded from 0.055 to 0.034 (homework D2)
- Model comparison report: OLS, Ridge, LightGBM, NN on 18 features — no model reaches 5% significance on expanded features. Inverted ranking (OLS/Ridge > GBM > NN) vs. GKX production ranking (homework D3)

### Key Caveats Established
- Features matter more than model architecture — GBM and NN are indistinguishable (p = 0.99) on 7 features, 174 stocks
- IC of 0.02–0.05 is realistic; GBM IC = 0.046 sits at the upper end. GBM does not significantly outperform single-feature momentum (p = 0.57)
- The "virtue of complexity" requires scale: on 7 features/174 stocks, depth-8 trees destroy significance (t = 1.27, p = 0.21) while depth-3 works (t = 2.63, p = 0.011)
- Feature expansion can degrade signals on small universes — 18 features on 174 stocks yields no significant model
- Signal strength (IC) and profitability (net Sharpe) diverge: 79% monthly turnover makes the strategy impractical despite IC = 0.046
- Implementation sensitivity: same GBM model produces Sharpe range 0.77–0.98 across different code paths (random seeds, NaN handling, train/test boundaries)
- Excess kurtosis > 3.0 means Sharpe understates tail risk
- Static fundamental ratios carry full look-ahead bias (time-varying data had 79% missing); PIT-contaminated fundamentals hurt IC, not inflate it
- VIX circularity when classifying regimes for an S&P 500 alpha model

### Reusable Downstream
- Can assume: IC/rank IC/ICIR computation and interpretation, cross-sectional prediction framework, LightGBM for tabular financial features, SHAP-based feature importance, long-short decile portfolio construction, turnover and transaction cost estimation, the fundamental law of active management, feature knockout methodology, substitution effect concept
- Can reference by name: `AlphaModelPipeline` class, extended feature matrix (18 features), model comparison report, the GKX framework, feature_matrix_ml.parquet (from Week 3)
- Don't re-teach: what IC is, why gradient boosting dominates tabular financial data, why features matter more than models in cross-sectional prediction, the alternative data taxonomy, substitution effect mechanics

---

## After Week 5: Backtesting, Research Discipline & Transaction Costs

### Concepts Taught
- The seven canonical backtest failure modes: look-ahead bias, survivorship bias, overfitting, data snooping, cherry-picking, ignoring costs, ignoring market impact — anchored by a look-ahead bug demonstration (IC = 1.0 flawed vs. -0.013 corrected) and survivorship simulation (3.07% annual premium via simulated delisting) (COVERED)
- Purged cross-validation: label-aware train/test splitting that prevents information leakage through forward-looking label construction. Purging effect is frequency-dependent — near-zero on monthly data with 1-month labels (IC delta = -0.0011, t = -0.030) because the contamination zone compresses to a single observation; material on daily data with multi-day labels (10-30% IC reduction per Lopez de Prado 2018) (COVERED)
- Combinatorial purged cross-validation (CPCV): k=6 produces 15 paths, enabling Probability of Backtest Overfitting (PBO) computation. PBO = 0.267 indicates IS winner maintains relative ranking OOS, but PBO < 0.5 does not confirm statistical significance — all three models fail t = 1.96 (GBM t = 1.387, NN t = 1.027, Ridge t = -0.058). PBO measures rank stability, not signal reality (COVERED)
- Multiple testing correction: Harvey-Liu-Zhu t > 3.0 threshold, Benjamini-Hochberg-Yekutieli (BHY) procedure, 50-variant FDR simulation showing 0/50 false positives at p < 0.05 (COVERED)
- Transaction cost decomposition: spread costs, market impact (Almgren-Chriss square-root law), slippage. Three-regime evaluation (zero, fixed 5 bps, tiered 10/20/30 bps). At 139.9% monthly turnover: spread drag = 1.68%/yr at 5 bps, market impact = 1.02%/yr at eta=0.1/$100M AUM. Gross Sharpe 0.871 → net tiered 0.624 (28% reduction) (COVERED)
- Deflated Sharpe Ratio (DSR): adjusts observed Sharpe for multiple testing, non-normality, and track record length. DSR surface over T×M grid spanning [0.159, 1.000]. Non-normality (excess kurtosis = 4.23) inflates MinTRL by ~10x through the (gamma4+2)/4 coefficient — strategy needs 174 months (14.5 years) to confirm Sharpe at M=10 vs. 10 months under Gaussian assumption (COVERED)
- MinTRL (Minimum Track Record Length): at M=1 (no trial penalty) = 77.5 months; at M=10 = 174 months. Observed track record = 67 months — insufficient under both assumptions. The trial count is a 2x multiplier on required track record (COVERED)
- Responsible vs. naive backtesting: naive Sharpe = 0.877, responsible Sharpe = 0.575, gap = 0.302 (34% reduction). DSR@M=10 = 0.414 (NO-DEPLOY). Sub-period IC decline: 0.054 → 0.039 (27%). The gap between naive and responsible evaluation IS the information (COVERED)
- Survivorship bias × DSR interaction: gross Sharpe of 0.88 includes survivorship inflation of 0.1-0.3 annual units; survivorship-corrected DSR would be more pessimistic. DSR results are an upper bound on statistical significance (COVERED)
- Feasibility frontier: 9×6 cost-sensitivity grid showing net Sharpe across spread × turnover-reduction dimensions. Strategy viable at large-cap spreads (2-5 bps), breaks at mid-cap (12-25 bps). Breakeven spread = 15 bps at 0% turnover reduction. 46/54 cells feasible (COVERED)

### Skills Acquired
- Diagnose backtest failure modes by IC fingerprint: look-ahead (IS/OOS ratio >> 5x), survivorship (ratio near 1x with elevated IS+OOS), correct (ratio near 1x with low IC)
- Implement purged k-fold CV from scratch (PurgedKFold class with sklearn BaseCrossValidator interface), verify correctness via zero-leakage invariant (n_leaking = 0)
- Run CPCV with k=6, compute PBO from 15 paths, interpret PBO alongside IC t-statistics (PBO measures rank stability, not signal significance)
- Apply BHY multiple testing correction to Fama-MacBeth or model-comparison p-values
- Decompose transaction costs into spread, market impact, and slippage components using a TransactionCostModel class
- Compute DSR surfaces over trial-count × track-record-length grids; compute MinTRL incorporating skewness and kurtosis
- Build a three-layer responsible backtest report: tearsheet + DSR verdict, CPCV + PBO + BHY correction, MinTRL + deployment readiness
- Evaluate strategy feasibility across a cost-sensitivity grid and identify the feasibility frontier
- Compare naive vs. responsible backtest evaluation and quantify the methodology gap

### Finance Vocabulary Established
purged cross-validation, embargo period, combinatorial purged CV (CPCV), probability of backtest overfitting (PBO), deflated Sharpe ratio (DSR), minimum track record length (MinTRL), Harvey-Liu-Zhu threshold (t > 3.0), Benjamini-Hochberg-Yekutieli (BHY) correction, transaction cost decomposition, bid-ask spread (half-spread), market impact (Almgren-Chriss square-root model), slippage, turnover (one-way monthly), cost regime, feasibility frontier, breakeven spread, responsible backtest, naive backtest, backtest overfitting, seven sins of backtesting, DEPLOY/NO-DEPLOY verdict, sub-period stability, signal decay

### Artifacts Built
- `PurgedKFold` class: sklearn-compatible cross-validator with label-aware purging and embargo, supports DatetimeIndex and positional fallback, correctness-verified (n_leaking = 0 at k=5 and k=10) (homework D1)
- `TransactionCostModel` class: fit(ohlcv) + transform(gross_returns) interface, three-regime cost decomposition (spread + impact + slippage), correctness-verified on synthetic 2-asset portfolio (exact match) (homework D2)
- Three-layer responsible backtest report: quantstats tearsheet + DSR@M=10 verdict (DEPLOY/NO-DEPLOY), CPCV + PBO + BHY correction, MinTRL + deployment readiness assessment (homework D3)
- 9×6 cost-sensitivity heatmap with feasibility frontier overlay (seminar Ex3)
- DSR calibration surface: 5×5 grid using net SR with per-slice moment re-estimation (seminar Ex4)
- Three-signal look-ahead bug hunt diagnostic (seminar Ex1)

### Key Caveats Established
- Purging's numerical impact is frequency-dependent: near-zero on monthly data with 1-month labels (IC delta = -0.0011), material on daily data with multi-day labels. The mechanism is universal; the effect size is setup-specific
- PBO < 0.5 does not confirm statistical significance — a model can persistently rank best among weak alternatives (low PBO) while producing a signal indistinguishable from zero (IC t < 1.96). Both PBO and t-statistics serve different purposes
- Non-normality is the dominant term in the DSR formula, not a footnote. Excess kurtosis of 4.23 inflates MinTRL by ~10x relative to Gaussian assumption. Most strategies with 3-5 year track records cannot confirm their Sharpe at 95% confidence
- DSR@M=10 = 0.504 (D3) vs. 0.414 (S6) on the same strategy under different cost specifications — the DEPLOY verdict is fragile and cost-assumption-dependent
- Survivorship bias inflates the gross Sharpe by 0.1-0.3 annual units, making DSR results an upper bound; survivorship-corrected DSR would be more pessimistic
- High turnover (139.9% monthly) is the multiplier that converts small per-trade costs into large annual drag. The same strategy is viable at large-cap spreads (2-5 bps) and unviable at mid-cap spreads (12-25 bps)
- The responsible-vs-naive Sharpe gap of 34% is at the favorable end of production experience (40-60% live degradation typical) because our cost assumptions are research-grade, not institutional-scale
- All IC t-statistics in Week 5 are below 1.96 — the signal is statistically insignificant at conventional confidence levels. This is the honest result for a monthly cross-sectional alpha model on 68 OOS months with 174 stocks

### Reusable Downstream
- Can assume: purged CV implementation and interpretation, PBO computation, DSR and MinTRL computation, transaction cost decomposition (spread + impact), feasibility frontier analysis, naive-vs-responsible backtest comparison, the seven backtest failure modes, multiple testing correction (BHY), cost-sensitivity grid methodology
- Can reference by name: `PurgedKFold` class, `TransactionCostModel` class, responsible backtest report format, DSR surface, feasibility frontier, the Week 4 long-short portfolio's performance characteristics (gross Sharpe 0.871, net tiered 0.624, turnover 139.9%, OOS 68 months)
- Don't re-teach: what purged CV does, why DSR penalizes non-normality, how transaction costs decompose, what PBO measures, the seven sins taxonomy, why naive and responsible backtests differ, what MinTRL means