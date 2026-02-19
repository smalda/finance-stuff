# Shared Data Reference

> **Consumers:** Expectations agent (Step 3, primary — for data plan
> decisions), Code agent (Step 4A — for data_setup.py), Observation
> agent (Step 5 — for data-quality context), Notebook agents (Step 7 —
> for translating shared data layer to standalone API calls). This is
> the factual inventory of what's available, what it enables, and where
> it lies.

---

## At a Glance

| Dataset | Source | Coverage | PIT status | Bias |
|---------|--------|----------|------------|------|
| Equity OHLCV | yfinance | 456 tickers, 2000–2025, daily | **Clean** | Survivorship (current constituents) |
| FF / Carhart factors | Ken French | 1926–2025, monthly + daily | **Clean** | None (gold standard) |
| KF sorted portfolios | Ken French | 1926–2025, monthly + daily | **Clean** | None (proper PIT sorts) |
| FRED market data | FRED | 2000–2025, varies | **Clean** | None (market-observed) |
| FRED macro series | FRED | 2000–2025, varies | **Revision risk** | Latest-revision, not real-time |
| Fundamentals | yfinance | ~455 tickers, ~4 yr history | **Contaminated** | PIT (reporting lag + restatements) |
| Options chains | yfinance | Current snapshot only | **Clean** | Not historical |
| SEC EDGAR filings | SEC | Recent 5 filings/ticker | **Clean** | As-originally-reported text; filing dates are real |
| Crypto prices | yfinance | 8–21 tokens, 2014–2025 | **Clean** | Short histories for many tokens |
| Synthetic LOB | Generated | Configurable | **N/A** | Synthetic — no real market dynamics |

---

## Equity Prices (OHLCV)

**What it is.** Daily Open, High, Low, Close (adjusted), and Volume for
456 tickers. Sourced from yfinance. Cached as `all_ohlcv.parquet`
(110 MB) and `all_prices.parquet` (19 MB, Close-only fast path).

**Universe.** Three concentric lists:
- `SP500_TICKERS` (503) — current S&P 500 constituents (Feb 2026)
- `SP400_TICKERS` (~150) — curated mid-cap stocks
- `ALL_EQUITY_TICKERS` = SP500 + SP400 (deduplicated)
- `ETF_TICKERS` (~70) — across sectors, factors, fixed income,
  international, commodity, volatility, REIT, thematic
- `DEMO_TICKERS` (20) — quick-loading subset for demos

After the completeness filter (>50% non-null days over 2000–2025),
456 of 511 attempted tickers survived. Dropped tickers are mostly
post-2010 IPOs (ABNB, COIN, DASH, PLTR, UBER, etc.).

**Access.** `load_sp500_prices(start, end, tickers)` → Close only.
`load_sp500_ohlcv(start, end, tickers, fields)` → full OHLCV with
MultiIndex columns (field, ticker).

**What it enables.**
- Return computation at any frequency (daily, weekly, monthly)
- Price-derived features: momentum (1m–12m), reversal, realized
  volatility, Amihud illiquidity, volume trends, beta, idiosyncratic
  volatility, max daily return, zero-trading-day fraction
- Cross-sectional analysis on ~200–450 stocks (depending on date range)
- Correlation/covariance estimation for portfolio construction
- Pairs trading and cointegration (stock-level prices)
- GARCH and volatility modeling

**Limitations.**
- **Survivorship bias.** The universe is today's S&P 500/400 projected
  backward. Companies that were delisted, went bankrupt, or were removed
  from the index are missing. This inflates backtest returns by an
  estimated 1–4% annually. Any cross-sectional result should note:
  `Universe: current S&P 500 — subject to survivorship bias.`
- **Adjusted close methodology.** yfinance applies split and dividend
  adjustments retroactively. This is standard (CRSP does the same), but
  the adjustment chain can differ from CRSP's methodology in edge cases.
  Not a meaningful concern for the course.
- **No intraday data.** Daily bars only — no tick, minute, or order book
  data. Microstructure analysis (Week 13) uses the synthetic LOB instead.
- **Data gaps.** Some tickers have missing days (holidays, halts,
  delistings). The completeness filter catches the worst cases, but
  2–5% missing days are normal for mid-caps.

---

## Fama-French & Carhart Factor Returns

**What it is.** Monthly and daily returns for the Fama-French 3-factor,
5-factor, and 6-factor models, plus the Carhart 4-factor model. Sourced
from Ken French's Data Library via the `getfactormodels` package.

**Factors available:**

| Model | Factors | Monthly from | Daily from |
|-------|---------|-------------|------------|
| FF3 | Mkt-RF, SMB, HML, RF | 1926-07 | 1926-07 |
| FF5 | Mkt-RF, SMB, HML, RMW, CMA, RF | 1963-07 | 1963-07 |
| FF6 | FF5 + UMD (momentum) | 1963-07 | 1963-07 |
| Carhart | Mkt-RF, SMB, HML, MOM | 1926-11 | 1926-11 |

**Access.** `load_ff_factors(model='3'|'5'|'6', frequency='M'|'D')`,
`load_carhart_factors(frequency='M'|'D')`.

**What it enables.**
- Time-series factor regressions (CAPM, FF3, FF5, Carhart)
- Factor risk attribution and alpha estimation
- Factor timing and factor momentum studies
- Risk-free rate series (RF column) for excess return computation
- Long-history analysis (98 years monthly for FF3)

**Why it's clean.** Constructed by Ken French's team from CRSP (prices)
and Compustat (fundamentals) with proper point-in-time methodology. The
accounting data uses the standard Fama-French lag convention: book equity
from fiscal year ending in calendar year t-1, matched with market equity
from June of year t, portfolios held July t to June t+1. This is the
gold standard for academic factor research.

**Limitations.**
- **Aggregate, not stock-level.** These are portfolio-level factor
  returns. You cannot decompose them into individual stock contributions
  or use them for stock selection directly.
- **Value-weighted.** The factor returns reflect cap-weighted portfolios.
  Equal-weighted factor returns (which often show stronger premia) are
  not provided by default.
- **Returns in percent.** FF factors use percent format (1.5 = 1.5%).
  Remember to divide by 100 when mixing with yfinance returns (which
  are in decimal form).

---

## Ken French Sorted Portfolios

**What it is.** Pre-sorted portfolio returns from Ken French's Data
Library. Monthly and daily returns for portfolios formed by sorting
stocks on various characteristics. 16 datasets cached.

**Datasets available:**

| Key | Description | Portfolios |
|-----|-------------|------------|
| `25_size_bm` | 5×5 double sort on size × book-to-market | 25 |
| `25_size_mom` | 5×5 double sort on size × prior 12-2 return | 25 |
| `25_size_inv` | 5×5 double sort on size × investment | 25 |
| `25_size_op` | 5×5 double sort on size × oper. profitability | 25 |
| `6_size_bm` | 2×3 sort on size × B/M (FF factor construction) | 6 |
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

**Access.** `load_ff_portfolios(name, frequency='M'|'D')`.
Returns in percent.

**What it enables.**
- **PIT-clean benchmarks for fundamental sorts.** The decile portfolios
  on B/M, E/P, CF/P, D/P, investment, and OP use the same economic
  concepts as yfinance fundamental features — but sorted with proper
  Compustat PIT methodology. Compare self-built sorts against these to
  measure PIT contamination.
- Factor model test assets (25 size/BM is the standard test)
- Industry analysis and industry-neutral factor construction
- Momentum strategy evaluation
- Long-short portfolio construction (top decile minus bottom decile)
- Cross-sectional dispersion studies

**Why it's powerful for PIT contrast.** The `10_dp` (dividend yield)
portfolios are a natural control — dividends are announced and paid in
real time, so PIT bias is near-zero. If the yfinance-vs-Ken-French gap
is small for D/P but large for B/M or E/P, that isolates the PIT effect
to accounting data specifically.

**Limitations.**
- **Portfolio-level only.** No stock-level membership or weights. You
  can see that "the top B/M decile returned X%" but not which stocks
  were in it.
- **Returns in percent** (same as FF factors — divide by 100).
- **US equities only.** NYSE/AMEX/NASDAQ universe. No international.
- **Value-weighted by default.** The first table in each CSV is
  value-weighted. Equal-weighted tables exist but are not parsed by
  the current loader.

---

## FRED Economic & Market Data

**What it is.** 60+ economic and market time series from the Federal
Reserve Economic Data (FRED) service. Downloaded via the public CSV
endpoint (no API key required). Each series cached individually.

**Access.** `load_fred_series(series_ids, start, end)`. Also accepts
ad-hoc series IDs not in the pre-cached list — they download and cache
on first call.

### Market-observed series (fully clean)

These are prices or rates observed in real time. Never revised.

| Category | Series | Notes |
|----------|--------|-------|
| **Treasury yields** (11) | DGS1MO through DGS30 | Full yield curve, daily. Zero revision risk. |
| **TIPS real yields** (4) | DFII5, DFII10, DFII20, DFII30 | Breakeven inflation = nominal − TIPS. Daily. |
| **Fed funds** | FEDFUNDS (monthly), DFF (daily) | Policy rate. |
| **VIX** | VIXCLS | CBOE implied vol index, daily. |
| **Credit spreads** (5) | BAMLH0A0HYM2 (HY OAS), BAMLC0A4CBBB (BBB), BAMLC0A0CM (IG), TEDRATE, DPRIME | Market-priced credit risk. Daily. |
| **FX rates** (5) | DTWEXBGS (dollar index), DEXJPUS, DEXUSEU, DEXCHUS, DEXUSUK | Spot rates, daily. |
| **Commodity prices** (5) | DCOILWTICO, DCOILBRENTEU, GOLDAMGBD228NLBM, DHHNGSP, PCOPPUSDM | Spot/fixing prices. |
| **Inflation expectations** (4) | T5YIE, T10YIE (breakevens), T5YIFR (forward), MICH (survey) | Market-derived except MICH. |
| **Financial conditions** (3) | NFCI, STLFSI2, DRTSCILM | Index values, not revised. |

**What these enable.** Yield curve modeling (bootstrapping, PCA on the
curve), credit cycle analysis, regime detection (VIX regime, yield curve
inversion → recession), macro factor features for alpha models, interest
rate sensitivity analysis, cross-asset correlation studies.

### Macro series (revision risk)

These are published by government agencies and revised after initial
release. FRED stores the **latest revision**, not the first print.

| Series | What | Revision severity |
|--------|------|-------------------|
| GDP, GDPC1 | Nominal/real GDP (quarterly) | **High** — advance, second, third estimates, plus annual revisions. GDP can revise by ±1pp. |
| CPIAUCSL, CPILFESL, PCEPI | Inflation measures (monthly) | **Low** — seasonal adjustment revisions are small (<0.1pp typically). |
| UNRATE | Unemployment rate (monthly) | **Low** — rarely revised materially. |
| PAYEMS | Nonfarm payrolls (monthly) | **Medium** — initial prints revised by ±50K regularly. Large annual benchmark revision. |
| INDPRO | Industrial production (monthly) | **Medium** — revised with about a 1-month lag, sometimes materially. |
| UMCSENT | Consumer sentiment (monthly) | **Low** — preliminary vs. final, small differences. |
| M2SL, WALCL | Money supply, Fed balance sheet | **Low** — minor revisions only. |
| USREC | NBER recession indicator (monthly) | **Special** — recession start/end dates are declared retroactively, months or years after the fact. The indicator is definitionally look-ahead. |
| HOUST, CSUSHPISA | Housing starts, home prices | **Low to medium.** |
| ICSA, JTSJOL, AWHAETP | Labor market (weekly/monthly) | **Low to medium.** |

**What macro series enable.** Macro regime features for alpha models,
economic cycle analysis, recession prediction, yield curve modeling with
macro context.

**The revision problem in practice.** For this course, FRED macro
revision risk is much less severe than fundamental PIT bias:
- Most monthly series revise by <0.1pp — statistically negligible
- GDP is the worst case but is quarterly (low frequency, fewer data
  points affected)
- Production use of FRED data requires the Philadelphia Fed's Real-Time
  Data Set — worth mentioning but not implementing in the course

When using FRED macro as features in predictive models, apply a
**1-month publication lag** as a conservative mitigation (analogous to
the +90 day fundamental lag but much shorter). Most monthly series are
published with a 1–3 week delay from the reference period.

**Special case: USREC.** The NBER recession indicator is backdated —
NBER declares recessions 6–18 months after they begin. Do NOT use
USREC as a predictive feature. Use it only as a labeling/evaluation
tool (e.g., "how did the signal perform in recessions?").

---

## Fundamentals

**What it is.** Balance sheet, income statement, cash flow, sector,
industry, market cap, shares outstanding, and valuation ratios for ~455
tickers. Sourced from yfinance `.balance_sheet`, `.income_stmt`,
`.cashflow`, `.info`.

**Access.** `load_sp500_fundamentals(tickers)` → dict with keys:
`balance_sheet`, `income_stmt`, `cashflow`, `sector`, `industry`,
`market_cap`, `shares`, `ratios`.

**Fields available:**
- **Balance sheet:** Stockholders Equity, Total Assets, Shares, Total
  Debt, Net Debt, Cash, Current/Non-Current Assets & Liabilities,
  Invested Capital, Goodwill, Working Capital (14 fields)
- **Income statement:** Revenue, Gross Profit, Operating Income, EBITDA,
  Net Income, Pretax Income, R&D, Cost of Revenue, SG&A, EPS (12 fields)
- **Cash flow:** Operating CF, CapEx, FCF, Dividends, Buybacks, Debt
  Issuance/Repayment (8 fields)
- **Ratios:** P/E, P/B, EV/EBITDA, EV/Revenue, ROE, ROA, D/E, current
  ratio, margins, growth rates, short interest, institutional holdings
  (28 fields) — **current snapshot only, not historical**

**What it enables.** Cross-sectional fundamental features (B/M, E/P,
profitability, investment, leverage), sector analysis, market-cap
weighting, value factor construction, earnings-based signal research.

### Why it's contaminated — the PIT problem

This is the **only dataset in shared/ with point-in-time contamination**.
Two distinct biases:

**1. Reporting-lag bias.** yfinance indexes fundamentals by fiscal
period end (e.g., Q4 ending Dec 31). But the 10-K filing arrives ~55
days later (median for S&P 500), with a tail to 90+ days. Using Dec 31
data for a Jan 1 prediction is look-ahead. The course applies a +90 day
conservative lag as default mitigation.

**2. Restatement bias.** Companies restate financials. The numbers
available today for FY2020 may differ from the original filing. yfinance
silently overwrites originals. A model trained on restated data sees
history no investor had. No mitigation exists with free data.

**3. Snapshot-only ratios.** The `ratios` sub-dict (P/E, P/B, ROE, etc.)
reflects *today's* values, not historical. Using current P/E as a
feature for a 2018 prediction is pure look-ahead. These ratios are
useful only for current-state analysis, not for backtesting.

### Required mitigations (from rigor.md)

- Load with `load_sp500_fundamentals(pit_lag_days=90)` — shifts all
  fiscal-period-end dates forward by 90 days at read time (cache is
  never modified).
- For fundamental-heavy sections: run lag sensitivity analysis
  (`pit_lag_days=0` vs `pit_lag_days=90`).
- Always state the mitigation in structured output.
- Restatement bias cannot be mitigated with structured yfinance data.
  SEC EDGAR filings contain as-originally-reported text (see SEC EDGAR
  section), but extracting structured values requires parsing.
  Acknowledge the residual bias when using yfinance fundamentals.
- See `rigor.md` §1.3 rule 4 and Appendix A for the engineering rules.

### Other limitations

- **~4 years of history.** yfinance typically returns 4–5 annual
  periods. Deeper history requires paid sources (Compustat, Refinitiv).
- **Coverage gaps.** Not all tickers have all fields. Operating Income
  is well-covered; some balance sheet items (Goodwill, Invested Capital)
  have 15–20% missing.
- **Quarterly data unreliable.** The course uses annual statements
  (`.balance_sheet`, `.income_stmt`) by default. Quarterly statements
  are available but have more coverage gaps and worse PIT contamination
  (10-Q filings lag less but occur 4x more often).

---

## Options Chains

**What it is.** Current-snapshot option chains for configurable tickers.
Calls and puts across the nearest N expiration dates. Sourced from
yfinance. Cached with a freshness check (re-downloads if >7 days old).

**Access.** `load_options_chain(tickers, near_expiries=4, max_age_days=7)`.

**Fields.** ticker, expiry, strike, type (call/put), lastPrice, bid,
ask, volume, openInterest, impliedVolatility, inTheMoney, spot price,
moneyness (S/K), time-to-expiry (days and years).

**What it enables.** Black-Scholes pricing with real market data, Greeks
computation and validation against market IVs, vol surface fitting
(SABR), put-call parity verification, option strategy payoff analysis.

**Limitations.**
- **Snapshot, not historical.** This is today's option chain. Historical
  option data requires paid sources (OptionMetrics, IVolatility). You
  cannot backtest option strategies.
- **Stale quotes.** Low-volume strikes may have lastPrice from hours or
  days ago. Use mid = (bid + ask) / 2 when bid/ask are available; fall
  back to lastPrice only for deep OTM options with no current quotes.
- **No Greeks from the source.** yfinance provides impliedVolatility but
  not Greeks. Compute them from BS (which is the teaching exercise).
- **Limited expiries.** Only the nearest N expiries are downloaded.
  Long-dated LEAPS are excluded by default.
- **Not reproducible.** Unlike every other dataset in shared/, options
  chains change on every download — different strikes, expiries, IVs,
  and spot prices. Two runs of the same notebook weeks apart produce
  entirely different data. Exercises must not reference specific
  contracts (e.g., "the AAPL 200 call expiring March 21") or hardcode
  expected numerical outputs. Design exercises around *relationships*
  (put-call parity, vol smile shape, Greeks sensitivity) that hold
  regardless of the specific snapshot. Pre-rendered notebook outputs
  for options sections will always diverge from a student's live run.

---

## SEC EDGAR Filings

**What it is.** Full text of 10-K and 10-Q filings from SEC EDGAR,
with real filing dates (actual SEC submission timestamps). Cached
per-ticker per filing type.

**Access.** `load_sec_filings(tickers, filing_type='10-K', max_filings=5)`.

**What it enables.** NLP analysis with genuine point-in-time text:
sentiment extraction, readability scoring, MD&A tone analysis, filing
length as complexity proxy, word embedding features — all tied to the
actual date the market could see the filing.

**Why it's PIT-clean.** The filing date from EDGAR is when the document
was submitted to the SEC. The text is the original filing (not amended).
This is one of the most PIT-clean data sources in the shared layer.

**Restatement-free.** Because the text is the original filing, any
financial values embedded in it reflect what was reported at the time —
not later restatements. This makes EDGAR the only free source of
as-originally-reported fundamental data. Extracting structured values
(revenue, net income, etc.) from the text requires parsing (XBRL inline
tags or regex on financial tables), which is non-trivial but feasible.
yfinance fundamentals, by contrast, silently reflect the latest
restatement.

**Limitations.**
- **Amendments.** Companies can file 10-K/A (amended annual reports).
  The submissions API returns the original filing, not amendments, but
  amendment existence is not flagged.
- **HTML artifacts.** The text is HTML-stripped, but formatting artifacts
  (table structures, page breaks) may remain.
- **Text truncation.** Filings are truncated to ~500K characters to keep
  cache size reasonable. Very long filings may lose appendices/exhibits.
- **5 filings per ticker by default.** Expand `max_filings` for deeper
  history, at the cost of download time (SEC rate-limits to 10 req/sec).
- **CIK mapping.** Some tickers may not map to a CIK (name changes,
  recent IPOs). These return empty results silently.

---

## Crypto Prices

**What it is.** Daily close prices for crypto/DeFi tokens. Sourced from
yfinance. Two tiers of tokens:

- **Layer 1 / major:** BTC, ETH, SOL, BNB, ADA, AVAX, DOT, MATIC,
  ATOM, NEAR
- **DeFi governance:** UNI, AAVE, LINK, CRV, MKR, COMP, SNX, SUSHI,
  LDO
- **Layer 2:** ARB, OP

**Access.** `load_crypto_prices(start, end, tickers)`. After the
completeness filter, ~8–21 tokens have data (many DeFi tokens launched
post-2020 with short histories).

**What it enables.** DeFi market analysis (Week 18), crypto-equity
correlation studies, AMM dynamics context, volatility comparison
(crypto vs. equities).

**Limitations.**
- **Short histories.** BTC from 2014, ETH from 2015, most DeFi tokens
  from 2020+. Not enough for long-horizon time-series analysis.
- **24/7 markets.** Crypto trades on weekends — daily close is midnight
  UTC, not a market close. This affects correlation studies with
  equity markets (which have weekends off).
- **No on-chain data.** Prices only — no TVL, trading volume by DEX,
  gas fees, or protocol-level metrics.
- **Exchange risk.** yfinance aggregates from a single exchange
  (typically CoinGecko-sourced). Prices may differ from other exchanges.

---

## Synthetic Limit Order Book

**What it is.** Generated LOB snapshots with configurable parameters.
Not downloaded — produced by `generate_synthetic_lob()`. Models
mean-reverting mid-price, bid-ask spread dynamics, exponentially
decaying depth, autocorrelated order flow, and simulated trades.

**Access.** `generate_synthetic_lob(n_levels=10, n_snapshots=10_000,
mid_price=100.0, tick_size=0.01, avg_spread_ticks=3.0, seed=42)`.

**What it enables.** Limit order book visualization, bid-ask spread
analysis, OFI (order flow imbalance) computation, the OFI-price
relationship, Almgren-Chriss execution simulation.

**Limitations.**
- **Synthetic.** No real market dynamics — the generator uses simple
  stochastic models (Poisson depth, AR(1) OFI). Real LOBs have
  strategic behavior, hidden orders, and queue priority effects.
- **Single day, single asset.** The generator produces one continuous
  session. Multi-day or multi-asset LOB analysis requires either paid
  data (LOBSTER, ITCH) or extending the generator.
- **No strategic agents.** Real LOBs feature market makers, HFT, and
  institutional flow. The synthetic version has random flow only.
- **Will likely need reimplementation.** The current generator is a
  minimal placeholder. Week 13 (Microstructure & Execution) will
  probably need a more sophisticated generator tailored to its specific
  exercises — e.g., configurable market impact, multi-day sessions,
  realistic queue dynamics. Treat the shared version as a starting point
  for demos, not as the production generator for that week.

---

## Data Quality Summary by Analysis Type

A quick reference for which data to use, and what to watch for.

| Analysis | Primary data | PIT-clean? | Key limitation |
|----------|-------------|------------|----------------|
| Factor regressions (CAPM, FF3/5) | FF factors + KF portfolios | Yes | Portfolio-level only |
| Cross-sectional alpha (price features) | Equity OHLCV | Yes | Survivorship bias in universe |
| Cross-sectional alpha (fundamental features) | Fundamentals | **No** | +90 day lag required |
| Value/earnings factor construction | Fundamentals + KF `10_bm`/`10_ep` as benchmark | **Partial** | Compare to KF for PIT calibration |
| Volatility modeling (GARCH) | Equity OHLCV | Yes | Daily only, no intraday |
| Yield curve / fixed income | FRED Treasury yields | Yes | Nominal yields only (add TIPS for real) |
| Credit cycle analysis | FRED credit spreads | Yes | Spread = market price, not default rate |
| NLP / text analysis | SEC EDGAR filings | Yes | 5 filings/ticker default, text truncated |
| Macro regime features | FRED macro series | **Revision risk** | Use 1-month pub lag; don't use USREC as feature |
| Derivatives pricing | Options chains | Yes | Current snapshot only, no backtest |
| Microstructure | Synthetic LOB | N/A | Not real market data |
| Portfolio construction | Any of the above as inputs | Depends on inputs | — |
| Backtesting | Equity prices + factors | Yes (prices) | Survivorship inflates returns |
| Crypto / DeFi | Crypto prices | Yes | Short histories, no on-chain data |

---

## Universe Bias Inventory

Every cross-sectional analysis in this course inherits universe bias.
Document which bias applies in structured output.

| Universe | Bias | Severity | Mitigation |
|----------|------|----------|------------|
| `SP500_TICKERS` (current) | Survivorship: today's constituents projected backward. Excludes delistings, bankruptcies, demotions. | **High** for returns (1–4% annual inflation). **Moderate** for factor structure. | Note in output. Compare to KF portfolios (full NYSE/AMEX/NASDAQ). |
| `SP400_TICKERS` (current) | Same survivorship bias as SP500, plus size-band bias (mid-caps that grew into SP500 or shrank out). | **Moderate.** | Combine with SP500 for broader coverage. |
| `ALL_EQUITY_TICKERS` | Broader than SP500 alone, but still current-day only. | **Moderate.** Somewhat reduced by broader coverage. | Best available free universe. |
| Ken French portfolios | Full NYSE/AMEX/NASDAQ including delistings. Proper PIT. | **None** (this is the clean benchmark). | Use as ground truth. |
| `DEMO_TICKERS` (20) | Extreme selection bias. All mega-caps, all survivors. | **Very high.** No cross-sectional power. | Demo only — never for statistical conclusions. |

---

## Choosing Between PIT-Contaminated and PIT-Clean Data

When an analysis involves fundamental or accounting concepts, the
available data sources have different PIT properties. This section maps
the tradeoffs to help make the source selection decision explicitly.

### The three source tiers

| Tier | Source | PIT status | Structured? | Stock-level? | Restatement-free? |
|------|--------|-----------|-------------|-------------|-------------------|
| **1. PIT-clean, structured, portfolio-level** | Ken French sorted portfolios (`10_bm`, `10_ep`, `10_op`, `10_inv`, etc.) | Clean — Compustat PIT methodology | Yes (returns) | No (portfolio deciles only) | Yes (proper PIT construction) |
| **2. PIT-clean, unstructured, stock-level** | SEC EDGAR filings (`load_sec_filings()`) | Clean — real filing dates | No (raw text) | Yes | Yes (original filing text) |
| **3. PIT-contaminated, structured, stock-level** | yfinance fundamentals (`load_sp500_fundamentals()`) | Contaminated — fiscal period end dates | Yes (financial statement fields) | Yes | **No** (latest restatement) |

No single source has all three desirable properties (PIT-clean,
structured, stock-level). Every choice is a tradeoff.

### Decision guide

**"I need stock-level fundamental features for an ML model"**
→ Use **Tier 3** (yfinance fundamentals) with `pit_lag_days=90`. This is
the only source with structured financial statement fields at the stock
level. See the Fundamentals section above for required mitigations.

**"I need to validate that my fundamental-based results aren't inflated
by PIT bias"**
→ Compare against **Tier 1** (Ken French sorted portfolios). The KF
`10_bm`, `10_ep`, `10_dp` decile portfolios use the same economic
concepts but with proper Compustat PIT sorts. If your self-built factor
returns diverge significantly from KF, PIT contamination is a likely
contributor. The `10_dp` (dividend yield) portfolios are a natural
control — dividends are announced in real time, so PIT bias is near-zero.

**"I need text-based features with genuine filing dates"**
→ Use **Tier 2** (SEC EDGAR). Filing dates from EDGAR are the actual SEC
submission timestamps. NLP features (sentiment, readability, MD&A tone)
are naturally PIT-clean when anchored to the filing date.

**"I need to know when a specific financial statement was actually
available to the market"**
→ Combine **Tier 2** filing dates with **Tier 3** financial values. Use
`load_sec_filings()` to get real filing dates, then align with yfinance
fundamental values from the corresponding fiscal period. This eliminates
reporting-lag bias precisely but does not fix restatement bias. Use this
when the +90 day lag is too conservative (e.g., for fast-filing
large-caps) or when you need per-ticker filing dates rather than a
blanket lag.

### What each tier cannot do

| Tier | Cannot do |
|------|-----------|
| **KF portfolios** | Stock selection, ML features, cross-sectional regressions (no stock-level data) |
| **SEC EDGAR** | Direct financial statement features without NLP extraction (text only) |
| **yfinance fundamentals** | Guarantee no look-ahead without mitigation; provide as-originally-reported values |

