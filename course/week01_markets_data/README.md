# Week 1: How Markets Work & Financial Data

**Band:** 1
**Implementability:** HIGH

## Learning Objectives

After this week, students will be able to:

1. Explain how a trade executes — from order submission through the matching engine to settlement — including the roles of exchanges, market makers, and clearinghouses
2. Describe the structure of a limit order book and compute bid-ask spreads from order book snapshots
3. Identify survivorship bias, look-ahead bias, and corporate action distortions in financial datasets, and explain why each corrupts model training
4. Build a data pipeline that downloads, cleans, validates, and stores multi-asset financial data using Polars and Parquet
5. Distinguish between adjusted and unadjusted prices, explain when each is appropriate, and compute adjustment factors from raw data
6. Evaluate data quality systematically — detecting missing days, stale prices, outlier returns, and adjustment errors — and quantify their impact on downstream analysis
7. Describe how commodities, FX, and crypto markets differ structurally from equities (at a landscape level)

## Prerequisites

- **Required:** None (entry point)
- **Assumed ML knowledge:** Familiarity with tabular data, basic pandas/NumPy, comfort reading and writing Python
- **Prerequisite explanations in this week:** None — all financial concepts are introduced from scratch

## Narrative Arc

Financial markets look simple from the outside. Prices go up and down. You want to predict which way. But between "I want to buy this stock" and "I own this stock," a remarkable amount of machinery operates: exchanges match orders in microseconds, market makers provide liquidity in exchange for the bid-ask spread, regulators enforce rules about how orders are routed — and every layer of this machinery leaves fingerprints in the data you'll eventually feed to a model.

This week's central question is: **"What is financial data, really?"** The ML expert's instinct is to treat it as a feature matrix — rows of dates, columns of prices. But every number in that matrix carries hidden assumptions. An adjusted close price embeds decisions about how to handle splits and dividends. A "complete" dataset of S&P 500 stocks is missing the companies that failed. A daily bar compresses thousands of individual trades into four numbers, erasing the market microstructure that generated them. If you don't understand where these numbers come from and what's been done to them, your model will learn from ghosts.

The week resolves this tension by building understanding from the ground up: first how markets generate data (the mechanics of trading), then how that data reaches you (aggregation, adjustment, vendor pipelines), and finally how the data can betray you (survivorship bias, look-ahead bias, corporate action distortions). By the end, students should look at a CSV of stock prices with healthy skepticism — and know exactly which questions to ask before feeding it to a model.

## Lecture Outline

### Section 1: How a Trade Actually Happens

- **Hook:** "You click 'buy AAPL' on your phone. What happens in the next 50 microseconds? Spoiler: your order probably never touches the NYSE."
- **Core concepts:** Order submission → broker → order router → execution venue (exchange or dark pool or wholesaler). The matching engine: price-time priority. Order types: market orders (immediate, price-uncertain), limit orders (price-certain, execution-uncertain), stop orders (conditional triggers). The concept of "best execution" and why your broker's routing decision matters.
- **Key demonstration:** Trace a single AAPL market buy order through the system using a diagram/flowchart. Show the decision tree: exchange vs. dark pool vs. wholesaler. Annotate each node with approximate latencies and costs.
- **Acceptance criteria:**
  - Flowchart renders with ≥ 4 nodes showing distinct stages (order submission → routing → venue → execution)
  - At least 3 venue types shown (exchange, dark pool, wholesaler)
  - Latency annotations are order-of-magnitude correct (microseconds for matching, milliseconds for routing)
- **Bridge:** "Every order that enters the system lands in a structure called the order book. That's where price discovery actually happens."

### Section 2: The Order Book — Where Price Discovery Lives

- **Hook:** "The price you see on Google Finance is not 'the price.' It's the last trade price — a historical artifact. The real action is in the order book, which most retail investors never see."
- **Core concepts:** The limit order book (LOB): bids stacked below, asks stacked above, the spread in between. L1 data (best bid/ask) vs. L2 data (full depth). Bid-ask spread as the market's "admission fee" — you pay it every time you trade. Mid-price as the fair value estimate. How the LOB generates trades: when a market order arrives, it "eats" resting limit orders.
- **Key demonstration:** Build and visualize a synthetic order book in Python — bids on the left, asks on the right, volumes as horizontal bars. Compute the spread. Show what happens when a large market buy order arrives (it walks up the ask side, widening the spread temporarily). This is a simulation, not real LOB data — real LOB data comes in Week 13.
- **Acceptance criteria:**
  - Order book has ≥ 5 price levels on each side (bid and ask)
  - Bid-ask spread > 0 and visible in visualization
  - After large market buy: top-of-book ask price increases (demonstrates walking the book)
  - Spread widens after the large order vs. before
- **"Did you know?" moment:** The SEC's November 2025 round-lot reform replaced the fixed 100-share round lot with price-tiered lots (stocks above $250 get smaller lot sizes: 40, 10, or 1 share). This only affects about 200 high-priced stocks, but it changes what counts as "top of book" for those names. Starting May 2026, SIPs will disseminate odd-lot quotes priced at or better than the NBBO — a big deal for retail investors who trade in odd lots.
- **Bridge:** "The spread is the market maker's compensation. But who are market makers, and why do they exist?"

### Section 3: Market Makers — The Invisible Infrastructure

- **Hook:** "Citadel Securities — one company — executes about 25% of all US equity volume. If they shut down for a day, the US stock market would feel it."
- **Core concepts:** Market makers provide liquidity by continuously quoting bids and asks. They earn the spread but face adverse selection risk (trading against informed traders). The maker-taker model: exchanges pay rebates to limit orders (makers) and charge fees to market orders (takers). Payment for Order Flow (PFOF): wholesalers pay brokers for retail order flow, then internalize it — retail orders are valuable because they're uninformed. The EU banned PFOF under MiFID III in 2025; the US hasn't. This creates a useful contrast for understanding how market structure is a regulatory choice, not a law of nature.
- **Key demonstration:** Compute a market maker's theoretical P&L from a simple spread-capture strategy: buy at bid, sell at ask, repeat. Show how adverse selection (large informed orders that move the price against you) eats into profits. Use a simple simulation with random vs. informed order flow.
- **Acceptance criteria:**
  - Market maker P&L is positive (profitable) with uninformed (random) order flow
  - Market maker P&L is reduced or negative with informed order flow
  - P&L difference between uninformed vs. informed scenarios is statistically visible (not noise)
  - Simulation runs ≥ 100 trades for stable results
- **Bridge:** "Market makers, exchanges, and brokers together produce a torrent of data — NYSE alone generates roughly 1 TB of message data per day. Nobody builds ML models on raw ticks. So how does that data get turned into something usable?"

### Section 4: From Tick Data to OHLCV Bars — The Aggregation Pipeline

- **Hook:** "A single trading day for AAPL generates tens of thousands of individual trades. You'll work with 4 numbers: Open, High, Low, Close. What gets lost in that compression — and does it matter?"
- **Core concepts:** Tick data (every trade) → time bars (OHLCV at fixed intervals: 1min, 5min, daily). What each field means: Open is the first trade, High/Low are extremes, Close is the last trade, Volume is total shares. Adjusted Close: a synthetic field that retroactively modifies history to account for splits and dividends (more on this in Section 6). Alternative bar types exist — volume bars, dollar bars, tick bars — that we'll revisit in later weeks.
- **Key demonstration:** Download daily OHLCV data for 5 tickers (AAPL, TSLA, JPM, SPY, GE) using yfinance. Examine the DataFrame structure. Show the columns, dtypes, index. Compute simple daily returns from Close prices: `r_t = (P_t - P_{t-1}) / P_{t-1}`. Plot the price series and return series side by side. Note the stylized fact visible even in daily data: returns are roughly symmetric and mean-zero, prices are not.
- **Acceptance criteria:**
  - DataFrame has ≥ 2500 rows per ticker (10+ years of daily data)
  - All 5 expected columns present (Open, High, Low, Close, Volume)
  - Mean daily return within ±0.002 for each ticker (approximately mean-zero)
  - Price series and return series are plotted side-by-side with visible contrast (trending vs. mean-reverting)
- **Bridge:** "This data looks clean and tidy. It is not. The next two sections show how financial data lies to you."

### Section 5: The Ghost Stocks — Survivorship Bias

- **Hook:** "If you download today's S&P 500 constituents and backtest a strategy to 2010, you'd miss every company that went bankrupt, was acquired, or was delisted in the last 15 years. Your backtest would only include survivors — the stocks that, by definition, didn't fail."
- **Core concepts:** Survivorship bias: using a current-day universe and projecting it backward. The result: inflated returns (you exclude the losers), understated risk (you exclude the blow-ups), and overstated strategy performance. Look-ahead bias: using information at time T that wouldn't have been available until time T+k (e.g., using final restated earnings instead of initially reported earnings). Point-in-time databases: the solution — they record what was known when, not what's known now. Professional datasets (CRSP, Compustat via WRDS) handle this; yfinance does not.
- **Key demonstration:** A small, concrete thought experiment with 10 simulated stocks: 7 survive, 3 go bankrupt. Compute the average return of survivors vs. the average return of the full universe. Show the gap. Then mention the empirical magnitude: studies consistently find survivorship bias inflates equity backtest returns by 1-4% per year, which over a 14-year backtest compounds to 15-70% cumulative overstatement. Students will measure this themselves on real data in the seminar.
- **Acceptance criteria:**
  - Simulation uses exactly 10 stocks: 7 survivors, 3 bankruptcies
  - Survivor-only portfolio return > full-universe portfolio return (bias is always positive)
  - Gap between survivor and full-universe returns is > 1% annualized
  - Bankrupt stocks reach $0 or near-zero — they're not just "underperformers"
- **Bridge:** "Even with the right stock universe, the numbers can still lie. The next silent data corruptor: corporate actions."

### Section 6: Corporate Actions — The Silent Data Corruptors

- **Hook:** "Apple's stock 'dropped' from $700 to $100 in June 2014. Nobody panicked — it was a 7-for-1 split. But if your model sees a 85% single-day loss in the training data, it will learn very wrong things."
- **Core concepts:** Stock splits (forward and reverse), dividends (cash and stock), mergers and spinoffs. What "adjusted" means: the data vendor retroactively modifies ALL historical prices so that the series is continuous and returns are correct across corporate action dates. The adjustment factor and how it accumulates over time. The critical rule: **always use adjusted prices for return computation, and unadjusted prices for level analysis** (e.g., comparing to strike prices, computing dollar volumes). When adjustment goes wrong: some vendors update adjustments with a lag, some use different adjustment bases (dividend-adjusted vs. split-only).
- **Key demonstration:** Plot AAPL's unadjusted Close vs. Adjusted Close from 2010 to present. The 2014 7:1 split creates a dramatic visual discontinuity in the unadjusted series. Compute returns from each: the unadjusted series shows an 85% "crash" on split day; the adjusted series shows a normal day. Lesson: one wrong column choice = catastrophically wrong training data.
- **Acceptance criteria:**
  - Reconstructed nominal AAPL price on 2014-06-06 > $600 (pre-split)
  - Reconstructed nominal AAPL price on 2014-06-09 < $100 (post-split)
  - Nominal daily return on split date < -80%
  - Adjusted daily return on split date within ±5% (normal trading day)
  - Plot shows visually dramatic discontinuity in nominal line vs. smooth adjusted line
  - NOTE: yfinance's `Close` with `auto_adjust=False` is already split-adjusted. To show truly unadjusted (nominal) prices, reverse split adjustments using `yf.Ticker().splits` data.
- **Bridge:** "You now know the traps. Time to build a data pipeline that handles them — and to meet the tools that the industry is actually using."

### Section 7: The Modern Financial Data Stack

- **Hook:** "Five years ago, 'financial data' meant a Bloomberg terminal and an Excel spreadsheet. Today it means Python, Parquet files, and APIs. The tooling has changed more than the finance."
- **Core concepts:** Data sources: yfinance (free, educational — unreliable for production), CRSP/Compustat via WRDS (academic gold standard, survivorship-free, point-in-time), Databento/Polygon.io (API-first professional vendors), Bloomberg (legacy terminal, still dominant at banks). Storage formats: CSV (human-readable, slow, bloated) vs. Parquet (columnar, compressed, 5-10x faster reads). Processing: pandas (established, universal) vs. Polars (newer, faster, memory-efficient — the industry is migrating for new projects). The practical recommendation: learn both, prefer Polars for new work, use Parquet always.
- **Key demonstration:** Download 15 years of daily data for 50 stocks via yfinance. Save to CSV and to Parquet. Read both back with pandas and with Polars. Benchmark the read/write times. Polars + Parquet should be visibly faster. Show the file size difference (Parquet will be 3-5x smaller than CSV). This is the data pipeline students will build properly in the homework.
- **Acceptance criteria:**
  - Data downloaded for ≥ 50 tickers spanning ≥ 10 years
  - Parquet file size < 50% of CSV file size (should be ~3-5x smaller)
  - Polars read time < pandas read time for the same format (should be 2-5x faster)
  - Benchmark table shows all 4 combinations: CSV+pandas, CSV+Polars, Parquet+pandas, Parquet+Polars
- **"Did you know?" moment:** UChicago's "Full Stack Quantitative Finance" course (FINM 32900, Winter 2026) teaches an entire 9-week module on data engineering tooling — Git workflows, WRDS/SQL access, build automation. Their canonical dataset list (CRSP, Compustat, FINRA TRACE, OptionMetrics, EDGAR, NYSE HF data, Treasury auctions) is what you'd encounter at a real fund. We're complementary: they teach the engineering pipeline; we teach the domain knowledge and ML.

### Closing

- **Summary table:** Key concepts (order book, bid-ask spread, survivorship bias, look-ahead bias, corporate actions, adjusted prices, Parquet) with one-line definitions
- **Asset class sidebar (MENTIONED ABSTRACTLY):** Commodities (physical delivery, contango/backwardation, seasonality), FX (24h, decentralized, spot vs. forward, carry trade), crypto (24/7, no closing prices, no corporate actions but hard forks). Each gets 2-3 sentences of landscape awareness — not instruction.
- **KDB/Q sidebar (MENTIONED ABSTRACTLY):** "You may encounter KDB/Q at legacy HFT firms. It's a columnar time-series database optimized for tick data. It's being steadily replaced by Polars, DuckDB, and Parquet-based stacks. If you see a job listing requiring it, that's what it is."
- **Bridge to Week 2:** "You now have clean, stored data. Next question: what statistical properties do financial time series have? Spoiler — they violate almost every assumption your ML models were built on. Stationarity, normality, independence — all broken. Week 2 shows how, and what to do about it."
- **Suggested reading with annotations**

## Seminar Outline

### Exercise 1: The Survivorship Trap — Measuring Ghost Returns

- **Task type:** Guided Discovery
- **The question:** How much does survivorship bias inflate equity backtest returns? Is it 0.5%/year? 2%/year? 5%/year?
- **Data needed:** Current S&P 500 constituent list (scrape from Wikipedia), historical daily prices from yfinance (2010-present)
- **Tasks:**
    1. Scrape the current S&P 500 ticker list from Wikipedia
    2. Download daily adjusted close prices for all ~500 tickers from 2010 to present via yfinance
    3. Identify which tickers have data starting significantly later than 2010 (these are late entrants — companies that were added to the index AFTER 2010, often because they succeeded)
    4. Compute an equal-weight portfolio return using ALL current constituents (the "survivorship-biased" portfolio — it includes future winners and excludes past losers)
    5. Compute an equal-weight portfolio return using ONLY tickers with complete data from 2010 (a rough "survivors-only" portfolio — still biased but less so)
    6. Estimate the annual survivorship bias premium: how much higher is the biased portfolio's annualized return?
- **Expected insight:** The bias is not trivial. Typical results show 1-4% per year of excess return from survivorship bias alone. Over a 14-year backtest, that compounds to a 15-70% cumulative overstatement. Students should realize: if your backtest shows 15% annualized return, 2-4 of those percentage points might be ghosts. This is the single most common data error in amateur quantitative research, and it silently inflates every metric — Sharpe ratio, max drawdown, hit rate.
- **Acceptance criteria:**
  - ≥ 400 tickers successfully downloaded (some S&P 500 tickers may fail in yfinance)
  - At least 20 tickers identified as having data starting after 2011 (late entrants)
  - Survivorship-biased portfolio annualized return > survivors-only portfolio return
  - Annual bias magnitude > 0.5% (should be 1-4%)
  - Cumulative bias over the full period > 5%

### Exercise 2: Corporate Action Forensics — Splits, Dividends, and the Slow Drift

- **Task type:** Guided Discovery
- **The question:** The lecture showed the dramatic case (AAPL's 7:1 split). But most corporate action damage is subtle, not dramatic. How much do dividends distort historical prices over a decade, and can you detect corporate actions programmatically from price data alone?
- **Data needed:** Daily Close and Adjusted Close from yfinance for 5 tickers with known corporate actions: AAPL (7:1 split Jun 2014, 4:1 split Aug 2020), TSLA (5:1 split Aug 2020, 3:1 split Aug 2022), GE (1:8 reverse split Aug 2021), NVDA (4:1 split Jul 2021, 10:1 split Jun 2024), JNJ (steady dividend payer, no recent splits — the "slow drift" example)
- **Tasks:**
    1. For each ticker, compute the adjustment factor over time: `adj_factor = Adj_Close / Close`
    2. Plot the adjustment factor for all 5 tickers on the same chart. Identify where splits create step-function jumps and where dividends create a slow, steady downward drift
    3. Build a simple split detector: flag dates where the adjustment factor changes by more than 10% in a single day. Cross-reference with known split dates. How accurate is your detector?
    4. For JNJ (no splits, regular dividends): compute the cumulative adjustment drift over 10 years. How much do unadjusted and adjusted total returns diverge?
    5. Compute daily returns using Close vs. Adjusted Close for all 5 tickers. Measure the return error (RMSE, max absolute error). On what dates is the error catastrophic vs. negligible?
- **Expected insight:** Splits are easy to detect (step changes in adjustment factor > 10%). Dividends are insidious — JNJ's 2-3% annual dividend creates a ~25-30% cumulative adjustment drift over 10 years that a naive analysis would completely miss. For a stock like JNJ, using unadjusted prices understates total return by ~25% over a decade. The GE reverse split (1:8) is interesting because it makes the pre-split price look 8x higher — a ML model trained on unadjusted data would see GE as a catastrophic decline when it was actually a restructuring.
- **Acceptance criteria:**
  - Adjustment factor plot shows visible step-function jumps for AAPL, TSLA, GE, NVDA at known split dates
  - JNJ adjustment factor shows smooth downward drift (no step jumps)
  - Split detector catches ≥ 90% of known splits in the 5-ticker set (with < 2 false positives)
  - JNJ cumulative adjustment drift > 15% over 10 years
  - Return RMSE between Close-based and Adj Close-based returns > 0 for all tickers with corporate actions
  - Max absolute return error on split dates > 50% (catastrophic, not subtle)

### Exercise 3: Data Quality Gauntlet — How Clean Is "Clean" Data?

- **Task type:** Skill Building
- **The question:** You downloaded data from a reputable free source (yfinance) for 20 diverse stocks. How many data quality issues can you find — and what do they cost a simple strategy?
- **Data needed:** Daily OHLCV from yfinance for 20 tickers spanning large-cap, mid-cap, small-cap, and ETFs (e.g., AAPL, MSFT, JPM, TSLA, SPY, QQQ, IWM, XLE, GE, META, AMZN, GOOG, BRK-B, V, UNH, PFE, INTC, BA, DIS, NFLX) — 10 years of history
- **Tasks:**
    1. **Missing data audit:** For each ticker, count the number of expected trading days (using NYSE calendar) vs. actual rows. Report the completeness ratio. Identify any gaps longer than 5 consecutive trading days and investigate (delistings? suspensions? data errors?)
    2. **Stale price detection:** Find consecutive days with identical Close prices (to 4 decimal places). How many occurrences per ticker? Are they legitimate (penny stocks, illiquid names) or data errors?
    3. **Outlier return detection:** Flag daily returns exceeding ±15%. For each flagged date, determine whether the return is genuine (COVID crash, earnings, FDA approval) or a data artifact (split not adjusted, erroneous price). Build a simple classifier: genuine vs. artifact
    4. **Consistency checks:** Verify that High ≥ max(Open, Close) and Low ≤ min(Open, Close) for every row. Count violations. Verify that Volume > 0 on all trading days. Check that Adjusted Close ≤ Close when cumulative dividends are positive
    5. **Impact assessment:** Take a simple buy-and-hold strategy on each ticker. Compute the return using raw data vs. data after your quality fixes. How much does the return estimate change for the worst-case ticker?
- **Expected insight:** Even yfinance data for major S&P 500 stocks has quality issues — missing days around delistings, occasional stale prices, rare OHLC consistency violations. The issues are MORE frequent for smaller, less liquid names. The impact on a simple buy-and-hold is usually small (<0.5% annualized) for large-caps but can be material (>2%) for names with messy corporate action histories. The takeaway: automated quality checks are not optional — they're the foundation of any research pipeline.
- **Acceptance criteria:**
  - All 20 tickers downloaded with ≥ 2000 rows each
  - Completeness ratio computed for all tickers; at least one ticker has ratio < 100%
  - Stale price detection finds ≥ 1 instance across the 20-ticker set
  - Outlier return detection flags ≥ 5 dates across all tickers (March 2020 COVID crash should be among them)
  - OHLC consistency check runs on all rows; violation count reported per ticker
  - Impact assessment shows return difference between raw and cleaned data for at least the worst-case ticker

## Homework Outline

### Mission Framing

You've been hired as the first data engineer at a small quantitative fund. There are three portfolio managers and a handful of researchers. They all need data. Right now, everyone downloads their own CSVs, cleans them (or doesn't), and stores them on their laptops. Your job: build the fund's data pipeline — the single source of truth that every analyst and model will depend on.

If the data is wrong, every model downstream is wrong. Every backtest is wrong. Every risk report is wrong. This isn't a software engineering exercise dressed up as finance — it's the actual first task a new quant data engineer faces. The pipeline you build this week is the foundation for every homework in this course. You'll reuse it in Weeks 2-6 and extend it further in Week 13.

### Deliverables

1. **A `FinancialDataLoader` class**
   - **Task type:** Construction
   - Handles the complete data lifecycle:
    - **Download:** Fetch daily OHLCV data for a configurable list of tickers and date range via yfinance. Handle API failures gracefully (retries, partial downloads, informative error messages). Support downloading at least 50 tickers in a single call.
    - **Clean:** Detect and handle missing trading days (forward-fill with a configurable limit, flag gaps > N days), remove weekends/holidays, handle timezone alignment. Detect and flag stale prices.
    - **Validate:** Automated quality checks — OHLC consistency, volume positivity, return outlier detection (flag but don't remove), adjustment factor continuity. Produce a per-ticker quality score.
    - **Store:** Save to Parquet format using Polars. Support both wide (tickers as columns) and long (ticker as a column) formats. Include metadata (download date, source, quality scores) as Parquet metadata or a sidecar JSON.
    - **Load:** Read back from Parquet efficiently. Support filtering by ticker, date range, and minimum quality score.
   - **Acceptance criteria:**
    - Downloads ≥ 50 tickers without crashing (gracefully handles any that fail)
    - Forward-fill handles gaps ≤ configurable limit; gaps > limit are flagged, not silently filled
    - Quality score computed for each ticker (completeness, OHLC consistency, stale price count)
    - Parquet output round-trips without data loss (read back == original, within float precision)
    - Load method correctly filters by ticker subset and date range
    - Class is instantiable with different ticker lists and date ranges (not hardcoded)

2. **A data quality report**
   - **Task type:** Skill Building
   - For the downloaded universe:
    - Per-ticker metrics: completeness ratio, number of stale price days, number of OHLC violations, number of detected corporate actions (from adjustment factor jumps), overall quality grade (A/B/C/D)
    - Universe-level summary: how many tickers pass a "production quality" bar (e.g., >99% complete, zero OHLC violations), distribution of quality grades
    - This should be structured as a template with clear headings that students fill in based on what they find — not a pre-written essay
   - **Acceptance criteria:**
    - Per-ticker metrics computed for all downloaded tickers (not a subset)
    - Quality grades assigned with clear, documented thresholds (not arbitrary)
    - At least one ticker receives a grade below A (data is imperfect — the report should surface this)
    - Universe summary includes a count/percentage at each grade level

3. **A storage format benchmark**
   - **Task type:** Skill Building
    - Save the same dataset to CSV and Parquet. Read back with pandas and with Polars (4 combinations: CSV+pandas, CSV+Polars, Parquet+pandas, Parquet+Polars)
    - Report: file sizes, write times, read times
    - Brief analysis: when does the format/library choice matter? (Answer: it matters more as data grows — show the scaling behavior if possible by varying the number of tickers)
   - **Acceptance criteria:**
    - All 4 combinations (CSV+pandas, CSV+Polars, Parquet+pandas, Parquet+Polars) tested and timed
    - Parquet file size < CSV file size (should be 3-5x smaller)
    - Timing results presented in a clean comparison table
    - Data integrity verified: read-back matches original for all 4 combinations

4. **A return statistics summary** (bridge to Week 2)
   - **Task type:** Skill Building
    - For 10 representative tickers across sectors, compute: mean daily return, annualized volatility, skewness, kurtosis, min/max daily returns, number of days with |return| > 3 standard deviations
    - Present in a clean summary table
    - Brief observations: are returns normally distributed? (No.) Is kurtosis close to 3? (No — it's much higher.) Do all stocks behave the same? (No — compare a utility like DUK to TSLA.)
    - This sets up the precise questions that Week 2 will answer
   - **Acceptance criteria:**
    - Statistics computed for all 10 tickers (not fewer)
    - Kurtosis > 3 for every ticker (fat tails are universal — should be 4-30+)
    - At least one volatile ticker (e.g., TSLA) has kurtosis > 10
    - Number of > 3σ days exceeds the normal distribution expectation (~0.3%) for every ticker
    - Summary table is clean and readable with consistent formatting

### Expected Discoveries

- Building a reliable pipeline for 5 stocks is easy. Extending it to 50 reveals edge cases: tickers that changed symbols, IPOs that have short histories, ETFs with different trading calendars, BRK.B's yfinance ticker format (`BRK-B` not `BRK.B`)
- Parquet files are 3-5x smaller than CSV and 5-10x faster to read, especially as the dataset grows. The advantage is dramatic for wide-format data with many tickers
- Polars is 3-5x faster than pandas for common operations (groupby, rolling, joins). The speed difference is most visible on larger datasets
- Data quality varies enormously: large-cap liquid stocks (AAPL, MSFT, SPY) are nearly perfect; less liquid names and stocks with complex corporate action histories have real problems
- Some "anomalies" in the data are real events (COVID crash: SPY -12% on March 16, 2020) and some are data artifacts. The pipeline must distinguish between the two — flagging without automatically removing
- Return kurtosis is much higher than 3 (the normal distribution value) for every stock examined. TSLA might have kurtosis > 20. This single observation motivates all of Week 2

## Key Papers & References

- **Ernst & Spatt, "Regulating Market Microstructure" (Annual Review of Financial Economics, Vol. 17, pp. 173-187, Nov 2025)** — Comprehensive survey of equity, option, and fixed-income market regulation. Covers best execution, Reg NMS, PFOF, tick sizes, access fees. Read this for the regulatory big picture.
- **Harris, "Trading and Exchanges: Market Microstructure for Practitioners" (2003)** — The standard reference on how markets work. Dense but readable. Chapters 1-6 are directly relevant to this week.
- **de Prado, "Advances in Financial Machine Learning" (2018), Chapter 2** — The definitive treatment of financial data structures (bars, labeling, sampling). Introduces dollar bars and information-driven bars. We'll revisit this in later weeks, but it's worth skimming now.
- **Boulton, Shohfi & Walz, "How Does Payment for Order Flow Influence Markets?" (SEC DERA Working Paper, Jan 2025)** — Uses Robinhood Crypto as a natural experiment to measure PFOF's effect on market quality. Finds crypto PFOF fees are 4.5-45x higher than equity PFOF, and the introduction increased daily trading costs by ~$4.8M. Concrete evidence for the PFOF discussion.

## Career Connections

- **Quant data engineer / data infrastructure:** At firms like Two Sigma, Citadel, or DE Shaw, someone's entire job is building and maintaining the data pipeline — downloading from vendors, cleaning corporate actions, validating quality, and serving clean data to researchers. The `FinancialDataLoader` class from this week's homework is a toy version of what they build for hundreds of thousands of instruments across asset classes. A junior hire's first month often involves debugging exactly the data quality issues in Exercise 3.
- **Quantitative researcher / analyst:** Every quant researcher's first step on any project is pulling clean data and sanity-checking it. Survivorship bias (Exercise 1) is the #1 question a PM will ask when reviewing a backtest: "Is your universe point-in-time?" If you can't answer that confidently, the research doesn't proceed. Data quality is career currency — the researcher who can explain *why* their data is trustworthy gets their strategies allocated to.
- **Risk analyst at a multi-strategy fund:** Risk teams run quality checks similar to Exercise 3 daily — monitoring for stale prices, missing data, and anomalous returns across the fund's entire position book. A stale price in a portfolio's NAV calculation means misstated risk. The automated pipeline students build this week mirrors the daily "data health check" that risk teams at firms like Millennium or Balyasny depend on.
- **Buy-side or sell-side technologist:** At banks and asset managers, the data stack discussion (Section 7) is a daily reality. Teams are actively migrating from pandas+CSV to Polars+Parquet or DuckDB stacks. Knowing both ecosystems — and when to use which — is a hiring signal that distinguishes a quant-aware engineer from a generic Python developer.

## Data Sources

- **yfinance** (free, no API key): Daily OHLCV + Adjusted Close for any publicly traded US stock. Covers splits, dividends. Unreliable under heavy load (rate limits, occasional schema changes). Sufficient for all exercises this week.
- **Wikipedia S&P 500 constituent list** (free): Current list of S&P 500 tickers. Used in the seminar survivorship bias exercise. Scrapeable with pandas `read_html()`.
- **Approximate data sizes:** 50 stocks x 15 years of daily data ≈ ~190K rows in long format. CSV: ~15-25 MB. Parquet: ~3-5 MB. Download time via yfinance: 1-3 minutes depending on rate limiting.
- **Professional alternatives (mentioned, not required):** CRSP/Compustat via WRDS (academic standard, survivorship-free), Databento (tick-level, API-first), Polygon.io (real-time + historical), Bloomberg Terminal (institutional standard). Students should know these exist for when they move beyond educational use.
