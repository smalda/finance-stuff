# Week 1 — Financial Markets, Data Structures & Microstructure

> **Every ML-for-finance failure you'll ever read about started here — with someone who didn't understand their data.**

## Prerequisites
None. This is the first week of the course. We assume strong Python/pandas skills and familiarity with core ML concepts (train/test splits, overfitting, loss functions, gradient descent). We assume zero knowledge of financial markets, instruments, or terminology.

## The Big Idea

Here's something that should unsettle you: the single most common reason ML models fail in finance has nothing to do with architecture, hyperparameters, or loss functions. It's the data. Specifically, it's that the person who built the model didn't understand the data they were feeding it — didn't know that stock prices can't be treated like pixel values, didn't know that their "clean" dataset had quietly removed every company that went bankrupt, didn't know that the number their model was predicting had already been reported three days before the timestamp said it would be.

This week is the foundation for everything that follows. We're going to take an ML engineer who knows how to build a transformer from scratch but has never thought about what a bid-ask spread is, and give them the mental model they need to work with financial data without making the mistakes that have cost real firms real money. When Knight Capital deployed untested code in August 2012, they lost $440 million in 45 minutes — roughly $10 million per minute. The bug reactivated old test logic that bought high and sold low, over and over. That's not a model failure. That's a data and systems failure. The model did exactly what it was told. It was told the wrong thing.

We'll start from the absolute basics — what is an exchange, what is an order book, what happens when you click "buy" — and build up to the data structures you'll actually use as inputs to ML models. Along the way, we'll encounter the four horsemen of financial data pathology: survivorship bias, look-ahead bias, non-stationarity, and fat tails. Each one can silently destroy a model that would otherwise look brilliant in a backtest.

By the end of this week, you'll be able to download financial data, clean it properly, compute returns correctly, understand why your return distributions look nothing like a Gaussian, and build alternative bar types (volume bars, dollar bars) that produce better-behaved inputs for ML models. You'll also have a healthy paranoia about data quality that will serve you for the rest of the course — and the rest of your career.

## Lecture Arc

### Opening Hook

"Here's something that might surprise you: when you click 'buy' on Robinhood, your order doesn't go to 'the stock market.' It goes to Citadel Securities — a single company that handles about 25% of all US equity trades. They look at your order, decide whether to fill it themselves, and pocket a fraction of a penny for their trouble. They made $7.5 billion in revenue in 2022. On pennies per share. To understand why that system exists, why those fractions of pennies add up to billions, and why it matters for every ML model you'll ever build in finance — we need to start from the beginning. We need to understand how markets actually work."

### Section 1: How Financial Markets Actually Work
**Narrative arc:** We start from the simplest possible mental model — two people wanting to trade — and build up to the modern exchange. The tension is that what seems simple (buying a stock) is actually a complex multi-party negotiation happening in microseconds. The resolution is the order book, which makes the negotiation transparent and mechanical.

**Key concepts:**
- Exchanges (NYSE, NASDAQ, CME) and their role as neutral matching engines
- The order book: a queue of buy orders (bids) and sell orders (asks)
- Limit orders vs. market orders
- The bid-ask spread: the price of immediacy
- Market makers: the people standing in the middle
- Dark pools: off-exchange venues where large orders hide

**The hook:** "Imagine you're buying a used car. The sticker says $20,000. But you can't just pay the sticker price — that's the seller's asking price. You counter at $19,500. That $500 gap is the bid-ask spread. On Wall Street, this gap is pennies for liquid stocks — Apple's spread is typically 1 cent on a $190 share. But for a micro-cap stock, the spread can be 50 cents on a $5 share — that's 10%. Your model needs to be right by more than 10% just to break even on a round trip. Most models aren't right by 1%."

**Key formulas:** The bid-ask spread:
"We can express the cost of trading as the spread relative to the price. The round-trip cost is:"

$$\text{Round-trip cost} = \frac{\text{ask} - \text{bid}}{\text{mid-price}} = \frac{2 \times \text{half-spread}}{\text{mid-price}}$$

"For Apple at bid=$189.99, ask=$190.01: that's about 0.01%. For a micro-cap at bid=$4.75, ask=$5.25: that's 10%. Same formula, wildly different implications for your strategy."

**Code moment:** We'll pull a real-time snapshot of the AAPL order book (or a cached one) and visualize it as two facing histograms — bids on the left (green), asks on the right (red), with the spread visible as the gap in the middle. Students should notice that the book isn't symmetric: there's more volume on one side than the other. That imbalance is a signal — it tells you which direction the market is leaning.

**"So what?":** "Every time your model says 'buy,' the bid-ask spread is a tax that comes out of your returns. A strategy that turns over daily at 10 bps round-trip burns 25% per year in transaction costs alone. For reference, the average hedge fund's gross return is about 10-15%. You'd be spending twice your expected revenue on shipping costs. That's not a strategy — it's a donation to market makers."

### Section 2: Financial Data Types and Their Quirks
**Narrative arc:** We move from how markets work to what data they produce. The tension is that ML engineers are used to clean, well-structured datasets (ImageNet, MNIST), but financial data is messy, multi-modal, and full of traps. The resolution is building awareness of each data type's specific quirks.

**Key concepts:**
- Price data: OHLCV (Open, High, Low, Close, Volume)
- What OHLCV actually represents: a compression of thousands of trades into 5 numbers
- Fundamental data: earnings, book value, revenue — and the point-in-time problem
- Alternative data: news sentiment, satellite imagery, social media, credit card transactions
- Adjusted vs. unadjusted prices: stock splits and dividends

**The hook:** "On June 9, 2022, Amazon did a 20:1 stock split. The stock went from ~$2,447 to ~$122 overnight. If your model is looking at raw prices, it sees a 95% crash. If it's looking at adjusted prices, it sees... nothing happened. Same company, same value, same everything — but one version of the data says the world ended and the other says it was a quiet Tuesday. This is the adjusted price problem, and it's the first of many ways financial data will try to trick your model."

**Key formulas:** None in this section — this is conceptual groundwork.

**Code moment:** Download AAPL data with `yfinance` and show both adjusted and unadjusted close prices. The split is visible as a massive discontinuity in unadjusted data but invisible in adjusted data. Then show the `auto_adjust` parameter and explain why we almost always want adjusted prices.

```python
import yfinance as yf

aapl = yf.download("AAPL", start="2014-01-01", end="2024-01-01")
# Show columns: Open, High, Low, Close, Adj Close, Volume
# Plot Close vs Adj Close — the splits are visible as steps in Close
```

**"So what?":** "Point-in-time is the financial data equivalent of data leakage. If you use a company's 2023 annual earnings to make a 'prediction' in March 2023, you're using data that wasn't available until February 2024 when the annual report was filed. Your backtest looks amazing. Your live strategy loses money. This is look-ahead bias, and it's the most common way quants accidentally cheat."

### Section 3: Data Pathology #1 — Survivorship Bias
**Narrative arc:** We introduce the single most insidious data quality problem in finance. The setup: you download S&P 500 stock data and everything looks great. The tension: your dataset is lying to you by omission — it only contains companies that survived. The resolution: understanding the scale of the problem and how to (partially) address it.

**Key concepts:**
- Survivorship bias: only currently-listed stocks appear in most free datasets
- Delisted stocks vanish from history
- The effect on backtests: systematically overstates returns
- Quantifying the bias: ~2-3% per year in overstated returns

**The hook:** "Let's play a game. Name the 10 biggest US companies from the year 2000. You'll probably remember Microsoft, GE, Walmart. You probably won't remember WorldCom — #20 by market cap — which filed the largest bankruptcy in US history two years later. Or Enron — #7 by revenue — which ceased to exist by December 2001. If your training data starts in 2005, these companies simply don't exist. Your model will learn that large-cap stocks always survive. They don't."

**Key formulas:**
"We can estimate survivorship bias as the difference in average return between a dataset with only survivors and the full dataset:"

$$\text{Survivorship Bias} = \bar{R}_{\text{survivors only}} - \bar{R}_{\text{full universe}}$$

"Elton, Gruber & Blake (1996) found this to be roughly 0.9% per year for mutual funds. For individual stocks, it's larger — Shumway & Warther (1999) estimated 0.5-1.0% per month for delisted stocks, which compounds to enormous errors over a backtest horizon."

**Code moment:** We'll download the current S&P 500 constituent list and show that it contains Amazon (added 2005), Tesla (added 2020), and Meta (added 2013). Then ask: what was in those slots before? Show a historical list from 2000 — Enron, WorldCom, Lehman Brothers, Bear Stearns, Washington Mutual. All gone. Then quantify: of the S&P 500 in 2000, approximately 150 companies are no longer in the index by 2024. That's 30% turnover. A model trained only on survivors has never seen 30% of the actual market.

**"So what?":** "Think of survivorship bias like training on a dataset where all the failed examples have been removed. You'd get great training accuracy and terrible real-world performance. That's exactly what happens when you backtest on Yahoo Finance data. Your model has never seen a company die. It will be very confused when one does."

### Section 4: Data Pathology #2 — Non-Stationarity and Fat Tails
**Narrative arc:** We confront the two statistical properties that make financial data fundamentally different from most ML datasets. The setup: ML models assume their inputs come from a stable distribution. The tension: stock prices violate this assumption spectacularly. The resolution: returns (not prices) plus an awareness of fat tails — which we'll address formally in Week 2.

**Key concepts:**
- Non-stationarity: the data-generating process changes over time
- Why raw prices can't be model inputs: the distribution shifts constantly
- Returns as the (partial) fix: first-differencing removes the trend
- Fat tails: extreme events happen far more often than Gaussian models predict
- Volatility clustering: big moves follow big moves
- Excess kurtosis as the measure of "fat-tailedness"

**The hook:** "On October 19, 1987, the S&P 500 dropped 22.6% in a single day. Under a Gaussian model with the historical mean and standard deviation, a move that large has a probability of about 10^-160. To put that in perspective, the universe is about 10^17 seconds old. If you watched the market every second since the Big Bang, across 10^50 parallel universes, you still wouldn't expect to see this event. And yet it happened on a Monday."

**Key formulas:**
"Start with raw prices. Apple's price in 2015 was around $130. In 2024, it's around $190. If you train a model on 2015 data and test on 2024 data, every number in the test set is outside the training range. The model is extrapolating every single prediction. This is non-stationarity.

The partial fix is returns — the percentage change from one period to the next:"

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} \quad \text{(simple return)}$$

$$R_t = \ln\left(\frac{P_t}{P_{t-1}}\right) \quad \text{(log return)}$$

"Returns are roughly stationary — they fluctuate around a mean that's close to zero. But 'roughly' is doing a lot of work in that sentence. We'll spend all of Week 2 on what 'roughly' means and how to do better."

Then introduce kurtosis as the measure of tail thickness:

$$\text{Excess Kurtosis} = \frac{E[(X - \mu)^4]}{\sigma^4} - 3$$

"A Gaussian has excess kurtosis of 0. The S&P 500 has excess kurtosis of about 20. That means the tails are 20 times thicker than a Gaussian would predict. Every risk model that assumes normality is lying to you — it just hasn't been caught yet."

**Code moment:** Two plots, side by side. Left: the distribution of daily S&P 500 returns overlaid with a Gaussian with the same mean and standard deviation. "If markets were the well-behaved system that most textbooks assume, these curves would overlap perfectly. They won't. Watch the tails." Right: a QQ-plot showing the same data. The points should curve away from the diagonal in both tails, dramatically. Students should see that the 1987 crash, the 2008 crisis, and the COVID crash of March 2020 are all events that "shouldn't happen" under Gaussian assumptions.

**"So what?":** "If you train an LSTM on raw Apple prices from 2015 ($130) and test on 2024 ($190), the model has never seen numbers in the test range. It's extrapolating every single prediction. This is the equivalent of training an image classifier on cats and testing on dogs — except it's harder to notice because the numbers look similar."

### Section 5: Returns Math — Simple vs. Log Returns
**Narrative arc:** We now teach the correct way to compute returns, starting from a puzzle that reveals why naive arithmetic is wrong. The tension: simple returns have a compounding trap. The resolution: log returns fix it, and we explain when to use which.

**Key concepts:**
- Simple (arithmetic) returns
- Log (geometric/continuously compounded) returns
- The compounding trap: why simple returns don't add across time
- Log returns are additive across time but not across assets
- Simple returns are additive across assets (portfolio returns) but not across time
- Annualization: how to scale daily returns to annual

**The hook:** "Quick puzzle. You invest $100. It goes up 10% in January — great, you have $110. Then it drops 10% in February. Are you back to $100? Nope. You're at $99. You lost money on two moves that should cancel out. This is the compounding trap, and it's bitten more junior quants than any bug in production code."

**Key formulas:**
"Log returns fix this. Take the log of 1.10, add the log of 0.90, and you get a small negative number — honest about the loss. They're the only return type that adds up cleanly across time:

$$R_{t_1 \to t_n} = \sum_{i=1}^{n} R_{t_i} = \sum_{i=1}^{n} \ln\left(\frac{P_{t_i}}{P_{t_{i-1}}}\right) = \ln\left(\frac{P_{t_n}}{P_{t_0}}\right)$$

For annualization, we scale by the number of trading days in a year (approximately 252):

$$\mu_{\text{annual}} = \mu_{\text{daily}} \times 252$$
$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$

The square root comes from the assumption that daily returns are (roughly) independent — variance adds, so standard deviation scales with the square root of time. Note: this assumption is approximate at best, and we'll challenge it in Week 2 when we discuss autocorrelation."

**Code moment:** Build it step by step:

```python
# Step 1: simple returns
simple_ret = prices.pct_change()

# Step 2: log returns
log_ret = np.log(prices / prices.shift(1))

# Step 3: show they're close for small returns, diverge for large ones
comparison = pd.DataFrame({
    'simple': simple_ret,
    'log': log_ret,
    'difference': simple_ret - log_ret
})
```

Show that for daily returns (typically <1%), the difference between simple and log is negligible. But for a big crash day like March 16, 2020 (-12%), the difference matters. Plot the two side by side.

**"So what?":** "For the rest of this course, we'll use log returns for time-series analysis (because they're additive across time) and simple returns for portfolio analysis (because they're additive across assets). Mixing them up is a classic mistake — it won't crash your code, but it will quietly bias every number downstream."

### Section 6: Alternative Bars — Volume Bars and Dollar Bars
**Narrative arc:** This is the payoff section — the Lopez de Prado content that distinguishes this course from a textbook. The setup: time bars (daily OHLCV) are the default, but they have a fatal flaw. The tension: not all time periods are equal — a Tuesday in August has a fraction of the volume of a Monday after an earnings release. The resolution: sample by activity instead of by clock time, and the returns become better-behaved.

**Key concepts:**
- Time bars: one observation per fixed time interval (the default)
- Tick bars: one observation per fixed number of trades
- Volume bars: one observation per fixed number of shares/contracts traded
- Dollar bars: one observation per fixed dollar amount traded (Lopez de Prado's recommendation)
- Why dollar bars are preferred: they normalize for both volume and price changes
- The Jarque-Bera test for normality

**The hook:** "Think about what a daily bar actually represents. On a quiet summer day, Apple might trade 30 million shares. On the day after an earnings announcement, it might trade 150 million shares. Same bar. Same weight in your dataset. But one of them contains 5x more information than the other. You'd never train an image classifier by giving some images 1 pixel and others 500 pixels and calling them the same thing. That's exactly what time bars do with financial data."

**Key formulas:**
"The idea is simple: instead of sampling once per day, sample once every time a certain dollar volume trades through the market:

$$\text{Dollar volume}_t = P_t \times V_t$$

$$\text{Cumulative dollar volume} = \sum_{i=1}^{t} P_i \times V_i$$

Every time the cumulative dollar volume crosses a threshold (say, $1 billion for SPY), we create a new bar. This means we get more bars during high-activity periods (lots of information flowing) and fewer bars during quiet periods (little information flowing)."

**Code moment:** Build dollar bars from scratch in stages:

```python
# Attempt 1: the napkin version
dollar_vol = spy['Close'] * spy['Volume']
cum_dollars = dollar_vol.cumsum()
bar_ids = (cum_dollars // threshold).astype(int)
dollar_bars = spy.groupby(bar_ids).agg({
    'Open': 'first', 'High': 'max', 'Low': 'min',
    'Close': 'last', 'Volume': 'sum'
})
```

"That actually works — 5 lines. But look at the output. We get bars of wildly different durations — some are 30 minutes of trading, others are 3 days. The short bars happen during earnings announcements and market crises. The long bars happen during quiet August weeks. That's the point — we're sampling information, not time."

Then show the punchline: compute the Jarque-Bera test statistic on returns from time bars, volume bars, and dollar bars. Dollar bar returns should be closest to Gaussian — lower kurtosis, lower Jarque-Bera statistic. This replicates Lopez de Prado's Figure 2.4 from *Advances in Financial Machine Learning*.

**"So what?":** "ML models generally perform better when their inputs are approximately Gaussian. Dollar bars move the returns distribution closer to Gaussian without any feature engineering or transformation — just by being smarter about when we sample. This is a free lunch, and it comes from understanding the data rather than from a fancier model."

### Section 7: Building a Clean Data Pipeline
**Narrative arc:** We take everything from the previous sections and combine it into a reusable data pipeline. The tension is that there are many ways to get financial data, and most of them will introduce at least one of the biases we discussed. The resolution is a disciplined, documented pipeline with explicit choices.

**Key concepts:**
- Data sources: `yfinance` (free, limited), WRDS/CRSP (academic gold standard), Polygon.io (professional)
- Handling missing data: NaN vs. forward-fill vs. drop — each has consequences
- Split/dividend adjustment
- The `DataLoader` pattern: a reusable class for the rest of the course
- Documenting your data choices (what you know, what you're ignoring)

**The hook:** "Every quant firm has a data pipeline they've spent years building and debugging. At Two Sigma, the data engineering team is larger than the research team. You won't build anything that sophisticated this week, but you will build something correct — and in this business, correct beats sophisticated every time."

**Code moment:** We'll build a `DataLoader` class step by step that:
1. Downloads data from `yfinance` for a configurable universe
2. Handles missing data (with a documented policy)
3. Computes both simple and log returns
4. Can produce time bars, volume bars, or dollar bars
5. Flags known data quality issues (split days, extreme moves, low-volume periods)

The class will be used throughout the rest of the course.

**"So what?":** "This DataLoader is your foundation. Every model you build for the next 17 weeks starts here. If it's wrong, everything downstream is wrong. If it's right, you can trust your results — and that trust is worth more than any architectural improvement you'll ever make to a model."

### Closing Bridge

"Let's take stock of what you now know that you didn't know 90 minutes ago. You know that 'the stock market' is actually a complex network of exchanges, dark pools, and market makers — and that every trade has a cost that most backtests ignore. You know that your data has been quietly lying to you: removing dead companies, leaking future information, and pretending that extreme events can't happen. You know that returns, not prices, are the right input for ML — and that log returns and simple returns each have their place. And you know that sampling by dollar volume instead of by clock time gives you better-behaved data for free.

Next week, we're going to confront the deeper problem that returns only partially solve: stationarity. We'll discover that taking first differences (returns) throws away too much information, that raw prices keep too much, and that there's a mathematically elegant sweet spot between the two called fractional differentiation. It sounds exotic. It's actually just calculus being clever."

## Seminar Exercises

### Exercise 1: Fat-Tail Safari Across Asset Classes
**The question we're answering:** Is non-Gaussianity a SPY-specific problem, or does every corner of the market have fat tails — and which corners are fattest?

**Setup narrative:** "The lecture showed you SPY's fat tails. Now you're going on a safari: how do tails compare across 10 different ETFs spanning equities, bonds, commodities, and volatility products? Not every asset misbehaves in the same way, and the differences matter for every risk model you'll build."

**What they build:** Download daily data for 10 ETFs from diverse asset classes (SPY, QQQ, TLT, GLD, USO, EEM, IWM, HYG, XLU, VXX/VIXY) from 2010 to present. Compute log returns for each. Generate a comparison table: mean, std, skewness, excess kurtosis, Jarque-Bera statistic. Produce QQ-plots for all 10 on a single multi-panel figure. Rank by excess kurtosis.

**What they'll see:** Kurtosis varies dramatically: VXX/VIXY (volatility products) may exceed 30, oil (USO) will be high (~15-20), bonds (TLT) will be lower but still non-Gaussian (~5-8), and utilities (XLU) will be the tamest. Skewness also varies: equity ETFs tend to be left-skewed (crashes), while VXX is right-skewed (volatility spikes). GLD has near-zero skewness but still fat tails.

**The insight:** "Fat tails aren't a single phenomenon — they have different shapes and magnitudes across asset classes. A risk model that assumes the same distributional shape for equities and bonds is making two different mistakes in two different ways. Your ML model needs features that capture asset-specific tail behavior, not a one-size-fits-all normality assumption."

### Exercise 2: Dollar Bars at Scale — 10 ETFs
**The question we're answering:** Does the dollar-bar advantage generalize beyond SPY, and how does the improvement vary by liquidity?

**Setup narrative:** "The lecture built dollar bars for SPY and showed they produce more Gaussian returns. But SPY is the most liquid security on the planet — it's an easy case. What happens when you try dollar bars on a less liquid commodity ETF, or a volatile emerging-markets fund? Does the improvement scale, shrink, or vanish?"

**What they build:** Using the 10 ETFs from Exercise 1, implement time bars, volume bars, and dollar bars for each. Choose per-ETF thresholds (use median daily dollar volume as the starting point for the dollar-bar threshold, adjusting for each ETF's liquidity). Compute returns for all three bar types across all 10 ETFs. Run Jarque-Bera tests. Build a comparison table: ETF, bar type, excess kurtosis, Jarque-Bera statistic, number of bars generated.

**What they'll see:** The dollar-bar advantage is strongest for liquid, high-volume ETFs (SPY, QQQ) and weakest for low-volume ETFs (XLU, USO). For illiquid ETFs, the threshold choice matters enormously — too low produces noisy micro-bars, too high produces few bars. The kurtosis reduction from time bars to dollar bars ranges from 20-50% for liquid names to 5-15% for illiquid names.

**The insight:** "Dollar bars aren't a magic bullet — their benefit depends on the liquidity and volume profile of the asset. For highly liquid assets with huge intraday volume variation (like SPY during earnings vs. August), the improvement is dramatic. For illiquid assets with sparse trading, the threshold selection becomes a new source of researcher discretion. Know when your free lunch has a price tag."

### Exercise 3: Survivorship Bias in Action
**The question we're answering:** How much does survivorship bias actually inflate backtest returns?

**Setup narrative:** "Let's see survivorship bias with real numbers. We'll take the S&P 500 constituent list from 2010 and 2024, and compare the average return of stocks that survived to 2024 versus the full 2010 universe."

**What they build:** Download the S&P 500 constituent list as of 2010 (from Wikipedia snapshots or provided CSV). Download the current list. Identify stocks that were removed between 2010 and 2024. For the survivors: compute average annualized return. For the removed stocks (where data is available): compute average return up to removal. Compare the two.

**What they'll see:** The survivors will have meaningfully higher average returns — somewhere in the range of 2-4% per year higher, depending on the exact period. This is the survivorship bias premium: the return you think your strategy earned but actually never existed.

**The insight:** "If you use `yfinance` to download 'S&P 500 stocks' today and backtest to 2010, you're only getting the winners. The losers — the Lehman Brothers, the GE (removed 2018 after 122 years), the Xerox — have been quietly deleted from your universe. Your model learned that large-cap stocks always survive. They don't."

### Exercise 4: Transaction Cost Reality Check
**The question we're answering:** How much do transaction costs eat into a strategy's returns?

**Setup narrative:** "Most academic backtests ignore transaction costs entirely. Let's see what happens when we don't."

**What they build:** Take a simple momentum strategy on SPY (buy when 50-day moving average is above 200-day, sell otherwise). Compute returns with and without transaction costs at different levels: 0 bps, 5 bps, 10 bps, 20 bps per side (so 0, 10, 20, 40 bps round-trip). Plot cumulative returns for each cost level.

**What they'll see:** At 0 bps, the strategy looks reasonable. At 10 bps per side, the returns drop noticeably — especially for a strategy that trades frequently. At 20 bps, the strategy may underperform buy-and-hold. The effect is more dramatic for higher-turnover strategies.

**The insight:** "Transaction costs are not a rounding error — they're a first-order determinant of whether a strategy works. A strategy that earns 8% per year gross but trades weekly at 10 bps round-trip loses roughly 5.2% per year to costs, netting only 2.8%. Buy-and-hold with zero costs often wins. This is why every model we build in this course must include a transaction cost estimate."

## Homework: "The Data Quality Audit"

### Mission Framing

Your mission is straightforward but important: take a universe of 200 US equities and put the data through the most skeptical scrutiny you can manage. Think of yourself as an auditor hired to certify that a dataset is safe to use for ML — because that's exactly what quants do before they let any model near real capital.

You'll download the data yourself, compute returns both ways, build alternative bar types, hunt for data quality issues, and ultimately produce a clean `DataLoader` class that handles the common pitfalls. The class you build this week will be the foundation of every pipeline you construct for the rest of the course.

Here's the thing that makes this homework more than busywork: every issue you find in this dataset is an issue that exists in production systems at hedge funds. The difference is that their datasets have thousands of stocks, decades of history, and millions of dollars riding on the results. Yours has 200 stocks and a grade riding on it. The habits are the same.

### Deliverables

1. **Download and inspect data for 200 US equities (2010-2024) via `yfinance`.** Use the current S&P 500 constituent list as your starting universe (you'll note the survivorship issue — document it explicitly). Compute both simple and log returns. Produce QQ-plots for at least 10 stocks spanning different sectors and market caps. Report kurtosis and skewness for the full universe.

2. **Implement volume bars and dollar bars.** Using SPY's daily data (or intraday data if available), construct volume bars and dollar bars. Choose sensible thresholds (hint: for daily data, a dollar-bar threshold around the median daily dollar volume works). Compare return distributions across bar types using the Jarque-Bera test. Replicate the qualitative result from Lopez de Prado Figure 2.4: dollar bars should produce returns closest to Gaussian.

3. **Identify and document at least 3 specific data quality issues.** Inspect the data for: missing dates, zero-volume days, extreme returns (>15% in a day — are they real moves or data errors?), stock splits that weren't properly adjusted, NaN values, and any other anomalies. For each issue, document: what it is, which stocks are affected, how it would bias an ML model, and how you fixed it.

4. **Build a clean data pipeline as a `DataLoader` class.** The class should: accept a list of tickers and a date range; download and cache the data; compute returns (simple and log); handle missing data with a documented policy; flag known anomalies; and be reusable for future weeks. Include a `get_dollar_bars()` method.

5. **Write a 1-page data quality report** (as markdown cells in the notebook) summarizing: what biases exist in this dataset, what you fixed, and what remains unfixable with free data. Be honest — this is practice for the kind of documentation that separates a trustworthy backtest from a misleading one.

### What They'll Discover

- The kurtosis of daily returns varies dramatically across stocks: a boring utility like Duke Energy might have excess kurtosis around 5, while a volatile meme stock like GameStop can exceed 50. The "one-size-fits-all" distributional assumption is even more wrong than you thought.

- Dollar bars do produce more Gaussian-like returns, but the improvement is less dramatic for daily data than for intraday data. The real win comes when you have tick-level data and the difference between a quiet Monday and an earnings-day Friday is a factor of 20x in trading volume.

- At least 10-20 of your 200 stocks will have obvious data quality issues: missing days, extreme returns from un-adjusted splits, or zero-volume periods. These aren't edge cases — they're the norm. If you don't check for them, your model will silently train on garbage.

- The survivorship bias in your dataset is structurally unfixable with `yfinance`: you can document it, you can acknowledge it, but you cannot download data for stocks that have been delisted. This is why professional quants pay for survivorship-bias-free databases like CRSP (which costs thousands of dollars per year for academic access).

### Deliverable
A complete Jupyter notebook containing: all analysis, visualizations (QQ-plots, return distribution comparisons, bar type comparisons), the `DataLoader` class, and the data quality report. Code should be clean, commented, and reusable.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Market microstructure (order book, bid-ask spread) | Demo: visualize AAPL order book, compute spread cost | Not covered (done in lecture) | Reference: transaction cost awareness in data quality report |
| OHLCV data & adjusted prices | Demo: download AAPL, show split discontinuity | Not covered (done in lecture) | At scale: download and inspect 200 equities, flag split issues |
| Survivorship bias | Demo: compare S&P 500 lists from 2000 vs. 2024 | Exercise 3: quantify bias with real 2010-2024 survivor vs. removed stocks | Integrate: document bias in the 200-stock universe |
| Fat tails & non-Gaussianity | Demo: SPY histogram + QQ-plot vs. Gaussian | Exercise 1: compare kurtosis/QQ-plots across 10 diverse ETFs | At scale: QQ-plots and kurtosis for 10+ stocks spanning sectors |
| Returns math (simple vs. log) | Demo: compute both, show divergence on crash days | Not covered (done in lecture) | At scale: compute both for 200 stocks, use in DataLoader |
| Alternative bars (volume, dollar) | Demo: build SPY dollar bars, Jarque-Bera comparison | Exercise 2: build dollar bars for 10 ETFs, vary thresholds by liquidity | At scale: implement `get_dollar_bars()` in DataLoader class |
| Transaction costs | Demo: compute round-trip cost for liquid vs. illiquid stocks | Exercise 4: apply cost tiers to a momentum strategy, plot degradation | Integrate: acknowledge cost impact in data quality report |
| Data pipeline (`DataLoader` class) | Demo: build class step by step | Not covered (built in homework) | Build: complete `DataLoader` with caching, returns, bars, anomaly flags |

## Key Stories & Facts to Weave In

1. **Knight Capital, August 2012.** Knight Capital Group deployed new software that reactivated old test code. The code bought at the ask and sold at the bid — exactly backwards — across 154 stocks simultaneously. In 45 minutes, Knight lost $440 million. The company's market cap was $365 million. They were acquired by Getco within months. The lesson: data and systems failures are more expensive than model failures.

2. **Citadel Securities.** One company handles approximately 25% of all US equity trades. They're not Citadel the hedge fund (that's Citadel LLC, different entity). They're the plumbing — a market maker that stands between buyers and sellers, earning fractions of a penny per share. Revenue in 2022: $7.5 billion. On pennies per share. The scale of modern market microstructure is staggering.

3. **The 1987 crash.** On Black Monday (October 19, 1987), the Dow Jones fell 22.6% in a single day. Under Gaussian assumptions, this was a 25-sigma event — probability approximately 10^-160. The universe has experienced approximately 4.3 × 10^17 seconds since the Big Bang. This event was supposed to be impossible on timescales that make the age of the universe look like a coffee break. It happened on a Monday.

4. **The Flash Crash of May 6, 2010.** The Dow Jones fell about 1,000 points (~9%) in minutes, then recovered most of the drop within 20 minutes. Accenture briefly traded at 1 cent per share. Procter & Gamble dropped 37% in seconds. The SEC investigation found that a single mutual fund's automated selling algorithm triggered a cascade of high-frequency trading responses. The market's plumbing broke for 20 minutes, and trillions of dollars evaporated temporarily.

5. **Enron and WorldCom.** In 2000, Enron was the 7th largest US company by revenue. WorldCom was #20 by market cap. Both ceased to exist by 2003 — Enron due to fraud (stock went from $90 to $0.26), WorldCom due to an $11 billion accounting fraud. If your dataset starts in 2005, these companies don't exist. Your model has never learned that a $90 stock can go to zero.

6. **The XIV blowup, February 5, 2018.** The XIV ETN (inverse volatility product) had $1.9 billion in assets and had delivered exceptional returns for years by betting that volatility would stay low. On February 5, 2018, the VIX doubled in a single day. XIV lost 96% of its value overnight. It was liquidated within the week. Investors who thought volatility was mean-reverting to "low" were right — until they weren't. Survivorship bias in strategy selection: the strategies that blow up don't show up in "best-performing ETF" lists.

7. **Amazon's 20:1 stock split, June 2022.** The stock went from ~$2,447 to ~$122 overnight. Same company, same market cap, same everything. An ML model looking at unadjusted prices sees a 95% crash. An ML model looking at adjusted prices sees a quiet day. If your pipeline doesn't handle split adjustments, this one event would inject a catastrophic outlier into your training data, and your model would learn that stocks can lose 95% in a day without any fundamental change.

8. **The bid-ask spread in practice.** Apple (AAPL) has a bid-ask spread of about 1 cent on a ~$190 share — roughly 0.005%. A small-cap stock like Cato Corporation (CATO) might have a spread of 5-10 cents on a $7 share — roughly 1-1.5%. For a model that predicts a 0.5% daily move, trading Apple costs almost nothing, but trading CATO costs 2-3% round-trip — the strategy is dead on arrival for small caps. This is why almost every profitable quant strategy focuses on liquid, large-cap stocks.

## Cross-References
- **Builds on:** Nothing — this is Week 1. We assume Python/pandas fluency and ML fundamentals.
- **Sets up:** The `DataLoader` class built this week will be used throughout the course. The concepts of non-stationarity and fat tails introduced here are formalized in Week 2 (stationarity tests, fractional differentiation). The transaction cost awareness introduced here becomes a quantitative model in Week 3 (constant bps) and Week 18 (market impact models). Survivorship bias awareness informs the cross-sectional stock selection in Weeks 4-5.
- **Recurring thread:** The "your model vs. reality" thread starts here. Every week will include at least one moment where the textbook assumption meets the messy reality of markets. This week: Gaussian vs. fat-tailed distributions. Week 2: stationary vs. non-stationary. Week 3: optimal vs. estimated covariance matrices. And so on.

## Suggested Reading
- **Lopez de Prado, *Advances in Financial Machine Learning*, Chapter 2 (Data Structures)** — The authoritative treatment of alternative bar types. Lopez de Prado makes a detailed case for dollar bars with empirical evidence. Dense but essential.
- **Larry Harris, *Trading and Exchanges* (2003)** — The definitive market microstructure textbook. Dated in its technology (pre-HFT era) but timeless in its economics. If you want to understand why markets are structured the way they are, this is the book.
- **Stefan Jansen, *Machine Learning for Algorithmic Trading*, Chapter 2 (Market & Fundamental Data)** — The most practical treatment of financial data for ML engineers. Covers everything from tick data to fundamental data to alternative data, with working Python code.
