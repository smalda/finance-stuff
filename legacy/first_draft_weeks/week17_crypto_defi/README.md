# Week 17 — Crypto & DeFi ML

> **Crypto isn't just another asset class. It's a parallel financial system where the order book is a math formula, every transaction is public, and the market never closes.**

## Prerequisites
- **Week 5 (Tree-Based Methods):** XGBoost feature engineering and model training. You'll use XGBoost as your primary model for crypto return prediction, with a feature set that includes on-chain data unavailable in traditional finance.
- **Week 8 (LSTM/GRU):** Sequential modeling. LSTM on crypto price sequences is a natural extension, and you'll compare it against tree-based methods.
- **Week 1 (Markets & Data Structures):** Market structure basics, OHLCV data, returns, volatility. All the same concepts apply to crypto, but with a 24/7 twist.
- **Week 16 (Market Making):** Understanding how Avellaneda-Stoikov works helps you understand why AMMs have impermanent loss — the problems are structurally similar.
- **Week 3 (Portfolio Theory & Risk):** Sharpe ratio, drawdown metrics. You'll evaluate crypto strategies the same way, but the numbers will be wilder.

## The Big Idea

Here's what makes crypto fundamentally different from everything we've done so far, and it's not the price volatility (though Bitcoin's annualized volatility of 60-80% makes the S&P 500's 15-20% look like a savings account). It's the *data*.

In traditional finance, your data pipeline starts with price and volume — and that's largely where it ends for most practitioners. Sure, you can pay for fundamental data (earnings, revenue, balance sheets), alternative data (satellite imagery, credit card transactions), or sentiment data (news, social media). But the core trading data — who bought what, from whom, at what price — is private. You see the output (OHLCV bars). You don't see the process.

In crypto, you see *everything*. Every Bitcoin transaction that has ever occurred, from the genesis block on January 3, 2009, to the one confirmed 10 minutes ago, is publicly visible on the blockchain. You can see wallets accumulating. You can see coins moving from cold storage to exchanges (a selling signal). You can see the exact moment a whale deposits 10,000 BTC on Binance. You can see how many addresses are active, how much was transacted, whether holders are in profit or loss. This is an entirely new category of features — on-chain data — and it doesn't exist in traditional finance because traditional finance doesn't have a public ledger.

Then there's DeFi — decentralized finance — where the exchanges themselves are mathematical formulas running on blockchain smart contracts. Uniswap doesn't have an order book. It has a constant product equation: x * y = k. You trade against a pool of liquidity, not against another human. The price is determined by the ratio of assets in the pool, not by the highest bid and lowest ask. This is a fundamentally different market structure, and it creates both unique opportunities (arbitrage between AMMs and centralized exchanges, liquidity provision as a yield strategy) and unique risks (impermanent loss, MEV/sandwich attacks, smart contract vulnerabilities).

This week bridges everything you've learned — XGBoost, LSTM, feature engineering, risk management — to a market that's more volatile, more transparent, less efficient, and never closes. The techniques transfer. The data ecosystem is new. And the opportunities, while real, decay fast: crypto markets are less efficient than equities, which means alpha signals exist, but more participants discover and exploit them every month.

## Lecture Arc

### Opening Hook

On May 7, 2022, a cryptocurrency called Terra (UST) — a "stablecoin" that was supposed to maintain a 1:1 peg to the US dollar — began to wobble. It dropped to $0.98. Then $0.90. Then $0.50. Within a week, UST was worth $0.02 and its sister token LUNA collapsed from $80 to effectively zero, destroying approximately $40 billion in market value. The on-chain data told the story before the price did: in the 72 hours before the depeg, large wallets withdrew $2.2 billion from the Anchor protocol (Terra's main yield platform), and exchange inflows of LUNA spiked 300% above normal. Anyone monitoring on-chain features — which are free, public, and real-time — would have seen the bank run developing three days before the mainstream media reported it. That's the promise of on-chain analytics: the data is there, for everyone, if you know how to read it.

### Section 1: Crypto Market Structure — How It Differs from TradFi
**Narrative arc:** We establish the structural differences between crypto and traditional markets. These aren't superficial — they change how you collect data, engineer features, handle risk, and evaluate strategies.

**Key concepts:** Centralized exchanges (CEX) vs. decentralized exchanges (DEX), 24/7 markets, stablecoins, global/unregulated markets, retail dominance.

**The hook:** Let's count the differences. Crypto markets never close — they trade 24 hours a day, 7 days a week, 365 days a year. There's no "overnight gap" (a major feature in equity volatility models). There are no "business days" (your pandas resampling code needs to handle weekends). There's no SEC enforcement (market manipulation that would be illegal on the NYSE happens openly on crypto exchanges). The market is global — a whale in Singapore can move the price while you're asleep in New York. And the participants are overwhelmingly retail: about 80% of crypto volume comes from individual traders, compared to maybe 25% in US equities. Retail-dominated markets are less efficient, which means patterns that have been arbitraged away in equities may still exist in crypto. That's your opportunity — and your competition is getting smarter every month.

**Key formulas:** No formulas yet — this section is structural. Key numbers to ground the concepts:

| Property | US Equities | Crypto |
|----------|-------------|--------|
| Trading hours | 6.5 hrs/day, 252 days/yr | 24/7/365 |
| Annualized volatility (benchmark) | 15-20% (S&P 500) | 60-80% (Bitcoin) |
| Daily bars per year | 252 | 365 |
| Bid-ask spread (liquid) | 0.5-2 bps | 2-10 bps (BTC on Binance) |
| Retail participation | ~25% of volume | ~80% of volume |
| Regulatory oversight | SEC, FINRA | Minimal/fragmented |
| Public transaction data | None | Everything (blockchain) |

**Code moment:** Pull BTC and SPY data using `ccxt` and `yfinance`. Compare return distributions: BTC has fatter tails, higher kurtosis, no overnight gap (the 24h return distribution is smoother than the equity open-to-close + gap distribution), and about 4x the volatility. Plot them on the same axes with the same scale. Output: the BTC distribution dwarfs SPY — the tails extend far beyond where SPY's distribution ends.

**"So what?":** Every model we've built in this course was trained on data with 252 bars per year, 15-20% volatility, and well-regulated markets. Crypto has 365 bars per year, 60-80% volatility, and the regulatory equivalent of the Wild West. Your models will transfer — XGBoost doesn't care what market it's running on — but your assumptions about transaction costs, volatility regimes, and data quality need to be recalibrated.

### Section 2: On-Chain Data — Features That Don't Exist in TradFi
**Narrative arc:** This is the crown jewel of the lecture. On-chain data is a genuinely new feature category that has no analogue in traditional finance. We introduce four key metrics, explain the intuition, and show that they have predictive power for returns.

**Key concepts:** Active addresses, exchange flows, MVRV ratio, SOPR, funding rates.

**The hook:** Imagine if you could see, in real time, every deposit and withdrawal at every bank in the world. Imagine you could tell whether people were moving money from savings accounts to checking accounts (preparing to spend), or from checking to brokerage (preparing to invest). Imagine you could count exactly how many people had bank accounts and whether that number was growing or shrinking. In traditional finance, this data doesn't exist — it's private, scattered across thousands of institutions, and released in aggregate with a 6-week delay (if at all). In crypto, it's all public, all real-time, all free. The blockchain is a public ledger, and every transaction ever recorded is visible to anyone who cares to look.

**Key formulas:**

**Active Addresses:** The count of unique blockchain addresses that sent or received a transaction in a given period. This is a proxy for network adoption and activity. A rising active address count is bullish; declining suggests waning interest.

**Exchange Net Flows:**

$$\text{Exchange Net Flow}_t = \sum_{\text{deposits to exchanges}} v_i - \sum_{\text{withdrawals from exchanges}} v_i$$

Positive net flow = coins moving *to* exchanges = potential selling pressure (people move to exchanges to sell). Negative net flow = coins moving *off* exchanges = accumulation (people move to cold storage for long-term holding). This is remarkably intuitive: when the exchange inflows spike, it often precedes a price decline.

**MVRV Ratio (Market Value / Realized Value):**

$$\text{MVRV} = \frac{\text{Market Cap}}{\text{Realized Cap}} = \frac{\text{Price} \times \text{Supply}}{\sum_{\text{all UTXOs}} \text{price\_at\_last\_move}_i \times \text{amount}_i}$$

Realized cap values each coin at the price when it was last moved on the blockchain. If Bitcoin is $60,000 and someone bought their coins at $30,000, their contribution to realized cap is $30,000, not $60,000. MVRV > 3 historically signals "overheated" (unrealized profits are extreme, selling likely). MVRV < 1 signals "undervalued" (most holders are at a loss, capitulation may be ending).

**SOPR (Spent Output Profit Ratio):**

$$\text{SOPR} = \frac{\sum \text{value of outputs at current price}}{\sum \text{value of outputs at creation price}}$$

SOPR > 1: coins are being sold at a profit. SOPR < 1: coins are being sold at a loss (capitulation). SOPR = 1 is a key psychological level — holders are break-even, and prices often bounce off this level.

**Funding Rates:**

On perpetual futures (crypto's most traded derivative), longs pay shorts when funding is positive, and shorts pay longs when funding is negative. Extreme positive funding = crowded longs = mean-reversion signal (price tends to fall). Extreme negative funding = crowded shorts = mean-reversion signal (price tends to rise). Funding rates are the strongest mean-reversion signal in crypto — annualized funding above 100% is common during manias and reliably precedes corrections.

**Code moment:** Fetch active addresses and exchange flows from the Glassnode free tier (or use a pre-downloaded CSV). Merge with BTC daily price data. Compute the Spearman IC (information coefficient) between each on-chain feature and next-day returns. Output: MVRV z-score and funding rate z-score have the highest IC (~0.05-0.08 for daily returns). Exchange flows have lower IC but the sign is reliably correct (inflows negative for price). Active addresses have weak daily IC but strong monthly IC.

**"So what?":** These features are *genuinely predictive* — not in the "2% R²" sense that we saw in equity return prediction, but with ICs of 0.05-0.08, which in a single-asset strategy translates to a meaningful edge. The catch: alpha decays fast in crypto. Features that worked in 2020-2021 may be priced in by 2025. On-chain analytics is a real field with real firms (Glassnode, CryptoQuant, IntoTheBlock) — and the free tier gives you enough to build a proof of concept.

### Section 3: AMMs and the Constant Product Formula — The New Market Structure
**Narrative arc:** We move from data to market structure. Decentralized exchanges don't have order books — they have math. Understanding how AMMs work is essential for any ML practitioner working in DeFi.

**Key concepts:** Automated Market Maker (AMM), constant product formula, liquidity pools, liquidity providers (LPs), concentrated liquidity (Uniswap V3).

**The hook:** In Week 16, you built a market maker that continuously quotes bid and ask prices, earning the spread while managing inventory risk. Uniswap takes a radically different approach: it replaces the market maker with a *mathematical formula*. There's no human deciding where to quote. There's no RL agent optimizing spread width. There's just a pool of two assets — say, ETH and USDC — and a rule: the product of the two quantities must remain constant. If someone buys ETH from the pool, the ETH quantity decreases and the USDC quantity increases, such that x * y stays the same. The price is the ratio. No order book, no market makers, no latency competition. Just math.

**Key formulas:**

The constant product formula (Uniswap V2):

$$x \cdot y = k$$

where $x$ is the quantity of token A in the pool and $y$ is the quantity of token B. The spot price of A in terms of B is:

$$P = \frac{y}{x}$$

When a trader buys $\Delta x$ of token A, they pay $\Delta y$ of token B, such that:

$$(x - \Delta x)(y + \Delta y) = k \implies \Delta y = \frac{y \cdot \Delta x}{x - \Delta x}$$

The key insight: the price changes *continuously* with every trade, not discretely. Buying a large amount of token A raises its price. This is the AMM's built-in market impact function — and it's deterministic, unlike the stochastic impact in traditional markets.

**Uniswap V3: Concentrated Liquidity**

V3 lets liquidity providers choose a price range $[p_{\text{low}}, p_{\text{high}}]$. Their capital is only active when the price is within this range. This is more capital-efficient (you earn more fees per dollar deposited) but introduces a key risk: if the price moves outside your range, you stop earning fees AND you're fully exposed to one asset.

**Impermanent Loss:**

The cost of providing liquidity to an AMM. If you deposit equal value of ETH and USDC, and the price of ETH moves, your pool position is worth less than if you'd simply held the assets. The formula:

$$\text{IL} = \frac{2\sqrt{r}}{1 + r} - 1$$

where $r = P_{\text{new}} / P_{\text{initial}}$ is the price ratio. For a 2x price move ($r = 2$): IL = -5.7%. For a 5x move ($r = 5$): IL = -25.5%. It's called "impermanent" because it reverses if the price returns to its initial value. But in practice, prices rarely return exactly, and the IL is realized as actual loss.

**Code moment:** Simulate a Uniswap V2 pool with ETH/USDC. Starting: 100 ETH at $2,000 each + 200,000 USDC ($k = 100 \times 200{,}000 = 20{,}000{,}000$). Simulate 100 random trades and plot: (a) the price path, (b) the pool composition over time, (c) the impermanent loss vs. simply holding. Output: the AMM curve is the classic hyperbola. The IL is always negative (the LP always does worse than holding, ignoring fees). The fee income must exceed IL for LP to be profitable.

**"So what?":** AMMs are the most important innovation in crypto market structure. They enable trading without intermediaries, without order books, without market makers. But they have a fundamental cost — impermanent loss — that is essentially the AMM equivalent of adverse selection. The market maker in Week 16 feared informed traders; the LP in a Uniswap pool fears price movements. Same economic problem, different mathematical formulation.

### Section 4: MEV and Sandwich Attacks — The New Front-Running
**Narrative arc:** We introduce the dark side of DeFi: Maximal Extractable Value. Validators can reorder transactions to profit at your expense. This is front-running, but it's algorithmic, on-chain, and happening at scale.

**Key concepts:** MEV (Maximal Extractable Value), sandwich attacks, front-running, back-running, Flashbots, private transaction pools.

**The hook:** In traditional markets, front-running is illegal. If your broker sees your order to buy 10,000 shares of Apple and buys ahead of you, the SEC will shut them down. In DeFi, front-running is *automatic*. Here's how a sandwich attack works: you submit a transaction to buy ETH on Uniswap. Before your transaction is confirmed, a bot sees it in the public mempool (the queue of pending transactions), submits its own buy order *before* yours (pushing the price up), lets your order execute at the higher price, and then sells right after (pocketing the difference). Your trade happens at a worse price. The bot profits. And it's all perfectly "legal" — there's no regulator to complain to. In 2023, MEV bots extracted an estimated $600 million from Ethereum transactions. That's $600 million taken from regular DeFi users through algorithmic front-running.

**Key formulas:**

A sandwich attack on a Uniswap trade:

1. Victim submits: buy $\Delta x$ ETH at max slippage $s$
2. Attacker front-runs: buy $\Delta x_f$ ETH (price rises from $P$ to $P'$)
3. Victim's trade executes at $P'$ (worse price, up to slippage tolerance $s$)
4. Attacker back-runs: sell $\Delta x_f$ ETH at $P'' > P$ (price fell from trade)

Attacker profit: approximately $\Delta x_f \cdot (P'' - P) - 2 \times \text{gas fees}$

The victim's cost: they paid $P'$ instead of $P$ for their ETH — the difference is the "MEV tax."

**Flashbots** is the industry response: a private transaction pool that lets users submit transactions directly to validators, bypassing the public mempool. This prevents sandwich attacks (the bot can't see your transaction before it's confirmed) but introduces its own centralization concerns.

**Code moment:** No live MEV simulation (that would require a real blockchain node). Instead: show a simplified numerical example. Pool has 1000 ETH / 2,000,000 USDC. Victim wants to buy 10 ETH. Without sandwich: pays ~$20,100. With sandwich (attacker front-runs with 5 ETH): victim pays ~$20,250. Attacker profit: ~$45 after gas. Output: the table showing the price impact with and without the sandwich attack. The victim loses $150 on a $20,000 trade — 0.75% slippage from front-running alone.

**"So what?":** MEV is a tax on DeFi activity that doesn't exist in centralized exchanges (where the exchange operator prevents front-running, or at least is supposed to). If you're building DeFi strategies — especially large trades on AMMs — MEV protection is not optional. Understanding MEV also changes how you think about on-chain data: some of the "trading activity" you see is bots sandwiching other bots. Not all volume is genuine.

### Section 5: Crypto-Specific ML Opportunities
**Narrative arc:** We connect the data and market structure from the previous sections to concrete ML applications. Four opportunities, each with a clear edge source and a realistic assessment of alpha decay.

**Key concepts:** Funding rate mean-reversion, on-chain feature engineering, LP position optimization, cross-exchange arbitrage.

**The hook:** Here's the uncomfortable truth about crypto ML: the signals are stronger than in equities, but they decay faster. In 2019, a simple funding rate mean-reversion strategy on Binance BTC perpetuals returned 80% annually with a Sharpe above 3. By 2022, the same strategy returned 15% with a Sharpe of 0.8. By 2024, it was barely profitable after transaction costs. The signal was real. It was exploited. It decayed. This is the lifecycle of every crypto alpha signal — stronger initial returns than equities, faster convergence to efficiency. You need to run faster, not smarter.

**Key formulas:**

**Funding Rate Mean-Reversion Strategy:**

When funding rate $f_t$ exceeds a threshold (e.g., annualized $f_t > 50\%$):
- Longs are paying shorts excessively
- Go short the perpetual future
- Collect funding payments
- Expected return per funding period: $f_t$ (typically every 8 hours on Binance)

The trade: you earn the funding payment and bet that the price doesn't rally against your short by more than the funding you collected. In extreme funding environments ($f_t > 100\%$ annualized), the strategy has historically been profitable 75-85% of the time.

**On-Chain Feature Engineering for XGBoost:**

Combine market and on-chain features:

| Feature | Type | Intuition |
|---------|------|-----------|
| 5/10/20-day momentum | Market | Trend strength |
| 20-day realized vol | Market | Risk regime |
| Volume ratio (vs. 20-day avg) | Market | Unusual activity |
| MVRV z-score | On-chain | Valuation (overheated/undervalued) |
| Exchange flow ratio | On-chain | Selling pressure proxy |
| Address growth rate | On-chain | Adoption momentum |
| Funding rate z-score | Derivative | Crowding indicator |
| SOPR | On-chain | Holder profit/loss |

**Code moment:** Show the structure of a crypto feature pipeline using `ccxt` for market data and Glassnode free-tier API for on-chain data:

```python
import ccxt

exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=730)  # 2 years
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
```

Output: students see that `ccxt` provides a unified API across 100+ exchanges, no API key needed for public data. The data pipeline is simpler than `yfinance` — no need for tickers, just symbol pairs.

**"So what?":** The opportunity in crypto ML is real but time-limited. On-chain features provide genuinely new information that has predictive power. But the competitive landscape is accelerating: in 2020, most crypto traders were retail with no ML. In 2025, major quant funds (Jump Crypto, Wintermute, Alameda's successors) have entered. The window for simple signals is closing. Complex, multi-feature models — the kind you've been building all semester — are the current edge.

### Section 6: Key Differences for ML Practitioners — A Summary
**Narrative arc:** We close with a practical checklist of what changes when you move from equities to crypto. This is the "don't get caught" section — the differences that will break your code, ruin your backtest, or lose you money if you ignore them.

**Key concepts:** 24/7 data handling, higher regime-change frequency, wash trading and data quality, faster alpha decay.

**The hook:** A student in last year's cohort built a beautiful LSTM model for Bitcoin return prediction. It achieved a Sharpe of 2.1 in backtest. When they tried to paper-trade it, it immediately fell apart. The reason: their training data included weekends (crypto trades 24/7), but their rebalancing logic assumed 252 trading days per year (the equity convention). The model was making predictions for Saturday that were executed on Monday. A trivial calendar bug, caused by assuming crypto works like equities. It doesn't.

**Key formulas:** No new formulas — this is a checklist:

1. **Calendar:** 365 days/year, not 252. No overnight gap. No market close. Your `pd.bdate_range()` calls will be wrong.
2. **Volatility:** 60-80% annualized for BTC, not 15-20%. Your position sizing from Week 3 (which assumes equity-level vol) will put you in way too much risk.
3. **Transaction costs:** 2-10 bps on major CEXs, but 30-100 bps on DEXs (including gas fees and slippage). Your flat 5 bps assumption from equity backtests is too tight for DEX strategies.
4. **Data quality:** Wash trading is rampant on unregulated exchanges. Binance and Coinbase data is relatively clean; smaller exchanges may have 50-80% fake volume. Filter your data sources.
5. **Regime changes:** The Luna crash (May 2022) and FTX collapse (November 2022) are not fat-tail events in the equity sense — they're *structural breaks* where the entire market paradigm shifts. Your model needs to handle these, not just survive them.
6. **Alpha decay:** Signals that work for 2-3 years in equities may work for 6-12 months in crypto. Build retraining into your pipeline.

**Code moment:** Show the "gotcha" calendar fix:

```python
# WRONG (equity assumption):
trading_days = pd.bdate_range(start='2023-01-01', end='2023-12-31')  # 252 days

# RIGHT (crypto):
all_days = pd.date_range(start='2023-01-01', end='2023-12-31')  # 365 days
```

And the volatility adjustment:

```python
# Equity: annualize with sqrt(252)
# Crypto: annualize with sqrt(365)
btc_annual_vol = btc_daily_vol * np.sqrt(365)  # NOT sqrt(252)
```

**"So what?":** Crypto is not "equities with more volatility." It's a different market with different data, different microstructure, different participants, and different failure modes. Every technique from this course applies — XGBoost, LSTM, feature engineering, risk management, backtesting — but the implementation details change. This section is your migration guide.

### Closing Bridge

You've spent this week in a market that's simultaneously more transparent (public blockchain), more volatile (4x equities), less efficient (more alpha), and more dangerous (MEV, hacks, regulatory risk) than anything we've studied before. The techniques from the entire course transfer: XGBoost on on-chain features, LSTM on price sequences, risk management with crypto-adjusted volatility. But the data ecosystem is genuinely new — and on-chain analytics represents a category of features that simply doesn't exist in traditional finance. Next week — the finale — we return to the fundamental question that underlies everything: does your strategy actually work? We'll formalize backtesting methodology, confront the deflated Sharpe ratio, and build the capstone project that integrates at least two techniques from the entire course. Everything you've learned converges in Week 18.

## Seminar Exercises

### Exercise 1: Crypto Risk Calibration — Recalibrating Your Equity Intuitions
**The question we're answering:** If you applied your equity-trained risk models to crypto, how badly would they misjudge the risk?

**Setup narrative:** The lecture showed the distributional comparison between BTC and SPY — the tails, the volatility, the kurtosis. That was descriptive. This exercise is *prescriptive*: take the risk management tools from Week 3 (position sizing, VaR, max drawdown budgets) and show what happens when you naively apply equity-calibrated parameters to crypto. The goal is to build a crypto-specific risk calibration that you'll use in the homework.

**What they build:** Download 2 years of BTC/USDT and ETH/USDT via `ccxt` (building on the lecture's pipeline). Then: (a) compute the Week 3 position sizing rule (risk parity) using equity-calibrated vol (sqrt(252) annualization) vs. crypto-calibrated vol (sqrt(365)) — how much does the position size change? (b) Compute historical VaR at 1% and 5% for BTC vs. SPY — how much wider are crypto's tails? (c) Simulate a 60/40 BTC/ETH portfolio through the 2022 bear market using equity-calibrated drawdown stops (trigger at -10%) vs. crypto-calibrated stops (trigger at -25%). Which avoids whipsaw?

**What they'll see:** Equity-calibrated position sizing gives 3-4x too much BTC exposure. Historical VaR at 1% for BTC is roughly 3x wider than SPY. The equity-calibrated drawdown stop triggers constantly in crypto (15+ triggers in 2022), while the crypto-calibrated stop triggers 3-4 times. The wrong calendar convention (sqrt(252) vs. sqrt(365)) alone changes annual vol estimates by ~20%.

**The insight:** "Crypto is just more volatile equities" is a dangerous simplification. The distributional properties (fatter tails, no overnight gap, 24/7 risk exposure) require re-calibrating *every* risk parameter, not just scaling volatility. This exercise produces the calibrated risk parameters you'll need for the homework strategy.

### Exercise 2: On-Chain Feature Engineering — Multi-Horizon IC and Regime Stability
**The question we're answering:** At what time horizon and in which market regimes do on-chain features have the most predictive power?

**Setup narrative:** The lecture demonstrated that on-chain features have non-trivial IC against daily returns. But a single IC number hides critical structure: does MVRV predict better at daily, weekly, or monthly horizons? Is the signal stronger in bull markets or bear markets? This exercise runs the systematic analysis that a crypto quant fund would actually do before deploying capital.

**What they build:** Fetch on-chain metrics from Glassnode free tier (or provided CSV): active addresses, exchange net flows, MVRV ratio, funding rate. Merge with BTC price data. Compute: (a) IC at 1-day, 5-day, and 20-day forward horizons for each feature, (b) rolling 60-day IC to check stability, (c) regime-split IC: separate the data into "bull" (20-day return > 0) and "bear" (20-day return < 0) halves and compute IC in each regime separately. Produce an IC heatmap: features as rows, horizons as columns, with separate panels for bull/bear.

**What they'll see:** MVRV z-score's IC roughly doubles from daily to 20-day horizon (~0.04 to ~0.08). Funding rate z-score is the strongest at daily but decays at longer horizons (it's a short-term mean-reversion signal). Exchange flows are bear-market features — their IC is 2-3x stronger when markets are falling. Active addresses are almost useless at daily frequency but meaningful at monthly. The regime-split reveals that on-chain alpha is not uniform: it's concentrated in specific market states.

**The insight:** On-chain features are not "always predictive" — they're regime-and-horizon-specific. A crypto ML pipeline needs to match each feature to its optimal horizon, and ideally use a regime detector to weight features dynamically. This is the same "features are not equally useful everywhere" lesson from Week 5's SHAP analysis, applied to a genuinely new data category.

### Exercise 3: Impermanent Loss Simulation
**The question we're answering:** How expensive is it to provide liquidity on Uniswap, and does concentrated liquidity (V3) make it better or worse?

**Setup narrative:** If you've ever wondered "why would anyone provide liquidity to an AMM if there's impermanent loss?" — this exercise answers that question quantitatively. You'll simulate an LP position using real ETH price data and compute the net P&L: fees earned minus impermanent loss.

**What they build:** Using ETH/USDC historical prices, simulate a Uniswap V3 LP position with three range widths: narrow (price +/- 5%), medium (price +/- 15%), wide (price +/- 30%). For each, compute: total fees earned (assuming a fee tier and volume estimate), impermanent loss, net P&L vs. simply holding ETH. Run over a 90-day period.

**What they'll see:** Narrow range: highest fees per day in-range but highest IL and most time out-of-range. Wide range: lowest fees per day but most stable and least IL. Medium: the Goldilocks zone — usually the best net P&L. The result depends heavily on the volatility of the period: high-vol periods punish narrow ranges; low-vol periods reward them.

**The insight:** Concentrated liquidity is a bet on volatility — narrow ranges are short vol (you profit when price stays put, lose when it moves), wide ranges are closer to full-range (less sensitive to price). This connects directly to Week 2 (volatility) and Week 16 (market making): the LP is effectively a market maker whose spread is determined by the range width.

### Exercise 4: Discussion — What Transfers from TradFi to Crypto?
**The question we're answering:** Which techniques from Weeks 1-16 transfer directly, which need adaptation, and what's genuinely new?

**Setup narrative:** You've spent 16 weeks building an ML toolkit for equities. Now you're moving to crypto. This discussion maps the toolkit to the new domain and identifies gaps.

**What they discuss and document:**

- **Transfers directly:** XGBoost/LightGBM for tabular features, LSTM for sequences, Sharpe/drawdown metrics, SHAP for feature importance, walk-forward evaluation.
- **Needs adaptation:** Calendar (365 vs. 252), volatility scaling, transaction cost models (CEX vs. DEX), survivorship bias (crypto "deaths" are more dramatic — Luna, FTX).
- **Genuinely new:** On-chain features, AMM/LP dynamics, MEV awareness, funding rate signals, 24/7 risk management (no "close the book at 4 PM").

**The insight:** About 80% of your toolkit transfers without modification. The 20% that's different is mostly about data handling and risk calibration, not about new ML techniques. The genuine innovations are in the *feature set* (on-chain data) and the *market structure* (AMMs), not in the models themselves.

## Homework: "Crypto Return Prediction with On-Chain Features"

### Mission Framing

You're about to do something that wasn't possible in traditional finance until crypto came along: build a return prediction model that uses features derived from every transaction ever recorded on a public ledger. No Bloomberg terminal required. No WRDS subscription. No data licensing fees. Just a blockchain, a free API, and the same XGBoost you've been running all semester.

The mission has two parts. First, you'll build a crypto return prediction model and test whether on-chain features actually improve performance over price/volume features alone. This is the clean A/B test that everyone in crypto analytics claims to have run but few have published rigorously. Second, you'll simulate a Uniswap V3 LP position and compute whether providing liquidity is profitable after impermanent loss — the question that every DeFi participant faces and most answer incorrectly because they don't do the math.

The crypto market is more volatile, more transparent, and less efficient than equities. Your models from earlier weeks should work better here — higher signal-to-noise ratio, more predictive features. But the evaluation needs to be honest: higher returns come with higher risk, and the regime changes in crypto (Luna, FTX) are far more violent than anything in equities. Your walk-forward test will include at least one crypto crash. How your model handles it is the real test.

### Deliverables

1. **Market Data Pipeline (20 min):** Download 2 years of daily BTC and ETH OHLCV from Binance via `ccxt`. Compute: returns, 5/10/20-day momentum, 20-day realized volatility, volume ratio (current vs. 20-day average). Verify data quality: no gaps (crypto trades 24/7), no outliers from exchange glitches.

2. **On-Chain Feature Pipeline (30 min):** From Glassnode free tier or a provided dataset, obtain: BTC active addresses (daily), exchange net flows (daily), MVRV ratio (daily), funding rate (Binance BTC perpetual, 8-hourly aggregated to daily). Merge with price data. Compute z-scores for all on-chain features (standardize using a trailing 60-day window — no look-ahead).

3. **Feature Engineering (30 min):** Build a feature matrix with 10-15 features:
   - Market: 5-day momentum, 10-day momentum, 20-day momentum, 20-day vol, volume ratio, RSI(14)
   - On-chain: MVRV z-score, exchange flow ratio, active address growth rate, funding rate z-score, SOPR (if available)
   - Compute the IC (Spearman correlation) of each feature vs. next-day BTC return. Rank features by |IC|.

4. **Model Comparison (45 min):**
   - Model A: XGBoost on market features only (momentum, vol, volume)
   - Model B: XGBoost on market + on-chain features
   - Model C: LSTM on raw price sequences (20-day windows, similar to Week 8)
   - Model D: LSTM with on-chain features as additional inputs
   - Use expanding-window walk-forward evaluation (retrain monthly). Report: IC, directional accuracy, and Sharpe of a long/flat strategy (long when predicted return > 0, flat otherwise).

5. **Strategy Evaluation (30 min):** For the best model, build a simple long/flat strategy with 10 bps transaction costs per side. Report: annualized return, Sharpe, max drawdown, Calmar ratio. Compare against buy-and-hold BTC over the same period.

6. **Uniswap V3 LP Simulation (45 min):** Using ETH/USDC historical prices over 90 days:
   - Simulate a concentrated LP position with three range widths: narrow (0.5x ATR), medium (1.5x ATR), wide (3x ATR)
   - Assume a 0.3% fee tier and estimate fees from daily volume (use ETH/USDC Uniswap volume data or a reasonable estimate)
   - Compute for each: total fees earned, impermanent loss, net P&L, net P&L vs. holding ETH, net P&L vs. holding 50/50 ETH/USDC
   - Which range width is most profitable? Does the answer change if you pick a high-vol vs. low-vol 90-day window?

7. **Deliverable:** Complete notebook with: data pipelines, on-chain feature analysis with IC rankings, 4-model comparison table, walk-forward equity curves, LP simulation results. Include a 300-word section: "Do on-chain features add alpha, or is it noise?"

### What They'll Discover

- Model B (XGBoost + on-chain) will likely outperform Model A (XGBoost, market only) by 0.1-0.3 Sharpe points. The improvement is real but modest at daily frequency — on-chain features are more predictive at weekly/monthly horizons.
- The LSTM models (C, D) will likely underperform XGBoost on daily crypto data. Crypto returns have less autocorrelation structure than equities (the 24/7 market structure means there's no overnight gap to exploit), which reduces LSTM's advantage.
- Funding rate z-score will likely be the strongest single feature, with an IC of 0.06-0.10. MVRV z-score is the second strongest. Pure price/volume features will have lower IC.
- In the LP simulation, the medium range width (1.5x ATR) will likely win during normal volatility periods, but the wide range width wins during high-vol periods (it stays in range). Narrow ranges are risky: a single 5% move takes them out of range, and they earn zero fees until the price returns.
- The buy-and-hold benchmark is hard to beat in crypto bull markets but easy to beat in bear markets (the long/flat strategy avoids the worst drawdowns).

### Deliverable
A complete Jupyter notebook containing: `ccxt` data pipeline, on-chain feature engineering with IC analysis, 4-model walk-forward comparison, long/flat strategy evaluation, Uniswap V3 LP simulation with three range widths, and a written analysis section on the value of on-chain features.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Crypto market structure & data | Demo: pull BTC/SPY via ccxt, compare return distributions | Exercise 1: crypto risk calibration (position sizing, VaR, drawdown stops recalibrated) | Build: full ccxt data pipeline with quality checks |
| On-chain features (MVRV, exchange flows, etc.) | Demo: compute Spearman IC for each feature vs. daily returns | Exercise 2: multi-horizon IC + regime-split analysis (bull/bear), IC heatmap | At scale: feature matrix with 10-15 features, IC ranking, 4-model comparison |
| AMMs & constant product formula | Demo: simulate Uniswap V2 pool, show IL curve | Exercise 3: Uniswap V3 concentrated liquidity with 3 range widths, net P&L analysis | At scale: 90-day LP simulation with ATR-based ranges, vol-dependent profitability |
| MEV & sandwich attacks | Demo: numerical sandwich attack example, cost breakdown | Not covered (conceptual, done in lecture) | Not directly tested (awareness integrated into strategy design) |
| Crypto-specific ML adaptations | Demo: calendar fix (365 vs. 252), vol adjustment code | Exercise 1: quantify the cost of getting calibration wrong | Integrate: all models use sqrt(365), crypto-calibrated costs |
| XGBoost on crypto features | Demo: feature pipeline structure with ccxt | Not repeated (technique from W5) | Build: XGBoost with/without on-chain features, A/B comparison |
| LSTM on crypto sequences | Not covered (technique from W8) | Not covered | Build: LSTM on raw prices + LSTM with on-chain, compare vs. XGBoost |
| TradFi-to-crypto transfer | Not covered (reserved for seminar discussion) | Exercise 4: systematic mapping of what transfers, what needs adaptation, what's new | Integrate: written analysis on on-chain alpha |

## Key Stories & Facts to Weave In

- **The Terra/Luna Collapse (May 2022):** $40 billion in value destroyed in one week. UST (an algorithmic stablecoin) depegged from $1.00 to $0.02. LUNA went from $80 to $0.0001. On-chain data showed the bank run developing 72 hours before the media reported it: massive Anchor protocol withdrawals, spiking exchange inflows, and LUNA selling pressure from the Luna Foundation Guard's failed defense. Anyone monitoring on-chain metrics had a 3-day head start.

- **The FTX Collapse (November 2022):** FTX, the third-largest crypto exchange, collapsed after a CoinDesk article revealed that Alameda Research (FTX's sister trading firm) held $5.8 billion in illiquid FTT tokens as "assets." Customer withdrawals exceeded $6 billion in 72 hours. On-chain data showed massive outflows from FTX wallets days before the public announcement. Sam Bankman-Fried was later convicted of fraud.

- **The Bitcoin Pizza (May 22, 2010):** Laszlo Hanyecz paid 10,000 BTC for two Papa John's pizzas — the first real-world Bitcoin transaction. At today's prices, those pizzas cost approximately $600 million. The transaction is permanently recorded on the blockchain at block 57043. On-chain analytics can trace the coins' journey from that day to today.

- **Uniswap's Daily Volume (2024):** Uniswap, a decentralized exchange running on smart contracts with no employees processing trades, regularly handles $1-3 billion in daily trading volume. For comparison, the London Stock Exchange handles about $5-7 billion daily. A smart contract with a few hundred lines of Solidity code processes a meaningful fraction of global exchange volume.

- **MEV in Numbers (2023):** MEV bots extracted an estimated $600 million from Ethereum transactions in 2023. The most profitable single MEV transaction ever: approximately $2 million in a single block, from a complex arbitrage across three DeFi protocols. Flashbots (the MEV mitigation protocol) now processes over 90% of Ethereum blocks through its MEV-protected relay.

- **Glassnode and the On-Chain Analytics Industry:** Founded in 2018, Glassnode tracks 800+ on-chain metrics for Bitcoin and Ethereum. Professional subscriptions cost $799/month. The free tier provides enough data for this homework. Competitors include CryptoQuant, IntoTheBlock, and Santiment. The industry exists because blockchain transparency creates a data category that didn't exist before 2009.

- **Jump Crypto's $300M Loss (2022):** Jump Crypto, the crypto arm of Jump Trading (one of the largest traditional market makers), lost an estimated $300 million in the 2022 crypto downturn, partly through exposure to Terra/Luna. Even sophisticated TradFi firms underestimated crypto's tail risk. The lesson: crypto volatility is not just "higher vol" — it includes existential risk to entire protocols.

## Cross-References
- **Builds on:** Week 5 (XGBoost — same algorithm, new features), Week 8 (LSTM — same architecture, new data), Week 1 (market structure — AMMs are a fundamentally different market structure), Week 16 (market making — LP on an AMM is market making via math formula).
- **Sets up:** Week 18 (Capstone — a crypto strategy is a legitimate capstone project, especially one that uses on-chain features not available in traditional finance).
- **Recurring thread:** The "data tells a story" theme reaches its most dramatic expression: in crypto, *every* transaction is public. The blockchain is the ultimate dataset — complete, timestamped, immutable, and free. The challenge is not access but interpretation.

## Suggested Reading
- **CoinGecko Annual Report (2024):** A comprehensive overview of the crypto market: exchange volumes, DeFi TVL, stablecoin market caps, NFT activity. Read it for the numbers — it's the best free source for understanding the scale and structure of crypto markets.
- **Adams et al., "Uniswap V3 Core" (Whitepaper, 2021):** The technical specification of concentrated liquidity. Surprisingly readable for a protocol whitepaper. Focus on Sections 2 (concentrated liquidity) and 6 (oracle). Understanding the math behind the AMM is essential for anyone building DeFi strategies.
- **Easley, O'Hara & Lopez de Prado, "Microstructure of Cryptocurrency Markets" (2019):** The bridge between traditional microstructure theory and crypto markets. The same economists who developed VPIN applied their framework to Bitcoin. The result: crypto markets have higher informed trading probability and faster price discovery than equities, consistent with the higher signal-to-noise ratio we observe.
