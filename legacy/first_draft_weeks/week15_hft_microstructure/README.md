# Week 15 — HFT & Market Microstructure ML

> **You will never build an HFT system in Python. But understanding what happens at the microsecond level will make every model you've built in this course better.**

## Prerequisites
- **Week 1 (Markets & Data Structures):** The order book concept, bid-ask spread, market vs. limit orders. We introduced these briefly; now we go deep.
- **Week 3 (Portfolio Theory & Risk):** Transaction cost awareness. You've been subtracting 5-10 bps per side in your backtests — now you'll understand *where* those costs come from, and why 5 bps is optimistic for most strategies.
- **Week 5 (Tree-Based Methods):** Feature engineering. Microstructure features (order flow imbalance, VPIN) are among the most powerful predictive features in finance — and they're computed the same way you'd compute any feature for XGBoost.
- **Week 7 (Feedforward Nets):** CNN and LSTM architectures. DeepLOB uses a CNN to process limit order book snapshots, followed by an LSTM for temporal dynamics. You have the building blocks.
- **General:** Comfort with the idea that financial data has multiple resolutions — daily (what we've used so far), minute-level, tick-level, and the raw message stream. This week, we zoom in to the finest resolution.

## The Big Idea

Every model you've built in this course operates on daily OHLCV data — four prices and one volume number per stock per day. Here's a number that should recalibrate your entire understanding of financial data: for a liquid stock like Apple, the actual market generates about *100,000 events per second* during trading hours. Limit order submissions, cancellations, modifications, trades. By the time you see a daily OHLCV bar, the data has been compressed by a factor of roughly 40,000 to 1. That's like training an image classifier on a single pixel per image and wondering why your accuracy is mediocre.

This week is not about building an HFT system. You can't, and we won't pretend otherwise. The infrastructure costs $10 million or more per year — co-located servers in the exchange data center ($12,000/month for a CME 10-gigabit connection), FPGA cards that process market data in under a microsecond ($5,000-$50,000 per card), microwave towers that shave 3 milliseconds off the New York-to-Chicago link by traveling at the speed of light through air instead of fiber. Python's interpreter overhead puts a floor of 10-20 microseconds on any decision — and by the time your Python script has decided what to do, a Xilinx FPGA at the exchange has already processed 15,000 messages and placed orders on half of them.

Instead, this week is about understanding the *structure* beneath the data you've been using all semester. We'll think of market microstructure as three concentric rings. The inner ring is HFT execution — sub-10-microsecond, hardware-driven, where ML is irrelevant and C++/FPGA is everything. The middle ring is execution quality — seconds to minutes, where ML adds significant value for optimal order routing, trade timing, and impact estimation. This is where quant traders actually apply ML to microstructure, and it's accessible to you. The outer ring is microstructure-informed longer-horizon models — where understanding order flow dynamics, spread behavior, and liquidity patterns makes every model you've ever built meaningfully better. This is where 99% of you will apply what you learn today, and it's transformative.

We'll work with the FI-2010 benchmark dataset — the only freely available limit order book dataset in the world, covering 5 Finnish stocks over 10 trading days. It's tiny by HFT standards and ancient by data-science standards. But it's enough to build a DeepLOB model, verify Cont's OFI theory, and understand *what you're missing* when you work with daily data. The limitation of the data is the reason there's no homework this week — and the awareness of that limitation is itself a lesson.

## Lecture Arc

### Opening Hook

On May 6, 2010, at precisely 2:32 PM Eastern, the Dow Jones Industrial Average began falling. Within five minutes, it dropped 600 points. Then, in the next five minutes, it dropped another 600 points — a total decline of nearly 1,000 points, or about 9%, in ten minutes. Procter & Gamble, one of the safest stocks on Earth, briefly traded at a penny. Accenture traded at a penny. Apple briefly quoted at $100,000 per share. Then, just as abruptly, the market recovered. By 3:00 PM, most of the decline had been erased. The Flash Crash wasn't caused by a rogue trader or a terrorist attack. It was caused by a single mutual fund — Waddell & Reed — executing a $4.1 billion sell order using an algorithm that paid no attention to the price impact of its own trading. The algorithm kept selling into a market with no buyers. Every market maker pulled their quotes. The order book — the thing we're about to study — emptied out. And for nine terrifying minutes, the most sophisticated market in the world functioned like a medieval bazaar where nobody knew what anything was worth.

### Section 1: What HFT Firms Actually Do
**Narrative arc:** We demystify HFT by explaining the three main strategies. The tension: most people think HFT is "buying and selling really fast." It's not — it's about providing liquidity (market making), capturing tiny mispricings (latency arbitrage), and processing information faster than anyone else (news-based trading). The resolution: understanding the economic role of HFT changes how you think about markets.

**Key concepts:** Market making (passive liquidity provision), latency arbitrage (cross-exchange price discrepancies), statistical arbitrage at microsecond timescales, news-based trading.

**The hook:** Virtu Financial, one of the world's largest HFT firms, went public in 2015 and disclosed something remarkable: in five years of trading, they had lost money on only *one* day. Not one year — one *day*. Out of roughly 1,250 trading days. Their average daily revenue: about $1 million. Their secret wasn't predicting where the market would go. It was being the fastest market maker — providing liquidity to everyone else and earning the bid-ask spread, pennies at a time, millions of times per day. That one losing day? They suspect it was a data feed glitch.

**Key formulas:** No formulas yet — this section is narrative-driven. The key numbers:

- Market making accounts for 50-60% of US equity volume
- Average HFT profit: 0.1-0.5 cents per share traded
- Latency arbitrage window: typically 10-100 microseconds
- Aquilina, Budish & O'Neill (QJE 2022) estimated the latency arbitrage "tax" at approximately 0.5 basis points of total trading costs

**Code moment:** No code in this section — it's a conceptual foundation. The code starts in Section 4 when we touch data.

**"So what?":** When you build a strategy that trades daily and assume 5 bps transaction costs, you're making an assumption about the HFT ecosystem operating correctly. If liquidity dries up (as it did in the Flash Crash), your 5 bps cost can become 500 bps. Understanding *who* is on the other side of your trades — and what might make them stop — is risk management 101.

### Section 2: The Infrastructure Stack — Why Python Can't Compete
**Narrative arc:** We give students a visceral sense of the hardware gap. This isn't about making them feel bad about Python — it's about calibrating expectations. The resolution: Python is the right tool for research and backtesting; C++/FPGA is the right tool for execution. Knowing the boundary prevents you from wasting time on the wrong problem.

**Key concepts:** Co-location, FPGA order routing, kernel bypass (DPDK, RDMA), microwave towers, the Global Interpreter Lock.

**The hook:** In 2010, Spread Networks spent $300 million to lay a new fiber optic cable between New York and Chicago. Its advantage: it was 3 milliseconds faster than the existing route — 13.1ms vs. 16.0ms. The cable was profitable for about two years, until someone realized that microwaves travel through air at the speed of light (300 meters per microsecond), while light through fiber is about 30% slower (200 meters per microsecond). Microwave towers now connect every major exchange pair. The latency from Chicago to New York via microwave: 3.98 milliseconds. Via fiber: 6.55 milliseconds. Traders pay millions per year to save 2.57 milliseconds. In the time it takes you to blink (about 300 milliseconds), an HFT firm has completed roughly 75,000 round trips.

**Key formulas:** The latency stack, quantified:

| Layer | Latency | Who Pays |
|-------|---------|----------|
| FPGA tick-to-trade | 100-800 nanoseconds | HFT firms ($5K-$50K/card) |
| C++ application | 1-10 microseconds | Prop trading firms |
| Python with NumPy | 10-20 microseconds (minimum) | You, in this course |
| Python pure loop | 50-500 microseconds | Nobody in production |
| Network (co-located) | 1-5 microseconds | $12K/month (CME 10G) |
| Network (cross-city) | 4-7 milliseconds | Microwave: $millions/year |

Python's three performance killers:
1. **GIL (Global Interpreter Lock):** Only one thread can execute Python bytecode at a time. Parallel processing? Not in Python.
2. **Garbage collector:** A GC pause of 2ms is an eternity when your competitors react in 100ns. You can't predict when GC runs.
3. **Interpreter overhead:** Every operation goes through the interpreter. A C++ loop over 1000 elements: ~1 microsecond. Python: ~100 microseconds. The floor is the floor.

**Code moment:** A timing comparison to make the gap visceral:

```python
import time
import numpy as np

# "Can Python react faster than the market?"
start = time.perf_counter_ns()
decision = np.argmax(np.random.randn(100))  # simplest possible "decision"
elapsed_us = (time.perf_counter_ns() - start) / 1000
print(f"Python decision: {elapsed_us:.1f} μs")
# Typically: 15-40 μs
# FPGA: 0.1-0.8 μs
# Ratio: ~50-400x slower
```

Output: students see that even the simplest possible NumPy operation takes 15-40 microseconds — 50 to 400 times slower than an FPGA. And that's before you add any actual logic.

**"So what?":** Python is for *research*: backtesting, feature engineering, model training, strategy analysis. It is not for *execution* at HFT timescales. The industry practice is: research in Python (or R, or MATLAB), execution in C++ (or FPGA/Verilog). Knowing this boundary saves you from building something that's too slow to be useful and too complex to be research.

### Section 3: LOB Data — What You're Missing with Daily OHLCV
**Narrative arc:** We show students the actual data that HFT systems consume — the limit order book — and quantify exactly how much information is lost when you compress it to OHLCV. The resolution: understanding what's in the LOB changes how you think about features, even for daily models.

**Key concepts:** Limit order book (LOB), message types (add, cancel, modify, execute), LOB snapshots, best bid/offer (BBO), depth, queue position.

**The hook:** Here's a concrete example of the 40,000:1 compression. On a typical trading day for Apple (AAPL), the OHLCV bar tells you: Open $192.30, High $194.10, Low $191.80, Close $193.50, Volume 62 million shares. What actually happened: approximately 2.4 billion LOB messages — order submissions, cancellations, modifications, and trades — each timestamped to the nanosecond. The spread oscillated between 1 cent and 5 cents, widening at 10:00 AM (when economic data was released), tightening at 11:30 AM (lunch lull, less adverse selection), and widening again at 3:45 PM (end-of-day rebalancing). There were three brief "liquidity holes" where the book was thin and a market order would have moved the price 10 cents instead of 1 cent. None of this — absolutely none of it — is visible in the OHLCV bar.

**Key formulas:**

The limit order book at time t is a collection of price-quantity pairs:

$$\text{Bid side: } \{(p_1^b, q_1^b), (p_2^b, q_2^b), \ldots\} \quad p_1^b > p_2^b > \ldots$$
$$\text{Ask side: } \{(p_1^a, q_1^a), (p_2^a, q_2^a), \ldots\} \quad p_1^a < p_2^a < \ldots$$

The best bid $p_1^b$ and best ask $p_1^a$ define the **spread**: $s = p_1^a - p_1^b$.

The **mid-price**: $m = (p_1^a + p_1^b) / 2$.

For Apple, the spread is typically 1 cent on a $190 stock — 0.005%, or half a basis point. For a micro-cap, it might be 50 cents on a $5 stock — 10%. That spread is not a number — it's the tax you pay on every trade.

**Code moment:** Load the FI-2010 dataset. Visualize a LOB snapshot — the "depth chart" showing bid and ask quantities at each price level, like two mountains facing each other. Animate it over 100 timesteps. Output: students see the LOB breathing — orders appearing, disappearing, the spread tightening and widening. The visual makes the LOB *tangible* in a way that numbers alone cannot.

**"So what?":** Even if you never trade at HFT speeds, *knowing what the LOB looks like* changes how you think about transaction costs. That flat 5 bps you've been using in your backtests? It's an average that hides enormous variation — tight during midday on a blue-chip, wide at market open on a small-cap, and potentially infinite during a crisis. Microstructure awareness is the difference between a backtest that's "roughly right" and one that's "precisely wrong."

### Section 4: Order Flow Imbalance — The Most Important Microstructure Feature
**Narrative arc:** We introduce Cont et al.'s (2014) Order Flow Imbalance — the single most important feature in market microstructure. It's beautifully simple: count the net volume arriving at the best bid vs. best ask. This predicts short-term price changes with a linear relationship that holds across stocks, exchanges, and time periods.

**Key concepts:** Order Flow Imbalance (OFI), trade imbalance, price impact, informed vs. uninformed flow, adverse selection.

**The hook:** In 2014, Rama Cont, Arseniy Kukanov, and Sasha Stoikov published a paper with a deceptively simple claim: you can predict short-term price changes using a single number — the difference between buy-initiated and sell-initiated order flow at the best bid and offer. They called it Order Flow Imbalance. The relationship is approximately linear: ΔP ≈ beta * OFI. That's it. One feature, one coefficient, and it explains 60-70% of the variance in contemporaneous price changes. No machine learning needed. No neural nets. Just counting.

**Key formulas:**

Order Flow Imbalance at time t:

$$\text{OFI}_t = \underbrace{(\Delta q_t^b \cdot \mathbf{1}_{p_t^b \geq p_{t-1}^b})}_{\text{bid-side pressure}} - \underbrace{(\Delta q_t^a \cdot \mathbf{1}_{p_t^a \leq p_{t-1}^a})}_{\text{ask-side pressure}}$$

Simplified version (trade-based):

$$\text{OFI}_t^{\text{trade}} = \sum_{i \in \text{buys}} v_i - \sum_{i \in \text{sells}} v_i$$

The linear price impact model:

$$\Delta P_t = \alpha + \beta \cdot \text{OFI}_t + \epsilon_t$$

Cont et al. found beta > 0 universally — more buy pressure pushes the price up — and the R² is remarkably high (0.6-0.7 for 10-second intervals). This isn't a prediction model in the usual sense — it's a contemporaneous relationship — but it forms the basis for short-term forecasting and, critically, for understanding *why* prices move.

**Code moment:** Using the FI-2010 data, compute OFI at 10-second intervals. Regress mid-price changes on OFI. Plot the scatter with a regression line. Output: a clean linear relationship with R² ~ 0.5-0.65 (FI-2010 is less liquid than US equities, so the R² is slightly lower). The plot should surprise students — they've spent 14 weeks building complex models to predict returns, and here a single feature with a linear model explains 60% of short-term price changes.

**"So what?":** OFI won't help you predict tomorrow's close — it's a microstructure feature, not a return predictor. But it tells you something profound: at the shortest timescales, prices don't move because of "news" or "fundamentals." They move because someone is buying more aggressively than someone else is selling. Understanding this changes how you think about your own trading: when you place a market order, *you* are the OFI signal that moves the price against you. That's market impact, and it's the largest hidden cost in your backtest.

### Section 5: DeepLOB — Deep Learning on the Order Book
**Narrative arc:** We move from the single-feature OFI model to a deep learning approach that takes the entire LOB as input. DeepLOB (Zhang, Zohren & Roberts, 2019) uses a CNN to process LOB snapshots and an LSTM for temporal dynamics. F1 ~83% on the FI-2010 benchmark. We'll build a simplified version.

**Key concepts:** LOB feature extraction, CNN for spatial (price-level) features, LSTM for temporal features, multi-class classification (up/down/stationary), the FI-2010 benchmark.

**The hook:** The FI-2010 dataset is the ImageNet of market microstructure. Created by Adamantios Ntakaris and colleagues at the University of Turku, it contains 10 days of limit order book data for 5 Finnish stocks (Nokia, Outokumpu, Sampo, Rautaruukki, Wärtsilä) from the NASDAQ Nordic exchange. It's small — about 450,000 LOB snapshots — but it's the only freely available LOB benchmark in the world. And the benchmark results are clear: classical ML (SVM, random forest) achieves F1 around 60-65%. DeepLOB achieves ~83%. The most recent transformer-based model (TLOB, 2025) pushes that to ~87%. Deep learning on raw LOB data is a genuine win.

**Key formulas:**

DeepLOB input: for each timestep, the top 10 levels of the LOB:

$$X_t = \begin{bmatrix} p_1^a, q_1^a, p_1^b, q_1^b \\ p_2^a, q_2^a, p_2^b, q_2^b \\ \vdots \\ p_{10}^a, q_{10}^a, p_{10}^b, q_{10}^b \end{bmatrix} \in \mathbb{R}^{10 \times 4}$$

A sequence of T such snapshots: $\{X_{t-T+1}, \ldots, X_t\} \in \mathbb{R}^{T \times 10 \times 4}$.

Target: mid-price direction at horizon k events:

$$y_t = \begin{cases} +1 & \text{if } m_{t+k} > m_t (1 + \alpha) \\ -1 & \text{if } m_{t+k} < m_t (1 - \alpha) \\ 0 & \text{stationary} \end{cases}$$

where alpha is a small threshold (0.002 in the standard benchmark) to filter out noise.

DeepLOB architecture: Conv1D layers along the price-level dimension (extracting spatial features like depth imbalance), followed by an LSTM along the time dimension (capturing temporal dynamics like order flow momentum). Output: softmax over 3 classes.

**Code moment:** Build a simplified "DeepLOB-lite" — a CNN with 2 convolutional layers followed by a 1-layer LSTM. Train on FI-2010 with an 80/20 split. Report F1 per class and overall. Output: F1 around 70-75% for the simplified version (vs. ~83% for the full DeepLOB). The "stationary" class is hardest to predict. Students see that deep learning on LOB data genuinely works, but also that the simple version captures most of the signal.

**"So what?":** DeepLOB proves that deep learning can extract signal from raw market microstructure data. But — and this is critical — the FI-2010 benchmark covers 5 Finnish stocks over 10 days. Generalizing to US equities, different market regimes, and longer horizons is an open research problem. The model works beautifully on the benchmark; whether it works in production is a question nobody has publicly answered.

### Section 6: VPIN — The Flash Crash Predictor
**Narrative arc:** We close with the most dramatic application of microstructure ML: predicting liquidity crises before they happen. VPIN (Volume-Synchronized Probability of Informed Trading), developed by Easley, Lopez de Prado, and O'Hara, spiked hours before the 2010 Flash Crash. It's not ML in the neural net sense — it's a handcrafted feature — but it's arguably the most important real-time risk metric in market microstructure.

**Key concepts:** Probability of informed trading (PIN), volume-synchronized sampling, order flow toxicity, liquidity risk.

**The hook:** On May 6, 2010, VPIN for E-mini S&P 500 futures had been rising steadily since 10:00 AM — hours before the Flash Crash hit at 2:32 PM. By noon, it was in the 95th percentile of its historical distribution. A trading desk monitoring VPIN would have known, hours in advance, that liquidity was draining from the market and that the risk of a dislocation was elevated. Most desks weren't monitoring VPIN. Easley, Lopez de Prado, and O'Hara published their analysis in 2012. The title: "Flow Toxicity and Liquidity in a High-Frequency World." VPIN is now standard at most large market-making firms. Lopez de Prado later said it was the paper he was most proud of.

**Key formulas:**

VPIN is computed on volume bars (not time bars — this connects to Week 1's alternative bars):

1. Partition trades into buy-initiated and sell-initiated (using the tick rule or bulk classification)
2. Aggregate into volume bars of size V
3. For each bar n:

$$\text{VPIN}_n = \frac{\sum_{\tau=n-N+1}^{n} |V_\tau^B - V_\tau^S|}{N \cdot V}$$

In words: the average absolute order flow imbalance over the last N volume bars, normalized by bar size. High VPIN means flow is directional (informed traders dominating). Low VPIN means flow is balanced (normal two-sided trading).

The connection to dollar bars from Week 1: VPIN works best on volume-synchronized data because the "informed trader" concept is about volume, not time. When a large informed order hits the market, it generates volume. VPIN captures this in real-time.

**Code moment:** Compute VPIN on the FI-2010 data (or on daily SPY volume data as a simplified version). Show the time series. Highlight the days with the highest VPIN readings. Output: students see that VPIN spikes *before* large price moves, not after. This is a leading indicator for liquidity risk.

**"So what?":** VPIN connects two themes from the course: alternative sampling from Week 1 (volume bars) and real-time risk monitoring from Week 3. It's a microstructure feature that works at any timescale — from intraday (its original use case) to daily (as a risk flag for longer-horizon models). Adding VPIN or VPIN-like features to your daily models from Weeks 4-5 can improve drawdown-adjusted performance. We'll see this in Week 18 when we build the full strategy pipeline.

### Closing Bridge

You've spent this week zooming in — from the daily bars you've used all semester to the microsecond-level data that actually drives markets. The three rings framework is your guide: you'll never compete in the inner ring (leave that to the FPGAs), but the middle ring (execution quality) and outer ring (microstructure-informed modeling) are yours for the taking. Next week, we combine what you learned here with the RL framework from Week 13 to tackle the quintessential HFT strategy: market making. You'll implement the Avellaneda-Stoikov model — the elegant analytical solution — and then see whether an RL agent can improve on it. The microstructure intuition from this week will make every formula in Week 16 more meaningful.

## Seminar Exercises

### Exercise 1: LOB Feature Engineering — Quantifying What You See
**The question we're answering:** Can you extract quantitative features from the limit order book that capture the dynamics the lecture visualization showed?

**Setup narrative:** The lecture showed you the LOB depth chart animation — the mountains breathing, the spread oscillating, the liquidity holes appearing. That was visual intuition. Now you need to turn that intuition into *numbers*. This exercise moves from "seeing the LOB" to "measuring the LOB" — building the feature set that feeds into DeepLOB and every microstructure model.

**What they build:** Load the FI-2010 dataset. Compute five LOB features at each timestamp: (a) bid-ask spread, (b) depth imbalance (total bid volume minus total ask volume across 10 levels), (c) volume-weighted mid-price (weighted by depth at best levels), (d) book pressure ratio (volume at best bid / volume at best ask), (e) spread-to-depth ratio (spread divided by total depth — a liquidity quality metric). Plot each feature's time series over 5,000 timestamps. Compute the correlation between each feature and the next-event mid-price change.

**What they'll see:** Depth imbalance and book pressure ratio have the highest correlation with short-term price moves (~0.2-0.4). The spread-to-depth ratio spikes before large price moves — a proxy for liquidity risk. These five features are the building blocks of every microstructure model.

**The insight:** The LOB is not just something to visualize — it's a rich feature space. The five features you just computed are more predictive of short-term price changes than any daily OHLCV feature you've used all semester. Understanding *why* (they capture real-time supply/demand dynamics) is the conceptual breakthrough.

### Exercise 2: DeepLOB — Raw LOB vs. Engineered Features
**The question we're answering:** Does a CNN on raw LOB snapshots outperform a simpler model on the hand-crafted features you just built?

**Setup narrative:** The lecture demonstrated that a CNN+LSTM on raw LOB data achieves ~70-75% F1. You've now built five hand-crafted LOB features in Exercise 1. The natural question: does the neural net discover something beyond what your features capture? This exercise runs the comparison that matters — deep learning vs. feature engineering on the same prediction task.

**What they build:** Two models, same target (mid-price direction at k=10 events, 3 classes): (a) the lecture's DeepLOB-lite architecture (2 Conv1D + LSTM) on raw LOB snapshots, (b) a simple XGBoost classifier on the 5 hand-crafted features from Exercise 1 plus OFI. Train on 80% of FI-2010, test on 20%. Report: F1 per class, confusion matrix, overall accuracy for both. Compute: which specific examples does the CNN get right that XGBoost misses, and vice versa?

**What they'll see:** The CNN achieves F1 ~70-75%, XGBoost on features achieves F1 ~60-68%. The CNN wins overall, but XGBoost is competitive on the "up" and "down" classes — the gap is mostly in "stationary" predictions, where the CNN's temporal modeling helps. The error analysis reveals that the CNN captures *temporal patterns* (sequences of imbalance changes) that the point-in-time features miss.

**The insight:** Deep learning on LOB data genuinely extracts more signal than hand-crafted features — but not dramatically more. The marginal improvement is in temporal patterns. For practitioners who can't afford the CNN's compute cost (especially at HFT speeds), the hand-crafted features capture most of the value. This is the classic "complexity vs. interpretability" tradeoff from Week 5, applied to microstructure.

### Exercise 3: Verifying Cont's OFI Theory
**The question we're answering:** Does the linear relationship between Order Flow Imbalance and price changes hold in the FI-2010 data?

**Setup narrative:** Cont et al. (2014) claimed a remarkably clean linear relationship: ΔP ≈ beta * OFI. That's an extraordinary claim for financial data, which is usually noisy and nonlinear. We'll verify it ourselves and measure the R².

**What they build:** Compute OFI from the FI-2010 order book changes at 1-second, 10-second, and 60-second aggregation intervals. Regress mid-price changes on OFI at each interval. Plot the scatter with regression line and report R².

**What they'll see:** At 10-second intervals: R² around 0.5-0.65. At 1-second: R² around 0.3-0.4 (noisier). At 60-second: R² around 0.4-0.5 (signal decays). The relationship is strikingly linear. The slope (beta) varies by stock — more liquid stocks have lower beta (less price impact per unit of flow imbalance).

**The insight:** This is perhaps the most reliable quantitative relationship in market microstructure. Unlike return prediction (where R² above 0.01 is impressive), OFI explains more than half the variance in short-term price changes. The difference: return prediction is a forecasting problem (hard); OFI is a contemporaneous relationship (easier). But OFI with a small lead still has predictive power — and that's what execution algorithms exploit.

### Exercise 4: Discussion — Microstructure Knowledge for Daily Models
**The question we're answering:** What microstructure knowledge improves the daily-frequency models we've been building all semester?

**Setup narrative:** This is the bridging exercise. Everything we've learned today operates at microsecond-to-minute timescales. But most of you will work at daily or weekly frequencies. The question is: what microstructure knowledge transfers up?

**What they discuss and document:**
- **Transaction cost modeling:** Replace flat 5 bps with spread-dependent costs. Use average daily spread as a feature (wide spread = expensive to trade = potential alpha in providing liquidity).
- **Volume profiles:** Intraday volume follows a U-shape (high at open and close, low at lunch). A model that trades at the right time of day can save 20-50% on transaction costs.
- **VPIN as a risk signal:** Daily VPIN can be computed from daily volume data. High VPIN days precede volatility spikes — useful as a regime indicator for your models.
- **Limit order book depth as a liquidity feature:** Average daily depth (when available) predicts next-day volatility better than historical volatility alone.

**The insight:** You don't need tick data to benefit from microstructure knowledge. The conceptual framework — order flow, adverse selection, liquidity dynamics — changes how you think about transaction costs, risk, and feature engineering, even at daily frequency.

## Why No Homework

The FI-2010 dataset — the only freely available limit order book dataset — covers exactly 5 Finnish stocks (Nokia, Outokumpu, Sampo, Rautaruukki, Wärtsilä) over 10 trading days in June 2010. That's it. The next-best option, LOBSTER (which reconstructs NASDAQ order books from ITCH data), costs approximately $500 per stock per year for academic access. Real proprietary LOB data from firms like Refinitiv or Bloomberg costs tens of thousands per year.

A homework assignment requires enough data to train a meaningful model, evaluate it out of sample, and draw conclusions. FI-2010 is sufficient for in-class exercises — we can split it 80/20 and demonstrate that deep learning on LOB data works — but it's not enough for a rigorous homework. Five stocks, one exchange, ten days, from 15 years ago — any result from that data tells you about Finnish equities in 2010, not about markets in general.

This data constraint is itself a lesson. In HFT and market microstructure, *data access is the moat.* The firms that have LOB data have spent millions building infrastructure to capture, store, and process it. The FI-2010 benchmark exists because a group of Finnish academics had a specific data-sharing agreement with the exchange. Nothing comparable exists for US equities, and it probably never will — the exchanges sell that data for too much money.

We'll use the energy you'd spend on homework to prepare for Week 16, which is homework-heavy and builds directly on the microstructure intuition from today.

## Concept Matrix

*Note: Week 15 has no homework due to LOB data constraints (see "Why No Homework" section). The Homework column shows where concepts reappear in later weeks.*

| Concept | Lecture | Seminar | Homework (later weeks) |
|---------|---------|---------|----------|
| Limit order book structure | Demo: LOB depth chart animation, spread/mid-price definitions | Exercise 1: quantitative LOB feature engineering (5 features) | W16: LOB structure assumed in A-S simulator |
| Order Flow Imbalance (OFI) | Demo: compute OFI at 10s intervals, linear regression on mid-price changes | Exercise 3: verify Cont's OFI at 1s/10s/60s intervals, multi-stock comparison | W16: OFI as state feature for RL market maker |
| DeepLOB / CNN on LOB | Demo: simplified DeepLOB-lite architecture and F1 results | Exercise 2: CNN vs. hand-crafted features comparison, error analysis | Not revisited (specialized topic) |
| VPIN | Demo: compute VPIN, show it spikes before large price moves | Not covered (demonstrated in lecture) | W18: VPIN as risk signal in backtest pipeline |
| HFT infrastructure & latency | Demo: Python timing benchmark, latency stack table | Not covered (conceptual, done in lecture) | Not revisited (awareness topic) |
| Microstructure-informed daily models | Not covered (reserved for seminar discussion) | Exercise 4: discussion mapping microstructure knowledge to daily models | W18: realistic transaction cost modeling in capstone |

## Key Stories & Facts to Weave In

- **The Flash Crash, May 6, 2010:** The most dramatic single-day event in modern market history. 9% decline in 10 minutes. Caused by a single algorithm (Waddell & Reed's $4.1 billion sell order) that ignored its own market impact. Procter & Gamble briefly traded at a penny. Led to SEC circuit breakers and the "limit-up/limit-down" mechanism. VPIN had been elevated since morning.

- **Spread Networks' $300 Million Fiber Cable (2010):** A new fiber route from NYC to Chicago that saved 3 milliseconds. Profitable for about 2 years until microwave towers (speed of light through air > speed of light through glass) made it obsolete. The cable's story is told in Michael Lewis's *Flash Boys* and illustrates the arms race: spend millions to save milliseconds, only to have someone else spend millions to save microseconds.

- **Virtu Financial's One Losing Day (2015 IPO):** In their S-1 filing, Virtu disclosed 1 losing day out of 1,238 trading days (2009-2014). Average daily revenue: ~$1 million. The consistency comes from market making: earning the spread thousands of times per day. The math is simple — if your win rate is 51% on thousands of independent trades, the law of large numbers makes your daily P&L nearly deterministic.

- **Knight Capital, August 1, 2012:** A deployment error activated old test code on 147 NYSE-listed stocks. The code bought high and sold low, repeatedly, at machine speed. Knight lost $440 million in 45 minutes — approximately $10 million per minute. The firm was acquired by Getco the following week. The bug was in the deployment process, not in the algorithm itself. Lesson: in HFT, an operational error can destroy a company before a human can intervene.

- **Aquilina, Budish & O'Neill (QJE 2022) — Quantifying the HFT Arms Race:** Using proprietary data from the London Stock Exchange, they showed that latency arbitrage imposes a "tax" of about 0.5 basis points on all trading. That may sound small, but applied to trillions of dollars in annual equity volume, it's billions of dollars transferred from regular investors to HFT firms. The paper won the 2023 Fischer Black Prize.

- **The FI-2010 Dataset and Academic Inequality:** The fact that the only freely available LOB dataset covers 5 Finnish stocks from 2010 reveals a deep problem in financial ML research: the best data is locked behind institutional paywalls. Researchers at universities with WRDS access or exchange data agreements can do work that others simply cannot replicate. FI-2010 is the community's best attempt at a level playing field, but it's a very small field.

- **Citadel Securities' Market Share:** One company handles approximately 25% of all US equity trading volume. They're not a hedge fund (that's Citadel LLC, a separate entity). They're the plumbing — the market maker that fills your Robinhood order. Revenue in 2022: $7.5 billion. On fractions of a penny per share.

- **Python's Interpreter Overhead vs. FPGA:** In the time it takes Python to import NumPy (~150 milliseconds), a Xilinx FPGA at the NYSE has processed roughly 150,000 market data messages and placed orders on half of them. This is not a gap you close with better code or a faster CPU. It's a fundamental architectural difference.

## Cross-References
- **Builds on:** Week 1 (order book basics, alternative bars — VPIN uses volume bars), Week 3 (transaction costs — now we understand where they come from), Week 7 (CNN architecture — DeepLOB is a CNN+LSTM).
- **Sets up:** Week 16 (Market Making — the Avellaneda-Stoikov model assumes an LOB structure, and OFI provides features for the RL enhancement), Week 18 (Capstone — microstructure-aware transaction cost modeling is critical for realistic backtests).
- **Recurring thread:** The "what's hidden in the data" theme from Week 1 (why OHLCV is lossy) reaches its most dramatic expression here. The 40,000:1 compression ratio is the number students should remember every time they work with daily data.

## Suggested Reading
- **Harris, "Trading and Exchanges" (2003):** The definitive textbook on market microstructure. Written by a former chief economist of the SEC. Yes, it's from 2003, and yes, markets have changed since then. But the fundamental concepts — order types, market makers, adverse selection, price discovery — are timeless. Read Chapters 1-6 for the conceptual framework.
- **Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events" (Quantitative Finance, 2014):** The OFI paper. Short (20 pages), elegant, and empirically rock-solid. If you read one paper on market microstructure, make it this one. The linear relationship between order flow and price changes is the single most important result in the field.
- **Lewis, "Flash Boys" (2014):** Michael Lewis at his best — the story of HFT told through the characters who exposed the latency arbitrage game. Not technical, but riveting, and it will make every number in this week's lecture feel real. Read it on a plane; you'll finish it before you land.
