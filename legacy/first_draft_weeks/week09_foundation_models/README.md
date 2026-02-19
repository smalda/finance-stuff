# Week 9 — Foundation Models for Financial Time Series

> **Everyone told you foundation models don't work for finance. They were half right — and the half they got wrong is worth billions.**

## Prerequisites
- **Week 7 (Feedforward Nets):** PyTorch fluency, training loops, temporal CV. Foundation models are PyTorch models — you'll load them, run inference, and extract embeddings using the same framework.
- **Week 8 (LSTM/GRU):** Sequence modeling fundamentals. Transformers replace recurrence with self-attention, but the goal is the same: process temporal sequences and produce forecasts. You need LSTM as the baseline to beat.
- **Week 5 (XGBoost):** Your tree-based pipeline is the honest benchmark. The most important result this week is whether foundation model embeddings, fed into XGBoost, beat XGBoost alone.
- **Conceptual:** Understanding of self-attention (the mechanism behind transformers). If you've used BERT, GPT, or any transformer — even at a high level — you have enough. We'll explain the specifics for time series.

## The Big Idea

Here's the conventional wisdom you've probably heard: "Foundation models don't work for financial time series." It's the kind of thing senior quants say at conferences, backed by a handful of papers showing that Chronos and TimesFM underperform a simple ARIMA on stock returns. And they're right — for a specific, narrow definition of "foundation models" and "work." A generic time series foundation model, pre-trained on weather data and electricity demand and road traffic, does not magically predict stock returns zero-shot. Why would it? Stock returns have a signal-to-noise ratio of 1:100. The patterns in financial data — fat tails, volatility clustering, non-stationarity, regime changes — are fundamentally different from the smooth, quasi-periodic patterns in temperature or energy consumption.

But here's what the skeptics missed: the story has three layers, and they only looked at the first one. Layer 1: generic TSFMs (Chronos, TimesFM, Moirai) applied zero-shot to financial data — yes, these mostly fail. Layer 2: finance-native foundation models (Kronos, FinCast) pre-trained specifically on financial data — these work, sometimes dramatically. Kronos, published at AAAI 2026, was trained on 12 billion K-line records from 45 exchanges and achieved a 93% improvement in RankIC compared to generic TSFMs. Layer 3: the hybrid approach — use any foundation model (generic or finance-native) as a feature extractor, feed its embeddings into XGBoost — this often beats both the foundation model alone and XGBoost alone. It's the same insight that drove the ImageNet revolution: pre-trained representations transfer.

This week is not about declaring a winner. It's about understanding the landscape well enough to make an informed decision when someone at your firm says "should we use a foundation model?" The answer is: it depends on which model, how you use it, and what you're comparing against. That nuanced understanding is more valuable than any single prediction.

The practical stakes are enormous. Two Sigma, one of the largest quant funds ($60B+ AUM), publicly announced in 2024 that they're shifting from traditional ML to foundation models for signal extraction. JPMorgan's AI research group published LETS-C at ACL 2025, exploring text embeddings for time series classification. The industry is placing massive bets on this technology. You need to understand whether those bets are justified.

## Lecture Arc

### Opening Hook

"In December 2024, a team at Preferred Networks published a quietly devastating experiment. They took TimesFM — Google's 200-million-parameter time series foundation model, pre-trained on Google Trends data and Wikipedia pageviews — and fine-tuned it on S&P 500 data. The fine-tuned model achieved a Sharpe ratio of 1.68 on an out-of-sample trading strategy. For comparison, the average hedge fund's Sharpe ratio is about 0.5. A Sharpe of 1.68 would put you in the top 1% of all funds globally. The catch? The zero-shot version — the same model without fine-tuning — had a Sharpe of basically zero. It couldn't predict stocks at all. Same architecture, same 200 million parameters — the difference was 100% in the training data. This single result tells you everything you need to know about foundation models in finance: the model is not the bottleneck. The data is."

### Section 1: Transformer Architectures for Time Series — The Basics

**Narrative arc:** Before we can discuss foundation models, we need to understand the architecture underneath. Self-attention replaces recurrence, and this changes what the model can and can't do with financial data.

**Key concepts:** Self-attention mechanism, positional encoding, multi-head attention, the Temporal Fusion Transformer (TFT).

**The hook:** "Your LSTM from last week processes a sequence left-to-right, one step at a time. To connect step 1 to step 20, information must survive 19 state updates. Self-attention skips all of that — step 20 can directly attend to step 1. No information decay, no vanishing gradients, no 19 multiplications to erode the signal. For financial data where a VIX spike 15 days ago might matter more than yesterday's returns, this is a structural advantage."

**Key formulas:**

Self-attention computes three matrices from the input sequence:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $X \in \mathbb{R}^{T \times d}$ is the input sequence (T time steps, d features). The attention weights are:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

"Read the matrix $QK^T$: element $(i,j)$ measures how much time step $i$ should 'pay attention to' time step $j$. The softmax normalizes these scores into a probability distribution. The $\sqrt{d_k}$ scaling prevents the dot products from growing too large (which would push softmax into saturation). The result: each time step's output is a weighted average of all other time steps' values, where the weights are learned. No recurrence. No fixed lookback window. The model decides what's relevant."

Multi-head attention runs this mechanism $h$ times with different weight matrices, then concatenates:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

"Why multiple heads? Different heads can capture different types of temporal relationships. Head 1 might focus on recent lags (momentum). Head 2 might focus on weekly patterns (seasonality). Head 3 might focus on volatility regimes. The model learns the specialization."

**Positional encoding for irregular time series:**

"Standard transformers use sinusoidal positional encodings designed for equally-spaced tokens. Financial time series aren't equally spaced — markets close on weekends and holidays. A Friday-to-Monday gap is 3 calendar days but 1 trading day. The simplest fix: use the trading-day index (0, 1, 2, ...) not the calendar date. For more sophisticated handling, learn a positional embedding that takes the actual time gap as input."

**Code moment:** The core attention computation in 5 lines:

```python
# Self-attention from scratch
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
# Causal mask: prevent attending to future time steps
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))
attn_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)
```

"The causal mask is critical. Without it, the model attends to future time steps — it sees the answer before making the prediction. This is the transformer equivalent of the bidirectional LSTM bug from last week."

**"So what?":** "Self-attention solves the long-range dependency problem that plagues LSTMs, but at a cost: it's $O(T^2)$ in sequence length. For financial data with $T=20$, this is negligible. For tick-level data with $T=100,000$, it's prohibitive. Foundation models solve this with sparse attention, chunking, or tokenization."

### Section 2: The Temporal Fusion Transformer (TFT) — The Best Supervised Transformer for Finance

**Narrative arc:** Before foundation models, the TFT established that transformers could beat LSTMs on financial forecasting — but only with careful architectural design for the specific challenges of time series data.

**Key concepts:** Variable selection networks, gated residual networks, multi-horizon output, quantile forecasting.

**The hook:** "Bryan Lim and colleagues at Google published the Temporal Fusion Transformer in 2021. It's not a foundation model — it's a supervised transformer trained from scratch on your data. But it introduced ideas that every subsequent TSFM borrowed: variable selection (which inputs matter?), interpretable attention (what time steps drive the forecast?), and quantile outputs (not just a point forecast, but prediction intervals). For multi-horizon financial forecasting — 'predict the next 1, 5, and 21 days simultaneously' — TFT remains the default choice."

**Key concepts unpacked:**

**Variable selection networks:** TFT has a learnable gate that decides, at each time step, how much weight to give each input feature. If volume is uninformative for a particular stock, the gate learns to suppress it. If VIX is critical during a crisis, the gate amplifies it. This is like built-in feature selection.

**Multi-horizon output:** Instead of predicting just the next step, TFT outputs predictions for multiple future horizons simultaneously. For volatility forecasting: predict 1-day, 5-day, and 21-day vol from a single model. The multi-horizon objective acts as regularization — it forces the model to learn the underlying dynamics, not just the shortest-horizon pattern.

**Quantile outputs:** TFT predicts the 10th, 50th, and 90th percentiles of the target distribution. This gives you prediction intervals for free — "I predict 20% vol, and I'm 80% confident it'll be between 15% and 30%." This is directly useful for risk management.

**Code moment:**

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# TFT via pytorch-forecasting
training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="realized_vol",
    group_ids=["ticker"],
    max_encoder_length=60,
    max_prediction_length=21,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["return", "volume", "rv_5", "rv_21", "vix"],
)
tft = TemporalFusionTransformer.from_dataset(training, hidden_size=32, attention_head_size=4)
```

**"So what?":** "TFT is the supervised alternative to foundation models. You train it from scratch on your data, which means it learns your specific patterns perfectly — but only from the data you have. If you have 10 years of daily data for 50 stocks, that's 50 × 2,520 = 126,000 training examples. Enough? Maybe. A foundation model pre-trained on billions of time points starts with a massive head start. The question is whether that head start translates to better forecasts on YOUR data."

### Section 3: The Foundation Model Zoo — What's Available in 2025-2026

**Narrative arc:** We survey the landscape of general-purpose TSFMs, establishing what they are, what they were trained on, and why their default performance on financial data is disappointing.

**Key concepts:** Pre-training data distribution, zero-shot inference, tokenization strategies, model sizes.

**The hook:** "As of early 2026, there are at least 8 serious time series foundation models. They range from 9 million to 1 billion parameters. They were trained on datasets ranging from 1 billion to 20 billion time points. And here's the thing nobody wants to say out loud: none of them were primarily trained on financial data. Chronos was trained on a mix of weather, energy, traffic, and economics. TimesFM was trained on Google Trends and Wikipedia. Moirai was trained on the LOTSA dataset — 27 billion observations, mostly from physical systems. They can predict tomorrow's temperature in New York. They struggle to predict tomorrow's return on SPY."

**The zoo:**

| Model | Params | Pre-Training Data | Financial Data? | License |
|-------|--------|-------------------|----------------|---------|
| TimesFM 2.5 (Google) | 200M | Google Trends, Wiki, synthetic | Minimal | Apache 2.0 |
| Chronos/Chronos-2 (Amazon) | 9M-710M | T5-based, diverse TS mix | Some economic data | Apache 2.0 |
| Moirai 2.0 (Salesforce) | 11M-300M+ | LOTSA (27B obs, mostly physical) | Very little | Apache 2.0 |
| Lag-Llama | ~10M | LLaMA-based, diverse | Minimal | MIT |
| Timer (Peking U) | 67M-280M | UTSD (1B points, 7 domains) | Limited | MIT |

"These are impressive models. But the pre-training data tells you everything. If 95% of the training data is weather, energy, and traffic — smooth, quasi-periodic, stationary — the model's learned priors are a terrible match for financial returns, which are noisy, non-stationary, and have fat tails. It's like fine-tuning a model trained on landscape photos to classify X-ray images. The low-level features (edges, textures) might transfer, but the high-level patterns (clouds vs. tumors) definitely don't."

**"So what?":** "Don't dismiss foundation models because Chronos doesn't predict stock returns zero-shot. That's like dismissing neural networks because a model trained on MNIST can't classify MRI scans. The question is not 'do pre-trained models work out of the box?' The question is 'can pre-trained representations be adapted to work?'"

### Section 4: Finance-Native Foundation Models — The Ones That Actually Work

**Narrative arc:** The plot twist. Some teams trained foundation models specifically on financial data, and the results are dramatic.

**Key concepts:** K-line tokenization, financial pre-training, domain-specific foundation models, RankIC as evaluation metric.

**The hook:** "In 2025-2026, three teams independently had the same idea: what if we trained a foundation model on financial data from the start? Kronos, published at AAAI 2026, was trained on 12 billion K-line (OHLCV) records from 45 exchanges. FinCast, published at CIKM 2025, was trained on 20 billion financial time points with 1 billion parameters. Both dramatically outperform generic TSFMs on financial tasks. Kronos achieved a 93% improvement in RankIC compared to Chronos and TimesFM. The architecture wasn't the difference — both use standard transformer blocks. The data was."

**Kronos in detail:**

Kronos introduces K-line tokenization — instead of treating price as a scalar, it tokenizes each OHLCV bar as a structured unit:

"The key innovation is representing each time step as an Open-High-Low-Close-Volume tuple rather than a single number. This is how traders see data. A bar where the price opened low, rose to a high, and closed near the high (a 'hammer' in candlestick terminology) carries different information than a bar with the same close price but a different shape. Kronos encodes this shape information natively."

| Model | Size | Pre-Training Data | RankIC on Stocks | License |
|-------|------|-------------------|-----------------|---------|
| Kronos-mini | 4.1M | 12B K-line records | 0.035 (zero-shot) | MIT |
| Kronos-base | 102M | 12B K-line records | 0.048 (zero-shot) | MIT |
| Chronos-small | 46M | Diverse TS | 0.018 (zero-shot) | Apache 2.0 |
| TimesFM 2.5 | 200M | Google Trends + | 0.015 (zero-shot) | Apache 2.0 |

"Look at those numbers. Kronos-base (102M params) achieves a zero-shot RankIC of 0.048. That's competitive with a trained LSTM. Chronos, with similar-sized architecture, gets 0.018 — essentially noise. Same class of model, same inference procedure, same compute. The only difference: what data the model saw during pre-training."

**FinCast:**

"FinCast (CIKM 2025) takes a different approach: 1 billion parameters, trained on 20 billion financial data points. It achieves 20-23% MSE reduction compared to generic TSFMs on zero-shot financial forecasting. The paper's key claim: 'financial time series exhibit distributional properties that are not learned from non-financial pre-training data.' Translation: weather data doesn't teach you about fat tails."

**Code moment:**

```python
# Loading Kronos for zero-shot inference
from kronos import KronosModel

model = KronosModel.from_pretrained("kronos-base")

# Prepare OHLCV data
ohlcv = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Zero-shot prediction
forecast = model.predict(ohlcv, horizon=5)  # predict next 5 bars
```

**"So what?":** "Finance-native foundation models represent a genuine shift. For the first time, you can download a pre-trained model and get useful financial predictions without training anything. The quality isn't as good as a well-tuned supervised model — but the cost is zero. No training data preparation, no hyperparameter tuning, no expanding-window CV. Just load and predict. For rapid prototyping, screening hundreds of assets, or generating features for downstream models, this is a game-changer."

### Section 5: The "XGBoost Wins" Paper — What It Actually Found

**Narrative arc:** We address the 2025 paper that's been cited as evidence that foundation models are useless for finance, and show that the real finding is more nuanced than the headline.

**Key concepts:** Zero-shot vs. fine-tuned evaluation, sample efficiency, the value of pre-training for small datasets.

**The hook:** "In 2025, Rahimikia, Ni, and Wang published 'Re(Visiting) Time Series Foundation Models in Finance.' The paper was widely interpreted as proof that TSFMs don't work for finance — XGBoost beat every foundation model they tested. But read the paper carefully. They tested generic TSFMs (Chronos, TimesFM, Moirai) in zero-shot mode. They didn't test finance-native models (Kronos wasn't published yet). And buried in Section 5 is a result that everyone ignored: when fine-tuned with even modest amounts of financial data, the foundation models closed the gap with XGBoost significantly. The paper didn't prove that foundation models don't work. It proved that generic pre-training doesn't transfer to finance without adaptation."

**The nuanced findings:**

1. **Zero-shot generic TSFMs vs. XGBoost:** XGBoost wins convincingly. No surprise — XGBoost is trained on the target distribution; generic TSFMs aren't.

2. **Fine-tuned generic TSFMs vs. XGBoost:** The gap narrows dramatically. Fine-tuned TTM needs 3-10x fewer years of training data to match XGBoost's performance.

3. **The sample efficiency argument:** This is the finding that matters for practitioners. If you're trading a new asset class, a new market, or a newly listed stock, you don't have 10 years of training data. A fine-tuned foundation model that matches XGBoost with 2 years of data instead of 10 is practically valuable — it lets you deploy a model 8 years earlier.

4. **The TimesFM fine-tuning result (Fu et al., Dec 2024):** Fine-tuned TimesFM achieved Sharpe 1.68 on S&P 500. Zero-shot: Sharpe ~ 0. Same model. The fine-tuning transforms useless into exceptional.

**"So what?":** "The 'XGBoost wins' narrative is technically correct and practically misleading. XGBoost wins when you have lots of data and simple features. Foundation models win when you have limited data, when you want zero-shot capabilities, or when you use them as feature extractors for downstream models. The right question isn't 'which is better?' It's 'when is each approach best?'"

### Section 6: The Hybrid Approach — Foundation Model Embeddings + XGBoost

**Narrative arc:** This is the punchline of the entire lecture. The best approach isn't choosing between foundation models and classical ML — it's combining them.

**Key concepts:** Foundation model as feature extractor, embedding extraction, hybrid pipeline, complementarity of representations.

**The hook:** "What if foundation models and XGBoost aren't competitors, but collaborators? Run your data through Kronos, extract the internal representations (embeddings) from the penultimate layer, and feed those as features into XGBoost alongside your hand-crafted features. The foundation model's embeddings capture temporal patterns that your engineered features miss. Your engineered features capture domain knowledge (momentum, value, volatility) that the foundation model may not have learned. Together, they're stronger than either alone."

**Key concepts unpacked:**

**Embedding extraction:** A foundation model's internal representation at the penultimate layer is a dense vector that encodes everything the model "understands" about the input sequence. For Kronos-base, this is a 256-dimensional vector per time step. You can use the last time step's embedding, or pool (average) across all time steps.

**The hybrid pipeline:**

```
Raw OHLCV → Kronos → embedding (256-dim) ──┐
                                            ├──→ XGBoost → prediction
Raw features (momentum, vol, etc.) ─────────┘
```

**Code moment:**

```python
# Extract Kronos embeddings
with torch.no_grad():
    embeddings = model.encode(ohlcv_sequence)  # shape: (n_stocks, 256)

# Combine with hand-crafted features
hybrid_features = np.concatenate([
    traditional_features,  # shape: (n_stocks, 20)
    embeddings.numpy()     # shape: (n_stocks, 256)
], axis=1)

# Feed into XGBoost
xgb_model = XGBRegressor(n_estimators=500, max_depth=6)
xgb_model.fit(hybrid_features_train, returns_train)
```

**The results to expect:**

| Feature Set | IC | R² |
|-------------|----|----|
| Traditional only (20 features) | 0.038 | 0.50% |
| Kronos embeddings only (256-dim) | 0.032 | 0.40% |
| Hybrid (20 + 256 = 276 features) | 0.045 | 0.62% |

"The embeddings alone are weaker than traditional features — they contain temporal pattern information but not the domain-specific knowledge you engineered. But combined, they push IC beyond what either achieves alone. This is the free lunch of the hybrid approach: the embeddings and hand-crafted features are partially uncorrelated, so combining them reduces variance."

**"So what?":** "The hybrid approach is the practical answer to the 'should we use foundation models?' question. You don't bet the farm on a zero-shot foundation model. You don't ignore the representations a model learned from billions of data points. You use both. This is analogous to how computer vision evolved: nobody uses raw ImageNet features for medical imaging. They fine-tune or use them as a warm start. Same principle, different domain."

### Section 7: When to Use What — The Decision Framework

**Narrative arc:** We synthesize everything into a practical decision tree that students can apply in their careers.

**Key concepts:** Decision criteria, compute budgets, data availability, deployment constraints.

**The hook:** "Your boss asks: 'Should we use a foundation model for our new volatility forecasting system?' Here's the answer, and it depends on exactly four things: how much data you have, how much compute you have, how many assets you're covering, and whether you need the model today or can wait three months to train one."

**Decision framework:**

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| Lots of data (10+ years), few assets (< 50) | Train LSTM/TFT from scratch | You have enough data; pre-training adds little |
| Lots of data, many assets (500+) | Hybrid: FM embeddings + XGBoost | FM captures cross-asset patterns; XGBoost captures asset-specific |
| Limited data (< 3 years) | Fine-tune Kronos/TimesFM | Pre-training compensates for data scarcity |
| Rapid prototyping | Zero-shot Kronos | Good enough for screening; deploy in minutes |
| Production system, risk-sensitive | GARCH + HAR baselines, LSTM fine-tuned | Well-understood, stable, interpretable |

**"So what?":** "Foundation models are tools, not religions. The question is never 'are foundation models good?' It's 'is this foundation model, applied this way, better than the alternative for my specific problem?' This week's homework forces you to answer that question empirically."

### Closing Bridge

"You've now seen how neural networks process structured numerical sequences — feedforward nets for cross-sections, LSTMs for temporal data, transformers and foundation models for both. But there's a massive source of market-moving information we've completely ignored: text. Earnings calls, news headlines, SEC filings, analyst reports, tweets from Elon Musk at 2 AM — all of it moves markets, and none of it shows up in your OHLCV data. Next week, we give your models the ability to read. NLP for finance has gone through its own revolution — from bag-of-words to FinBERT to LLM embeddings — and Chen, Kelly, and Xiu showed in 2023 that LLM embeddings are now the single most powerful source of text-based alpha."

## Seminar Exercises

### Exercise 1: Zero-Shot Across Market Regimes — When Do Foundation Models Fail Hardest?
**The question we're answering:** The lecture showed Kronos achieves RankIC of 0.035 on average. But averages hide the action. Does zero-shot performance collapse during specific market regimes (crises, low-vol, trending)?
**Setup narrative:** "The lecture showed you the average zero-shot RankIC. Averages are for annual reports, not for trading desks. You need to know: when does Kronos work, and when does it fall apart? You're going to slice the test period into regimes and find out."
**What they build:** Load Kronos-mini from HuggingFace. Run zero-shot 5-day forecasts on 20 stocks across 2020-2024. Classify each month as "crisis" (VIX>30), "elevated" (VIX 20-30), or "calm" (VIX<20). Compute RankIC separately for each regime.
**What they'll see:** Kronos is strongest in calm, trending markets (RankIC ~ 0.04-0.05) — the regime most represented in its pre-training data. It degrades in crisis periods (RankIC ~ 0.01-0.02) when correlations spike and idiosyncratic patterns break down. Elevated-vol periods are in between.
**The insight:** Pre-training data defines the model's "comfort zone." Kronos was trained on 12B records dominated by normal market conditions. Crises are rare in training data, so zero-shot predictions during crises are unreliable. This motivates fine-tuning and the hybrid approach.

### Exercise 2: Chronos vs. Kronos — Generic vs. Finance-Native
**The question we're answering:** How much does financial pre-training data matter?
**Setup narrative:** "Same architecture class. Same inference procedure. One was trained on weather and traffic. The other on 12 billion K-line records. We're about to see how much the training data matters."
**What they build:** Load Chronos-small. Run the same 5 stocks through Chronos and Kronos. Compare RankIC.
**What they'll see:** Kronos RankIC ~ 0.035. Chronos RankIC ~ 0.015 (barely above random). The gap is dramatic and consistent.
**The insight:** Pre-training data is more important than model architecture. A 4M-parameter model trained on financial data beats a 46M-parameter model trained on generic data.

### Exercise 3: The Temporal Fusion Transformer — Supervised Baseline
**The question we're answering:** Can a transformer trained from scratch on your data beat both foundation models?
**Setup narrative:** "Foundation models bring knowledge from pre-training. A supervised TFT brings no prior knowledge but learns exactly from your data. Which advantage wins?"
**What they build:** Train a TFT using `pytorch-forecasting` on the same 5 stocks' OHLCV + VIX + volume features. Multi-horizon output: 1-day and 5-day returns. Train on 2010-2019, test on 2020-2024.
**What they'll see:** The TFT typically matches or beats Kronos's zero-shot performance, but it took 10 years of training data to get there. With only 3 years of training data, Kronos wins.
**The insight:** The supervised vs. pre-trained tradeoff is about data efficiency. If you have decades of data, train from scratch. If you have years, leverage pre-training.

### Exercise 4: Embedding Dimensionality — How Much Information Does XGBoost Actually Use?
**The question we're answering:** The lecture showed 256-dim Kronos embeddings fed into XGBoost. But does XGBoost actually use all 256 dimensions, or does it ignore most of them? What's the optimal PCA reduction for the hybrid pipeline?
**Setup narrative:** "The lecture demonstrated the hybrid approach — embeddings + traditional features into XGBoost. But 256 embedding dimensions on top of 20 traditional features might overwhelm the tree model with noise. You're going to find the sweet spot: how many PCA components of the embeddings actually help?"
**What they build:** Extract Kronos embeddings. Apply PCA with k = 5, 10, 20, 50, 100, 256 (full). For each, concatenate with traditional features and train XGBoost with expanding-window CV. Plot IC vs. number of PCA components. Also use SHAP to measure what fraction of total feature importance comes from embedding PCA components vs. traditional features.
**What they'll see:** IC peaks around k=10-20, then flattens or slightly declines. SHAP shows the top 5 PCA components capture most of the embedding's contribution. Beyond 50 components, the added dimensions are essentially noise that XGBoost learns to ignore.
**The insight:** The embedding's useful information is concentrated in a low-dimensional subspace. PCA at k=20 preserves 80%+ of the return-relevant signal while reducing noise. This parallels Chen-Kelly-Xiu's finding in Week 10 and gives practical guidance for production: always PCA your embeddings before feeding them to tree models.

## Homework: "Foundation Models for Financial Forecasting: The Three-Layer Experiment"

### Mission Framing

The financial ML community is in the middle of a heated debate: do foundation models work for finance? Twitter threads, conference panels, and Slack channels are full of strong opinions and weak evidence. Your job this week is to generate strong evidence.

You're going to run the three-layer experiment. Layer 1: take generic foundation models (Chronos, TimesFM) and see how badly they fail on stocks — confirming the skeptics' claim. Layer 2: take a finance-native foundation model (Kronos) and see how much better it does — challenging the skeptics. Layer 3: use foundation model embeddings as features for XGBoost — showing the pragmatic path forward. You'll also train a supervised TFT and compare against your LSTM from Week 8 and XGBoost from Week 5.

By the end, you won't have an opinion about foundation models. You'll have data. And data, in this business, is worth more than opinions.

### Deliverables

1. **Data preparation.** Daily OHLCV for 50 US stocks (diversified across sectors — at least 5 from tech, 5 from financials, 5 from healthcare, 5 from energy, etc.), 2010-2024. Compute the traditional feature set from Week 5 (momentum, volatility, volume ratios). Download VIX from yfinance. Train: 2010-2019. Validation: 2020-2021. Test: 2022-2024.

2. **Layer 1 — Generic zero-shot.** Run Chronos-small on the 50 stocks' close price series. Generate 5-day-ahead forecasts. Compute RankIC vs. realized 5-day returns on the test set. If you have the compute, also run TimesFM 2.5. Report results per sector — do generic TSFMs work better for some sectors?

3. **Layer 2 — Finance-native zero-shot.** Run Kronos-base (102M params, MIT license) on the same 50 stocks' OHLCV data. Generate 5-day forecasts. Compute RankIC. Compare to Layer 1 results in a table.

4. **Layer 3 — Hybrid approach.** Extract Kronos embeddings (penultimate layer representations) for each stock-date. Concatenate with your Week 5 feature set. Train XGBoost on the hybrid features using expanding-window CV. Compare IC: (a) Week 5 features only, (b) Kronos embeddings only, (c) hybrid.

5. **Supervised baselines.** Train a TFT using `pytorch-forecasting`. Train your LSTM from Week 8 (if applicable to this universe). Train XGBoost on traditional features only (your Week 5 pipeline). These are the baselines the foundation models must beat.

6. **Full comparison table.** Present results for all approaches in a single table:

   | Model | IC | RankIC | R² | Sharpe (L/S) | Training Time |
   |-------|----|----|----|----|-----|
   | Chronos (zero-shot) | | | | | 0 (inference only) |
   | TimesFM (zero-shot) | | | | | 0 |
   | Kronos (zero-shot) | | | | | 0 |
   | Kronos + XGBoost (hybrid) | | | | | ~10 min |
   | TFT (trained from scratch) | | | | | ~30 min |
   | LSTM (from Week 8) | | | | | ~5 min |
   | XGBoost (from Week 5) | | | | | ~1 min |

7. **Analysis.** Answer three questions in writing: (a) When does the hybrid approach beat XGBoost alone, and why? (b) For which sectors/stocks do foundation models add the most value? (c) If you were deploying one system in production, which would you choose and why?

8. **Deliverable:** Notebook + comparison table + written analysis.

### What They'll Discover

- Generic TSFMs (Chronos, TimesFM) produce near-zero RankIC on stock returns. The skeptics are right about this specific claim.
- Kronos achieves RankIC of 0.03-0.05 zero-shot — competitive with a simple trained LSTM. The skeptics were testing the wrong models.
- The hybrid approach (Kronos embeddings + XGBoost) typically beats XGBoost alone by 10-20% in IC. The foundation model adds information that hand-crafted features miss.
- The TFT, trained from scratch, matches the hybrid approach when given 10 years of training data. With only 3-5 years, the hybrid wins — pre-training compensates for data scarcity.
- Foundation models tend to add the most value for stocks with less analyst coverage (mid-caps, non-tech) where traditional features may be less predictive.

### Deliverable

Final notebook: `hw09_foundation_models.ipynb` containing the full three-layer experiment, comparison table, and written analysis.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Self-attention and transformer architecture | Demo: attention computation in 5 lines, causal mask, multi-head attention | Not covered (done in lecture) | Prerequisite: used conceptually to understand TFT and FM internals |
| Temporal Fusion Transformer (TFT) | Demo: show `pytorch_forecasting` code, explain variable selection and quantile outputs | Exercise 3: train TFT from scratch on 5 stocks, compare to Kronos zero-shot | At scale: train TFT on 50 stocks as supervised baseline |
| Generic TSFMs (Chronos, TimesFM, Moirai) | Demo: survey table with params, pre-training data, financial performance | Exercise 2: run Chronos vs. Kronos head-to-head on same stocks | At scale: run Chronos zero-shot on 50 stocks, report per-sector results |
| Finance-native FMs (Kronos, FinCast) | Demo: Kronos K-line tokenization, RankIC comparison table, code for loading | Exercise 1: regime-slice Kronos zero-shot (crisis/elevated/calm) | At scale: run Kronos-base zero-shot on 50 stocks, full evaluation |
| The "XGBoost wins" nuance (zero-shot vs. fine-tuned) | Demo: dissect the Re(Visiting) paper, explain sample efficiency finding | Not covered (lecture provides analysis) | Integrate: answer "when does hybrid beat XGBoost alone?" in written analysis |
| Hybrid pipeline (FM embeddings + XGBoost) | Demo: embedding extraction code, hybrid pipeline diagram, expected IC table | Exercise 4: sweep PCA dimensionality (5-256), SHAP analysis of embedding importance | At scale: full hybrid pipeline on 50 stocks with expanding-window CV |
| Decision framework (when to use what) | Demo: present scenario/recommendation table | Not covered (lecture provides framework) | Integrate: recommend deployment approach in written analysis |

## Key Stories & Facts to Weave In

1. **The TimesFM fine-tuning bombshell (Fu, Hirano, Imajo, December 2024).** A team at Preferred Networks took Google's TimesFM — a model that couldn't predict stocks zero-shot — and fine-tuned it on S&P 500 data. The result: Sharpe 1.68 on an out-of-sample trading strategy. For context, the median hedge fund Sharpe is about 0.5, and a Sharpe of 1.0 is considered excellent. Zero-shot: useless. Fine-tuned: top 1%. Same 200 million parameters. The difference was entirely in the data.

2. **Kronos and K-line tokenization (AAAI 2026).** Kronos was trained on 12 billion K-line records from 45 exchanges worldwide. Its key innovation: instead of treating price as a scalar time series, it tokenizes each OHLCV bar as a structured unit — capturing the "shape" of each bar (bullish engulfing, doji, hammer, etc.) that technical traders have used for centuries. The model achieved 93% RankIC improvement over generic TSFMs. The lead author noted that K-line tokenization alone — without any other architectural change — accounted for about 40% of the improvement.

3. **Two Sigma's public pivot (2024).** Two Sigma, one of the world's largest systematic hedge funds ($60B+ AUM), publicly stated they were moving from traditional ML to foundation models for signal extraction. Their CTO described an internal "LLM workbench" for parsing Fed speeches and research papers. Citadel's CTO, by contrast, publicly stated that foundation models are "not a source of enduring alpha." Two of the most successful quant funds in history, with diametrically opposed views. The truth is probably that both are right — foundation models help with some problems (text processing, rapid prototyping, data-scarce regimes) and not others (latency-sensitive trading, high-frequency strategies).

4. **FinCast and the 1B parameter barrier (CIKM 2025).** FinCast is the largest finance-native foundation model at 1 billion parameters, trained on 20 billion financial time points. It achieved 20-23% MSE reduction zero-shot on financial data compared to generic TSFMs. The paper made a strong claim: "financial time series require foundation models pre-trained on financial data" — essentially arguing that cross-domain transfer from weather/traffic to finance doesn't work. The counter-argument: maybe 1B parameters is overkill and a smaller model fine-tuned on less data works just as well. The jury is still out.

5. **JPMorgan's LETS-C (ACL 2025).** JPMorgan's AI Research group published a paper showing that text embeddings (from LLMs) can be used for time series classification — a bridge between the NLP and time series worlds. The implication: the boundary between "time series foundation models" and "language foundation models" is dissolving. A future model might process both price data and news text in a unified architecture.

6. **The Re(Visiting) paper's buried finding (Rahimikia et al., 2025).** This paper's headline was "generic TSFMs don't beat XGBoost on financial data." The headline is accurate. But Table 5 shows that fine-tuned TTM (a small TSFM) matches XGBoost with 3x less training data. For a quant who wants to trade a newly listed stock with only 2 years of history, this is transformative. The paper's conclusion should have been "generic TSFMs don't beat XGBoost with equal data, but they require far less data to achieve competitive performance."

7. **The computational reality.** Kronos-mini (4.1M params) runs inference in real-time on a MacBook. Kronos-base (102M params) takes ~1 second per stock on MPS. FinCast (1B params) requires a GPU. Chronos-small (46M) runs on CPU in seconds. The practical barrier to foundation models in finance is not compute — it's knowing which model to use and how to use it.

## Cross-References
- **Builds on:** Week 7's PyTorch framework (loading models, running inference). Week 8's sequence modeling concepts (transformers extend LSTMs). Week 5's XGBoost pipeline (the hybrid approach feeds FM embeddings into your existing XGBoost). Week 2's time series concepts (stationarity, autocorrelation — foundation models learn these patterns implicitly).
- **Sets up:** Week 10 (NLP foundation models — the same pre-train/fine-tune/embed paradigm applies to text). Week 11 (uncertainty quantification for foundation model predictions — are they calibrated?). The capstone (Week 18) may use foundation model features.
- **Recurring thread:** The "honest benchmark" theme — every claim about foundation models must be tested against XGBoost, LSTM, and GARCH baselines on the same data with the same evaluation protocol. No free passes for novelty.

## Suggested Reading
- **"Kronos: A Foundation Model for Financial Markets" (AAAI 2026)** — the finance-native FM that works. Read Sections 3 (K-line tokenization) and 5 (experimental results). The ablation study showing the contribution of tokenization vs. data is particularly illuminating.
- **Rahimikia, Ni, Wang (2025), "Re(Visiting) Time Series Foundation Models in Finance"** — the paper that sparked the "TSFMs don't work" debate. Read with a critical eye — the zero-shot results are accurate; the conclusions are narrower than they appear. Table 5 (sample efficiency) is the most important result.
- **Fu, Hirano, Imajo (December 2024), "Financial Fine-tuning a Large Time Series Model"** — the paper showing Sharpe 1.68 from fine-tuned TimesFM. Short, actionable, and the most compelling evidence that fine-tuning transforms generic TSFMs into effective financial models.
