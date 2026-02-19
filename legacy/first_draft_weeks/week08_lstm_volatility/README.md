# Week 8 — Sequence Models: LSTM/GRU for Volatility & Returns

> **Volatility is the one thing in finance that's genuinely predictable. This week, you build the model that proves it.**

## Prerequisites
- **Week 7 (Feedforward Nets):** PyTorch fluency — `nn.Module`, custom training loops, MPS acceleration, temporal train/val/test splitting. You need this because the LSTM is built on the same PyTorch foundation, just with a different internal architecture.
- **Week 2 (Time Series & Stationarity):** GARCH(1,1), volatility clustering, autocorrelation. GARCH is the baseline we're trying to beat. You built one in Week 2 — now you'll see whether a neural network can do better.
- **Week 3 (Portfolio/Risk):** Realized volatility, risk metrics. You'll need to understand what volatility IS before you can forecast it.
- **Week 5 (Trees/XGBoost):** Feature engineering for financial data — momentum, volume, volatility measures. Some of these become LSTM input features.

## The Big Idea

Last week, your feedforward network looked at each stock-month as an independent observation — a frozen snapshot of 20 features with no temporal context. But financial data is fundamentally sequential. Volatility clusters: a high-vol day is more likely to be followed by another high-vol day. Momentum persists: stocks that went up last month tend to go up this month (until they don't). Earnings announcements create patterns in volume and returns that repeat quarterly. A feedforward network can't see any of this unless you manually engineer lagged features. An LSTM sees it natively — it carries a hidden state that remembers what happened yesterday, last week, and last month.

We focus this week on volatility forecasting rather than return prediction for a specific reason: volatility is genuinely predictable. The autocorrelation of squared returns is strong, persistent, and well-documented. GARCH(1,1), a model from 1986 that uses exactly 3 parameters, captures about 70-80% of the variance in next-day volatility. This is remarkable. Try finding a 3-parameter model that explains 70% of next-day returns — it doesn't exist. Returns are close to a martingale; volatility is not. This means volatility forecasting is a problem where we can meaningfully evaluate whether a neural network adds value over a classical baseline, because the baseline is actually strong.

The practical stakes are enormous. Every options desk in the world needs a volatility forecast. If you sell an option, you're implicitly betting that realized volatility will be lower than the implied volatility embedded in the option price. If your volatility forecast is 1% better than the market's, you have an edge on every option trade. Risk management systems use volatility forecasts to set position limits, compute Value at Risk, and trigger hedging. Portfolio optimizers need a covariance matrix, which is just a matrix of volatilities and correlations. Better volatility forecasts make everything downstream better.

The showdown this week is LSTM vs. GARCH. It's a fair fight: GARCH has the advantage of simplicity, interpretability, and decades of theoretical justification. The LSTM has the advantage of flexibility — it can incorporate multiple input features (not just past returns), learn nonlinear dynamics, and adapt to regime changes. The honest result, which you'll replicate in your homework: the LSTM beats GARCH, but not by as much as you'd hope. The improvement is typically 5-15% in MSE, concentrated in volatile periods where GARCH's rigid parametric form can't keep up. In calm markets, GARCH is nearly optimal, and the LSTM's extra complexity buys you almost nothing.

## Lecture Arc

### Opening Hook

"On February 5, 2018, the VIX — Wall Street's 'fear gauge' — doubled in a single day, from 17 to 37. An ETF called XIV, which bet against volatility by shorting VIX futures, lost 96% of its value overnight. It had $1.9 billion in assets that morning. By Friday, it was liquidated. The people who bought XIV believed that low volatility would stay low. They had evidence: the VIX had been below 15 for most of 2017. But volatility clusters. The market remembers its shocks. And a model that captures the dynamics of volatility — that understands how today's calm relates to tomorrow's storm — is the difference between managing risk and being destroyed by it. Today we build that model."

### Section 1: Why Volatility (and Not Returns) is the Right Target

**Narrative arc:** We establish why we're choosing volatility as our forecasting target — it's the rare quantity in finance that's both genuinely predictable and economically important.

**Key concepts:** Realized volatility, implied volatility, the volatility clustering phenomenon, autocorrelation of squared returns.

**The hook:** "Here's a puzzle. Plot the autocorrelation of S&P 500 daily returns. It's barely distinguishable from zero at any lag — the market is roughly a random walk. Now plot the autocorrelation of squared daily returns. At lag 1, it's around 0.25. At lag 5, it's 0.15. At lag 20, it's still 0.05. Squared returns — a proxy for volatility — are predictable weeks into the future. This is one of the most robust stylized facts in all of finance, and it's been documented in every market, every asset class, every time period since Mandelbrot noticed it in 1963."

**Key formulas:**

Realized volatility (the target variable). The simplest estimator — close-to-close:

$$\text{RV}_t^{(h)} = \sqrt{\sum_{i=1}^{h} r_{t-h+i}^2}$$

where $r$ are log returns and $h$ is the horizon (e.g., $h=21$ for monthly realized vol). This is the sum of squared returns over a window — essentially, how much the price wiggled over the last $h$ days.

Better estimators use intraday information. Parkinson (1980) uses high-low range:

$$\text{RV}_t^{\text{Parkinson}} = \sqrt{\frac{1}{4 \ln 2} (\ln H_t - \ln L_t)^2}$$

Garman-Klass (1980) adds open-close:

$$\text{RV}_t^{\text{GK}} = \sqrt{0.5(\ln H_t - \ln L_t)^2 - (2\ln 2 - 1)(\ln C_t - \ln O_t)^2}$$

"The Parkinson and Garman-Klass estimators are more efficient — they use more price information per bar — but they're biased when markets are closed (overnight). For daily data from yfinance, close-to-close is the safest default. For intraday data, Garman-Klass is standard."

**Code moment:** Compute all three estimators for SPY and plot them:

```python
# Close-to-close realized vol (21-day rolling)
rv_cc = returns.rolling(21).std() * np.sqrt(252)

# Parkinson
hl = np.log(df['High'] / df['Low'])
rv_park = np.sqrt((hl**2 / (4 * np.log(2))).rolling(21).mean() * 252)

# Garman-Klass
rv_gk = np.sqrt((0.5 * hl**2 - (2*np.log(2)-1) * np.log(df['Close']/df['Open'])**2).rolling(21).mean() * 252)
```

Plot all three. They track each other closely, but Garman-Klass is less noisy — it squeezes more signal out of the same daily bars.

**"So what?":** "The target variable matters as much as the model. A better volatility estimator gives your LSTM a cleaner target to learn from, which translates directly to better forecasts. In industry, options desks typically use 5-minute squared returns for intraday realized vol. With daily data, Garman-Klass is the sweet spot between accuracy and simplicity."

### Section 2: GARCH(1,1) — The Baseline That Refuses to Die

**Narrative arc:** Before building the LSTM, we need to understand what it's competing against. GARCH(1,1) is 40 years old, has 3 parameters, and is still the default volatility model at most banks.

**Key concepts:** Conditional variance, GARCH(1,1) specification, maximum likelihood estimation, volatility persistence.

**The hook:** "Tim Bollerslev published the GARCH model in 1986. It's 2026 and people still use it. In ML terms, that's like if logistic regression from 1958 still beat neural networks on ImageNet. GARCH doesn't beat LSTMs — but it comes embarrassingly close. Three parameters versus three thousand, and GARCH gives up maybe 10% in MSE. That should make you humble about the value of complexity."

**Key formulas:**

GARCH(1,1) says the conditional variance tomorrow is a weighted average of three things:

$$\sigma_t^2 = \underbrace{\omega}_{\text{long-run floor}} + \underbrace{\alpha \cdot r_{t-1}^2}_{\text{yesterday's shock}} + \underbrace{\beta \cdot \sigma_{t-1}^2}_{\text{yesterday's variance}}$$

Read it left to right: $\omega$ is the baseline variance the process reverts to. $\alpha$ controls how much today's return surprises the model — a big move yesterday makes the model expect a big move today. $\beta$ controls persistence — yesterday's forecast carries forward. Typically $\alpha \approx 0.05$, $\beta \approx 0.93$, so $\alpha + \beta \approx 0.98$ — meaning shocks to volatility decay slowly, over many weeks.

The unconditional (long-run) variance is:

$$\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta}$$

"When $\alpha + \beta$ is close to 1, the long-run variance is large relative to $\omega$, and volatility shocks take a long time to decay. This is the 'persistence' parameter. For the S&P 500, it's typically 0.97-0.99 — meaning a volatility shock has a half-life of about 20-50 trading days."

**Code moment:**

```python
from arch import arch_model

garch = arch_model(returns * 100, vol='Garch', p=1, q=1)
result = garch.fit(disp='off')
print(result.summary())  # omega, alpha, beta, log-likelihood
```

Show the fitted parameters. Then extract the conditional variance series and compare to realized vol.

**"So what?":** "GARCH(1,1) is your baseline. Any model you build that doesn't beat GARCH is adding complexity for nothing. And here's the uncomfortable truth: on univariate daily data for a single asset, beating GARCH by more than 10-15% is genuinely hard. The LSTM's advantage comes from multivariate inputs — it can see volume, VIX, other stocks' returns — things GARCH can't use."

### Section 3: The HAR Model — A Smarter Baseline

**Narrative arc:** Before jumping to LSTMs, we introduce one more classical baseline that's deceptively powerful — and that teaches an important lesson about what timescales matter for volatility.

**Key concepts:** Heterogeneous Autoregressive (HAR) model, multi-horizon aggregation, daily/weekly/monthly components.

**The hook:** "Fulvio Corsi published the HAR model in 2009. It's a linear regression with three features: average realized vol over the last day, last week, and last month. Three features, linear model, no deep learning required. It consistently beats GARCH for multi-day volatility horizons. The reason is simple: different market participants operate at different timescales. Day traders react to today's vol. Swing traders react to this week's vol. Portfolio managers react to this month's vol. The HAR model captures all three, and GARCH doesn't."

**Key formulas:**

$$\text{RV}_t^{(h)} = \beta_0 + \beta_d \cdot \text{RV}_{t-1}^{(1)} + \beta_w \cdot \text{RV}_{t-1}^{(5)} + \beta_m \cdot \text{RV}_{t-1}^{(22)} + \varepsilon_t$$

where $\text{RV}_{t}^{(k)}$ is the average realized vol over the preceding $k$ days. "That's it. Four parameters. The coefficients tell you how much of tomorrow's volatility is explained by each timescale. Typically $\beta_d > \beta_w > \beta_m$ — short-term vol matters most — but all three are significant."

**Code moment:** Implement HAR in 10 lines of sklearn:

```python
from sklearn.linear_model import LinearRegression

# Compute RV at three timescales
rv_d = returns.rolling(1).std()  # daily
rv_w = returns.rolling(5).std()  # weekly
rv_m = returns.rolling(22).std()  # monthly

features = pd.DataFrame({'rv_d': rv_d, 'rv_w': rv_w, 'rv_m': rv_m}).dropna()
target = rv_d.shift(-1).reindex(features.index)  # predict tomorrow's vol

har = LinearRegression().fit(features, target)
```

**"So what?":** "The HAR model tells you something profound about the LSTM's job: the LSTM needs to learn these multi-timescale aggregations automatically from the raw data, plus whatever nonlinear interactions exist beyond them. If the LSTM can't beat HAR, it hasn't learned anything that a linear model with smart features can't capture."

### Section 4: RNNs, LSTMs, and GRUs — Giving the Network a Memory

**Narrative arc:** We build from vanilla RNNs (broken) to LSTMs (fixed) to GRUs (simpler fix), explaining why each step was necessary.

**Key concepts:** Vanishing gradients, gating mechanisms, forget/input/output gates (LSTM), reset/update gates (GRU), hidden state as "memory."

**The hook:** "A vanilla RNN processes a sequence by maintaining a hidden state: $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$. In theory, $h_t$ remembers everything that happened before time $t$. In practice, it forgets within 5-10 time steps. The math is straightforward: when you backpropagate through 20 time steps, you're multiplying the gradient by $W_h$ twenty times. If the largest eigenvalue of $W_h$ is less than 1, the gradient shrinks exponentially — vanishing gradients. If it's greater than 1, the gradient explodes. In financial data, where you might want to remember that volatility was elevated 20 trading days ago, a vanilla RNN is useless."

**Key formulas:**

The LSTM's solution is four gates that control information flow. At each time step $t$:

**Forget gate** — what to discard from the cell state:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input gate** — what new information to store:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate cell state** — proposed new memory:
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell state update** — the core memory:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output gate** — what to output:
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden state** — the output:
$$h_t = o_t \odot \tanh(C_t)$$

"Read the cell state update equation carefully. The forget gate $f_t$ is a value between 0 and 1 for each dimension. When $f_t = 1$, the old memory passes through perfectly — no gradient decay. When $f_t = 0$, the old memory is erased. This additive update ($C_t = f \odot C_{t-1} + ...$) is the key insight: it creates a 'gradient highway' where information can flow unchanged across many time steps. The vanilla RNN's multiplicative update ($h_t = \tanh(Wh_{t-1} + ...)$) has no such highway."

The GRU simplifies to two gates (reset and update):

$$z_t = \sigma(W_z [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

"The GRU has 25% fewer parameters than the LSTM (2 gates instead of 3, no separate cell state). In practice, they perform similarly on most financial forecasting tasks. The GRU trains faster. The LSTM is more widely used in the literature, so it's easier to compare results. Pick one and be consistent."

**Code moment:**

```python
class VolatilityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)
```

"Notice: we take only the last hidden state `lstm_out[:, -1, :]`. The LSTM processes the entire sequence, building up its memory step by step, and we only ask for a prediction at the end. This is a many-to-one architecture — the standard setup for forecasting."

**"So what?":** "The LSTM is a feedforward network with memory. Everything you learned last week — batch norm, dropout, training loops, temporal splits — still applies. The difference is that instead of seeing a single snapshot of features, the LSTM sees a *movie* of the last 20 trading days. For volatility forecasting, that movie is the signal."

### Section 5: Architecture Decisions for Financial Sequences

**Narrative arc:** We go through the practical decisions: how long should the sequence be? How many layers? Bidirectional or not? These choices matter more than they do in NLP or speech.

**Key concepts:** Sequence length, number of layers, hidden size, bidirectional LSTMs, teacher forcing.

**The hook:** "In NLP, bidirectional LSTMs are standard — you read a sentence both forwards and backwards to understand it. Try that with stock data and you've just given the model the answer. A bidirectional financial LSTM sees the future. This is the most common bug in financial LSTM code, and it shows up in published papers more often than you'd think. The fix is easy: `bidirectional=False`. Always."

**Key concepts unpacked:**

**Sequence length:** How many past trading days does the LSTM see? Too short (5 days) and it misses the monthly volatility cycle. Too long (100 days) and it struggles to learn — the signal from 100 days ago is very weak, and the long sequence makes training slower. For daily volatility forecasting, 20 trading days (approximately one month) is the sweet spot. This matches the longest timescale in the HAR model, which isn't a coincidence.

**Number of layers:** 1-2 layers is usually sufficient. Financial sequences are much shorter than NLP sequences (20 steps vs. 512 tokens), so the extra representational capacity of deep LSTMs isn't needed. More layers = more parameters = more overfitting risk.

**Hidden size:** 32-64 is the typical range. Bigger hidden sizes can memorize training data. "If your LSTM has hidden_size=256, it has more parameters than your dataset has useful data points. That's not deep learning — that's an expensive lookup table."

**Teacher forcing:** Used in sequence-to-sequence models where you predict multiple future steps. For one-step forecasting (predict tomorrow's vol), it's not applicable — just predict the single next value.

**Code moment:** Show the effect of sequence length on validation MSE. Train the same LSTM with seq_len = 5, 10, 20, 40, 60. Plot validation MSE vs. seq_len. The curve typically has a U-shape with a minimum around 20.

**"So what?":** "Every hyperparameter in a financial LSTM is a statement about how markets work. Sequence length = how far back does relevant information extend? Number of layers = how much nonlinear transformation does the signal need? Hidden size = how complex are the patterns? These aren't engineering choices — they're scientific hypotheses."

### Section 6: Loss Functions for Volatility

**Narrative arc:** MSE seems natural for regression, but volatility has a specific statistical property (heteroskedasticity) that makes MSE suboptimal. QLIKE is better — and the reason why teaches you something important about loss function design.

**Key concepts:** MSE, QLIKE (quasi-likelihood loss), MSE on log-volatility, heteroskedasticity-robust evaluation.

**The hook:** "Imagine two days. On Day 1, actual vol is 10% and your model predicts 12% — error of 2 percentage points. On Day 2, actual vol is 50% and your model predicts 52% — also 2 percentage points. MSE penalizes both equally. But the first error is a 20% relative mistake, and the second is a 4% relative mistake. In volatility forecasting, percentage errors matter more than absolute errors because volatility itself varies by 5x between calm and crisis periods. QLIKE penalizes relative errors."

**Key formulas:**

QLIKE loss:

$$\mathcal{L}_{\text{QLIKE}} = \frac{1}{T} \sum_{t=1}^{T} \left( \frac{\sigma_t^2}{\hat{\sigma}_t^2} - \ln \frac{\sigma_t^2}{\hat{\sigma}_t^2} - 1 \right)$$

where $\sigma_t^2$ is realized variance and $\hat{\sigma}_t^2$ is the forecast. QLIKE is the loss function that corresponds to the Gaussian quasi-likelihood — it's the "right" loss if you assume returns are conditionally Gaussian with time-varying variance. It penalizes under-prediction more than over-prediction (asymmetric), which is appropriate for risk management — underestimating volatility is more dangerous than overestimating it.

A simpler alternative: MSE on log-volatility:

$$\mathcal{L}_{\text{log-MSE}} = \frac{1}{T} \sum_{t=1}^{T} (\ln \hat{\sigma}_t - \ln \sigma_t)^2$$

"Log-MSE is symmetric in percentage terms. It's less theoretically justified than QLIKE but easier to implement and often performs comparably."

**Code moment:**

```python
def qlike_loss(pred_var, actual_var):
    """QLIKE loss — better for heteroskedastic targets."""
    ratio = actual_var / (pred_var + 1e-8)
    return (ratio - torch.log(ratio) - 1).mean()

def log_mse_loss(pred_vol, actual_vol):
    """MSE on log-volatility — percentage error."""
    return ((torch.log(pred_vol + 1e-8) - torch.log(actual_vol + 1e-8))**2).mean()
```

**"So what?":** "The loss function you train on determines what your model is good at. MSE-trained models are best at predicting average volatility. QLIKE-trained models are best during crises — exactly when you need them most. For risk management applications, always use QLIKE or log-MSE. For portfolio optimization where you just need a reasonable vol estimate, MSE is fine."

### Section 7: The Showdown — LSTM vs. GARCH vs. HAR

**Narrative arc:** We run the head-to-head comparison. The results are nuanced — LSTM wins overall but the margin depends heavily on the market regime and evaluation metric.

**Key concepts:** Out-of-sample evaluation, regime-conditional performance, directional accuracy, the cost of complexity.

**The hook:** "Dozens of papers have run this horse race. The typical result: LSTM beats GARCH by 5-15% in MSE, beats HAR by 3-8% in MSE, and beats both more convincingly during volatile periods. In calm markets, the differences nearly vanish. The question isn't whether the LSTM wins — it does. The question is whether the improvement justifies the complexity. A GARCH fits in 50 milliseconds. An LSTM takes 5 minutes. If you're forecasting vol for 500 stocks daily, that's 50 milliseconds × 500 = 25 seconds (GARCH) vs. 5 minutes × 500 = not happening without a GPU cluster. Production systems use GARCH for the bulk and LSTMs for the top-tier stocks where marginal improvement matters."

**Code moment:** The comparison table:

```python
comparison = pd.DataFrame({
    'Model': ['GARCH(1,1)', 'HAR', 'LSTM (1-layer)', 'LSTM (2-layer)', 'GRU (2-layer)'],
    'MSE (calm)': [0.82, 0.78, 0.77, 0.76, 0.76],
    'MSE (crisis)': [3.15, 2.90, 2.45, 2.38, 2.42],
    'QLIKE': [0.45, 0.42, 0.38, 0.36, 0.37],
    'Dir. Accuracy': ['54%', '56%', '58%', '59%', '58%'],
    'Train Time': ['50ms', '10ms', '3min', '5min', '4min'],
})
```

"These numbers are representative, not exact — your results will vary. But the pattern is consistent: LSTM wins by more during crises than during calm, and the 2-layer LSTM barely beats the 1-layer. The GRU matches or beats the LSTM despite having fewer parameters."

**"So what?":** "The right model depends on your use case. If you're running a risk system for 5,000 stocks, use GARCH — it's fast enough to run in real time. If you're running a volatility trading strategy on 20 highly liquid stocks where a 10% improvement in vol forecasting translates to $5 million in P&L, use the LSTM. Know your ROI."

### Closing Bridge

"Your LSTM now carries a memory of the last 20 trading days. But the architecture has a limitation: it processes one stock at a time, using only that stock's own history. It can't see that when Apple's volatility spikes, TSMC's usually follows two days later. It can't learn that energy stocks move in sync during oil price shocks. And it's trained from scratch on your small dataset — it hasn't seen the patterns that repeat across thousands of stocks and decades of history. Next week, we ask a radical question: what if someone already trained a model on billions of financial time series? Foundation models for financial time series are the fastest-moving area in quant ML, and the results are both more nuanced and more interesting than you've been led to believe."

## Seminar Exercises

### Exercise 1: Volatility Estimators Across Asset Classes — Does the Target Variable Choice Transfer?
**The question we're answering:** The lecture demonstrated three RV estimators on SPY. Does the best estimator for equities also work best for other asset classes — or does the "right" target depend on the market microstructure?
**Setup narrative:** "You saw the three estimators on SPY in the lecture. But equities trade on exchanges with discrete open/close auctions. Crypto trades 24/7 with no official close. FX has overlapping sessions across time zones. The Garman-Klass estimator assumes a meaningful open-close gap. Does it still win when that gap doesn't exist?"
**What they build:** Compute close-to-close, Parkinson, and Garman-Klass realized vol for SPY (equity), BTC-USD (crypto, via yfinance), and EUR/USD (FX proxy ETF or yfinance). For each asset, train the same LSTM on each target. Evaluate with QLIKE on a held-out period.
**What they'll see:** For SPY, Garman-Klass is best (confirming the lecture). For BTC-USD, the difference between estimators is smaller (continuous trading = less open/close information). For FX, Parkinson performs surprisingly well (high-low range captures intraday moves cleanly).
**The insight:** The "best" volatility target depends on market microstructure. Garman-Klass exploits the open-close gap, which is meaningful for equities but less so for 24/7 markets. Always validate your target variable choice on your specific asset class.

### Exercise 2: How Fast Do Baselines Adapt? — Measuring Lag During Regime Shifts
**The question we're answering:** The lecture showed that GARCH and HAR are strong baselines overall. But during regime transitions (calm-to-crisis, crisis-to-calm), how many days do they take to "catch up"? Is there a window of opportunity where the LSTM's faster adaptation matters?
**Setup narrative:** "You already know GARCH and HAR are competitive on average. The more useful question is: when do they fail? If there's a predictable window after each regime shift where classical models lag, that's exactly where the LSTM earns its keep."
**What they build:** Using SPY data from 2018-2024, identify the 5 largest volatility regime shifts (using a CUSUM or rolling z-score detector). For each event, compute GARCH, HAR, and a pre-trained LSTM's daily forecast errors for the 20 days following the regime change. Plot the "adaptation curve" — cumulative QLIKE error as a function of days-since-shift.
**What they'll see:** GARCH takes 5-10 days to fully adapt after a crisis onset (because $\alpha + \beta \approx 0.98$ means slow mean-reversion). HAR adapts faster (3-5 days, because the daily component responds immediately). The LSTM adapts in 1-2 days because its multivariate inputs (VIX, volume) react instantly.
**The insight:** The LSTM's advantage is concentrated in a narrow window after regime shifts. If you're sizing the LSTM's value, don't look at average metrics — look at the adaptation lag.

### Exercise 3: Sequence Length as a Scientific Hypothesis — What Memory Does the Market Have?
**The question we're answering:** The lecture built the LSTM with `seq_len=20` (one month of trading days). But what if the market's memory for volatility is shorter or longer? Can we use the LSTM's performance across different sequence lengths to measure how far back the signal extends?
**Setup narrative:** "Every hyperparameter is a hypothesis. Sequence length says 'the last N days contain useful information for tomorrow's volatility.' You're about to turn hyperparameter tuning into a scientific experiment: sweep seq_len from 5 to 60 and map the volatility memory curve."
**What they build:** Train the lecture's 2-layer LSTM (hidden=64, input = lagged returns + volume + VIX) with seq_len = 5, 10, 15, 20, 30, 40, 60. For each, record validation QLIKE. Plot the "memory curve" — QLIKE vs. seq_len. Also compute HAR-equivalent features at each timescale for comparison.
**What they'll see:** QLIKE improves rapidly from seq_len=5 to 20, then flattens. The optimal is usually around 20-25 — closely matching the HAR model's monthly component. Beyond 40, performance slightly degrades (noise from distant lags).
**The insight:** The LSTM independently discovers that ~20 trading days (one month) is the relevant memory horizon for volatility — the same timescale that Corsi's HAR model uses. The LSTM doesn't just match HAR's inductive bias; it validates it from the data.

### Exercise 4: Multivariate LSTM vs. Univariate GARCH
**The question we're answering:** Does adding extra features (VIX, volume, sector vol) to the LSTM create a meaningful advantage over GARCH, which can only use past returns?
**Setup narrative:** "Now we give the LSTM the information it was born to use: multiple input streams across time. GARCH gets past returns. The LSTM gets past returns, past volume, VIX, and 5-day rolling cross-sectional volatility. This is the real fight."
**What they build:** LSTM with input = [lagged returns (5), lagged RV (5), lagged volume (5), VIX]. Compare to univariate GARCH.
**What they'll see:** The multivariate LSTM beats GARCH by 8-15% in MSE, with the gap widening during volatile periods (COVID, 2022 rate hikes).
**The insight:** The LSTM's real advantage isn't architectural — it's informational. It can ingest signals that GARCH structurally cannot. When you add the VIX (a forward-looking measure derived from option prices), you're giving the LSTM access to the market's own volatility forecast. It learns to blend this with historical vol.

## Homework: "Volatility Forecasting Showdown"

### Mission Framing

This is a horse race, and you're the judge. You have four contestants: GARCH(1,1), the grizzled veteran with 40 years on the job. HAR, the clever academic who noticed that different timescales matter. Your LSTM, the neural network newcomer with more parameters than experience. And your GRU, the LSTM's leaner cousin who claims to do the same job with less.

Your job is not just to declare a winner — it's to understand when and why each model wins. You'll discover that the LSTM dominates during volatile regimes but barely beats GARCH in calm markets. You'll find that the HAR model, despite being a linear regression with 4 parameters, punches well above its weight on multi-day horizons. And you'll wrestle with the fundamental question of financial ML: is the neural network's marginal improvement worth its marginal complexity?

The answer depends on the application. For a risk system monitoring 5,000 stocks, the GARCH is better — it's fast enough to run every minute. For a volatility trading desk where a 10% improvement in forecasting translates to millions in P&L on 20 options chains, the LSTM pays for itself many times over. This homework makes you the expert who can articulate that tradeoff.

### Deliverables

1. **Data preparation.** For 20 liquid US stocks + SPY, download daily OHLCV data from 2010 to 2024. Compute realized volatility using the Garman-Klass estimator (21-day rolling). Compute the VIX from FRED or yfinance (^VIX). Compute features: 5 lags of RV, 5 lags of returns, 5 lags of volume (normalized by 20-day average), and VIX level. Train on 2010-2019, validate on 2020-2021, test on 2022-2024.

2. **GARCH baseline.** Fit GARCH(1,1) to each stock using the `arch` library. Also fit EGARCH to capture asymmetric effects (bad news increases vol more than good news). Generate 1-day-ahead volatility forecasts using recursive out-of-sample evaluation (refit every 252 days).

3. **HAR baseline.** Fit the HAR model (daily, weekly, monthly RV as features) using OLS. Generate 1-day-ahead forecasts.

4. **LSTM model.** Build a 2-layer LSTM (hidden_size=64, dropout=0.2) in PyTorch. Input: the 16-dimensional feature vector (5 lagged RV + 5 lagged returns + 5 lagged volume + VIX), sequence length = 20 trading days. Output: next-day realized vol. Train with QLIKE loss. Use Adam (lr=1e-3), early stopping on validation QLIKE, patience=10.

5. **GRU variant.** Replace the LSTM with a GRU of the same size. Compare training speed and final performance. Are they meaningfully different?

6. **Evaluation.** For each model and each stock, compute on the 2022-2024 test set: MSE, QLIKE, directional accuracy (did the model correctly predict whether vol would go up or down?). Aggregate across stocks. Report the results in a table.

7. **Regime analysis.** Split the test set into "calm" months (VIX < 20) and "crisis" months (VIX >= 20). Report MSE separately for each regime. The key question: which model benefits most from the LSTM's extra complexity?

8. **Stretch goal.** Add an attention mechanism to the LSTM: instead of using only the last hidden state, compute a weighted average across all time steps, where the weights are learned. Does attention help? Visualize which lags get the most attention.

9. **Deliverable:** Notebook + comparison table + regime analysis + one-paragraph conclusion.

### What They'll Discover

- GARCH(1,1) achieves surprisingly good QLIKE — typically within 5-10% of the best neural model during calm periods. In a bull market, you don't need deep learning for volatility.
- The LSTM's advantage is concentrated in crisis periods. During COVID (March 2020) and the 2022 rate hike cycle, the LSTM's multi-step MSE is 15-25% lower than GARCH. It adapts faster to regime changes because it has more input channels.
- HAR is shockingly competitive. Despite being a linear regression with 4 coefficients, it beats GARCH on 5-day and 21-day horizons and comes within 3-5% of the LSTM. If you can only deploy one model, HAR is the pragmatic choice.
- GRU and LSTM produce nearly identical results. The GRU trains about 25% faster. There's no compelling reason to prefer one over the other for this task.
- The attention mechanism (stretch goal) typically improves results by 1-3% and reveals that the model pays most attention to lags 1, 5, and 20 — approximately the same timescales as the HAR model. The LSTM independently discovers the daily/weekly/monthly structure.

### Deliverable

Final notebook: `hw08_volatility_showdown.ipynb` containing all models, the comparison table, regime analysis, and conclusions.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Realized volatility estimators (CC, Parkinson, GK) | Demo: compute all three for SPY, plot and compare | Exercise 1: test across asset classes (equity, crypto, FX) — does the best estimator transfer? | Integrate: use Garman-Klass as the target for all models |
| GARCH(1,1) baseline | Demo: fit with `arch`, extract conditional variance, explain parameters | Exercise 2: measure adaptation lag during regime shifts (5 largest vol events) | At scale: fit GARCH + EGARCH per stock, recursive OOS forecasting |
| HAR model | Demo: implement in 10 lines, explain multi-timescale intuition | Exercise 2: compare adaptation speed to GARCH after regime shifts | At scale: fit HAR per stock, report alongside GARCH and LSTM |
| LSTM/GRU architecture (gates, memory) | Demo: build `VolatilityLSTM`, explain forget/input/output gates, vanishing gradients | Not re-built (done in lecture) | At scale: train 2-layer LSTM + GRU variant, compare speed and performance |
| Sequence length and financial memory | Demo: mention 20-day sweet spot, explain matching HAR timescales | Exercise 3: sweep seq_len 5-60, plot the "memory curve," validate HAR's timescale hypothesis | Integrate: use optimal seq_len from seminar findings |
| QLIKE and log-MSE loss functions | Demo: implement both, explain heteroskedasticity argument | Not covered (done in lecture) | At scale: train LSTM with QLIKE, evaluate all models with QLIKE + MSE + directional accuracy |
| Multivariate LSTM vs. univariate GARCH | Demo: conceptual comparison table (illustrative numbers) | Exercise 4: build multivariate LSTM, test gap widening during volatile periods | At scale: full 16-feature LSTM vs. GARCH, regime-conditional evaluation |
| Regime-conditional evaluation | Demo: mention calm vs. crisis differences | Exercise 2: measure adaptation lag per regime | At scale: split test set by VIX, report MSE per regime per model |

## Key Stories & Facts to Weave In

1. **The XIV implosion (February 5, 2018).** XIV was an inverse VIX ETF — it profited when volatility was low and lost when volatility spiked. It had $1.9 billion in assets. On February 5, the VIX doubled in a single day (from 17 to 37), and XIV lost 96.3% of its value overnight — about $1.8 billion evaporated. The event was caused by a feedback loop: as vol spiked, XIV had to buy VIX futures to rebalance, which pushed vol higher, which forced more buying. Credit Suisse, the issuer, terminated the product within a week. A model that forecasted the volatility regime shift even 24 hours earlier would have saved billions.

2. **Bollerslev's GARCH (1986).** Tim Bollerslev was Robert Engle's PhD student. Engle had invented ARCH in 1982 (which won him the Nobel Prize in 2003). Bollerslev's generalization — adding the lagged variance term — is one of the most cited papers in economics (60,000+ citations). The model is now so standard that it's built into every statistical package, used by every central bank, and taught in every finance PhD program. When people say "volatility clustering," they're usually thinking about GARCH even if they don't know it.

3. **The volatility persistence puzzle.** For the S&P 500, GARCH(1,1) parameters are typically $\alpha \approx 0.05$, $\beta \approx 0.93$, so $\alpha + \beta = 0.98$. This means that after a volatility shock (like a crash), it takes about 35 trading days for the conditional variance to revert halfway to its long-run level. Academic debate: is volatility actually mean-reverting ($\alpha + \beta < 1$) or permanently persistent ($\alpha + \beta = 1$, the "integrated GARCH" case)? The practical difference is small, but the theoretical implications are profound — if volatility has a unit root, there's no "normal" to revert to.

4. **The Corsi HAR model (2009).** Fulvio Corsi's observation was deceptively simple: aggregate realized vol at three timescales (daily, weekly, monthly) and regress tomorrow's vol on all three. The model was published in the Journal of Financial Econometrics and has become the benchmark for multi-horizon vol forecasting. Its implicit theory — that different market participants operate at different frequencies — is one of the foundational ideas in financial economics (the "heterogeneous market hypothesis").

5. **Volatility at options desks.** At an options market-making desk like Citadel Securities or Jane Street, volatility forecasting directly determines P&L. When you sell an option, you're short volatility — you profit if realized vol is lower than the implied vol you sold at. A 1% improvement in your vol forecast, across thousands of option trades per day, can be worth millions of dollars per year. This is why options desks were among the earliest adopters of ML for volatility.

6. **GARCH variants nobody uses.** EGARCH, GJR-GARCH, APARCH, FIGARCH, HYGARCH — the GARCH zoo has dozens of members. In practice, GARCH(1,1) and EGARCH (which captures asymmetric effects — bad news increases vol more than good news) are the only ones you'll encounter in production. The others add complexity without meaningfully improving out-of-sample forecasts. This is a recurring pattern in finance: simple models with good features beat complex models with the same features.

7. **The "realized vol" revolution.** Before 2000, volatility was estimated from daily close-to-close returns (inherently noisy — one observation per day). Andersen and Bollerslev (1998) showed that using 5-minute returns gives a far more precise estimate of daily vol — you get ~78 observations per day instead of 1. This "realized volatility" approach transformed the field and is now the standard target for vol forecasting models. The limitation: 5-minute data is expensive and not freely available for most stocks. With daily data from yfinance, Garman-Klass is the best proxy.

## Cross-References
- **Builds on:** Week 2's GARCH(1,1) and volatility clustering (the baseline we beat). Week 7's PyTorch framework (same training infrastructure, different architecture). Week 5's feature engineering (momentum, volume features become LSTM inputs).
- **Sets up:** Week 9 (foundation models use transformer architectures that extend the sequence modeling idea from LSTMs — self-attention replaces recurrence). Week 11 (the LSTM with dropout becomes a Bayesian model when you leave dropout on at inference — MC Dropout gives uncertainty estimates on your volatility forecasts, enabling you to say "I predict 20% vol with confidence" vs. "I predict 20% vol but I'm not sure").
- **Recurring thread:** The "honest comparison" theme — every new model must prove it beats the old model, and the margin must be worth the complexity. LSTM vs. GARCH is the paradigmatic case: yes, the neural net wins, but by how much, and at what cost?

## Suggested Reading
- **Corsi (2009), "A Simple Approximate Long-Memory Model of Realized Volatility"** — the HAR paper. Short, elegant, and foundational. Shows how much you can do with a linear model and the right features.
- **Zhang & Zohren (2024), "Deep Learning in Quantitative Trading," Chapter 5** — the best modern treatment of LSTMs for financial time series. Includes practical architecture guidance and honest benchmarks.
- **Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity"** — the original GARCH paper. Dense but historically important. Read the first 5 pages for the setup and equation (1) for the punchline. Everything else in the paper follows from that one equation.
