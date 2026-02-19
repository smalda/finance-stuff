# Week 2 — Time Series Properties & Stationarity

> **Your model's biggest assumption is that the future looks like the past. This week, we learn when that's true, when it's a lie, and what to do about it.**

## Prerequisites
Week 1 — specifically: the concepts of non-stationarity and fat tails (Section 4), the difference between simple and log returns (Section 5), and the `DataLoader` class. Students should be comfortable computing returns and understand why raw prices can't be fed directly into ML models.

## The Big Idea

Last week, we discovered that raw stock prices are non-stationary — the distribution shifts over time, so a model trained on 2015 data is extrapolating on 2024 data. The obvious fix is returns: take the first difference (or, equivalently, compute percentage changes), and you get a roughly stationary series. Problem solved, right?

Not quite. Here's the dilemma that will define this entire week: when you take returns (first-differencing, d=1), you get stationarity, but you throw away all the memory in the price series. The autocorrelation drops to near zero. Your model can't learn that Apple has been trending upward for six months, because returns are memoryless — each one is just a percentage change from yesterday. On the other hand, if you keep raw prices (d=0), you preserve all the memory, but the series is non-stationary and your model is extrapolating. This is the integer differentiation problem, and it's been hiding in plain sight since the invention of the ARIMA model in the 1970s.

Marcos Lopez de Prado, in Chapter 5 of *Advances in Financial Machine Learning*, proposed an elegant solution: fractional differentiation. Instead of taking the 0th derivative (prices) or the 1st derivative (returns), you take the 0.4th derivative. Or the 0.6th. You find the minimum amount of differencing needed to make the series stationary, and you stop there — preserving as much memory as possible while eliminating non-stationarity. It sounds like something a mathematician made up to win an argument. But it works, and we'll prove it with code.

But fractional differentiation is only part of this week's story. We also need to understand the classical time series toolkit — AR, MA, ARIMA, and especially GARCH. Not because you'll use ARIMA as your final model (you won't), but because these models are the baselines your neural networks are competing against. And in the case of GARCH, the baseline is genuinely hard to beat. GARCH(1,1) — a model from 1986 with exactly three parameters — remains competitive with LSTMs for volatility forecasting. That fact should humble you. It certainly humbles me. Understanding why it works so well will make you a better model builder, regardless of what architecture you ultimately use.

## Lecture Arc

### Opening Hook

"In 2018, a quant at a mid-tier hedge fund showed me his LSTM for predicting S&P 500 returns. Beautiful architecture — three layers, attention mechanism, trained on five years of minute-by-minute data. Out-of-sample R-squared: 0.001. Essentially random. He asked me what went wrong. I asked him one question: 'Did you check if your input series was stationary?' He hadn't. He'd fed raw prices into the LSTM and asked it to learn that $250 and $400 are the same stock at different times. The model was spending all its capacity learning the trend — the easy, meaningless part — and had nothing left for the signal. He re-ran it on returns. The R-squared went up to 0.03. Not impressive by ML standards. But in a universe of 500 stocks, rebalanced monthly, an R-squared of 0.03 is a career. The difference between 0.001 and 0.03 was a single line of code: `.pct_change()`. This week is about understanding why that line matters so much — and whether we can do even better."

### Section 1: Stationarity — What It Is and Why Your Model Needs It
**Narrative arc:** We formalize the concept of stationarity that was introduced informally in Week 1. The setup: ML models assume that the training distribution equals the test distribution. The tension: financial time series violate this constantly. The resolution: formal tests (ADF, KPSS) that tell you exactly how bad the problem is.

**Key concepts:**
- Weak (covariance) stationarity: constant mean, constant variance, autocovariance depends only on lag
- Strong stationarity: the full joint distribution is time-invariant (almost never tested in practice)
- The Augmented Dickey-Fuller (ADF) test: null hypothesis = unit root (non-stationary)
- The KPSS test: null hypothesis = stationary (the complement of ADF)
- Why you should use both: ADF and KPSS can disagree, and the disagreement is informative

**The hook:** "Here's a question that seems too simple to be interesting: 'Is this time series stationary?' In ML, you'd barely think about this. Your training set and test set come from the same distribution — that's the whole point of the train/test split. In finance, that assumption is violated every day. Apple's mean return in 2020 was about 0.35% per day (it doubled during COVID). In 2022, it was about -0.10% per day (the tech crash). Same stock, same model, fundamentally different data-generating process. If you trained on 2020 data, your model learned that Apple goes up. If you tested on 2022 data, your model was confidently wrong."

**Key formulas:**
"The ADF test is checking whether the coefficient $\phi$ in this regression is zero:

$$\Delta y_t = \alpha + \phi \cdot y_{t-1} + \sum_{i=1}^{p} \beta_i \Delta y_{t-i} + \epsilon_t$$

If $\phi = 0$, then $y_t$ is a random walk (non-stationary) — past levels don't predict future changes. If $\phi < 0$, then the series is mean-reverting (stationary) — big deviations from the mean get pulled back. The test statistic has a non-standard distribution (Dickey-Fuller distribution, not Gaussian), which is why you can't just use a t-test.

In practice, you'll use `statsmodels`:

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(series, autolag='AIC')
# result[0] = test statistic, result[1] = p-value
# p < 0.05 → reject null → series IS stationary
```

The KPSS test flips the null hypothesis: it assumes stationarity and tests for a unit root. When ADF says 'stationary' and KPSS says 'non-stationary,' you have a trend-stationary process. When both agree, you can be more confident."

**Code moment:** Run ADF and KPSS on three series: (a) raw SPY prices, (b) SPY log returns, (c) SPY log prices. Show that raw prices fail ADF (p >> 0.05, non-stationary), returns pass easily (p << 0.01, stationary), and the intermediate case is illuminating.

**"So what?":** "Every supervised ML model implicitly assumes that the patterns it learned in training still hold in testing. If your input series is non-stationary, that assumption is violated. The model isn't wrong — it's answering a different question than you think it is."

### Section 2: Autocorrelation, Partial Autocorrelation, and Market Efficiency
**Narrative arc:** We introduce autocorrelation as the tool for measuring how much memory a time series has. The tension: returns have almost zero autocorrelation (markets are 'efficient'), but squared returns have strong autocorrelation (volatility is predictable). The resolution: this asymmetry is the reason GARCH works and why volatility forecasting is easier than return forecasting.

**Key concepts:**
- Autocorrelation function (ACF): correlation between $y_t$ and $y_{t-k}$
- Partial autocorrelation function (PACF): direct correlation at lag k, removing intermediate lags
- The efficient market hypothesis (EMH) in one graph: returns ACF ≈ 0 at all lags
- Volatility clustering in one graph: squared returns ACF decays slowly
- Ljung-Box test: formal test for whether any autocorrelation exists

**The hook:** "The efficient market hypothesis, in its weak form, says you can't predict returns from past returns. Here's the weird thing: it's approximately true. Plot the autocorrelation function of daily S&P 500 returns and you'll see... nothing. The correlations at every lag are statistically indistinguishable from zero. The market has already priced in whatever information was in yesterday's return. But now plot the autocorrelation of squared returns — which measure volatility, not direction. Suddenly, you see strong positive autocorrelation out to 60+ trading days. High-volatility days follow high-volatility days. Low-volatility days follow low-volatility days. The market remembers its shocks. Returns are unpredictable, but the size of returns is highly predictable. That asymmetry is the foundation of the entire volatility forecasting industry."

**Key formulas:**
"The autocorrelation function at lag k:

$$\rho_k = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)} = \frac{\gamma_k}{\gamma_0}$$

For returns, $\rho_k \approx 0$ for all $k > 0$. For squared returns (a proxy for volatility), $\rho_k > 0$ and decays slowly, often remaining significant for $k > 50$ trading days. This is volatility clustering — the phenomenon that large moves tend to follow large moves, regardless of direction."

**Code moment:** Two ACF plots side by side. Left: ACF of daily SPY returns. The bars are all within the 95% confidence bands — no significant autocorrelation. Right: ACF of squared SPY returns. Strong positive autocorrelation persisting for months. The visual contrast is dramatic and makes the point instantly.

**"So what?":** "This is why we'll spend Week 8 forecasting volatility, not returns. Volatility is genuinely predictable. Returns are barely predictable. The entire options market is built on volatility forecasts. And the baseline for volatility forecasting — GARCH(1,1) — is just a model that formalizes what the ACF of squared returns is showing you."

### Section 3: Classical Time Series Models — AR, MA, ARIMA
**Narrative arc:** We cover the classical toolkit rapidly but respectfully. The tension: these models seem old-fashioned to ML engineers, but they encode important ideas about how time series work. The resolution: you won't use ARIMA as your final model, but understanding it tells you what your neural net is competing against — and sometimes losing to.

**Key concepts:**
- AR(p): autoregressive model — current value depends on p past values
- MA(q): moving average model — current value depends on q past shocks
- ARMA(p,q): combination of both
- ARIMA(p,d,q): ARMA with differencing (d = number of times you difference)
- Box-Jenkins methodology: identify (ACF/PACF), estimate, diagnose
- AIC/BIC for model selection

**The hook:** "ARIMA was published by Box and Jenkins in 1970. It's older than the personal computer. And yet, in the M4 forecasting competition (2018), simple statistical models like ARIMA and exponential smoothing were competitive with neural networks across 100,000 time series. Not because ARIMA is brilliant, but because neural networks are bad at learning from short, noisy, non-stationary sequences — which is exactly what financial time series are. Know your enemy."

**Key formulas:**
"An AR(1) model is just a linear regression of the series on its own past:

$$y_t = c + \phi_1 y_{t-1} + \epsilon_t$$

If $|\phi_1| < 1$, the series is stationary and mean-reverts. If $\phi_1 = 1$, you have a random walk. If $|\phi_1| > 1$, the series explodes.

An ARIMA(p,d,q) model applies d differences first, then fits an ARMA(p,q):

$$\Delta^d y_t = c + \sum_{i=1}^{p} \phi_i \Delta^d y_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t$$

The 'd' in ARIMA is always an integer: 0 or 1 (sometimes 2). This is the integer differentiation problem: d=0 leaves non-stationarity, d=1 throws away all memory. What if d could be 0.4?"

**Code moment:** Fit ARIMA to SPY returns using `statsmodels.tsa.arima.model.ARIMA`. Show the residual diagnostics — the residuals should be white noise if the model is correct. Use `auto_arima` from `pmdarima` (or manual AIC search) to find the best (p,d,q). The result will likely be something boring like ARIMA(1,0,1) or ARIMA(0,0,0) — confirming that daily returns are nearly unpredictable.

**"So what?":** "If ARIMA(0,0,0) — literally 'the best prediction is the mean' — is the best model for daily returns, then your LSTM needs to find patterns that this trivial model can't. That's a high bar. It also tells you that the signal-to-noise ratio in daily returns is extremely low. An R-squared of 0.03 isn't a failure — it's excellent. Calibrate your expectations accordingly."

### Section 4: GARCH(1,1) — The Model That Won't Die
**Narrative arc:** This is the most important classical model in the course. The setup: returns themselves are nearly unpredictable, but their variance is highly predictable. The tension: how can a model from 1986 still be competitive? The resolution: GARCH(1,1) captures volatility clustering with exactly three parameters, and the parsimony is a feature, not a bug.

**Key concepts:**
- The ARCH effect: heteroskedasticity — variance changes over time
- GARCH(1,1): the workhorse model for conditional volatility
- The three parameters: omega (baseline variance), alpha (reaction to shocks), beta (persistence)
- Why GARCH(1,1) and not GARCH(2,1) or EGARCH: parsimony wins in practice
- Volatility persistence: alpha + beta < 1 ensures mean-reversion in variance

**The hook:** "On February 5, 2018, the VIX — Wall Street's 'fear gauge' — doubled in a single day. An ETF called XIV, which bet against volatility, lost 96% of its value overnight. It had $1.9 billion in assets that morning. It was liquidated within the week. The people who bought it thought volatility was low and would stay low. GARCH(1,1) — a model you can fit in three lines of Python — would have told them that volatility persistence (alpha + beta) was around 0.98. That means today's volatility explains 98% of tomorrow's. Low vol tends to stay low. But when it spikes, the spike persists. The XIV investors learned this lesson at a cost of $1.9 billion."

**Key formulas:**
"Start with the question: what should tomorrow's variance be? If you just use the historical average, you're saying March 2020 and March 2019 have the same expected volatility. That's absurd.

GARCH(1,1) says: tomorrow's variance is a weighted average of three things:

$$\sigma_t^2 = \underbrace{\omega}_{\text{long-run baseline}} + \underbrace{\alpha \cdot r_{t-1}^2}_{\text{yesterday's shock}} + \underbrace{\beta \cdot \sigma_{t-1}^2}_{\text{yesterday's variance}}$$

Read that left to right:
- $\omega$ pulls the variance back to the long-run average ($\omega / (1 - \alpha - \beta)$)
- $\alpha$ reacts to yesterday's surprise: big return → big increase in variance
- $\beta$ creates persistence: yesterday's variance carries over

For the S&P 500, typical values are $\alpha \approx 0.05$, $\beta \approx 0.93$, so $\alpha + \beta \approx 0.98$. That means 98% of today's variance carries over to tomorrow. Volatility is incredibly sticky."

**Code moment:** Fit GARCH(1,1) to SPY returns using the `arch` library:

```python
from arch import arch_model
garch = arch_model(returns * 100, vol='GARCH', p=1, q=1)
result = garch.fit(disp='off')
print(result.summary())
# Plot conditional volatility vs realized volatility
```

Show the conditional volatility time series. Students will see it spike during COVID (March 2020), during the 2022 bear market, and during August 2015 — all matching real market stress. Then overlay the VIX for comparison. GARCH(1,1) and the VIX tell a remarkably similar story, despite GARCH using only past returns.

**"So what?":** "GARCH(1,1) is the only GARCH variant you need. GARCH(2,1), EGARCH, GJR-GARCH — they exist, people publish papers about them, and in practice they barely beat (1,1). We'll test this ourselves in the seminar. GARCH is also the baseline your LSTM will compete against in Week 8. Spoiler: beating it is harder than you think."

### Section 5: The Integer Differentiation Problem
**Narrative arc:** This is the intellectual centerpiece of the week — the bridge between classical econometrics and ML. The setup: we've seen that d=0 (prices) is non-stationary and d=1 (returns) throws away memory. The tension: what if there's a sweet spot? The resolution: fractional differentiation finds the minimum d needed for stationarity, preserving maximum memory.

**Key concepts:**
- The differentiation spectrum: d=0 (prices) → d=0.3 → d=0.5 → d=0.7 → d=1 (returns)
- Fractional differentiation via the binomial series
- The fixed-width window (FFD) method for practical computation
- Finding optimal d*: the minimum d where ADF rejects non-stationarity
- The tradeoff: stationarity vs. memory preservation

**The hook:** "Here's the core dilemma of financial time series, and Lopez de Prado calls it the single most important concept in his book. When you take returns (d=1), you get stationarity — but you throw away all memory. The autocorrelation drops to zero. Your model doesn't know that Apple has been trending up for six months. It only knows what happened yesterday. When you keep prices (d=0), you preserve all memory — but the series is non-stationary. Your model extrapolates. This is a lose-lose tradeoff... unless you realize that d doesn't have to be an integer."

**Key formulas:**
"Fractional differentiation extends the binomial series to non-integer d:

$$(1 - B)^d = \sum_{k=0}^{\infty} \binom{d}{k} (-B)^k$$

where B is the backshift operator ($B^k x_t = x_{t-k}$) and the generalized binomial coefficient is:

$$\binom{d}{k} = \frac{d(d-1)(d-2)\cdots(d-k+1)}{k!} = \prod_{i=0}^{k-1} \frac{d-i}{k-i}$$

For integer d=1, this gives: $x_t - x_{t-1}$ (standard first difference).
For d=0.5, you get an infinite weighted sum of past values, with weights that decay slowly.

In practice, we use the Fixed-Width Window (FFD) method from Lopez de Prado, which truncates the infinite sum at the point where the weights drop below a threshold $\tau$:

$$\tilde{x}_t^{(d)} = \sum_{k=0}^{K} w_k x_{t-k}, \quad \text{where } K = \min\{k : |w_k| < \tau\}$$

The weights $w_k$ for a given d:

$$w_0 = 1, \quad w_k = -w_{k-1} \frac{d - k + 1}{k}$$

These weights decay, but slowly — for d=0.5, the 100th weight is still about 0.04. That's the memory we're preserving."

**Code moment:** Build FFD from scratch in NumPy, step by step:

```python
def get_weights_ffd(d, threshold=1e-5):
    """Compute FFD weights for a given d."""
    w = [1.0]
    k = 1
    while abs(w[-1]) >= threshold:
        w.append(-w[-1] * (d - k + 1) / k)
        k += 1
    return np.array(w[::-1])  # reverse for convolution

# Show weights for d=0.3, 0.5, 0.7, 1.0
for d in [0.3, 0.5, 0.7, 1.0]:
    w = get_weights_ffd(d)
    print(f"d={d}: {len(w)} weights, range [{w.min():.4f}, {w.max():.4f}]")
```

Plot the weights for different d values. Students should see that d=1.0 gives exactly [1, -1] (first difference), while d=0.5 gives a long, slowly decaying tail — that tail is the memory being preserved.

**"So what?":** "This is the free lunch of financial feature engineering. Instead of feeding your model returns (which have no memory) or prices (which are non-stationary), you feed it fractionally differentiated prices — stationary, but with memory. In the homework, you'll prove that this actually improves out-of-sample prediction."

### Section 6: Finding the Optimal d*
**Narrative arc:** The practical payoff of Section 5. The setup: we have the theory of fractional differentiation. The tension: how do we choose d? Too low and the series is still non-stationary; too high and we throw away memory. The resolution: a systematic search for d* — the minimum d where ADF rejects non-stationarity.

**Key concepts:**
- The optimization problem: minimize d subject to ADF rejecting non-stationarity at 5% significance
- The correlation with the original series: higher is better (more memory preserved)
- d* varies across stocks: low-volatility utilities need less differencing than volatile tech stocks
- Practical considerations: d* can change over time (it depends on the sample)

**The hook:** "Finding d* is like finding the minimum effective dose of a medication. Too little and the disease (non-stationarity) persists. Too much and you kill the patient (memory). The right dose varies by patient (stock) — a boring utility needs barely any differencing, while a meme stock might need d close to 1.0."

**Key formulas:**
"The optimization:

$$d^* = \min \{ d \in [0, 1] : \text{ADF p-value}(\tilde{x}^{(d)}) < 0.05 \}$$

We also track the correlation between the fractionally differentiated series and the original:

$$\rho(d) = \text{Corr}(\tilde{x}^{(d)}, x)$$

As d increases from 0 to 1, the ADF p-value drops (good — more stationary) and the correlation drops (bad — less memory). d* is the sweet spot."

**Code moment:** For 10 diverse stocks (AAPL, JNJ, XOM, JPM, NVDA, KO, TSLA, PG, META, GS), compute d* using a grid search from 0 to 1 in steps of 0.05. For each d, compute the ADF statistic and the correlation with the original series. Plot both as functions of d. Students should see that d* varies: JNJ (defensive healthcare) might have d* ≈ 0.25, while TSLA (volatile tech/meme) might have d* ≈ 0.65.

**"So what?":** "The optimal d* is stock-specific. A model that applies the same d to all stocks is leaving information on the table. In the homework, you'll build a `FractionalDifferentiator` class that finds d* per stock and uses it as a feature transformation — compatible with sklearn's pipeline framework."

### Section 7: Putting It Together — From Raw Prices to ML-Ready Features
**Narrative arc:** We close the loop, connecting all the tools from this week into a coherent feature engineering pipeline. The tension: you now have multiple representations (prices, returns, fractionally differentiated series, GARCH volatility) — which one should you use? The resolution: use them all, and let the model decide.

**Key concepts:**
- Feature stacking: prices (via fracdiff), returns, volatility (GARCH), lagged features
- The pipeline: raw data → `DataLoader` (Week 1) → `FractionalDifferentiator` → feature matrix
- Sklearn compatibility: `TransformerMixin` pattern
- Preview of Week 4: this feature matrix is exactly what we'll feed into Ridge/Lasso

**The hook:** "You now have more ways to represent a stock's price history than a stock has ways to move. The question isn't 'which representation is best' — it's 'which combination gives the model the most to work with while keeping everything stationary.' We're going to build a pipeline that does this automatically."

**Code moment:** Build a `FractionalDifferentiator` class that inherits from `sklearn.base.BaseEstimator` and `TransformerMixin`, implements `fit()` (finds d* per feature) and `transform()` (applies FFD), and plugs directly into an sklearn `Pipeline`. Then show a complete pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ('fracdiff', FractionalDifferentiator()),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])
```

**"So what?":** "This pipeline will be the starting point for Week 4, where we add 20+ features and train cross-sectional prediction models. The fractional differentiator you build this week is a genuine contribution to your ML toolkit — it's not a toy."

### Closing Bridge

"Let's step back and see the full picture. Last week, you learned that raw prices can't be fed into ML models. This week, you learned that returns — the obvious fix — throw away too much information. The real answer is fractional differentiation, which preserves as much memory as possible while achieving stationarity. You also learned that volatility is more predictable than returns (GARCH captures this with three parameters), and that classical time series models, while limited, set the bar your neural networks must clear.

Next week, we shift from individual stocks to portfolios. We'll learn what 'alpha' actually means, why the Sharpe ratio is the metric everyone in finance cares about, and why the 'optimal' portfolio from a textbook blows up the moment you use it with real data. We'll also meet the Fundamental Law of Active Management — the equation that tells you whether your ML model has any hope of making money, before you ever run a backtest."

## Seminar Exercises

### Exercise 1: Stationarity Testing Marathon
**The question we're answering:** How many of the common financial time series are actually stationary?

**Setup narrative:** "We're going to test everything: prices, returns, log returns, volatility, volume. Some of these will surprise you. The goal is to build an intuition for what stationarity 'looks like' in practice."

**What they build:** For SPY and 4 other assets (a bond ETF like TLT, a commodity like GLD, a volatile stock like TSLA, a defensive stock like JNJ): compute ADF and KPSS tests on raw prices, log prices, simple returns, log returns, rolling 20-day volatility, and volume. Build a table of p-values.

**What they'll see:** Prices and log prices will be non-stationary (ADF p >> 0.05). Returns will be stationary (ADF p << 0.01). Rolling volatility will be borderline — sometimes stationary, sometimes not, depending on the sample period (this is because volatility has long memory but is technically mean-reverting). Volume is usually non-stationary due to trends in market participation.

**The insight:** "Returns are stationary but memoryless. Volatility has memory but questionable stationarity. This asymmetry is the fundamental challenge: the series with the most predictive content is the hardest to work with."

### Exercise 2: Fractional Differentiation — The Cross-Sectional d* Map
**The question we're answering:** Does the optimal differencing order d* vary across asset classes, and can you predict which assets need more differencing from their economic properties?

**Setup narrative:** "The lecture built FFD for SPY and showed the d* sweet spot for a single series. Now you're going to find d* for 15 diverse assets — equities, bonds, commodities, FX — and discover that the 'right' amount of differencing is asset-specific. The pattern reveals something deep about how different markets carry memory in their prices."

**What they build:** Using the FFD implementation from the lecture as a starting point, apply it to 15 assets spanning asset classes (5 equity ETFs: SPY, QQQ, IWM, EEM, XLU; 3 bond ETFs: TLT, IEF, HYG; 3 commodities: GLD, USO, SLV; 2 currency pairs via ETFs: UUP, FXE; 2 volatility-adjacent: VXX/VIXY, SVXY). For each: run the d* grid search from 0 to 1 in steps of 0.05, recording ADF p-value and correlation with original series at each d. Plot d* by asset class. Build a scatter plot: d* vs. trailing 1-year realized volatility — is there a relationship?

**What they'll see:** d* varies dramatically: bond ETFs (TLT, IEF) will have low d* (~0.15-0.30) because their prices are already close to mean-reverting. Commodity ETFs (USO, GLD) will have higher d* (~0.40-0.60). Equity ETFs will be intermediate (~0.30-0.50). Volatility products will be unusual — VXX has structural decay that affects the test. The scatter plot will show a rough positive correlation between volatility and d* — more volatile assets need more differencing.

**The insight:** "The optimal d* is not a universal constant — it's a property of the asset and its data-generating process. Slow-moving, mean-reverting assets (bonds, utilities) need barely any differencing. Fast-moving, trending assets (momentum stocks, commodities in supply shocks) need much more. This is why applying d=1 (standard returns) to everything is wasteful — for bonds, you're throwing away 85% of the information to solve a stationarity problem that d=0.2 would fix."

### Exercise 3: GARCH(1,1) — Fitting and Interpreting
**The question we're answering:** How well does a 3-parameter model from 1986 capture real-world volatility dynamics?

**Setup narrative:** "GARCH(1,1) is going to be your baseline for the rest of the course. Let's fit it properly, understand its parameters, and see what it tells us about the market."

**What they build:** Fit GARCH(1,1) to SPY, TSLA, JNJ, and GLD using the `arch` library. Extract parameters (omega, alpha, beta). Compute persistence (alpha + beta). Plot conditional volatility vs. rolling realized volatility. Evaluate 1-day-ahead forecasts on a held-out period using MSE and QLIKE loss.

**What they'll see:** SPY will have high persistence (alpha + beta ≈ 0.98), moderate alpha (~0.05), high beta (~0.93). TSLA will have higher alpha (more reactive to shocks) and lower beta (less persistent). The conditional volatility will track realized volatility well, especially during crisis periods. QLIKE loss will be lower (better) than MSE for most stocks, because QLIKE handles heteroskedastic targets better.

**The insight:** "GARCH(1,1) captures the most important statistical property of financial markets — that big moves cluster — with just three numbers. The persistence parameter (alpha + beta) tells you how 'sticky' volatility is for each asset. When alpha + beta approaches 1.0, shocks persist almost indefinitely. When it's lower, the market 'forgets' faster."

### Exercise 4: The GARCH Variants — Do They Matter?
**The question we're answering:** Is GARCH(1,1) really all you need, or do the fancier variants add value?

**Setup narrative:** "There are at least 50 GARCH variants in the literature — EGARCH, GJR-GARCH, TARCH, and counting. Let's test whether any of them meaningfully beat the simplest version."

**What they build:** For SPY and TSLA, fit GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), and GARCH(2,1). Compare by AIC, BIC, out-of-sample MSE, and out-of-sample QLIKE.

**What they'll see:** AIC/BIC will sometimes favor EGARCH or GJR-GARCH (they capture the asymmetric volatility response — volatility increases more after negative returns than positive ones). But out-of-sample performance differences will be tiny — typically within 1-2% of each other. The added complexity of fancier models rarely translates into meaningfully better forecasts.

**The insight:** "GJR-GARCH's leverage effect is real — negative returns do increase volatility more than positive returns of the same magnitude. But the out-of-sample improvement is marginal. In production, simplicity has value: GARCH(1,1) is easier to fit, more stable, and less likely to overfit. Unless you have a strong reason to use a variant, don't."

## Homework: "The Fractional Differentiation Study"

### Mission Framing

The lecture made a bold claim: that there's a sweet spot between raw prices (too much memory, non-stationary) and returns (stationary but memoryless). Fractional differentiation, Lopez de Prado argues, finds that sweet spot. You're going to test this claim on 50 real stocks and see if it holds up or if Lopez de Prado was having us on.

You'll find the optimal d* for each stock, build a prediction model using three different feature representations, and compare them head-to-head. You'll also fit GARCH(1,1) to every stock and forecast volatility — building the baseline that your LSTM will compete against in Week 8. Along the way, you'll build a `FractionalDifferentiator` class that's compatible with sklearn, which means it plugs directly into the pipelines you'll use for the rest of the course.

Here's what makes this homework more than a replication exercise: the optimal d* varies dramatically across stocks, and the reasons are economically meaningful. Defensive utilities with low volatility and slow-moving prices need barely any differencing. High-volatility tech stocks need much more. Meme stocks are a story unto themselves. The pattern reveals something deep about how different stocks carry information in their price histories.

### Deliverables

1. **For a universe of 50 US stocks (spanning sectors and market caps), find the optimal d* per stock.** Use the FFD method with a grid search from 0.0 to 1.0 in steps of 0.01. For each stock, d* is the minimum d where the ADF test rejects non-stationarity at the 5% level. Report a table: ticker, sector, d*, correlation with original series at d*. Plot d* by sector — is there a pattern?

2. **Build a prediction model (Ridge regression) using three feature sets:**
   - (a) Raw log returns (d=1)
   - (b) Log prices (d=0) — yes, this is non-stationary, but test it anyway as a baseline
   - (c) Fractionally differentiated log prices at d* (per stock)

   For each: use 5 lagged values as features. Train with expanding-window CV (train on all data up to month t, predict 1-day return at month t+1). Report out-of-sample R-squared and correlation between predicted and actual returns.

3. **Compare out-of-sample R-squared across all three feature sets.** Show that (c) typically preserves more signal than (a) while being better-behaved than (b). Create a scatter plot: R-squared of fracdiff features vs. R-squared of return features. Are there stocks where fracdiff helps more than others?

4. **Fit GARCH(1,1) to each of the 50 stocks.** Report parameters (omega, alpha, beta) and persistence (alpha + beta) for each. Forecast 1-day-ahead volatility on a held-out period. Evaluate with QLIKE loss. Identify the stocks with highest and lowest persistence — do they match your intuition (e.g., are utilities low-persistence and tech stocks high-persistence)?

5. **Build a `FractionalDifferentiator` class** compatible with sklearn's `TransformerMixin`. It should:
   - `fit(X)`: find d* for each column (feature) in X
   - `transform(X)`: apply FFD at the fitted d* to each column
   - `get_params()`: return the fitted d* values
   - Be usable in an sklearn `Pipeline`

### What They'll Discover

- The optimal d* for utility stocks (Duke Energy, Southern Company) is barely above 0.1-0.2, while volatile tech stocks (NVDA, TSLA) need d* near 0.5-0.7. The "right" amount of differentiation depends on how mean-reverting the stock is. Stable, slow-moving stocks have prices that are already close to stationary — they need barely any help. Volatile, trending stocks need much more aggressive differencing.

- Fractionally differentiated features typically improve out-of-sample R-squared by 5-15% over standard returns for stocks with low d*. For stocks where d* is close to 1.0, there's almost no improvement — because d* ≈ 1 means fractional differentiation is basically just taking returns.

- GARCH persistence (alpha + beta) is remarkably consistent within sectors: financials tend to have high persistence (shocks to bank volatility last a long time), while consumer staples have lower persistence (volatility mean-reverts faster). This isn't an accident — it reflects how information propagates through different parts of the economy.

- The raw log prices model (d=0) will occasionally have the highest R-squared — but this is a trap. It's fitting the trend, not the signal. On genuinely out-of-sample data (e.g., data from a different market regime), it will fail catastrophically. This is a lesson in why stationarity tests matter.

### Deliverable
A complete Jupyter notebook containing: the d* analysis table, the three-way comparison of feature sets, GARCH(1,1) results for all 50 stocks, and the `FractionalDifferentiator` class. Code should be clean, modular, and ready for reuse in Week 4.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Stationarity (ADF/KPSS tests) | Demo: run ADF/KPSS on SPY prices, returns, log prices | Exercise 1: test 5 assets × 6 representations, build full p-value table | At scale: ADF grid search for 50 stocks to find per-stock d* |
| Autocorrelation & market efficiency | Demo: ACF of SPY returns vs. squared returns side by side | Not covered (done in lecture) | Reference: interpret ACF structure when analyzing GARCH residuals |
| Classical time series (AR, MA, ARIMA) | Demo: fit ARIMA to SPY, show residual diagnostics | Not covered (done in lecture) | Reference: ARIMA as context for why fractional d matters |
| GARCH(1,1) volatility modeling | Demo: fit GARCH to SPY, plot conditional vs. realized vol | Exercise 3: fit GARCH to 4 diverse assets, compare persistence by asset type | At scale: GARCH for 50 stocks, report parameters and QLIKE loss |
| GARCH variants comparison | Not covered (deferred to seminar) | Exercise 4: compare GARCH(1,1) vs. EGARCH vs. GJR-GARCH on 2 assets | Reference: confirm (1,1) sufficiency for homework baseline |
| Fractional differentiation (FFD) | Demo: build FFD from scratch in NumPy, show weights for various d | Exercise 2: find d* for 15 assets across asset classes, map d* vs. volatility | At scale: d* for 50 stocks, build `FractionalDifferentiator` class |
| Finding optimal d* | Demo: grid search for SPY, plot ADF p-value and correlation vs. d | Exercise 2: cross-asset d* comparison | At scale: per-stock d* with sector-level patterns |
| ML-ready feature pipeline | Demo: show `FractionalDifferentiator` with sklearn `TransformerMixin` | Not covered (built in homework) | Build: sklearn-compatible `FractionalDifferentiator` class |

## Key Stories & Facts to Weave In

1. **The M4 Competition (2018).** The M4 forecasting competition tested 60 methods on 100,000 time series. The winner was a hybrid of exponential smoothing and a neural network. Pure ML methods (standalone LSTMs, etc.) generally underperformed statistical methods. This was a shock to the ML community and led to significant soul-searching about whether neural networks are actually good at time series forecasting. The lesson: don't underestimate simple baselines.

2. **Robert Engle's Nobel Prize (2003).** Robert Engle won the Nobel Prize in Economics for developing ARCH (Autoregressive Conditional Heteroskedasticity) in 1982. His co-laureate, Clive Granger, won for cointegration. Between them, they provided the foundational tools for understanding financial time series. GARCH, the extension by Tim Bollerslev (1986), is still used daily at every major bank and hedge fund. Three parameters, four decades of dominance.

3. **LTCM's collapse (1998).** Long-Term Capital Management had two Nobel laureates on staff (Myron Scholes and Robert Merton), $125 billion in assets, and models that said a loss of their magnitude wouldn't happen in the lifetime of the universe. It happened in four months. Their models assumed that the distributions of returns were Gaussian and stationary. They were neither. The firm lost $4.6 billion and nearly brought down the global financial system. The Federal Reserve organized a $3.6 billion bailout — not because LTCM deserved saving, but because its counterparties included every major bank on Wall Street.

4. **The VIX doubling of February 5, 2018 (Volmageddon).** The VIX doubled from ~17 to ~37 in a single day. XIV, an inverse volatility ETN with $1.9 billion in assets, lost 96% of its value. Credit Suisse (XIV's issuer) terminated the note. Thousands of retail investors lost their life savings. The product had returned ~400% over five years before it blew up. GARCH would have shown that volatility persistence was 0.98 — meaning that while low vol tends to persist, when it spikes, the spike persists too. The trade was profitable 95% of the time and catastrophic 5% of the time. No amount of backtesting on the 95% prepares you for the 5%.

5. **Lopez de Prado's career arc.** Marcos Lopez de Prado managed over $13 billion in assets as head of machine learning at AQR Capital Management and previously at Guggenheim Partners. His book *Advances in Financial Machine Learning* (2018) was unusual: a senior portfolio manager publicly sharing his methodology. The fractional differentiation chapter (Chapter 5) was the most controversial — some academics argued it was theoretically unsound, but practitioners found it useful. Lopez de Prado's response: "The test of a methodology is not whether it's theoretically pure, but whether it makes money. FFD passes that test."

6. **Why markets are 'almost efficient.'** Eugene Fama won the Nobel Prize in 2013 for the Efficient Market Hypothesis. Robert Shiller won the same year for showing that markets are not efficient — they exhibit predictable bubbles and crashes. The Nobel committee gave the prize to both, essentially saying: "We don't know who's right." The practical truth: markets are efficient enough that daily return autocorrelation is near zero, but inefficient enough that a 0.03 information coefficient can sustain a career. Your models live in that narrow gap.

7. **The square-root-of-time rule and its failures.** The annualization formula $\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$ assumes that daily returns are independent. They're not — volatility clustering means that high-vol days follow high-vol days. The square-root-of-time rule underestimates risk over short horizons (when vol is clustered) and can overestimate it over very long horizons (when vol mean-reverts). JPMorgan's RiskMetrics system (1994) was the first to account for this using EWMA (exponentially weighted moving average) volatility — a simplified version of GARCH.

8. **Andrew Lo's Adaptive Markets Hypothesis (2004).** MIT's Andrew Lo proposed an alternative to Fama's EMH: markets are efficient, but efficiency varies over time as participants adapt. In calm periods, arbitrage opportunities are quickly eliminated (high efficiency). After crises, participants are disoriented, patterns change, and efficiency temporarily drops. This has direct implications for ML: your model's performance will degrade during regime changes, and you need to account for that.

## Cross-References
- **Builds on:** The non-stationarity and fat-tails discussion from Week 1 (Section 4), the returns math from Week 1 (Section 5), and the `DataLoader` class from Week 1's homework.
- **Sets up:** The `FractionalDifferentiator` class becomes a standard feature transformation in Weeks 4-5. The GARCH(1,1) model becomes the baseline for the LSTM volatility forecasting in Week 8. The stationarity testing methodology (ADF/KPSS) is used throughout the course whenever we introduce new time series features. The expanding-window CV concept (introduced here informally) is formalized in Week 4.
- **Recurring thread:** The "simple vs. sophisticated" thread starts here. GARCH(1,1) vs. GARCH variants is the first instance of a pattern we'll see repeatedly: simple models that work surprisingly well against complex ones. Week 5 (trees vs. linear models), Week 8 (GARCH vs. LSTM), Week 9 (XGBoost vs. foundation models) — each time, the simple baseline refuses to die gracefully.

## Suggested Reading
- **Lopez de Prado, *Advances in Financial Machine Learning*, Chapter 5 (Fractional Differentiation)** — The primary source for the FFD method. Lopez de Prado's prose is dense and academic, but the ideas are genuinely novel. Focus on the intuition and the implementation, not the proofs.
- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity"** — The original GARCH paper. Surprisingly readable for a 1986 econometrics paper. Worth skimming to see how the ideas developed.
- **Makridakis, Spiliotis & Assimakopoulos (2018), "The M4 Competition"** — The paper that made the ML community confront the fact that statistical methods beat neural networks on many forecasting tasks. Essential reading for anyone who thinks "deep learning always wins."
