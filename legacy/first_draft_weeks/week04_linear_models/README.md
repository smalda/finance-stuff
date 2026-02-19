# Week 4 — Cross-Sectional Return Prediction: Linear Models

> **The bread and butter of quantitative asset management: rank 500 stocks by predicted return, buy the top, short the bottom, repeat monthly. It's been working since 1990.**

## Prerequisites
Weeks 1-3, specifically: the `DataLoader` class (Week 1), fractional differentiation and stationarity testing (Week 2), the Information Coefficient (IC), Sharpe ratio, and Fundamental Law of Active Management (Week 3), and transaction cost modeling (Week 3). Students should understand what alpha means, why breadth matters, and how to evaluate a strategy with proper risk metrics. Familiarity with scikit-learn's `Ridge`, `Lasso`, and `Pipeline` is assumed from general ML background.

## The Big Idea

Everything you've learned so far has been building toward this moment. You have clean data (Week 1). You know how to make it stationary (Week 2). You understand what alpha means and how to evaluate it (Week 3). Now we're going to use it.

The cross-sectional prediction problem is the core of quantitative asset management, and it works like this: at the end of each month, you have N stocks, each described by a vector of features — momentum, volatility, size, value, and so on. Your job is to predict which stocks will have the highest returns next month, relative to the other stocks. Not the absolute return — the ranking. You go long the top quintile and short the bottom quintile. If your ranking is even slightly better than random, you make money. And the Fundamental Law (IC × sqrt(BR)) tells you that "slightly better than random" across 500 stocks is more than enough.

This is not a hypothetical exercise. Gu, Kelly, and Xiu published "Empirical Asset Pricing via Machine Learning" in 2020 — a landmark paper that tested 8 model classes on 30,000 stocks over 60 years using 94 firm characteristics. They found that even Ridge regression — the simplest regularized linear model — produces meaningful out-of-sample predictions of stock returns. The out-of-sample R-squared was about 0.4%, which sounds tiny until you realize that it translates to significant portfolio performance when applied across thousands of stocks. Neural nets and gradient-boosted trees did better (R-squared ~1.8% for neural nets), but the linear baseline was not zero. There is signal in the cross-section, and linear models can find some of it.

This week, you'll replicate the core of the Gu-Kelly-Xiu framework: build a feature matrix of firm characteristics, train OLS/Ridge/Lasso/Elastic Net with expanding-window cross-validation, measure out-of-sample IC, and construct a long-short portfolio. The critical methodology lesson is expanding-window CV — the only correct way to evaluate financial ML models. Never shuffle across time. Train on everything up to month t, predict month t+1. This is the evaluation framework we'll use for the remaining 14 weeks of the course.

## Lecture Arc

### Opening Hook

"In 2020, Shihao Gu, Bryan Kelly, and Dacheng Xiu published a paper that ended a decades-long debate: can machine learning predict stock returns? They tested 8 model classes — from OLS to neural networks — on 30,000 stocks over 60 years, using 94 firm characteristics as features. The answer was yes. Neural networks achieved an out-of-sample monthly R-squared of about 0.4%. Before you scoff at that number, let's do some math. An R-squared of 0.4% on 3,000 stocks, rebalanced monthly, translates to a long-short portfolio with a Sharpe ratio above 2. Two. That's better than 99% of hedge funds. And that's from a model that 'explains' less than half a percent of return variation. Welcome to the economics of cross-sectional prediction, where tiny edges compound across enormous breadth."

### Section 1: The Cross-Sectional Prediction Problem
**Narrative arc:** We frame the problem precisely, distinguishing it from time-series prediction (which is what most ML engineers instinctively think of when they hear "stock prediction"). The setup: you're not predicting whether Apple will go up tomorrow — you're predicting whether Apple will outperform Microsoft. The tension: this is a ranking problem, not a regression problem, and the metrics are different. The resolution: IC and quantile portfolio analysis are the right evaluation tools.

**Key concepts:**
- Cross-sectional vs. time-series prediction: predicting across stocks vs. across time
- Why cross-section wins: massive breadth (500+ independent bets per month)
- The feature matrix: N stocks × K features at each time t
- Panel data: stacking the cross-sections over time → (N × T) × K matrix
- The prediction target: typically next-month excess return (return minus market return)
- Why we predict excess returns, not raw returns: removing market risk

**The hook:** "Here's a question that separates ML engineers from quant researchers: 'Predict whether Apple stock will go up next month.' An ML engineer starts thinking about LSTMs, attention mechanisms, and historical price sequences. A quant researcher says: 'Wrong question.' The right question is: 'Will Apple outperform the median stock next month?' This reframing — from absolute to relative prediction — is the most important conceptual shift in this entire course. It changes the loss function, the features, and the evaluation metric. And it makes the problem dramatically more tractable."

**Key formulas:**
"The cross-sectional prediction problem at time t:

Given features $\mathbf{x}_{i,t}$ for stocks $i = 1, ..., N_t$, predict excess returns:

$$r_{i,t+1}^{e} = r_{i,t+1} - r_{m,t+1}$$

The model:

$$\hat{r}_{i,t+1}^{e} = f(\mathbf{x}_{i,t}; \theta_t)$$

where $f$ can be linear (this week) or nonlinear (Weeks 5 and 7+).

We evaluate with the Information Coefficient — the rank correlation between predictions and realized returns:

$$IC_t = \text{Spearman}(\hat{r}_{i,t+1}^{e}, r_{i,t+1}^{e})$$

In ML terms, this is just Spearman's $\rho$ between your predictions and labels. In finance, it has a name and a mystique, but it's the same thing."

**Code moment:** Show the shape of the data we're building toward. A pandas DataFrame with a MultiIndex (date, ticker), columns for features (momentum, volatility, etc.), and a target column (next-month excess return). This is the cross-sectional panel that all subsequent models will consume.

**"So what?":** "Cross-sectional prediction is why quant funds hire ML engineers. It's a well-posed supervised learning problem with massive training data (thousands of stocks × hundreds of months), clear features (firm characteristics), and a measurable objective (IC). The catch: the signal-to-noise ratio is atrocious. An IC of 0.05 is excellent. An R-squared of 1% is excellent. Calibrate your expectations now."

### Section 2: Feature Engineering for Stocks — The 94 Characteristics
**Narrative arc:** We build the feature set that drives cross-sectional prediction. The setup: ML engineers are used to having features handed to them (or learning them from raw data). The tension: in finance, feature engineering is the craft — decades of academic research have identified the features that predict returns. The resolution: we implement the most important ones and explain the economic intuition behind each.

**Key concepts:**
- Momentum features: past winners tend to keep winning (1, 3, 6, 12-month returns)
- The "skip" in 12-month momentum: skip the most recent month (short-term reversal)
- Value features: book-to-market ratio, earnings-to-price
- Size: market capitalization (small stocks earn more, historically)
- Volatility: realized volatility (high-vol stocks tend to underperform — the "low volatility anomaly")
- Volume features: turnover, Amihud illiquidity
- Reversal: 1-month return (recent losers tend to bounce back)
- Quality/profitability: ROE, gross profit margin

**The hook:** "In 1993, Jegadeesh and Titman published one of the most influential papers in finance: stocks that went up in the past 12 months tend to keep going up, and stocks that went down tend to keep going down. This is the momentum anomaly, and it's been replicated in almost every market and time period ever studied. It earned a Nobel Prize's worth of citations and remains the single most robust stock market anomaly. Your ML model will rediscover momentum. But it'll also find something more interesting: momentum interacts with volatility in a way that linear models can't capture. That nonlinear interaction is where trees and neural nets earn their keep — but that's Week 5's story."

**Key formulas:**
"The standard momentum feature:

$$\text{MOM}_{12,1}(i, t) = \frac{P_{i,t-1}}{P_{i,t-13}} - 1$$

The 'skip-1' means we use the return from t-13 to t-1, excluding the most recent month. Why? Because the most recent month shows reversal (recent losers bounce, recent winners fade). Momentum and reversal are opposite effects operating at different timescales.

Short-term reversal:

$$\text{REV}(i, t) = r_{i, t-1}$$

Just the prior month's return. Negative correlation with next month's return.

Volatility (realized, 20-day):

$$\text{VOL}_{20}(i, t) = \sqrt{\frac{252}{20} \sum_{j=0}^{19} r_{i, t-j}^2}$$

Annualized standard deviation of the last 20 trading days.

Amihud illiquidity:

$$\text{ILLIQ}(i, t) = \frac{1}{D} \sum_{d=1}^{D} \frac{|r_{i,d}|}{\text{DolVol}_{i,d}}$$

Price impact per dollar of volume. Higher = less liquid = harder to trade."

**Code moment:** Build the feature matrix step by step:

```python
# Momentum features
features['mom_1m'] = prices.pct_change(21)    # 1-month
features['mom_3m'] = prices.pct_change(63)    # 3-month
features['mom_12m_skip1'] = prices.shift(21).pct_change(252)  # 12-month, skip 1

# Volatility
features['vol_20d'] = returns.rolling(20).std() * np.sqrt(252)

# Volume
features['turnover'] = volume / shares_outstanding
features['amihud'] = (returns.abs() / dollar_volume).rolling(21).mean()
```

Build features for 200+ stocks. Show the resulting DataFrame shape, examine summary statistics, and note which features have the most missing data (value features like book-to-market are only available quarterly).

**"So what?":** "These features aren't arbitrary — each one represents a decades-old research finding about what predicts stock returns. Momentum works because investors underreact to information. Value works (sometimes) because investors overpay for growth. The low-volatility anomaly works because institutional investors are forced to buy high-beta stocks to target their return benchmarks. Your ML model doesn't need to know these stories — it just needs the features. But understanding the economics helps you build better features and interpret the model's decisions."

### Section 3: Linear Models — OLS, Ridge, Lasso, Elastic Net
**Narrative arc:** We apply the standard ML regularization toolkit to the cross-sectional prediction problem. The setup: OLS is the natural starting point but overfits badly with 20+ correlated features. The tension: which regularization is right for financial data? The resolution: Ridge preserves all features (good when all features carry some signal), Lasso selects features (good when most features are noise), Elastic Net hedges between the two.

**Key concepts:**
- Why OLS fails with many correlated features: multicollinearity → unstable coefficients
- Ridge (L2): shrinks coefficients toward zero but keeps all features
- Lasso (L1): shrinks coefficients to exactly zero — automatic feature selection
- Elastic Net: combination of L1 and L2
- The bias-variance tradeoff in the context of noisy financial returns
- Regularization as a form of Bayesian prior (Ridge = Gaussian prior, Lasso = Laplace prior)

**The hook:** "In the Gu-Kelly-Xiu study, OLS (no regularization) had the worst out-of-sample performance of all 8 model classes. It overfit spectacularly — fitting noise in 94 features as if it were signal. Ridge regression, with a single hyperparameter controlling the penalty, nearly doubled the out-of-sample R-squared. One hyperparameter. The lesson: in financial prediction, where signal-to-noise is extremely low, regularization isn't optional — it's the difference between a model that works and a model that hallucinates."

**Key formulas:**
"Ridge regression minimizes:

$$\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^{N} (r_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \alpha \|\boldsymbol{\beta}\|_2^2$$

The penalty $\alpha \|\boldsymbol{\beta}\|_2^2$ shrinks coefficients toward zero. Larger $\alpha$ → more shrinkage → less overfitting → more bias. The solution is analytical:

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{r}$$

Compare with OLS: $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T \mathbf{r}$. The only difference is $\alpha \mathbf{I}$ — a small diagonal addition that stabilizes the matrix inversion. Same idea as Ledoit-Wolf shrinkage from Week 3 — add structure to a noisy estimate.

Lasso replaces the L2 penalty with L1:

$$\mathcal{L}_{\text{Lasso}} = \sum_{i=1}^{N} (r_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \alpha \|\boldsymbol{\beta}\|_1$$

No analytical solution — requires coordinate descent. But the L1 penalty drives coefficients to exactly zero, performing automatic feature selection. In a world with 94 features and a signal-to-noise ratio of 0.001, knowing which features to ignore is almost as valuable as knowing which to use."

**Code moment:** Train all four models on the cross-sectional panel data. Show that OLS coefficients are wild (some features have coefficients of ±100 while others are near zero — multicollinearity is visible). Ridge coefficients are smooth and stable. Lasso selects perhaps 8-12 of the 20 features and zeros out the rest. Compare in-sample R-squared (OLS wins — it always does) vs. out-of-sample IC (Ridge wins).

**"So what?":** "Ridge regression is the Bayesian prior that says 'all features are probably small.' Lasso is the prior that says 'most features are zero.' In finance, where features are correlated and signal is weak, Ridge tends to win because it retains the small, correlated signals that Lasso discards. But Lasso's feature selection is interpretable — it tells you which characteristics the market is currently pricing. Both are useful."

### Section 4: Expanding-Window Cross-Validation — The Only Way to Evaluate Financial ML
**Narrative arc:** This is the methodology section that separates correct financial ML from incorrect financial ML. The setup: standard k-fold CV shuffles data randomly. The tension: in financial time series, shuffling violates causality — you're training on the future and predicting the past. The resolution: expanding-window CV respects temporal order and is the standard evaluation method for the rest of the course.

**Key concepts:**
- Why k-fold CV fails in finance: temporal dependence, information leakage
- Expanding-window CV: train on all data up to time t, predict time t+1
- Walk-forward analysis: same idea, applied to strategy evaluation
- The train/validation/test split in time: training period → validation gap → prediction period
- Hyperparameter tuning within the expanding window: nested cross-validation in time

**The hook:** "If you use standard 5-fold cross-validation on financial data, you will get a beautiful, impressive, and completely fake result. Here's why: imagine your training set contains data from January 2020 and your test fold contains data from December 2019. Your model has seen the future — it knows the COVID crash is coming — and it's using that information to predict the past. The CV score looks great, but it's measuring your model's ability to time-travel, not to predict. In production, your model doesn't have a time machine."

**Key formulas:**
"Expanding-window CV at time t:

- **Training set:** all cross-sections from time 1 to time t-1
- **Prediction:** the cross-section at time t

Formally: for each month $t = T_0, T_0+1, ..., T$:

$$\text{Train: } \{(\mathbf{x}_{i,s}, r_{i,s+1})\}_{s=1, i=1}^{t-1, N_s}$$
$$\text{Predict: } \hat{r}_{i,t+1} = f(\mathbf{x}_{i,t}; \hat{\theta}_{1:t-1})$$
$$\text{Evaluate: } IC_t = \text{Spearman}(\hat{r}_{i,t+1}, r_{i,t+1})$$

The final metric is the average IC across all out-of-sample months:

$$\overline{IC} = \frac{1}{T - T_0 + 1} \sum_{t=T_0}^{T} IC_t$$

This is also called 'rolling IC' or 'walk-forward IC' in the industry."

**Code moment:** Implement expanding-window CV from scratch:

```python
results = []
for t in range(min_train_months, total_months):
    # Everything up to month t is training data
    train = panel[panel['month'] <= t]
    test = panel[panel['month'] == t + 1]

    model = Ridge(alpha=best_alpha)
    model.fit(train[features], train['target'])

    preds = model.predict(test[features])
    ic = spearmanr(preds, test['target']).correlation
    results.append({'month': t + 1, 'ic': ic})
```

Plot the rolling IC over time. Students should see that IC fluctuates — positive in some months, negative in others — but the average should be around 0.02-0.05. Plot the cumulative IC to show the trend.

Then show the comparison: expanding-window IC vs. standard 5-fold IC. The 5-fold IC will be systematically higher (because of information leakage). This is a vivid demonstration of why temporal CV matters.

**"So what?":** "Expanding-window CV is the standard evaluation method for the rest of this course. Every model from Week 4 to Week 18 will be evaluated this way. Never shuffle across time. Never put future data in the training set. This is not just good practice — it's the difference between a backtest that's informative and one that's a fairy tale."

### Section 5: The Gu-Kelly-Xiu Framework
**Narrative arc:** We place our work in the context of the field's most important benchmark paper. The setup: Gu, Kelly, and Xiu tested 8 models on the most comprehensive dataset ever assembled. The tension: which model wins? The resolution: trees and neural nets capture nonlinear interactions that linear models miss, but the differences are quantitatively specific and worth studying.

**Key concepts:**
- The GKX dataset: 30,000 stocks × 720 months × 94 characteristics
- The 8 models: OLS, ENet, PLS, PCR, Random Forest, GBT, NN1, NN2-5
- Out-of-sample R-squared as the metric
- The ranking: NN > GBT > RF > PCR > ENet > PLS > OLS
- The key interaction: momentum × volatility — trees and NNs capture this, linear models can't
- What "R-squared of 0.4%" means in portfolio terms

**The hook:** "Gu, Kelly, and Xiu's paper has over 3,000 citations in four years. It settled a question that had been debated for decades: yes, ML can predict stock returns out of sample. The surprise wasn't that neural nets work — it was that the improvement over regularized linear models was modest. Neural nets achieved monthly R-squared of ~0.4%. Elastic Net achieved ~0.2%. The gap is real but not enormous. The lesson: most of the predictive power comes from the features, not the model. The model architecture is a second-order effect."

**Key formulas:**
"GKX's out-of-sample R-squared:

$$R^2_{oos} = 1 - \frac{\sum_{(i,t) \in \text{test}} (r_{i,t} - \hat{r}_{i,t})^2}{\sum_{(i,t) \in \text{test}} r_{i,t}^2}$$

Note: the denominator is not the total variance around the mean — it's the sum of squared returns. This is because the unconditional expected return in the cross-section is approximately zero (by construction, since we're predicting excess returns). This is the standard measure in the asset pricing literature.

Key GKX results:
- OLS: $R^2_{oos} \approx 0.08\%$ (barely positive — but positive!)
- Elastic Net: $R^2_{oos} \approx 0.20\%$
- Random Forest: $R^2_{oos} \approx 0.22\%$
- Gradient Boosted Trees: $R^2_{oos} \approx 0.28\%$
- Neural Net (3-layer): $R^2_{oos} \approx 0.40\%$

These numbers look tiny. But they translate to economically significant portfolio returns when applied to 3,000+ stocks."

**Code moment:** Reference the numbers from the paper and contextualize. Show a table comparing model performance. Then compute: if IC = 0.05 and BR = 500 × 12 = 6,000, what's the IR? What annualized Sharpe does this imply for a long-short portfolio? Use the Fundamental Law from Week 3 to connect the abstract R-squared numbers to concrete portfolio performance.

**"So what?":** "Your Ridge/Lasso models this week won't hit GKX's numbers — you have 200 stocks and 20 features, not 30,000 stocks and 94 features. But the methodology is identical. And the relative ordering of models will hold: regularized linear models provide a meaningful baseline. Trees and neural nets (Weeks 5 and 7) will improve on this baseline by capturing nonlinear interactions. The improvement is real but modest."

### Section 6: Quantile Portfolio Analysis — From Predictions to Profits
**Narrative arc:** We close the loop from predictions to portfolio performance. The setup: you have a ranked list of stocks. The tension: how do you turn rankings into a portfolio, and how do you evaluate whether the portfolio makes money? The resolution: quantile portfolio sorts and long-short analysis.

**Key concepts:**
- Quantile sorts: divide stocks into quintiles (or deciles) by predicted return
- Long-short portfolio: long top quintile, short bottom quintile
- Monotonic spread: if the model works, returns should increase from bottom to top quintile
- `alphalens`: the standard tool for factor/signal analysis
- Factor returns, IC, turnover, and the factor decay analysis

**The hook:** "The acid test of a cross-sectional prediction model isn't its R-squared or even its IC. It's this: if you sort stocks into quintiles by predicted return, does the top quintile actually outperform the bottom quintile? If the spread is monotonic — Q1 < Q2 < Q3 < Q4 < Q5 — your model has learned something real about the cross-section. If it's not monotonic, you might have noise that correlates with the target by accident."

**Key formulas:**
"Long-short portfolio return at time t:

$$r_{LS,t} = \frac{1}{N_{Q5}} \sum_{i \in Q5} r_{i,t} - \frac{1}{N_{Q1}} \sum_{i \in Q1} r_{i,t}$$

where Q5 is the top quintile (highest predicted returns) and Q1 is the bottom.

This portfolio is approximately dollar-neutral (zero net investment) and market-neutral (approximately zero beta), so its return is close to pure alpha."

**Code moment:** Sort stocks into quintiles by prediction each month. Compute the average return of each quintile. Plot the quintile bar chart — it should show an upward slope from Q1 to Q5. Then compute the long-short portfolio returns and plot cumulative performance. Use `alphalens` for the complete factor analysis:

```python
import alphalens
factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
    predictions,
    returns,
    quantiles=5,
    periods=(1, 5, 21)
)
alphalens.tears.create_full_tear_sheet(factor_data)
```

The tear sheet will show IC, factor returns by quantile, and turnover — a complete picture of whether the signal is investable.

**"So what?":** "This is the standard evaluation pipeline at every quant fund. You don't publish an R-squared — you show a quintile spread chart. If the spread is positive and monotonic, you have a signal worth trading. If the spread includes transaction costs and is still positive, you have a strategy worth deploying."

### Closing Bridge

"You've just built your first cross-sectional alpha model — the bread and butter of quantitative asset management. It uses simple linear models (Ridge, Lasso), standard features (momentum, volatility, value), and rigorous expanding-window evaluation. The IC is small — maybe 0.02-0.05 — but the Fundamental Law tells us that's enough.

Next week, we replace the linear model with gradient-boosted trees — XGBoost and LightGBM. These models dominate production quant finance, and the reason is specific: they capture nonlinear interactions between features that linear models miss. The Gu-Kelly-Xiu paper showed that the momentum × volatility interaction is the single most important nonlinear effect in stock prediction. Trees find it automatically. Linear models can't. You'll also discover SHAP values — a way to see exactly which features and interactions your model is using, which is essential for trust (both your own trust in the model and your boss's)."

## Seminar Exercises

### Exercise 1: Building the Feature Matrix
**The question we're answering:** Can we construct a clean, complete feature matrix for 200+ stocks using only free data?

**Setup narrative:** "This is the foundation of everything we'll build for the next 14 weeks. We're going to compute 20+ features for 200+ stocks, handle missing data, and produce a panel dataset ready for ML. Think of this as building your feature store."

**What they build:** Using `yfinance` and the `DataLoader` from Week 1, compute the following features for 200+ stocks (e.g., S&P 500 constituents), monthly, from 2005-2024:
- Momentum: 1-month, 3-month, 6-month, 12-month (skip 1)
- Reversal: prior month return
- Volatility: 20-day and 60-day realized vol
- Volume: average daily turnover, Amihud illiquidity
- Size: log market cap
- Technical: 50-day / 200-day moving average ratio, RSI

Produce a clean pandas DataFrame with MultiIndex (date, ticker).

**What they'll see:** The feature matrix will have significant missing data — especially for smaller stocks and earlier dates. Some features (like book-to-market) require fundamental data that `yfinance` provides inconsistently. The features are correlated: momentum variants are highly correlated with each other, volatility measures are correlated.

**The insight:** "Feature engineering in finance is 80% data cleaning and 20% creativity. The features themselves are well-known — the craft is in computing them correctly with imperfect data."

### Exercise 2: Regularization Path — How Lambda Shapes the Cross-Section
**The question we're answering:** What does the model "see" at different regularization strengths, and when does aggressive shrinkage help versus hurt?

**Setup narrative:** "The lecture demonstrated that Ridge beats OLS. But how much regularization is optimal, and does the answer change over time? You're going to trace the full regularization path and watch coefficients appear, shrink, and vanish as you dial lambda from zero to infinity."

**What they build:** For one expanding-window fold (e.g., train on 2005-2018, test on 2019), sweep Ridge alpha from 0.001 to 10,000 on a log scale (50 values). For each alpha: record all 20 coefficients and the out-of-sample IC. Plot the regularization path (coefficients vs. log-alpha). Repeat for Lasso — plot the same path and note which features survive at each alpha. Then do a time-stability check: pick the best alpha from 2019 and test whether that same alpha is still optimal in 2020, 2021, 2022. Plot "best alpha over time."

**What they'll see:** The Ridge path shows all coefficients shrinking smoothly toward zero — momentum features are the last to shrink (they carry the most signal). The Lasso path shows features dropping to zero one by one — the order of elimination reveals a feature importance ranking that's complementary to SHAP (Week 5). The best alpha is not constant over time — it fluctuates, with higher regularization preferred during volatile regimes (when signal-to-noise drops). The time-stability check will show moderate alpha drift, suggesting annual re-tuning is enough.

**The insight:** "The regularization path is a diagnostic tool, not just a tuning method. The order in which Lasso kills features tells you which features are carrying real signal vs. noise. The fact that optimal alpha shifts across regimes tells you that the signal-to-noise ratio itself is non-stationary — your model's confidence should vary with market conditions."

### Exercise 3: The Leakage Trap — 5-Fold vs. Expanding-Window vs. Purged
**The question we're answering:** How dramatically does evaluation methodology change your assessment of a model's skill?

**Setup narrative:** "You're about to see why methodology is the most important thing in financial ML — more important than the model, more important than the features. The same Ridge model, evaluated three different ways, will produce three very different answers. Only one of them is real."

**What they build:** Take the best Ridge model from Exercise 2. Evaluate it three ways on the same data: (a) standard 5-fold CV (shuffled), (b) expanding-window CV (temporal, but without purging), (c) expanding-window CV with a 21-day purge buffer (remove training samples within 21 days of the test boundary). Report IC, R-squared, and quintile spread for each method. Plot the three IC series over time on the same axes.

**What they'll see:** Shuffled 5-fold: IC ≈ 0.08-0.12 (inflated by temporal leakage). Expanding-window: IC ≈ 0.03-0.05 (better, but still leaks through label overlap). Expanding-window with purge: IC ≈ 0.02-0.04 (honest). The gap between (a) and (c) is typically 3-5x. The quintile spread follows the same pattern — it looks great with shuffled CV and merely decent with purged CV.

**The insight:** "The gap between your first number and your last number is the amount of self-deception in your evaluation. Every impressive backtest you've ever seen that used shuffled CV in finance is lying — not maliciously, but structurally. This exercise is a permanent calibration: whenever someone shows you a financial ML result, your first question should be 'what CV method did you use?'"

### Exercise 4: IC Analysis by Feature
**The question we're answering:** Which individual features predict returns, and how strong are they?

**Setup narrative:** "Before we build complex models, let's understand each feature's raw predictive power. Sometimes the best feature wins — and knowing which features work individually helps you understand what the model is doing."

**What they build:** For each feature individually, compute the rolling IC with next-month returns. Rank features by average IC. Then compute the IC of the Ridge model's prediction vs. each feature's IC. Is the model doing better than the best single feature?

**What they'll see:** Momentum (12-month, skip 1) will likely be the strongest single feature, with IC around 0.04-0.06. Reversal (1-month) will also be significant. Volatility features will show a negative IC (high-vol stocks underperform). The Ridge model's IC should be modestly higher than any single feature's IC — but not by much. The model is mostly a weighted average of a few strong features.

**The insight:** "In linear models, the model's prediction is approximately a weighted average of the features. If no feature has IC > 0.03, the model won't have IC > 0.05. The ceiling is low. Trees (Week 5) will raise this ceiling by finding nonlinear interactions."

## Homework: "Cross-Sectional Alpha Model v1 (Linear)"

### Mission Framing

This is the first real alpha model you'll build in the course, and it follows exactly the workflow used at every quantitative asset management firm: engineer features, train a model, evaluate with expanding-window CV, construct a portfolio, and report risk-adjusted performance. The methodology is identical whether you're managing $100 or $100 billion — the difference is scale, not approach.

Your mission is to build the best linear model you can for predicting cross-sectional stock returns. "Best" means highest out-of-sample IC and Sharpe ratio, net of transaction costs. You'll compare OLS, Ridge, Lasso, and Elastic Net, and you'll discover that the regularized models win decisively — not because they're fancier, but because they refuse to fit noise.

In Week 5, you'll extend this pipeline with XGBoost and LightGBM. In Week 7, with neural networks. Each week builds on the same codebase. So build it clean, build it modular, and build it right — because you'll be living with this code for a long time.

### Deliverables

1. **Build a feature matrix with at least 20 features for 200+ US stocks, monthly frequency, 2005-2024.** Features should include: momentum variants (1m, 3m, 6m, 12m-skip-1), reversal (prior month return), volatility (20-day, 60-day), volume (turnover, Amihud illiquidity), size (log market cap), and at least 5 additional features of your choice (moving average ratios, RSI, price-to-52-week-high, etc.). Handle missing data with a documented policy (cross-sectional median imputation is the standard approach). Cross-sectionally rank-normalize all features to [0, 1] — this removes outliers and makes features comparable.

2. **Implement expanding-window CV.** Train on all data up to month t, predict month t+1. The minimum training period should be at least 60 months (5 years). Report IC and R-squared for each out-of-sample month.

3. **Train OLS, Ridge, Lasso, and Elastic Net.** For regularized models, tune the regularization parameter using a validation set within the expanding window (e.g., use the most recent 12 months of the training window for validation). Report the best hyperparameter for each model and how it changes over time.

4. **Construct a long-short portfolio: long top decile predictions, short bottom decile.** Compute monthly returns. Report annualized return, Sharpe ratio, Sortino ratio, maximum drawdown, and annualized turnover.

5. **Include 10 bps round-trip transaction costs.** Compute net returns for the long-short portfolio. How much do costs degrade the Sharpe? Which model produces a portfolio with the lowest turnover?

6. **Generate an `alphalens` tear sheet** for your best model's predictions. The tear sheet should show: IC over time, mean IC, quintile returns, and turnover.

### What They'll Discover

- OLS will have the highest in-sample R-squared but the worst out-of-sample IC — classic overfitting. The gap between in-sample and out-of-sample metrics is dramatic and serves as a permanent warning about the importance of regularization.

- Ridge will typically be the best linear model, with IC around 0.02-0.04. Lasso will be competitive but slightly lower, because it zeros out features that carry small but genuine signal. The difference between Ridge and Lasso is small — within sampling error in most datasets.

- The 12-month momentum (skip-1) feature will likely have the highest individual IC, confirming decades of academic research. But the model's IC will be higher than any single feature's IC, confirming that combining features adds value even in a linear framework.

- Transaction costs will reduce the Sharpe ratio by 0.2-0.5 points, depending on turnover. Lasso's sparser predictions may produce lower turnover than Ridge, partially offsetting its lower IC. The net-of-cost comparison might favor Lasso even if Ridge has higher gross IC.

- The IC will vary over time — it will be higher during some market regimes (trending markets, where momentum works) and lower or negative during others (market crashes, regime changes). This is not a bug; it's the reality of financial prediction.

### Deliverable
A complete Jupyter notebook containing: the feature matrix construction, expanding-window CV implementation, model comparison table, long-short portfolio performance, `alphalens` tear sheet, and a discussion of results. The code should be modular — the feature engineering and CV code will be reused in Weeks 5, 6, and 7.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Cross-sectional prediction framing | Demo: show panel data shape, explain cross-section vs. time-series | Not covered (done in lecture) | Integrate: build full cross-sectional pipeline for 200+ stocks |
| Feature engineering (momentum, vol, value, size) | Demo: compute key features, show code patterns | Exercise 1: build complete 20+ feature matrix for 200+ stocks | At scale: 20+ features for 200+ stocks, handle missing data, rank-normalize |
| OLS / Ridge / Lasso / Elastic Net | Demo: train all four, compare coefficients and in-sample vs. OOS | Exercise 2: trace regularization path, study lambda stability over time | At scale: tune all four models with expanding-window CV, report IC/Sharpe |
| Expanding-window CV | Demo: implement from scratch, show rolling IC | Exercise 3: compare 5-fold vs. expanding-window vs. purged CV on same model | At scale: full expanding-window evaluation for all models |
| Gu-Kelly-Xiu framework | Demo: contextualize results in GKX's model comparison | Not covered (done in lecture) | Reference: compare your linear results to GKX benchmarks |
| Quantile portfolio analysis (alphalens) | Demo: quintile sorts, long-short, alphalens tear sheet | Not covered (done in homework) | Build: full alphalens tear sheet for best model, quintile spread chart |
| Per-feature IC analysis | Not covered (deferred to seminar) | Exercise 4: compute rolling IC per feature, rank features by predictive power | Integrate: use feature-level IC to interpret model behavior |
| Transaction cost integration | Demo: reference Week 3 cost model | Not covered (done in homework) | At scale: net-of-cost Sharpe for long-short portfolio at 10 bps |

## Key Stories & Facts to Weave In

1. **Gu, Kelly & Xiu (2020) — the landmark paper.** "Empirical Asset Pricing via Machine Learning" tested 8 model classes on 30,000 stocks over 60 years using 94 firm characteristics. It was published in the *Review of Financial Studies*, the top journal in the field. The paper's most cited finding: neural networks achieve the highest out-of-sample R-squared (~0.4% monthly), but even OLS is positive (~0.08%). The paper also showed that the momentum × volatility interaction is the single most important nonlinear effect — trees and neural nets capture it, linear models can't.

2. **Jegadeesh & Titman (1993) — the momentum discovery.** Narasimhan Jegadeesh and Sheridan Titman showed that a strategy of buying past winners and shorting past losers earns about 1% per month. The paper has over 12,000 citations. The momentum anomaly has been replicated in 40+ countries, across asset classes, and over centuries of data. It remains the most robust stock market anomaly — and your model will rediscover it.

3. **The death of the value premium.** The Fama-French HML factor earned approximately 5% per year from 1927-2007. From 2007-2024, it has been approximately flat — value stocks have not outperformed growth stocks. Some researchers (including Fama and French themselves) attribute this to increased market efficiency. Others argue it's a cyclical phenomenon that will reverse. Your feature analysis will show this: book-to-market will have a much lower IC in the 2010-2024 period than in earlier decades.

4. **AQR Capital Management.** Cliff Asness founded AQR (Applied Quantitative Research) in 1998 with $1 billion in assets. By 2024, the firm manages over $100 billion, primarily through systematic strategies based on the same factors you're implementing today — momentum, value, quality, and low volatility. AQR's research team publishes academic papers about their own strategies, making them the most transparent large quant fund. Their paper "Value and Momentum Everywhere" (2013) showed these factors work across equities, bonds, currencies, and commodities.

5. **The information coefficient in practice.** At most quant funds, an IC of 0.05 is considered excellent. An IC of 0.10 would be world-class — and suspicious (check for look-ahead bias). Most published factor ICs are between 0.02 and 0.06. To put this in perspective: in image classification, a rank correlation of 0.05 between predictions and labels would be useless. In finance, it's a career. The Fundamental Law (IR = IC × sqrt(BR)) explains why: breadth amplifies tiny signal.

6. **Kenneth French's data library.** Kenneth French (Dartmouth Tuck School of Business) maintains a free online data library with factor returns going back to 1926. This is the most-used free dataset in academic finance. Every asset pricing paper references it. The Fama-French factors (market, size, value, momentum, profitability, investment) are computed from CRSP data using a methodology that's been standardized for decades. It's available at mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html — bookmark it, because you'll use it frequently.

7. **Cross-sectional rank normalization — the trick that everyone uses.** Raw features have wildly different scales: market cap ranges from $1 billion to $3 trillion, while momentum ranges from -50% to +100%. More importantly, raw features have extreme outliers (a stock that rose 300% in a month will dominate the regression). The standard solution in quant finance: rank-normalize each feature cross-sectionally at each time step, mapping to [0, 1]. This removes outliers, makes features comparable, and makes the model robust to distributional shifts. It's used at virtually every quant fund and is so standard that papers don't even mention it.

8. **The "factor zoo" problem.** As of 2024, academic researchers have published over 400 "factors" that supposedly predict stock returns: momentum, value, accruals, asset growth, betting against beta, earnings surprises, insider trading, and hundreds more. Harvey, Liu & Zhu (2016) showed that the standard statistical threshold (t-stat > 2) is too low when you account for the multiple testing problem — with 400 factors tested, you'd expect 20 to be "significant" by chance. The true threshold should be t-stat > 3 or even higher. This is the "factor zoo" problem, and it connects directly to Week 6's discussion of backtest overfitting and Week 18's deflated Sharpe ratio.

## Cross-References
- **Builds on:** The `DataLoader` class and returns computation from Week 1. The fractional differentiation and stationarity concepts from Week 2 (fractionally differentiated features can be added to the feature matrix). The Information Coefficient, Sharpe ratio, and Fundamental Law from Week 3. Transaction cost modeling from Week 3.
- **Sets up:** The feature matrix and expanding-window CV framework built this week are reused directly in Week 5 (trees), Week 6 (financial ML methodology), and Week 7 (neural nets). The `alphalens` analysis is the standard signal evaluation used throughout. This week's Ridge/Lasso results become the linear baseline that all subsequent models are compared against.
- **Recurring thread:** The "signal-to-noise ratio" thread intensifies here. IC of 0.03-0.05, R-squared of 0.1-0.4%, signal buried in noise — this is the reality of financial prediction, and calibrating expectations is as important as building the model. Also: the "simple vs. sophisticated" thread — regularized linear models provide a shockingly hard-to-beat baseline.

## Suggested Reading
- **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning"** — The paper that defined the field. Dense but essential. Focus on Tables 3-5 (model comparison) and Figure 3 (feature importance / interactions). This is the benchmark everyone cites.
- **Stefan Jansen, *Machine Learning for Algorithmic Trading*, Chapters 4-7** — The most practical treatment of alpha factor research with Python code. Chapters 4-5 cover feature engineering, Chapters 6-7 cover linear and tree models. Working code for everything.
- **Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns"** — The "factor zoo" paper that showed most published factors are probably false positives. Essential context for why we need rigorous methodology (Week 6) and honest evaluation (Week 18).
