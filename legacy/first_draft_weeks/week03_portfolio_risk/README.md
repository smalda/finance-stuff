# Week 3 — Portfolio Theory, Factor Models & Risk

> **Alpha is what's left after you account for all the risks you took. Most people discover they have none.**

## Prerequisites
Week 1 (data structures, returns math, transaction cost awareness) and Week 2 (stationarity, GARCH, the concept of volatility clustering). Specifically: students must understand log returns, annualization ($\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$), and the difference between stationary and non-stationary series. The `DataLoader` class from Week 1 is used for data access.

## The Big Idea

Every ML model you'll build in this course produces a prediction. "Stock A will go up 2% next month." "Stock B will outperform Stock C." But a prediction is not a strategy. The distance between "I predicted the return correctly" and "I made money" is filled with a set of ideas that finance has been developing for 70 years: portfolio theory, factor models, and risk management.

Here's the question that motivates this entire week: suppose your ML model has an information coefficient (IC) of 0.05 — its predictions have a 0.05 rank correlation with actual returns. Is that useful? It sounds pathetic. In image classification, a correlation of 0.05 between your predictions and the labels would get you fired. But in finance, with 500 stocks and monthly rebalancing, an IC of 0.05 translates to an information ratio (IR) of about 1.1 — which is better than 90% of hedge funds. The equation that tells you this is the Fundamental Law of Active Management: IR = IC × sqrt(BR), where BR is the number of independent bets you make per year. A tiny edge, applied across many stocks, many times, compounds into a serious business. Breadth — the number of independent bets — matters more than the accuracy of any single prediction.

But there's a trap. Harry Markowitz won the Nobel Prize in 1952 for showing that you can find the "optimal" portfolio — the one with the highest return for a given level of risk. Beautiful theory. In practice, Markowitz's optimizer produces portfolios that are maximally wrong, because the optimization is incredibly sensitive to estimation errors in the covariance matrix. A small error in how you estimate the correlation between two stocks ripples through the optimizer and produces a portfolio that bets 40% of your wealth on a single stock. DeMiguel, Garlappi & Uppal (2009) showed that a naive equal-weight portfolio (1/N) beats mean-variance optimization in most realistic settings. That paper has over 5,000 citations. It's basically saying: the most sophisticated portfolio optimization technique from 70 years of financial theory loses to the simplest possible approach. We'll understand why, and we'll learn about better alternatives — including Hierarchical Risk Parity (HRP), which Lopez de Prado developed as a tree-based solution that avoids the covariance estimation problem entirely.

This week is also the authoritative introduction to risk metrics. Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, VaR, CVaR — these are the KPIs of quantitative finance. Every model you build for the rest of the course will be evaluated by these numbers. Learn them now, deeply, so that when you see a Sharpe ratio of 2.5 in a backtest, your first reaction is suspicion rather than celebration.

## Lecture Arc

### Opening Hook

"In 1998, Long-Term Capital Management had $125 billion in assets, two Nobel laureates on staff, and a portfolio optimizer that said their positions were perfectly hedged. They lost $4.6 billion in four months and nearly brought down the global financial system. In 2009, three professors published a paper showing that a strategy as dumb as 'put equal money in everything' beats the Nobel-Prize-winning portfolio optimization technique in most real-world tests. Both of these facts are true, and together they tell you something important about the gap between theory and practice in portfolio construction. This week, we're going to learn the theory — Markowitz, CAPM, factor models — understand why it's beautiful, understand why it fails, and then learn what actually works."

### Section 1: Mean-Variance Optimization — The Beautiful Theory
**Narrative arc:** We introduce Markowitz's framework as one of the great intellectual achievements in finance, then show why it fails in practice. The setup: if you know expected returns and the covariance matrix, the optimal portfolio is a solved problem. The tension: you don't know either of those things — you estimate them, badly. The resolution (deferred to Section 6): better estimation methods and alternative approaches.

**Key concepts:**
- Portfolio return: weighted average of individual returns
- Portfolio risk: not the weighted average of individual risks (diversification!)
- The efficient frontier: the set of portfolios with the highest return for each risk level
- The tangency portfolio: the portfolio with the highest Sharpe ratio
- Why it fails: estimation error in expected returns and covariance matrix

**The hook:** "Diversification is the only free lunch in finance. If you hold two stocks that aren't perfectly correlated, the portfolio's risk is less than the weighted average of the individual risks. That's not a model assumption — it's a mathematical fact. Harry Markowitz proved it in 1952 and won the Nobel Prize. The tragedy is that the optimizer he invented to exploit this fact is so sensitive to estimation errors that it usually produces portfolios worse than equal-weighting."

**Key formulas:**
"For a portfolio with weights $\mathbf{w}$ and asset returns $\mathbf{r}$:

Portfolio return:
$$R_p = \mathbf{w}^T \boldsymbol{\mu}$$

Portfolio variance:
$$\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$$

The mean-variance optimization problem:
$$\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{s.t. } \sum_i w_i = 1$$

This is just maximizing the Sharpe ratio. The solution is analytical:

$$\mathbf{w}^* = \frac{\boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu} - r_f \mathbf{1})}{\mathbf{1}^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu} - r_f \mathbf{1})}$$

Beautiful. And completely useless in practice, because $\boldsymbol{\Sigma}^{-1}$ amplifies every estimation error in $\boldsymbol{\Sigma}$. For a portfolio of 500 stocks, the covariance matrix has 125,250 entries to estimate from maybe 1,260 daily observations (5 years). You have fewer observations than parameters. The matrix is nearly singular. The optimizer goes insane."

**Code moment:** Use `PyPortfolioOpt` to build the efficient frontier for 20 stocks. Plot it. Then show the tangency portfolio weights — they'll be extreme (some stocks at 30-40%, many at 0%). Then add a small perturbation to the expected returns (e.g., add random noise with std = 0.1 × original std) and re-optimize. The new weights will be completely different. This demonstrates the instability of mean-variance optimization.

**"So what?":** "If your ML model produces expected return predictions, and you feed those into a Markowitz optimizer, the optimizer will take your small prediction errors and amplify them into enormous position errors. The portfolio won't reflect your model's views — it'll reflect your model's noise. We need a better way to go from predictions to positions."

### Section 2: CAPM and Factor Models — What "Alpha" Actually Means
**Narrative arc:** We introduce the Capital Asset Pricing Model as the simplest framework for decomposing a stock's return into "market risk" and "everything else." The tension: CAPM says the market is the only risk factor, but empirically, small stocks and value stocks earn more than CAPM predicts. The resolution: Fama-French factors, which capture the missing risk premia.

**Key concepts:**
- CAPM: expected return = risk-free rate + beta × (market return - risk-free rate)
- Beta: sensitivity to the market. Beta = 1 means the stock moves with the market
- Alpha: the return that remains after accounting for market exposure
- Why CAPM fails: small stocks and value stocks have positive alpha under CAPM
- Fama-French 3-factor model: market + size (SMB) + value (HML)
- Fama-French 5-factor model: adds profitability (RMW) and investment (CMA)
- Momentum factor (MOM / UMD): winners keep winning, losers keep losing — for a while

**The hook:** "When a hedge fund manager tells you they earned 15% last year, the first question isn't 'Is that good?' It's 'How much risk did you take to get it?' If the market was up 20% and you earned 15% with a beta of 1.5, your alpha was actually 15% - 1.5 × 20% = -15%. You underperformed massively. You just took a lot of market risk and the market happened to go up. Alpha is what's left after you subtract the return you would have earned just by being exposed to known risk factors. Most managers, after this subtraction, have no alpha at all."

**Key formulas:**
"CAPM says:
$$E[R_i] - R_f = \beta_i (E[R_m] - R_f)$$

But empirically, small-cap stocks earn about 2-3% per year more than CAPM predicts, and value stocks earn about 3-5% per year more. Fama and French captured this with:

$$R_i - R_f = \alpha_i + \beta_i^{MKT}(R_m - R_f) + \beta_i^{SMB} \cdot SMB + \beta_i^{HML} \cdot HML + \epsilon_i$$

$\alpha_i$ is now the return after controlling for market, size, and value exposures. If $\alpha_i > 0$ and is statistically significant, the stock (or strategy) is genuinely adding value.

The 5-factor model adds profitability (RMW: robust minus weak) and investment (CMA: conservative minus aggressive):

$$R_i - R_f = \alpha_i + \beta_i^{MKT}(R_m - R_f) + \beta_i^{SMB} \cdot SMB + \beta_i^{HML} \cdot HML + \beta_i^{RMW} \cdot RMW + \beta_i^{CMA} \cdot CMA + \epsilon_i$$

The more factors you control for, the harder it is to have alpha. This is by design — the whole point is to distinguish genuine skill from known risk premia."

**Code moment:** Download Fama-French factors from the Kenneth French Data Library (pandas-datareader makes this one line). Run a time-series regression for a few stocks. Show that a stock with a high beta (say, NVDA with beta ≈ 1.8) has a huge raw return but may have negative alpha after controlling for market exposure. Show that a boring utility (say, DUK with beta ≈ 0.5) has a low raw return but may have positive alpha.

**"So what?":** "When we build ML models to predict stock returns, we're implicitly searching for alpha — the return component that known risk factors can't explain. If your model just learns to buy high-beta stocks, it hasn't learned anything useful — it's just taking market risk. The Fama-French regression tells you whether your model found genuine alpha or just rediscovered beta."

### Section 3: Risk Metrics — The KPIs of Quantitative Finance
**Narrative arc:** This is the reference section for the entire course. Every metric defined here will be used in every evaluation going forward. The tension: there are many ways to measure performance, and they can tell contradictory stories. The resolution: each metric captures a different aspect of risk/return, and you need several to get the full picture.

**Key concepts:**
- Sharpe ratio: risk-adjusted return (the primary metric)
- Sortino ratio: like Sharpe but only penalizes downside risk
- Maximum drawdown: worst peak-to-trough loss
- Calmar ratio: annual return / max drawdown
- Value at Risk (VaR): the loss threshold at a given confidence level
- Conditional VaR (CVaR / Expected Shortfall): average loss beyond VaR

**The hook:** "A strategy with a Sharpe ratio of 0.5 is mediocre. A strategy with a Sharpe ratio of 1.0 is good. A strategy with a Sharpe ratio of 2.0 is excellent. A strategy with a Sharpe ratio of 3.0 is almost certainly wrong — either the backtest is overfitted, the costs are wrong, or the data has survivorship bias. Jim Simons' Medallion Fund, the best-performing hedge fund in history, has a Sharpe ratio of about 2.5 after fees. If your homework Sharpe is higher than Simons', you have a bug, not a breakthrough."

**Key formulas:**
"The Sharpe ratio:

$$SR = \frac{E[R_p - R_f]}{\sigma(R_p - R_f)} = \frac{\mu_{\text{excess}}}{\sigma_{\text{excess}}}$$

For daily data annualized: $SR_{\text{annual}} = SR_{\text{daily}} \times \sqrt{252}$

The Sortino ratio replaces the denominator with downside deviation:

$$\text{Sortino} = \frac{\mu_{\text{excess}}}{\sigma_{\text{downside}}}, \quad \sigma_{\text{downside}} = \sqrt{E[\min(R_p - R_f, 0)^2]}$$

Maximum drawdown:

$$\text{MDD} = \max_{t \in [0,T]} \left( \max_{s \in [0,t]} V_s - V_t \right) / \max_{s \in [0,t]} V_s$$

Value at Risk at confidence level $\alpha$ (typically 95% or 99%):

$$\Pr(R_p < -\text{VaR}_\alpha) = 1 - \alpha$$

'There's a 5% chance you'll lose more than VaR(95%) in a given day.'

Conditional VaR (Expected Shortfall) — the average loss in the worst (1-$\alpha$) of cases:

$$\text{CVaR}_\alpha = E[-R_p \mid R_p < -\text{VaR}_\alpha]$$

CVaR is a better risk measure than VaR because it tells you the average size of the catastrophe, not just the threshold. VaR says 'there's a 5% chance of losing more than X.' CVaR says 'when you do lose more than X, you lose Y on average.' Y is what actually determines whether you survive."

**Code moment:** Compute all metrics for SPY over the full sample. Then split into periods: 2017-2019 (calm), March 2020 (COVID crash), 2021-2022 (recovery then bear). Show how the metrics change dramatically across regimes. The Sharpe is great in 2017-2019, terrible in 2020, and mediocre overall. Max drawdown happens in March 2020 and dominates the entire sample's risk profile. CVaR during COVID is 3-4x the calm-period CVaR.

Use `QuantStats` to generate an automatic tear sheet:

```python
import quantstats as qs
qs.reports.html(returns, "SPY", output="spy_report.html")
```

**"So what?":** "When you report a Sharpe ratio for your ML strategy, also report the max drawdown and the Calmar ratio. A strategy with Sharpe 1.5 and max drawdown of 40% is very different from Sharpe 1.5 with max drawdown of 10%. The first one will get you fired during the drawdown, regardless of its long-run performance."

### Section 4: The Fundamental Law of Active Management
**Narrative arc:** This is the equation that connects ML model quality to portfolio performance. The setup: you have an ML model with a certain predictive accuracy. The tension: how do you know if it's good enough to make money? The resolution: Grinold's law tells you exactly — and the answer is surprising.

**Key concepts:**
- Information Coefficient (IC): rank correlation between predictions and realized returns
- Breadth (BR): number of independent bets per year
- Information Ratio (IR): the portfolio-level Sharpe ratio of active returns
- The Fundamental Law: IR = IC × sqrt(BR)
- Why breadth matters more than accuracy

**The hook:** "Here's the single most important equation in quantitative finance for an ML engineer:

$$IR = IC \times \sqrt{BR}$$

Your information coefficient (IC) is 0.05 — your model's rank correlation with actual returns. Pathetic, right? But if you trade 500 stocks monthly, your breadth is BR = 500 × 12 = 6,000. So your information ratio is 0.05 × sqrt(6,000) ≈ 3.9. That's world-class. Jim Simons territory. A terrible-sounding prediction accuracy, applied across many stocks many times, compounds into extraordinary performance. This is why cross-sectional models (predicting all stocks simultaneously) dominate quant finance — they have enormous breadth."

**Key formulas:**
"The Fundamental Law:

$$IR = IC \times \sqrt{BR}$$

where:
- $IC = \text{Corr}(\hat{r}_i, r_i)$ — rank correlation between predicted and realized returns
- $BR$ — number of independent bets per year (stocks × rebalancing frequency)
- $IR$ — information ratio, the Sharpe ratio of your active returns (returns above the benchmark)

In practice, the 'independent' part of BR is tricky. If you hold 500 stocks but they're all tech stocks that move together, your effective BR is much less than 500. Correlation reduces effective breadth."

**Code moment:** Build a simulation showing the Fundamental Law in action. Generate synthetic signals with IC = 0.05 for 500 stocks, monthly. Construct a long-short portfolio (long top quintile, short bottom quintile). Show that the portfolio Sharpe is approximately IC × sqrt(500 × 12) ≈ 3.9. Then reduce to 50 stocks — the Sharpe drops to about 1.2. Then reduce IC to 0.02 — the Sharpe drops to about 1.5 with 500 stocks. The message: breadth is incredibly powerful.

**"So what?":** "This is why we'll build cross-sectional models (predicting many stocks at once) rather than time-series models (predicting one stock at a time) for most of the course. Cross-sectional models have massive breadth. A model with IC = 0.03 across 500 stocks and monthly rebalancing has an IR of about 2.3. The same IC for a single stock gives IR = 0.10. Breadth is the multiplier."

### Section 5: Covariance Estimation — Where Everything Goes Wrong
**Narrative arc:** We return to the Markowitz problem and diagnose exactly why it fails. The setup: mean-variance optimization needs the covariance matrix. The tension: estimating a 500×500 matrix from finite data is a fundamentally ill-posed problem. The resolution: shrinkage estimators and random matrix theory.

**Key concepts:**
- Sample covariance matrix and its problems: noise, near-singularity
- Ledoit-Wolf shrinkage: blend the sample covariance with a structured target
- Random matrix theory: the Marchenko-Pastur distribution
- Which eigenvalues are signal and which are noise
- How this connects to PCA (the ML engineer's familiar territory)

**The hook:** "If you have 500 stocks and 1,260 trading days (5 years), your covariance matrix has 125,250 unique entries. You're estimating 125,250 parameters from 1,260 observations. That's not statistics — that's hallucination. The smallest eigenvalues of the sample covariance matrix are pure noise. And the Markowitz optimizer puts the most weight on exactly those smallest eigenvalues, because they correspond to 'diversification opportunities' that don't actually exist. The optimizer is fitting noise and calling it alpha."

**Key formulas:**
"Ledoit-Wolf shrinkage blends the sample covariance $\mathbf{S}$ with a structured target $\mathbf{F}$ (typically scaled identity or constant-correlation):

$$\hat{\boldsymbol{\Sigma}} = \delta \mathbf{F} + (1 - \delta) \mathbf{S}$$

where $\delta$ is the optimal shrinkage intensity, estimated from the data. The target $\mathbf{F}$ provides structure (it's well-conditioned by construction); the sample $\mathbf{S}$ provides information. The blend is better than either alone.

From random matrix theory, the Marchenko-Pastur distribution tells us what eigenvalues to expect from a purely random covariance matrix:

$$f(\lambda) = \frac{T}{2\pi\sigma^2 N} \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{\lambda}$$

where $\lambda_{\pm} = \sigma^2(1 \pm \sqrt{N/T})^2$. Eigenvalues inside this range are indistinguishable from noise. Eigenvalues outside this range carry signal. For 500 stocks and 5 years of daily data, typically only the top 5-10 eigenvalues contain genuine signal. The other 490+ are noise."

**Code moment:** Compute the sample covariance matrix for 100 stocks. Plot the eigenvalue distribution and overlay the Marchenko-Pastur theoretical distribution. Students will see that most eigenvalues cluster inside the MP bounds — they're noise. Only the top few (corresponding to market, sector, and size factors) stand out. Then compare portfolio weights from: (a) sample covariance, (b) Ledoit-Wolf shrinkage, (c) after removing noise eigenvalues. The Ledoit-Wolf weights will be much more reasonable — fewer extreme positions, better diversification.

**"So what?":** "Random matrix theory tells you that for a 500-stock portfolio, your covariance estimate is mostly noise. The Markowitz optimizer treats that noise as signal and bets on it. Ledoit-Wolf shrinkage is the minimum fix — always use it. But even better approaches exist, which brings us to the next section."

### Section 6: Hierarchical Risk Parity — The Tree-Based Solution
**Narrative arc:** This is the Lopez de Prado innovation — a portfolio allocation method that sidesteps the covariance estimation problem entirely by using hierarchical clustering. The setup: Markowitz requires inverting the covariance matrix (amplifying noise). The tension: can we build a portfolio without inverting anything? The resolution: HRP uses the covariance matrix only for clustering (distances), never for inversion.

**Key concepts:**
- The key insight: covariance matrix inversion is the source of Markowitz's instability
- Hierarchical clustering of assets (Ward's method or single linkage)
- The dendrogram: visualizing which assets cluster together
- Recursive bisection: allocating risk top-down through the hierarchy
- Why HRP is more stable than mean-variance out of sample

**The hook:** "Lopez de Prado developed HRP while managing over $13 billion at Guggenheim Partners. His insight was that portfolio construction should respect the hierarchical structure of markets — tech stocks cluster with tech stocks, utilities cluster with utilities — and allocate risk within and between clusters. The result is a method that never inverts the covariance matrix, never produces extreme weights, and consistently outperforms Markowitz out of sample."

**Key formulas:**
"HRP works in three steps:

1. **Tree clustering:** compute the distance matrix from correlations:
$$d_{ij} = \sqrt{\frac{1}{2}(1 - \rho_{ij})}$$
Then apply hierarchical clustering (Ward's method).

2. **Quasi-diagonalization:** reorder the covariance matrix so that correlated assets are adjacent. This is the dendrogram's leaf order.

3. **Recursive bisection:** split the assets at the top of the hierarchy. Allocate risk inversely proportional to each cluster's variance:
$$w_1 = \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2}, \quad w_2 = 1 - w_1$$
Then recurse into each cluster.

The key: we never invert $\boldsymbol{\Sigma}$. We only use it to compute distances (step 1) and cluster-level variances (step 3). These operations are much more robust to estimation error than matrix inversion."

**Code moment:** Build HRP from scratch using `scipy.cluster.hierarchy` and `PyPortfolioOpt`. Show the dendrogram for 50 stocks — students will see that banks cluster together, tech stocks cluster together, and the hierarchy mirrors the real economic structure. Then compare portfolio weights: Markowitz (wild, concentrated), HRP (reasonable, diversified). Then compare out-of-sample Sharpe ratios over rolling windows.

**"So what?":** "HRP is the default portfolio construction method for this course. When you need to go from ML predictions to portfolio weights, HRP gives you stable, reasonable allocations without the instability of Markowitz. In the homework, you'll compare all three approaches (equal-weight, Markowitz with Ledoit-Wolf, HRP) and see the out-of-sample results for yourself."

### Section 7: Transaction Costs — The Silent Killer
**Narrative arc:** A short but essential section that introduces the simplest transaction cost model. This connects back to Week 1's intuition about bid-ask spreads and makes it quantitative.

**Key concepts:**
- The constant-bps model: assume a fixed cost per dollar traded
- Round-trip costs: buy + sell
- Turnover: the fraction of the portfolio that changes each period
- Net Sharpe = Gross Sharpe - (turnover × cost × sqrt(252) / sigma_portfolio)
- Why high-turnover strategies die first

**The hook:** "A strategy that turns over 100% per month at 10 bps round-trip loses 12% per year to costs. If your gross return is 10%, your net return is -2%. You're paying market makers for the privilege of losing money."

**Key formulas:**
"Total transaction cost over a period:

$$\text{TC}_t = c \times \sum_{i=1}^{N} |w_{i,t} - w_{i,t-1}| \times V_t$$

where $c$ is the cost per dollar traded (e.g., 5 bps = 0.0005 per side), and the sum is the turnover. For a simple model:

$$\text{Annual TC} = c \times \text{annual turnover rate}$$

If you rebalance monthly with 30% turnover per rebalance: annual turnover = 360%, annual cost at 10 bps round-trip = 3.6%. That's a lot — it's roughly equal to the equity risk premium."

**Code moment:** Take a simple momentum strategy and compute returns with and without costs at 5, 10, and 20 bps per side. Plot cumulative returns for each cost level. Show that the strategy that looks great at 0 bps becomes mediocre at 10 bps and unprofitable at 20 bps. This is the reality check that every backtest needs.

**"So what?":** "From now on, every homework includes transaction costs. 10 bps round-trip is our standard assumption for liquid large-cap US equities. We'll revisit this in Week 18 with more realistic market impact models."

### Closing Bridge

"You now have the complete toolkit for evaluating any strategy or ML model in finance. You know what alpha means (the return that known risk factors can't explain), why the Sharpe ratio is the metric (risk-adjusted return, annualized, comparable across strategies), and why most 'optimal' portfolios blow up (covariance estimation error amplified by matrix inversion). You've seen that a tiny prediction accuracy (IC = 0.05), applied across many stocks, can generate world-class performance (IR ≈ 4) — the Fundamental Law of Active Management tells you exactly when your model is good enough.

Next week, we put all of this to work. We'll build our first cross-sectional return prediction model — the bread and butter of quantitative asset management. You'll engineer features (momentum, value, volatility, size), train Ridge and Lasso regressions, and evaluate them with the IC we introduced today and expanding-window CV (formalized next week). For the first time in the course, you'll see a number come out of a model and know exactly what it means — and whether it's good enough to trade."

## Seminar Exercises

### Exercise 1: Fama-French Factor Regressions
**The question we're answering:** Do individual stocks have alpha, or are their returns just compensation for known risk factors?

**Setup narrative:** "We're going to download the Fama-French factors and run regressions for 20 stocks. Most of them will have no statistically significant alpha. The few that do will surprise you — they're not the stocks you'd expect."

**What they build:** Download Fama-French 5 factors + momentum from the Kenneth French Data Library. Select 20 stocks spanning sectors (tech, financials, healthcare, utilities, energy). Run time-series regressions for each stock: $R_i - R_f = \alpha + \beta^{MKT}(R_m - R_f) + \beta^{SMB} \cdot SMB + \beta^{HML} \cdot HML + \beta^{RMW} \cdot RMW + \beta^{CMA} \cdot CMA + \epsilon$. Report alpha, t-statistic, R-squared, and factor loadings.

**What they'll see:** Most alphas will be small and statistically insignificant (t-stat < 2). A few stocks might show significant alpha — but often it's a stock with unusual characteristics (e.g., a defensive stock during a bull market, or a growth stock during the value premium's disappearance). The R-squared will typically be 0.3-0.7, meaning factors explain 30-70% of return variation. The rest is idiosyncratic.

**The insight:** "If the factor model explains 50% of a stock's return, your ML model is fighting over the remaining 50% — and much of that is pure noise. The IC of 0.05 that seems small is 0.05 of the signal that's left after factors take their share."

### Exercise 2: Covariance Estimation — How Many Eigenvalues Are Signal?
**The question we're answering:** Can random matrix theory tell us how much of our covariance estimate is real structure versus pure noise?

**Setup narrative:** "The lecture showed that the Markowitz optimizer amplifies estimation noise. Now we're going to look directly at the noise itself. You'll compute the covariance matrix for 100 stocks, decompose it into eigenvalues, and use the Marchenko-Pastur distribution to draw the line between signal and noise. The number of signal eigenvalues will be shockingly small."

**What they build:** Download 5 years of daily returns for 100 S&P 500 stocks. Compute the sample covariance matrix and its eigenvalues. Overlay the Marchenko-Pastur theoretical distribution (compute $\lambda_+$ and $\lambda_-$ for the ratio N/T = 100/1260). Count how many eigenvalues fall outside the MP bounds. Then run two experiments: (a) build portfolios using all eigenvalues vs. only the signal eigenvalues (by reconstructing the covariance from the top-k eigenvectors) — compare out-of-sample portfolio variance; (b) vary the number of stocks (50, 100, 200) with the same time window and show how the N/T ratio changes the proportion of noise eigenvalues.

**What they'll see:** For 100 stocks and 5 years of daily data, typically only 5-10 eigenvalues will exceed the MP upper bound — these correspond to market, sector, and style factors. The other 90-95 are noise. The denoised covariance (using only signal eigenvalues) will produce portfolios with lower realized variance out-of-sample. As N/T increases (more stocks, same history), the noise problem gets worse — the MP bounds widen and more eigenvalues are swallowed by noise.

**The insight:** "Your 100×100 covariance matrix has 5,050 unique entries, but only 5-10 dimensions of real structure. The Markowitz optimizer treats all 5,050 numbers as truth and bets on the noise. Denoising via random matrix theory or shrinkage is not optional — it's the difference between a portfolio built on signal and one built on statistical hallucination."

### Exercise 3: Hierarchical Risk Parity vs. Markowitz vs. Equal-Weight
**The question we're answering:** Does the fancier approach actually win when it counts — out of sample?

**Setup narrative:** "DeMiguel et al. (2009) showed that 1/N beats Markowitz in most realistic settings. Lopez de Prado claims HRP beats both. Let's test all three on real data with rolling out-of-sample evaluation."

**What they build:** For 30-50 stocks over 10 years, implement a rolling out-of-sample test: estimate covariance matrix on the past 252 days, construct three portfolios (equal-weight, Markowitz max-Sharpe with Ledoit-Wolf, HRP), hold for 21 trading days (one month), then re-estimate and rebalance. Track returns, Sharpe, max drawdown, and turnover for all three.

**What they'll see:** HRP will typically have the best risk-adjusted returns out of sample, with lower drawdowns and lower turnover than Markowitz. Equal-weight will be surprisingly competitive. Markowitz will have the highest Sharpe in some periods but the worst drawdowns.

**The insight:** "The ranking is often: HRP > 1/N > Markowitz in terms of out-of-sample Sharpe and especially risk-adjusted metrics. The more stocks you add, the worse Markowitz gets (the covariance estimation problem gets harder). HRP scales better because it never inverts the covariance matrix."

## Homework: "Factor Model & Portfolio Construction"

### Mission Framing

You're about to build your first complete investment pipeline: from raw data to factor analysis to portfolio construction to performance evaluation. This is the workflow that every quantitative analyst at every quant fund executes daily. The difference between your version and theirs is scale (they have 3,000 stocks and 50+ factors), not methodology.

Your job is to answer a specific question: does sophisticated portfolio construction add value over naive approaches? The theory says yes — Markowitz proved it. The empirics are less kind. DeMiguel, Garlappi & Uppal (2009) showed that equal-weighting beats mean-variance optimization for most realistic parameter settings. Lopez de Prado's HRP claims to fix this. You're going to run the horse race yourself, with real data, real factor models, and real transaction costs, and report the honest results.

The twist: you'll also run Fama-MacBeth regressions to estimate factor risk premia. This is the cross-sectional regression methodology that Weeks 4-5 will build on — you're getting a preview of the engine that powers cross-sectional alpha research.

### Deliverables

1. **Download 5 years of daily returns for 100 US stocks + Fama-French 5 factors.** Use large-cap, liquid stocks (e.g., S&P 100 + 50 additional mid-caps). Download Fama-French factors from Kenneth French Data Library. Align dates. Handle missing data. Document your universe selection (and acknowledge the survivorship bias inherent in picking stocks that are currently listed).

2. **Run cross-sectional Fama-MacBeth regressions to estimate factor risk premia.** For each month: run a cross-sectional regression of that month's stock returns on their factor loadings (betas estimated from the prior 60 months). The average of the monthly coefficient estimates = the factor risk premium. Report: average risk premium, t-statistic, and R-squared for each factor. Which factors are priced? Is the value premium (HML) still significant in recent data?

3. **Construct 3 portfolios with monthly rebalancing:** (a) equal-weight (1/N), (b) mean-variance optimized with Ledoit-Wolf shrinkage (maximize Sharpe, long-only constraint), (c) Hierarchical Risk Parity. Use `PyPortfolioOpt` for (b) and (c).

4. **Evaluate all three portfolios with:** annualized return, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, annualized turnover. Report results in a clean comparison table.

5. **Show how transaction costs (assume 10 bps round-trip) degrade each strategy differently.** Compute net-of-cost returns for each portfolio. Which portfolio suffers most from costs? (Hint: it's the one with the highest turnover — almost certainly Markowitz.)

6. **Generate a full `QuantStats` tear sheet** for each portfolio, comparing all three against SPY as benchmark.

### What They'll Discover

- The Fama-MacBeth regression will show that the market risk premium (MKT) and momentum (MOM) are the most robust factors. The value premium (HML) has weakened significantly since 2010 — value investing has been underperforming for over a decade, and the data will show it. This is a live debate in academic finance: has the value premium disappeared, or is it just in a cyclical drawdown?

- Equal-weight (1/N) will be shockingly competitive. It will likely have a higher Sharpe ratio than Markowitz with sample covariance, and will be within 0.1-0.2 Sharpe points of the best optimizer. The DeMiguel result holds.

- HRP will typically produce the best Calmar ratio (return/max drawdown) — it avoids the concentrated positions that cause Markowitz to blow up during market stress. Its turnover will be lower than Markowitz, making it more robust to transaction costs.

- Transaction costs will hit the Markowitz portfolio hardest, because it has the highest turnover (the optimizer suggests large position changes each month as the covariance estimate shifts). After 10 bps round-trip costs, Markowitz may actually underperform equal-weight. The most "sophisticated" approach loses to the dumbest approach, after costs.

### Deliverable
A complete Jupyter notebook containing: factor regression analysis, all three portfolio constructions, performance comparison table, `QuantStats` tear sheets, and a 1-page discussion of results and limitations.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Mean-variance optimization & efficient frontier | Demo: build frontier for 20 stocks, show perturbation instability | Not covered (done in lecture) | At scale: construct Markowitz portfolio for 100 stocks with Ledoit-Wolf, monthly rebalance |
| CAPM & factor models (Fama-French) | Demo: FF regression for a few stocks, show beta vs. alpha | Exercise 1: FF 5-factor regressions for 20 stocks across sectors, test for significant alpha | At scale: Fama-MacBeth cross-sectional regressions for 100 stocks |
| Risk metrics (Sharpe, Sortino, MDD, VaR, CVaR) | Demo: compute all metrics for SPY across regimes, QuantStats tear sheet | Not covered (done in lecture) | At scale: full comparison table for 3 portfolios, QuantStats tear sheets |
| Fundamental Law of Active Management | Demo: simulate IC × sqrt(BR) with synthetic signals | Not covered (done in lecture) | Reference: interpret portfolio IR in context of the Fundamental Law |
| Covariance estimation & random matrix theory | Demo: eigenvalue distribution vs. Marchenko-Pastur for 100 stocks | Exercise 2: denoise covariance using MP bounds, vary N/T ratio | Integrate: use Ledoit-Wolf shrinkage in Markowitz portfolio |
| Hierarchical Risk Parity (HRP) | Demo: build HRP for 50 stocks, show dendrogram and weight comparison | Exercise 3: HRP vs. Markowitz vs. 1/N rolling OOS for 30-50 stocks | At scale: HRP as one of 3 portfolio methods for 100 stocks |
| Transaction costs (constant bps model) | Demo: momentum strategy returns at multiple cost levels | Not covered (done in lecture) | At scale: net-of-cost comparison for all 3 portfolio strategies |

## Key Stories & Facts to Weave In

1. **Harry Markowitz's ironic confession.** When asked how he invested his own retirement money, Markowitz — the inventor of mean-variance optimization and a Nobel laureate — admitted he used a simple 50/50 split between stocks and bonds. "I should have computed the efficient frontier. Instead, I split my contributions 50/50 between stocks and bonds, to minimize my future regret." The inventor of the optimal portfolio used the dumbest possible approach for his own money.

2. **LTCM's collapse (1998), from the portfolio perspective.** LTCM's models said their portfolio was diversified — they held thousands of positions across dozens of markets. But during the Russian debt crisis, correlations spiked to nearly 1.0 across all markets simultaneously. The "diversification" was an illusion based on historical correlations that broke down during stress. Their $125 billion in assets required a $3.6 billion bailout organized by the Federal Reserve. Correlation is not constant — it increases during crises, exactly when you need diversification most.

3. **DeMiguel, Garlappi & Uppal (2009) — "1/N".** This paper, with over 5,000 citations, showed that a naive equal-weight portfolio beats 14 different optimization methods, including Markowitz, across most datasets. The reason: optimization error in estimating the covariance matrix overwhelms any benefit from optimization. You need approximately 3,000 months (250 years) of data for mean-variance optimization to reliably beat 1/N for a 25-stock portfolio. This paper should be required reading for anyone who thinks "more sophisticated" means "better."

4. **Jim Simons and the Medallion Fund.** Renaissance Technologies' Medallion Fund has earned approximately 66% annual returns before fees (39% after the punishing 5-and-44 fee structure) since 1988. The estimated Sharpe ratio is around 2.5 after fees. To achieve this, they trade thousands of instruments with tiny edge per trade but enormous breadth. The Fundamental Law in action: IC × sqrt(BR) = massive IR. Note: the fund has been closed to outside investors since 1993 — the $10 billion is all employee money.

5. **The disappearing value premium.** From 1927-2007, value stocks (high book-to-market) outperformed growth stocks by about 5% per year — the HML factor. Since 2007, the premium has essentially vanished. Fama and French themselves published a paper in 2020 acknowledging that the value premium has weakened. Some argue it was arbitraged away (quants exploited it until it disappeared). Others argue it's cyclical and will return. Your Fama-MacBeth regressions will show this — the HML coefficient is likely insignificant in recent data.

6. **The Sharpe ratio of the S&P 500.** Over the long run (1926-2024), the S&P 500 has a Sharpe ratio of approximately 0.4. A Sharpe of 0.4 means you earn 0.4 units of return per unit of risk, annualized. Most hedge funds aim for Sharpe 1.0-2.0. If your strategy has a backtest Sharpe above 2.5, you should be deeply suspicious — either you've found the next Renaissance Technologies, or (much more likely) you have a bug in your backtest. The deflated Sharpe ratio in Week 18 will formalize this suspicion.

7. **Maximum drawdown in practice.** The S&P 500's maximum drawdown during the 2008 financial crisis was about 55% (from peak in October 2007 to trough in March 2009). At most funds, a 20% drawdown triggers serious conversations. A 30% drawdown gets people fired. A 50% drawdown closes the fund. The Calmar ratio (annual return / max drawdown) captures this asymmetry: investors care disproportionately about drawdowns because drawdowns end careers.

8. **The Marchenko-Pastur law in practice.** Joel Bun, Jean-Philippe Bouchaud, and Marc Potters at Capital Fund Management (a quant hedge fund managing $14 billion) have used random matrix theory to separate signal from noise in covariance matrices. They found that for typical equity portfolios, only the top 5-10 eigenvalues contain genuine market structure (market, sector, country factors). The remaining eigenvalues are indistinguishable from random noise. Using the full covariance matrix for portfolio optimization is like fitting a model to noise in 490 out of 500 dimensions.

## Cross-References
- **Builds on:** Returns math and the `DataLoader` class from Week 1. The concept of fat tails (Week 1, Section 4) — VaR and CVaR are direct responses to the fat-tail problem. Volatility clustering from Week 2 — GARCH(1,1) volatility forecasts can be used as inputs to portfolio optimization.
- **Sets up:** The Information Coefficient (IC) and expanding-window CV methodology are formalized in Week 4 and become the standard evaluation framework for all ML models. Factor models (Fama-French) provide the decomposition used to evaluate alpha in all subsequent weeks. The Sharpe ratio and other risk metrics defined here are used in every homework from Week 4 onward. HRP becomes the default portfolio construction method. The transaction cost model (10 bps round-trip) is the baseline for all backtests; Week 18 extends this to realistic market impact.
- **Recurring thread:** The "simple vs. sophisticated" thread. Equal-weight vs. Markowitz is the same pattern as GARCH(1,1) vs. GARCH variants (Week 2), and simple features vs. complex ones (Weeks 4-5). The parsimonious approach wins more often than ML engineers expect.

## Suggested Reading
- **Grinold & Kahn, *Active Portfolio Management* (1999)** — The source of the Fundamental Law of Active Management. Dense and mathematical, but Chapter 2 (on IC and the Fundamental Law) is essential reading for anyone building ML models for finance. It tells you whether your model has any chance of making money before you ever run a backtest.
- **DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?"** — The paper that showed equal-weighting beats optimization. Over 5,000 citations. Short, readable, and humbling.
- **Lopez de Prado, *Advances in Financial Machine Learning*, Chapter 16 (Hierarchical Risk Parity)** — HRP explained by its inventor. The motivation is compelling even if the mathematical details are dense. The key insight is that hierarchical structure avoids matrix inversion entirely.
