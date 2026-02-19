# Week 18 — Backtesting, Strategy Evaluation & Capstone Integration

> **The most dangerous number in quantitative finance is a backtest Sharpe ratio. This week, you learn why — and how to compute the number that actually matters.**

## Prerequisites
- **Week 3 (Portfolio Theory & Risk):** Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio. These were introduced in Week 3; this week we show why they're necessary but insufficient.
- **Week 4 (Linear Models):** Expanding-window cross-validation. Walk-forward optimization formalizes what you've been doing since Week 4.
- **Week 6 (Financial ML Methodology):** Triple-barrier labeling, purged k-fold CV, meta-labeling. The methodology framework from Week 6 is the foundation for honest backtesting.
- **Week 5 (Tree-Based Methods):** XGBoost model building. Your capstone will likely use XGBoost or a neural net from later weeks.
- **All prior weeks:** The capstone integrates at least 2 techniques from the course. Everything you've learned is fair game.
- **Week 15 (HFT & Microstructure):** Transaction cost understanding. You now know where the 5-10 bps per side comes from, and why it's optimistic.

## The Big Idea

Here's a dirty secret about quantitative finance: most backtests are lies. Not intentional lies — the practitioners running them are usually honest, diligent people. But the process itself is designed to produce flattering results. You try a model. It doesn't work. You try another. Better, but not great. You tweak the features, adjust the lookback window, change the rebalancing frequency. After fifty iterations, you have a strategy with a Sharpe ratio of 2.0. You write it up, present it to the portfolio manager, and — if you're unlucky enough to trade it — watch it lose money for six months straight.

The problem isn't the model. The problem is that you tested fifty strategies and reported the best one. If you flip fifty coins and pick the one with the most heads, that coin looks like it's biased — but it isn't. The same logic applies to strategies: the best of fifty random strategies will have a Sharpe ratio well above zero, purely by chance. Bailey and Lopez de Prado formalized this in 2014 with the **deflated Sharpe ratio** — a correction that accounts for how many strategies you tried. It's the financial equivalent of Bonferroni correction in statistics: adjust for multiple testing, or your results are meaningless.

This week brings together three themes that have been building all semester. First, we formalize the backtesting methodology: walk-forward optimization (the proper way to evaluate strategies over time), realistic transaction costs (not the flat 5 bps we've been using, but market-impact-aware costs that depend on position size and liquidity), and capacity analysis (your strategy works on $1 million; does it work on $100 million?). Second, we compute the deflated Sharpe ratio — the number that honest quants report instead of the raw Sharpe. Third, we introduce causal inference in factor investing: Lopez de Prado's 2023 argument that most published factors aren't causal relationships but statistical associations that break out of sample.

And then: the capstone. You'll build a complete, production-quality ML trading strategy that integrates at least two techniques from the course. This isn't a homework assignment — it's a portfolio piece. The kind of notebook you'd show at a quant firm interview. Data pipeline, feature engineering, model training, portfolio construction, walk-forward backtest, honest evaluation. Every step done correctly, every assumption stated explicitly, every limitation acknowledged. The code is important. The analysis is more important.

## Lecture Arc

### Opening Hook

In 2016, Harvey, Liu, and Zhu published "... and the Cross-Section of Expected Returns" — a paper whose title intentionally echoes the hundreds of papers it debunks. They compiled a database of 316 factors that academic researchers had published as "significant" predictors of stock returns. Momentum, value, size, accruals, asset growth, investment, profitability — 316 of them, published in top journals with t-statistics above 2.0. Harvey et al. showed that, after correcting for multiple testing, the appropriate significance threshold wasn't t > 2.0 — it was t > 3.0. Roughly half the published factors failed this corrected threshold. Half of thirty years of academic finance research — thousands of papers, cited tens of thousands of times — was likely measuring noise. Lopez de Prado later estimated that the true number of tested-and-discarded strategies behind each published factor was not the "3 or 4" that papers report, but more like 100-200. When you account for the full "garden of forking paths," almost nothing survives.

### Section 1: The Backtesting Lie — Why Most Backtests Don't Work in Practice
**Narrative arc:** We catalog the ways that backtests deceive you — not through fraud but through structural biases that are almost impossible to eliminate entirely. The tension: every model we've built in this course has a backtest. How confident should we be? The resolution: specific tools (deflated Sharpe, walk-forward, realistic costs) that separate signal from noise.

**Key concepts:** Look-ahead bias, survivorship bias, selection bias, transaction cost modeling, market impact, capacity constraints, alpha decay.

**The hook:** A quant researcher at a major hedge fund once told me: "Every strategy I've ever backtested has made money. About 10% of them make money when we trade them. And 10% of *those* make money after a year." The funnel is brutal: 100 backtested strategies → 10 that survive realistic simulation → 1 that makes it to production → 0.5 that are still profitable after 12 months. The backtesting process itself is the bottleneck, because it's so easy to accidentally build a strategy that exploits historical artifacts rather than genuine market dynamics.

**Key formulas:**

The backtesting hierarchy of lies, from most to least subtle:

1. **Look-ahead bias:** Using information not available at decision time. The worst version: using the close price to decide whether to trade at the close. The subtle version: using an earnings surprise that's timestamped at market close but was actually released at 4:15 PM.

2. **Survivorship bias:** Training on stocks that exist today, ignoring the ones that delisted. The S&P 500 adds 20-30 stocks per year and removes 20-30. If your training data only includes current members, you've removed every bankruptcy, every merger, every failure.

3. **Selection bias (the biggest one):** You tried 50 strategies. You're presenting the best one. The expected maximum Sharpe ratio of N independent strategies, each with true Sharpe = 0:

$$\mathbb{E}[\max(SR_1, \ldots, SR_N)] \approx \sqrt{2 \ln N}$$

For N = 50: $\sqrt{2 \ln 50} \approx 2.8$. A Sharpe of 2.8 from pure noise. That's higher than most published strategies. This is the problem we solve with the deflated Sharpe ratio.

4. **Transaction costs (the honest tax):** Every trade costs money. The bid-ask spread is the most visible cost, but market impact — the price moving against you because of your own order — is often larger.

Simple cost model (what we've been using):
$$\text{cost} = c \cdot |\Delta w| \cdot V$$

where $c$ is cost per dollar traded (5-10 bps), $\Delta w$ is the weight change, and $V$ is portfolio value.

Square-root market impact model (more realistic for large orders):
$$\text{impact} = \sigma \cdot \sqrt{\frac{Q}{V_{\text{daily}}}}$$

where $Q$ is order size, $V_{\text{daily}}$ is average daily volume, and $\sigma$ is daily volatility. A $10M order in a stock with $50M daily volume and 2% daily vol: impact = $0.02 \times \sqrt{10/50} = 0.009 = 0.9\%$. That's 90 basis points — 9x what your "5 bps flat" assumption would predict.

**Code moment:** Take a strategy from Week 5 (XGBoost top-5 long/short). Backtest it with three cost models: (a) zero costs, (b) 5 bps flat, (c) square-root impact. Plot the three equity curves on the same axes. Output: the zero-cost backtest looks amazing (Sharpe ~2.5). The flat-cost version is decent (Sharpe ~1.2). The square-root impact version is mediocre (Sharpe ~0.5). The gap between (a) and (c) is the "reality gap" — and it's enormous.

**"So what?":** If your backtested Sharpe drops by more than 50% when you add realistic transaction costs, your strategy is a transaction cost donor, not a trading strategy. This is the single most common failure mode in quantitative finance, and it kills more strategies than any other factor.

### Section 2: The Deflated Sharpe Ratio — The Number That Matters
**Narrative arc:** We introduce Bailey and Lopez de Prado's deflated Sharpe ratio (DSR), the tool that corrects for the multiple testing problem in strategy selection. This was previewed in Week 6; here we give the full treatment with implementation.

**Key concepts:** Multiple testing, selection bias, the haircut formula, non-normal return adjustments, probability of backtest overfitting.

**The hook:** In Week 6, we introduced the idea that the Sharpe ratio is inflated when you test multiple strategies. Now we quantify exactly how much. If you test N strategies over T periods, the expected maximum Sharpe ratio under the null hypothesis (all strategies are zero-alpha) is:

$$SR_0 = \sqrt{V[\hat{SR}]} \cdot \left[ (1 - \gamma) \Phi^{-1}\left(1 - \frac{1}{N}\right) + \gamma \Phi^{-1}\left(1 - \frac{1}{N} e^{-1}\right) \right]$$

where $\gamma \approx 0.5772$ (Euler-Mascheroni constant) and $V[\hat{SR}]$ is the variance of the Sharpe ratio estimator:

$$V[\hat{SR}] = \frac{1}{T}\left[1 - \gamma_3 \cdot SR + \frac{\gamma_4 - 1}{4} SR^2\right]$$

Here $\gamma_3$ and $\gamma_4$ are the skewness and excess kurtosis of returns. The deflated Sharpe ratio tests whether your observed Sharpe exceeds this threshold. If DSR > 0.95, your strategy is significantly better than the best of N random strategies at the 5% level.

Let's unpack this with a concrete example. You tried 100 strategies over 5 years of daily data (T = 1,260). Your best strategy has a Sharpe of 1.5. Is it real?

$$SR_0 = \sqrt{\frac{1}{1260}} \cdot \left[(1 - 0.5772) \cdot 3.89 + 0.5772 \cdot 3.72\right] \approx 0.028 \times 3.79 \approx 0.106$$

Wait — that seems low. But that's for Gaussian returns. With fat tails (kurtosis = 6, common in finance), the variance of the Sharpe estimator increases, and $SR_0$ rises. With realistic parameters, $SR_0$ for N=100 over 5 years is typically 0.8-1.2. Your Sharpe of 1.5 might still be significant — but barely. And that's assuming you honestly counted all 100 strategies. Most people undercount.

**Code moment:**

```python
def deflated_sharpe_ratio(observed_sr, sr_variance, n_trials):
    """
    Test whether observed Sharpe exceeds what you'd expect
    from the best of n_trials random strategies.
    """
    from scipy.stats import norm
    gamma = 0.5772  # Euler-Mascheroni
    sr0 = np.sqrt(sr_variance) * (
        (1 - gamma) * norm.ppf(1 - 1/n_trials) +
        gamma * norm.ppf(1 - 1/n_trials * np.exp(-1))
    )
    # Z-test
    z = (observed_sr - sr0) / np.sqrt(sr_variance)
    p_value = 1 - norm.cdf(z)
    return sr0, z, p_value
```

Apply it to the students' own course results. Question: "How many models/strategies did you try during this entire course?" The honest answer, including hyperparameter sweeps, feature experiments, and abandoned approaches: probably 50-200. Plug that number in and watch Sharpe ratios that looked impressive suddenly become insignificant.

**"So what?":** The deflated Sharpe ratio is the difference between "my strategy has a Sharpe of 2.0" (impressive) and "my strategy has a Sharpe of 2.0, but I tried 200 things first, so the deflated Sharpe is 0.85" (not significant). Every honest quant report should include the DSR. If it doesn't, the strategy hasn't been properly evaluated.

### Section 3: Walk-Forward Optimization — The Proper Backtest
**Narrative arc:** We formalize walk-forward optimization, upgrading the expanding-window evaluation from Week 4 to include hyperparameter re-tuning at each step. This is the gold standard for strategy evaluation.

**Key concepts:** Walk-forward optimization, expanding-window CV, anchored vs. rolling windows, hyperparameter re-optimization frequency, in-sample vs. out-of-sample ratio.

**The hook:** In Week 4, you learned expanding-window cross-validation: train on data up to time t, predict at t+1, expand the window, repeat. That's honest for model evaluation. But it has a gap: the *hyperparameters* (learning rate, max depth, number of features, rebalancing frequency) were chosen once, on the full training set, and never re-optimized. In production, you'd re-tune periodically as market conditions change. Walk-forward optimization does this: at each step, it re-optimizes hyperparameters on the in-sample period, then evaluates on the out-of-sample period. It's more realistic, slower, and sometimes reveals that your "robust" strategy was actually dependent on a specific hyperparameter setting that worked in one regime.

**Key formulas:**

Walk-forward optimization with expanding window:

For each evaluation point $t_k$:
1. In-sample period: $[t_0, t_k]$
2. Hyperparameter search: find $\theta^* = \arg\max_\theta \text{CV-Sharpe}([t_0, t_k]; \theta)$
3. Retrain model with $\theta^*$ on $[t_0, t_k]$
4. Out-of-sample evaluation: $[t_k, t_{k+1}]$

Concatenate all OOS periods: the combined OOS equity curve is your "real" backtest.

The in-sample to out-of-sample ratio matters. Too much IS: you overfit. Too little IS: the model doesn't have enough data to learn. Typical choices: IS:OOS = 4:1 (train 4 months, test 1 month) or 12:1 (train 12 months, test 1 month).

**Code moment:** Implement walk-forward on the XGBoost strategy from Week 5. Monthly re-optimization with quarterly hyperparameter re-tuning (full Optuna sweep every 3 months, fixed hyperparameters otherwise). Compare: (a) static model (train once, test forever), (b) expanding-window model (retrain monthly, fixed hyperparameters), (c) full walk-forward (retrain monthly + re-tune quarterly). Output: the walk-forward version typically has slightly lower Sharpe than the expanding-window version (because the hyperparameter search adds noise), but more consistent performance across regimes. The static model shows significant performance decay after 12-18 months.

**"So what?":** Walk-forward optimization is the closest thing to simulating what you'd do in production. It reveals whether your strategy adapts to changing markets or whether it's a one-trick pony that worked in 2020 but not 2022. The computational cost is real (each evaluation point requires a full training + optional hyperparameter search), but it's the price of honest evaluation.

### Section 4: Realistic Market Impact — The Full Treatment
**Narrative arc:** We upgrade from "flat 5 bps" to a proper market impact model. This connects Week 15's microstructure knowledge to practical strategy evaluation.

**Key concepts:** Temporary vs. permanent impact, linear and square-root impact models, Almgren-Chriss framework, capacity constraints.

**The hook:** In Week 15, you learned that the bid-ask spread is only part of your transaction cost. The other part — often larger — is market impact: the price moving against you *because* of your own trading. When you buy 10,000 shares of a stock that normally trades 50,000 shares per day, you're 20% of the day's volume. The market notices. Other participants pull their quotes. The price rises against you before you've finished buying. The total cost of executing that order — spread plus impact — can be 5x what the flat 5 bps model assumes.

**Key formulas:**

The Almgren-Chriss framework (simplified):

**Temporary impact** (affects only the executing trade):
$$g(v) = \eta \cdot v$$

where $v$ is the trade rate (shares per unit time) and $\eta$ is a liquidity parameter. Linear in trade speed.

**Permanent impact** (shifts the price for everyone):
$$h(v) = \beta \cdot v$$

The total implementation shortfall for executing $X$ shares over $T$ periods:

$$\text{IS} = \underbrace{\eta \sum_t v_t^2}_{\text{temporary}} + \underbrace{\beta \sum_t v_t \cdot x_t}_{\text{permanent}} + \underbrace{\lambda \sigma^2 \sum_t x_t^2}_{\text{timing risk}}$$

The square-root model (empirically better for large orders):

$$\text{impact} = \sigma \cdot c \cdot \text{sgn}(Q) \cdot \sqrt{\frac{|Q|}{V_{\text{daily}}}}$$

where $c \approx 0.1$ is a market constant. For a $5M order in a stock with $20M daily volume and 2% daily vol: $0.02 \times 0.1 \times \sqrt{5/20} = 0.001 = 10$ bps. For a $50M order: $0.02 \times 0.1 \times \sqrt{50/20} = 0.003 = 30$ bps. Impact scales with the square root of order size — doubling your order doesn't double your cost, but it does increase it by 40%.

**Capacity constraint:** The maximum AUM where your strategy is still profitable:

$$\text{AUM}_{\max} = \frac{\alpha^2 \cdot V_{\text{daily}}}{\sigma^2 \cdot c^2}$$

where $\alpha$ is your expected return per trade. A strategy with 20 bps expected return trading stocks with $50M daily volume: $\text{AUM}_{\max} \approx \frac{0.002^2 \times 50M}{0.02^2 \times 0.01} = 50M$. Your strategy maxes out at $50 million. Beyond that, the impact costs eat your edge. This is why most hedge fund strategies have capacity limits, and why the largest funds don't run the same strategies as the smallest.

**Code moment:** Take the same strategy, backtest at three AUM levels: $1M, $10M, $100M. With square-root impact, the $1M backtest shows Sharpe ~1.5. The $10M backtest: Sharpe ~1.0. The $100M backtest: Sharpe ~0.2. Plot the "Sharpe vs. AUM" curve — it's monotonically decreasing. Output: the visual is the most important in the lecture. Most strategies look great at $1M and are worthless at $100M.

**"So what?":** Capacity is the hidden dimension of every strategy. A Sharpe of 3.0 at $1M capacity isn't a strategy — it's an interesting simulation. A Sharpe of 0.8 at $1B capacity is a career. Understanding capacity changes which strategies are worth pursuing.

### Section 5: Causal Inference in Factor Investing — The Frontier
**Narrative arc:** We introduce Lopez de Prado's 2023 argument that most "factors" are statistical associations, not causal relationships. This is the most intellectually ambitious section of the course — and the most important for anyone who wants to do factor research that actually holds up out of sample.

**Key concepts:** Associational vs. causal factors, the factor mirage, causal graphs (DAGs), confounders, Lopez de Prado's research protocol.

**The hook:** In 1992, Fama and French published "The Cross-Section of Expected Stock Returns," introducing the three-factor model: market, size, and value. It launched an industry. By 2025, there were 400+ published factors. Momentum, profitability, investment, volatility, accruals, asset growth, share issuance — the list grew by 20-40 factors per year. The problem: most of these factors are *associational*, not *causal*. Small stocks outperformed large stocks from 1963 to 1990 — but the size effect largely disappeared after publication. Value stocks outperformed growth stocks for six decades — and then underperformed for a decade (2010-2020). Were these real relationships that changed, or statistical artifacts that were never real? Lopez de Prado argues the latter for many published factors, and his 2023 book *Causal Factor Investing* proposes a protocol for telling the difference.

**Key formulas:**

The traditional factor research protocol:
1. Hypothesize a factor (e.g., "low-volatility stocks outperform")
2. Backtest it (sort stocks by volatility, go long low-vol, short high-vol)
3. Report the t-statistic (if t > 2.0, publish)
4. Problem: no causal mechanism required. The factor might work because low-vol stocks happen to be large-cap stocks, which happened to outperform in your sample period. The factor is a proxy for something else.

Lopez de Prado's causal protocol:
1. **Theory first:** Propose a causal mechanism. *Why* would low-volatility stocks outperform? (e.g., "behavioral biases cause investors to overpay for high-volatility stocks, creating a systematic overpricing.")
2. **Causal graph (DAG):** Draw the directed acyclic graph showing the hypothesized causal relationships. Identify confounders. Does size confound the volatility-return relationship?
3. **Testable predictions:** Derive predictions that the causal model makes but the associational model doesn't. (e.g., "If the low-vol anomaly is caused by behavioral biases, it should be stronger among retail-dominated stocks and weaker among institutionally-held stocks.")
4. **Backtest with controls:** Test the predictions, controlling for confounders. If the low-vol effect disappears when you control for size and sector, it wasn't a real factor — it was a proxy.
5. **Out-of-sample validation:** Only after steps 1-4 do you run a full backtest.

The key formula is the "corrected" factor return:

$$R_f^{\text{causal}} = R_f - \beta_{\text{confounder}} \cdot R_{\text{confounder}}$$

If the factor return disappears after removing the confounder's contribution, the factor was associational, not causal.

**Code moment:** Take a well-known factor — say, momentum (12-month return predicts next-month return). Compute raw momentum factor returns for the S&P 500 over 2000-2024. Then control for sector (is momentum just "buying the hot sector?") and size (is momentum just "buying large-cap winners?"). Plot the raw factor return vs. the controlled version. Output: for momentum, the controlled return is still positive — momentum is one of the more robust factors. For other factors (e.g., asset growth, accruals), the controlled return often approaches zero.

**"So what?":** If you're building a factor-based strategy for your capstone (or for a job), the causal protocol separates real factors from mirages. It's more work upfront but dramatically reduces the risk of deploying a strategy that worked in backtest because of a confounding variable and fails in production because the confounder changed. This is the frontier of quantitative finance research.

### Section 6: The Full Pipeline — From Signal to Strategy
**Narrative arc:** We synthesize the entire course into a single pipeline. Signal generation, portfolio construction, execution, risk management, evaluation. Each step maps to specific weeks in the course. This section is the capstone blueprint.

**Key concepts:** Signal generation, position sizing, portfolio construction, execution, risk management, the complete pipeline.

**The hook:** A quant fund's process is a pipeline. Your ML model — the thing you've spent 17 weeks learning to build — is one stage. The model produces a signal: "AAPL will go up, TSLA will go down, MSFT will stay flat." But a signal isn't a strategy. You need to convert signals to positions (how many shares of each?), positions to orders (market or limit?), and orders to executions (at what cost?). You need risk management (what if your model is wrong about everything at once?). And you need evaluation (is your backtest honest?). This section maps each step to the tools you've already learned.

**Key formulas:**

The pipeline:

1. **Signal generation** (Weeks 4-12): Model outputs $\hat{y}_i$ for each stock $i$.
2. **Portfolio construction** (Week 3): Convert signals to weights:
   $$w_i = \frac{\text{rank}(\hat{y}_i)}{\sum_j \text{rank}(\hat{y}_j)}$$
   or use mean-variance optimization: $w^* = \arg\max_w w'\mu - \frac{\lambda}{2} w'\Sigma w$.
3. **Execution** (Weeks 15-16): Estimate transaction costs: $c_i = \sigma_i \cdot c \cdot \sqrt{|Q_i| / V_i}$.
4. **Risk management** (Week 3 + Week 11): Position limits, sector exposure limits, drawdown stops. Uncertainty quantification (Week 11) provides a "confidence" filter.
5. **Evaluation** (this week): Walk-forward backtest, deflated Sharpe ratio, capacity analysis.

**Code moment:** Pseudocode for the complete pipeline:

```python
for date in walk_forward_dates:
    # 1. Generate signals
    features = compute_features(data, date)
    predictions = model.predict(features)

    # 2. Construct portfolio
    target_weights = rank_normalize(predictions)
    target_weights = apply_constraints(target_weights, max_position=0.05)

    # 3. Estimate transaction costs
    turnover = compute_turnover(current_weights, target_weights)
    impact = estimate_impact(turnover, volumes, volatilities)

    # 4. Execute (subtract costs)
    current_weights = target_weights
    costs += impact

    # 5. Track performance
    daily_return = (current_weights * asset_returns).sum() - costs
    performance.append(daily_return)

# 6. Evaluate
sharpe = compute_sharpe(performance)
dsr = deflated_sharpe_ratio(sharpe, n_trials=100)
```

Output: students see the complete pipeline in ~30 lines of pseudocode. Each line maps to a specific tool they've already learned. The capstone is implementing this pipeline for real.

**"So what?":** Your ML model is maybe 20% of a trading strategy. The other 80% is everything around it: data cleaning, feature engineering, portfolio construction, execution, risk management, and honest evaluation. The capstone project is about demonstrating that you can do all of it, not just the sexy part.

### Closing Bridge

This is the last lecture. You started 18 weeks ago learning what a bid-ask spread is. You've now built models that predict returns, estimate risk, hedge options, make markets, and analyze blockchain data. The tools are powerful. But the most important lesson of this course — the one that separates the practitioners who survive from the ones who blow up — is intellectual honesty. Honest evaluation of your backtest. Honest accounting of your transaction costs. Honest assessment of how many strategies you tried before finding the one you're presenting. The deflated Sharpe ratio is just a formula. The habit of applying it — the habit of asking "is this real?" before asking "is this profitable?" — is what makes a quant.

Your capstone is your chance to prove you've internalized this. Build something real. Evaluate it honestly. And when the numbers don't look as good as you hoped — which they won't — have the integrity to say so. That's the skill that gets you hired.

## Seminar Exercises

### Exercise 1: The Capacity Curve — From $1M to $100M
**The question we're answering:** At what AUM does your strategy stop working, and what drives the capacity constraint?

**Setup narrative:** The lecture demonstrated that transaction costs degrade Sharpe ratios — that was the wake-up call. This exercise goes further: you'll compute the *capacity curve* for your best strategy, finding the exact AUM at which your edge is consumed by market impact. This is the analysis that hedge fund allocators actually request — and most academic papers never provide.

**What they build:** Take the best strategy from any prior week. Backtest it at 10 AUM levels: $100K, $500K, $1M, $5M, $10M, $25M, $50M, $100M, $250M, $500M. At each level, compute transaction costs using the square-root impact model (from the lecture), adjusting for the actual daily volume of each stock in the universe. Report: Sharpe at each AUM level, the breakeven AUM (where Sharpe drops below 0.5), and the maximum AUM (where Sharpe hits 0). Plot the capacity curve (AUM on x-axis, Sharpe on y-axis). Identify: which stocks in your universe are the bottleneck (lowest capacity)?

**What they'll see:** Most strategies have Sharpe > 1.0 at $1M but drop below 0.5 by $10-50M. The capacity bottleneck is usually 2-3 small-cap or low-volume stocks in the universe. Removing those stocks and re-running raises capacity by 2-5x but slightly reduces the pre-cost Sharpe. The capacity curve is the most honest single plot about a strategy's viability.

**The insight:** A strategy isn't just a signal and a backtest — it's a signal, a backtest, and a *capacity*. The lecture showed that costs eat returns; this exercise shows *how much* they eat at every scale. The capacity curve is the answer to "could anyone actually trade this?" — the question that separates academic exercises from investable strategies.

### Exercise 2: Computing Your Deflated Sharpe Ratio
**The question we're answering:** After 18 weeks of trying different models and strategies, is your best result statistically significant?

**Setup narrative:** This is the honesty exercise. You'll count — honestly — how many distinct strategies/models/parameter settings you tried during this course. Then you'll compute the deflated Sharpe ratio for your best result. Most students are surprised by the answer.

**What they build:** A list of "trials" — every model, feature set, and parameter combination they tried across all 17 homeworks. The deflated Sharpe ratio computation for their best out-of-sample Sharpe. A significance assessment.

**What they'll see:** Most students have tried 50-200 distinct approaches (counting hyperparameter sweeps). For N=100 and T=500 (2 years of daily data), the DSR threshold is approximately Sharpe ~0.8-1.2, depending on return distribution. Many students' best results will fail the significance test.

**The insight:** The deflated Sharpe ratio is humbling. It doesn't mean your models are useless — it means you can't be *confident* they're useful. This is the same lesson as statistical power in any science: with limited data and many trials, significance is hard to achieve. The solution isn't to stop trying — it's to be honest about the number of trials.

### Exercise 3: Alpha Decay Analysis — Is Your Signal Getting Weaker?
**The question we're answering:** Does your model's predictive power decay over time, and if so, how fast?

**Setup narrative:** The lecture demonstrated walk-forward optimization and showed that static models degrade. This exercise asks the deeper question: *why* do they degrade? Is it because the signal decays (the market adapts), because the model overfits (the parameters drift), or because the regime changes (the features' relationships shift)? Diagnosing the cause determines the fix.

**What they build:** Using expanding-window evaluation on their best model, compute: (a) rolling 6-month IC (information coefficient) of the model's predictions vs. actual returns — plot the time series, (b) rolling 6-month feature importance (SHAP or XGBoost feature_importances_) — do the top features change over time? (c) rolling 6-month Sharpe — does strategy performance track signal quality or are there other factors? Overlay all three time series on a single chart with dual y-axes. Compute the half-life of the IC: fit an exponential decay to the IC time series and report the half-life in months.

**What they'll see:** IC typically decays with a half-life of 12-24 months for equity strategies. Feature importance shifts: features that dominate in 2020 (pandemic regime) lose importance in 2022 (rate-hike regime). Strategy Sharpe tracks IC with a lag of 1-2 months — confirming that signal decay is the primary driver of performance decay, not transaction costs or execution.

**The insight:** Alpha decay isn't mysterious — it's measurable. A half-life of 18 months means you need to retrain (and possibly redesign features) at least every 6 months to stay ahead of the decay curve. This is the operational reality of quantitative trading: the model is a living system, not a one-time build. Walk-forward optimization (from the lecture) is the mechanism; alpha decay analysis is the diagnostic that tells you *when* to retrain.

### Exercise 4: The Garden of Forking Paths
**The question we're answering:** How much does random variation in the modeling process affect final strategy performance?

**Setup narrative:** Build the same strategy 20 times with different random seeds and small hyperparameter perturbations. The spread of Sharpe ratios across the 20 runs tells you how much of your result is signal vs. noise. This is the "researcher degrees of freedom" audit.

**What they build:** 20 runs of their strategy with: 5 different random seeds x 4 hyperparameter settings (minor variations: max_depth in {4,5,6,7}, learning_rate in {0.05, 0.08, 0.10, 0.12}). Distribution of Sharpe ratios. Box plot showing the range.

**What they'll see:** The Sharpe ratio varies by 0.3-0.8 across the 20 runs, even with minor hyperparameter changes. The best run may have Sharpe 1.5; the worst may have Sharpe 0.3. If you only report the best run, you're lying — even though you didn't mean to.

**The insight:** This is the visual proof of why the deflated Sharpe ratio matters. The "garden of forking paths" — all the small choices that affect the result — produces a *distribution* of outcomes, not a single number. Reporting the median of this distribution (not the max) is honest. Reporting confidence intervals is better.

## Homework: "End-to-End ML Trading Strategy" (Capstone)

### Mission Framing

This is it. Eighteen weeks of data structures, statistical tests, machine learning models, deep learning architectures, reinforcement learning agents, microstructure analysis, and crypto analytics — all converging into one notebook. Your capstone isn't just a homework assignment. It's a demonstration that you can do financial ML *correctly*, from data acquisition to honest evaluation, with every assumption stated and every shortcut justified.

The requirements are deliberately flexible: any asset universe, any model, any strategy. The constraints are deliberately strict: at least two techniques from the course, proper financial cross-validation, realistic transaction costs, a deflated Sharpe ratio, and — most importantly — an honest written analysis of what works, what doesn't, and why.

The best capstones from prior cohorts have one thing in common: they aren't the ones with the highest Sharpe ratio. They're the ones with the most honest analysis. A notebook that achieves a Sharpe of 0.6 and explains clearly why it's not higher — identifying the specific limitations (alpha decay, transaction costs, regime dependence) with supporting evidence — is worth more than a notebook that claims a Sharpe of 3.0 without mentioning that it tested 200 strategies first.

Build something you'd be proud to show at an interview. Build something that proves you understand not just the ML, but the finance.

### Deliverables

1. **Data Pipeline (30 min):** Clean data acquisition for your chosen universe. Handle survivorship bias (if applicable). Document data sources, date ranges, any exclusions. Minimum: 3 years of data, 20+ assets (equities, ETFs, crypto, or a mix). Show basic data quality checks (missing values, outliers, return distribution sanity).

2. **Feature Engineering (45 min):** At least 15 features spanning at least 3 categories: momentum (e.g., 5/10/20-day returns), volatility (e.g., realized vol, GARCH vol from Week 8), volume (e.g., volume ratio, OBV), and at least one "advanced" category (sentiment from Week 10, on-chain from Week 17, uncertainty from Week 11, or factor exposures from Week 3). Compute IC for each feature. Drop features with |IC| < 0.01.

3. **Model Training (45 min):** At least one ML model trained with proper financial cross-validation (purged k-fold from Week 6 or expanding window from Week 4). Recommended: XGBoost or LightGBM as primary, with an optional neural net comparison. Report in-sample and out-of-sample performance. SHAP analysis for the primary model.

4. **Labeling Strategy (20 min):** Triple-barrier labeling (Week 6) or clearly justified alternative (e.g., next-day return sign, fixed-horizon return). If using triple-barrier, document your barrier parameters and justify them.

5. **Portfolio Construction (30 min):** Signal-to-portfolio conversion. Options: rank-based (long top quintile, short bottom), signal-weighted (weight proportional to prediction strength), or optimized (mean-variance with your model's predictions as expected returns). Apply position limits (max 5-10% per stock).

6. **Walk-Forward Backtest (45 min):** Expanding-window or full walk-forward evaluation. Monthly rebalancing (minimum). Realistic transaction costs: at least 5 bps per side, preferably square-root impact model. Track: gross returns, net returns, turnover, transaction costs. Minimum out-of-sample period: 12 months.

7. **Evaluation Suite (45 min):**
   - Full QuantStats tear sheet (or equivalent): Sharpe, Sortino, Calmar, max drawdown, monthly returns heatmap, rolling Sharpe
   - Deflated Sharpe ratio: be honest about n_trials. Count every model, parameter set, and feature combination you tried in the course
   - Capacity analysis: estimate the maximum AUM at which your strategy is profitable (using square-root impact)
   - Benchmark comparison: vs. buy-and-hold, vs. equal-weight, vs. at least one "smart" benchmark

8. **Written Analysis (60 min):** Minimum 5 pages (or 1,500 words). Must include:
   - Strategy description and rationale
   - What worked and what didn't (be specific — which features helped, which were noise?)
   - Honest limitations (regime dependence, data quality, capacity constraints)
   - What you'd do differently with more time/data/compute
   - The deflated Sharpe verdict: is your result significant?

9. **Deliverable:** Complete Jupyter notebook (or Python package) with all code, visualizations, and inline analysis, plus the 5-page written document (can be the final section of the notebook).

### What They'll Discover

- Most capstone strategies will achieve raw Sharpe ratios of 0.5-1.5. After transaction costs, 0.3-1.0. After the deflated Sharpe correction (with honest n_trials), many will be insignificant. This is the honest truth about financial ML with limited data.
- Feature importance will reveal that simple features (momentum, volatility) dominate in most models. Advanced features (sentiment, on-chain) provide marginal improvement — typically 0.05-0.15 Sharpe points. This matches the industry experience: feature engineering matters, but the marginal feature is rarely transformative.
- Capacity analysis will show that most student strategies are viable at $1-10M but unprofitable above $50M. This is realistic for small quant funds and prop desks, not for BlackRock.
- The written analysis is where the real learning happens. Students who honestly confront their strategy's limitations learn more than students who present inflated numbers.

### Deliverable
A complete capstone project containing: data pipeline, feature engineering with IC analysis, trained model with SHAP, walk-forward backtest with realistic costs, full evaluation suite with QuantStats tear sheet and deflated Sharpe ratio, capacity analysis, and a minimum 1,500-word written analysis. This is the portfolio piece.

## Concept Matrix

| Concept | Lecture | Seminar | Homework (Capstone) |
|---------|---------|---------|----------|
| Transaction cost modeling | Demo: three cost models (zero, flat, sqrt impact) on a Week 5 strategy | Exercise 1: capacity curve across 10 AUM levels, identify bottleneck stocks | Build: realistic cost model in capstone backtest, capacity analysis |
| Deflated Sharpe ratio | Demo: implement DSR, apply to hypothetical strategy | Exercise 2: compute personal DSR across all course trials | Build: report DSR for capstone strategy with honest n_trials |
| Walk-forward optimization | Demo: three evaluation modes (static, expanding, full walk-forward) | Not repeated (lecture establishes methodology) | Build: expanding-window or full walk-forward backtest for capstone |
| Alpha decay & signal diagnostics | Demo: conceptual discussion of why models degrade | Exercise 3: rolling IC, feature importance shifts, IC half-life estimation | Integrate: written analysis of strategy limitations and regime dependence |
| Researcher degrees of freedom | Demo: expected max Sharpe of N random strategies formula | Exercise 4: garden of forking paths (20 runs, different seeds/hyperparams) | Integrate: honest accounting of trials in written analysis |
| Causal inference in factors | Demo: Lopez de Prado protocol, raw vs. controlled factor returns | Not covered (advanced lecture topic) | Optional: apply causal thinking in capstone factor selection rationale |
| Full pipeline (signal to strategy) | Demo: 30-line pseudocode pipeline mapping to course weeks | Not covered (blueprint for capstone) | Build: complete pipeline (data, features, model, portfolio, backtest, evaluation) |
| Market impact & capacity | Demo: sqrt impact model, Sharpe vs. AUM curve | Exercise 1: full capacity curve with stock-level bottleneck analysis | Build: capacity estimate for capstone strategy |

## Key Stories & Facts to Weave In

- **Harvey, Liu & Zhu, "... and the Cross-Section of Expected Returns" (2016):** Compiled 316 published factors. Showed that the appropriate significance threshold (after multiple testing correction) is t > 3.0, not the standard t > 2.0. Roughly half of published factors fail this threshold. The paper sent shockwaves through academic finance and is the foundation for the deflated Sharpe ratio.

- **AQR's "factor drought" (2018-2020):** AQR Capital Management, a $140 billion quantitative hedge fund built on academic factors (value, momentum, quality), lost money for three consecutive years. The flagship Absolute Return Strategy returned -10%, -3%, -8% in 2018-2020. Cliff Asness (AQR's founder) published several defensive papers arguing the factors were real but experiencing a "drawdown." By 2024, AQR had fully recovered. The lesson: even real factors can have multi-year drawdowns, and the difference between "the factor is dead" and "the factor is in a drawdown" is only visible in hindsight.

- **Renaissance Technologies' Medallion Fund:** Averaged 66% annual return before fees (39% after) from 1988-2018. Net of fees, $1 invested in 1988 would be worth $28,000 by 2018. The fund is closed to outside investors and manages "only" $10 billion — because capacity constraints limit its strategies. Even the best quants in history can't scale past a certain point.

- **Lopez de Prado's "Probability of Backtest Overfitting" (2014):** Demonstrated that the probability of selecting an overfitted strategy from a set of backtested strategies exceeds 50% when the number of trials exceeds a remarkably low threshold. The paper formalized what practitioners had long suspected: most backtested strategies are overfit.

- **Knight Capital (revisited for the finale):** The $440M loss in 45 minutes was a backtesting failure as much as an operational failure. The old code that was accidentally activated had been "backtested" (it worked in the test environment). But the test environment didn't have the same market conditions as production. The lesson for capstone: a backtest is only as good as its assumptions, and the gap between simulation and reality is where money dies.

- **Long-Term Capital Management (1998):** LTCM's models were backtested over decades. They had two Nobel laureates, $125 billion in assets, and models that said their positions were safe. They lost $4.6 billion in four months. The models assumed stable correlations; the Russian debt crisis broke that assumption. The lesson: every backtest has assumptions, and every assumption is a potential point of failure.

- **The "Factor Zoo" Problem:** By 2023, academic researchers had published over 400 "significant" factors for stock returns. Cochrane (2011) called this the "factor zoo." Lopez de Prado (2023) estimated that each published factor represents 100-200 unpublished tests, making the true multiple-testing burden orders of magnitude larger than what's reported. The deflated Sharpe ratio is the antidote.

- **Two Sigma's Research Protocol:** Two Sigma, a $60 billion quant fund, reportedly requires researchers to pre-register their hypotheses before running backtests — similar to clinical trial pre-registration. This prevents p-hacking and reduces the multiple-testing problem. The practice is considered extreme in the quant world; most firms don't require it.

## Cross-References
- **Builds on:** Week 3 (basic risk metrics — now advanced with DSR, capacity analysis), Week 5 (XGBoost — your primary model for the capstone pipeline and the benchmark everything is measured against), Week 6 (financial ML methodology — DSR was previewed there, fully developed here), Week 4 (expanding-window CV — now formalized as walk-forward optimization), Week 15 (microstructure — realistic transaction costs), Every prior week (capstone integrates 2+ techniques).
- **Sets up:** Nothing — this is the finale. But the habits established here (honest evaluation, deflated Sharpe, capacity analysis) should persist in every project the student does afterward.
- **Recurring thread:** The "does it actually work?" question that's been building since Week 1 reaches its definitive answer here. The deflated Sharpe ratio is the final exam question: after everything you've tried, is any of it real?

## Suggested Reading
- **Bailey & Lopez de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality" (Journal of Portfolio Management, 2014):** The paper that introduced the DSR. Readable for a finance paper. Focus on Sections 2 (the formula) and 4 (examples). After reading this, you'll never look at a Sharpe ratio the same way.
- **Lopez de Prado, "Causal Factor Investing" (Cambridge University Press, 2023):** The philosophical manifesto for a new approach to factor research. Not easy reading — it requires comfort with causal inference (DAGs, do-calculus) — but the argument is important: if you can't explain *why* a factor works, you can't expect it to keep working.
- **Harvey, Liu & Zhu, "... and the Cross-Section of Expected Returns" (Review of Financial Studies, 2016):** The factor zoo paper. Every quant researcher should read at least the introduction and Table 1. It's the empirical proof that multiple testing is not a theoretical concern — it's the dominant source of false discoveries in quantitative finance.
