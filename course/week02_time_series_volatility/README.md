# Week 2: Financial Time Series & Volatility

**Band:** 1
**Implementability:** HIGH

## Learning Objectives

After this week, students will be able to:

1. Explain why financial time series violate the core assumptions behind standard ML models (stationarity, normality, independence) and quantify each violation on real data
2. Apply and interpret stationarity tests (ADF, KPSS) and diagnose whether a financial series needs differencing, detrending, or fractional differentiation before modeling
3. Read ACF and PACF plots of financial returns, identify volatility clustering from squared-return autocorrelation, and connect these patterns to ARMA/GARCH model selection
4. Fit GARCH(1,1), EGARCH, and GJR-GARCH models to equity return data using the `arch` library and interpret every parameter in financial terms
5. Explain the leverage effect, volatility clustering, and fat tails as stylized facts of financial returns, and demonstrate each empirically
6. Compute realized volatility from daily returns at multiple horizons and compare it to GARCH-implied conditional volatility
7. Apply fractional differentiation to a price series to preserve memory while achieving stationarity, and explain why this matters for ML feature construction

## Prerequisites

- **Required:** Week 1 — specifically: OHLCV data handling, adjusted vs. unadjusted prices, return computation (`pct_change()`), data loading from Parquet, yfinance usage, Polars/pandas familiarity
- **Assumed ML knowledge:** Maximum likelihood estimation, basic understanding of conditional distributions (e.g., mixture models, heteroskedastic regression), hypothesis testing mechanics (p-values, null vs. alternative). Students know general ML/DL — they do NOT know econometric time series models (ARMA, GARCH) or financial econometrics tests (ADF, KPSS).
- **Prerequisite explanations in this week:** Extended time series primer (~20 min in Lecture Section 1) covering: white noise, random walk, strict vs. weak stationarity, AR models (what autoregressive means, the AR(1) equation, mean-reversion), MA models (past forecast errors, NOT smoothing — a common confusion), ARMA as the combination, the lag operator as notation, ACF and PACF (what they measure, how to read the plots, how they identify AR vs. MA structure), and the concept of conditional variance (variance that changes over time — the bridge to GARCH). This is the econometric vocabulary that ML experts typically lack. The primer teaches these as tools, then the rest of the lecture applies them to financial data.

## Narrative Arc

Week 1 ended with a provocation: you computed return statistics for 10 stocks and found kurtosis far above 3, tail events far more frequent than a normal distribution predicts, and behavior that varies wildly across assets. The data told you something was off. Week 2 explains what, why, and what to do about it.

The central question is: **"Why do financial time series break your ML assumptions, and how does the finance industry handle it?"** An ML expert's default toolkit assumes stationary inputs, roughly Gaussian noise, and independent observations. Financial returns violate all three. They are non-stationary in levels (prices trend), fat-tailed in returns (extreme moves happen 10-100x more often than Gaussian models predict), and temporally dependent in their second moment (calm days cluster together, as do volatile days). If you feed raw financial data into a standard ML pipeline without understanding these properties, your model will silently learn from statistical ghosts — spurious trends, understated tail risk, and false independence.

The week resolves this tension through a single coherent arc: first, diagnose the violations empirically (stationarity tests, ACF/PACF analysis, distributional tests). Then, understand the theoretical framework the finance industry built to handle them (GARCH family — the canonical parametric volatility model). Finally, learn the practical tools that bridge time series analysis and ML feature engineering (realized volatility, fractional differentiation). By the end, students should look at any financial time series and immediately know: is it stationary? what's its autocorrelation structure? is volatility clustering present? do I need to difference, and if so, by how much? These are not academic exercises — they are the first five minutes of any quantitative research project, and getting them wrong invalidates everything downstream.

## Lecture Outline

### Section 1: The Time Series Toolkit — From White Noise to ARMA

- **Hook:** "Your ML models assume i.i.d. data. Financial returns are not i.i.d. — and the way they fail to be i.i.d. is specific, predictable, and exploitable. But the tools for diagnosing and modeling this non-i.i.d. behavior come from econometrics, not ML. This section gives you the vocabulary and toolkit — then we'll spend the rest of the lecture applying it to real financial data."
- **Core concepts (this is a ~20 min primer — the longest section):**
  - **White noise:** Uncorrelated, zero-mean, constant variance — the simplest stochastic process. The "nothing to model here" baseline.
  - **Random walk:** Cumulative sum of white noise — `P_t = P_{t-1} + epsilon_t`. A model of prices under the efficient market hypothesis. Key property: non-stationary (variance grows with time). Differencing a random walk gives back white noise — this is why we work with returns instead of prices.
  - **Stationarity:** Strict stationarity (full distribution invariant to time shifts) vs. weak/covariance stationarity (constant mean, constant variance, autocovariance depends only on lag — the practical definition). Why it matters for ML: a non-stationary input means your training distribution differs from your test distribution by construction. Financial prices are non-stationary (they trend). Financial returns are approximately stationary — but "approximately" hides important details that the rest of the lecture unpacks.
  - **AR (autoregressive) models:** `y_t = c + phi * y_{t-1} + epsilon_t` (AR(1)). The current value depends linearly on its own past values. If |phi| < 1, the series is stationary and mean-reverting — it pulls back toward c/(1-phi). AR models are the time series analog of the recurrence relations ML practitioners know from RNNs, except the relationship is linear and parameters are fit by MLE or OLS, not gradient descent. AR(p) extends to p lags.
  - **MA (moving average) models:** `y_t = c + epsilon_t + theta * epsilon_{t-1}` (MA(1)). **Important:** the "moving average" in MA is NOT a smoothing window average — it's a weighted combination of current and past *forecast errors* (shocks). This confuses almost everyone the first time. MA(q) means the current value depends on the last q shocks. MA models capture short-lived effects that dissipate after q periods.
  - **ARMA:** The combination — `y_t = c + phi * y_{t-1} + epsilon_t + theta * epsilon_{t-1}` (ARMA(1,1)). AR captures persistence, MA captures shock propagation. ARMA(p,q) is the general workhorse for stationary time series. In practice for financial returns, ARMA structure is weak (returns are nearly white noise), but understanding ARMA is essential scaffolding for GARCH — which is basically "ARMA applied to the variance process."
  - **The lag operator (notation only):** `L y_t = y_{t-1}`. Compact notation for writing AR/MA polynomials: `(1 - phi*L) y_t = epsilon_t`. Students should be able to read equations using L but won't need to manipulate the algebra — it appears in papers and library docs.
  - **ACF (autocorrelation function):** Correlation between `y_t` and `y_{t-k}` for each lag k, plotted as a bar chart with 95% confidence bands. If a bar exceeds the band, that lag has statistically significant autocorrelation. Key signatures: AR processes show geometric ACF decay; MA(q) shows ACF that cuts off sharply after lag q; white noise shows all bars inside the bands.
  - **PACF (partial autocorrelation function):** Correlation between `y_t` and `y_{t-k}` after removing the effects of intermediate lags. The mirror image of ACF for model identification: AR(p) shows PACF cutoff after lag p; MA processes show geometric PACF decay. Reading ACF + PACF together is the classical method for choosing the AR and MA orders — the "Box-Jenkins methodology."
  - **Conditional variance — the bridge to GARCH:** ML practitioners know conditional distributions (e.g., a Gaussian whose mean depends on features). The same idea applies to variance: what if the variance of today's return depends on yesterday's return? A calm day predicts another calm day; a volatile day predicts more volatility. This is "conditional heteroskedasticity" — the variance (skedasticity) changes (hetero) over time, and it's predictable from past data (conditional). GARCH (Section 5) formalizes this into a model. The key intuition: returns may be unpredictable (ACF ≈ 0), but the SIZE of returns is predictable (ACF of squared returns >> 0). Unpredictable direction + predictable magnitude = the core asymmetry of financial time series.
- **Key demonstration:** Generate three synthetic series (2500 steps each): (1) a random walk (cumulative Gaussian noise), (2) a stationary AR(1) with phi=0.7, (3) white noise. Plot all three. Compute and display ACF for each (6-panel figure: 3 series + 3 ACFs). The random walk's ACF decays very slowly (non-stationary); the AR(1)'s ACF decays geometrically; the white noise ACF is flat. Then show a toy conditional-variance process: generate returns where today's variance = 0.001 + 0.9 * yesterday's squared return. Plot the returns AND their rolling variance — this "homemade GARCH" should visibly cluster calm and volatile periods.
- **Acceptance criteria:**
  - Random walk variance at step 2500 is > 10x the variance at step 250 (demonstrates growing variance)
  - AR(1) series has variance within a factor of 2 between first half and second half (approximately stationary)
  - Random walk ACF decays very slowly (ACF at lag 50 > 0.5); AR(1) ACF decays geometrically (ACF at lag 10 < 0.2 * ACF at lag 1)
  - White noise ACF: all lags 1-20 inside 95% confidence bands
  - Toy conditional-variance series shows visible volatility clustering (visual, not asserted numerically)
  - 6-panel figure (3 series + 3 ACFs) plus the toy conditional-variance figure both produced
- **Bridge:** "That's the toolkit. Now let's apply it to real financial data. The results are striking — and they violate almost every assumption you'd make if you hadn't looked."

### Section 2: Stylized Facts — What Makes Financial Returns Special

- **Hook:** "There's a list of empirical regularities that financial returns exhibit across every asset class, every market, and every time period ever studied. If your model doesn't respect these stylized facts, it's wrong — no matter how good its backtest looks."
- **Core concepts:** The six canonical stylized facts of financial returns: (1) Returns are approximately uncorrelated (ACF near zero at all lags — no easy linear predictability). (2) Squared returns (and absolute returns) ARE correlated — volatility clusters. (3) Returns are fat-tailed (kurtosis >> 3, extreme events far more frequent than Gaussian). (4) Returns exhibit negative skewness (crashes are larger than rallies — the leverage effect). (5) Volatility is persistent — high-vol periods last days to weeks, not hours. (6) The "volatility smile" in options prices (forward pointer to Week 9) reflects the market's awareness of facts 3-4. Connect each fact to a modeling consequence: fact 1 means linear return prediction is nearly hopeless; fact 2 means volatility prediction IS feasible; fact 3 means your risk models underestimate tails; fact 4 means symmetric loss functions are wrong.
- **Key demonstration:** Using 15 years of daily returns for SPY (S&P 500 ETF), produce a 4-panel diagnostic figure: (a) return time series showing volatility clustering visually, (b) histogram of returns overlaid with a fitted Gaussian of the same mean/variance — the fat tails should be dramatically visible, (c) ACF of returns (should be near zero at all lags), (d) ACF of squared returns (should show significant positive autocorrelation out to 20+ lags). This single figure demonstrates facts 1, 2, and 3 simultaneously. Run the Jarque-Bera test (a hypothesis test for normality based on skewness and kurtosis — null: the data is normally distributed; rejection means non-normal) and print the result — it should reject overwhelmingly.
- **Acceptance criteria:**
  - ACF of raw returns: absolute value < 0.05 for all lags 1-20 (approximately uncorrelated)
  - ACF of squared returns: value at lag 1 > 0.10 (significant volatility clustering)
  - ACF of squared returns: at least 10 lags out of the first 20 exceed the 95% confidence band
  - Return kurtosis > 5 (fat tails — Week 1 homework found this, now we visualize it)
  - Histogram shows visible excess density in tails relative to Gaussian overlay
  - Jarque-Bera test rejects normality (p-value < 0.01)
- **Bridge:** "Returns are uncorrelated but NOT independent — their squares are correlated. This is the signature of volatility clustering. Before we model it, we need formal tools to test whether a series is stationary."

### Section 3: Testing for Stationarity — ADF and KPSS in Practice

- **Hook:** "Is this series stationary? The eyeball test is unreliable — financial data loves to fool visual inspection. We need formal hypothesis tests. But here's the twist: the two standard tests can contradict each other, and knowing what that contradiction means is the real skill."
- **Core concepts:** A "unit root" means the AR coefficient phi = 1 exactly — a random walk. If phi = 1, shocks never decay and the series wanders forever (non-stationary). If phi < 1, shocks decay and the series is stationary. The stationarity question reduces to: is phi = 1 or phi < 1? Augmented Dickey-Fuller (ADF) test — null hypothesis: unit root (non-stationary), so rejection means "stationary." KPSS test — null hypothesis: stationary, so rejection means "non-stationary." The critical distinction: ADF looks for evidence AGAINST stationarity; KPSS looks for evidence FOR stationarity. When they agree (both reject or both fail to reject), the picture is clear. When they disagree — ADF rejects (says stationary) but KPSS also rejects (says non-stationary) — the series may be trend-stationary or fractionally integrated. This disagreement is COMMON with financial data and motivates fractional differentiation later. Practical guidance: always run BOTH tests. Report both p-values. Make decisions based on the joint result, not a single test.
- **Key demonstration:** Run ADF and KPSS on four series from real data: (1) raw SPY prices (expect: ADF fails to reject, KPSS rejects — non-stationary), (2) SPY daily returns (expect: ADF rejects, KPSS fails to reject — stationary), (3) SPY log prices (expect: similar to raw prices — still non-stationary), (4) SPY cumulative returns (expect: ADF may give ambiguous results — possibly trend-stationary). Present results in a clean summary table with test statistics and p-values.
- **Acceptance criteria:**
  - Raw SPY prices: ADF p-value > 0.05 (fails to reject unit root)
  - Raw SPY prices: KPSS p-value < 0.05 (rejects stationarity)
  - SPY daily returns: ADF p-value < 0.01 (rejects unit root — stationary)
  - SPY daily returns: KPSS p-value > 0.05 (fails to reject stationarity)
  - Summary table has 4 rows (one per series) and columns for ADF stat, ADF p-value, KPSS stat, KPSS p-value, and diagnosis
- **Bridge:** "Returns are stationary. Good. But 'stationary' doesn't mean 'independent.' The autocorrelation function tells us what kind of temporal dependence remains — and for financial data, the answer is surprising."

### Section 4: Diagnosing Financial Returns — ACF, PACF, and the Limits of ARMA

- **Hook:** "In a first pass, financial returns look like white noise — the ACF is flat. Most practitioners stop here. But plot the ACF of SQUARED returns and the picture changes completely. That gap — between uncorrelated returns and correlated volatility — is where the entire GARCH literature lives."
- **Core concepts:** Apply the ACF/PACF toolkit from Section 1 to real financial returns. For financial returns, ACF and PACF both show little structure — Box-Jenkins methodology says returns are approximately white noise. But apply the same analysis to squared or absolute returns and strong, slowly decaying autocorrelation appears — this is the "ARCH effect" (autoregressive conditional heteroskedasticity), the empirical signature that variance clusters. Fit an ARMA(1,1) to returns as a baseline: the model captures negligible mean dynamics, and crucially, the RESIDUALS still show the squared-return autocorrelation pattern. This proves ARMA is not enough — it models the mean, but the variance is where the action is. The Ljung-Box test formalizes this: applied to raw residuals it says "mean captured" (high p-value), applied to squared residuals it says "variance NOT captured" (low p-value). This is the setup for GARCH.
- **Key demonstration:** Produce ACF and PACF plots for SPY returns (4 panels): (1) ACF of returns, (2) PACF of returns, (3) ACF of squared returns, (4) ACF of absolute returns. Panels 1-2 should look like white noise. Panels 3-4 should show significant, slowly decaying autocorrelation. Fit an ARMA(1,1) model to SPY returns using statsmodels and display diagnostics — the residuals should still exhibit the squared-return autocorrelation pattern (ARMA captures the mean, not the variance dynamics). Explain the Ljung-Box test: it's a hypothesis test for "is there remaining autocorrelation in this series?" — null hypothesis is "no autocorrelation." Run it on raw residuals (expect pass) and squared residuals (expect fail).
- **Acceptance criteria:**
  - ACF of returns: fewer than 2 lags out of 1-20 exceed the 95% confidence band (near-white-noise)
  - ACF of squared returns: at least 10 lags out of 1-20 exceed the 95% confidence band (strong clustering)
  - ACF of absolute returns: at least 10 lags out of 1-20 exceed the 95% confidence band
  - ARMA(1,1) residuals: Ljung-Box p-value on raw residuals > 0.05 (mean captured), Ljung-Box p-value on squared residuals < 0.05 (variance NOT captured)
  - 4-panel ACF figure produced with clear visual contrast between returns and squared returns
- **Bridge:** "ARMA captures the mean dynamics. The variance is still wild. We need a model for the conditional variance — and the finance industry settled on one answer over 30 years ago."

### Section 5: GARCH — The Industry's Canonical Volatility Model

- **Hook:** "Robert Engle won the 2003 Nobel Prize in Economics for ARCH models. His student Tim Bollerslev generalized it to GARCH in 1986. Forty years later, GARCH(1,1) is still the single most widely used parametric volatility model in risk management, options pricing, and quantitative trading. Why has nothing replaced it?"
- **Core concepts:** GARCH(1,1) specification: `sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2`, where `epsilon_t` is the return shock (return minus mean) and `sigma_t^2` is the conditional variance (today's variance, given everything we know up to yesterday). Three parameters, one equation, captures volatility clustering. Note the ARMA parallel from Section 1: GARCH is essentially an ARMA model applied to the squared returns — the `alpha` term is the AR-like component (yesterday's shock), the `beta` term is the MA-like component (yesterday's variance estimate). Interpretation: `alpha` = the reaction coefficient (how much today's shock matters), `beta` = the persistence coefficient (how long high volatility lasts), `alpha + beta` = persistence (how quickly volatility decays to the long-run mean `omega/(1 - alpha - beta)`). Typical equity values: `alpha` ~ 0.05-0.10, `beta` ~ 0.85-0.95, persistence ~ 0.95-0.99. Why GARCH endures: (a) parsimonious — 3 parameters beat 300 for risk management where overfitting is deadly, (b) interpretable — regulators understand and accept it, (c) well-calibrated for VaR (Value at Risk — "the maximum expected loss at a given confidence level") and ES (Expected Shortfall — "the average loss beyond VaR"), the two standard risk metrics in finance, (d) good enough — hybrid GARCH+DL models are a growing research direction (GARCH features as inputs to neural networks), but standalone GARCH remains the production standard.
- **Key demonstration:** Fit GARCH(1,1) to SPY daily returns using the `arch` library. Display the estimated parameters with standard errors. Plot the conditional volatility (sigma_t) overlaid on absolute returns. The conditional volatility should visibly track the volatility regime — rising during 2008, 2020 (COVID), and 2022, falling during calm periods. Print the persistence (alpha + beta) and the implied long-run annualized volatility.
- **Acceptance criteria:**
  - GARCH(1,1) converges (optimization status is 0 or "Optimization terminated successfully")
  - alpha > 0 and beta > 0 (both positive — meaningful volatility dynamics)
  - alpha + beta > 0.90 and < 1.0 (high persistence, stationary — typical for equity indices)
  - Conditional volatility plot shows visible spikes during known crisis periods (2008-2009, March 2020)
  - Long-run annualized volatility between 10% and 30% (reasonable range for SPY)
- **Bridge:** "GARCH(1,1) captures the level and persistence of volatility. But it treats positive and negative shocks symmetrically. Financial markets don't — a crash increases volatility more than a rally of the same magnitude. Asymmetric models fix this."

### Section 6: Asymmetric GARCH — The Leverage Effect

- **Hook:** "When the market drops 3%, volatility spikes. When it rallies 3%, volatility barely moves. This asymmetry has been documented in every equity market in the world and is called the leverage effect — not because anyone literally uses leverage, but because of a 1970s theory about balance-sheet leverage amplifying negative shocks. The theory turned out to be wrong, but the name stuck and the empirical effect is rock-solid."
- **Core concepts:** EGARCH: the log-volatility formulation naturally captures asymmetry through separate coefficients for positive and negative shocks. A negative gamma means negative returns increase log-vol more than positive returns. GJR-GARCH: adds an indicator function for negative shocks — an explicit "bad news premium" on volatility. Model comparison: fit GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) to the same data. Compare using BIC (Bayesian Information Criterion) — BIC penalizes complexity, so if EGARCH or GJR-GARCH wins despite having an extra parameter, the asymmetry is real and material.
- **Key demonstration:** Fit all three models (GARCH, EGARCH, GJR-GARCH) to SPY returns using the `arch` library. Report parameters, log-likelihood, and BIC for each in a comparison table. Plot conditional volatility from all three models overlaid. Compute the "news impact curve" for each model: how does a shock of size epsilon affect next-period variance? Plot these curves — GARCH gives a symmetric V-shape, EGARCH and GJR give an asymmetric curve tilted toward negative shocks.
- **Acceptance criteria:**
  - All three models converge successfully
  - EGARCH gamma parameter is negative (negative shocks increase volatility more)
  - GJR-GARCH gamma parameter is positive (indicator for negative shocks adds volatility)
  - BIC: at least one asymmetric model (EGARCH or GJR-GARCH) has lower (better) BIC than GARCH(1,1)
  - News impact curve plot shows visible asymmetry for EGARCH and GJR, symmetry for GARCH
  - Comparison table includes columns: model name, omega, alpha, beta, gamma (where applicable), log-likelihood, BIC
- **Bridge:** "GARCH models give you conditional volatility — a model's estimate of how volatile the market IS. But you can also MEASURE volatility directly from the data, without any model at all. That's realized volatility."

### Section 7: Realized Volatility and Volatility Forecasting

- **Hook:** "GARCH estimates volatility from a model. Realized volatility measures it from the data. In an ideal world, they'd agree. In practice, they diverge — and the divergence tells you something."
- **Core concepts:** Realized volatility: the standard deviation of returns over a backward-looking window. Daily realized vol at horizon h = `std(r_{t-h+1}, ..., r_t)`. Common horizons: 5-day (1 trading week), 21-day (1 month), 63-day (1 quarter). The rolling window tradeoff: short windows are noisy but responsive; long windows are smooth but laggy. RV is model-free — it's a measurement, not a forecast. GARCH conditional volatility is a forecast — sigma_t is the model's prediction for today's volatility given yesterday's information. Comparing them: GARCH should LEAD realized volatility (it's forward-looking), and the gap between GARCH's forecast and subsequent RV is the forecast error. Forward pointer: when high-frequency intraday data is available, realized volatility computed from 5-minute returns is far more precise than daily — the HAR (Heterogeneous Autoregressive) model is the workhorse for forecasting it. That's beyond this week's scope but is a key tool in Weeks 8 and 13.
- **Key demonstration:** Compute rolling realized volatility for SPY at three horizons (5-day, 21-day, 63-day). Plot all three alongside the GARCH(1,1) conditional volatility from Section 5 (annualized, all on the same scale). Compute the correlation between GARCH conditional vol and next-day 21-day realized vol — this measures GARCH's forecasting ability.
- **Acceptance criteria:**
  - Rolling RV computed for all three horizons (5, 21, 63 days)
  - 5-day RV is visibly noisier than 63-day RV in the plot (shorter window = more noise)
  - GARCH conditional vol and 21-day RV have correlation > 0.50 (GARCH has genuine forecasting power)
  - All volatility series are annualized (multiplied by sqrt(252)) for interpretable scale
  - Plot shows 4 overlaid series with a legend distinguishing each
- **Bridge:** "We can model volatility and we can measure it. But there's one more problem: when you difference a price series to make it stationary, you destroy the long-memory structure that might be useful for ML models. Fractional differentiation offers a middle path."

### Section 8: Fractional Differentiation — Stationarity Without Amnesia

- **Hook:** "Standard practice: difference your prices to get returns. Returns are stationary. But you just threw away the memory — the long-run level information that might help your ML model. Lopez de Prado's fractional differentiation lets you difference 'just enough' to achieve stationarity while preserving as much memory as possible."
- **Core concepts:** Integer differentiation: d=0 (prices, non-stationary, full memory), d=1 (returns, stationary, no memory of levels). Fractional differentiation: d between 0 and 1 — difference by a fractional amount. Implementation: the fracdiff operator uses a truncated binomial series expansion. The key parameter is d — the minimum d that makes the series pass the ADF test. This d is typically 0.3-0.7 for equities, meaning you preserve substantially more memory than full differencing. Why it matters for ML: features with more memory have more predictive information. A fractionally differentiated series is both stationary (safe for ML) and memory-rich (potentially more informative than returns). This is directly from Lopez de Prado's "Advances in Financial Machine Learning" (2018), Chapter 5 — one of the most influential ideas in financial ML of the last decade.
- **Key demonstration:** Take SPY log prices. Apply fractional differentiation at d = 0.0, 0.3, 0.5, 0.7, 1.0. For each, run the ADF test and report the p-value. Find the minimum d (to two decimal places) where ADF rejects the unit root at 5% significance. Plot all five differenced series on a single figure — d=0 looks like prices (trending), d=1 looks like returns (noisy, mean-zero), intermediate d values look like a hybrid. Print a table: d, ADF p-value, correlation with original price series (memory measure).
- **Acceptance criteria:**
  - d=0.0 (raw prices): ADF p-value > 0.05 (non-stationary)
  - d=1.0 (returns): ADF p-value < 0.01 (stationary)
  - Minimum d for stationarity is between 0.2 and 0.8 (typical for equity indices)
  - Correlation with original prices decreases as d increases (memory loss)
  - Fractionally differenced series at minimum d has ADF p-value < 0.05 AND correlation with original prices > correlation of d=1.0 series with original prices (the whole point: more memory retained)
  - Table and multi-panel plot both produced

### Closing

- **Summary table:** Key concepts and their practical significance:

| Concept | What It Is | Why It Matters |
|---|---|---|
| Stationarity | Constant mean, variance, autocovariance | Non-stationary inputs = shifting train/test distributions |
| ADF / KPSS | Stationarity hypothesis tests | Always run both; disagreement signals fractional integration |
| Stylized facts | Empirical regularities of returns | Any model violating these is wrong by construction |
| Volatility clustering | Calm and volatile periods persist | Squared-return autocorrelation; motivates GARCH |
| GARCH(1,1) | Conditional variance model | Industry standard for 40 years; 3 interpretable parameters |
| EGARCH / GJR-GARCH | Asymmetric volatility models | Leverage effect: crashes spike vol more than rallies |
| Realized volatility | Model-free vol measurement | The target variable for vol forecasting models |
| Fractional differentiation | Stationarity with memory preservation | Better ML features than integer-differenced returns |

- **Forward pointer (GARCH+DL):** GARCH outputs (conditional volatility, standardized residuals) are increasingly used as input features to neural network volatility models. The hybrid approach — GARCH for structure, DL for residual nonlinearities — is a growing research direction. Students will encounter this directly in Week 8 (Deep Learning for Financial Time Series).
- **Bridge to Week 3:** "You now understand how individual financial time series behave — their distributional properties, their volatility dynamics, their memory structure. Next question: how do MULTIPLE assets relate to each other? Week 3 introduces factor models — the finance industry's framework for understanding why stocks move together, and how to decompose returns into systematic and idiosyncratic components."
- **Suggested reading:**
  - **Cont, R. (2001), "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues," *Quantitative Finance*** — The definitive catalog of stylized facts. Short, readable, universally cited. Read this first.
  - **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity," *Journal of Econometrics*** — The original GARCH paper. Surprisingly accessible. 3 parameters, one equation, a Nobel Prize.
  - **Lopez de Prado, M. (2018), "Advances in Financial Machine Learning," Chapter 5** — Fractional differentiation for financial ML. The single most practical chapter in the book for time series feature engineering.
  - **Engle, R. (2001), "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics," *Journal of Economic Perspectives*** — Engle's own readable overview of the GARCH family. Written for economists, accessible to ML practitioners.
  - **Tsay, R. (2010), "Analysis of Financial Time Series," Chapters 1-3** — Comprehensive textbook treatment of everything in this week. Denser than the above but the standard reference.

## Seminar Outline

### Exercise 1: The Stationarity Landscape — How 20 Stocks Behave Differently

- **Task type:** Guided Discovery
- **The question:** In Week 1, you computed return statistics for 10 stocks and found fat tails everywhere. But do all stocks have the SAME kind of non-stationarity? Do large-cap ETFs and small-cap growth stocks fail stationarity tests in the same way? Is the degree of non-stationarity uniform across asset types?
- **Data needed:** 15 years of daily adjusted close prices from yfinance for 20 diverse tickers: a mix of large-cap (AAPL, MSFT, JPM, JNJ), high-volatility (TSLA, NVDA, MARA), ETFs (SPY, QQQ, IWM, TLT, GLD), sector ETFs (XLE, XLF, XLK), and a few mid-caps/special cases (GE, INTC, BA, PFE, DIS).
- **Tasks:**
    1. Compute daily log returns for all 20 tickers
    2. Run ADF and KPSS tests on both prices and returns for every ticker. Build a 20-row summary table with columns: ticker, ADF p-value (prices), KPSS p-value (prices), ADF p-value (returns), KPSS p-value (returns), diagnosis (non-stationary / stationary / ambiguous)
    3. For each ticker, compute return kurtosis, skewness, and the Ljung-Box p-value on squared returns at lag 20 (a test for ARCH effects / volatility clustering). Add these to the summary table.
    4. Rank the 20 tickers by kurtosis. Which asset types have the fattest tails? Which have the least? Is there a pattern by sector, cap size, or asset class?
    5. Identify any tickers where the ADF/KPSS joint diagnosis on returns is NOT "clearly stationary." Investigate: what's different about these returns?
- **Expected insight:** All prices fail stationarity tests (trivially — they trend). All returns pass (trivially — differencing works). The interesting variation is in degree: TSLA and NVDA have kurtosis > 15, while SPY and TLT are closer to 5-7. Bonds (TLT) have lower kurtosis than equities. Commodities proxy (GLD) has different skewness patterns. The Ljung-Box test on squared returns should reject for nearly every ticker — volatility clustering is truly universal — but the strength of the ARCH effect varies. Students should discover that the stylized facts are universal in direction but heterogeneous in magnitude, and that this heterogeneity is systematic (high-vol stocks have fatter tails; bonds differ from equities).
- **Acceptance criteria:**
  - ADF and KPSS run on all 20 tickers for both prices and returns (80 tests total)
  - All 20 tickers' prices: ADF p-value > 0.05 (non-stationary)
  - All 20 tickers' returns: ADF p-value < 0.05 (stationary)
  - Return kurtosis > 3 for all 20 tickers
  - At least 3 tickers have kurtosis > 10 (the volatile names)
  - Ljung-Box test on squared returns: p-value < 0.05 for at least 18 of 20 tickers (volatility clustering is near-universal)
  - Summary table has all 20 rows with complete columns

### Exercise 2: GARCH Model Comparison — Which Flavor of Volatility Wins?

- **Task type:** Skill Building
- **The question:** The lecture showed GARCH(1,1), EGARCH, and GJR-GARCH on SPY. Does the ranking hold across different assets? Does the "best" model depend on the stock?
- **Data needed:** Daily returns for 5 tickers with diverse volatility profiles: SPY (index), AAPL (large-cap tech), JPM (financials), TSLA (high-volatility growth), TLT (bonds — does GARCH even make sense here?)
- **Tasks:**
    1. Fit GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) to each of the 5 tickers using the `arch` library. Use `mean='Constant'` and `dist='Normal'` for comparability across models.
    2. For each ticker, record: all model parameters, log-likelihood, AIC, BIC. Build a 5x3 comparison table (5 tickers, 3 models).
    3. For each ticker, identify the best model by BIC. Is it always an asymmetric model?
    4. Compare the persistence (alpha + beta for GARCH, equivalent for EGARCH/GJR) across tickers. Which assets have the most persistent volatility? Does persistence correlate with average volatility level?
    5. For TLT (bonds): examine the GARCH fit. Is volatility clustering as strong as for equities? Does the leverage effect exist for bonds? (Interest rate dynamics are different — negative returns on bonds mean rising rates, which have different economic implications than equity crashes.)
    6. Produce a 5-panel figure: one row per ticker, each panel showing the conditional volatility from the best-fit model overlaid on absolute returns.
- **Expected insight:** Asymmetric models (EGARCH or GJR-GARCH) win by BIC for most equity tickers due to the leverage effect, but the margin varies. TSLA may be an exception — its extreme volatility might make the asymmetry parameter harder to estimate precisely. TLT should show weaker volatility clustering and a less clear leverage effect compared to equities, revealing that GARCH's structure implicitly assumes equity-like dynamics. Persistence is high for all equities (alpha + beta > 0.93) but varies meaningfully — SPY may have higher persistence than individual stocks because index diversification smooths out idiosyncratic vol spikes.
- **Acceptance criteria:**
  - All 15 model fits converge (5 tickers x 3 models)
  - Best model by BIC identified for each ticker
  - At least 3 of 5 tickers show an asymmetric model as best by BIC
  - Persistence (alpha + beta) > 0.90 for all equity tickers
  - TLT squared-return ACF at lag 1 is lower than SPY squared-return ACF at lag 1 (weaker clustering for bonds)
  - Comparison table is complete with all parameters, log-likelihoods, and information criteria
  - 5-panel conditional volatility figure produced

### Exercise 3: Fractional Differentiation in Practice — Finding the Minimum d

- **Task type:** Guided Discovery
- **The question:** Lopez de Prado argues that integer differencing (d=1) throws away too much information. How much memory can you actually preserve by using fractional differentiation? Is the optimal d the same for all stocks? What does the fractionally differenced series look like as a potential ML feature?
- **Data needed:** 15 years of daily log prices for 5 tickers: SPY, AAPL, TSLA, TLT, GLD
- **Tasks:**
    1. For each ticker, compute fractionally differenced series at d = 0.0, 0.1, 0.2, ..., 1.0 (11 values). Use a fixed-width fracdiff window (Lopez de Prado recommends a truncation threshold of 1e-5 for the weights).
    2. For each ticker and each d, run the ADF test and record the p-value. Plot d (x-axis) vs. ADF p-value (y-axis) for all 5 tickers on the same figure. Identify the minimum d where ADF p-value < 0.05 (the "stationarity threshold").
    3. For each ticker, compute the correlation between the fractionally differenced series (at minimum d) and the original log price series. Compare to the correlation between returns (d=1) and the original prices.
    4. Build a summary table: ticker, minimum d for stationarity, correlation with original prices at minimum d, correlation with original prices at d=1, memory gain (difference in correlations).
    5. For one ticker (SPY), plot the fractionally differenced series at d=0, d=minimum d, and d=1 side by side to visualize the progression from prices to "just stationary enough" to returns.
- **Expected insight:** Minimum d varies across tickers (typically 0.3-0.7). Equity indices like SPY may require lower d than individual volatile stocks like TSLA. The correlation with original prices at minimum d is substantially higher than at d=1 — meaning the fractionally differenced series retains meaningful level information that returns discard. This is the core value proposition: you get a stationary series (safe for ML) that remembers more about where prices have been. TLT and GLD may have different optimal d values than equities, reflecting different memory structures in bond and commodity markets.
- **Acceptance criteria:**
  - Fractional differentiation computed at all 11 d values for all 5 tickers
  - Minimum d for stationarity is between 0.1 and 0.9 for all 5 tickers
  - For each ticker: correlation at minimum d > correlation at d=1.0 (more memory preserved)
  - Memory gain (correlation difference) > 0.05 for at least 4 of 5 tickers
  - ADF p-value plot shows clear transition from > 0.05 to < 0.05 as d increases for each ticker
  - Summary table complete with all 5 tickers
  - 3-panel SPY visualization produced (d=0, d_min, d=1)

### Exercise 4: Volatility Forecasting Evaluation — How Good Is GARCH?

- **Task type:** Skill Building
- **The question:** GARCH gives you a one-step-ahead volatility forecast. How accurate is it? What's the right way to measure forecast quality for volatility, where the "true" value is itself an estimate?
- **Data needed:** 15 years of daily SPY returns. Split: first 10 years for in-sample fitting, last 5 years for out-of-sample evaluation.
- **Tasks:**
    1. Fit a GARCH(1,1) model on the first 10 years of SPY returns (in-sample period). Record parameters.
    2. Generate rolling one-step-ahead forecasts for the out-of-sample period: for each day in the last 5 years, use the fitted model to forecast tomorrow's variance. Use the `arch` library's `forecast()` method with appropriate horizon.
    3. Compute the "realized" benchmark: 21-day rolling realized variance (annualized), lagged by one day so it represents what actually happened AFTER the forecast.
    4. Evaluate the GARCH forecast using three metrics: (a) QLIKE loss (quasi-likelihood loss — the standard loss function for volatility forecasting: `QLIKE = log(sigma^2_forecast) + RV / sigma^2_forecast`), (b) MSE of conditional variance vs. squared return (the crude proxy), (c) Mincer-Zarnowitz regression: regress realized variance on forecast variance — the slope should be close to 1 and R-squared > 0 for a useful forecast.
    5. Produce a time-series plot of GARCH forecast vol vs. rolling realized vol over the out-of-sample period.
- **Expected insight:** GARCH produces genuinely useful volatility forecasts — R-squared in the Mincer-Zarnowitz regression is typically 0.15-0.40 for daily conditional variance vs. realized variance. Not perfect, but far better than a constant-volatility model. The forecast tracks the broad regime (high vol during crises, low vol during calm), but lags on sudden transitions. QLIKE is the preferred loss function because it penalizes both underestimation and overestimation in a way that's natural for positive-valued targets. MSE of variance is dominated by extreme observations and is a less stable metric. Students should appreciate that volatility forecasting is FEASIBLE (unlike return forecasting, which is nearly hopeless for daily data) — this is the fundamental asymmetry that makes GARCH useful.
- **Acceptance criteria:**
  - GARCH model fit on in-sample period converges
  - Out-of-sample forecast generated for at least 1000 trading days
  - Mincer-Zarnowitz R-squared > 0.05 (GARCH has genuine forecasting power — typically 0.15-0.40)
  - Mincer-Zarnowitz slope > 0 (forecast moves in the right direction)
  - QLIKE loss computed and reported as a single number
  - Time-series plot shows GARCH forecast tracking broad volatility regimes in the out-of-sample period

## Homework Outline

### Mission Framing

You're two weeks into your role at a quantitative fund. Last week you built the data pipeline. This week the head of research drops by your desk: "We need a volatility analysis toolkit. Every new research project starts with characterizing the volatility of the assets we're trading — are they clustered? persistent? asymmetric? What's the best model for each? And I keep hearing about fractional differentiation for ML features — can you build something that finds the optimal d for any ticker?"

This isn't a one-off analysis. It's a toolkit your researchers will use every time they onboard a new asset or strategy. The deliverables should be modular, reusable, and well-documented — not scripts that work once on SPY and break on everything else.

The homework integrates everything from the lecture and seminar into a production-grade analysis pipeline. Where the seminar exercises explored concepts one at a time, the homework combines them into a coherent workflow: characterize the time series, select the right volatility model, evaluate its forecasts, and produce ML-ready features via fractional differentiation.

### Deliverables

1. **A `VolatilityAnalyzer` class**
   - **Task type:** Construction
   - Build a reusable class that takes a return series (or prices + a flag to compute returns) and produces a comprehensive volatility analysis:
     - **Stationarity diagnostics:** Run ADF and KPSS on both prices and returns. Report test statistics, p-values, and a joint diagnosis (stationary / non-stationary / ambiguous). Handle edge cases (series too short, constant series).
     - **Stylized fact verification:** Compute kurtosis, skewness, Ljung-Box test on squared returns (for ARCH effects), Jarque-Bera test for normality. Return a dictionary of results with pass/fail flags against stylized-fact expectations (kurtosis > 3, significant Ljung-Box, etc.).
     - **GARCH model fitting:** Fit GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1). Return fitted parameters, log-likelihood, AIC, BIC. Select the best model by BIC. Handle convergence failures gracefully (skip the failed model, report which ones failed, proceed with the rest).
     - **Volatility forecasting:** Generate one-step-ahead conditional volatility from the best model. Compute rolling realized volatility at a configurable horizon. Return both series aligned by date.
   - **Acceptance criteria:**
     - Class instantiates and runs on at least 10 different tickers without crashing
     - Stationarity diagnostics return correct results for known cases (prices: non-stationary, returns: stationary)
     - Stylized facts dictionary includes at least: kurtosis, skewness, ljung_box_pvalue, jarque_bera_pvalue, with correct types (floats)
     - GARCH fitting handles convergence failure for at least one model variant without crashing (test with a deliberately short or odd series)
     - Best model selected by BIC matches manual comparison of the three BIC values
     - Conditional volatility series has the same length as the input return series (minus any initial burn-in)
     - Class works on both single-ticker Series and multi-ticker DataFrames (or clearly documents its input contract)

2. **A multi-asset volatility comparison report**
   - **Task type:** Skill Building
   - Run your `VolatilityAnalyzer` on 10 diverse tickers (spanning equities, ETFs, bonds, and commodities proxies). Produce:
     - A comparison table: ticker, kurtosis, skewness, ARCH effect (yes/no), best GARCH model, persistence, annualized long-run vol
     - A panel figure (2x5 grid): conditional volatility from the best model for each ticker, with absolute returns overlaid. All panels on the same y-axis scale so cross-asset vol comparisons are visual and immediate.
     - A brief summary (programmatic, not prose): which tickers have the highest persistence? The strongest leverage effect? The fattest tails? Any surprises (e.g., bonds behaving differently from equities)?
   - **Acceptance criteria:**
     - Analysis completed for all 10 tickers
     - Comparison table has all specified columns, no missing values
     - Panel figure has 10 subplots with matched y-axes
     - At least one bond or commodity ticker shows materially different GARCH characteristics (lower persistence, weaker leverage) than equity tickers
     - All persistence values are in [0, 1) (stationary GARCH)
     - Long-run annualized vol is between 5% and 80% for all tickers (sanity check: no obviously wrong values)

3. **A fractional differentiation feature builder**
   - **Task type:** Construction
   - Build a function (or method on your `VolatilityAnalyzer`) that:
     - Takes a price series and finds the minimum fractional differentiation order d (to 0.05 precision) that achieves stationarity (ADF p-value < 0.05)
     - Uses a binary-search or grid-search over d in [0, 1] with the fracdiff operation
     - Returns the optimal d, the fractionally differentiated series, and a diagnostic dictionary (ADF p-value at optimal d, correlation with original series, correlation of d=1 series with original)
     - Handles edge cases: series already stationary at d=0 (return d=0), series not stationary even at d=1 (return d=1 with a warning)
   - Run this on 10 tickers and produce a table: ticker, optimal d, correlation at optimal d, correlation at d=1, memory gain.
   - **Acceptance criteria:**
     - Optimal d found for all 10 tickers
     - All optimal d values in [0, 1]
     - Fractionally differenced series at optimal d passes ADF test (p-value < 0.05) for all tickers
     - Correlation at optimal d > correlation at d=1.0 for at least 8 of 10 tickers (memory preserved)
     - Function runs in < 30 seconds per ticker (not computationally prohibitive)
     - Edge case: if a ticker's prices are already stationary at d=0, function returns d=0 (not an error)

4. **A GARCH forecast evaluation pipeline**
   - **Task type:** Investigation (Baseline + Smarter layers)
   - **Baseline (Layer 1):** Implement a proper out-of-sample GARCH forecast evaluation:
     - Split data: first 70% in-sample, last 30% out-of-sample
     - Fit GARCH(1,1) on in-sample data
     - Generate rolling one-step-ahead forecasts for the out-of-sample period
     - Evaluate using: QLIKE loss, MSE of variance, Mincer-Zarnowitz regression (slope, intercept, R-squared)
     - Run on 5 tickers. Report metrics in a clean comparison table.
   - **Smarter (Layer 2):** Compare GARCH(1,1) to a simple rolling-window realized volatility forecast (21-day rolling vol as the forecast for tomorrow). Use the same evaluation metrics. Answer: does GARCH actually beat the naive estimator? By how much? For which assets?
   - Report which model wins on QLIKE for each ticker and by what margin. Discuss: when is the added complexity of GARCH justified?
   - **Acceptance criteria:**
     - Out-of-sample forecasts generated for all 5 tickers in both models
     - Mincer-Zarnowitz R-squared > 0 for GARCH on at least 4 of 5 tickers (genuine forecasting power)
     - QLIKE computed for both GARCH and rolling-window models on all 5 tickers
     - Comparison table includes all metrics for both models side by side
     - At least one ticker shows GARCH winning over rolling-window, and the discussion addresses why
     - Layer 2 comparison clearly identifies conditions where GARCH adds value over the naive estimator (e.g., regime transitions, post-shock periods)

### Expected Discoveries

- Building a reusable `VolatilityAnalyzer` reveals that not all tickers behave the same: the "best" GARCH variant changes depending on the asset class. Students will find that the vanilla GARCH(1,1) is rarely the best choice for equities — the leverage effect is too strong to ignore.
- The multi-asset comparison will show that volatility persistence is near-universal (alpha + beta > 0.93 for equities), but the strength of asymmetry and the level of long-run volatility vary systematically by sector and asset class. Bonds and gold behave measurably differently from equities.
- Fractional differentiation reveals that most equities need d between 0.3 and 0.7 for stationarity — meaning integer differencing (d=1) discards 30-70% of the available memory. The optimal d correlates loosely with how "trending" the asset is (lower d for mean-reverting assets, higher d for trending ones).
- In the forecast evaluation, GARCH(1,1) does better than the naive rolling-window estimator on QLIKE for most equities, but the margin is modest. The biggest advantage comes during regime transitions (entering/exiting crises), where GARCH's parametric structure adapts faster than a backward-looking rolling window. For very stable assets, the rolling window may be sufficient.
- The Mincer-Zarnowitz R-squared is modest but positive (typically 0.15-0.40) — volatility is predictable in a way that returns are not. This asymmetry (forecastable variance, unforecastable mean) is the most important stylized fact for ML practitioners entering finance.

## Key Papers & References

- **Cont, R. (2001), "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues," *Quantitative Finance*** — The single best reference on what financial returns look like empirically. 11 stylized facts, clean presentation, universally cited. Start here.
- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity," *Journal of Econometrics*** — The GARCH paper. Three parameters, a Nobel-adjacent contribution. Read it to understand why such a simple model endured.
- **Engle, R. (2001), "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics," *Journal of Economic Perspectives*** — The Nobel laureate's own accessible tutorial. Written for economists, friendly to ML practitioners.
- **Lopez de Prado, M. (2018), "Advances in Financial Machine Learning," Chapter 5** — Fractional differentiation for financial ML features. The practical recipe: find minimum d, preserve memory, feed to ML. The most directly actionable reference for this week.
- **Nelson, D. (1991), "Conditional Heteroskedasticity in Asset Returns: A New Approach," *Econometrica*** — The EGARCH paper. Introduced log-volatility modeling and the asymmetric news impact curve. Foundational for understanding leverage effects.
- **Glosten, L., Jagannathan, R., Runkle, D. (1993), "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks," *Journal of Finance*** — The GJR-GARCH paper. The explicit "bad news indicator" approach to asymmetry.
- **Tsay, R. (2010), "Analysis of Financial Time Series," Chapters 1-3** — Comprehensive textbook treatment of return properties, ARMA, and GARCH. The reference text for deep dives.
- **Patton, A. (2011), "Volatility Forecast Comparison Using Imperfect Volatility Proxies," *Journal of Econometrics*** — Why QLIKE is the right loss function for volatility forecasting when the "true" volatility is a proxy. Essential reading for understanding the forecast evaluation in Deliverable 4.

## Career Connections

- **Quantitative researcher (buy-side):** At firms like Two Sigma, DE Shaw, or Citadel, the first step in any new research project is characterizing the target assets' statistical properties — stationarity, volatility regime, distributional shape. The `VolatilityAnalyzer` from this week's homework is a stripped-down version of what a quant researcher runs before writing a single model. The decision between GARCH(1,1) and EGARCH is not academic — it determines which risk model the portfolio uses, which directly affects position sizing and P&L.
- **Risk analyst / risk quant:** Risk teams at multi-strategy funds (Millennium, Balyasny, Point72) run GARCH-based VaR and Expected Shortfall calculations daily for every book in the fund. The volatility forecast feeds directly into position limits — if your GARCH model says vol is spiking, the risk system automatically reduces allowable exposure. Understanding GARCH parameters (especially persistence and asymmetry) is daily operational knowledge, not textbook theory.
- **ML engineer / data scientist at a fund:** Fractional differentiation is used in production feature engineering pipelines at systematic funds. When building ML features from price data, the choice between d=0 (prices), d=1 (returns), and d=0.4 (fractionally differenced) directly impacts model performance. A data scientist who understands WHY d matters — and can explain the stationarity-memory tradeoff to a PM — is more valuable than one who just applies standard differencing because a textbook said to.
- **Quant trader:** Traders at vol-focused firms (Optiver, IMC, Susquehanna) use conditional volatility estimates to size trades and price options. A GARCH conditional vol that's 20% above realized vol is a trading signal — implied vol may be mispriced. Understanding the gap between model-implied and realized volatility is the core of volatility trading, and it starts with the exact analysis this week covers.

## Data Sources

- **yfinance** (free, no API key): Daily adjusted close prices for all tickers used this week. Download 15+ years of history to have enough data for stable GARCH estimation and meaningful out-of-sample evaluation. Use the same `FinancialDataLoader` from Week 1's homework or download directly with `yf.download()`.
- **Tickers used:** SPY, QQQ, IWM, TLT, GLD (ETFs), AAPL, MSFT, JPM, JNJ, TSLA, NVDA (large-cap equities), MARA (high-vol), GE, INTC, BA, PFE, DIS (mid-cap / special cases), XLE, XLF, XLK (sector ETFs). Not all exercises use all tickers — each exercise specifies its subset.
- **Approximate data sizes:** 20 tickers x 15 years x 252 days = ~75,600 rows in long format. Download time via yfinance: < 1 minute. CSV: ~5 MB. Parquet: ~1 MB.
- **Key libraries:** `arch` (v8.0+ for GARCH family), `statsmodels` (ADF, KPSS, ARMA, Ljung-Box, ACF/PACF), `fracdiff` or manual implementation for fractional differentiation, `matplotlib` for all plotting, `numpy`/`pandas` for computation.
- **Professional alternatives (mentioned, not required):** Oxford-Man Institute Realized Library (open, high-frequency realized measures for major indices), WRDS (academic — tick-level data for precise RV computation). These provide the intraday data needed for high-frequency realized volatility, which is beyond this week's scope but relevant in Week 8.