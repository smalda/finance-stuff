# Week 3: Factor Models & Cross-Sectional Analysis

**Band:** 1
**Implementability:** HIGH

## Learning Objectives

After this week, students will be able to:

1. Explain the economic intuition behind systematic risk factors (size, value, profitability, investment, momentum) and why certain firm characteristics predict cross-sectional stock returns
2. Build factor portfolios from scratch using the portfolio-sorting methodology — downloading fundamental data, computing firm characteristics, ranking stocks into quantiles, and constructing long-short factor returns
3. Implement and interpret Fama-MacBeth two-step cross-sectional regression to test whether factors are priced in the cross-section of returns
4. Distinguish between the Fama-French (portfolio-sorting) and Barra (cross-sectional regression) approaches to factor modeling and explain when each is used in practice
5. Validate student-constructed factors by comparing them to official Ken French factors and diagnosing discrepancies (correlation > 0.95 as success criterion)
6. Compute factor exposures (betas) for individual stocks and portfolios via time-series regression and decompose portfolio returns into systematic and idiosyncratic components
7. Apply out-of-sample validation methodology to factor models, diagnose the "factor zoo" problem, and explain why economic intuition matters as much as statistical significance
8. Construct a cross-sectional feature matrix from fundamental data (book-to-market, size, profitability, asset growth) that will serve as input to ML alpha models in Week 4

## Prerequisites

- **Required:** Week 1 (data downloading with yfinance, return computation, DataFrame manipulation, survivorship bias awareness, corporate action handling), Week 2 (time-series regression intuition, stationarity, panel data structure)
- **Assumed ML knowledge:** Linear regression (coefficients, R², t-statistics, p-values), hypothesis testing, regularization concepts (Ridge/LASSO intuition), cross-validation mechanics. Students do NOT know financial econometrics (Fama-MacBeth, panel regression) or factor model theory — these are introduced from scratch this week.
- **Prerequisite explanations in this week:** Fundamental data structure (15 min): What is a balance sheet? Income statement? Key fields (total equity, net income, total assets, book value per share). How quarterly vs. annual data works. How to align fundamental data with price data at the correct point-in-time (financial statements are reported with a lag — Q4 2023 earnings might not be public until February 2024). Brief intro to cross-sectional regression as distinct from time-series regression — regressing all assets' returns on their characteristics at each time period, separately for each month, then averaging coefficients over time. This is the Fama-MacBeth logic.

## Narrative Arc

Week 2 answered "how does ONE financial time series behave?" You learned that returns are fat-tailed, volatility clusters, and GARCH models capture conditional variance. But real portfolios hold many assets, and understanding a portfolio's risk requires understanding how assets move TOGETHER. Week 3 shifts from single-series analysis to CROSS-SECTIONAL analysis — comparing hundreds of stocks at the same moment in time.

The central question is: **"Why do stocks move together, and what drives cross-sectional differences in returns?"** The efficient market hypothesis says all securities should earn the same risk-adjusted return. Reality violates this spectacularly: small-cap stocks outperform large-caps over long periods, "value" stocks (high book-to-market) beat "growth" stocks, profitable firms outperform unprofitable ones. These patterns — called "factor premia" — have persisted for decades and form the foundation of quantitative equity investing. But there's a catch: hundreds of proposed factors exist in academic literature (the "factor zoo"), most are data-mined, and many disappear out-of-sample.

This week resolves the tension between "factors are the most important concept in quantitative finance" and "most published factors are statistical noise" through a disciplined empirical approach. First, you'll learn the canonical factors (CAPM, Fama-French 3/5-factor) and their economic stories — WHY should small-caps outperform? What risk does value exposure compensate you for? Then you'll build these factors from scratch using real fundamental data, validate them against authoritative benchmarks, and implement the statistical machinery (Fama-MacBeth regression) that tests whether a factor is genuinely priced. By the end, you'll understand that factor models are simultaneously:
1. The industry's standard risk decomposition framework (every portfolio manager talks about "factor exposures")
2. The feature engineering toolkit for ML-based alpha models (Week 4 will use these exact firm characteristics as inputs)
3. A cautionary tale about data mining and the replication crisis in empirical finance

The week closes with a practical validation workflow: build a factor → test it in-sample → validate it out-of-sample → compare construction methods (Fama-French vs. Barra) → connect to feature engineering for ML. This is the bridge from classical finance to modern quantitative research.

## Lecture Outline

### Section 1: The CAPM — The Simplest Factor Model

- **Hook:** "The Capital Asset Pricing Model says only one thing matters: how much does a stock move with the market? If you diversify perfectly, you eliminate all risk except market risk — and the market compensates you for that risk with the equity risk premium. This is the intellectual foundation for passive investing, the beta coefficient, and the entire factor model literature. It's also spectacularly wrong as a complete description of returns. But it's the right place to start."
- **Core concepts:** The CAPM equation: `E[R_i] = R_f + beta_i * (E[R_m] - R_f)`, where `R_i` is the return on asset i, `R_f` is the risk-free rate, `R_m` is the market return, and `beta_i` measures asset i's sensitivity to market movements. Beta as systematic risk — the risk you can't diversify away. The market portfolio as the single factor. The intuition: risk-averse investors demand compensation for bearing systematic risk; idiosyncratic risk is diversifiable and earns no premium. The empirical test: regress individual stock returns on the market return (time-series regression) to estimate beta, then test whether high-beta stocks earn higher returns than low-beta stocks (cross-sectional prediction). Historical result: the CAPM fails spectacularly in cross-section — the beta-return relationship is far weaker than the theory predicts, and OTHER characteristics (size, value) predict returns better than beta. This failure motivated the factor model revolution.
- **Key demonstration:** Download 10 years of daily returns for SPY (market proxy) and 10 individual stocks spanning small-cap, large-cap, value, and growth. For each stock, run a time-series regression: `R_i,t = alpha_i + beta_i * R_m,t + epsilon_i,t`. Display betas. A defensive utility (e.g., DUK) should have beta < 1; a high-vol tech stock (e.g., NVDA) should have beta > 1. Then produce a scatter plot: x-axis = estimated beta, y-axis = average realized return. The CAPM predicts a positive relationship — high beta → high return. Empirically, the relationship is noisy at best. Compute the cross-sectional correlation and R² — typically R² < 0.10, meaning beta explains almost none of the variation in average returns across stocks.
- **Acceptance criteria:**
  - Beta estimates computed for all 10 stocks via time-series regression
  - At least one stock has beta < 0.8 (defensive), at least one has beta > 1.2 (aggressive)
  - Time-series R² for each regression > 0.10 (beta captures some co-movement with market)
  - Cross-sectional R² (beta predicting average returns) < 0.20 (CAPM fails cross-sectionally). Note: R² depends on market regime and sample period — the 2004-2023 period includes the post-GFC tech rally, which weakens the beta-return relationship further. In earlier decades (1960s-1990s), R² could reach 0.10-0.15. The key lesson holds regardless: beta alone explains very little of the cross-section of average returns.
  - Scatter plot shows positive but weak relationship between beta and average returns
- **Bridge:** "Beta is not enough. The CAPM leaves enormous variation in returns unexplained. In the early 1990s, Fama and French documented two additional factors that dominate beta in explaining cross-sectional returns: size and value."

### Section 2: Fama-French 3-Factor — Size and Value Join the Party

- **Hook:** "In 1992, Fama and French published a paper that shook empirical finance: 'The Cross-Section of Expected Stock Returns.' They showed that two firm characteristics — market capitalization (size) and book-to-market ratio (value) — explain returns far better than beta. Small-cap stocks outperform large-caps. High book-to-market ('value') stocks outperform low book-to-market ('growth') stocks. These patterns persist across decades and countries. The question is: are they risk factors or market anomalies?"
- **Core concepts:** The three Fama-French factors: (1) Market (MKT or Rm-Rf) — same as CAPM, (2) SMB (Small Minus Big) — the return difference between small-cap and large-cap portfolios, (3) HML (High Minus Low) — the return difference between high book-to-market and low book-to-market portfolios. The Fama-French 3-factor model: `R_i,t - R_f,t = alpha_i + b_i,MKT * MKT_t + b_i,SMB * SMB_t + b_i,HML * HML_t + epsilon_i,t`. Factor construction via portfolio sorting: (1) At the end of each month, rank all stocks by market cap (or book-to-market), (2) Split into quantiles (typically deciles or terciles — Fama-French uses 2x3 sorts), (3) Form portfolios (equal-weight or value-weight), (4) Compute portfolio returns next month, (5) SMB = (small-cap portfolio return) - (large-cap portfolio return), HML = (high B/M portfolio return) - (low B/M portfolio return). Economic intuition: SMB captures size risk — small firms are riskier than large firms (higher bankruptcy risk, less liquidity). HML captures distress risk — high B/M firms are distressed (book value high relative to market value means the market expects trouble); the value premium compensates investors for holding these risky firms. Alternative story: these are behavioral anomalies (overreaction to past performance), not risk premia. The debate continues, but the empirical patterns are robust.
- **Key demonstration:** Explain the Fama-French portfolio-sorting methodology with a worked example on a small dataset (100 stocks, 1 month). Show the steps: download market caps and book-to-market ratios, rank stocks, split into terciles (small/medium/big, value/neutral/growth), form 6 portfolios (2x3 sort), compute equal-weight portfolio returns. Then show the factors: SMB = average of (small-value, small-neutral, small-growth) minus average of (big-value, big-neutral, big-growth). HML = average of (small-value, big-value) minus average of (small-growth, big-growth). This is the "6 portfolios formed on size and book-to-market" methodology from Ken French's website. Emphasize: factors are PORTFOLIOS, not abstract statistical constructs. They're tradeable — you can actually buy the small-cap portfolio and short the large-cap portfolio to earn SMB.
- **Acceptance criteria:**
  - Sorting methodology demonstrated on >= 50 stocks for a single month
  - Exactly 6 portfolios formed (small/value, small/neutral, small/growth, big/value, big/neutral, big/growth)
  - SMB and HML computed using the exact Fama-French weighting scheme (average across value/neutral/growth for SMB, average across small/big for HML)
  - SMB has economically reasonable sign (typically positive — small outperforms large historically). Caveat: in the 2004-2023 S&P 500 sample, SMB may be near-zero or slightly negative due to large-cap tech dominance; the long-run positive premium is a full-universe (NYSE/AMEX/NASDAQ) result over 1926-present.
  - HML has economically reasonable sign (typically positive — value outperforms growth historically). Caveat: HML has been negative in recent decades (the "death of value" post-2007); in a 2004-2023 sample, negative HML is expected and pedagogically valuable — it illustrates that factor premia are not guaranteed.
  - Portfolio returns are in decimal format (not percentages) and have realistic magnitudes (monthly returns of -10% to +10%)
- **Bridge:** "SMB and HML capture size and value effects. But Fama and French didn't stop there. In 2015 they added two more factors — profitability and investment — based on new empirical evidence."

### Section 3: Fama-French 5-Factor — Profitability and Investment

- **Hook:** "The 3-factor model worked well for 20 years. Then researchers noticed two more patterns: profitable firms outperform unprofitable firms (controlling for everything else), and firms that grow assets aggressively (high investment) underperform conservative firms. The 5-factor model adds RMW (Robust Minus Weak — profitability) and CMA (Conservative Minus Aggressive — investment). The model's explanatory power jumps from R² ≈ 0.90 to R² ≈ 0.95. But there's a twist: HML becomes nearly redundant — the value premium is largely explained by profitability and investment."
- **Core concepts:** The two new factors: (1) RMW (Robust Minus Weak) — the return difference between high-profitability and low-profitability portfolios, where profitability = operating income / book equity, (2) CMA (Conservative Minus Aggressive) — the return difference between low-investment and high-investment portfolios, where investment = year-over-year change in total assets / lagged total assets. The 5-factor model: same as 3-factor plus `+ b_i,RMW * RMW_t + b_i,CMA * CMA_t`. Economic intuition: RMW — profitable firms are safer and more attractive; the market rewards profitability. CMA — firms that grow assets rapidly (via acquisitions, capex) often destroy value; conservative firms that don't overinvest earn higher returns. The "value is subsumed" finding: in the 5-factor model, HML often has insignificant coefficients because RMW and CMA capture the underlying sources of the value premium — value stocks tend to be unprofitable and/or conservative in investment. Practical implication: the 5-factor model is NOW the academic standard, but many practitioners still use the 3-factor model because it's simpler and HML has a cleaner interpretation.
- **Key demonstration:** Show how to compute RMW and CMA from fundamental data. Use yfinance to download `income_stmt` (for operating income) and `balance_sheet` (for total assets and total equity) for 100 stocks. Compute profitability = operating income / book equity. Compute investment = (total assets this year - total assets last year) / total assets last year. Rank stocks by profitability → form terciles → long high-profitability, short low-profitability → RMW. Same for investment (but NOTE: CMA is conservative minus aggressive, so LOW investment is the long leg). Show a side-by-side table: SMB, HML, RMW, CMA for a single month. Explain the signs: RMW should be positive (profitable > unprofitable), CMA should be positive (conservative > aggressive). Show that sorting on profitability or investment AFTER controlling for size and value is nontrivial — you need a 2x3x3 sort (size x value x profitability) or independent sorts. Full implementation is complex; the demonstration shows the intuition.
- **Acceptance criteria:**
  - Profitability and investment computed from fundamental data for >= 50 stocks
  - Profitability = operating income / book equity (dimensionless ratio)
  - Investment = (total assets_t - total assets_t-1) / total assets_t-1 (growth rate)
  - RMW computed as high-profitability portfolio return minus low-profitability portfolio return
  - CMA computed as low-investment portfolio return minus high-investment portfolio return (note the sign!)
  - RMW has plausible sign (typically positive)
  - CMA has plausible sign (typically positive). Caveat: CMA is the weakest of the FF5 factors in recent data; with S&P 500 stocks (which skew toward large, established firms), CMA may be near-zero or slightly negative. The positive CMA premium is strongest in the full-universe long-run sample.
  - Data handling: missing values (stocks with no operating income or assets) are dropped, not filled with zeros
- **Bridge:** "The 5-factor model is the portfolio-sorting approach — Fama and French's signature methodology. But there's a completely different way to construct factors: cross-sectional regression. This is the Barra approach, and it's what risk models in production use."

### Section 4: Cross-Sectional Regression — The Barra Approach

- **Hook:** "Fama-French builds factors by forming portfolios. Barra builds factors by running regressions. The end goal is the same — decompose returns into systematic and idiosyncratic components — but the mechanics are totally different. Understanding both approaches is essential because you'll see 'Fama-French' in academic papers and 'Barra risk model' in job descriptions at asset managers."
- **Core concepts:** The Barra methodology: at each time period t, regress ALL assets' returns on ALL characteristics simultaneously: `R_i,t = b_0,t + b_1,t * size_i,t + b_2,t * value_i,t + b_3,t * momentum_i,t + ... + epsilon_i,t`, where `size_i,t` is stock i's market cap at time t (standardized), `value_i,t` is its book-to-market, etc. The coefficients `b_1,t, b_2,t, ...` are the FACTOR RETURNS at time t — they tell you how much the market rewarded (or punished) size exposure that month. The key distinction from Fama-French: (1) Factors are regression coefficients, not portfolio returns, (2) You're using characteristics DIRECTLY as exposures (not sorting into portfolios), (3) You estimate everything jointly in one multivariate regression per period (not sequential univariate sorts), (4) The output is a time series of factor returns + a covariance matrix of factors (the core of a risk model). Barra risk models also include INDUSTRY factors (dummy variables for each sector — financials, tech, healthcare, etc.), which Fama-French ignores. The full Barra USE4 model has ~70 industry factors + ~10 style factors + country factors. Why practitioners use this: (1) Risk decomposition — you can attribute portfolio return to size, value, industry, and specific stock picks, (2) Risk forecasting — the factor covariance matrix predicts portfolio volatility, (3) Portfolio construction — you can explicitly control factor exposures (be neutral to value, tilt toward momentum, avoid energy sector).
- **Key demonstration:** Implement a simplified "Barra-style" cross-sectional regression for a single month using 100 stocks. Regress the month's returns on standardized characteristics (size, book-to-market, momentum). Display the coefficients — these are the factor returns for that month. Compare to the Fama-French portfolio-sorting approach on the same data and same characteristics. Key insight: the TWO METHODS give similar (but not identical) factor returns. Correlation between Barra-style size factor and Fama-French SMB is typically 0.70-0.90 — they're capturing the same underlying phenomenon but with different statistical machinery. The Barra approach is more flexible (you can add any characteristic as a factor, including interactions) but less transparent (you're not forming actual tradeable portfolios).
- **Acceptance criteria:**
  - Cross-sectional regression run on >= 100 stocks for at least one month
  - Regression includes at least 3 factor exposures: size, value, momentum (standardized to mean 0, std 1)
  - Regression coefficients (factor returns) have plausible magnitudes (monthly returns of -5% to +5%)
  - Regression R² reported and is > 0.03 (some cross-sectional explanatory power). Note: cross-sectional R² is inherently low in single-month regressions — individual months are dominated by idiosyncratic noise. The R² captures the AVERAGE explanatory power of characteristics across stocks in one month, which is typically 3-8% for size/value/momentum. Pooled or time-averaged R² is much higher.
  - Side-by-side comparison: Barra-style size coefficient vs. Fama-French SMB for the same month
  - Qualitative discussion: "Both methods capture the size effect, but Barra uses continuous exposures while FF uses discrete portfolio sorts"
- **Bridge:** "You've seen two factor construction methods. Now the question: how do you TEST whether a factor is actually priced? The Fama-MacBeth regression is the canonical statistical test for this."

### Section 5: Fama-MacBeth Regression — Testing If Factors Are Priced

- **Hook:** "Here's the core question in empirical asset pricing: does exposure to a factor predict returns? If you hold a high-beta stock, do you earn a higher return on average? If you're exposed to the value factor, do you get compensated? The Fama-MacBeth two-step regression is the gold-standard methodology for answering this. It's 50 years old and still the dominant approach in academic finance."
- **Core concepts:** The two-step procedure: **Step 1 (time-series):** For each asset i, run a time-series regression of returns on factors: `R_i,t = alpha_i + beta_i,MKT * MKT_t + beta_i,SMB * SMB_t + ... + epsilon_i,t`. This gives you factor loadings (betas) for each stock. **Step 2 (cross-sectional):** For each time period t, run a cross-sectional regression of returns on the betas from Step 1: `R_i,t = lambda_0,t + lambda_MKT,t * beta_i,MKT + lambda_SMB,t * beta_i,SMB + ... + eta_i,t`. The coefficients `lambda_t` are the RISK PREMIA — they tell you how much return you earn per unit of factor exposure that month. **Aggregation:** Average the lambdas over time: `mean(lambda_MKT)`, `mean(lambda_SMB)`, etc. Test if the average is significantly different from zero using a t-test. If yes, the factor is "priced" — exposure to it earns a significant risk premium. The intuition: Step 1 measures each asset's exposure to the factor (via time-series covariation). Step 2 tests whether that exposure predicts returns in the cross-section (do high-exposure stocks earn more?). Why the two-step structure matters: standard pooled OLS on panel data produces biased standard errors because residuals are correlated across assets and time. Fama-MacBeth's cross-sectional regressions naturally account for cross-sectional correlation. The limitation: errors-in-variables problem — the betas from Step 1 are estimates, not true values, which biases Step 2 standard errors downward (makes factors look more significant than they are). Shanken correction adjusts for this, but it's rarely applied in practice.
- **Key demonstration:** Implement Fama-MacBeth on a small universe (50 stocks, 5 years of monthly returns). Step 1: For each stock, regress its returns on MKT, SMB, HML (use official Ken French factors downloaded via `getfactormodels`). Extract betas. Step 2: For each month, regress that month's returns on the betas from Step 1. Extract lambdas. Aggregation: Compute `mean(lambda_MKT)`, `mean(lambda_SMB)`, `mean(lambda_HML)` and t-statistics. Display a summary table: Factor, Mean Risk Premium, t-stat, p-value. Interpretation: if `mean(lambda_SMB)` > 0 and t-stat > 2, SMB is priced — small-cap exposure earns a positive risk premium. Also show the implementation using the `linearmodels` library's `FamaMacBeth` class for comparison — it automates the two-step procedure and computes Newey-West standard errors (which correct for autocorrelation in the lambda time series).
- **Acceptance criteria:**
  - Step 1: Factor loadings (betas) estimated for >= 50 stocks using time-series regression
  - Step 2: Cross-sectional regression run for each month in the sample (>= 60 months)
  - Lambda time series has the same length as the number of months
  - Mean lambda computed and t-statistic calculated for each factor
  - At least one factor (MKT or SMB or HML) has |t-stat| > 2.0 (statistically significant risk premium)
  - Summary table includes: Factor name, Mean lambda, Std error, t-stat, p-value
  - `linearmodels.FamaMacBeth` implementation shown as validation (results should be similar to manual implementation)
- **Bridge:** "You now know how to build factors and test if they're priced. But there's a problem: the academic literature has proposed HUNDREDS of factors. Most don't replicate. This is the factor zoo."

### Section 6: The Factor Zoo Problem — Why Most Factors Are Noise

- **Hook:** "In 2016, Harvey, Liu, and Zhu published 'And the Cross-Section of Expected Returns.' They catalogued 316 proposed factors from academic papers published since 1970. Most have never been replicated. Many disappear out-of-sample. Some were data-mined. The factor zoo is finance's replication crisis — and it's a warning about what happens when you torture data long enough."
- **Core concepts:** The problem: With enough computing power and researcher degrees of freedom (which transformations? which sample periods? which weighting schemes?), you can find a "significant" factor by chance. The multiple testing issue: if you test 100 characteristics and use a 5% significance threshold, you expect 5 false positives. The factor zoo papers test 300+ characteristics — dozens will look significant by pure luck. The publication bias: journals favor novel, significant results; null results rarely get published. The post-publication decay: once a factor is published and widely known, it often weakens or disappears (the "alpha decay" problem — traders arbitrage it away, or it was never real). The solutions: (1) Out-of-sample validation — test the factor on a different time period or different geography, (2) Economic intuition — why SHOULD this characteristic predict returns? What's the risk story? (3) Multiple testing correction — adjust p-values for the number of tests (Bonferroni correction, false discovery rate), (4) Higher significance thresholds — Harvey-Liu-Zhu recommend t-stat > 3.0 (not 2.0) for new factors, (5) Parsimony over complexity — a model with 5 intuitive factors beats a model with 50 data-mined factors. The practical takeaway: Stick to factors with (a) decades of evidence, (b) clear economic intuition, (c) robustness across markets and time periods. The Fama-French 5 factors, momentum, and liquidity are the "core set" that meet this bar. Everything else is speculative.
- **Key demonstration:** Show a factor validation workflow on a toy example. Create a synthetic dataset with 50 random "factors" (just Gaussian noise) and 100 stocks over 10 years. For each "factor," run a Fama-MacBeth regression and test if it's priced. Count how many factors appear significant at the 5% level. You'll get ~2-3 by pure chance (5% of 50 = 2.5). This is the false positive problem. Then show a real-world example: download 10 characteristics from real stock data (market cap, book-to-market, momentum, profitability, investment, leverage, liquidity, volatility, earnings yield, dividend yield). Split the sample: 2000-2015 (in-sample), 2016-2023 (out-of-sample). For each characteristic, compute its Fama-MacBeth t-stat in-sample. Then compute its t-stat out-of-sample. Many factors that looked significant in-sample will have t-stats near zero out-of-sample. The correlation between in-sample t-stat and out-of-sample t-stat is typically only ~0.30-0.50 — evidence that many "discoveries" are overfitting.
- **Acceptance criteria:**
  - Synthetic example: 50 random "factors" tested, at least 1 appears significant at 5% level by chance
  - Real-world example: 6 canonical factors (MKT, SMB, HML, RMW, CMA, MOM) tested using official Ken French data with IS (1963-1999) vs. OOS (2000-2023) split. This uses the natural historical split: FF3 was published in 1993 using pre-1990s data, so 2000+ is genuine out-of-sample.
  - Data split: at least 10 years in-sample, at least 3 years out-of-sample
  - In-sample t-stats computed for all factors
  - Out-of-sample t-stats computed for all factors
  - At least one factor degrades from significant IS to not significant OOS (demonstrates that even well-established factors can lose power)
  - At least one factor remains significant out-of-sample (some risk premia are genuine and persistent)
  - Discussion: "Most factors don't replicate. Economic intuition and out-of-sample validation are essential."
- **Bridge:** "The factor zoo is a cautionary tale. But the validated factors — size, value, momentum, profitability, investment — are real and useful. Now let's implement the full pipeline: build factors from scratch, validate them, and prepare them as features for ML models."

### Section 7: Building Factors from Scratch — The Full Pipeline

- **Hook:** "You've seen the theory. You've seen Ken French's official factors. Now it's your turn: download fundamental data for 500 stocks, compute firm characteristics, sort into portfolios, construct factor returns, and compare your results to the official benchmarks. If your correlation with Ken French's SMB is > 0.95, you did it right. If it's < 0.80, something's wrong — and diagnosing what went wrong is where the learning happens."
- **Core concepts:** The data pipeline: (1) Download prices and market caps from yfinance (daily, 10+ years), (2) Download fundamental data from yfinance: `balance_sheet` (for total equity, total assets), `income_stmt` (for operating income, net income), (3) Compute characteristics: size = market cap, value = book equity / market cap (B/M ratio), profitability = operating income / book equity, investment = YoY change in total assets / lagged total assets, momentum = past 12-month return (skipping the most recent month), (4) At the end of each month, rank all stocks by each characteristic, (5) Form quantile portfolios (typically terciles for 2x3 sorts or deciles for single sorts), (6) Compute portfolio returns for the next month (equal-weight or value-weight), (7) Construct factor returns: SMB = small portfolio - big portfolio, HML = high B/M - low B/M, etc., (8) Annualize, align dates, handle missing data. The validation step: Download official Fama-French factors from Ken French's data library via `getfactormodels`. Compute correlation and RMSE between your factors and the official factors. Target: correlation > 0.95 for SMB, HML, MKT. If lower, diagnose: Did you handle delistings correctly? Did you align fundamental data with the right lag (fundamentals from Q3 2023 should affect portfolio formation in 2024, not immediately)? Did you use the same weighting scheme (equal-weight vs. value-weight)? The gotchas: (1) Fundamental data lags — you can't use Q4 2023 book value in December 2023 (it's not reported yet); you need to carry forward the last known value, (2) Survivorship bias — yfinance only has current listings; you're missing delisted stocks, which biases returns upward, (3) Micro-caps — stocks with market cap < $100M have illiquid, erratic returns; Ken French excludes the smallest NYSE stocks from his universe (NYSE breakpoints), (4) Corporate actions — splits and dividends affect market cap computation; use adjusted prices, (5) Breakpoint methodology — Fama-French uses NYSE-only breakpoints (the 50th percentile of NYSE market cap) applied to the full NYSE/AMEX/NASDAQ universe; this ensures stable breakpoints that don't shift with the composition of the investable universe.
- **Key demonstration:** The full pipeline implemented on a simplified universe (S&P 500 constituents, 10 years, monthly rebalancing). Show the DataFrame at each step: (1) Raw prices + fundamentals, (2) Computed characteristics, (3) Ranked and sorted, (4) Portfolio assignments, (5) Portfolio returns, (6) Factor returns. Compare student-constructed SMB, HML, and MKT to official Ken French factors for the overlapping period. Plot time series side-by-side. Compute correlation, RMSE, and mean absolute error. Discuss: "Our SMB has 0.92 correlation with Ken French's SMB. Why not 1.0? We're using S&P 500 (survivorship-biased, large-cap only), while Ken French uses the full NYSE/AMEX/NASDAQ universe with NYSE breakpoints. We're using yfinance fundamental data (which has reporting lags we're not handling perfectly), while Ken French uses Compustat with exact point-in-time data. Despite these differences, 0.92 correlation proves we've got the methodology correct."
- **Acceptance criteria:**
  - Full pipeline runs on >= 500 stocks over >= 5 years of monthly data
  - Characteristics computed: size, book-to-market, profitability, investment (momentum optional)
  - At least 3 factors constructed: MKT (or Rm-Rf), SMB, HML
  - Student-constructed factors validated against official Ken French factors
  - Correlation with Ken French factors: MKT > 0.95, SMB > 0.85, HML > 0.85 (targets may be lower due to universe differences)
  - Time-series plot shows student factor and official factor tracking each other visually
  - Mean returns have economically sensible signs: SMB ≈ 0.2-0.4% monthly (small > large historically), HML ≈ 0.3-0.5% monthly (value > growth), MKT ≈ 0.8-1.2% monthly (equity risk premium)
  - Volatilities (annualized standard deviation) are realistic: MKT ≈ 15-20%, SMB ≈ 10-15%, HML ≈ 10-15%
  - No missing values in factor return time series (gaps handled via forward-fill or exclusion)

### Closing

- **Summary table:** Factor models at a glance

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| Systematic risk | Risk shared across many assets (market movements, sector trends) | Can't be diversified away; earns a risk premium |
| Idiosyncratic risk | Stock-specific risk (earnings surprises, management changes) | Diversifiable; earns no premium in equilibrium |
| Beta | Sensitivity to a factor (e.g., market beta measures how much a stock moves with the market) | The coefficient from time-series factor regression; quantifies exposure |
| Risk premium | Expected return per unit of factor exposure | The lambda in Fama-MacBeth; tells you if a factor is priced |
| SMB (Size) | Small-cap return minus large-cap return | Small stocks outperform large historically; captures size risk or liquidity risk |
| HML (Value) | High B/M return minus low B/M return | Value stocks outperform growth; captures distress risk or behavioral overreaction |
| RMW (Profitability) | Profitable firms minus unprofitable firms | Profitable firms are safer and more attractive |
| CMA (Investment) | Conservative firms minus aggressive firms | Firms that overinvest destroy value |
| UMD (Momentum) | Past winners minus past losers | Momentum persists 6-12 months; behavioral or risk-based explanations debated |
| Portfolio-sorting (FF) | Form portfolios by ranking stocks, compute long-short returns | Intuitive, tradeable, the academic standard for 30 years |
| Cross-sectional regression (Barra) | Regress returns on characteristics each period, extract factor returns | Used in risk models; more flexible than portfolio sorts |
| Fama-MacBeth regression | Two-step procedure to test if a factor is priced in the cross-section | The statistical gold standard for factor validation |
| The factor zoo | 300+ proposed factors in academic literature, most are noise | Lesson: Out-of-sample validation and economic intuition are essential |

- **Career connections:** How this week's skills map to daily work:
  - **Quant researcher (buy-side):** A quant researcher at a statistical arbitrage fund (Two Sigma, AQR, Citadel) uses factor models daily in two ways: (1) Risk management — decompose portfolio returns into factor exposures to understand where risk and P&L are coming from (Are we accidentally long the value factor? Is our alpha coming from sector bets or true stock-picking?), (2) Alpha generation — factors are FEATURES for ML models. Week 4 will use SMB, HML, momentum, profitability as inputs to gradient boosting and neural net alpha models. A researcher who understands factor construction can engineer better features than one who treats factors as black boxes.
  - **Portfolio manager / factor investor:** Factor investing (also called "smart beta") is a $2 trillion industry. Asset managers (AQR, BlackRock, Vanguard) run funds that explicitly target factor exposures: value ETFs, momentum funds, quality strategies. The PM's job is to decide HOW MUCH exposure to each factor, given current valuations and crowding. This requires deep understanding of factor behavior — when do factors perform well? what's their correlation? how do they interact? Everything in this week's lecture is daily working knowledge.
  - **Risk quant at asset manager:** Risk teams use Barra-style factor models to decompose portfolio risk. A typical Monday morning risk meeting: "The portfolio lost 2% last week. How much was due to market exposure? How much to sector bets? How much to the value factor? How much is unexplained (specific risk)?" The risk quant runs a factor attribution report using the Barra USE4 model (or an internal model built on the same principles). The methodology is exactly what Section 4 demonstrated.
  - **Vendor quant (MSCI, S&P, BlackRock Aladdin):** MSCI sells the Barra risk models (USE4, GEM3) to asset managers globally. S&P sells factor indices. BlackRock's Aladdin platform provides factor analytics to institutional clients. A quant at these firms builds and maintains factor models — researching new factors, validating existing ones, computing factor covariances, and ensuring the models work across asset classes and geographies. This role requires mastery of both Fama-French and Barra methodologies plus the statistical machinery to test factor robustness.

- **Bridge to Week 4:** "You now understand how to decompose returns into systematic factors. But factor models are LINEAR — they assume returns are a weighted sum of factor exposures. Real-world returns have nonlinearities, interactions, and complex dependence structures that linear models miss. Week 4 introduces ML for alpha generation: using the firm characteristics from this week (size, value, momentum, profitability) as FEATURES in gradient boosting models and neural networks. The question shifts from 'what are the factors?' to 'can we predict returns better than linear factor models by using ML to learn complex patterns in these characteristics?' The factors you built this week become the input features to next week's models."

- **Suggested reading with annotations:**
  - **Fama & French (1993), "Common Risk Factors in the Returns on Stocks and Bonds," *Journal of Financial Economics*** — The original 3-factor paper. Introduces SMB and HML. Still readable 30 years later. The most-cited paper in empirical asset pricing.
  - **Fama & French (2015), "A Five-Factor Asset Pricing Model," *Journal of Financial Economics*** — The 5-factor extension. Adds RMW (profitability) and CMA (investment). Shows HML becomes redundant. This is the current academic standard.
  - **Fama & French (2020), "Comparing Cross-Section and Time-Series Factor Models," *Review of Financial Studies*** — Compares portfolio-sorting (their approach) to Fama-MacBeth cross-sectional regressions. Discusses when each is appropriate and how results differ. Essential for understanding the Barra vs. FF distinction.
  - **Fama & MacBeth (1973), "Risk, Return, and Equilibrium: Empirical Tests," *Journal of Political Economy*** — The original two-step regression paper. Old but still the standard methodology. Read this to understand the statistical machinery behind factor testing.
  - **Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns," *Review of Financial Studies*** — The definitive factor zoo paper. Catalogs 316 proposed factors and shows most don't replicate. Essential reading for understanding the replication crisis in finance. Proposes t-stat > 3.0 as the new significance threshold for factor discovery.
  - **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning," *Review of Financial Studies*** — The bridge to Week 4. Shows how ML (gradient boosting, neural nets) can use firm characteristics (the factors from this week) to predict returns with R² = 3-5%, compared to R² < 1% for linear factor models. The empirical framework for modern quantitative equity research.
  - **Lopez de Prado (2018), *Advances in Financial Machine Learning*, Chapter 8** — "Feature Importance Analysis." Shows how to use SHAP values and permutation importance to understand which characteristics (factors) matter most in ML models. Week 4 and Week 10 will revisit this.

## Seminar Outline

### Exercise 1: From Fundamentals to Factors — Building SMB and HML by Hand

- **Task type:** Skill Building
- **The question:** Can you replicate the Fama-French SMB and HML factors using only free data? How close can you get to the official benchmarks, and what do the discrepancies teach you about data quality and methodology?
- **Data needed:** Current S&P 500 constituent list (from Wikipedia or yfinance), 10 years of daily adjusted close prices (yfinance), quarterly fundamental data (yfinance: `ticker.balance_sheet`, `ticker.info['marketCap']`)
- **Tasks:**
  1. Download the current S&P 500 ticker list and 10 years of daily adjusted close prices for all ~500 tickers
  2. Download fundamental data for each ticker: total equity (from balance sheet), market cap (from `ticker.info` or price × shares outstanding)
  3. At the end of each June (Fama-French's annual rebalancing month), compute: (a) Size = market cap as of June 30, (b) Book-to-market = (book equity from most recent fiscal year end) / (market cap as of June 30)
  4. For each June, rank stocks by size and split into 2 groups (small = bottom 50%, big = top 50% by NYSE breakpoints — approximate using your universe's median as the breakpoint)
  5. Independently rank stocks by book-to-market and split into 3 groups (value = top 30%, neutral = middle 40%, growth = bottom 30%)
  6. Form 6 portfolios (2x3 sort: small-value, small-neutral, small-growth, big-value, big-neutral, big-growth). Each portfolio is equal-weighted.
  7. For each month from July Year t to June Year t+1, compute portfolio returns. Rebalance annually each June.
  8. Compute SMB and HML: SMB = (1/3) * (small-value + small-neutral + small-growth) - (1/3) * (big-value + big-neutral + big-growth), HML = (1/2) * (small-value + big-value) - (1/2) * (small-growth + big-growth)
  9. Download official Ken French SMB and HML factors via `getfactormodels` (monthly frequency, same time period)
  10. Compare: compute correlation, RMSE, plot time series side-by-side, compute mean/std/Sharpe for both your factors and official factors
- **Expected insight:** Students will get correlations of 0.80-0.95 with official factors — not perfect, but strong. The discrepancies come from: (1) Survivorship bias in the S&P 500 universe (current constituents only, no delistings), (2) yfinance fundamental data has reporting lags students aren't handling perfectly (should use stale book values from 6+ months ago, not the latest value), (3) S&P 500 is large-cap only, while Ken French uses the full NYSE/AMEX/NASDAQ universe with NYSE breakpoints, (4) Equal-weighting vs. value-weighting (Ken French uses value-weighted portfolios for the official factors). Despite these issues, the HIGH correlation proves the methodology is correct. Students learn that factor construction is sensitive to data quality and that proprietary data (CRSP, Compustat) exists BECAUSE these details matter. The exercise builds intuition for how much "data cleaning" and "methodology precision" affect quantitative research results — a 0.85 correlation is "good enough" for educational purposes but would be unacceptable in a production risk model.
- **Acceptance criteria:**
  - SMB and HML computed for at least 5 years of monthly data (>= 60 months)
  - Portfolio formation uses 2x3 sort (size x value) with exactly 6 portfolios
  - Rebalancing occurs annually in June (Fama-French standard)
  - Correlation with official Ken French factors: SMB >= 0.75, HML >= 0.75 (lower bounds due to data quality differences)
  - Mean returns have correct signs: SMB > 0 (small > large historically), HML > 0 (value > growth historically)
  - Time-series plot shows student factors tracking official factors with visible co-movement
  - At least one source of discrepancy identified and discussed (survivorship bias, universe composition, weighting scheme, or fundamental data lag)

### Exercise 2: Factor Exposures and Return Decomposition — What Drives Your Portfolio?

- **Task type:** Guided Discovery
- **The question:** You hold a portfolio of 20 stocks. How much of your return comes from market exposure? Size? Value? Stock-specific selection? Factor models let you decompose portfolio returns into systematic and idiosyncratic components. This is the core of risk attribution.
- **Data needed:** Pick 20 stocks spanning size and value spectrums (e.g., 5 large-value: JPM, JNJ, PFE, XOM, CVX; 5 large-growth: AAPL, MSFT, GOOGL, META, TSLA; 5 small-value: regional banks or industrials; 5 small-growth: biotech or small-cap tech). Download 5 years of monthly returns. Download official Fama-French MKT, SMB, HML factors via `getfactormodels`.
- **Tasks:**
  1. Construct an equal-weight portfolio of your 20 stocks (rebalanced monthly to equal weights)
  2. Regress the portfolio's monthly returns on the Fama-French 3 factors: `R_p,t - R_f,t = alpha + b_MKT * MKT_t + b_SMB * SMB_t + b_HML * HML_t + epsilon_t`
  3. Report the estimated factor loadings (betas) with t-statistics and p-values. Interpret each: "Our portfolio has a market beta of 1.15, meaning it's 15% more volatile than the market. Our SMB loading is 0.30 — we're tilted toward small-caps. Our HML loading is -0.10 — we're slightly tilted toward growth (negative HML)."
  4. Compute the R² of the regression. What percentage of your portfolio's return variance is explained by the 3 factors? What's left over (1 - R²) is idiosyncratic risk.
  5. Decompose a single month's return: `R_p,t - R_f,t = b_MKT * MKT_t + b_SMB * SMB_t + b_HML * HML_t + (alpha + epsilon_t)`. Example: "In March 2020, our portfolio lost 18%. The market factor contributed -12% (beta_MKT * MKT_March_2020), the size factor contributed -2%, value contributed -1%, and the remaining -3% is specific to our stock picks (alpha + residual)."
  6. Now run the same regression for each individual stock (20 separate regressions). Produce a table: Stock, Beta_MKT, Beta_SMB, Beta_HML, Alpha (annualized), R². Identify which stocks are driving your portfolio's factor exposures.
  7. Portfolio-level vs. stock-level: Does the portfolio's SMB loading equal the average of the individual stocks' SMB loadings? (It should, if equal-weighted and monthly rebalanced.)
- **Expected insight:** Students discover that 70-90% of portfolio variance is explained by systematic factors (R² = 0.70-0.90 is typical for equity portfolios regressed on FF3). The remaining 10-30% is idiosyncratic — the risk you took by picking these specific stocks. The factor loadings make intuitive sense: a portfolio of AAPL, MSFT, GOOGL has negative SMB (large-cap tilt) and negative HML (growth tilt). A portfolio of regional banks has positive SMB and positive HML. The alpha (intercept) is typically near zero and statistically insignificant — consistent with the EMH (no free lunch). A positive, significant alpha would suggest genuine stock-picking skill (or luck). The decomposition of a crisis month (March 2020) shows that MOST of the loss came from market beta, not stock selection — a humbling realization about the limits of diversification during market crashes. The stock-level regressions reveal heterogeneity: some stocks have high R² (90%+ of their variance is systematic), others have low R² (50% — they move idiosyncratically). This connects to the CAPM's insight: idiosyncratic risk can be diversified away; systematic risk cannot.
- **Acceptance criteria:**
  - Portfolio of 20 stocks constructed and rebalanced monthly to equal weights for >= 5 years
  - Fama-French 3-factor regression run on portfolio returns
  - Factor loadings (betas) estimated with t-stats and p-values reported
  - Regression R² > 0.50 (at least half of variance explained by factors — should be much higher for equity portfolios)
  - Single-month return decomposition shown with numerical contributions from each factor
  - Individual stock regressions run for all 20 stocks (20 separate regressions)
  - Summary table includes: Stock, Beta_MKT, Beta_SMB, Beta_HML, Alpha, R²
  - At least one stock has R² > 0.80 (highly systematic), at least one has R² < 0.60 (more idiosyncratic)
  - Discussion: "Our portfolio's SMB loading is driven primarily by [these 5 small-cap stocks], while our growth tilt comes from [these 5 large-growth names]."

### Exercise 3: Testing New Factors with Fama-MacBeth — Is Leverage Priced?

- **Task type:** Investigation
- **The question:** The Fama-French factors (size, value, profitability, investment, momentum) are well-established. But what about other characteristics? Is financial leverage (debt-to-equity ratio) priced in the cross-section? Does higher leverage predict higher returns (compensation for risk) or lower returns (distressed firms underperform)? Use Fama-MacBeth to test it.
- **Data needed:** S&P 500 stocks, 5 years of monthly returns, balance sheet data from yfinance (total debt, total equity)
- **Tasks:**
  1. Download balance sheet data for S&P 500 constituents. Compute leverage = total debt / total equity for each stock. Handle missing values (drop stocks with no debt or equity data).
  2. Also compute the standard Fama-French characteristics: size (market cap), book-to-market, momentum (past 12-month return, skip most recent month)
  3. **Step 1 (time-series):** For each stock, regress its excess returns on MKT, SMB, HML (use official factors from Ken French). This gives you factor-adjusted returns and residuals. Estimate betas for each stock.
  4. **Step 2 (cross-sectional):** For each month, regress that month's stock returns on: (a) the betas from Step 1 (MKT, SMB, HML), (b) the stock's leverage ratio (standardized to mean 0, std 1). This tests whether leverage has explanatory power BEYOND the FF3 factors.
  5. Extract the lambda coefficients for leverage from each month's cross-sectional regression. Compute the time-series mean and t-statistic. Is mean(lambda_leverage) significantly different from zero?
  6. Compare to a "naive" test: instead of Fama-MacBeth, just regress average returns on leverage across all stocks (single cross-sectional regression). Do you get the same answer? (Probably not — the standard errors will be wrong.)
  7. Validate with `linearmodels.FamaMacBeth`: Re-run the full procedure using the library and compare to your manual implementation.
- **Expected insight:** Leverage is NOT a robust factor in recent data — empirical results are mixed. Some studies find high-leverage firms underperform (consistent with distress risk), others find no relationship. The Fama-MacBeth t-stat for leverage is typically |t| < 1.5, meaning it's NOT significantly priced after controlling for size, value, and momentum. This is a NEGATIVE result — and negative results are educational. Students learn that not every characteristic is a factor. The "naive" cross-sectional regression (regressing average returns on leverage, ignoring time variation) will likely give a different answer with incorrectly narrow standard errors — this demonstrates WHY Fama-MacBeth exists (to handle cross-sectional and time-series correlation properly). The manual implementation should match the `linearmodels` library output (within numerical precision) — if not, the student has a bug.
- **Acceptance criteria:**
  - Leverage computed for >= 200 stocks with valid balance sheet data
  - Step 1: FF3 betas estimated for each stock via time-series regression
  - Step 2: Cross-sectional regressions run for each month (>= 60 months)
  - Lambda_leverage time series extracted with >= 60 observations
  - Mean(lambda_leverage) and t-stat computed
  - |t-stat| < 2.0 for leverage (likely result — leverage is NOT priced after controlling for FF3)
  - Naive cross-sectional regression run for comparison (single regression, all stocks, average returns ~ leverage)
  - `linearmodels.FamaMacBeth` validation: results match manual implementation within 10%
  - Discussion: "Leverage does not appear to be priced in our sample after controlling for size, value, and momentum. This could mean: (a) leverage risk is subsumed by value (high-leverage firms tend to be value stocks), or (b) leverage is not a systematic risk factor."

### Exercise 4: Out-of-Sample Factor Validation — Do Factors Persist?

- **Task type:** Investigation
- **The question:** A factor looks strong in-sample (2000-2015). Does it work out-of-sample (2016-2023)? How much does factor performance degrade, and is the degradation uniform across factors? This exercise simulates the real-world process of validating a trading strategy.
- **Data needed:** S&P 500 stocks, 20 years of monthly data split: 2004-2015 (in-sample), 2016-2023 (out-of-sample). Fundamental data (size, book-to-market, profitability, investment, momentum).
- **Tasks:**
  1. Using ONLY in-sample data (2004-2015), construct 5 factors: MKT, SMB, HML, RMW, CMA using the portfolio-sorting methodology from Exercise 1
  2. For each factor, compute in-sample performance: mean return, volatility, Sharpe ratio, t-stat (is mean return significantly > 0?)
  3. Now apply the SAME portfolio formation rules to out-of-sample data (2016-2023). Crucially, do NOT re-optimize breakpoints or methodology — use the EXACT SAME quantile cutoffs, weighting scheme, rebalancing frequency. This is a true out-of-sample test.
  4. Compute out-of-sample performance for each factor: mean return, volatility, Sharpe ratio, t-stat
  5. Build a comparison table: Factor, Mean Return (IS), Mean Return (OOS), Sharpe (IS), Sharpe (OOS), t-stat (IS), t-stat (OOS)
  6. Compute the out-of-sample degradation: OOS Sharpe / IS Sharpe (typical result: 0.4-0.7 — OOS performance is 40-70% of IS performance)
  7. Test correlation stability: Compute the correlation matrix of the 5 factors in-sample. Compute the same correlation matrix out-of-sample. How similar are they? (Use Frobenius norm or element-wise RMSE.)
  8. Investigate: Which factor degrades the most? Which is most stable? Does momentum (UMD) break down in recent years (a common claim)? Does profitability (RMW) strengthen (consistent with academic findings)?
- **Expected insight:** ALL factors degrade out-of-sample — this is the norm, not the exception. The typical pattern: in-sample Sharpe = 0.6-0.8, out-of-sample Sharpe = 0.3-0.5 (a 30-50% drop). Some factors degrade more than others. Value (HML) has weakened substantially post-2015 — this is a well-documented shift often attributed to the rise of intangible assets (tech stocks with low book values but high market values). Momentum and profitability tend to be more stable. The factor correlation structure is MOSTLY stable (e.g., SMB and HML have near-zero correlation both IS and OOS) — this is evidence that the factors are capturing distinct sources of risk, not just noise. A factor whose correlation with others CHANGES dramatically OOS is suspect (likely overfit IS). Students learn the hard lesson: a backtest from 2000-2015 that shows Sharpe = 1.2 will likely realize Sharpe = 0.6-0.8 going forward. This is alpha decay, and it's universal. The implication: strategies must be robust to 50% performance degradation, or they'll fail in production.
- **Acceptance criteria:**
  - All 5 factors (MKT, SMB, HML, RMW, CMA) constructed in-sample (2004-2015) and out-of-sample (2016-2023) using identical methodology
  - In-sample performance computed: mean, vol, Sharpe, t-stat for all 5 factors
  - Out-of-sample performance computed: mean, vol, Sharpe, t-stat for all 5 factors
  - Comparison table complete with IS vs. OOS columns for each metric
  - At least one factor has OOS Sharpe < 0.7 * IS Sharpe (performance degradation)
  - At least one factor has OOS t-stat < IS t-stat (weaker statistical significance)
  - Correlation matrix computed IS and OOS; Frobenius norm or RMSE of difference reported
  - Discussion: "All factors degrade OOS, consistent with alpha decay. Value (HML) shows the largest degradation, while profitability (RMW) is relatively stable. Factor correlations remain similar IS vs. OOS, suggesting the factors are capturing real, persistent risk dimensions."

## Homework Outline

### Mission Framing

The head of research drops by again: "I need a factor analytics platform. We're launching new strategies and I want to understand their factor exposures before we allocate capital. Build me a system that takes any stock universe, computes all the major factors from scratch, validates them against benchmarks, runs Fama-MacBeth tests, and produces a clean risk attribution report for any portfolio. And make it reusable — we'll run this on US equities, international equities, and eventually other asset classes."

This is not a one-off backtest. It's infrastructure. The tools you build this week will be used in Week 4 (as feature inputs to ML models), Week 5 (for backtest evaluation), and Week 6 (for portfolio risk management). Every quantitative fund has some version of this system. You're building the educational equivalent.

The homework integrates everything: data pipelines from Week 1, time-series analysis from Week 2, and this week's factor machinery. By the end, you'll have a modular, well-documented factor analytics toolkit that demonstrates both financial domain knowledge and software engineering discipline.

### Deliverables

1. **A `FactorBuilder` class**
   - **Task type:** Construction
   - Build a reusable class that takes a universe of stocks (tickers + date range) and produces Fama-French-style factors from scratch:
     - **Data acquisition:** Download prices (adjusted close, daily) and fundamental data (balance sheet, income statement) from yfinance. Handle API failures, missing tickers, and partial data gracefully.
     - **Characteristic computation:** Compute monthly characteristics for each stock: size (market cap), value (book-to-market), profitability (operating income / book equity), investment (YoY asset growth), momentum (past 12-month return, skip most recent month). Handle missing fundamentals (drop stocks with missing data, don't fill with zeros).
     - **Portfolio formation:** At each rebalancing date (configurable: monthly, quarterly, or annually), rank stocks by each characteristic, split into quantiles (configurable: terciles, quintiles, or deciles), form portfolios (equal-weight or value-weight), compute next-period portfolio returns.
     - **Factor construction:** Construct factor returns: MKT (Rm - Rf, using 1-month T-bill rate as Rf), SMB, HML, RMW, CMA, UMD (momentum). Return a DataFrame of factor returns (dates as index, factors as columns).
     - **Validation:** Compare constructed factors to official Ken French factors (download via `getfactormodels`). Return correlation, RMSE, mean difference, and a diagnostic report.
   - **Acceptance criteria:**
     - Class instantiates with configurable parameters: ticker list, date range, rebalancing frequency, quantiles, weighting scheme
     - Runs on >= 500 tickers over >= 5 years without crashing (handles tickers with missing data)
     - Computes at least 5 characteristics: size, value, profitability, investment, momentum
     - Constructs at least 4 factors: MKT, SMB, HML, RMW (CMA and UMD optional but encouraged)
     - Factor returns have no missing values (gaps handled via exclusion or forward-fill)
     - Validation against Ken French factors: correlation >= 0.80 for MKT, >= 0.75 for SMB and HML (lower thresholds due to data quality differences)
     - Returns a diagnostic report as a dict or DataFrame: {factor: {corr: 0.87, rmse: 0.02, mean_diff: 0.001}}

2. **A Fama-MacBeth regression engine**
   - **Task type:** Construction
   - Build a function or class that runs the two-step Fama-MacBeth procedure:
     - **Step 1 (time-series):** For each asset in a universe, regress its returns on a set of factors (e.g., MKT, SMB, HML). Extract factor loadings (betas) for each asset. Handle short time series (assets with < 36 months of data are excluded).
     - **Step 2 (cross-sectional):** For each time period, regress assets' returns on their betas from Step 1. Extract risk premia (lambdas) for each period.
     - **Aggregation:** Compute mean(lambda) across time, standard error, t-statistic, p-value. Support Newey-West standard errors (autocorrelation correction).
     - **Output:** Return a summary table: Factor, Mean Risk Premium, Std Error, t-stat, p-value, with interpretation ("MKT is priced: mean lambda = 0.8% per month, t = 3.2, p < 0.01").
   - Run the engine on two test cases: (a) S&P 500 stocks, Fama-French 3 factors (MKT, SMB, HML), 2010-2020, (b) S&P 500 stocks, 5 characteristics (size, value, profitability, investment, momentum) directly as factors (no Step 1 — just regress returns on characteristics each month).
   - **Acceptance criteria:**
     - Step 1: betas estimated for >= 200 stocks
     - Step 2: lambdas estimated for >= 60 time periods
     - Mean lambda, t-stat, and p-value computed for each factor
     - At least one factor has |t-stat| > 2.0 using standard SE (statistically significant)
     - Summary table shows both standard and Newey-West standard errors side-by-side, so students can see how the NW correction inflates SE and reduces t-statistics
     - Test case (a): at least one factor should have |t-stat| > 2 (standard SE), confirming it's priced
     - Test case (b): at least one characteristic (e.g., momentum) should have t-stat > 2

3. **A portfolio factor attribution report**
   - **Task type:** Skill Building
   - Take a portfolio (either a predefined list of 30 stocks or a synthetic "momentum" portfolio — top 50 stocks by past 12-month return) and produce a factor attribution report:
     - **Factor exposure:** Regress portfolio returns on Fama-French 5 factors (MKT, SMB, HML, RMW, CMA). Report betas, t-stats, R².
     - **Return decomposition:** For a selected period (e.g., 2020 full year or a single crisis month like March 2020), decompose the portfolio's return into contributions from each factor: R_p = beta_MKT * MKT + beta_SMB * SMB + ... + (alpha + residual). Present as a waterfall chart or table.
     - **Risk decomposition:** Using the factor covariance matrix (computed from historical factor returns), decompose the portfolio's variance: Var(R_p) = sum over factors of (beta_i^2 * Var(factor_i)) + sum over pairs of (beta_i * beta_j * Cov(factor_i, factor_j)) + Var(residual). Report: what % of portfolio risk is systematic (explained by factors) vs. idiosyncratic (specific to stock picks)?
     - **Comparison:** Run the same analysis on a market-cap-weighted S&P 500 portfolio (passive benchmark). Compare factor exposures and risk decomposition to your active portfolio.
   - **Acceptance criteria:**
     - Portfolio of >= 30 stocks constructed (or top 50 by momentum)
     - FF5 regression run on portfolio returns; betas, t-stats, R² reported
     - Regression R² > 0.70 (most portfolio variance explained by factors)
     - Return decomposition shown for at least one time period (full year or single month)
     - Decomposition sums to total return within 0.1% (numerical accuracy check)
     - Risk decomposition computed: systematic risk %, idiosyncratic risk %
     - Systematic risk is > 70% of total portfolio risk (typical for equity portfolios)
     - Comparison to passive S&P 500 benchmark: active portfolio has different factor exposures (e.g., higher SMB or UMD if momentum portfolio)

4. **An out-of-sample factor validation study**
   - **Task type:** Investigation
   - Split your data: 2004-2015 (in-sample), 2016-2023 (out-of-sample). Build a validation pipeline:
     - **In-sample:** Construct SMB, HML, RMW, CMA, UMD using your `FactorBuilder` on the in-sample period. Compute mean return, vol, Sharpe, t-stat for each factor.
     - **Out-of-sample:** Apply the EXACT SAME portfolio formation rules to the OOS period (same breakpoints, weighting, rebalancing). Do NOT re-optimize. Compute the same metrics.
     - **Comparison:** Build a performance comparison table (IS vs. OOS). Compute degradation: OOS Sharpe / IS Sharpe. Identify which factors are most robust.
     - **Factor zoo test:** Create 5 "fake" factors by randomly assigning stocks to portfolios (shuffle the characteristic rankings each month). Run the same IS/OOS analysis. Show that fake factors have near-zero performance OOS (they were never real).
     - **Economic intuition test:** For each factor, write 2-3 sentences explaining WHY it should persist OOS. Example: "Momentum persists because behavioral biases (underreaction to news) are structural features of human psychology, not data artifacts." Then assess: do the factors with strong economic stories perform better OOS?
   - **Acceptance criteria:**
     - All 5 factors (SMB, HML, RMW, CMA, UMD) constructed IS and OOS with identical methodology
     - IS and OOS performance metrics (mean, vol, Sharpe, t-stat) computed for all factors
     - Comparison table complete; at least one factor has OOS Sharpe < 0.7 * IS Sharpe (degradation is common)
     - Fake factors: 5 random "factors" created by shuffling rankings
     - Fake factors have mean Sharpe < 0.2 OOS (near-zero performance — they were never real)
     - At least 2 of 5 real factors have |t-stat| > 1.0 OOS (persistent, though weaker than IS — note that the OOS period 2016-2023 is relatively short, reducing statistical power)
     - Economic intuition provided for each real factor (2-3 sentences)
     - Discussion: "Factors with clear risk stories (SMB, RMW) are more robust OOS than those with weaker intuition. Value (HML) shows the largest degradation, consistent with recent market shifts. Fake factors have no OOS performance, confirming that statistical significance IS is not sufficient — economic intuition is essential."

### Expected Discoveries

- Building a production-grade `FactorBuilder` reveals dozens of edge cases: tickers that changed symbols, stocks with IPOs mid-sample (short history), stocks with missing fundamental data (biotechs with negative book equity), and stocks that were delisted (survivorship bias). A robust implementation handles all of these gracefully — dropping stocks with insufficient data rather than crashing or returning NaNs.
- Validating against Ken French factors shows correlations of 0.80-0.95 — strong but not perfect. The main sources of discrepancy: (1) survivorship bias in yfinance data (Ken French uses CRSP, which includes delistings), (2) universe differences (S&P 500 vs. full NYSE/AMEX/NASDAQ), (3) fundamental data lags (yfinance doesn't provide exact point-in-time data — students use the most recent balance sheet, while Ken French uses the balance sheet from 6+ months ago to ensure it was publicly available when portfolios were formed). Understanding WHAT causes these discrepancies is as educational as getting high correlations.
- The Fama-MacBeth regression confirms that MKT, SMB, and HML are priced (t-stats > 2) in most samples, but RMW and CMA are borderline (t-stats ~ 1.5-2.5) — they're weaker factors with less consistent risk premia. Momentum (UMD) is strongly priced in-sample but degrades substantially out-of-sample (consistent with academic findings that momentum weakens post-publication as traders arbitrage it).
- Portfolio factor attribution reveals that 80-90% of a diversified equity portfolio's variance is systematic (explained by factors), with only 10-20% idiosyncratic. This is a humbling result: most of your portfolio's risk comes from broad market movements, not your stock picks. The alpha (intercept) is typically insignificant — consistent with EMH. A positive, significant alpha would be evidence of skill (or luck), but it's rare.
- Out-of-sample validation shows universal performance degradation: in-sample Sharpe of 0.8 becomes out-of-sample Sharpe of 0.4-0.6 (a 50% drop is normal, not exceptional). Value (HML) shows the largest degradation (Sharpe drops from 0.6 IS to 0.2 OOS), which students will connect to real-world shifts (the rise of intangible assets, tech dominance). The fake factors have Sharpe near zero OOS — proving that statistical significance in-sample is not sufficient without economic intuition. Factors with clear risk stories (SMB = size risk, RMW = profitability) are more robust than factors with weaker narratives.

## Key Papers & References

- **Fama & French (1993), "Common Risk Factors in the Returns on Stocks and Bonds," *Journal of Financial Economics*** — The 3-factor model. Introduces SMB and HML. The foundation of modern factor investing. Read this first.
- **Fama & French (2015), "A Five-Factor Asset Pricing Model," *Journal of Financial Economics*** — Adds RMW (profitability) and CMA (investment). Shows HML becomes less important. The current academic standard.
- **Fama & French (2020), "Comparing Cross-Section and Time-Series Factor Models," *Review of Financial Studies*** — Compares portfolio-sorting to Fama-MacBeth cross-sectional regressions. Essential for understanding when to use each approach.
- **Fama & MacBeth (1973), "Risk, Return, and Equilibrium: Empirical Tests," *Journal of Political Economy*** — The original two-step regression methodology. Still the gold standard for testing if factors are priced.
- **Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns," *Review of Financial Studies*** — The factor zoo paper. Catalogs 316 factors and shows most don't replicate. Required reading for understanding the replication crisis. Proposes t > 3.0 as the new significance threshold.
- **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning," *Review of Financial Studies*** — Uses gradient boosting and neural nets to predict returns from firm characteristics. R² = 3-5% (vs. < 1% for linear models). The bridge to Week 4.
- **Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section," *Journal of Financial Economics*** — Proposes using machine learning (PCA, IPCA) to reduce the factor zoo to a parsimonious set. A modern take on factor selection.
- **Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere," *Journal of Finance*** — Shows value and momentum work across asset classes (equities, bonds, commodities, FX) and countries. Evidence that these are universal risk premia, not US equity anomalies.

## Career Connections

- **Quant researcher (buy-side):** At systematic equity funds (AQR, Two Sigma, Citadel), factor models are used daily in two ways: (1) **Risk management** — decompose P&L into factor contributions ("we lost 2% last week; 1.5% was market beta, 0.3% was momentum, 0.2% was stock selection"), (2) **Alpha generation** — the characteristics from this week (size, value, momentum, profitability) become input features to ML models (Week 4). A researcher who can BUILD factors from scratch (not just download them) can design custom factors specific to the fund's strategy.
- **Portfolio manager / factor investor:** Factor investing is a $2T industry. PMs at AQR, BlackRock, and Vanguard run portfolios that explicitly target factor exposures (value funds, momentum ETFs, quality strategies). The PM's job: decide HOW MUCH to allocate to each factor given current valuations, crowding, and correlations. This requires understanding factor behavior — when do they work? when do they fail? how do they interact? This week's validation exercises (in-sample vs. out-of-sample, correlation stability) are the exact analyses PMs run before launching new strategies.
- **Risk quant (buy-side or sell-side):** Risk teams use Barra-style factor models to attribute portfolio risk. A typical risk report: "Your portfolio has 1.2x market beta, +0.5 SMB, -0.3 HML, +0.7 momentum. Systematic risk is 85% of total variance. Your largest risk is market beta (contributing 60% of variance), followed by momentum (15%)." The risk quant builds and maintains these models. This role requires fluency in BOTH Fama-French (understanding factors) and Barra (cross-sectional regression) methodologies.
- **Vendor quant (MSCI, S&P, BlackRock Aladdin):** MSCI sells Barra risk models (USE4, GEM3) to institutional clients. S&P sells factor indices. BlackRock's Aladdin platform provides factor analytics. Quants at these firms: (1) Research new factors and validate existing ones, (2) Maintain factor models across asset classes and geographies, (3) Ensure models are stable and don't overfit, (4) Explain factor performance to clients ("Why did value underperform this quarter?"). This requires mastery of factor construction, validation methodology, and the factor zoo problem.

## Data Sources

- **yfinance (free, no API key):** Download prices (`yf.download()`) and fundamentals (`ticker.balance_sheet`, `ticker.income_stmt`, `ticker.info['marketCap']`). Coverage: all US publicly traded stocks. Limitations: survivorship bias (current listings only), fundamental data has reporting lags (not true point-in-time), occasional schema changes. Sufficient for all exercises this week.
- **getfactormodels (free, no API key):** Download official Fama-French factors from Ken French's data library. Supported models: FF3, FF5, Carhart 4-factor, momentum (UMD), liquidity, q-factors, AQR factors, and more. Usage: `import getfactormodels as gfm; df = gfm.get_factors(model='ff5', frequency='m')`. This is the validation benchmark for student-constructed factors.
- **Approximate data sizes:** 500 stocks x 10 years x monthly = 60K stock-months in long format. With prices + 5 fundamental fields = ~6 columns. CSV: ~20 MB. Parquet: ~4 MB. Download time via yfinance: 5-10 minutes (due to rate limits on fundamental data API calls).
- **Professional alternatives (mentioned, not required):**
  - **CRSP (via WRDS):** Survivorship-bias-free stock prices and returns for all US-listed securities from 1926 onward. Handles delistings, mergers, exact delisting returns. The academic gold standard. Requires institutional subscription (~$1000s/year).
  - **Compustat (via WRDS):** Point-in-time fundamental data with exact reporting dates. Ensures you only use information that was publicly available when portfolios were formed. Essential for production-grade factor models but overkill for educational purposes.
  - **Ken French's data library (free):** Pre-constructed factors (FF3, FF5, momentum, industry portfolios) dating back to 1926. Available at [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Accessible via `getfactormodels` or direct download.
