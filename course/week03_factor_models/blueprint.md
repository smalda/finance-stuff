# Week 3: Factor Models & Cross-Sectional Analysis

**Band:** 1 — Essential Core
**Implementability:** HIGH

## Learning Objectives

1. Estimate CAPM betas and interpret the security market line, including its empirical failures
2. Construct Fama-French factors (SMB, HML, RMW, CMA) from raw price and fundamental data using portfolio double-sorts
3. Validate self-constructed factors against official benchmark data and diagnose sources of divergence
4. Execute the Fama-MacBeth two-step procedure to test whether factors carry significant cross-sectional risk premia
5. Build a simplified Barra-style cross-sectional regression and decompose portfolio returns into factor and specific components
6. Engineer a cross-sectional feature matrix from firm characteristics with proper standardization
7. Explain why the factor zoo creates large-scale overfitting risk and how multiple testing correction addresses it

## Prerequisites

- **Required prior weeks:** 1, 2
- **Assumed from curriculum_state:** OHLCV data handling, return computation, adjusted vs. unadjusted prices, yfinance usage, Polars/pandas familiarity, Parquet I/O (Week 1); stationarity testing, stylized facts of returns, volatility clustering concept (Week 2)
- **Assumed ML knowledge:** Linear regression, cross-validation, feature engineering, regularization, multiple testing concepts
- **Prerequisite explanations in this week:** None

## Opening Hook

Every year, academic journals publish dozens of new "factors" that predict stock returns. By one count, over 400 have been proposed — more factors than months of out-of-sample data to test them. The industry calls this the "factor zoo," and navigating it is the largest-scale overfitting problem in quantitative finance. This week, you build the tools to tell the real animals from the taxidermy.

## Narrative Arc

**The setup.** You're an ML expert. Features are features — throw them into a model, let gradient descent sort them out. You've built feature matrices with hundreds of columns and trusted regularization to separate signal from noise. Cross-sectional prediction? That's just supervised learning with stocks as observations and characteristics as features. What could finance possibly add?

**The complication.** Finance has been doing "feature engineering for cross-sectional prediction" since 1964, and they called it "factor modeling." The humbling part: a three-factor linear model from 1993 still explains most of the cross-section of stock returns, and trillions of dollars are managed using it daily. The dangerous part: of those 400+ proposed factors, the majority are ghosts — artifacts of data mining that vanish out of sample. Your ML intuition to "add more features" is exactly the instinct that created this mess. The standard t-statistic threshold of 2.0 doesn't protect you when you've tested hundreds of hypotheses.

**The resolution.** Factor models aren't a primitive precursor to ML — they're the economic logic layer that ML needs to avoid drowning in noise. By week's end, you'll understand why cross-sectional ranking dominates absolute prediction, how to construct and validate factors from raw data, and how to build the principled feature matrix that Week 4's ML models will consume. The bridge between "factor" and "feature" is where the real alpha research happens.

## Lecture Outline

### Section 1: The One-Factor World → `s1_capm.py`

- **Hook:** "CAPM says one number — beta — explains all expected returns. It's elegant, Nobel Prize–winning, and empirically wrong. But you can't escape it: every performance report, every cost-of-capital estimate, every 'alpha' you've ever heard of is defined relative to this model."
- **Core concepts:** systematic vs. idiosyncratic risk, beta estimation (OLS of excess returns on market excess return), security market line, Jensen's alpha, R² as explanatory power
- **Key demonstration:** Estimate betas for diverse stocks across market caps and sectors, plot the empirical security market line, show the gap between CAPM prediction and realized returns
- **Key formulas:** CAPM: E[Rᵢ] = Rf + βᵢ(E[Rm] − Rf); Jensen's alpha: αᵢ = R̄ᵢ − [Rf + β̂ᵢ(R̄m − Rf)]
- **"So what?":** Alpha — the most overused word in finance — is literally "the part of returns CAPM can't explain." Every factor model that follows is an attempt to shrink alpha.
- **Bridge:** If beta can't explain the cross-section, what can?

### Section 2: When One Factor Fails — The Fama-French Three-Factor Model → `s2_ff3_model.py`

- **Hook:** "In 1992, Fama and French showed that beta explains almost nothing about which stocks earn more. Two characteristics — size and book-to-market — explained what CAPM couldn't. A one-factor model became three."
- **Core concepts:** size effect (small beats large), value effect (cheap beats expensive), SMB (Small Minus Big), HML (High Minus Low), time-series factor regression, alpha shrinkage under FF3 vs. CAPM
- **Key demonstration:** Download official FF3 factors, regress stock/portfolio returns on all three, show dramatic R² improvement over CAPM, visualize cumulative factor returns over decades
- **Key formulas:** FF3: Rᵢ − Rf = αᵢ + βᵢ,MKT(Rm − Rf) + βᵢ,SMB·SMB + βᵢ,HML·HML + εᵢ
- **"So what?":** FF3 redefined alpha. Strategies that looked brilliant under CAPM turned out to be loading on well-known, compensated risks. Your "stock-picking genius" was just a size and value bet.
- **Bridge:** How are these factors actually built from raw data?

### Section 3: Building Factors from Scratch — Portfolio Sorts → `s3_factor_construction.py`

- **Hook:** "The Fama-French factors aren't magic numbers from a database. They're portfolios — long the stocks with one characteristic, short those with the opposite. The construction methodology is both more mechanical and more subtle than most people realize."
- **Core concepts:** single and double sorts, breakpoints (median market cap, 30/70 book-to-market), long-short portfolio construction, value-weighted returns, annual rebalancing at June, the lag between fiscal year end and sort date
- **Key demonstration:** Construct SMB step-by-step from raw price and fundamental data — downloading stocks, computing book-to-market, applying breakpoints, forming portfolios, computing factor returns
- **"So what?":** Understanding construction reveals that "the value factor" isn't a platonic concept — it's an artifact of specific methodological choices. Change the breakpoints, the weighting, or the rebalancing frequency, and you get a different factor.
- **Bridge:** Three factors were good. Are five better?

### Section 4: The Five-Factor Model and the Momentum Orphan → `s4_ff5_momentum.py`

- **Hook:** "In 2015, Fama and French admitted three factors weren't enough. They added profitability and investment — but refused to include momentum, the anomaly that won't die and won't fit their theory."
- **Core concepts:** profitability factor (RMW: Robust Minus Weak), investment factor (CMA: Conservative Minus Aggressive), the FF5 model, momentum (UMD/WML), why momentum is excluded from FF5 (no risk-based explanation), factor redundancy analysis
- **Key demonstration:** Compare explanatory power — CAPM vs. FF3 vs. FF5 — across test portfolios, show diminishing alpha as more factors are added, visualize momentum's strong returns alongside its crash risk
- **Key formulas:** FF5: Rᵢ − Rf = αᵢ + βᵢ,MKT(Rm − Rf) + βᵢ,SMB·SMB + βᵢ,HML·HML + βᵢ,RMW·RMW + βᵢ,CMA·CMA + εᵢ
- **"So what?":** Going from 1 factor to 5 is principled. Going from 5 to 500 is data mining. The line between "adding explanatory power" and "fitting noise" is the central tension of modern factor research.
- **Bridge:** Constructing factors tests if characteristics PREDICT. But do they carry a risk PREMIUM?

### Section 5: Are Factors Priced? — The Fama-MacBeth Methodology → `s5_fama_macbeth.py`

- **Hook:** "Constructing a factor and proving it earns a risk premium are fundamentally different questions. The first is a portfolio exercise. The second requires the most elegant two-step procedure in financial econometrics."
- **Core concepts:** time-series step (estimate betas for each stock), cross-sectional step (regress returns on betas each period), time-series average of cross-sectional slopes as risk premium estimate, Newey-West standard errors, the errors-in-variables problem
- **Key demonstration:** Manual Fama-MacBeth implementation (pedagogical), then the production version using `linearmodels.FamaMacBeth`, compare results, interpret which factors carry significant premia
- **Key formulas:** Cross-sectional step: Rᵢₜ = γ₀ₜ + γ₁ₜβ̂ᵢ₁ + ... + γₖₜβ̂ᵢₖ + εᵢₜ; Risk premium: γ̄ₖ = (1/T)Σγₖₜ
- **"So what?":** Fama-MacBeth is the standard test for asset pricing models. If your ML model discovers a "new factor," this is how you prove it's not noise.
- **Bridge:** Academics sort portfolios. But how do practitioners manage trillions?

### Section 6: The Practitioner's Lens — Barra-Style Risk Models → `s6_barra_risk_models.py`

- **Hook:** "While academics debate whether factors are risk or anomaly, MSCI's Barra models are used to manage over $15 trillion. Their approach — regress all stock returns on all characteristics simultaneously, every day — is what an ML engineer would call 'linear regression with feature engineering.' Finance just got there first."
- **Core concepts:** cross-sectional regression (all stocks, all characteristics, each period), factor returns as regression coefficients, style factors vs. industry factors, risk decomposition (factor risk + specific risk), contrast with FF portfolio-sort methodology
- **Key demonstration:** Build a simplified Barra-style regression — regress cross-sectional returns on standardized characteristics and industry dummies, extract daily factor returns, decompose a sample portfolio's risk
- **Sidebar:** Industry factors (MENTIONED) — Barra USE4 uses 55-70 industry classifications; simplified sector dummies used here for tractability
- **"So what?":** FF answers "do these characteristics predict returns?" Barra answers "what are the sources of my portfolio's risk?" Both matter — FF for alpha research, Barra for risk management.
- **Bridge:** Factors are the finance community's name for features with economic theory behind them. Time to build the feature matrix.

### Section 7: From Factors to Features — Cross-Sectional Feature Engineering → `s7_feature_engineering.py`

- **Hook:** "Here's the secret that bridges your ML world and the finance world: factors are features with economic theory behind them. The difference is that in finance, you can't just throw 400 features into XGBoost and hope for the best — that's how the factor zoo was born."
- **Core concepts:** firm characteristics as features (P/E, P/B, ROE, asset growth, momentum, reversal, volatility), cross-sectional standardization (rank-transform, z-score within each period), winsorizing extremes, handling missing data, feature matrix construction at scale
- **Key demonstration:** Build a cross-sectional feature matrix for a broad stock universe — download fundamentals, compute ratios, standardize cross-sectionally, handle missing data — and show the resulting matrix ready for ML consumption
- **Sidebar:** The factor zoo (MENTIONED) — Harvey, Liu & Zhu (2016) showed the standard t > 2.0 threshold is insufficient for the 400+ factors tested; t > 3.0 is needed after multiple testing correction
- **Key formulas:** Cross-sectional z-score: zᵢₜ = (xᵢₜ − x̄ₜ) / σₜ
- **"So what?":** This feature matrix is the input to every cross-sectional ML model from Week 4 onward. The standardization and cleaning decisions made here propagate through every downstream model.
- **Bridge:** The feature matrix is ready. Next week, gradient boosting and neural networks ask: can ML extract signal that linear factor models miss?

## Narrative Hooks

1. **The factor zoo's body count.** Of the 400+ published factors, roughly two-thirds fail to replicate when tested with proper methodology. The bar for a "new" factor should be t > 3.0, not the traditional 2.0 — but journals kept publishing at the lower threshold for decades. *(Section 7 sidebar or transition into Section 7)*

2. **CAPM: wrong but inescapable.** CAPM explains very little cross-sectional return variation, yet it remains the foundation of corporate finance worldwide. Every company's cost of capital, every equity analyst's "alpha," every fund manager's performance attribution is defined relative to this model. The most influential wrong model in economics. *(Section 1)*

3. **The $15 trillion regression.** MSCI's Barra risk models — at their core, cross-sectional linear regressions — are used to manage over $15 trillion in assets. The methodology is conceptually identical to what any ML engineer would build with scikit-learn. Finance discovered it independently and built an industry around it. *(Section 6)*

4. **Momentum's crash and survival.** The momentum factor has generated strong positive returns for nearly a century — except for catastrophic crashes during sharp market reversals. Despite this, it remains one of the strongest anomalies in finance, used by virtually every systematic fund. No risk-based theory explains it, which is why Fama and French excluded it from FF5. *(Section 4)*

## Seminar Outline

### Exercise 1: Can You Replicate Fama-French?

- **Task type:** Guided Discovery
- **The question:** Construct SMB and HML factors from raw stock data and compare your self-built factors to official Ken French returns. Where do they diverge, and why?
- **Expected insight:** Self-constructed factors from free data correlate strongly but not perfectly with official Ken French factors. The gap comes from survivorship bias (free sources only have current tickers), different universe coverage (Ken French uses all NYSE/AMEX/NASDAQ), and data quality differences. This demonstrates concretely why institutional data (CRSP/Compustat) exists and what you sacrifice with free data.

### Exercise 2: Which Factors Carry a Risk Premium?

- **Task type:** Guided Discovery
- **The question:** Run a Fama-MacBeth regression on a cross-section of stocks using market beta, size, value, profitability, and momentum as characteristics. Which factors carry statistically significant risk premia with Newey-West standard errors?
- **Expected insight:** Factor premia significance varies across time periods. Market beta often shows a flat or negative cross-sectional relationship (the "beta anomaly"), size and value premia have weakened in recent decades, while momentum and profitability tend to remain more robust. The "are factors risk or anomaly?" debate is empirically messy — t-statistics depend heavily on the sample period.

### Exercise 3: Decompose Your Portfolio's Risk

- **Task type:** Skill Building
- **The question:** Take a portfolio of ~20 stocks (equal-weighted). Using a simplified Barra-style cross-sectional regression, decompose the portfolio's return variance into factor risk and specific (idiosyncratic) risk. How much of the portfolio's risk is explained by common factors?
- **Expected insight:** For a reasonably diversified equity portfolio, common factor exposure explains the majority of return variance. A concentrated portfolio (e.g., all tech) has higher factor risk because stocks share common factor exposures. This is why risk managers obsess over factor tilts — most portfolio risk isn't stock-specific.

### Exercise 4: The Factor Zoo Safari

- **Task type:** Investigation
- **The question:** Given 8-10 candidate characteristics (some well-known like momentum and value, some constructed from noise or weak signals), test each for cross-sectional predictive power using Fama-MacBeth. Apply multiple testing correction (Bonferroni or Benjamini-Hochberg). How many "significant" factors survive correction?
- **Expected insight:** Without correction, a majority of characteristics appear significant at p < 0.05. After multiple testing correction, only a handful survive. This demonstrates the factor zoo problem firsthand — the standard significance threshold produces alarming false positives when many hypotheses are tested simultaneously.

## Homework Outline

### Deliverable 1: The Factor Factory

- **Task type:** Construction
- **Mission framing:** Build a complete factor construction pipeline that downloads raw data, extracts fundamental metrics, performs portfolio double-sorts, and produces monthly factor returns for all five Fama-French factors plus momentum. The pipeline validates its output against official Ken French data and reports correlation and tracking error for each factor. This is the infrastructure that every cross-sectional strategy rests on.
- **Scope:** `FactorBuilder` class handling data download, fundamental ratio computation (B/M, OP, INV), breakpoint calculation, double-sort portfolio formation, value-weighted return computation, factor return aggregation, and validation against `getfactormodels` benchmark data.

### Deliverable 2: The Cross-Sectional Feature Matrix

- **Task type:** Construction
- **Mission framing:** Build a reusable feature engineering system that constructs a standardized cross-sectional feature matrix from raw stock data. This is the bridge between factor models (this week) and ML models (Week 4) — the feature matrix you build here is the direct input to next week's gradient boosting and neural network models.
- **Scope:** `FeatureEngineer` class computing fundamental features (P/E, P/B, ROE, asset growth, earnings yield), technical features (momentum at multiple horizons, short-term reversal, volatility), cross-sectional standardization (rank-transform and z-score), winsorizing, missing data handling, and output as a panel DataFrame ready for ML consumption.
- **Requires:** Deliverable 1 (factor returns for validation)

### Deliverable 3: The Factor Model Horse Race

- **Task type:** Investigation
- **Mission framing:** Which factor model best explains the cross-section of returns? Compare CAPM, FF3, and FF5 using Fama-MacBeth regressions on your stock universe. Report which factors are priced, how explanatory power changes as factors are added, and what remains unexplained — the residual alpha that ML might capture in Week 4.
- **Scope:** Fama-MacBeth regressions for three nested models, Newey-West t-statistics, adjusted R² comparison, analysis of residual alpha patterns, written interpretation of which model wins and why.
- **Requires:** Deliverable 1 (factor data), Deliverable 2 (feature matrix for characteristics)

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| CAPM & beta estimation | Demo: SML, alpha estimation | — | — |
| FF3 size/value anomalies | Demo: factor returns, R² improvement | Ex 1: replicate from raw data | — |
| Factor construction (portfolio sorts) | Demo: SMB walkthrough | — | D1: full 6-factor pipeline |
| FF5 + momentum | Demo: model comparison, momentum crash | — | D3: model horse race |
| Fama-MacBeth regression | Demo: manual + linearmodels | Ex 2: test which factors are priced | — |
| Barra-style risk decomposition | Demo: simplified cross-sectional regression | Ex 3: decompose portfolio risk | — |
| Cross-sectional feature engineering | Demo: feature matrix construction | — | D2: feature matrix builder |
| Factor zoo / multiple testing | Demo: the problem | Ex 4: test with correction | — |

## Career Connections

- **Quant Researcher at buy-side fund (AQR, Two Sigma, Citadel):** Tests whether new characteristics predict cross-sectional returns using Fama-MacBeth and spanning regressions. Builds factor-based signal models. Debates whether factors represent risk premia or behavioral anomalies — the answer shapes portfolio construction.

- **Risk Analyst at MSCI/Barra or vendor firm:** Builds and maintains production risk models using the cross-sectional regression methodology. Calibrates style and industry factor exposures for client portfolios spanning trillions in AUM. Investigates factor behavior during crises to update model parameters.

- **Portfolio Manager at systematic fund (AQR, Dimensional, Bridgewater):** Uses factor exposures to construct portfolios with targeted risk/return profiles. Monitors unintended factor tilts. Decides how much to tilt toward value, momentum, or quality based on regime analysis and capacity constraints.

- **Data Scientist at asset management firm:** Builds feature engineering pipelines that bridge traditional factor research with ML models. Constructs factor replicas from alternative data sources. Evaluates whether new features add explanatory power beyond established factors.

## Data Sources

- yfinance — daily equity prices, fundamental data (balance sheet, income statement, cash flow)
- getfactormodels — official factor return data (Fama-French 3/5/6, Carhart, q-factors)
- linearmodels — Fama-MacBeth regression implementation with Newey-West standard errors

## Key Papers & References

### Core

- **Sharpe (1964)** — "Capital Asset Prices: A Theory of Market Equilibrium." CAPM — where it all started.
- **Fama & French (1993)** — "Common Risk Factors in the Returns on Stocks and Bonds." Introduces FF3; launched the multifactor revolution.
- **Fama & French (2015)** — "A Five-Factor Asset Pricing Model." Extends FF3 with profitability and investment; the current canonical model.
- **Fama & MacBeth (1973)** — "Risk, Return, and Equilibrium: Empirical Tests." The cross-sectional regression methodology; still the standard test for factor pricing.

### Advanced

- **Harvey, Liu & Zhu (2016)** — "...and the Cross-Section of Expected Returns." Quantifies the factor zoo's multiple testing problem; the t > 3.0 argument.
- **Gu, Kelly & Xiu (2020)** — "Empirical Asset Pricing via Machine Learning." Bridge to Week 4 — applies ML to the cross-sectional prediction framework that factor models established.
- **Fama & French (2020)** — "Comparing Cross-Section and Time-Series Factor Models." Directly compares portfolio-sorting vs. cross-sectional regression approaches.

## Bridge to Next Week

This week gave you the economic logic layer: factors are features with theory behind them, and cross-sectional analysis is the paradigm. Week 4 asks the ML question — can gradient boosting and neural networks extract signal from the feature matrix that linear factor models miss? The Gu-Kelly-Xiu framework picks up exactly where Fama-MacBeth leaves off.
