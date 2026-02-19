# Course Outline: ML for Quantitative Finance

**Governing document:** OUTLINE_SPEC.md (definitions, rules, topic dispositions)
**Research basis:** RESEARCH_SYNTHESIS.md (topic tiers, career paths, evidence)

---

## 1. Preamble

### 1.1 Audience

ML/DL experts entering quantitative finance. They know neural architectures, training methodology, standard ML, RL, LLMs, and generative models. They do NOT know financial markets, instruments, domain-specific evaluation, or the finance-specific adaptations of ML.

### 1.2 Ordering Logic

Weeks are ordered by balancing two pressures:

**Pressure A (prerequisite correctness):** Financial domain knowledge must precede its ML application.

**Pressure B (industry relevance density):** Each successive week is less likely to be studied. Front-load the highest career value per unit of time.

Resolution: **Banded ordering.** Within each band, order by prerequisites. Between bands, order by industry relevance. The outline has 3 primary bands (18 weeks) plus an optional Band 4 (Weeks 19-20).

### 1.3 The Prefix Property

Any first-N-weeks subset forms a coherent, self-contained learning experience. No week introduces something that REQUIRES a later week to make sense.

### 1.4 Career Track Weighting

| Priority | Career Paths | Rationale |
|---|---|---|
| **1 — Primary** | ML Engineer at Fund, NLP/AI Engineer, Data Scientist at Fund, Quant Researcher (buy-side) | Highest ML relevance, buy-side, most accessible transitions |
| **2 — Strong secondary** | Execution Quant, Quant Developer / Research Engineer, Investment Engineer, Fintech/Vendor Quant | Buy-side or accessible, ML-driven growth |
| **3 — Sell-side/specialized** | Model Validation / MRM Quant, Quant Trader, Risk Quant | Increasing ML use, but further from ML background |
| **4 — Deep sell-side** | Desk Quant / Strat, XVA Quant | Requires stochastic calculus + C++ + derivatives depth beyond scope |

Universal topics come first. Priority 1 topics before Priority 2. Topics serving only Priority 3-4 are mentioned/sidebared, not given dedicated weeks.

### 1.5 Coverage Depth Key

| Depth | Meaning |
|---|---|
| **COVERED** | Primary focus; full treatment, implementation, exercises |
| **TOUCHED** | Significant component (~1/3 to 1/2 of a week); dedicated section, partial implementation |
| **MENTIONED** | 5-15 min section; conceptual awareness, no student implementation |
| **MENTIONED ABSTRACTLY** | 1-3 paragraphs; awareness that the area exists |
| **WOVEN IN** | Appears across weeks through assignments/standards, never standalone |
| **PREREQ EXPLAINED** | Background section within a week; enough to follow the application, not comprehensive |

### 1.6 Implementability Summary

| Grade | Definition | Count |
|---|---|---|
| **HIGH** | 70%+ of the week is implemented in code | 13 weeks |
| **MEDIUM-HIGH** | 55-70% implemented | 4 weeks |
| **MEDIUM** | 40-55% implemented | 3 weeks |
| **LOW** | <40% implemented | 0 weeks |

No LOW implementability weeks exist in this outline; all conceptual content is delivered through a data-driven or model-driven narrative.

### 1.7 Key Context from Research

Two findings that shape the entire outline:

1. **Finance knowledge is explicitly NOT required at most top firms** (Jane Street, Two Sigma, RenTech, G-Research, D.E. Shaw, Tower Research). Firms teach domain knowledge on the job but cannot teach mathematical maturity, programming, or research methodology. Implication: teach frameworks and intuition; don't attempt encyclopedic coverage.

2. **The biggest competition is physics/math PhDs, not other ML engineers.** These candidates have deeper mathematical foundations. Implication: the course builds financial mathematical vocabulary and probabilistic intuition — the areas where ML experts are weakest relative to their competition.

---

## 2. Week-by-Week Outline

---

### BAND 1: Essential Core

**Theme:** Universal foundations + the complete alpha pipeline — data, features, model, backtest, portfolio.
**Who it serves:** ALL career paths. Every student should complete this band.
**Research grounding:** All MUST KNOW topics are COVERED or WOVEN IN within this band.

---

#### Week 1: How Markets Work & Financial Data

**Band:** 1

**Topics:**
- How markets work: exchanges, order types, bid-ask spreads, market makers, clearing and settlement, asset classes overview (COVERED)
- Working with large-scale financial datasets: survivorship bias, point-in-time databases, look-ahead bias in data, corporate actions, data vendors and their quirks (COVERED)
- Commodities / energy markets (MENTIONED ABSTRACTLY) — how these markets differ from equities
- FX markets (MENTIONED ABSTRACTLY) — 24h markets, spot vs. forward
- KDB/Q (MENTIONED ABSTRACTLY) — "you may encounter this at legacy HFT firms; being replaced by Polars/DuckDB"

**Career paths served:** ALL (universal foundation)

**Research synthesis grounding:** MUST KNOW: How Markets Work, Working with Large-Scale Financial Datasets. NICHE: Commodities, FX. DECLINING: KDB/Q.

**Implementability:** HIGH — download tick and daily data, build data pipelines, explore order book snapshots, compute bid-ask spreads, detect survivorship bias in a dataset

**Prerequisites:** None (entry point)

**Prerequisite explanations needed:** None

**Multi-topic note:** Markets and data are tightly coupled — you understand markets BY working with their data. Asset class sidebars provide landscape awareness without requiring separate treatment.

---

#### Week 2: Financial Time Series & Volatility

**Band:** 1

**Topics:**
- Financial time series: return computation, stationarity tests, autocorrelation structure, ARMA models, fractional differentiation (COVERED)
- GARCH family: GARCH, EGARCH, GJR-GARCH — the industry's canonical volatility model (COVERED)
- Realized volatility, volatility clustering, stylized facts of financial returns (TOUCHED)

**Career paths served:** ALL (especially QR, data scientist, ML researcher, quant trader)

**Research synthesis grounding:** MUST KNOW: Time-Series Analysis (ARMA, GARCH, cointegration). SHOULD KNOW: Volatility Modeling (fundamentals).

**Implementability:** HIGH — fit GARCH to real data, compute realized vol, test stationarity, apply fractional differentiation to price series, compare vol forecasts

**Prerequisites:** Week 1

**Prerequisite explanations needed:** Brief refresher on stochastic processes — white noise, random walk, stationarity definitions. (10 min)

**Multi-topic note:** Single theme: how financial time series behave. GARCH is not a separate topic — it is THE answer to "how does the industry model volatility?"

---

#### Week 3: Factor Models & Cross-Sectional Analysis

**Band:** 1

**Topics:**
- Factor models: CAPM, Fama-French 3/5-factor, Barra-style risk models, cross-sectional regressions, Fama-MacBeth methodology (COVERED)
- Financial feature engineering: firm characteristics, technical indicators, fundamental ratios, constructing cross-sectional features at scale (COVERED — begins here, continues in Week 4)

**Career paths served:** QR buy-side, ML researcher, data scientist, asset management, vendor quant (MSCI/Barra)

**Research synthesis grounding:** SHOULD KNOW: Factor Models (elevated to COVERED — central to the cross-sectional prediction framework). MUST KNOW: Financial Feature Engineering (begins here).

**Implementability:** HIGH — download returns + fundamentals data, compute factor exposures, run Fama-MacBeth regressions, build a feature matrix from firm characteristics

**Prerequisites:** Week 1, Week 2

**Prerequisite explanations needed:** None

**Multi-topic note:** Factors and features are two sides of the same coin — factors are the finance community's name for features that explain cross-sectional returns.

---

#### Week 4: ML for Alpha — From Features to Signals

**Band:** 1

**Topics:**
- ML for alpha/signal generation: gradient boosting, neural nets applied to cross-sectional data, financial loss functions (IC-based, Sharpe-based), evaluation metrics (IC, rank IC, turnover-adjusted IC) (COVERED)
- Financial feature engineering continued: interaction features, non-linear transformations, feature importance in financial context (COVERED)
- Alternative data: satellite imagery, web data, news flow — as feature sources for alpha models (TOUCHED)

**Career paths served:** QR buy-side, ML researcher, data scientist, NLP/AI engineer

**Research synthesis grounding:** MUST KNOW: ML for Alpha/Signal Generation, Financial Feature Engineering. SHOULD KNOW: Alternative Data (introduction).

**Implementability:** HIGH — build cross-sectional prediction model (Gu-Kelly-Xiu framework), evaluate with IC, construct long-short portfolio from predictions, compare tree-based vs. neural approaches

**Prerequisites:** Week 2, Week 3

**Prerequisite explanations needed:** None (ML methodology assumed known)

**Multi-topic note:** Alpha generation and feature engineering are inseparable in practice. Alternative data enters as a feature source, not a standalone concept.

---

#### Week 5: Backtesting, Research Discipline & Transaction Costs

**Band:** 1

**Topics:**
- Backtesting methodology: look-ahead bias, purged cross-validation, combinatorial purged CV, walk-forward optimization, backtest overfitting quantification (COVERED)
- Research methodology / data snooping: multiple testing correction, deflated Sharpe ratio, probability of backtest overfitting, the "7 deadly sins of financial ML" (COVERED)
- Transaction cost modeling: bid-ask spread costs, market impact, slippage, the gap between backtest P&L and live P&L (COVERED)
- Hypothesis testing in finance: multiple comparisons, family-wise error rate, false discovery rate in strategy evaluation (COVERED — finance-specific aspects)

**Career paths served:** ALL (universal methodology)

**Research synthesis grounding:** MUST KNOW: Backtesting Methodology, Transaction Cost Modeling, Research Methodology / Data Snooping, Hypothesis Testing (finance-specific).

**Implementability:** HIGH — build a proper backtesting pipeline with purged CV, compute deflated Sharpe, add realistic transaction costs, measure the gap between naive and rigorous backtests

**Prerequisites:** Week 4 (need a model to backtest)

**Prerequisite explanations needed:** None

**Multi-topic note:** These four topics form a single coherent theme: "how to not fool yourself in financial ML." They are inseparable in practice and must be taught together.

---

#### Week 6: Portfolio Construction & Risk Management

**Band:** 1

**Topics:**
- Portfolio construction: mean-variance optimization (Markowitz), risk parity, Black-Litterman, hierarchical risk parity (COVERED)
- Risk metrics: VaR, Expected Shortfall, drawdown measures, stress testing basics (TOUCHED)
- Position sizing / Kelly criterion: optimal bet sizing under uncertainty (TOUCHED)
- ESG/climate risk (MENTIONED) — NGFS scenarios, carbon credit markets, data quality challenges

**Career paths served:** QR buy-side, quant PM, risk quant (Priority 3), asset management, vendor quant

**Research synthesis grounding:** MUST KNOW: Portfolio Construction & Risk Management. SHOULD KNOW: Risk Metrics, Position Sizing. GOOD TO KNOW: ESG/Climate Risk.

**Implementability:** HIGH — implement efficient frontier, compare optimization approaches on real data, compute VaR/ES, implement Kelly sizing, see how portfolio choice changes with risk constraints

**Prerequisites:** Week 3 (factor context), Week 4 (signals to combine), Week 5 (methodology)

**Prerequisite explanations needed:** Convex optimization basics — quadratic programming intuition, what the efficient frontier means as an optimization surface, how solvers work. (10 min)

**Multi-topic note:** Portfolio construction bridges signals to strategy. Risk management and position sizing are its inseparable companions. ESG is a brief sidebar within the portfolio context.

---

### =============================================
### CUTOFF MARKER: ESSENTIAL CORE (After Week 6)
### =============================================

**What the student has:** Market knowledge, financial data skills, time-series analysis, factor models, ML for alpha, rigorous backtesting, portfolio construction. They can independently build, evaluate, and backtest a quantitative equity strategy.

**Suitable for:** Entry-level data scientist at fund, junior QR, ML engineer transitioning to finance (basic preparation).

### =============================================

---

### BAND 2: High-Value Applications

**Theme:** Major applied ML domains with high career value + essential domain knowledge that broadens the quant education beyond equities.
**Who it serves:** Priority 1 and most Priority 2 career paths.
**Research grounding:** All SHOULD KNOW topics reach at least TOUCHED depth by the end of this band.

---

#### Week 7: NLP & LLMs for Financial Alpha

**Band:** 2

**Topics:**
- NLP/LLM for financial text: sentiment analysis, earnings call parsing, SEC filing analysis, FinBERT vs. general LLM embeddings for alpha (COVERED)
- Alternative data continued: text as the dominant alternative data source; combining textual and numerical features in alpha models (TOUCHED)
- Agentic AI for research augmentation (MENTIONED) — what LLMs can and cannot do in quant research workflows; the "agentic humility" reality check

**Career paths served:** NLP/AI Engineer (Priority 1), QR buy-side, data scientist, ML researcher

**Research synthesis grounding:** SHOULD KNOW: NLP/LLM for Financial Text (elevated to COVERED — fastest-growing role category). SHOULD KNOW: Alternative Data (continued). NICHE: Agentic AI.

**Implementability:** HIGH — process earnings call transcripts, extract sentiment features, build combined textual+numerical alpha model, evaluate in backtesting framework from Week 5

**Prerequisites:** Week 4 (alpha framework), Week 5 (backtesting)

**Prerequisite explanations needed:** None (LLM/NLP methodology assumed known)

**Multi-topic note:** NLP and alternative data are deeply intertwined — text IS alternative data. Agentic AI is a brief sidebar within the NLP context, not a separate theme.

---

#### Week 8: Deep Learning for Financial Time Series

**Band:** 2

**Topics:**
- DL for financial time series: LSTM/GRU for volatility forecasting, temporal convolutions, attention mechanisms on financial data (COVERED)
- Financial foundation models: Kronos, FinCast, the "do financial time-series FMs work?" debate, generic vs. domain-specific FMs (TOUCHED)
- The "simple model beats DL" reality: when DL works in finance (NLP, signal combination, execution) vs. when it doesn't (end-to-end price prediction from price data alone) — integrated throughout as a framing lens (TOUCHED)

**Career paths served:** ML researcher (Priority 1), QR buy-side, DL-focused firms (XTX, G-Research, Point72/Cubist)

**Research synthesis grounding:** SHOULD KNOW: DL for Financial Time Series (elevated to COVERED). GOOD TO KNOW: Financial Foundation Models.

**Implementability:** HIGH — LSTM vs. GARCH vs. HAR showdown on real volatility data, compare financial FMs to fine-tuned baselines, benchmark everything with proper temporal splits from Week 5

**Prerequisites:** Week 2 (financial time series), Week 4 (ML framework), Week 5 (evaluation methodology)

**Prerequisite explanations needed:** None (DL architectures assumed known — brief "here's why these architectural properties matter for THIS financial problem" framing only)

**Multi-topic note:** Foundation models are a natural extension of "DL for time series." The honest comparison with simpler models is not a separate topic but a lens applied throughout.

---

#### Week 9: Derivatives Pricing & the Greeks

**Band:** 2

**Topics:**
- Derivatives / options pricing: Black-Scholes, put-call parity, option payoffs, the Greeks (delta, gamma, vega, theta), hedging strategies (COVERED)
- Implied volatility: vol smile, vol surface, SABR model introduction (TOUCHED)
- Numerical methods for derivatives: Monte Carlo pricing, binomial trees (TOUCHED)
- Regulatory context (MENTIONED) — Basel III/FRTB and how they affect derivatives businesses; SR 11-7 and model governance

**Career paths served:** Quant trader (Priority 3), vol researcher, desk quant (Priority 4), risk quant (Priority 3), QR buy-side (derivatives strategies)

**Research synthesis grounding:** SHOULD KNOW: Derivatives/Options Pricing (elevated to COVERED — 100% of MFE programs, major gap in prior course versions). SHOULD KNOW: Volatility Modeling (IV surfaces). SHOULD KNOW: Numerical Methods. GOOD TO KNOW (sell-side): Regulatory Frameworks.

**Implementability:** MEDIUM-HIGH — implement BS with autograd for Greeks computation, fit SABR to market vol surface, Monte Carlo pricing for exotics, compare model prices to market data

**Prerequisites:** Week 2 (volatility fundamentals)

**Prerequisite explanations needed:** Stochastic calculus intuition — Ito's lemma as "chain rule with an extra term," risk-neutral pricing as "price = discounted expectation under a special probability measure," BS derivation at high level. Skip: SDE convergence proofs, measure-theoretic foundations, Girsanov theorem derivation. (15 min)

**Multi-topic note:** Options, vol surfaces, and numerical methods form a tight conceptual unit. The regulatory sidebar is brief sell-side awareness context.

---

#### Week 10: Causal Inference & Modern Factor Investing

**Band:** 2

**Topics:**
- Causal inference for finance: directed acyclic graphs (DAGs), collider bias, confounders in financial data, do-calculus intuition, instrumental variables (COVERED)
- Causal factor investing: the "factor mirage" concept, why correlation-based factors are causally misspecified, how causal reasoning changes portfolio construction (COVERED)
- Interpretable/explainable ML: SHAP values in financial models, attention visualization, regulatory requirements for explainability, the "white-box finance" movement (TOUCHED)

**Career paths served:** QR buy-side, factor investing, asset management, MRM quant (Priority 3 — interpretability)

**Research synthesis grounding:** SHOULD KNOW: Causal Inference (elevated to COVERED — "rapidly growing differentiator"). SHOULD KNOW: Interpretable/Explainable ML.

**Implementability:** MEDIUM-HIGH — build causal graphs for financial systems, detect collider bias in simulated and real factor data, compute SHAP values for financial models, compare causal vs. correlational factor portfolios

**Prerequisites:** Week 3 (factor models), Week 4 (ML models to interpret)

**Prerequisite explanations needed:** Causal graph basics — DAGs, d-separation, interventions vs. observations. Not "math" but domain-neutral methodology that many ML experts may not know. (10 min)

**Multi-topic note:** Causal inference and interpretable ML share a theme: "understanding what your model is actually learning and whether it will generalize." Causal factor investing is the direct financial application.

---

#### Week 11: Bayesian Methods, Uncertainty & Decision-Making

**Band:** 2

**Topics:**
- Bayesian DL for finance: MC dropout, deep ensembles, calibration, predictive uncertainty in financial predictions (TOUCHED to COVERED)
- Bayesian statistics in financial context: informative priors for financial parameters, posterior shrinkage, connection to Ridge/regularization (TOUCHED)
- Position sizing revisited: connecting uncertainty estimates to bet sizing, from Week 6 Kelly to uncertainty-aware position sizing (TOUCHED)
- Conformal prediction (MENTIONED ABSTRACTLY) — distribution-free uncertainty quantification, regulatory appeal
- Model risk management (MENTIONED ABSTRACTLY) — SR 11-7, growing need for ML model validation in banking

**Career paths served:** QR buy-side, risk management, ML researcher, MRM quant (Priority 3)

**Research synthesis grounding:** GOOD TO KNOW (buy-side): Bayesian DL & Uncertainty (elevated to TOUCHED-to-COVERED). SHOULD KNOW: Bayesian Statistics, Position Sizing (continued from Week 6). NICHE: Conformal Prediction, Model Risk Management.

**Implementability:** HIGH — implement MC dropout and deep ensembles on financial predictions, produce calibration plots, build uncertainty-aware position sizer, compare to fixed-size positions on real data

**Prerequisites:** Week 4 (ML models), Week 6 (position sizing basics)

**Prerequisite explanations needed:** Bayesian inference basics — prior, likelihood, posterior; conjugate priors for linear regression; connection to Ridge regression. Skip: variational inference derivation, MCMC convergence diagnostics. (10 min)

**Multi-topic note:** Theme: "what do we NOT know, and how does that change our decisions?" Bayesian DL produces uncertainty, position sizing consumes it. This is a decision pipeline: estimate → quantify uncertainty → size position.

---

#### Week 12: Fixed Income & Interest Rates

**Band:** 2

**Topics:**
- Fixed income fundamentals: bonds, yield curves, duration, convexity, bootstrapping zero-coupon rates from treasury data (TOUCHED)
- Interest rate models: Vasicek/CIR intuition, term structure modeling concepts (TOUCHED — conceptual, not derivation-heavy)
- Credit risk / default prediction (MENTIONED) — structural vs. reduced-form models, CDS basics, the "quant credit takes center stage" trend
- C++ in finance (MENTIONED) — importance for pricing libraries and low-latency systems, career paths that require it, resources for self-study

**Career paths served:** Rates quant, risk quant (Priority 3), vendor quant (Bloomberg, BlackRock), desk quant (Priority 4)

**Research synthesis grounding:** SHOULD KNOW: Fixed Income (TOUCHED — "half of global capital markets"). GOOD TO KNOW (buy-side): Credit Risk. SHOULD KNOW: C++ (MENTIONED — course cannot teach it but must acknowledge importance).

**Implementability:** MEDIUM — download yield curve data from FRED, bootstrap zero-coupon rates, compute duration/convexity, analyze how bond portfolios differ from equity, basic credit spread analysis

**Prerequisites:** Week 2 (time series), Week 6 (portfolio context)

**Prerequisite explanations needed:** None beyond what is woven into the content

**Multi-topic note:** Fixed income and credit risk are naturally related (bonds → credit spreads → default). C++ is mentioned here because it is most relevant for the sell-side/pricing roles that fixed income naturally points toward.

---

### ====================================================
### CUTOFF MARKER: STRONGLY RECOMMENDED (After Week 12)
### ====================================================

**What the student has:** Essential core + NLP for alpha, DL for time series, derivatives pricing, causal inference, Bayesian uncertainty, fixed income awareness. Broad applied competence across equities, text, options, and bonds.

**Suitable for:** Competitive for Priority 1 roles (ML Engineer, NLP/AI Engineer, Data Scientist, QR buy-side) and most Priority 2 roles.

### ====================================================

---

### BAND 3: Important Specializations

**Theme:** Deeper domain knowledge and advanced ML applications for specific career tracks.
**Who it serves:** Specific Priority 1/2 roles (execution quant, stat-arb QR, ML researcher) and Priority 3 roles.
**Research grounding:** All GOOD TO KNOW (buy-side) topics reach at least TOUCHED depth by the end of this band.

---

#### Week 13: Market Microstructure & Optimal Execution

**Band:** 3

**Topics:**
- Market microstructure (advanced): limit order book dynamics, market impact models (Kyle, Almgren-Chriss), order flow imbalance (OFI), price impact estimation (COVERED)
- Execution algorithms: TWAP, VWAP, Implementation Shortfall, Almgren-Chriss optimal execution framework (TOUCHED)
- Stochastic control / HJB (MENTIONED) — how optimal execution relates to HJB equations; conceptual bridge for interested students

**Career paths served:** Execution quant (Priority 2), HFT, market maker, e-trading

**Research synthesis grounding:** GOOD TO KNOW (buy-side): Market Microstructure advanced (elevated to COVERED — highly implementable, underappreciated career path). GOOD TO KNOW (sell-side): Stochastic Control / HJB.

**Implementability:** HIGH — visualize LOB data, compute OFI, verify the linear OFI-price relationship, implement Almgren-Chriss, simulate execution with market impact

**Prerequisites:** Week 1 (market fundamentals), Week 5 (backtesting)

**Prerequisite explanations needed:** None

**Multi-topic note:** Microstructure and execution are tightly coupled — understanding WHY markets move (microstructure) directly informs HOW to execute optimally.

---

#### Week 14: Statistical Arbitrage & Regime Detection

**Band:** 3

**Topics:**
- Statistical arbitrage: cointegration (Engle-Granger, Johansen), Ornstein-Uhlenbeck process, pairs and basket trading, mean-reversion strategies (TOUCHED to COVERED)
- Regime detection: Hidden Markov Models for bull/bear markets, Markov-switching GARCH, online changepoint detection (TOUCHED)
- Geopolitical risk modeling (MENTIONED ABSTRACTLY) — risk dashboards, ML for tail risk, regime changes from geopolitical events

**Career paths served:** QR buy-side (stat-arb firms), systematic macro, risk management

**Research synthesis grounding:** GOOD TO KNOW (buy-side): Statistical Arbitrage, Regime Detection / HMMs. NICHE: Geopolitical Risk.

**Implementability:** HIGH — test cointegration on real stock pairs, fit OU model and estimate parameters, build a pairs trading strategy, fit HMM to market data, compare detected regimes to known market events

**Prerequisites:** Week 2 (time series), Week 5 (backtesting)

**Prerequisite explanations needed:** Cointegration as a concept — why two non-stationary series can have a stationary linear combination, and why this matters for trading. (5 min)

**Multi-topic note:** Stat arb and regime detection address related questions: "what's the current market regime?" and "can we exploit mean-reverting patterns within it?" Geopolitical risk is a brief sidebar on regime triggers.

---

#### Week 15: Reinforcement Learning for Finance

**Band:** 3

**Topics:**
- RL for finance: financial MDP formulation, reward shaping with financial objectives (Sharpe, drawdown penalties), environment design for trading and execution (COVERED)
- Hybrid RL: why pure RL fails in finance (non-stationarity, adversarial environment, sparse/noisy rewards), integrating domain knowledge and financial constraints into RL agents (TOUCHED)
- Multi-agent RL for market making (MENTIONED ABSTRACTLY) — near production at major firms, ICAIF research
- Pure RL decline (MENTIONED) — historical context; shift from 85% pure RL (2020) to hybrid dominance (2025)

**Career paths served:** Execution quant (Priority 2), ML researcher (Priority 1), portfolio optimization

**Research synthesis grounding:** GOOD TO KNOW (buy-side): RL for Finance (elevated to COVERED — ML experts already know RL; the financial formulation IS the content). DECLINING: Pure RL for Trading.

**Implementability:** HIGH — build a financial RL environment, train agent for execution optimization, compare pure RL vs. hybrid with domain constraints, benchmark against Almgren-Chriss from Week 13

**Prerequisites:** Week 5 (backtesting/evaluation), Week 6 (portfolio context). Week 13 recommended but not required.

**Prerequisite explanations needed:** None (RL fundamentals assumed known — brief "here's PPO/SAC recap and why these properties matter for financial MDPs" framing only)

**Multi-topic note:** Single theme: RL applied to finance. The hybrid approach and multi-agent research are subtopics within this theme, not separate topics.

---

#### Week 16: Deep Hedging & Neural Derivatives Pricing

**Band:** 3

**Topics:**
- Deep hedging: neural network surrogates for derivatives pricing, learning hedging strategies directly from data, hedging under transaction costs where BS fails (TOUCHED to COVERED)
- Neural calibration: using DL to calibrate stochastic vol models (Heston, SABR), speed vs. accuracy tradeoffs (TOUCHED)
- XVA (MENTIONED ABSTRACTLY) — CVA/DVA/FVA: what these adjustments are, why the sell-side cares, one resource
- Rough volatility (MENTIONED ABSTRACTLY) — frontier research, DL calibration connection

**Career paths served:** Derivatives QR, desk quant (Priority 4 — modern DL approach), vol researcher

**Research synthesis grounding:** GOOD TO KNOW (buy-side): Deep Hedging. NICHE: XVA, Rough Volatility.

**Implementability:** MEDIUM-HIGH — implement deep hedging agent under transaction costs, train neural net to calibrate Heston model, compare neural pricing speed vs. Monte Carlo

**Prerequisites:** Week 9 (derivatives fundamentals required)

**Prerequisite explanations needed:** None beyond Week 9

**Multi-topic note:** Deep hedging and neural calibration are two sides of the same coin: using neural networks to solve derivatives problems that are computationally expensive with classical methods. XVA and rough vol are brief contextual sidebars for sell-side awareness.

---

#### Week 17: Generative Models & Graph Networks for Finance

**Band:** 3

**Topics:**
- Diffusion models for finance: synthetic financial data generation, LOB simulation (TRADES framework), scenario generation, the "beyond Monte Carlo" argument for certain applications (TOUCHED)
- GNNs for finance: graph construction from correlation, sector, and supply-chain relationships; dynamic relational graphs; when GNN beats tree-based baselines (TOUCHED)
- Synthetic data generation: landscape overview, privacy-preserving generation, diffusion replacing GANs (MENTIONED)
- GANs for financial data (MENTIONED) — historical context, why diffusion models are displacing them

**Career paths served:** ML researcher (Priority 1), frontier QR, risk quant (scenario generation)

**Research synthesis grounding:** GOOD TO KNOW (buy-side): Diffusion Models, GNNs, Synthetic Data. DECLINING: GANs for Financial Data.

**Implementability:** MEDIUM-HIGH — train diffusion model on financial returns, generate synthetic data and evaluate statistical fidelity, build financial knowledge graph, train GNN for stock prediction, compare to tree-based baselines

**Prerequisites:** Week 4 (ML framework), Week 8 (DL context)

**Prerequisite explanations needed:** Diffusion model basics — forward (noise addition) / reverse (denoising) process, connection to score matching. GNN basics — message passing, graph convolution. Brief recaps for those who haven't worked with these recently. (10 min total)

**Multi-topic note:** Two distinct ML paradigms (generative and graph-based) both representing the research frontier of financial ML. Bundled because each alone is too narrow for a full week at TOUCHED depth, and both address the question "what novel ML architectures bring genuine value to finance?"

---

#### Week 18: DeFi, Emerging Markets & Frontier Topics

**Band:** 3

**Topics:**
- DeFi/tokenization: DEX mechanisms (AMMs, liquidity pools), tokenized real-world assets, TradFi-DeFi convergence, BlackRock/Franklin Templeton in production (MENTIONED to TOUCHED)
- ESG/climate risk (MENTIONED) — NGFS scenarios, carbon credit markets, data quality challenges, "gold-rush phase is over" reality
- Commodities / energy markets (MENTIONED ABSTRACTLY) — complexity from renewables, nodal pricing, growing market
- FX markets (MENTIONED ABSTRACTLY) — 24h market structure, carry trade, e-trading
- Agentic AI revisited (MENTIONED) — where the field stands after "agentic humility," research augmentation vs. autonomous trading
- Quantum computing for finance (MENTIONED ABSTRACTLY) — HSBC and Goldman pilot results, realistic timeline assessment

**Career paths served:** DeFi quant (emerging), macro roles, specialized asset class roles

**Research synthesis grounding:** GOOD TO KNOW (buy-side): DeFi/Tokenization, ESG/Climate Risk. NICHE: Commodities, FX, Agentic AI, Quantum Computing.

**Implementability:** MEDIUM — interact with DeFi protocols programmatically, analyze AMM dynamics and impermanent loss, apply ML to NGFS climate scenarios

**Prerequisites:** Week 1 (market fundamentals)

**Prerequisite explanations needed:** Blockchain/smart contract basics for DeFi section. (5 min)

**Multi-topic note:** Deliberate multi-topic survey week. Common thread: "financial markets and technologies beyond traditional equities." Each topic gets its labeled depth; bundled because none warrants a standalone week at the required depth. This is the most optional week in Band 3.

---

### =============================================
### CUTOFF MARKER: FULL COURSE (After Week 18)
### =============================================

**What the student has:** Essential core + high-value applications + specializations across microstructure, stat arb, RL, derivatives ML, generative models, and emerging markets. All MUST KNOW, SHOULD KNOW, and GOOD TO KNOW (buy-side) topics are addressed at their required depth.

**Suitable for:** Competitive for all Priority 1-3 roles. Strong specialization options based on which Band 3 weeks the student emphasizes.

### =============================================

---

### BAND 4: Optional Extensions

**Theme:** Deeper sell-side preparation and capstone integration. The core 18 weeks cover all required material at specified depths; these weeks provide additional depth for specific career paths.

---

#### Week 19 (Optional): Sell-Side Deep Dive

**Band:** 4

**Topics:**
- Stochastic calculus (deeper treatment): Ito calculus worked examples, risk-neutral pricing derivation, change of measure (TOUCHED)
- Advanced derivatives: exotic options, local vol / stochastic vol models, model calibration depth (TOUCHED)
- Regulatory frameworks: Basel III/FRTB details, capital requirements, model governance (TOUCHED)
- XVA deeper treatment: exposure profiles, collateral modeling, CVA computation (TOUCHED)
- Model risk management: validation methodology, SR 11-7 compliance requirements (TOUCHED)

**Career paths served:** Desk quant (Priority 4), XVA quant (Priority 4), risk quant (Priority 3), MRM quant (Priority 3)

**Research synthesis grounding:** SHOULD KNOW: Stochastic Calculus (deeper). NICHE: XVA, Model Risk Management. GOOD TO KNOW (sell-side): Regulatory Frameworks. All elevated from MENTIONED to TOUCHED for students who need it.

**Implementability:** MEDIUM — implement local vol calibration, compute CVA for a simple portfolio, apply model validation statistical tests

**Prerequisites:** Week 9, Week 16

**Prerequisite explanations needed:** None (builds on prior treatments)

**Multi-topic note:** This week exists for students explicitly targeting sell-side roles. It provides the additional depth those paths require but that the buy-side-focused core does not prioritize. Can be treated as a resource collection rather than a taught week.

---

#### Week 20 (Optional): The Full Pipeline — Integration & Career Navigation

**Band:** 4

**Topics:**
- The investment process as a system: data → signal → portfolio → execution → risk, and how these pieces interact (COVERED)
- The Sisyphus paradigm: why team-based approaches succeed and solo researchers fail; the meta-strategy paradigm (TOUCHED)
- Alpha decay as a systemic reality: building a sustainable research process, crowding risk, signal shelf-life (TOUCHED)
- Career path mapping: which weeks are most relevant for each career goal, what to learn next (TOUCHED)

**Career paths served:** ALL (meta-integration)

**Research synthesis grounding:** Non-obvious findings #3 (signal decay), #7 (risk = alpha in importance), #8 (Sisyphus paradigm).

**Implementability:** MEDIUM-HIGH — full pipeline capstone: team-based strategy from data to paper-trading

**Prerequisites:** Band 1 + at least 3 weeks from Band 2

**Prerequisite explanations needed:** None

**Multi-topic note:** Integration week. Not new content but synthesis of everything. The capstone project ties together the complete pipeline.

---

## 3. Cutoff Markers Summary

| Cutoff | After Week | What the Student Has | Target Roles |
|---|---|---|---|
| **Essential Core** | 6 | Complete alpha pipeline: data → features → model → backtest → portfolio | Entry-level data scientist at fund, junior QR, ML engineer (basic) |
| **Strongly Recommended** | 12 | Core + NLP, DL, derivatives, causal inference, uncertainty, fixed income | Priority 1 and most Priority 2 roles |
| **Full Course** | 18 | Everything above + microstructure, stat arb, RL, deep hedging, generative models, DeFi | All Priority 1-3 roles; strong specialization options |
| **Extended** | 20 | Full course + sell-side depth + integration capstone | Includes Priority 4 paths; full synthesis |

---

## 4. Career Track Analysis

### 4.1 Career Paths This Course Primarily Serves

**ML Engineer at Fund** (Priority 1) — Most direct skill transfer.
- Core weeks: Band 1 (all), Week 7, Week 8, Week 15, Week 17
- Gap remaining after course: C++ for production systems, firm-specific infrastructure
- Assessment: Course provides all financial context needed for this role

**NLP/AI Engineer** (Priority 1) — Fastest-growing role type.
- Core weeks: Band 1 (all), Week 7 (primary), Week 8, Week 10
- Gap remaining: LLM production deployment at scale, firm-specific data pipelines
- Assessment: Course provides financial NLP expertise that is in extreme demand with few qualified candidates

**Data Scientist at Fund** (Priority 1) — Most accessible entry point.
- Core weeks: Band 1 (all), Week 7, Week 10, Week 14
- Gap remaining: Firm-specific tooling, heavy SQL/data pipeline work (woven in but not deep)
- Assessment: Course bridges the primary gap — financial domain knowledge — that data scientists lack

**Quant Researcher, Buy-Side** (Priority 1) — Core transition target, broadest scope.
- Core weeks: ALL of Bands 1 and 2, most of Band 3
- Gap remaining: Firm-specific research process, probability puzzle interview prep
- Assessment: Course provides the broadest and deepest preparation of any career path

### 4.2 Career Paths with Partial Coverage

**Execution Quant** (Priority 2) — Strong partial coverage.
- Key weeks: Week 1, Week 13 (primary), Week 15
- Coverage gap: Low-latency programming, C++ depth, tick-data infrastructure at scale
- Where to look next: "Trading and Exchanges" (Harris), C++ courses, QuantLib

**Quant Developer / Research Engineer** (Priority 2) — Moderate coverage.
- Key weeks: Band 1 (all), Week 12 (C++ awareness)
- Coverage gap: C++ proficiency (major), system design, distributed computing
- Where to look next: "C++ Design Patterns and Derivatives Pricing" (Joshi), system design resources

**Quant Trader** (Priority 3) — Moderate coverage.
- Key weeks: Week 1, Week 9 (derivatives), Week 6
- Coverage gap: Real-time decision-making, options intuition at speed, probability puzzles
- Where to look next: "Heard on the Street" (Crack), "A Practical Guide to Quantitative Finance Interviews" (Zhou), paper trading practice

**Risk Quant** (Priority 3) — Moderate coverage.
- Key weeks: Week 6, Week 9, Week 11, Week 12
- Coverage gap: Regulatory depth (Basel III/FRTB), stress testing methodologies, product breadth
- Where to look next: FRM certification, "Market Risk Analysis" (Alexander)

**Fintech/Vendor Quant** (Priority 2) — Moderate coverage.
- Key weeks: Band 1 (all), Week 9, Week 12
- Coverage gap: Product breadth (Bloomberg requires ALL asset classes), client-facing development
- Where to look next: Firm-specific documentation, CFA for breadth

### 4.3 Career Paths Explicitly Out of Scope

**Desk Quant / Strat** (Priority 4) — Week 19 (optional) provides a starting point, but this path requires mastery-level stochastic calculus, production C++, and deep derivatives pricing across multiple asset classes. Start with: Shreve's "Stochastic Calculus for Finance," Joshi's "C++ Design Patterns and Derivatives Pricing," CQF certification.

**XVA Quant** (Priority 4) — Week 19 (optional) provides awareness, but this path requires advanced counterparty credit risk modeling, Monte Carlo for exposure simulation, and regulatory capital computation. Start with: Green's "XVA: Credit, Funding and Capital Valuation Adjustments," CQF XVA module.

---

## 5. Coverage Matrix

### MUST KNOW Topics

| Topic | Week(s) | Depth | Notes |
|---|---|---|---|
| Probability & Statistics | ALL | WOVEN IN | Finance-specific applications (IC, information ratio, deflated Sharpe) within their contexts |
| Time-Series Analysis (ARMA, GARCH, frac diff) | 2 | COVERED | Full treatment with real data |
| How Markets Work | 1 | COVERED | Full treatment with data exploration |
| Portfolio Construction & Risk Management | 6 | COVERED | Full optimization + risk treatment |
| Backtesting Methodology | 5 | COVERED | Purged CV, backtest overfitting, WFO |
| Transaction Cost Modeling | 5 | COVERED | Spread, impact, slippage |
| ML for Alpha/Signal Generation | 4 | COVERED | Gu-Kelly-Xiu framework, financial loss functions |
| Financial Feature Engineering | 3, 4 | COVERED | Starts in W3, continues in W4 |
| Hypothesis Testing (finance-specific) | 5 | COVERED | Multiple testing, deflated Sharpe |
| Python | ALL | WOVEN IN | Assumed known; used throughout |
| Production Code Quality | ALL | WOVEN IN | Enforced through homework standards |
| Git / Version Control | ALL | WOVEN IN | Assumed known |
| Working with Financial Datasets | 1 | COVERED | Survivorship bias, point-in-time, corporate actions |
| Communication | ALL | WOVEN IN | Written analysis in assignments |
| Research Methodology / Data Snooping | 5 | COVERED | Lopez de Prado methodology |

### SHOULD KNOW Topics

| Topic | Week(s) | Depth | Notes |
|---|---|---|---|
| Derivatives / Options Pricing | 9 | COVERED | BS, Greeks, hedging, put-call parity |
| Factor Models | 3 | COVERED | CAPM → FF → Barra |
| Fixed Income | 12 | TOUCHED | Yield curves, duration, bootstrapping |
| Stochastic Calculus | 9 (prereq section) | PREREQ EXPLAINED | Ito intuition, risk-neutral pricing concept |
| Optimization Methods | 6 | WOVEN IN | Within portfolio construction as QP |
| NLP/LLM for Finance | 7 | COVERED | FinBERT, LLM embeddings, earnings |
| DL for Financial Time Series | 8 | COVERED | LSTM, attention, financial FMs |
| Volatility Modeling | 2, 9 | TOUCHED | GARCH basics in W2, IV surfaces in W9 |
| Alternative Data | 4, 7 | TOUCHED | Features in W4, NLP text in W7 |
| Causal Inference | 10 | COVERED | DAGs, factor mirage, causal factors |
| Bayesian Statistics | 11 | TOUCHED | Priors, posteriors, financial context |
| Risk Metrics (VaR, ES) | 6 | TOUCHED | Within portfolio/risk week |
| Position Sizing / Kelly | 6, 11 | TOUCHED | Kelly in W6, uncertainty-aware in W11 |
| Interpretable/Explainable ML | 10 | TOUCHED | SHAP, regulatory context |
| Data Engineering / ETL | ALL | WOVEN IN | Messy data by default in assignments |
| PyTorch / TensorFlow | ALL | WOVEN IN | Used as tool, not taught |
| SQL | ALL | WOVEN IN | Through data exercises |
| C++ | 12 | MENTIONED | Importance acknowledged, resources provided |
| Linux | ALL | WOVEN IN | Through tooling |
| Numerical Methods | 9 | TOUCHED | MC and binomial trees within derivatives |
| Cross-Functional Collaboration | ALL | WOVEN IN | Team components in projects |

### GOOD TO KNOW (Buy-Side) Topics

| Topic | Week(s) | Depth | Notes |
|---|---|---|---|
| Market Microstructure (advanced) | 13 | COVERED | LOB, OFI, market impact |
| Statistical Arbitrage | 14 | TOUCHED-COVERED | Cointegration, OU, pairs trading |
| Regime Detection / HMMs | 14 | TOUCHED | HMM, changepoint detection |
| RL for Finance | 15 | COVERED | Financial MDP, hybrid RL |
| Bayesian DL & Uncertainty | 11 | TOUCHED-COVERED | MC dropout, deep ensembles |
| Deep Hedging | 16 | TOUCHED-COVERED | Neural pricing, hedging under TC |
| Diffusion Models for Finance | 17 | TOUCHED | Synthetic data, LOB simulation |
| Financial Foundation Models | 8 | TOUCHED | Kronos, FinCast debate |
| Synthetic Data Generation | 17 | MENTIONED | Overview within generative models |
| DeFi/Tokenization | 18 | MENTIONED-TOUCHED | AMMs, tokenized assets |
| GNNs for Finance | 17 | TOUCHED | Graph construction, baseline comparison |
| GPU Computing | ALL | WOVEN IN | Through DL assignments |
| ESG/Climate Risk | 6, 18 | MENTIONED | Sidebars in portfolio + emerging topics |
| Credit Risk | 12 | MENTIONED | Sidebar in fixed income week |

### GOOD TO KNOW (Sell-Side) Topics

| Topic | Week(s) | Depth | Notes |
|---|---|---|---|
| Regulatory Frameworks (Basel III, FRTB, SR 11-7) | 9 | MENTIONED | Sidebar in derivatives week |
| Stochastic Control / HJB | 13 | MENTIONED | Sidebar in execution week |

### NICHE Topics

| Topic | Week(s) | Depth | Notes |
|---|---|---|---|
| XVA (CVA/DVA/FVA) | 16 | MENTIONED ABSTRACTLY | Sidebar in deep hedging week |
| Model Risk Management (SR 11-7) | 11 | MENTIONED ABSTRACTLY | Sidebar in uncertainty week |
| Commodities / Energy Markets | 1, 18 | MENTIONED ABSTRACTLY | Sidebars in markets + emerging topics |
| FX Markets | 1, 18 | MENTIONED ABSTRACTLY | Sidebars in markets + emerging topics |
| Conformal Prediction | 11 | MENTIONED ABSTRACTLY | Brief note in uncertainty section |
| Multi-Agent RL | 15 | MENTIONED ABSTRACTLY | Brief note in RL week |
| Geopolitical Risk | 14 | MENTIONED ABSTRACTLY | Sidebar in regime detection |
| Agentic AI | 7, 18 | MENTIONED | Sidebar in NLP + revisited in emerging |
| Rough Volatility | 16 | MENTIONED ABSTRACTLY | Brief note in deep hedging |
| Quantum Computing | 18 | MENTIONED ABSTRACTLY | Brief note in emerging topics |

### DECLINING Topics

| Topic | Treatment | Notes |
|---|---|---|
| KDB/Q | MENTIONED ABSTRACTLY (W1) | "You may encounter this; being replaced" |
| Pure RL for Trading | Absorbed into hybrid RL (W15) | Taught as what NOT to do |
| GANs for Financial Data | MENTIONED (W17) | Historical context; diffusion models preferred |
| R Programming | SKIP | Python covers all use cases |
| MATLAB | SKIP | Essentially disappeared from curricula |
| VBA | SKIP | Legacy |

---

## 6. Gap List

### Topics NOT Covered (with justification)

| Topic | Reason for Exclusion |
|---|---|
| Behavioral Finance | Minimal direct demand in job listings. Skip per research synthesis. |
| Operational Risk | Bank-specific; "almost never taught in MFE programs." Skip. |
| Regulatory Capital (deep treatment) | Irrelevant for buy-side target audience. MENTIONED briefly in Week 9 sidebar; deeper treatment only in optional Week 19. |
| R Programming | Python covers all use cases. Not worth screen time. |
| MATLAB | Essentially disappeared from modern curricula. |
| VBA | Legacy technology with no growth. |
| Rust | Too early-stage for quant finance. Growing in crypto but not yet relevant for most roles. |
| Federated Learning | Too early, too niche (<5% of roles). |
| Energy Trading (deep) | Specialized sub-domain; MENTIONED ABSTRACTLY only. |
| Post-Quantum Cryptography | Infrastructure/engineering concern, not quant-specific. |

### Topics with Less Depth Than Some Students May Need

| Topic | What We Cover | What's Missing | Who Needs More |
|---|---|---|---|
| Stochastic Calculus | Intuition only (prereq section in W9) | Full measure-theoretic treatment, Girsanov, SDE convergence | Desk quant, XVA quant |
| C++ | Mentioned importance + resources (W12) | Any actual programming instruction | Quant developer, HFT |
| Derivatives Pricing | Core options + vol surfaces (W9, W16) | Exotic products, multi-asset, deep model calibration | Desk quant, vol trader |
| Fixed Income | Yield curves + duration (W12) | Interest rate derivatives, convexity trades, MBS | Rates quant |
| Regulatory Frameworks | Brief awareness (W9 sidebar) | Implementation knowledge for Basel III/FRTB | Risk quant, MRM quant |
| Low-Latency Systems | Not covered | C++/FPGA, network optimization, co-location | HFT, quant developer |

---

## 7. Woven-In Topic Manifest

### Supporting Topics (never standalone, appear through assignments/standards)

| Woven-In Topic | Where It Appears | Mechanism |
|---|---|---|
| Probability & Statistics | ALL weeks | Used in every analysis; finance-specific metrics (IC, Sharpe, information ratio) taught within their application context |
| Python | ALL weeks | Every exercise uses Python; NumPy/Pandas idioms throughout |
| Production Code Quality | ALL homework | Grading criteria include code organization, documentation, proper abstractions |
| Git / Version Control | ALL homework | Submission via version control; branching for projects |
| Communication | Weeks 4-18 | Written analysis sections in every homework; discussion components in seminars |
| Data Engineering / ETL | Weeks 1, 4, 7, 8, 13, 14 | Every applied week starts with messy data; pipeline quality graded |
| SQL | Weeks 1, 3, 4, 12 | Data-loading exercises use SQL where appropriate |
| PyTorch / TensorFlow | Weeks 4, 8, 9, 11, 15, 16, 17 | Used as tool in DL weeks; never taught as topic |
| Linux | Weeks 1, 8, 15 | Through tooling and environment setup |
| GPU Computing | Weeks 8, 15, 17 | DL assignments benefit from GPU; practical notes included |
| Optimization Methods | Week 6 | Markowitz as QP within portfolio construction |
| Cross-Functional Collaboration | Weeks 15, 20 | Team-based project components |
| Research Methodology (generic) | ALL modeling weeks | Proper temporal CV, purged splits, honest evaluation as default |

### Recurring Themes from Non-Obvious Findings

| Theme | Where It Appears | How It Manifests |
|---|---|---|
| Signal decay (5-10%/year) | Weeks 4, 5, 7, 8, 10, 14 | Recurring theme in every signal-generation week; crowding risk, alpha shelf-life |
| Risk = alpha generation in importance | Weeks 5, 6, 11, 14, 15 | Risk thinking integrated into strategy weeks; not relegated to one week |
| Simple models often beat complex in production | Weeks 4, 8, 15 | Honest comparison of simple vs. complex in every ML application week |
| Data cleaning is 70-80% of the job | ALL applied weeks | Messy data by default; pipeline quality graded; data cleaning is not a topic but the REALITY |
| Interview-relevant probability | Weeks 2, 5, 6, 9 | Financial concepts framed through probability lens where natural |
| AI coding assistants: use thoughtfully | ALL homework | Assignment design accounts for LLM assistance; analytical reasoning cannot be outsourced |
| Kaggle as credible entry signal | Weeks 4, 8, 17 | Competition-format assignments where appropriate |
| Sisyphus paradigm (team > solo) | Weeks 15, 20 | Team-based projects; investment process taught as a system |

---

## 8. Non-Obvious Findings Integration

Mapping the 12 non-obvious findings from RESEARCH_SYNTHESIS.md Section 6 to their integration points:

| # | Finding | Integration Mode | Where |
|---|---|---|---|
| 1 | Finance knowledge NOT required at top firms | Preamble context | Section 1.7 |
| 2 | Simple regression often beats DL in production | Core teaching moment | Week 8 (framing), Weeks 4, 15 |
| 3 | Signals decay 5-10% per year | Recurring theme | Weeks 4, 5, 7, 8, 10, 14 |
| 4 | Data cleaning is 70-80% of the job | WOVEN IN | Every applied week; messy data by default |
| 5 | Competition is physics/math PhDs | Preamble context | Section 1.7; motivates financial mathematical vocabulary |
| 6 | Interview topics haven't changed much | WOVEN IN | Probability/statistics framed through interview lens in Weeks 2, 5, 6, 9 |
| 7 | Risk management = alpha generation | Recurring theme | Weeks 5, 6, 11, 14, 15 |
| 8 | Sisyphus paradigm (solo researchers fail) | Team project design | Weeks 15, 20; investment process as system |
| 9 | AI coding assistants make experienced devs slower | WOVEN IN | Assignment design philosophy throughout |
| 10 | C++ demand surging | MENTIONED | Week 12 + sidebar awareness throughout |
| 11 | Kaggle as credible entry signal | Assignment design | Competition-format exercises in Weeks 4, 8, 17 |
| 12 | DeFi-TradFi convergence | Dedicated content | Week 18 |

---

## 9. What This Course Does Not Prepare You For

### Career Paths Beyond Scope

**Desk Quant / Strat** — Requires mastery-level stochastic calculus, production C++, and deep derivatives pricing across multiple asset classes. The course provides awareness and intuition (Weeks 9, 16, optional Week 19) but not the depth needed. Resources: Shreve's "Stochastic Calculus for Finance," Joshi's "C++ Design Patterns and Derivatives Pricing," CQF certification.

**XVA Quant** — The most technically demanding sell-side specialty. Requires advanced counterparty credit risk, exposure modeling, and regulatory capital computation. The course provides brief awareness (Week 16 sidebar, optional Week 19). Resources: Green's "XVA: Credit, Funding and Capital Valuation Adjustments," CQF XVA module.

### Skills Beyond Scope

**C++ Programming** — Acknowledged as important (surging demand per research) but teaching C++ to ML/DL experts is a full course in itself. Resources: Joshi's "C++ Design Patterns and Derivatives Pricing," Meyers' "Effective Modern C++."

**Low-Latency Systems Design** — Critical for HFT and execution infrastructure. Not covered. Resources: "Trading and Exchanges" (Harris), "All About High-Frequency Trading" (Durbin).

**Full Stochastic Calculus** — Only intuition-level treatment provided (Week 9 prereq section). Students targeting derivatives-heavy roles need: Shreve's "Stochastic Calculus for Finance" (Volumes I and II).

**Interview Preparation** — The course builds knowledge but does not explicitly train for quant interview formats (probability puzzles, brain teasers, live coding). Resources: "Heard on the Street" (Crack), "A Practical Guide to Quantitative Finance Interviews" (Zhou), "The Green Book" (Joshi).

### Topics Not Addressed

Behavioral finance, operational risk, full regulatory compliance (Basel III/FRTB implementation details), Rust programming, federated learning for finance, quantum computing (beyond brief mention).
