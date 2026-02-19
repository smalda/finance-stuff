# Quant Finance Skill Map for ML/DL Experts (2025–2026)
*A comprehensive, exploration-first guide to what an experienced ML/DL practitioner should learn to pivot into quantitative finance and adjacent “money” fields.*

> **Audience:** Senior/experienced ML/DL engineers/researchers (not ML beginners)  
> **Scope:** Global (US/UK/EU/Asia), senior pivot expectations  
> **Purpose:** A standalone landscape map to drive course restructuring (18-week “ML for Quant Finance”)  
> **Method:** Work backward from industry demand (job descriptions + curricula + community signal), then classify.

---

## 0) How to read this document

### What this guide *does*
- Maps the **full landscape** of quant/finance knowledge relevant to ML/DL experts.
- Separates **table-stakes vs differentiators vs niche**, and links them to **role families**.
- Emphasizes **finance-specific application** of ML/DL: data quirks, feature engineering, losses, metrics, risk constraints, and deployment realities.
- Identifies **2025–2026 growth areas** (LLMs, alternative data, modern execution, model risk governance, distributed compute) and **declining areas** (legacy tools, over-theoretical derivations in many roles).

### What this guide *does not*
- Teach ML/DL fundamentals (backprop, transformer derivations, LSTM gating, etc.).
- Provide “one true curriculum” for every career path. Instead it provides a **skill map** and **priorities**.

### Tier definitions (priority levels)
| Tier | Description | Action |
|---|---|---|
| **MUST KNOW** | Required for ~80%+ of relevant roles; table stakes | Core course content |
| **SHOULD KNOW** | Required for ~40–80%; strong differentiator | Include if space; integrate into projects |
| **GOOD TO KNOW** | Required for ~15–40%; valuable but specialized | Electives / optional modules |
| **NICHE** | Required for <15%; highly specialized | Brief awareness + resources |
| **DECLINING** | Historically important but being commoditized/automated | Mention for context; avoid deep training |

---

## 1) The quant finance role landscape (career paths & what they actually do)

Quant finance is not one job. Senior pivots succeed faster when they pick a **role family** and learn the relevant subset deeply.

### 1.1 Role families (global)
1. **Buy-side Quant Researcher (Alpha / Systematic Research)**
   - **Goal:** Discover predictive signals (“alpha”), build and validate strategies, research new data sources.
   - **Work:** Data assembly, feature engineering, model training, backtesting, risk/portfolio integration, monitoring.
   - **Skill profile:** Heavy stats + ML + time-series + research discipline + engineering hygiene.

2. **Quant Trader / Vol Trader / Market Maker**
   - **Goal:** Trade and manage risk, often in options/volatility or market-making contexts.
   - **Work:** Strategy execution, quoting, hedging, risk limits, intraday analysis, microstructure tactics.
   - **Skill profile:** Derivatives + microstructure + fast decision-making + strong coding + risk instincts.

3. **Sell-side Quant / Strat (Desk Quant)**
   - **Goal:** Build pricing/risk tools and models for a trading desk; support products and risk.
   - **Work:** Pricing libraries, calibration, scenario analytics, P&L explain, model documentation and approval.
   - **Skill profile:** Stochastic calculus + derivatives + numerical methods + robust engineering + governance.

4. **Quant Developer (Trading Systems / Platform / Research Infrastructure)**
   - **Goal:** Productionize models & build performant systems (data, execution, risk, research tooling).
   - **Work:** Low-latency systems, data pipelines, APIs, distributed compute, monitoring, reliability engineering.
   - **Skill profile:** Software engineering + performance + systems + enough quant literacy to implement correctly.

5. **Risk Quant / Model Risk / Validation (Bank/Asset Manager/Fintech)**
   - **Goal:** Validate models, quantify risk, ensure compliance with regulatory expectations.
   - **Work:** Independent testing, stress testing, documentation, explainability, bias controls, governance.
   - **Skill profile:** Risk + stats + interpretability + documentation + stakeholder communication.

6. **Quant PM (Systematic Portfolio Manager / Multi-strat PM)**
   - **Goal:** Combine signals into portfolios; manage risk/turnover; deliver stable risk-adjusted returns.
   - **Work:** Portfolio construction, allocation, constraints, execution cost, capacity analysis, regime management.
   - **Skill profile:** Alpha + portfolio theory + optimization + transaction costs + decision processes.

7. **Finance Data Scientist / Applied ML (Fintech / Market Intelligence / Asset Manager Tech)**
   - **Goal:** ML for finance problems beyond alpha: forecasting, client risk, fraud/AML, pricing, analytics.
   - **Work:** Feature engineering, pipelines, model deployment, monitoring, stakeholder collaboration.
   - **Skill profile:** ML + data engineering + domain framing + interpretability + product sense.

### 1.2 The “T-shaped” requirement in 2025–2026
Senior hires are expected to be **T-shaped**:
- **Horizontal bar (breadth):** basic finance domain + data realities + evaluation + engineering literacy.
- **Vertical bar (depth):** one or two specialties (e.g., volatility + options; microstructure + execution; NLP/alt data alpha; risk governance).

### 1.3 Quick mapping: role → emphasis
| Role family | Domain (finance) | Modeling | Engineering | Communication |
|---|---:|---:|---:|---:|
| Buy-side Quant Researcher | High | Very high | High | High |
| Quant Trader / Vol / MM | Very high | Medium–High | High | High |
| Sell-side Strat | Very high | Medium | High | Very high |
| Quant Developer | Medium | Medium | Very high | High |
| Risk/Model Risk | Very high | Medium | Medium | Very high |
| Quant PM | Very high | High | Medium | Very high |
| Finance Data Scientist | Medium | High | High | High |

---

## 2) The quant finance knowledge landscape (what to learn)

This section is the core “topic universe,” prioritized for an ML/DL expert pivoting into quant finance.

### 2.1 Market & products (domain fundamentals)

#### MUST KNOW
- **Market basics:** equities, FX, rates, credit, commodities; how instruments trade and settle.
- **Return definitions:** log returns vs arithmetic, corporate actions, dividends, splits, survivorship issues.
- **Microstructure basics:** bid/ask, spreads, slippage, order types, order book; liquidity and impact.
- **Portfolio performance & risk metrics:** Sharpe, Sortino, max drawdown, volatility, beta, information ratio.
- **Transaction costs & turnover:** why they destroy naive alpha; capacity and impact.

#### SHOULD KNOW
- **Derivatives essentials:** options (calls/puts), forwards/futures, volatility products, Greeks and intuition.
- **Rates/fixed income essentials:** yield curves, duration/convexity, curve trades, carry/roll-down.
- **Volatility dynamics:** implied vs realized vol, skew/smile, term structure, vol risk premium.

#### GOOD TO KNOW
- **Market conventions by asset class:** day count, quoting conventions, contract specs.
- **Credit products:** CDS basics, spread curves, default modeling intuition.
- **Commodity specifics:** seasonality, storage/carry, term structure, convenience yield.

#### NICHE
- **Exotics & structured products** (deep pricing, XVA) — strat territory.
- **Energy/power trading domain specifics**, freight, emissions — specialized desks.
- **Insurance/actuarial** overlaps — niche unless targeting that.

---

### 2.2 Financial math & statistics (the quant core)

#### MUST KNOW
- **Probability & statistics for finance:** heavy tails, skewness, kurtosis, dependence structures.
- **Non-stationarity & regimes:** structural breaks, regime shifts, rolling distributions.
- **Time-series basics:** autocorrelation/partial autocorrelation, volatility clustering, heteroskedasticity.
- **Hypothesis testing in research:** multiple testing, p-hacking avoidance, confidence intervals, bootstrap.

#### SHOULD KNOW
- **Stochastic processes intuition:** Brownian motion, mean reversion, jump processes (conceptual).
- **Simulation:** Monte Carlo, variance reduction, scenario generation, pathwise risk.
- **Optimization:** convex/non-convex, constrained optimization, robust formulations (portfolio constraints).

#### GOOD TO KNOW
- **Econometrics:** ARIMA/VAR, cointegration, GARCH variants, state space models/Kalman filters.
- **Causal inference mindset:** when correlations mislead; selection bias; backtesting bias.

#### NICHE
- **Full stochastic calculus derivations:** Ito calculus deep proofs (needed for some strat roles).
- **Advanced PDE/numerical schemes** for pricing — specialized.

---

### 2.3 Data realities in finance (where many ML experts stumble)

#### MUST KNOW
- **Data alignment & leakage:** time zone alignment, look-ahead bias, point-in-time data, timestamp integrity.
- **Corporate actions & survivorship:** adjusted prices, delistings, index reconstitutions.
- **Backfill & revision bias:** fundamentals revised; vendor “as-of” snapshots matter.
- **Label definition:** what is the target? next-bar return, event response, risk-adjusted payoff.
- **Sampling & microstructure noise:** bid-ask bounce, stale quotes, irregularly spaced events.
- **Missingness:** holidays, outages, symbol changes, trading halts.

#### SHOULD KNOW
- **Alternative data pitfalls:** coverage drift, licensing/legal constraints, selection bias, spurious correlations.
- **Feature stability & drift:** monitoring, online recalibration, regime-conditioned models.

#### GOOD TO KNOW
- **Data vendor ecosystem:** Bloomberg/Refinitiv, exchange feeds, fundamentals providers, alt data brokers.

#### NICHE
- **Tick reconstruction & feed handler engineering:** very specific to HFT quant dev.

---

### 2.4 ML/DL in finance: what changes (applied focus)

> You already know the architectures; the question is how finance breaks naïve ML and what adaptations matter.

#### MUST KNOW (finance-specific ML competence)
- **Feature engineering for markets:**
  - Price/volume/volatility features, cross-sectional ranks, rolling z-scores, event-based features.
  - Regime features (vol regime, macro regime, liquidity regime).
  - Microstructure features (order book imbalance, flow toxicity proxies, realized spread).
- **Objective functions that reflect trading reality:**
  - P&L-based objectives, risk-adjusted returns, drawdown penalties, turnover/transaction-cost penalties.
  - Asymmetric objectives (tail risk, downside focus).
- **Evaluation that matches deployment:**
  - Walk-forward validation, purged/embargoed CV, time-series split discipline.
  - Out-of-sample stability under regime change.
  - Metrics: Sharpe/Sortino, hit rate, profit factor, max drawdown, tail metrics, turnover, capacity.
- **Overfitting control in finance:**
  - Multiple testing control, feature selection discipline, pre-registration mindset, backtest hygiene.
  - Ensembles and model risk buffers; simplicity bias when signal-to-noise is low.
- **Risk constraints:**
  - Position limits, leverage constraints, exposure constraints (sector, beta, duration), liquidity constraints.

#### SHOULD KNOW (widely used / differentiating)
- **Deep learning for signals:**
  - Transformers/LSTMs for sequences, CNN-like models for order book “images”.
  - Representation learning for cross-sectional modeling; embedding instruments.
- **NLP for finance (rapidly growing):**
  - Earnings call transcripts, news, filings, social text.
  - LLMs for extraction, summarization, event detection, and structured signal creation.
- **Model interpretability in finance:**
  - SHAP/feature attributions, monotonic constraints for risk-sensitive use cases.
  - Explainability narratives for stakeholders.

#### GOOD TO KNOW (growing but not universal)
- **RL for execution & decision-making:**
  - Optimal execution, inventory control, dynamic hedging (“deep hedging”).
  - Sample efficiency issues, safety constraints, sim-to-real gaps.
- **Generative models:**
  - Synthetic data, scenario generation, stress testing, distribution modeling.
- **Causal ML / structural models:**
  - Treatment effect modeling for policy/macro events; robustness.

#### NICHE
- **Ultra-low latency ML inference** for HFT (hardware constraints, FPGA) — specialized.
- **Very large multi-modal models** for funds with massive compute — top-end but not universal.

---

### 2.5 Engineering & production: the quant “reality layer”

#### MUST KNOW
- **Python proficiency** for research and pipelines (vectorization, memory management, correctness).
- **Data engineering basics:** SQL, data modeling, time-series indexing, reproducible pipelines.
- **Versioning & reproducibility:** experiments, data snapshots, deterministic builds, audit trails.
- **Testing discipline:** unit tests for data transforms, backtests, and metrics; “research code” still needs rigor.
- **Monitoring:** drift, performance decay, latency, incident response basics.

#### SHOULD KNOW
- **Performance engineering:**
  - Profiling, parallelism, GPUs, batching, vectorized computation.
  - Basic C++/Rust familiarity if targeting low-latency roles.
- **Distributed compute:** Ray/Spark/Dask, cluster usage patterns; job orchestration.
- **Deployment patterns:** batch scoring vs real-time; model serving; feature stores; governance.

#### GOOD TO KNOW
- **Cloud concepts:** object storage, compute orchestration, IAM/security basics.
- **Time-series databases:** kdb+/q awareness (common in some trading shops).
- **ML Ops stack:** CI/CD for models, model registry, automated retraining.

#### NICHE
- **Exchange connectivity & feed handlers** — quant dev / HFT engineering.
- **FPGA / kernel bypass networking** — niche.

---

### 2.6 Governance, regulation, and “model risk”

#### MUST KNOW (for bank/asset manager/regulated contexts; SHOULD otherwise)
- **Model risk management basics:** documentation, validation, sign-off process, limitations.
- **Stress testing & scenario analysis:** model behavior in extreme regimes.
- **Explainability requirements:** why models must be interpretable (especially for credit/AML/risk).
- **Data governance & compliance:** licensing constraints, privacy, auditability.

#### SHOULD KNOW
- **Bias and fairness** in consumer finance ML (credit/underwriting).
- **Regulatory landscape awareness:** high-level understanding of frameworks affecting ML in finance.

#### NICHE
- **Deep regulatory specialization** (e.g., SR 11-7 in detail) — for dedicated model risk roles.

---

## 3) Topic priority matrix (classification)

> A compact, actionable prioritization for a senior ML/DL pivot.

### 3.1 Table: topics by tier
| Topic | Tier | Where it’s required (role examples) | Trend | Depth recommendation |
|---|---|---|---|---|
| Financial time-series quirks (non-stationarity, vol clustering) | MUST | All trading/research roles | Growing | Mastery |
| Backtesting discipline (purged CV, leakage control) | MUST | Quant research, PM, DS | Growing | Mastery |
| Portfolio/risk metrics (Sharpe, drawdown, turnover, costs) | MUST | All buy-side; many sell-side | Stable | Mastery |
| Market microstructure basics | MUST | Trading, execution, HFT-adjacent | Growing | Working mastery |
| Data leakage + point-in-time correctness | MUST | All modeling roles | Growing | Mastery |
| Transaction cost analysis (TCA) basics | MUST | Trading/PM/execution | Growing | Working mastery |
| Derivatives intuition (Greeks, vol) | SHOULD | Vol trader, strat, risk, PM | Stable | Working knowledge |
| Yield curve/duration basics | SHOULD | Rates, macro, strat, risk | Stable | Working knowledge |
| Deep learning for signals | SHOULD | Research roles at ML-heavy firms | Growing | Working knowledge to mastery |
| NLP/LLMs for finance text | SHOULD | Research, fintech DS, macro | Rapidly growing | Working knowledge + 1–2 projects |
| Interpretability + stability analysis | SHOULD | Risk/credit/regulated roles | Growing | Working knowledge |
| Distributed compute & GPU utilization | SHOULD | Large-scale shops, ML-heavy | Growing | Working knowledge |
| RL for execution/hedging | GOOD | Execution quant, research | Growing | Conceptual + small project |
| Generative scenario models | GOOD | Risk, research | Growing | Conceptual + awareness |
| Econometrics (cointegration, GARCH) | GOOD | Stat arb, macro, risk | Stable | Working knowledge |
| Full stochastic calculus derivations | NICHE | Certain strats/pricers | Stable | Awareness unless targeting |
| kdb+/q | NICHE | Some trading shops | Stable | Awareness |
| MATLAB/R as primary | DECLINING | Legacy teams | Declining | Awareness only |

---

## 4) What employers test (interview reality for senior pivots)

### 4.1 Common interview components
1. **Math + probability + stats reasoning**
   - Distributions, conditional probability, expectation/variance, inference intuition, optimization reasoning.
2. **Coding (often timed)**
   - Python common; C++ for low-latency roles; focus on correctness, performance, clarity.
3. **Research or system design**
   - Researchers: define target, features, backtest and validation, costs, risk constraints.
   - Quant dev: design data/execution pipelines, performance constraints, reliability/monitoring.
4. **Finance intuition**
   - Basic asset class literacy; where alpha might come from; how costs and risk kill naive models.
5. **Communication**
   - Explain assumptions, trade-offs, and failure modes to mixed audiences.

### 4.2 What senior candidates are expected to demonstrate
- **Correct problem formulation** (objective + constraints + evaluation).
- **Finance ML hygiene** (leakage control, non-stationarity, costs, crowding).
- **Reproducible research** with clear artifacts and narratives.
- **Production awareness**: how a model lives and dies in a live market system.

---

## 5) Emerging trends (2025–2026): what’s new and growing

### 5.1 LLMs and generative AI in the quant workflow
- LLM use cases:
  - Parsing filings/earnings calls/news into structured event variables.
  - Research acceleration: feature ideation, code generation, summarization, debugging.
- New expectation: *use LLMs responsibly* (evaluation, leakage control, governance).

### 5.2 Alternative data maturity
- The differentiator is less “having alt data” and more:
  - Coverage drift management, licensing awareness, version control, and robust validation.

### 5.3 Execution and microstructure sophistication
- More emphasis on:
  - Impact models, capacity analysis, turnover and slippage modeling.
  - Latency-aware deployment for short-horizon strategies.

### 5.4 Governance and auditability
- Particularly in banks and large managers:
  - Documentation, explainability, validation, reproducibility are increasingly central.

---

## 6) What’s declining / being commoditized

- **Heavy theoretical derivations as a standalone skill** (in many roles): pricing libraries reduce marginal value.
- **Legacy tools as core differentiators** (MATLAB as primary, heavy VBA): present but not growing.
- **Simple textbook alphas**: arbitraged; advantage shifts to robustness, data discipline, and engineering.
- **Manual research workflows**: modern stacks expect automation, tracking, and scalable compute.

---

## 7) Recommended learning depth by role family (pivot plans)

### 7.1 Buy-side Quant Researcher (alpha)
- **Mastery:** time-series quirks; backtest hygiene; costs/turnover; risk metrics; feature engineering.
- **Working knowledge:** microstructure; derivatives intuition; deep learning signals; NLP basics.
- **Optional:** RL for execution; generative scenarios; causal inference.
- **Proof:** 2–3 research-grade projects with walk-forward testing and realistic cost models.

### 7.2 Quant Trader / Vol trader / market maker
- **Mastery:** derivatives/vol; hedging; microstructure; risk limits.
- **Working knowledge:** ML for vol/liquidity; execution cost modeling.
- **Optional:** RL inventory control; low-latency inference.
- **Proof:** options/vol project + hedging simulation + cost-aware execution study.

### 7.3 Sell-side Strat
- **Mastery:** derivatives pricing intuition; calibration concepts; numerical methods; governance writing.
- **Working knowledge:** ML surrogates; scenarios; explainability.
- **Optional:** deep PDE/stochastic calculus (if targeting pure pricing research).
- **Proof:** surrogate pricer/calibration speed-up + documentation package.

### 7.4 Quant Developer
- **Mastery:** systems design; data pipelines; performance; testing; reproducibility; monitoring.
- **Working knowledge:** finance domain; model integration; latency-aware tradeoffs.
- **Optional:** C++ performance; kdb+/q; exchange connectivity.
- **Proof:** end-to-end backtest + execution pipeline or research infrastructure component.

### 7.5 Risk / Model Risk
- **Mastery:** risk measures; stress tests; interpretability; validation protocols; governance.
- **Working knowledge:** ML testing; drift detection; fairness in credit contexts.
- **Proof:** model validation report + stress test suite + interpretability analysis.

---

## 8) Implications for an 18-week ML-for-Quant-Finance course (high-level)

> Not a full syllabus—an evidence-aligned structure you can convert into weekly topics.

### 8.1 Suggested module clusters
1. **Finance foundations for ML experts (fast, pragmatic)**
   - Instruments, microstructure, costs, portfolio/risk metrics
2. **Data in finance**
   - Point-in-time data, leakage, corporate actions, backfill/revisions, labeling and sampling
3. **Backtesting and evaluation**
   - Walk-forward testing, purged CV, robustness, statistical discipline, stress tests
4. **Modeling for alpha and decision-making**
   - Cross-sectional vs time-series modeling, regime models, uncertainty, constraint-aware objectives
5. **NLP/LLMs for finance**
   - Text sources, event extraction, signal construction, evaluation
6. **Execution and constraints**
   - TCA, impact models, turnover, capacity, constraint-aware optimization
7. **Production and governance**
   - Reproducibility, monitoring, documentation, validation mindset

### 8.2 The course “contract” for ML/DL experts
For every model class:
- **Recap:** 1–2 paragraphs (architecture & why it matters in finance)
- **Deep dive:** finance problem framing, domain pitfalls, features, loss/metrics, deployment constraints

---

## 9) Practical reading list (starting points)
*(Keep the course self-contained; use this list as optional depth.)*
- Interview-style math/quant prep (for problem-solving patterns)
- Derivatives/volatility primers (intuition-first)
- Portfolio construction & risk texts
- Time-series/volatility modeling references
- Recent ML-for-finance surveys + workshop proceedings (ICAIF, NeurIPS/ICML finance workshops)
- Practitioner newsletters/blogs focused on robustness and postmortems

---

## 10) Appendix A — Finance-specific pitfalls checklist (use in every project)

**Data & labels**
- [ ] Point-in-time correctness (no look-ahead)
- [ ] Corporate actions handled
- [ ] Survivorship bias avoided
- [ ] Label reflects tradeable outcome and horizon

**Backtest**
- [ ] Walk-forward or proper temporal splits
- [ ] Transaction costs and slippage included
- [ ] Robustness checks across regimes
- [ ] Multiple-testing control / feature selection hygiene

**Risk**
- [ ] Exposure constraints considered
- [ ] Tail risk measured
- [ ] Drawdowns and turnover quantified

**Production realism**
- [ ] Latency requirements considered
- [ ] Monitoring plan for drift/performance decay
- [ ] Reproducible pipeline + experiment tracking

---

## 11) Appendix B — One-page summary (executive)

If you are a senior ML/DL practitioner pivoting into quant finance in 2025–2026, focus on:
1. **Finance domain fluency** (microstructure, costs, instruments, risk/portfolio metrics)
2. **Finance ML hygiene** (leakage control, backtesting rigor, regime awareness)
3. **Feature engineering + finance objectives** (P&L/risk-aware loss functions)
4. **Engineering discipline** (reproducible pipelines, testing, monitoring)
5. **Differentiators:** NLP/LLMs, alternative data discipline, execution modeling, governance/explainability (role-dependent)

---

*End of document.*
