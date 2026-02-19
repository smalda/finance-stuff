# Research Synthesis: ML for Quantitative Finance Course Restructure
## Comprehensive Reference Document (February 2026)

**Purpose:** Standalone guide answering "What should an ML/DL expert learn to break into quantitative finance?" Synthesized from 5 raw research reports covering job listings (45-50 roles at 23 firms), MFE syllabi (13 degree programs, 4 certifications), community/practitioner sources, and emerging trends research.

**Audience context:** The target learner already has mastery of ML/DL fundamentals (neural architectures, training methodology, standard ML, RL, LLMs). This document maps the *financial* and *applied* knowledge they need.

---

## Table of Contents

1. [Career Paths Map](#section-1-career-paths-map)
2. [Master Topic Classification](#section-2-master-topic-classification)
3. [What's Genuinely New in 2025-2026](#section-3-whats-genuinely-new-in-2025-2026)
4. [What's Declining or Overhyped](#section-4-whats-declining-or-overhyped)
5. [The Balance Question](#section-5-the-balance-question)
6. [Non-Obvious Findings](#section-6-non-obvious-findings)
7. [Key Questions Answered](#section-7-key-questions-answered)
8. [Source Summary](#section-8-source-summary)

---

## Section 1: Career Paths Map

### 1.1 Quantitative Researcher (Buy-Side)

**Description:** The core "idea person" at hedge funds and systematic asset managers. Conceives trading strategies, builds predictive models from data, validates through backtesting, and often owns the full pipeline from hypothesis to production signal.

**Where it exists:** Hedge funds (Jane Street, Citadel, Two Sigma, D.E. Shaw, Millennium, Point72/Cubist, HRT, AQR, Man AHL, Bridgewater, WorldQuant, Balyasny, Renaissance), prop trading firms (XTX, G-Research), systematic asset managers.

**Key skills:** Python (essential), statistics and probability (deep), ML/time-series analysis, backtesting methodology, experiment design, communication. PhD preferred at most top firms.

**Sub-specializations:** Alpha/signal researcher, ML researcher (DL-focused), volatility researcher, execution researcher, alternative data researcher, NLP/LLM researcher.

**How it differs:** Broadest scope per individual. Optimizes for alpha generation and P&L. Less regulatory burden than sell-side. More entrepreneurial -- you own outcomes.

**Relevance for ML/DL expert:** HIGH. This is the most natural transition path. ML methodology transfers directly; the gaps are financial domain knowledge, understanding of signal-to-noise in markets, and backtesting discipline. Finance knowledge explicitly NOT required at Jane Street, Two Sigma, RenTech, G-Research, D.E. Shaw, Tower Research.

**Compensation:** $165K-$350K base; total comp often 2-4x base.

---

### 1.2 Quantitative Trader

**Description:** Hybrid role combining quantitative research with real-time trading decisions. Makes pricing and risk decisions, often with direct P&L responsibility. More common at prop trading firms.

**Where it exists:** Optiver, IMC, Jane Street, SIG, Tower Research, Akuna Capital, Citadel Securities.

**Key skills:** Mental math, probability (intuitive, under pressure), game theory, options/derivatives pricing and Greeks, market-making intuition, quick decision-making. Programming is needed but less emphasis on production code.

**How it differs from QR:** More real-time decision-making, more options/derivatives knowledge, less deep learning emphasis, more emphasis on intuitive probability. Some quant traders "don't even code at all" and are more macro oriented.

**Relevance for ML/DL expert:** MODERATE. Requires strong probabilistic thinking and market intuition that ML backgrounds don't naturally provide. Interview process is heavily probability-puzzle-oriented. The transition is harder than QR.

**Compensation:** $175K-$300K base + performance-based bonus.

---

### 1.3 Quantitative Developer / Research Engineer

**Description:** The engineer who partners with researchers to implement, optimize, and deploy trading systems and research tools. Bridge between research and production.

**Where it exists:** Citadel Securities, Citadel (GQS), SIG, Tower Research, Akuna, AQR, Two Sigma, most large hedge funds.

**Key skills:** C++ proficiency (required, not optional), Python, data structures and algorithms, system design, distributed computing, Linux, software engineering practices (testing, version control), low-latency considerations.

**How it differs:** Much stronger engineering emphasis, less original statistical research, more focus on scalability, reliability, and performance. C++ is table stakes.

**Relevance for ML/DL expert:** HIGH (for those with strong engineering backgrounds). ML engineers from tech companies often have the systems skills. The gap is C++ proficiency and financial systems knowledge. This can be an accessible entry point with pathway to ML researcher roles.

**Compensation:** $200K-$350K base.

---

### 1.4 Desk Quant / Strat (Sell-Side)

**Description:** Front-office quants at banks who sit on or near trading desks. Build and maintain derivative pricing models that traders use. Goldman Sachs calls these "Strats" with sub-types: Algo/Core Strats, Data Strats, Desk Strats.

**Where it exists:** JPMorgan, Goldman Sachs, Morgan Stanley, Barclays, Deutsche Bank, UBS, BNP Paribas, Citigroup.

**Key skills:** Stochastic calculus (Ito's lemma, SDEs, Girsanov theorem), derivatives pricing theory (Black-Scholes, local vol, stochastic vol), numerical methods (Monte Carlo, PDE/finite difference), C++ (pricing libraries), Python. Deep understanding of options pricing theory and quantitative models for pricing/hedging.

**How it differs from buy-side:** Builds pricing engines and risk tools FOR traders; does not generate alpha or manage P&L directly. Work is narrower and more specialized ("five people doing one HF quant's job"). Focus on pricing accuracy, not signal discovery. Regulatory knowledge matters.

**Relevance for ML/DL expert:** MODERATE-LOW without additional training. Requires significant stochastic calculus, derivatives pricing theory, and C++ that ML backgrounds rarely include. "Data Strats" roles at Goldman are the most accessible sub-type for ML practitioners.

**Compensation:** $110K-$250K base; more stable but lower ceiling than buy-side.

---

### 1.5 Risk Quant

**Description:** Middle-office roles focused on risk measurement, management, and reporting. Compute VaR, Expected Shortfall, stress test scenarios.

**Where it exists:** All major banks (JPMorgan, Goldman, Morgan Stanley, etc.), large asset managers, insurance companies.

**Key skills:** Derivatives pricing, risk metrics (VaR, CVaR, Expected Shortfall), fixed income and credit products, Python and SQL, stress testing and scenario analysis, communication skills to explain risk to non-quants. Critical thinking to anticipate failure modes.

**How it differs:** Less P&L-driven, more defensive/protective orientation. Regulatory knowledge (Basel III, FRTB) is important. Broader product knowledge needed but less depth on any single product.

**Relevance for ML/DL expert:** MODERATE. Risk management increasingly uses ML techniques, and the explainability requirements create demand for interpretable ML expertise. The gap is financial product knowledge and regulatory frameworks.

**Compensation:** $100K-$200K base (lower than front-office roles).

---

### 1.6 XVA Quant

**Description:** Specialized quants computing valuation adjustments for counterparty risk in OTC derivatives: CVA, DVA, FVA, KVA, MVA.

**Where it exists:** Major banks with large derivatives books (JPMorgan, Goldman, Citigroup, Barclays, Deutsche Bank, BNP Paribas). This role essentially does not exist on the buy-side.

**Key skills:** Detailed knowledge of CVA/DVA/FVA/KVA/MVA and their interactions, Monte Carlo simulations, exposure profiles (PFE, EPE, ENE), stochastic calculus, C++ and Python, QuantLib, collateral modeling, CSA terms.

**How it differs:** The most technically demanding sell-side specialty. Arose from post-2008 regulatory requirements. One of the fastest-growing areas of sell-side quant work.

**Relevance for ML/DL expert:** LOW without significant additional training. Requires deep stochastic calculus, derivatives pricing, and regulatory knowledge. ML applications are emerging (GenAI pricing at Citigroup per QuantMinds 2025) but the domain knowledge barrier is high.

**Compensation:** Premium over standard desk quants due to scarcity.

---

### 1.7 Execution Quant

**Description:** Focuses on market microstructure, order book analysis, transaction cost analysis, and execution algorithms (TWAP, VWAP, IS, Almgren-Chriss).

**Where it exists:** HRT, Citadel Securities, Graham Capital, Millennium Execution Services, Jump Trading, Tower Research, bank e-trading desks (eFX, eRates).

**Key skills:** Market microstructure, order book dynamics, transaction cost analysis (TCA), optimal execution models, Python/C++, low-latency programming, backtesting on tick data, RL for order scheduling (emerging).

**How it differs:** Distinct sub-field that many ML practitioners overlook. Less competition than alpha research. Combines elements of engineering, statistics, and market knowledge.

**Relevance for ML/DL expert:** HIGH. Execution optimization is increasingly ML-driven (RL for order scheduling, ML for market impact prediction). Strong demand with less competition. The gap is market microstructure knowledge and low-latency systems understanding.

**Compensation:** Competitive with QR roles.

---

### 1.8 Data Scientist at Fund

**Description:** Often a softer entry point for ML practitioners. More focused on data analysis, feature discovery, and building insights from diverse datasets. Less emphasis on original mathematical research.

**Where it exists:** Two Sigma, Point72, Citadel (Data Strategies Group), Millennium, larger multi-strategy funds.

**Key skills:** Python and SQL (always), ML algorithms (practical application), data wrangling/ETL/pipeline experience, communication and "narrative from data," applied statistics. Finance background explicitly NOT required (Two Sigma: "more than half of our employees come from outside finance").

**How it differs from QR:** Lower bar on mathematical depth, more emphasis on data engineering (SQL, ETL, pipelines), more emphasis on communication and storytelling. Can be a stepping stone to QR.

**Relevance for ML/DL expert:** VERY HIGH. The most accessible entry point. Skills transfer almost directly. Alternative data roles are particularly strong matches.

**Compensation:** $150K-$300K base.

---

### 1.9 ML Engineer at Fund

**Description:** Dedicated ML role focused on developing, training, and deploying ML models. Emphasis on ML methodology itself rather than financial intuition.

**Where it exists:** G-Research, XTX Markets, Point72/Cubist, Jane Street, Two Sigma (growing).

**Key skills:** Deep learning expertise, PyTorch/TensorFlow (explicit), GPU computing and distributed training, NeurIPS/ICML/ICLR publications valued, Python essential, C++ for performance-critical paths, RL, NLP, representation learning.

**How it differs from QR:** Deeper ML expertise required, less financial domain knowledge, publications and conference participation explicitly valued, involves research at the frontier of ML itself.

**Relevance for ML/DL expert:** HIGHEST. This is the most direct translation of existing skills. The gap is minimal -- mainly adapting to financial data characteristics (noise, non-stationarity).

**Compensation:** $200K-$500K+ (highest ceiling for senior AI talent).

---

### 1.10 NLP/AI Engineer

**Description:** A new and rapidly growing role type focused on applying LLMs, NLP, and GenAI to financial data. Extracting alpha from text.

**Where it exists:** Point72 (explicit role), Citadel, various hedge funds hiring "Head of LLM Research." Emerging across the industry.

**Key skills:** LLM expertise (fine-tuning, prompt engineering, RAG), NLP fundamentals, Python/SQL, software engineering (Git, testing), experience with both internal models and external APIs, benchmarking LLM approaches.

**How it differs:** Most specific to the 2024-2026 wave of LLM adoption. Combines textual and numerical data. Still niche (<10% of overall quant hiring) but fastest-growing role category.

**Relevance for ML/DL expert:** VERY HIGH for those with NLP/LLM experience. Extremely high demand with very few qualified candidates. One of the strongest competitive advantages an ML expert can have.

**Compensation:** $150K-$250K+ base (plus bonus).

---

### 1.11 Model Validation / Model Risk Management (MRM) Quant

**Description:** Independent review function required by regulators (SR 11-7). Validates and challenges models built by front-office quants. Must be independent of model builders.

**Where it exists:** All major banks (required by regulation). Does not exist in the same form on the buy-side.

**Key skills:** Python, R, SQL, statistical testing, ML, knowledge of regulatory guidelines (SR 11-7, Basel, CCAR, CECL), ability to critique models across all types. Certifications valued: FRM, PRM, CQF, GARP's AI in Risk.

**How it differs:** Adversarial role -- must find flaws and challenge assumptions. Requires breadth across all model types. Growing demand for people who can validate ML models under regulatory frameworks.

**Relevance for ML/DL expert:** MODERATE-HIGH. The growing use of ML in banking creates urgent need for validators who understand ML. This is a genuine gap in the market -- few people combine ML expertise with regulatory knowledge.

**Compensation:** $120K-$200K base; lower than front-office but more stable.

---

### 1.12 Fintech/Vendor Quant (Bloomberg, MSCI, BlackRock)

**Description:** Quants at data/analytics vendors who build models consumed by thousands of institutional clients. More emphasis on scalability, breadth, and productionization.

**Where it exists:** Bloomberg (quantitative analytics), MSCI (risk/factor models), BlackRock (Aladdin platform), Refinitiv, FactSet, S&P Global.

**Key skills vary by firm:**
- **Bloomberg:** PhD preferred, modern C++ and Python, full breadth of financial products and risk types, derivatives pricing across ALL asset classes.
- **MSCI:** Factor modeling, cross-sectional analysis, risk decomposition, both traditional statistics AND AI techniques.
- **BlackRock (Aladdin):** Math/stats/probability, ML and time series, SQL/Python/R/JavaScript, fixed income/equities/credit knowledge.

**How it differs:** Must understand the FULL breadth of financial products (Bloomberg serves 300K+ clients). Breadth over depth compared to bank desk quants. Platform orientation means more emphasis on scalability and user-facing design. Models become INDUSTRY STANDARDS (Barra risk models).

**Relevance for ML/DL expert:** MODERATE-HIGH. MSCI and BlackRock explicitly seek "cutting-edge AI techniques." Bloomberg needs quants who understand the full breadth of financial products. Good blend of ML skills with financial breadth.

**Compensation:** $155K-$285K (Bloomberg); competitive at MSCI and BlackRock.

---

### 1.13 Investment Engineer (Bridgewater-Specific)

**Description:** Bridgewater's unique hybrid of quant researcher and software engineer. Designs algorithms that codify economic and market understanding into trading systems.

**Where it exists:** Bridgewater Associates (unique to this firm).

**Key skills:** R or Python (Scala a plus), statistical modeling, time-series and cross-sectional data, system design for large-scale systems, deep interest in markets and investing, intellectual curiosity and evidence-based skepticism, excellent communication.

**Relevance for ML/DL expert:** HIGH. The combination of systems thinking and quantitative research maps well to ML engineers.

**Compensation:** $300K-$450K total.

---

## Section 2: Master Topic Classification

### Tier Definitions (from Course Requirements)

| Tier | Description | Action |
|------|-------------|--------|
| **MUST KNOW** | Required for 80%+ of quant roles; table stakes | Must be in the course |
| **SHOULD KNOW** | Required for 40-80% of roles; strong differentiator | Should be in the course if space permits |
| **GOOD TO KNOW** | Required for 15-40% of roles; valuable but specialized | Include if fits naturally; can be elective |
| **NICHE** | Required for <15% of roles; specialized domain | Brief mention at most; point to resources |
| **DECLINING** | Was important but being automated/replaced | Mention for awareness; don't build skills around it |

---

### Category A: Mathematical & Statistical Foundations

| Topic | Tier | Evidence | Roles Where Required | Trend | Recommended Depth |
|-------|------|----------|---------------------|-------|-------------------|
| **Probability & Statistics** | MUST KNOW | ~95% of job listings (Report 1); 100% of MFE programs (Report 3); "mother tongue" per practitioners (Report 4). Universal across every source. | All quant roles without exception | Stable | Mastery |
| **Linear Algebra** | MUST KNOW | Implicit in every ML and finance role; explicit at MSCI, Bridgewater, interview prep (Report 4: Green Book covers it); 100% of MFE programs | All quant roles | Stable | Working Knowledge (ML experts already have this) |
| **Time-Series Analysis (ARMA, GARCH, cointegration)** | MUST KNOW | ~70% of job listings (Report 1); 77% of MFE programs (Report 3); practitioners emphasize non-stationarity as core challenge (Report 4); universal in quant research | QR, QT, execution quant, risk quant, data scientist at fund | Stable | Mastery |
| **Stochastic Calculus** | SHOULD KNOW | 100% of MFE programs require it (Report 3); still tested in interviews (Report 4); BUT buy-side ML roles increasingly don't require it; practitioners note it's "less relevant since post-2008 derivatives scaling back" for non-derivatives roles (Report 5). Critical for sell-side. | Desk quant, XVA quant, risk quant, derivatives QR. Less critical: stat-arb QR, ML researcher, data scientist | Stable for sell-side; declining for buy-side ML roles | Working Knowledge for buy-side track; Mastery for sell-side track |
| **Optimization Methods (convex, LP/QP, dynamic programming)** | SHOULD KNOW | 54% of MFE programs (Report 3); required for portfolio construction (~35% of job listings, Report 1); Columbia has core optimization course | QR (portfolio construction), quant developer, risk quant, asset management | Stable | Working Knowledge |
| **Bayesian Statistics** | SHOULD KNOW | ~15% of job listings but growing (Report 1); Springer 2025 book on Bayesian ML in QF (Report 4); practitioners flag as "underrated" and "growing rapidly in importance" (Report 4); G-Research lists "Bayesian non-parametrics" | ML researcher, QR, risk quant | Growing | Working Knowledge |
| **Causal Inference** | SHOULD KNOW | Lopez de Prado's dominant 2023-2025 research agenda (Reports 4, 5); CFA Institute published on it; Cambridge Elements publication; emerging in interviews; "factor mirage" concept reshaping factor investing | QR, factor investing, portfolio construction | Rapidly growing | Working Knowledge |
| **Hypothesis Testing & Experiment Design** | MUST KNOW | ~40% of listings explicitly (Report 1); Two Sigma, Jane Street, Citadel emphasize "rigorous scientific method"; practitioners unanimously stress research methodology over model selection (Report 4) | QR, ML researcher, data scientist | Stable | Mastery |
| **Stochastic Control / HJB Equations** | GOOD TO KNOW | 31% of MFE programs (Report 3); Oxford core course; relevant for optimal execution and portfolio optimization | Execution quant, derivatives QR, academic-track roles | Stable | Conceptual Awareness |
| **Numerical Methods (Monte Carlo, PDE/finite diff, trees)** | SHOULD KNOW | 92% of MFE programs (Report 3); required for derivatives pricing; explicitly at CMU, Columbia, NYU, Oxford, Berkeley; required for sell-side roles (Report 2) | Desk quant, XVA quant, risk quant, derivatives-focused QR | Stable (but diffusion models may partially displace MC per Report 5) | Working Knowledge |

---

### Category B: Financial Domain Knowledge

| Topic | Tier | Evidence | Roles Where Required | Trend | Recommended Depth |
|-------|------|----------|---------------------|-------|-------------------|
| **How Markets Work (order books, execution, bid-ask, market making)** | MUST KNOW | ~85% of roles require basic market understanding; practitioners flag this as "routinely catches out prospective quants at interview" (Report 4); market microstructure at ~25% of listings (Report 1) but basic market mechanics needed everywhere | All quant roles | Stable | Working Knowledge |
| **Derivatives / Options Pricing (Black-Scholes, Greeks, hedging)** | SHOULD KNOW | ~30% of buy-side listings explicitly (Report 1); 100% of MFE programs (Report 3); dominant at Optiver, IMC, Akuna, SIG; critical for sell-side (Report 2). Huge gap in current course. | QT, desk quant, XVA quant, vol researcher, risk quant, sell-side roles. Less critical: stat-arb QR, data scientist | Stable | Working Knowledge for buy-side; Mastery for sell-side |
| **Portfolio Construction & Risk Management** | MUST KNOW | ~35% of listings explicitly mention portfolio construction (Report 1); 100% of MFE programs teach portfolio theory (Report 3); practitioners say risk management is "of equal importance to alpha generation" (Report 4); HRT, AQR, Man AHL, Citadel mention it | QR, QT, quant PM, risk quant, asset management | Growing emphasis | Working Knowledge |
| **Factor Models (Fama-French, Barra, cross-sectional)** | SHOULD KNOW | Central to AQR, MSCI, Vanguard, Bridgewater; core at most MFE programs; Lopez de Prado's "factor mirage" work shows this area is evolving (Reports 4, 5); Barra models are industry standard | QR, asset management, MSCI/vendor quants, risk quant | Evolving (causal factor methods growing) | Working Knowledge |
| **Fixed Income (yield curves, duration, interest rate models)** | SHOULD KNOW | ~20% of job listings mention FI (Report 1); 85% of MFE programs teach it (Report 3); "fixed income is half of global capital markets by notional" (Requirements doc); FI is UNIVERSALLY taught -- more than equities in MFE programs | Rates quant, desk quant, risk quant, Bloomberg/vendor quant. Less critical: equity stat-arb QR | Stable | Working Knowledge for buy-side; Mastery for sell-side |
| **Backtesting Methodology (overfitting, look-ahead bias, survivorship bias, transaction costs)** | MUST KNOW | ~65% of listings mention backtesting (Report 1); Lopez de Prado's "7 deadly sins of ML in finance" emphasize backtest overfitting (Report 4); practitioners unanimously stress this. "Most backtests are misleading" | QR, QT, data scientist at fund, execution quant | Stable | Mastery |
| **Market Microstructure (advanced: LOB modeling, market impact, optimal execution)** | GOOD TO KNOW | ~25% of job listings (Report 1); 46% of MFE programs (Report 3); dedicated roles at HRT, Citadel Securities, Jump Trading; practitioners call it "underappreciated specialization" (Report 1); becoming image-processing-based (Report 5) | Execution quant, HFT, market maker, e-trading | Growing | Working Knowledge for execution track; Conceptual Awareness otherwise |
| **Statistical Arbitrage / Pairs Trading** | GOOD TO KNOW | NYU has "Time Series Analysis & Statistical Arbitrage" elective (Report 3); HRT lists "3+ years in stat-arb" (Report 1); historically important quant strategies; cointegration framework | QR at stat-arb firms (HRT, Millennium, etc.) | Stable | Working Knowledge |
| **Credit Risk / Default Prediction** | GOOD TO KNOW | 62% of MFE programs cover credit (Report 3); growing "quant credit" trend at QuantMinds 2025 (Report 5); important for banking but less for buy-side; includes CDS, structural/reduced-form models | Desk quant (credit), risk quant, MRM quant, credit fund QR | Growing ("quant credit takes center stage" per QuantMinds 2025) | Working Knowledge |
| **XVA (CVA/DVA/FVA/KVA/MVA)** | NICHE | Only 2 MFE programs cover it (Report 3); critical for sell-side derivatives businesses (Report 2); fastest-growing sell-side specialty but barely exists buy-side | XVA quant (sell-side only) | Growing (sell-side) | Conceptual Awareness (unless targeting sell-side) |
| **Regime Detection / Hidden Markov Models** | GOOD TO KNOW | Ensemble HMM + tree-based approaches at QuantMinds 2025 (Report 5); DeePM regime-robust strategies show 2x outperformance; Quantocracy covers it; not in most MFE programs but important in practice | QR, systematic macro, portfolio management | Growing | Working Knowledge |
| **Volatility Modeling (stochastic vol, SABR, rough vol, IV surfaces)** | SHOULD KNOW | 31% of MFE programs have advanced vol courses (Report 3); Man AHL and Optiver have dedicated vol researcher roles (Report 1); rough volatility + DL calibration = "new sell-side frontier" at ICAIF 2025 (Report 5); CQF elective covers rough path theory | Vol researcher, derivatives QR, desk quant, options QT | Growing (rough vol + DL calibration) | Working Knowledge |
| **Regulatory Frameworks (Basel III, FRTB, SR 11-7, CCAR)** | GOOD TO KNOW | Essential for sell-side (Report 2); FRM covers extensively; MFE programs barely mention it (Report 3 identifies this as a gap); minimal relevance for buy-side | Risk quant, MRM quant, regulatory/capital quant, XVA quant | Stable (sell-side); irrelevant for buy-side | Conceptual Awareness (unless targeting sell-side) |
| **Alternative Data** | SHOULD KNOW | ~30% of listings (Report 1); Citadel Data Strategies Group works with "petabyte-scale alternative datasets"; 85% of leading hedge funds use 2+ alt datasets; $273B projected market (Report 5); NYU elective covers it | QR, data scientist at fund, NLP/AI engineer | Growing | Working Knowledge |
| **ESG/Climate Risk Quantification** | GOOD TO KNOW | 23% of MFE programs (Report 3); NGFS scenarios (Report 5); carbon markets growing but fragmented; nature/biodiversity risk emerging; 3 MFE programs offer it (MIT, Columbia, UCL) | ESG quant, climate risk analyst, some asset managers | Growing but cooling from peak hype (Report 5: "gold-rush phase is over") | Conceptual Awareness |
| **Commodities / Energy Markets** | NICHE | 15% of MFE programs (Report 3); ~15% of job listings (Report 1); CQF has energy trading elective; energy trading market growing to $12B by 2030 (Report 5); massive complexity from renewables | Commodities QR, energy trading | Growing (energy) | Conceptual Awareness |
| **FX Markets** | NICHE | 23% of MFE programs (Report 3); ~20% of job listings (Report 1); specialized e-trading roles at banks | FX QR, e-trading quant | Stable | Conceptual Awareness |
| **DeFi/Tokenization** | GOOD TO KNOW | 31% of MFE programs now offer DeFi courses -- Oxford, NYU, Berkeley (Report 3); tokenized RWAs at $33B+ and tripled in 2025 (Report 5); QuantMinds 2025 has DeFi sessions; BlackRock in production tokenization | DeFi quant, crypto trading, TradFi-DeFi bridge roles | Rapidly growing | Conceptual Awareness |
| **Behavioral Finance** | NICHE | Only 15% of MFE programs (Report 3); CQF elective; relevant to market anomalies; QuantMinds 2025 featured "Psychology of AI" | Limited direct demand | Stable | Skip (for this course) |

---

### Category C: ML/DL Applied to Finance

| Topic | Tier | Evidence | Roles Where Required | Trend | Recommended Depth |
|-------|------|----------|---------------------|-------|-------------------|
| **ML for Alpha/Signal Generation (feature engineering, financial loss functions, IC-based evaluation)** | MUST KNOW | ~85-90% of listings mention ML (Report 1); 85% of MFE programs now include ML (Report 3); AQR's 20% of flagship signals from ML (Report 4); Gu-Kelly-Xiu framework is canonical | QR, ML researcher, data scientist at fund | Growing | Mastery |
| **Financial Feature Engineering** | MUST KNOW | ~35% of listings explicitly (Report 1); Jane Street, IMC, Squarepoint emphasize it; practitioners stress this over model architecture (Report 4) | QR, ML researcher, data scientist | Stable | Mastery |
| **NLP/LLM for Financial Text (sentiment, earnings calls, filings)** | SHOULD KNOW | ~35% of listings and growing fast (Report 1); Point72 explicit NLP/AI role; Citadel, G-Research (Report 1); AQR using ML for earnings call parsing (Report 4); FinBERT, LLM embeddings research active; ICAIF 2025 dominated by LLM papers (Report 5) | NLP/AI engineer, QR (growing), data scientist | Rapidly growing | Working Knowledge |
| **Deep Learning for Financial Time Series** | SHOULD KNOW | ~55-60% of listings mention DL (Report 1); Oxford made DL core (Report 3); XTX, Point72/Cubist, G-Research explicitly require DL; BUT practitioners warn "simple regression often beats DL in production" (Report 4) | ML researcher, QR at DL-focused firms (XTX, G-Research, Point72) | Growing (with caveats) | Working Knowledge |
| **Financial Foundation Models (FinCast, Kronos, TFT)** | GOOD TO KNOW | NeurIPS 2025 paper on Kronos (Report 5); FinCast first finance-specific FM; critical finding: generic FMs fail on financial data, domain-specific training essential; very new area | ML researcher, QR (frontier) | Rapidly growing (new category) | Conceptual Awareness |
| **Reinforcement Learning for Finance (execution, portfolio optimization)** | GOOD TO KNOW | ~15% of job listings (Report 1); 38% of MFE programs (Report 3); Stanford has dedicated RL-for-finance course; JPMorgan and Goldman in production (Report 5); shift from pure RL to hybrid approaches; multi-agent RL for market making near production | Execution quant, ML researcher, portfolio optimization | Growing (hybrid approaches) | Working Knowledge |
| **GNNs for Finance (correlation graphs, supply chain, fraud)** | GOOD TO KNOW | Not widely in job listings; maturing in production for fraud detection (Report 5); conference papers on dynamic relational graphs; TFT-GNN hybrids for forecasting | ML researcher, risk/fraud roles | Maturing | Conceptual Awareness |
| **Bayesian DL & Uncertainty Quantification** | GOOD TO KNOW | G-Research lists "Bayesian non-parametrics" (Report 1); Springer 2025 book (Report 4); practitioners flag as "growing rapidly"; natural framework for position sizing and regime detection | QR, risk management, ML researcher | Growing | Working Knowledge |
| **Diffusion Models for Finance** | GOOD TO KNOW | NeurIPS 2025 Best Paper on diffusion for portfolio optimization (Report 5); TRADES framework for LOB simulation; displacing GANs; "Beyond Monte Carlo" paper; ICAIF 2025 papers; "the hottest technical area in quant finance right now" per emerging trends research | ML researcher, QR (frontier), risk (scenario generation) | Rapidly growing | Conceptual Awareness (emerging) |
| **Conformal Prediction** | NICHE | NeurIPS 2025 highlight paper (Report 5); distribution-free uncertainty quantification; regulatory appeal; still emerging | ML researcher, risk/MRM quant | Emerging | Conceptual Awareness |
| **Synthetic Data Generation (GANs, diffusion, VAEs)** | GOOD TO KNOW | CFA Institute 2025 report (Report 5); JP Morgan foundational work; Gartner predicts synthetic > real data for 60% of financial AI; diffusion models replacing GANs | QR, risk quant, model validation | Growing | Conceptual Awareness |
| **Deep Hedging (neural net surrogates for derivatives pricing)** | GOOD TO KNOW | Andrew Green's 2024 Wiley book (Report 4); PyQuant News covers it; QuantMinds sessions on dynamic hedging under model uncertainty (Rama Cont); academic momentum | Derivatives QR, desk quant (modern), vol researcher | Growing | Working Knowledge |
| **Interpretable/Explainable ML for Finance** | SHOULD KNOW | Banks require explainable models due to regulation (Report 2); SR 11-7 model governance (Report 2); ICAIF 2025 papers on interpretability; "White Box Finance" NeurIPS paper (Report 5); growing regulatory requirement | MRM quant, risk quant, any bank-facing ML role | Growing (regulatory-driven) | Working Knowledge |
| **Agentic AI / Multi-Agent Systems for Finance** | NICHE | Dominant topic at ICAIF 2025 (Report 5); TradingAgents, AgentQuant frameworks; BUT reality check: agents "fail to consistently outperform simple baselines"; 2025 saw "agentic humility" after 2024 "agentic hype" | ML researcher (frontier), QR (experimental) | Early stage / overhyped short-term | Conceptual Awareness |

---

### Category D: Programming & Engineering

| Topic | Tier | Evidence | Roles Where Required | Trend | Recommended Depth |
|-------|------|----------|---------------------|-------|-------------------|
| **Python (NumPy, Pandas, SciPy, scikit-learn)** | MUST KNOW | ~95% of all listings (Report 1); 100% of MFE programs (Report 3); universal across every source | All quant roles | Stable (dominant) | Mastery (ML experts already have this) |
| **PyTorch / TensorFlow** | SHOULD KNOW | ~30% of listings explicitly (Report 1); Point72/Cubist, XTX, G-Research require it; Oxford uses PyTorch in core DL course (Report 3) | ML researcher, DL-focused QR, NLP/AI engineer | Stable (PyTorch gaining over TF) | Working Knowledge (ML experts already have this) |
| **SQL** | SHOULD KNOW | ~50% of listings (Report 1); rising; more common at multi-strategy funds; required "nearly everywhere" for sell-side (Report 2); essential for data engineering | Data scientist, QR at multi-strategy funds, risk quant, sell-side roles | Growing | Working Knowledge |
| **C++** | SHOULD KNOW | ~65% of buy-side listings (Report 1); essential for sell-side pricing libraries (Report 2); C++ developer demand surging (UK salary up 17% YoY per Report 4); 93.94% of C++ quant ads mention low-latency; BUT only 46% of MFE programs require it (Report 3), and fading as universal requirement in academia | Quant developer, desk quant, XVA quant, HFT roles | Growing demand but narrowing to specific roles | Working Knowledge for dev roles; Conceptual Awareness for research-only |
| **Production Code Quality / Software Engineering** | MUST KNOW | ~40% of listings mention "production code" (Report 1); "production-quality research code" growing expectation at Citadel, HRT, IMC, Millennium; "2025: stopped chasing models, started shipping systems" (Report 4); MFE programs largely fail to teach this (Report 3 gap) | QR (increasingly), quant developer, ML engineer | Growing rapidly | Working Knowledge |
| **Git / Version Control** | MUST KNOW | ~25% of listings explicitly (Report 1); implicit everywhere; Point72 NLP role specifies "github, testing" | All quant roles | Stable | Working Knowledge (ML experts already have this) |
| **Linux Environment** | SHOULD KNOW | ~30% of listings (Report 1); Citadel, SIG, Squarepoint, Akuna | Quant developer, infrastructure roles, HFT | Stable | Working Knowledge |
| **Data Engineering / ETL Pipelines** | SHOULD KNOW | "70-80% of a quant's time is spent on data cleaning" (Report 4); barely covered in MFE programs (Report 3 gap); practitioners unanimously stress it; infrastructure maintenance dominates quant developer day-to-day | All quant roles (especially data scientist, QR) | Growing recognition | Working Knowledge |
| **GPU Computing / Distributed Training** | GOOD TO KNOW | Jane Street "tens of thousands of GPUs," XTX "25,000+ GPUs" (Report 1); NVIDIA CUDA 160x speedups (Report 5); JAX gaining traction; JAX-LOB for LOB simulation | ML researcher, quant developer, DL-focused QR | Growing | Working Knowledge |
| **Cloud Computing (AWS/GCP/Azure)** | GOOD TO KNOW | ~15% of listings (Report 1); zero MFE coverage (Report 3 gap); Squarepoint explicitly mentions cloud; hedge funds increasingly cloud-native (Report 5) | Infrastructure roles, quant developer, data engineering | Growing | Conceptual Awareness |
| **Polars + DuckDB (modern data stack)** | GOOD TO KNOW | "The new default research data stack" replacing Pandas + KDB+ (Report 5); fast, free, composable | QR, data scientist, quant developer | Rapidly growing | Conceptual Awareness |
| **KDB/Q** | NICHE | ~10-15% of listings (Report 1); Millennium, Squarepoint; pays well but declining (Report 5: 40% migration predicted by 2026; disrupted by open-source alternatives) | Legacy HFT, Millennium, some banks | Declining | Skip |
| **MLOps / Model Monitoring (MLflow, feature stores)** | GOOD TO KNOW | "2025: stopped chasing models, started shipping systems" (Report 4); MLflow widely adopted; Santander study: 12-17% regulatory capital savings from MLOps; EU AI Act compliance needs | ML engineer, quant developer, production-facing QR | Growing | Conceptual Awareness |
| **Rust** | NICHE | "Rust for Quant Finance 2025" book exists (Report 4); growing in crypto; RustQuant library; memory safety advantage; BUT "overwhelming majority of infrastructure remains C++ and Python" (Report 4) | Crypto/DeFi quant developer (niche) | Early growth | Skip (for now) |
| **R** | DECLINING | ~35% of listings but declining (Report 1); some MFE programs still use it; "often 'Python or R'" -- Python winning | Some legacy roles, model validation, Bridgewater | Declining vs. Python | Skip (Python preferred) |
| **MATLAB** | DECLINING | ~15% of listings (Report 1); "essentially disappeared from curricula" (Report 3); legacy mention | Legacy roles only | Declining | Skip |
| **VBA** | DECLINING | "Still the lingua franca of trading floors for ad hoc analysis" (Report 2); persistent but legacy | Some sell-side desk roles | Declining | Skip |

---

### Category E: Data & Infrastructure

| Topic | Tier | Evidence | Roles Where Required | Trend | Recommended Depth |
|-------|------|----------|---------------------|-------|-------------------|
| **Working with Large-Scale Financial Datasets** | MUST KNOW | ~90% of listings mention large data (Report 1); "petabyte-scale" at Citadel, Jane Street, XTX; data quality/cleaning is 70-80% of actual work (Report 4) | All quant roles | Stable | Working Knowledge |
| **Alternative Data Processing (satellite, NLP, web scraping)** | SHOULD KNOW | ~30% of listings (Report 1); $273B projected market (Report 5); Citadel Data Strategies Group; 85% of leading funds use 2+ alt datasets | QR, data scientist, NLP/AI engineer | Growing | Working Knowledge |
| **Market Data Vendors (Bloomberg, Refinitiv, FactSet)** | GOOD TO KNOW | Bloomberg BQL explicitly mentioned at some firms (Report 2); 300K+ Bloomberg Terminal users; understanding data sources matters in practice | All quant roles (practical knowledge) | Stable | Conceptual Awareness |
| **Time-Series Databases** | GOOD TO KNOW | KDB/Q at some firms (Report 1); TimescaleDB, ArcticDB emerging (Report 5); critical for tick data storage | Quant developer, HFT, data engineering | Evolving (away from KDB toward open-source) | Conceptual Awareness |
| **Streaming / Real-Time Data Processing** | NICHE | Relevant for HFT and e-trading; Spark at Two Sigma (Report 1) | HFT, e-trading, execution quant | Stable | Conceptual Awareness |

---

### Category F: Risk & Regulation

| Topic | Tier | Evidence | Roles Where Required | Trend | Recommended Depth |
|-------|------|----------|---------------------|-------|-------------------|
| **Risk Metrics (VaR, Expected Shortfall, stress testing)** | SHOULD KNOW | 85% of MFE programs (Report 3); FRM covers extensively; required for sell-side and risk roles; VaR still widely used despite ES adoption under FRTB (Report 2) | Risk quant, desk quant, MRM quant, QR (portfolio level) | Stable | Working Knowledge |
| **Position Sizing / Kelly Criterion** | SHOULD KNOW | Quantocracy covers it (Report 4); practitioners stress it; "risk management should be considered of equal importance to alpha generation" (Report 4); connected to uncertainty quantification | QR, QT, quant PM | Stable | Working Knowledge |
| **Transaction Cost Modeling** | MUST KNOW | Practitioners unanimously stress: "backtests that ignore realistic transaction costs, market impact, and slippage are worthless" (Report 4); ~40% of listings implicitly require it | QR, execution quant, QT | Stable | Working Knowledge |
| **Model Risk Management (SR 11-7)** | NICHE | Required by regulation for banks (Report 2); growing as ML models enter banking; "barely appears in curricula" (Report 3); creates demand for ML+regulatory expertise combo | MRM quant (bank-specific) | Growing (as ML enters banking) | Conceptual Awareness |
| **Regulatory Capital (Basel III, FRTB)** | NICHE | Critical for sell-side (Report 2); FRM covers in depth; MFE programs barely mention it (Report 3 gap); irrelevant for buy-side | Regulatory/capital quant, risk quant (sell-side) | Stable | Skip (unless targeting sell-side) |
| **Operational Risk** | NICHE | 20% of FRM Part II (Report 3); "almost NEVER taught in MFE programs" -- significant gap between FRM and academia (Report 3) | Risk quant (bank), operational risk roles | Stable | Skip |

---

### Category G: Soft Skills & Research Methodology

| Topic | Tier | Evidence | Roles Where Required | Trend | Recommended Depth |
|-------|------|----------|---------------------|-------|-------------------|
| **Communication / Explaining Models to Non-Technical Audiences** | MUST KNOW | ~85% of listings mention communication (Report 1); "explain complex concepts clearly and concisely" is near-universal language; practitioners say it "catches out technically brilliant candidates" (Report 4); CMU and MIT have dedicated business communication courses (Report 3) | All quant roles | Stable | Working Knowledge |
| **Research Methodology / Avoiding Data Snooping** | MUST KNOW | Lopez de Prado's "7 deadly sins" (Report 4); "having a research sample and discussing it has the highest signal-to-noise ratio in hiring" (Report 4); experiment design at ~40% of listings (Report 1) | QR, ML researcher, data scientist | Growing emphasis | Mastery |
| **Intellectual Curiosity / Self-Directed Learning** | MUST KNOW | ~60% of listings explicitly mention it (Report 1); assessed (not just mentioned) at Jane Street, Bridgewater, G-Research; "if the foundational reading material feels like a chore, the career probably isn't right" (Report 4) | All quant roles | Stable | N/A (character trait, not teachable) |
| **Cross-Functional Collaboration** | SHOULD KNOW | ~70% of listings mention teamwork (Report 1); "cross-functional squads" replacing siloed roles (Report 4); "no single researcher builds complete strategies alone" -- Sisyphus paradigm (Lopez de Prado) | All quant roles | Growing | Working Knowledge |
| **Coachability / Intellectual Humility** | SHOULD KNOW | Kris Abdelmessih identifies as #1 predictor of success (Report 4); "even brilliant hires at elite firms like SIG need humbling"; "defensive egos indicate poor fit" | All quant roles | Stable | N/A (character trait) |
| **Research Paper Reading / Staying at Frontier** | SHOULD KNOW | Two Sigma has reading circles; Man AHL works with Oxford-Man Institute; G-Research researchers attend top conferences (Report 1); "most of a quant researcher's day is spent reading" (Report 4) | QR, ML researcher | Stable | Working Knowledge |

---

### Category H: Emerging & Frontier Areas

| Topic | Tier | Evidence | Roles Where Required | Trend | Recommended Depth |
|-------|------|----------|---------------------|-------|-------------------|
| **Causal Factor Investing** | SHOULD KNOW | Lopez de Prado's Cambridge Elements publication (Report 5); CFA Institute coverage; "factor mirage" concept; "factor investing's predicament will not be resolved with more data; what is most needed is causal reasoning" (Reports 4, 5) | QR (factor investing), asset management | Rapidly growing | Working Knowledge |
| **Diffusion Models for Financial Data** | GOOD TO KNOW | NeurIPS 2025 Best Paper (Report 5); TRADES framework; displacing GANs and potentially MC; multiple ICAIF papers; Bloomberg contributions | ML researcher, frontier QR | Rapidly growing | Conceptual Awareness |
| **Financial Foundation Models** | GOOD TO KNOW | FinCast, Kronos (NeurIPS 2025), LIM (Report 5); critical finding: generic FMs fail, finance-specific essential | ML researcher, frontier QR | Rapidly growing | Conceptual Awareness |
| **Multi-Agent RL for Market Making** | NICHE | Multiple ICAIF 2025 papers (Report 5); near production at major firms; JPMorgan, Goldman in production for derivatives | Execution quant, ML researcher | Growing | Conceptual Awareness |
| **Geopolitical Risk Modeling** | NICHE | BlackRock dashboard; BBVA neural network models; LightGBM for energy tail risk; QuantMinds 2025 sessions (Report 5) | Macro QR, risk management | Growing | Conceptual Awareness |
| **Quantum Computing for Finance** | NICHE | 15% of MFE programs (Report 3); $1.67B market (Report 5); HSBC improved bond predictions by 34%; Goldman 25x faster risk processing; BUT still pilot/hybrid mode | Infrastructure (future), pricing (future) | Early but more real than expected | Skip (for now) |
| **Post-Quantum Cryptography** | NICHE | Government deadlines: 2030 critical systems (Report 5); NIST standards 2024; every financial institution needs migration plan; "potentially rivals LIBOR transition in scope" | Infrastructure/engineering (not quant-specific) | Urgent for infra | Skip (not quant-specific) |
| **Tokenization of Real-World Assets** | NICHE | $33B+ on-chain; BlackRock, Franklin Templeton in production (Report 5); QuantMinds sessions | DeFi/TradFi bridge roles | Rapidly growing | Conceptual Awareness |
| **Federated Learning for Finance** | NICHE | ICAIF 2025 papers (Report 5); privacy-preserving ML; $0.1B market growing at 27% CAGR | Privacy-focused ML roles, banking | Early stage | Skip |
| **Conformal Prediction** | NICHE | NeurIPS 2025 highlight (Report 5); distribution-free uncertainty; regulatory appeal | MRM quant, risk | Emerging | Conceptual Awareness |
| **Energy Trading Quantification** | NICHE | $7.5B to $12B market (Report 5); renewable integration creating complexity; thousands of nodal prices | Energy QR | Growing | Skip (unless targeting energy) |
| **Rough Volatility Models** | NICHE | ICAIF 2025 paper on DL calibration (Report 5); CQF covers rough path theory (Report 3); still primarily research | Vol researcher, academic-track | Growing (from research to practice) | Conceptual Awareness |

---

## Section 3: What's Genuinely New in 2025-2026

### 3.1 Causal Factor Investing

**What it is:** Moving from correlation-based to causation-based factor analysis. Identifying that many "validated" factors are causally misspecified due to collider bias and confounder bias. Lopez de Prado's "factor mirage" concept.

**Evidence it's growing:** Cambridge Elements publication (2025), CFA Institute coverage, 8 Lopez de Prado lectures (2023-2025), growing recognition that "factor investing's predicament will not be resolved with more data or more complex methods." Interview questions on causal inference emerging at some firms.

**Assessment: REAL.** This is not rebranded statistics. It addresses a fundamental flaw in how the industry has done factor investing for decades. The intellectual framework is rigorous and the problem it solves is genuine.

**Recommended action:** Include as a dedicated topic. Teach causal graphs, collider bias in financial contexts, and how causal reasoning changes factor model construction. This is a genuine differentiator for course graduates.

---

### 3.2 Financial Foundation Models

**What it is:** Pre-trained models specifically for financial time series (FinCast, Kronos) and financial language (LIM), analogous to GPT/BERT for text but built from scratch on financial data.

**Evidence it's growing:** Kronos presented at NeurIPS 2025; FinCast demonstrated zero-shot performance; critical finding that generic time-series FMs FAIL on financial data. This is a new category that didn't exist 2 years ago.

**Assessment: REAL but early.** The key insight -- that domain-specific training is essential -- is well-established. The models themselves are still maturing.

**Recommended action:** Cover as part of the time-series forecasting module (current W9). Teach the "do financial foundation models work?" debate with hands-on comparison of generic vs. finance-specific FMs.

---

### 3.3 Diffusion Models for Finance

**What it is:** Denoising diffusion probabilistic models applied to financial data generation, LOB simulation, scenario generation, and potentially replacing Monte Carlo for certain applications.

**Evidence it's growing:** NeurIPS 2025 Best Paper (factor-based conditional diffusion for portfolio optimization); TRADES framework for LOB simulation; "Beyond Monte Carlo" paper; multiple ICAIF 2025 papers; Bloomberg contributing.

**Assessment: REAL and technically substantial.** Diffusion models are displacing GANs across ML broadly, and the financial applications are genuine. The "replacing Monte Carlo" claim needs caveats but for certain applications it's viable.

**Recommended action:** Include as a topic in synthetic data generation or advanced generative models. Could replace or supplement GANs coverage.

---

### 3.4 LLM Agents for Finance

**What it is:** Multi-agent LLM systems where specialized agents (analyst, trader, risk manager) collaborate on financial analysis, research, or trading decisions.

**Evidence it's growing:** Dominant topic at ICAIF 2025; TradingAgents, AgentQuant, FinReflectKG frameworks; Goldman Sachs GS AI Assistant firmwide; 95% of hedge funds using GenAI.

**Assessment: OVERHYPED for trading; REAL for research augmentation.** Agents "fail to consistently outperform simple baselines" for trading. 2025 saw "agentic humility" after 2024 "agentic hype." But for research workflow acceleration, the value is proven.

**Recommended action:** Cover the practical uses (research augmentation, NLP extraction, knowledge management) honestly. Flag the gap between research papers and production reality. Do not oversell autonomous trading agents.

---

### 3.5 Conformal Prediction for Finance

**What it is:** Distribution-free uncertainty quantification that provides rigorous prediction intervals without assuming specific distributions. Particularly appealing for non-stationary financial data.

**Evidence it's growing:** NeurIPS 2025 highlight paper; addresses the exchangeability violation in time series; exactly what regulators want for model risk management.

**Assessment: REAL but niche.** Technically sound and addresses a genuine problem. Could become a standard tool for model validation.

**Recommended action:** Brief mention in uncertainty quantification module. Point to resources for self-study.

---

### 3.6 The "Models to Systems" Shift

**What it is:** The competitive advantage in quant finance shifting from model novelty to production infrastructure -- deployment, monitoring, governance, and operational resilience.

**Evidence it's growing:** "2025: The Year Quant Finance Stopped Chasing Models and Started Shipping Systems" (LLMQuant); multiple firms requiring "production-quality research code"; Santander study showing 12-17% capital savings from MLOps; EU AI Act compliance requirements.

**Assessment: REAL and important.** This affects what we teach about ML lifecycle management in finance.

**Recommended action:** Integrate production/MLOps considerations throughout the course, not as a separate module. Emphasize that "writing a model" is 20% of the job; getting it into production and keeping it running is 80%.

---

## Section 4: What's Declining or Overhyped

### 4.1 Stochastic Calculus as Universal Core Skill (SELECTIVELY DECLINING)

**What it is:** The mathematical framework (Ito calculus, SDEs, Girsanov theorem) underpinning derivatives pricing.

**Evidence it's declining:** Buy-side ML roles increasingly don't require it; practitioners note it's "less relevant since post-2008 derivatives scaling back" for non-derivatives roles; stat-arb and HFT prioritize statistics/data analysis over stochastic calc.

**What's replacing it:** ML methodology, causal inference, and applied statistics are absorbing the "analytical core" role for non-derivatives quant roles.

**Should it still be taught?** YES, but calibrate depth to career path. For buy-side ML-focused roles: conceptual awareness is sufficient (understand risk-neutral pricing intuitively, don't derive it). For sell-side derivatives roles: still requires mastery. 100% of MFE programs still teach it, so it remains important for credentialing even if daily use is declining for some roles. Recommendation: teach the intuition and financial implications, not the full mathematical derivation, for an ML-focused audience.

---

### 4.2 KDB/Q (DECLINING)

**What it is:** A specialized columnar time-series database and its associated programming language, historically dominant in HFT.

**Evidence it's declining:** $250K+ annual licenses; opaque pricing; exotic syntax; Gartner predicts 40% user migration to hybrid solutions by 2026; open-source alternatives (Polars, DuckDB, QuestDB, ArcticDB) offer 80% performance at fraction of cost. New firms won't adopt.

**What's replacing it:** Polars + DuckDB for research workloads; ArcticDB and TimescaleDB for production.

**Should it still be taught?** NO. Brief mention as context ("you may encounter this at Millennium or legacy HFT firms") but no skill-building.

---

### 4.3 Pure RL for Trading (DECLINING in favor of hybrid)

**What it is:** Using reinforcement learning alone, without domain knowledge integration, for trading decisions.

**Evidence it's declining:** Pure RL adoption dropped from 85% (2020) to 58% (2025). Hybrid approach (RL + domain knowledge) grew from 15% to 42% in the same period.

**What's replacing it:** Hybrid RL that integrates financial constraints, transaction costs, and domain knowledge into the reward function and environment design.

**Should it still be taught?** YES, but teach the HYBRID approach, not pure RL for trading. Emphasize why pure RL fails in finance (non-stationarity, adversarial environment, sparse/noisy rewards).

---

### 4.4 LLM Trading Agents (OVERHYPED in short term)

**What it is:** Using LLMs to make autonomous trading decisions.

**Evidence it's overhyped:** "Many AI trading tools are overhyped, overpriced, and underperforming"; agents "fail to consistently outperform simple baselines"; LLMs can be adversarially prompted to generate misleading financial guidance; 2024 "agentic hype" followed by 2025 "agentic humility."

**What's replacing it:** Using LLMs for research augmentation, NLP extraction, knowledge management -- tasks where hallucination risk is manageable. Multi-agent research workflows (not trading) are the productive direction.

**Should it still be taught?** YES BRIEFLY. Teach what LLMs CAN do in finance (NLP, research augmentation) and what they CANNOT reliably do (autonomous trading). The distinction is important.

---

### 4.5 GANs for Financial Data Generation (BEING REPLACED)

**What it is:** Using Generative Adversarial Networks for synthetic financial data generation.

**Evidence it's declining:** Diffusion models displacing GANs for financial data generation, mirroring the broader ML trend. TRADES framework, NeurIPS 2025 papers all use diffusion approaches. CFA Institute 2025 synthetic data report emphasizes diffusion.

**What's replacing it:** Denoising diffusion probabilistic models (DDPMs), score-based generative models, conditional diffusion models.

**Should it still be taught?** BRIEFLY. Mention GANs as historical context; focus on diffusion models as the current frontier for synthetic financial data.

---

### 4.6 R Programming (DECLINING relative to Python)

**What it is:** Statistical programming language historically popular in finance and academia.

**Evidence it's declining:** Only ~35% of listings mention R (Report 1), and almost always as "Python or R" with Python preferred. Declining in MFE curricula. Python has won the language war in quant finance.

**Should it still be taught?** NO. Python covers all use cases.

---

### 4.7 ESG Quant as Hot Area (COOLING FROM PEAK)

**What it is:** Quantitative approaches to environmental, social, and governance investing.

**Evidence it's cooling:** "Was very hot 2020-2023. Data quality and standardization issues remain unresolved. Greenwashing concerns have dampened enthusiasm. Still important but the gold-rush phase is over." Only 23% of MFE programs offer it.

**Should it still be taught?** BRIEFLY. Climate risk quantification has legs (NGFS scenarios, carbon markets) but the broader ESG narrative has cooled. Cover as a contextual topic, not a module.

---

## Section 5: The Balance Question

### What's the Right Balance Between Domain Knowledge, Applied Modeling, and Engineering?

Based on all evidence, the recommended balance for an ML/DL expert learning finance is:

**~40% Financial Domain Knowledge | ~35% Applied ML/Modeling for Finance | ~25% Engineering/Production Skills**

### Evidence for This Balance

**Why ~40% Domain Knowledge:**
- "Routinely catches out prospective quants at interview" (QuantStart, HN, recruiters) -- basic financial market knowledge is a common failure point for ML-to-finance transitions.
- Finance knowledge is the PRIMARY gap for ML experts. They already have the modeling skills; they lack the domain context.
- Understanding signal-to-noise in finance, non-stationarity, transaction costs, and market mechanics is essential to apply ML correctly.
- Even firms that say "no finance background required" (Jane Street, Two Sigma, RenTech) expect candidates to learn it quickly on the job. A course can accelerate this.
- Stochastic calculus + derivatives + fixed income remain the credentialing core at 100% of MFE programs.

**Why ~35% Applied ML/Modeling:**
- The audience already knows ML/DL fundamentals. The course should NOT reteach them.
- What they need: financial feature engineering, financial loss functions (IC-based, Sharpe-based), financial evaluation metrics, temporal train/val/test splitting, backtesting methodology, financial data quirks (non-stationarity, regime changes, low SNR).
- "The core failure mode of AI in finance is not insufficient learning capacity, but a mismatch between stationary assumptions and a competitive, reflexive environment" -- the applied adaptation is what matters.
- AQR's ML adoption: ML powers ~20% of flagship signals, used for NLP extraction and signal weighting, NOT for end-to-end prediction. This is the realistic use case.

**Why ~25% Engineering/Production:**
- "70-80% of a quant's time is spent on data cleaning" (practitioners) -- but the course shouldn't be 70% data cleaning. It should teach the PRINCIPLES and common patterns.
- "2025: stopped chasing models, started shipping systems" -- production skills are increasingly valued.
- "Production-quality research code" is a growing expectation at Citadel, HRT, IMC, Millennium.
- Infrastructure maintenance (failed cron jobs, broken APIs, data quality) consumes daily attention in practice.
- BUT: Engineering is learnable on the job. The conceptual frameworks (backtesting pipelines, production ML lifecycle, data quality) matter more than specific tools.

### Specific Practitioner Quotes Supporting This Balance

- **On domain knowledge being critical:** "It's all well and good being the best mathematician and programmer on the globe, but if you can't tell your stock from your bond, or your bank from your fund, you'll find it harder to pass HR screenings." (QuantStart)
- **On ML being necessary but insufficient:** "Machine learning is used a lot less than people think." (HN hiring manager). AQR's Cliff Asness: "It's not magic -- it's a more sophisticated analytical capability."
- **On engineering mattering more than expected:** "The competitive advantage has shifted from who has the best model to who has the best production infrastructure." (LLMQuant 2025 retrospective)
- **On the 70-80% data reality:** "Despite all the modeling theory, the bulk of quant work is data engineering -- getting disparate datasets aligned, handling API changes, fixing anomalies." (Multiple practitioners)

### Implications for 18-Week Course Design

Given 18 weeks:
- **~7 weeks** on financial domain knowledge (markets, derivatives, portfolio theory, fixed income basics, risk, execution)
- **~6 weeks** on applied ML for finance (financial feature engineering, NLP for alpha, time-series models, backtesting, financial loss functions, deep hedging, causal inference)
- **~4 weeks** on engineering/production (data pipelines, production code, backtesting infrastructure, MLOps concepts)
- **~1 week** on career/interview preparation and integration

---

## Section 6: Non-Obvious Findings

### Finding 1: Finance Knowledge is Explicitly NOT Required at Most Top Firms
Jane Street, Two Sigma, Renaissance Technologies, G-Research, D.E. Shaw, and Tower Research all explicitly state that no finance background is needed. Two Sigma: "more than half of our employees come from outside finance." The implication: firms teach domain knowledge on the job but CANNOT teach mathematical maturity, programming skill, and research methodology. This is counterintuitive but consistently supported across sources.

**Course implication:** Don't feel pressure to cover every financial topic exhaustively. Teach the frameworks and intuition; firms will provide the details.

### Finding 2: Simple Regression Often Beats Deep Learning in Production
"At one major investment firm, none of the deep learning made money except simple regression." This reflects the signal-to-noise reality of financial markets. Complex models overfit to noise. Geoffrey Hinton's framework applies: ML excels with complex structure and low noise; finance has simple structure and high noise. The opposite of where DL shines.

**Course implication:** Teach deep learning applications in finance honestly. Show WHEN it works (NLP, signal combination, execution optimization) and WHEN it doesn't (end-to-end price prediction from price). The "simple model beats complex model" result should be a core teaching moment.

### Finding 3: Signals Decay 5-10% Per Year
In highly electronic markets, the half-life of a quantitative signal is shockingly short. Dozens of participants rush to execute on the same signal simultaneously. Shared datasets, overlapping strategies, and common talent pools create correlated failure. The 2025 quant stumbles confirmed this: crowding and deleveraging triggered cascading losses.

**Course implication:** Alpha decay should be a theme, not just a topic. Every module on signal generation should address signal shelf-life, crowding risk, and the need for continuous research.

### Finding 4: Data Cleaning is 70-80% of the Job
Despite all the modeling theory, the bulk of quant work is data engineering -- getting disparate datasets aligned, handling API changes, fixing pricing anomalies. A typical quant developer's morning: "Check automated data tasks and overnight cron jobs, fix failed scripts." This is rarely mentioned in textbooks or course descriptions.

**Course implication:** Integrate data quality challenges into every applied module. Don't teach them as a separate "data cleaning" lecture -- make messy data the DEFAULT in assignments and projects.

### Finding 5: The Biggest Competition is Physics and Math PhDs, Not Other ML Engineers
Top quant firms recruit heavily from pure math, physics, and statistics programs. These candidates often have deeper mathematical foundations even if their programming is weaker. The "ML engineer from FAANG" profile has engineering skills but often lacks the probabilistic intuition and mathematical depth.

**Course implication:** The course should build mathematical confidence in financial contexts. ML experts have the tools; they need the financial mathematical vocabulary and intuition.

### Finding 6: Interview Topics Haven't Changed as Much as Expected
Despite all the ML hype, probability puzzles, brain teasers, and statistical reasoning still dominate quant interviews. ML questions are growing but supplement rather than replace the classics. The Green Book (probability, stochastic processes, finance) and Heard on the Street remain essential prep resources.

**Course implication:** Include interview-relevant probability and statistics applications throughout. Frame financial concepts through the lens of "how would this appear in an interview?"

### Finding 7: Risk Management is Equally Important to Alpha Generation but Rarely Taught Well
"Should be considered of equal importance to alpha generation, as opposed to a secondary function." Position sizing, risk allocation, drawdown management, and understanding when to NOT trade are skills that PMs value more than signal generation alone.

**Course implication:** Don't relegate risk management to one week. Integrate risk thinking into every strategy module. Teach position sizing, portfolio-level risk, and the difference between backtest Sharpe and live Sharpe.

### Finding 8: The "Sisyphus Paradigm" Explains Why Solo ML Researchers Fail in Finance
Lopez de Prado argues that building investment strategies requires specialized teams with divided subtasks. Individual researchers attempting complete strategies alone face insurmountable complexity. "Every successful quantitative firm applies the meta-strategy paradigm." The lone genius building a complete strategy is a myth.

**Course implication:** Design team-based projects. Teach the investment pipeline as a system (data, signal, portfolio construction, execution, risk) rather than as individual components.

### Finding 9: Experienced Developers Get SLOWER with AI Coding Assistants on Complex Tasks
A study found developers took 19% longer with AI assistance on complex analytical tasks, despite believing they were faster. 51.5% of coding session time was spent in LLM interaction states. AI coding assistants create a perception gap.

**Course implication:** Use LLM coding assistants thoughtfully in course assignments. They help with boilerplate but can mislead on analytical code. Teach students to use them appropriately.

### Finding 10: C++ Developer Demand is SURGING, Not Declining
UK median salary jumped 17.24% YoY; permanent roles nearly doubled. 93.94% of C++ quant job ads mention low-latency. Despite Python dominance for research, C++ remains essential for production trading systems.

**Course implication:** Don't dismiss C++ as legacy. For students targeting quant dev or HFT roles, recommend C++ as a complementary skill. The course itself probably shouldn't teach C++, but should acknowledge its importance.

### Finding 11: Kaggle Performance is a Credible Entry Signal
XTX Markets and G-Research explicitly value Kaggle performance: "Outstanding performance in any quantitative field or contest (Kaggle, hackathons, olympiads)." This provides a concrete, actionable pathway for ML practitioners to build credible credentials for quant roles.

**Course implication:** Consider designing assignments or projects in Kaggle-like competition format. Encourage students to participate in financial data competitions.

### Finding 12: The DeFi-TradFi Convergence is Real
QuantMinds (a mainstream quant conference) now has DeFi sessions. Tokenized real-world assets at $33B+. BlackRock, Franklin Templeton, Nasdaq in production tokenization. Oxford, NYU, and Berkeley offer DeFi courses. This is no longer a separate ecosystem.

**Course implication:** Brief coverage of DeFi/tokenization is warranted as context for how markets are evolving, even if it's not a core module.

---

## Section 7: Key Questions Answered

### Question 1: What does the full landscape of quant/finance knowledge look like for an ML/DL expert?

The landscape spans 7 major categories with approximately 60+ distinct topics. The Section 2 Master Topic Classification provides the full map. At the highest level:

- **8 MUST-KNOW topics:** Probability/statistics, time-series analysis, how markets work, portfolio construction/risk management, backtesting methodology, transaction cost modeling, financial feature engineering, ML for alpha/signal generation, communication, research methodology, Python, production code quality, hypothesis testing, working with large datasets.
- **~18 SHOULD-KNOW topics:** Derivatives pricing, factor models, fixed income, alternative data, NLP/LLM for finance, DL for financial time series, backtesting, causal inference, SQL, C++ (role-dependent), interpretable ML, data engineering, volatility modeling, risk metrics, Bayesian statistics, position sizing, cross-functional collaboration.
- **~20 GOOD-TO-KNOW topics:** Stat arb, credit risk, market microstructure, regime detection, DeFi, GNNs, RL for finance, diffusion models, financial foundation models, GPU computing, cloud, synthetic data, deep hedging, ESG, regulatory frameworks.
- **~15 NICHE/DECLINING topics:** XVA, quantum computing, KDB/Q, Rust, R, MATLAB, VBA, operational risk, FX-specific, commodities-specific, federated learning, energy trading, behavioral finance.

The landscape is BROADER than most ML practitioners expect, but the depth required varies enormously by topic.

### Question 2: What are the distinct career paths, and how do their knowledge requirements differ?

Section 1 maps 13 distinct career paths. The key insight is that these paths have VERY different knowledge requirements:

- **ML Researcher / NLP-AI Engineer / Data Scientist at Fund:** Most accessible for ML/DL experts. Minimal additional finance knowledge required at entry. Skills transfer directly.
- **Quant Researcher (buy-side):** Core transition target. Requires adding financial domain knowledge (backtesting, signal-to-noise, transaction costs) but NOT deep derivatives math.
- **Quant Developer / Research Engineer:** Requires strong C++ and systems skills. ML engineers from tech have the engineering background; need to add C++ and financial systems knowledge.
- **Execution Quant:** Underappreciated path with less competition. ML increasingly used for execution optimization. Requires market microstructure knowledge.
- **Desk Quant / XVA Quant / Risk Quant (sell-side):** Requires significant additional training in stochastic calculus, derivatives pricing, and regulatory frameworks. The furthest from an ML background.

### Question 3: What is table-stakes vs. differentiating vs. niche?

**Table stakes (MUST KNOW):** Probability/statistics, Python, ML methodology, time-series analysis, backtesting discipline, basic market knowledge, communication, research methodology, production code quality, financial feature engineering, transaction cost awareness.

**Differentiating (SHOULD KNOW):** Causal inference (rapidly growing differentiator), NLP/LLM for finance (high demand, few candidates), derivatives pricing (credentialing), interpretable ML (regulatory-driven), Bayesian methods (growing), alternative data processing.

**Niche:** XVA, quantum computing, operational risk, commodities-specific models, DeFi (growing but still small), Rust.

### Question 4: What's the balance between domain knowledge vs. modeling vs. engineering?

**~40% domain knowledge / ~35% applied ML / ~25% engineering.** See Section 5 for full evidence. The key insight: for ML experts, domain knowledge is the PRIMARY gap, but engineering/production skills are increasingly valued and often overlooked in academic curricula.

### Question 5: What's genuinely new or rapidly growing in 2025-2026?

See Section 3 for details. The top genuinely new areas:
1. **Causal factor investing** -- paradigm shift from correlation to causation (REAL)
2. **Financial foundation models** -- domain-specific FMs (FinCast, Kronos) (REAL but early)
3. **Diffusion models for finance** -- displacing GANs, potentially Monte Carlo (REAL)
4. **LLM agents for research augmentation** -- proven for workflow, not for trading (PARTIALLY REAL)
5. **The "models to systems" shift** -- production infrastructure > model novelty (REAL)
6. **Conformal prediction** -- distribution-free uncertainty quantification (EMERGING)

### Question 6: What's declining?

See Section 4 for details. Key declining areas:
1. **Stochastic calculus as universal requirement** (declining for non-derivatives roles)
2. **KDB/Q** (being replaced by Polars/DuckDB)
3. **Pure RL for trading** (shifting to hybrid approaches)
4. **GANs for financial data** (replaced by diffusion models)
5. **R programming** (replaced by Python)
6. **ESG quant as a hot area** (cooling from peak hype)

### Question 7: What non-obvious skills appear?

See Section 6 for the full list. The most important non-obvious findings:
1. Finance knowledge explicitly NOT required at most top firms
2. Simple regression often beats DL in production
3. Data cleaning is 70-80% of the job
4. Signals decay 5-10% per year
5. Coachability may matter more than raw intelligence
6. Risk management is as important as alpha generation
7. The "Sisyphus paradigm" -- solo researchers fail; teams succeed
8. Kaggle is a credible entry signal
9. C++ demand is surging, not declining
10. Interview topics haven't changed as much as the ML hype would suggest

---

## Section 8: Source Summary

### Reports Synthesized

| Report | Scope | Sources Consulted |
|--------|-------|-------------------|
| **Report 1: Quant Fund Jobs** | 45-50 job listings across 23 firms (Jane Street, Citadel, Two Sigma, D.E. Shaw, HRT, IMC, Optiver, SIG, XTX, G-Research, etc.) | Direct career pages, OpenQuant, BuiltIn, ZipRecruiter, QuantBlueprint, QuantStart, eFinancialCareers, Selby Jennings, Paragon Alpha |
| **Report 2: Bank/AM/Fintech Roles** | 22 search queries across 5 streams covering sell-side banks (GS, JPM, MS), asset managers (BlackRock, MSCI, Vanguard), and vendors (Bloomberg) | Career pages, eFinancialCareers, industry analysis, recruiter sources |
| **Report 3: MFE Syllabi & Certifications** | 13 MFE/MFin degree programs (CMU, Princeton, Baruch, NYU, Columbia, Stanford, MIT, Oxford, Imperial, ETH, UCL, Berkeley, Cornell) + 4 certifications (CQF, FRM, CFA, CAIA) + 6 online programs | Program websites, course catalogs, certification syllabi |
| **Report 4: Community/Practitioner Views** | Reddit r/quant, QuantNet, Wall Street Oasis, Hacker News, Wilmott forums; blogs from Kris Abdelmessih, Cliff Asness/AQR, Lopez de Prado, Paleologo, Derek Snow, Matt Levine; podcasts; 10+ recent books; interview prep resources; academic surveys | Forum threads, blog posts, newsletter archives, podcast episodes, book reviews, interview guides, arXiv papers |
| **Report 5: Emerging Trends & Wild Cards** | Conference programs (ICAIF 2025, NeurIPS 2025, QuantMinds 2025, ICML 2025, Quant Strats 2025); frontier research; industry reports (AIMA, McKinsey, Oliver Wyman); infrastructure analysis | Conference proceedings, arXiv papers, industry reports, market research, vendor analyses |

### Approximate Total Sources

- **Job listings reviewed:** ~50-60 distinct roles
- **Academic programs analyzed:** 13 degree + 4 certification + 6 online = 23
- **Community forum threads:** ~15-20 substantive discussions
- **Practitioner blog/newsletter sources:** ~15-20
- **Conference papers/presentations:** ~40-50
- **Books referenced:** ~15-20
- **Industry reports:** ~8-10
- **Total approximate sources:** 170-200+

### Research Date

All research conducted in February 2025-2026. Job listings, conference programs, and industry reports reflect the 2025-2026 landscape. Historical comparisons draw on 2022-2024 data where available.

### Limitations

- Job listings represent a snapshot; firm needs change quarterly.
- Conference papers represent frontier research, not necessarily production practice. There is a 2-5 year lag between conference presentation and production deployment.
- Community forums reflect vocal practitioners, who may not be representative of the full industry.
- Compensation data is approximate and varies significantly by firm, location, and individual.
- The "frequency in listings" metric from Report 1 is approximate (based on ~50 listings) and should be treated as directional, not precise.

---

*Synthesis compiled: February 2026*
*Based on 5 raw research reports covering ~170-200+ distinct sources*
*This document is designed to be standalone -- no reference to the raw reports is necessary to use it.*
