# Course Restructure: Topic Requirements & Research Methodology

## Context

We're restructuring an 18-week "ML for Quantitative Finance" course. The target audience is **ML/DL experts** who are learning finance applications — NOT beginners learning ML/DL fundamentals.

## Core Constraint: No Teaching Fundamental ML/DL

The audience already knows:
- Feedforward networks, CNNs, RNNs/LSTMs/GRUs, Transformers, attention mechanisms
- Training loops, loss functions, optimizers, regularization, batch norm, dropout
- Generative models (VAEs, GANs, diffusion models)
- LLMs, fine-tuning, embeddings, prompt engineering
- Standard ML (linear models, trees, ensemble methods, SVMs, etc.)
- RL fundamentals (MDPs, policy gradient, value methods, PPO, SAC, etc.)

**What the course SHOULD do with ML/DL:**
- Brief recap: "here's the architecture, here's why its properties matter for THIS financial problem" (1-2 paragraphs max)
- Deep dive into the **finance-specific application**: what's the financial problem, why does this model class solve it, what are the domain-specific pitfalls
- Focus on financial feature engineering, financial loss functions, financial evaluation metrics, financial data quirks

**What the course should NOT do:**
- Teach how backpropagation works
- Explain LSTM gating from scratch
- Walk through transformer self-attention derivation
- Explain what dropout/batch norm/skip connections are
- Build up from perceptrons to deep networks
- Teach RL from MDPs up

## Weeks Currently Flagged for Compression

### Must Compress (W7-W10, W14) — "Why it's used, not how it's built"
| Week | Current Topic | Problem | What to Keep | What to Cut |
|------|---------------|---------|--------------|-------------|
| W7 | Feedforward Nets for Asset Pricing | Teaches PyTorch fundamentals, explains what neural nets are | Gu-Kelly-Xiu replication, financial loss functions (IC-based, Sharpe-based), ensemble-over-seeds technique, temporal train/val/test splitting | PyTorch tensor basics, architecture explanation (depth/width/skip connections as concepts), why ReLU works |
| W8 | LSTM/GRU for Volatility & Returns | Explains RNN→LSTM→GRU evolution, vanishing gradients | Volatility forecasting application (HAR vs. GARCH vs. LSTM showdown), financial sequence construction, QLIKE loss | RNN/LSTM/GRU architecture explanation, vanishing gradient explanation |
| W9 | Foundation Models for Financial TS | Already well-focused on finance — mainly fine | Kronos/FinCast/TFT for finance, the "do TSFMs work for finance?" debate, hybrid approach | Any generic transformer explanation |
| W10 | NLP/LLM Embeddings for Finance | Already well-focused — mainly fine | FinBERT vs. LLM embeddings for alpha, Chen-Kelly-Xiu results, agentic AI for quant research | Any explanation of how attention/transformers work |
| W14 | Derivatives Pricing with Neural Networks | Currently named "Neural Options" — teaches Black-Scholes basics | Neural net surrogates for fast pricing, deep hedging, IV surface learning | Building up from call/put payoffs, BS derivation from scratch (brief mention OK) |

### Grey Area (W11-W13) — "Prefer why it's used over how it's built"
| Week | Current Topic | Assessment |
|------|---------------|------------|
| W11 | Bayesian DL & Uncertainty | MC Dropout and Deep Ensembles are arguably niche enough that a brief "how" is warranted. The financial application (position sizing, regime detection via uncertainty) is the real content. Keep the "how" very brief. |
| W12 | GNNs for Finance | GNNs are niche enough that the audience may not know them well. Brief architecture recap OK. The real content is graph construction for stocks (correlation, sector, supply chain) and when GNN beats XGBoost. |
| W13 | RL for Portfolios | RL formulation for trading is the content. Brief recap of PPO/SAC is fine. Focus on the financial MDP formulation, reward shaping, and why RL is hard in finance. |

## Previously Identified Major Gaps

From our earlier analysis, these financial topics are **missing** from the course:

### 1. Options & Derivatives (BIGGEST GAP)
- No coverage of: Black-Scholes intuition, implied volatility surfaces, Greeks, ML-driven delta hedging
- Huge fraction of quant roles involve derivatives
- ML applications: learning vol surface with neural nets (Horvath et al.), deep hedging under transaction costs, calibrating stochastic vol models with NNs
- W14 currently covers some of this but is titled "Neural Options" and buries derivatives knowledge under DL teaching

### 2. Execution Algorithms & Optimal Execution
- No coverage of: Almgren-Chriss optimal execution, TWAP/VWAP/IS algorithms, RL for order scheduling
- Many "quant" roles are actually execution quant roles
- W15 briefly mentions Almgren-Chriss but doesn't build it

### 3. Fixed Income (completely absent)
- No yield curve modeling, duration/convexity, interest rate models, ML for credit spreads
- Fixed income ≈ half of global capital markets by notional
- Defensible omission if course is "ML for equities," but title says "Quantitative Finance"

### 4. Regime Detection / Hidden Markov Models
- GARCH captures vol clustering (W2), but no explicit regime-switching models
- HMMs for bull/bear detection, Markov-switching GARCH, online changepoint detection
- Practically important; alpha decay (W18) is caused by regime shifts

### 5. Statistical Arbitrage / Pairs Trading
- No cointegration-based pairs trading, mean-reversion strategies, Ornstein-Uhlenbeck framework
- Some of the most intuitive and historically important quant strategies

### 6. Minor Gaps
- Credit risk / default prediction (important in banking, less for buy-side)
- Black-Litterman portfolio optimization (industry-standard, missing from W3)
- ESG quantification (emerging but increasingly relevant)
- Macro factor modeling (partially in Fama-French but no dedicated treatment)

## Research Methodology

### Goal
**Discover** which financial/quant topics are truly essential in 2025-2026. The research should be **exploration-first** — start from what the industry actually demands and work backward to course content, rather than starting from our existing topic list and looking for confirmation.

### Research Philosophy

The research has two phases:

**Phase 1: Open Discovery (most important)**
- Start from broad, unbiased sources: job listings, MFE syllabi, practitioner discussions, conference programs
- Extract **whatever topics appear**, not just the ones we've already identified
- Look for topics we haven't thought of at all — emerging areas, cross-disciplinary skills, tooling/infrastructure knowledge, soft skills that quant roles require
- Pay attention to the **relative frequency** of topics — what shows up in every listing vs. what's rare
- Note any surprising omissions from our current course or surprising emphases in the real world

**Phase 2: Classification & Depth Assessment**
- Only after open discovery, classify each discovered topic
- Include topics from our existing course that the research supports AND any new topics discovered
- Flag any current course topics that the research suggests are less important than we assumed

### Sources to Search (in priority order)

#### Tier 1: Job Market Reality (THE ground truth)
1. **Quant job listings from top firms** — Jane Street, Citadel/Citadel Securities, Two Sigma, DE Shaw, Jump Trading, HRT, Optiver, IMC, Millennium, Point72, Bridgewater, AQR, Man Group, WorldQuant, Susquehanna (SIG), Balyasny, Squarepoint, XTX Markets
2. **Bank quant/strat roles** — JPMorgan, Goldman Sachs, Morgan Stanley, Barclays, Deutsche Bank, UBS, BNP Paribas
3. **Asset managers & fintech** — Bloomberg, MSCI, BlackRock (Aladdin), Vanguard, State Street, Robinhood, Stripe
4. **Job aggregators** — efinancialcareers, Quantnet jobs, LinkedIn "quantitative researcher" / "quantitative developer"

**What to look for:** Don't just check skills lists. Read full job descriptions. Look for:
- Required vs. preferred qualifications
- Day-to-day responsibilities (what do they actually DO?)
- Which asset classes they mention
- Which tools/frameworks they name
- What they test in interviews (if mentioned)
- Any non-obvious requirements (communication, stakeholder management, regulatory knowledge, etc.)

#### Tier 2: Academic & Practitioner Curriculum
5. **Top MFE/MFin syllabi** — CMU MSCF, Princeton MFin, Baruch MFE, NYU MFE, Columbia MFE, Stanford MSFM, Oxford MCF, Imperial College, ETH Zurich, UCL
6. **Professional certifications** — CQF (Certificate in Quantitative Finance) syllabus, FRM syllabus, CFA Level III quant methods
7. **Recent ML-for-finance courses** — any publicly available university courses on ML applied to finance (not just traditional MFE courses)
8. **Conference programs** — ICAIF, NeurIPS/ICML finance workshops, Risk.net Quant Summit, Global Derivatives, QuantMinds

#### Tier 3: Community Signal & Emerging Trends
9. **Community forums** — r/quant, Wilmott forums, QuantNet forums, Nuclear Phynance — specifically threads about "what skills matter," "what I wish I knew," career advice
10. **Practitioner blogs/newsletters** — Flirting with Models, Quantocracy, Alpha Architect, AQR Cliff's Perspective, Matt Levine (Bloomberg), Kris Abdelmessih
11. **Books published 2024-2026** — what new quant finance textbooks cover; what's being written about NOW
12. **Industry reports** — Greenwich Associates, Coalition, Oliver Wyman reports on quant finance trends

#### Tier 4: Wild Cards (things we might be missing entirely)
13. **Adjacent fields bleeding into quant** — climate risk modeling, geopolitical risk quantification, supply chain analytics, computational social science
14. **Regulatory/compliance ML** — AML, KYC, model risk management (SR 11-7), explainability requirements
15. **Infrastructure & engineering skills** — what quant roles actually require beyond modeling (databases, cloud, APIs, production systems)
16. **Interview prep resources** — QuantNet interview guides, Heard on the Street, Green Book — what do firms actually test?

### Classification Framework

For each topic discovered (both from our existing list AND newly discovered), classify into:

| Tier | Description | Action |
|------|-------------|--------|
| **MUST KNOW** | Required for 80%+ of quant roles; table stakes | Must be in the course |
| **SHOULD KNOW** | Required for 40-80% of roles; strong differentiator | Should be in the course if space permits |
| **GOOD TO KNOW** | Required for 15-40% of roles; valuable but specialized | Include if fits naturally; can be elective content |
| **NICHE** | Required for <15% of roles; specialized domain | Brief mention at most; point to resources |
| **DECLINING** | Was important but being automated/replaced | Mention for awareness; don't build skills around it |

### What to Capture for Each Topic

1. **Tier classification** (MUST/SHOULD/GOOD/NICHE/DECLINING)
2. **Evidence** — specific sources that support this classification (quote job listings, name syllabi, cite forum threads)
3. **Job roles where this is required** — be specific (e.g., "vol trader at Citadel Securities" not just "quant"). List ALL relevant role types.
4. **Current industry trend** — growing, stable, or declining? Why?
5. **Recommended learning depth** — how deep should an ML/DL expert go into this to be competitive? (mastery, working knowledge, conceptual awareness, or skip)

### Key Questions the Research Must Answer

The output of this research is a **standalone guide**: "What should an ML/DL expert learn to break into quantitative finance and adjacent 'money' fields?" It is NOT about evaluating any specific course — it is a general-purpose map of the landscape.

1. **What does the full landscape of quant/finance knowledge look like for an ML/DL expert?** — every topic area that matters, from foundational to frontier, organized and prioritized
2. **What are the distinct career paths, and how do their knowledge requirements differ?** — quant researcher vs. quant developer vs. execution quant vs. risk quant vs. vol trader vs. strat vs. data scientist at a fund — map the paths and their skill profiles
3. **What is table-stakes vs. differentiating vs. niche?** — the tiering itself, grounded in frequency across job listings and syllabi
4. **What's the balance between domain knowledge (finance) vs. applied modeling vs. engineering/production skills?** — how much of each does the industry actually demand?
5. **What's genuinely new or rapidly growing in 2025-2026?** — emerging areas that weren't in 2023 syllabi or job listings but are showing up now
6. **What's declining or being automated away?** — areas where human expertise is being replaced by tools, LLMs, or standardized platforms
7. **What non-obvious skills or knowledge areas appear in job listings?** — things beyond the standard "stochastic calculus + ML" that firms actually ask for
