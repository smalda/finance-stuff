# PHASE 1: OPEN DISCOVERY - RAW FINDINGS
## Emerging Trends, Wild Cards, and Non-Obvious Areas in Quantitative Finance (2025-2026)
### Research Date: February 15, 2026

---

## 1. CONFERENCE PROGRAM ANALYSIS -- What Researchers Are Presenting NOW

### ICAIF 2025 (ACM Conference on AI in Finance) -- Singapore, Nov 15-18, 2025
- **Record submissions**: 349 papers submitted (38.5% increase over 2024, 74.5% growth since 2023) -- signals exploding academic interest
- **Program scale**: 10 workshops, 4 tutorials, 5 competitions, Industry Day
- **First time held outside North America** (Singapore) -- reflects the global shift of quant finance toward Asia-Pacific

**Key paper topics at ICAIF 2025:**
- Continuous-Time RL for Asset-Liability Management
- Interpretable Market Simulations via Optimal Transport
- Federated Financial Reasoning Distillation (training small financial experts from multiple teacher LLMs)
- Leveraging Deep Learning for Monte Carlo Calibration of Rough Stochastic Volatility Models
- IKNet: Interpretable Stock Price Prediction via Keyword-Guided News/Technical Integration
- Multi-Agent Reinforcement Learning for Market Making
- FinReflectKG: Agentic Construction of Financial Knowledge Graphs
- Large Language Model Agents for Investment Management
- FAITH: Framework for Assessing Intrinsic Tabular Hallucinations in Finance
- Structured Agentic Workflows for Financial Time-Series Modelling with LLMs

**Emergent themes from ICAIF 2025:**
- LLM agents are THE dominant topic (agents for investment, knowledge graph construction, agentic workflows)
- "Rough" stochastic volatility + deep learning calibration = the new sell-side frontier
- Federated/privacy-preserving approaches are entering the mainstream
- Interpretability and hallucination detection are no longer afterthoughts
- Multi-agent RL for market making signals production-readiness

### NeurIPS 2025 -- Workshop on Generative AI in Finance (San Diego, Dec 2025)

**Selected accepted papers (revealing frontier topics):**
- **Best Paper**: "Factor-Based Conditional Diffusion Model for Portfolio Optimization" (Gao, He -- CUHK)
- White Box Finance: Interpreting AI Decisions through Rules and Language Models
- Democratizing Alpha: LLM-Driven Portfolio Construction for Retail Investors
- Context-Masked Meta-Prompting for Privacy-Preserving LLM Adaptation in Finance
- EvoAlpha: Evolutionary Alpha Factor Discovery with Large Language Models
- **Kronos: A Foundation Model for the Language of Financial Markets** -- MAJOR: foundation model specifically for financial language
- Stress-Aware Scenario Generation for Reliable Portfolio Inference under Regime Shifts
- CTBench: Cryptocurrency Time Series Generation Benchmark
- CMS-VAE: Strategy-aware VAE for High-Fidelity Crypto Market Simulation
- Compliant Generative Diffusion for Finance
- FISCAL: Financial Synthetic Claim-document Augmented Learning for Fact-Checking
- FinAgentBench: Benchmark Dataset for Agentic Retrieval in Financial QA
- Structured Agentic Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback
- MASFIN: Multi-Agent System for Decomposed Financial Reasoning and Forecasting

**Bloomberg's contributions at NeurIPS 2025:**
- 2 papers published -- institutional engagement from data/analytics firms is growing

**Key NeurIPS 2024 finance contributions:**
- FinBen: A Holistic Financial Benchmark for LLMs (42 datasets, 24 financial tasks)
- Capital One: Scaling laws for large time-series models

### QuantMinds International 2025 (London, Nov 17-20, 2025)

**Plenary themes (revealing what practitioners care about):**
- "AI and the Future of Quantitative Finance" -- Panel with Rama Cont (Oxford), Hans Buehler (XTX), Andrey Chirikhin (Schonfeld), Nicole Konigstein (quantmate), Dara Sosulski (HSBC)
- "The Psychology of AI" -- Alexander Sokol (CompatibL) -- behavioral/cognitive dimension of AI in quant
- "Escaping Lock-in and Numerical Instability in Quantitative Analytics" -- infrastructure/engineering focus
- "Geopolitics, Markets, and the Future of the Derivatives Market" -- geopolitical risk modeling going mainstream
- "A Generic Approach to Statistical Arbitrage" -- Bruno Dupire (Bloomberg)
- **"The Rise of Emerging DeFi Markets and the Thawing of the Crypto Winter"** -- DeFi entering mainstream quant conferences

**Day Two key sessions:**
- "Data Management in 2025: New Data Sources, Alternative Data, Models, and Strategies"
- "The Rise of the Machines: Quant Credit Takes Center Stage" -- quant credit is a growing area
- "NLP, Hype Index and Identification of Financial Bubbles" -- Helyette Geman (Johns Hopkins)
- "Infrastructure's Role in Accelerating Quant Research" -- Maxim Morozov (ArcticDB)
- "From Tariff Shock to Network Signal" -- S&P Global -- supply chain/geopolitical network analysis
- "Latest Developments in GenAI Pricing" -- Youssef Elouerkhaoui (Citigroup)
- "Reconciling P- and Q-Calibration" -- Julien Guyon (ENPC/NYU) -- bridging risk-neutral and real-world measures
- "Dynamic Hedging and P&L Decomposition Under Model Uncertainty" -- Rama Cont

**Parallel tracks reveal four domains:** Delta (Derivatives Pricing), Theta (Quant Dev & Innovation), Alpha (Alpha & Allocation), Vega (Risk & Hedging)

**Notable: "Women in Quant Finance" lunch discussion and "Permission to Press Pause" wellness session signal cultural shifts in the industry**

### Quant Strats 2025 (London/New York)
- 600+ attendees; 70+ speakers
- Key themes: Data/AI/Applied Innovation, Modelling Alternative Markets, Portfolio Optimization & Risk Management, Front-to-Back Integration
- Debate: "Will LLMs or real-time sentiment analytics replace traditional macro risk modelling?"
- Focus on reinforcement learning for execution and portable alpha

### ICML 2025 (Vancouver)
- MACAW: Multi-Agentic Conversational AI Workflow for quantitative financial reasoning (Capital One)
- 3,300+ papers accepted total; finance-specific papers scattered across sessions

---

## 2. EMERGING FRONTIER AREAS -- Evidence of Real Adoption vs. Hype

### A. AI Agents for Quantitative Finance (HIGH SIGNAL -- GENUINELY NEW)

**Evidence of real development:**
- TradingAgents framework: Multi-agent LLM trading framework emulating a trading firm with specialized roles (fundamental analysts, sentiment analysts, technical analysts, traders with varied risk profiles, Bull/Bear researchers, risk management team)
- AgentQuant: Autonomous platform transforming stock lists into fully backtested strategies using AI agents
- AI-Trader: Benchmarking Autonomous Agents in Real-Time Financial Markets (arXiv Dec 2024)
- StockBench: Testing whether LLM agents can trade stocks profitably in real-world markets

**Reality check:**
- Most LLM agents exhibit poor returns and weak risk management vs. simple baselines
- General intelligence does not automatically translate to effective trading capability
- "While current LLM agents could operate profitably, they still struggle to consistently outperform simple baselines"
- LLMs can be prompted to conceal material risks, generate misleading narratives, or provide harmful guidance (red-teaming findings)

**Assessment: EARLY STAGE but with genuine paradigm-shift potential. The multi-agent architecture (specialist agents debating) is more promising than single-agent approaches. Production deployment is 2-3 years away for actual trading; nearer-term for research augmentation.**

### B. Synthetic Data Generation (MATURING -- REAL ADOPTION)

**Key developments:**
- CFA Institute published a major 2025 report: "Synthetic Data in Investment Management"
- Gartner predicted: By 2025, synthetic data will exceed real data in AI model training for 60% of financial applications
- Diffusion models are the new frontier (replacing GANs): TRADES framework generates realistic order flows; diffusion-based LOB simulation
- JP Morgan has published foundational work on synthetic data generation in finance
- Applications: model training, backtesting, portfolio optimization, risk modeling, sentiment analysis

**Specific technical advances:**
- Denoising Diffusion Probabilistic Models (DDPMs) + wavelet image transformation for synthetic financial time series
- Score-based generative models with cross-attention controls enabling "what-if" scenario design
- TRADES framework: Transformer-based diffusion for LOB simulation, covers 67% of real data distribution
- Image-based diffusion methods converting LOB data into structured images

**Assessment: REAL AND MATURING. Diffusion models for financial data generation is a genuine breakthrough area. The shift from GANs to diffusion models is significant and mirrors the broader ML field.**

### C. Causal Inference in Finance (GENUINELY NEW -- HIGH INTELLECTUAL VALUE)

**Key developments:**
- Marcos Lopez de Prado's "Causal Factor Investing" (Cambridge Elements, 2025) -- landmark publication
- CFA Institute published "Causality and Factor Investing: A Primer"
- "Factor Mirage" concept introduced: factor models that appear valid but are causally misspecified
- Collider bias and confounder bias in factor models exposed
- LLM-based causal discovery: using language models to automate expert judgment for causal discovery

**Technical advances:**
- Causal discovery algorithms applied to factor investing
- Causal Machine Learning (CML) for asset pricing -- shifting from prediction to explanation
- Causal methods identifying relationships that persist across regime changes (unlike correlation-based methods)
- "Causal Network Representations in Factor Investing" (Wiley, 2025)

**Assessment: GENUINELY NEW AND IMPORTANT. This is not rebranded statistics. The "factor mirage" concept alone could reshape how the industry thinks about factor investing. Lopez de Prado's work is creating a paradigm shift from correlation-based to causation-based factor analysis.**

### D. Climate Risk & Carbon Markets (GROWING BUT FRAGMENTED)

**Key developments:**
- NGFS published first short-term climate scenarios (May 2025) with new damage functions
- FSB Roadmap for Addressing Financial Risks from Climate Change (2025 update)
- G20 Common Carbon Credit Data Model (CCCDM) for standardization
- Carbon markets expanded from covering 7% to 24% of global emissions
- Modeling frameworks incorporating carbon price dynamics into credit default probability
- Nature/biodiversity risk emerging as the next frontier beyond climate

**Challenges:**
- No singular quantitative model captures carbon risk in portfolios
- Lack of standardization, inadequate monitoring, insufficient transparency in carbon markets
- Cross-border systems remain underdeveloped

**Assessment: IMPORTANT BUT INFRASTRUCTURE-HEAVY. The modeling challenges are genuine and create opportunities for quants. Nature/biodiversity risk is the "next climate risk" -- even less modeled. This is a multi-decade area of growth.**

### E. Quantum Computing in Finance (TRANSITIONING FROM HYPE TO EARLY PRODUCTION)

**Surprising findings -- more real than expected:**
- Market: $1.67B globally, finance holds ~20% of applications
- Funding tripled: $3.77B in first 9 months of 2025 vs. $1.3B all of 2024
- HSBC: Used IBM Heron quantum computer to improve bond trading predictions by 34%
- Goldman Sachs: Quantum algorithms for risk analysis achieving 25x faster processing
- JPMorgan Chase: Adopted quantum cryptography securing trillions in transaction volume
- One institution completed an analysis in 7 seconds that would traditionally take years

**Key application areas:**
- Corporate banking: 31% of total quantum computing market
- Risk and cybersecurity: 26%
- Retail banking: 14%
- Payments and wealth management: 13% each

**But also:**
- FCA published a research note on quantum computing applications in financial services
- Practical focus: hybrid quantum-classical methods for optimization, Monte Carlo simulation, targeted ML
- Post-quantum cryptography is an urgent parallel track (see below)

**Assessment: NO LONGER PURE HYPE. The HSBC and Goldman numbers are significant. However, most institutions are still in pilot/hybrid mode. The real near-term value is in post-quantum cryptography migration, which is urgent and well-funded.**

### F. Post-Quantum Cryptography Migration (WILD CARD -- UNDERAPPRECIATED)

**This is a major hidden trend:**
- Government deadlines: Critical financial systems must transition to PQC by 2030; full migration by 2035
- NIST released first PQC standards in 2024
- EU requires national strategies by Dec 2026
- Mastercard published a PQC migration white paper in 2025
- FS-ISAC published "The Timeline for Post Quantum Cryptographic Migration" (Sep 2025)
- Europol developed a prioritization/scoring framework for banks
- Previous migrations (SHA-1 to SHA-2) took over a decade; PQC is more complex

**Assessment: URGENT AND UNDERAPPRECIATED. This will create massive demand for quants/engineers who understand both cryptography and financial systems. Every financial institution needs a PQC migration plan.**

### G. Geopolitical Risk Modeling (GROWING -- INCREASINGLY QUANTITATIVE)

**Key developments:**
- BlackRock Geopolitical Risk Dashboard -- institutional adoption
- BBVA Research: ML approach using neural network language models (not keyword search) across 42 economies
- Random Forests delivering largest predictive accuracy gains for sovereign CDS spreads
- LightGBM optimal for energy market tail risk forecasting (geopolitical indicators = 19.15% of predictive power)
- Cross-sector contagion analysis using Quantile-on-Quantile Connectedness

**Assessment: GENUINELY BECOMING QUANTITATIVE. The shift from keyword-based approaches to neural network language models for geopolitical risk is real and significant. This area has strong demand from institutional investors.**

### H. Private Markets ML (LARGE OPPORTUNITY -- EARLY STAGE)

**Key findings:**
- 53% of PE firms expect to hire more digital transformation specialists
- 51% seeking more data scientists and AI specialists
- AI/ML accounted for 71% of total VC deal value in Q1 2025
- Private credit yields ~10% with low defaults -- analytics needed for this growing market
- Analytics used for operator management, service delivery, and alpha generation

**Assessment: LARGE UNDERSERVED MARKET. Public market quant techniques are well-developed; private markets analytics is still early. The data scarcity problem creates unique challenges that reward creative quantitative approaches.**

### I. Tokenization of Real World Assets (RAPIDLY GROWING)

**Key numbers:**
- On-chain tokenized RWAs tripled from ~$5.5B to ~$18.6B during 2025
- By October 2025: ~$33B total
- Projections: ~$2T by 2030
- Private credit: largest category at >$18.91B on-chain
- Tokenized US Treasuries: >$9B
- BlackRock, Franklin Templeton, Nasdaq moved from pilots to production-level tokenization

**Assessment: REAL AND ACCELERATING. This is no longer a crypto-native phenomenon. Traditional finance institutions are actively tokenizing. Creates demand for quants who understand both TradFi and DeFi.**

### J. Federated Learning in Finance (EARLY BUT STRATEGIC)

**Applications:**
- Fraud detection: sharing insights without exposing customer data
- Credit scoring: collaborative model building with privacy preservation
- AML: cross-institution pattern detection
- Techniques: homomorphic encryption, differential privacy, secure aggregation

**Market:**
- $0.1B in 2025, projected $1.6B by 2035 (27% CAGR)
- Large enterprises: 63.7% market share

**Assessment: STRATEGICALLY IMPORTANT BUT SLOW TO ADOPT. Regulatory pressure for privacy + need for cross-institution data will drive this. The technology works; adoption is constrained by legal/governance issues.**

### K. Foundation Models for Financial Time Series (GENUINELY NEW)

**Major developments:**
- **FinCast**: First foundation model specifically for financial time-series forecasting -- robust zero-shot performance
- **Kronos**: Foundation model for the language of financial markets (NeurIPS 2025)
- **Large Investment Model (LIM)**: Foundation model trained on vast financial datasets for universal market patterns
- **DeePM**: Regime-robust deep learning for macro portfolio management -- 2x net risk-adjusted returns vs. trend-following

**Key finding: Off-the-shelf pre-trained time series FMs perform poorly on financial data; models pre-trained FROM SCRATCH on financial data achieve substantial improvements. Domain-specific adaptation is essential.**

**Assessment: GENUINELY NEW AND IMPORTANT. Financial foundation models are an emerging category that didn't exist 2 years ago. The key insight that generic FMs fail but finance-specific FMs succeed is critical.**

### L. Diffusion Models in Finance (THE NEW GENERATIVE FRONTIER)

**Applications expanding rapidly:**
- NeurIPS 2025 Best Paper: Factor-Based Conditional Diffusion Model for Portfolio Optimization
- TRADES: Transformer-based diffusion for LOB simulation
- "Painting the market": Diffusion models for limit order book simulation and forecasting
- "Beyond Monte Carlo": Harnessing Diffusion Models for Financial Market Dynamics
- Data-driven generative simulation of SDEs using diffusion models
- "Compliant Generative Diffusion for Finance" -- regulatory-aware generation

**Assessment: THIS IS THE HOTTEST TECHNICAL AREA IN QUANT FINANCE RIGHT NOW. Diffusion models are displacing GANs for synthetic data, replacing Monte Carlo for scenario generation, and enabling new approaches to derivatives pricing. A "must-know" area.**

### M. Graph Neural Networks in Finance (MATURING)

**Applications:**
- Financial risk analysis and cross-market contagion modeling
- Fraud detection (outperforming traditional methods significantly)
- Stock price prediction using dynamic relational graphs
- TFT-GNN hybrids for daily stock price forecasting
- Multi-scale multimodal dynamic graph convolutional networks

**Assessment: MATURE AND DEPLOYED. GNNs are no longer frontier -- they're production tools for fraud detection and increasingly for risk contagion analysis. The frontier is now temporal and heterogeneous GNNs.**

### N. Reinforcement Learning in Trading (IN PRODUCTION)

**Evidence of real deployment:**
- JPMorgan and Goldman Sachs: production systems for options/derivatives pricing and trading
- Jump Trading: AI engine for real-time HFT strategy optimization
- Point72: NLP-powered sentiment analysis for options trading
- Deep Distributional RL (D4PG) with quantile regression for gamma/vega hedging
- Hybrid approach adoption: 15% (2020) --> 42% (2025); pure RL declined from 85% to 58%

**Key insight: The field has matured from "pure RL" to "hybrid RL + domain knowledge." Multiple specialist RL agents collaborating is the cutting edge.**

**Assessment: REAL AND IN PRODUCTION at major firms. The shift to hybrid approaches is the signal -- pure ML without domain knowledge underperforms. Multi-agent RL is the next frontier.**

### O. Conformal Prediction for Finance (EMERGING -- NON-OBVIOUS)

**Key development:**
- Distribution-free uncertainty quantification gaining traction for stock selection
- Achieving empirical coverage matching nominal confidence levels
- NeurIPS 2025 Highlight paper on conformal prediction for time series with change points
- Challenge: exchangeability assumption violated in time series data
- Solutions: reweighting calibration data, dynamically updating residual distributions

**Assessment: NON-OBVIOUS AND IMPORTANT. Conformal prediction provides rigorous uncertainty quantification without distributional assumptions -- exactly what regulators want. Could become a standard tool for model risk management.**

---

## 3. INFRASTRUCTURE & ENGINEERING REQUIREMENTS

### Programming Languages

**Python:** Dominant for research, prototyping, and increasingly for production. Enhanced by NumPy, Pandas, and specialized libraries. Quants with Python command highest average salaries.

**C++:** Still essential for low-latency trading (HFT) and production systems. "Milliseconds can mean the difference between profit and loss."

**Rust:** EMERGING. A few newer HFT firms (mostly crypto) building infrastructure in Rust. RustQuant library exists. Key advantage: memory safety without garbage collection = more predictable latency distribution. Most established firms won't port existing C++ infrastructure, but greenfield projects increasingly consider Rust.

**SQL:** Universal requirement for data engineering roles.

**R:** Still used but declining relative to Python.

### Data Stack Evolution

**KDB+/q -- Declining:**
- Still backbone of HFT at major banks (JP Morgan, Goldman, Citadel)
- BUT: $250K+ annual licenses, opaque pricing, exotic syntax
- Gartner predicted 40% of users will migrate to hybrid solutions by 2026
- Open-source alternatives offer 80% performance at fraction of cost

**Rising alternatives:**
- **Polars + DuckDB**: The new default research data stack. Fast, free, composable. "Build a lightning-fast research database" pattern is spreading.
- **Apache Iceberg**: Enabling bidirectional data movement between local analytics and cloud data lakes
- **Snowflake**: Cloud data warehousing for larger-scale analytics (but compute is expensive)
- **ArcticDB**: Specifically presented at QuantMinds 2025 for "accelerating quant research"
- **TimescaleDB**: Open-source, PostgreSQL-compatible time-series database

### MLOps & Production ML

**Key requirements:**
- MLflow: Most widely adopted open-source MLOps platform in 2025
- Feature stores: Tecton (enterprise gold standard), Hopsworks (regulated industries), Feast (open source)
- Real-time inference serving at millisecond latency for fraud detection, trading signals
- CI/CD pipelines for ML models with regulatory audit trails
- Model monitoring and drift detection

**Financial-specific MLOps challenges:**
- Regulatory compliance (SR 11-7, EU AI Act)
- Model explainability documentation
- Data lineage and reproducibility
- Santander study: MLOps-managed ML models saved 12.4-17% in regulatory capital

### GPU Computing

**Rapidly growing in quant finance:**
- NVIDIA CUDA acceleration: up to 160x speedups in portfolio optimization
- JAX-LOB: GPU-accelerated limit order book simulator
- VectorAlpha: Open-source GPU-accelerated trading libraries
- RAPIDS/cuDF for accelerated data processing
- JAX gaining traction for auto-differentiation + JIT compilation + GPU parallelization

### Cloud Infrastructure

**Market dynamics:**
- AWS, Azure, GCP collectively: 66% of global cloud infrastructure spending
- GenAI-specific cloud services: 140-180% growth in Q2 2025
- Hyperscaler capex: $400B+ annually on AI infrastructure
- Hedge funds increasingly cloud-native, but data security concerns persist

### Data Engineering Skills in Demand

- Python + SQL + time-series databases (core)
- Data pipeline architecture (ingestion, transformation, distribution)
- Market data vendor expertise (Bloomberg, Refinitiv, Factset)
- Alternative data processing (satellite, NLP, web scraping)
- Streaming/real-time data processing
- Data quality and governance

---

## 4. INDUSTRY TREND REPORTS -- Key Findings

### Hedge Fund Technology (AIMA Survey 2025)

**AI Adoption:**
- 95% of hedge funds use generative AI (up from 86% in 2023)
- 58% expect GenAI to play larger role in investment decisions (up from 20% in 2023)
- 90% of investors believe GenAI will enhance hedge fund performance over next 3 years
- 60% of investors more likely to allocate to managers investing in AI

**GenAI Use Cases:**
- Goldman Sachs: GS AI Assistant launched firmwide (mid-2025, after 10,000-employee pilot)
- Research and data analysis remain most common applications
- Front-office use growing fastest
- Larger multi-strategy managers ahead; smaller firms closing gap with public tools

### McKinsey Global Banking Annual Review 2025

**Key findings:**
- AI could cut bank operating costs by 15-20%
- But AI also disrupts traditional profit pools (customers using AI to manage finances)
- "Precision, not heft" as the defining characteristic of next era
- Early AI adopters: ROE increase of up to 4 percentage points
- Slow movers: risk long-term profit erosion
- Agentic AI could reduce banks' cost base by 15-20%

### Oliver Wyman

- Banks and fintechs must adapt as agentic AI evolves from helping customers to placing orders
- Investors may not have priced in full AI transformation potential

### Market Size & Growth

- Quant fund market: $2.5T in 2025, 12% CAGR to 2033
- Quant funds added $44B in assets early 2025
- Algorithmic trading market: projected to reach $43B by 2030
- Alternative data market: expected to reach $273B by 2032

### LLM Coding Assistants -- Reality Check

**Surprising findings:**
- AI coding assistants increase individual output 20-40% (vendor claims)
- BUT: Experienced developers took 19% longer with AI assistance
- Developers expected 24% speedup, believed they were faster even when they weren't
- 51.5% of coding session time spent in LLM interaction states (verifying, prompting, deferring)
- Code quality concerns: 4x growth in code clones attributed to AI assistants

**Assessment: The productivity gains from LLM coding assistants are REAL for certain tasks (boilerplate, documentation, simple scripts) but NEGATIVE for complex analytical work. Quant researchers should be aware of the perception gap.**

---

## 5. DECLINING/OVERHYPED AREAS

### Declining

**Stochastic Calculus (Selective Decline):**
- Less relevant since post-2008 derivatives scaling back
- Buy-side: ML skills now dominant requirement
- Stat-arb and HFT: statistical/data analysis skills more critical
- STILL relevant for: sell-side derivatives, fixed income quants, interest rate modeling
- Advice: "Keep it in your studies, but by no means the core of it"

**KDB+/q:**
- Being disrupted by open-source alternatives (Polars, DuckDB, QuestDB)
- New firms won't adopt; existing clients retained but not growing
- 40% user migration to hybrid solutions predicted by 2026

**Pure RL Trading (declining in favor of hybrid):**
- Pure RL adoption: 85% (2020) --> 58% (2025)
- Hybrid approach: 15% (2020) --> 42% (2025)
- Domain knowledge integration is essential

### Overhyped

**LLM Trading Agents (OVERHYPED in short term):**
- "Many AI trading tools are overhyped, overpriced, and underperforming"
- Companies "rely too heavily on LLMs like ChatGPT without offering any unique value"
- LLMs can be adversarially prompted to generate misleading financial guidance
- Most agents "fail to consistently outperform simple baselines"
- The multi-agent research approach is more promising than the "ChatGPT trades stocks" narrative

**AI Coding Productivity Claims:**
- Vendor claims of 20-40% productivity gains are overstated for complex work
- Experienced developers actually slower with AI for certain tasks
- 4x growth in code clones from AI assistants

**"AI Will Replace Quants" Narrative:**
- Pictet research: up to 50% of alpha in advanced quant strategies from AI-powered conditioning
- BUT: "human + machine, not machine alone" is the consensus
- ML is evolution, not revolution of quantitative investment process
- Most top firms invest more in execution ML than prediction ML

**ESG Quant (Cooling Off):**
- Was very hot 2020-2023
- Data quality and standardization issues remain unresolved
- Greenwashing concerns have dampened enthusiasm
- Still important but the gold-rush phase is over

---

## 6. ADJACENT FIELDS WITH QUANT DEMAND

### Energy Trading (HIGH GROWTH)
- US electricity demand projected to rise 25% by 2030
- Data centers: up to 11.7% of US power demand by 2030
- Thousands of nodal prices updating every 5 minutes
- Global energy trading market: $7.5B (2023) --> $12B (2030)
- Renewable energy integration creating massive complexity
- Demand for: algorithmic trading, predictive modeling, optimization, real-time analytics

### Sports Betting Analytics (ESTABLISHED AND GROWING)
- 123+ quantitative positions on Indeed alone
- $70K-$175K salary range
- Companies: Fanatics, DraftKings, BetMGM, Susquehanna International Group
- Skills overlap with quant finance: predictive modeling, risk management, real-time analytics
- Growing as legalization expands

### Insurance/Insurtech (LARGE AND GROWING)
- Insurtech market: $19B (2025) --> $133B (2034) at 24.1% CAGR
- AI/ML led the market by technology in 2025
- 50% of insurance claims automated using AI/ML by 2025
- Applications: underwriting, claims processing, fraud detection, dynamic pricing
- IPOs and M&A exits up 67% YoY

### DeFi/Crypto Quantitative Trading (INSTITUTIONALIZING)
- MEV arbitrage: institutional-grade, hyper-specialized industry
- Top 2 block builders capture 90%+ of Ethereum block auctions
- 90%+ of arbitrage through private MEV-Boost channels
- Automated Market Making: Loss Versus Rebalancing (LVR) is key metric
- US corporate bond trading: $50B+/day average in 2025

### Private Markets Analytics
- Massive underserved market
- PE firms actively hiring data scientists
- Private credit: $10%+ yields, growing rapidly
- Unique challenges: data scarcity, irregular valuations, limited transparency

---

## 7. TECHNOLOGY SHIFTS

### The Rise of Diffusion Models
- Displacing GANs for financial data generation
- Enabling new approaches to scenario generation and stress testing
- "Compliant" diffusion models designed with regulatory awareness
- Transforming limit order book simulation

### Financial Foundation Models
- FinCast, Kronos, LIM -- purpose-built for finance
- Critical finding: generic foundation models fail on financial data; domain-specific training essential
- Three modalities: language, time-series, visual-language

### Agentic AI Architecture
- Moving from single-model to multi-agent systems
- Specialist agents (analyst, trader, risk manager) debating and collaborating
- McKinsey estimates 15-20% cost reduction potential for banks
- Projections: AI systems completing 4 days of work without human oversight by 2027

### GPU Acceleration Going Mainstream
- 160x speedups in portfolio optimization
- JAX gaining traction (auto-diff + JIT + GPU in one framework)
- LOB simulators moving to GPU
- This makes previously intractable problems feasible

### Polars + DuckDB Replacing Pandas + KDB+
- For research workloads: Polars + DuckDB becoming standard
- For production HFT: KDB+ still entrenched but under pressure
- Apache Iceberg enabling hybrid local/cloud architectures

### Post-Quantum Cryptography
- NIST standards released 2024
- Migration deadlines: critical systems by 2030, full by 2035
- Every financial institution needs a migration plan
- Creates demand for crypto-engineering talent in finance

---

## 8. LLM/GenAI IMPACT -- Reality Check

### Where LLMs Are ACTUALLY Changing Quant Work (Not Hype)

**Proven, deployed applications:**
1. **Research augmentation**: Summarizing documents, earnings calls, regulatory filings -- Goldman's GS AI Assistant is firmwide
2. **Code generation for prototyping**: Accelerating Python/R script writing for data processing
3. **NLP sentiment extraction**: More sophisticated than keyword-based approaches
4. **Knowledge management**: Internal knowledge base search and synthesis
5. **Report generation**: Automating client reports and research summaries

**Emerging but promising:**
1. **Alpha factor discovery**: EvoAlpha (evolutionary LLM-driven factor discovery)
2. **Multi-agent research workflows**: Structured agentic workflows for financial analysis
3. **Financial knowledge graph construction**: FinReflectKG at ICAIF 2025
4. **Causal discovery automation**: Using LLMs to automate expert judgment for causal analysis
5. **Compliance and regulatory analysis**: Automated regulatory document processing

**Not yet proven / overhyped:**
1. **Direct trading decisions**: LLM agents don't consistently beat simple baselines
2. **Fully autonomous research**: Still requires heavy human oversight
3. **Replacing quantitative analysts**: Evolution, not revolution
4. **"AI-powered" retail trading tools**: Mostly marketing

### Key Statistics
- 95% of hedge funds use GenAI in some capacity (2025)
- 58% expect GenAI in investment decisions (up from 20% in 2023)
- Goldman Sachs: GS AI Assistant firmwide after 10,000 pilot
- Experienced developers: 19% SLOWER with AI coding assistants on complex tasks

---

## 9. WILD CARDS -- Things That Surprised Me

### 1. Post-Quantum Cryptography is a Financial Emergency
Most quant finance discussions ignore this entirely, but governments are setting hard deadlines (2030 for critical systems). Every bank needs a PQC migration plan. This will create a massive engineering program rivaling LIBOR transition in scope.

### 2. Diffusion Models Are Displacing Monte Carlo
The NeurIPS 2025 paper "Beyond Monte Carlo: Harnessing Diffusion Models to Simulate Financial Market Dynamics" signals a potential paradigm shift. If diffusion models can replace Monte Carlo simulation for certain applications, this changes core quant workflows.

### 3. The "Factor Mirage" Could Reshape Factor Investing
Lopez de Prado's causal factor investing framework exposes that many "validated" factors are causally misspecified. If this gains traction, it could invalidate large portions of current factor-based strategies.

### 4. Foundation Models Need Finance-Specific Training
Generic time-series foundation models FAIL on financial data. This means the "just use GPT/Claude for everything" approach doesn't work. Financial foundation models are a distinct and important category.

### 5. Experienced Developers Get SLOWER with AI Coding Assistants
The 19% slowdown finding contradicts the dominant narrative. The perception gap (developers believe they're faster when they're not) is particularly concerning for quant work where accuracy matters.

### 6. Sovereign Wealth Funds Are the Largest AI Investors
SWFs invested $66B in AI/digitalization in 2025. Saudi Arabia's PIF committed $36.2B. This capital allocation will reshape infrastructure availability and AI capabilities in finance.

### 7. The DeFi-TradFi Convergence Is Real
QuantMinds (a mainstream quant conference) has DeFi sessions. Tokenized assets at $33B+. BlackRock in production. This is no longer a separate ecosystem.

### 8. ICAIF Moving to Singapore Signals Asia-Pacific Quant Shift
The most important academic AI-in-finance conference moving to Asia for the first time, with record submissions, signals the center of gravity shifting.

### 9. Market Microstructure Research Is Becoming Image Processing
Converting LOB data into images and applying computer vision techniques (diffusion with inpainting) is a bizarre but apparently effective approach.

### 10. Multi-Agent RL for Market Making Is Near Production
Multiple ICAIF 2025 papers on multi-agent RL for market making suggest this is close to deployment at major firms.

---

## 10. ADDITIONAL FINDINGS THAT DON'T FIT ABOVE CATEGORIES

### Regime Detection Is Becoming Multi-Model
- Ensemble HMM + tree-based models for regime detection
- "Forest of Opinions" approach combining multiple models
- Regime-aware strategies (like DeePM) showing 2x outperformance

### The "Quant Credit" Trend
- Multiple conference sessions dedicated to quantitative credit strategies
- Quant techniques moving from equities to credit markets
- Systematic credit investing growing rapidly

### Behavioral AI / Psychology of AI
- QuantMinds 2025 featured "The Psychology of AI" and "Using behavioral psychology to design reliable AI workflows"
- This meta-topic about how humans interact with AI systems is becoming a research area

### Women in Quant Finance
- Dedicated lunch discussions at major conferences
- Diversity initiatives becoming more visible
- May affect talent pipeline and culture

### Data Vendor Landscape Shifting
- Bloomberg and Refinitiv still dominant
- But alternative data platforms growing rapidly ($273B projected market)
- $25M Series B for AI-driven event intelligence platform backed by investment banks

### The "Wellness" Signal
- QuantMinds 2025 included "Permission to Press Pause" wellness session
- Burnout and mental health in quant finance being acknowledged publicly

### GPU Cloud Spending May Be a Bubble
- Hyperscaler capex: $400B+ on AI infrastructure
- AI-related revenue: only ~$25B in 2025 (10% of spending)
- Only 25% of AI initiatives delivering expected ROI
- If AI adoption lags, this spending is unsustainable

### The "Copula Renaissance"
- QuantMinds 2025: "My Hot Shot Marvelous Copula" by Andrey Chirikhin (Schonfeld)
- Copula methods may be seeing renewed interest with modern computational tools

### Reconciling P and Q Measures
- Julien Guyon (ENPC/NYU) presenting at QuantMinds on "Reconciling P- and Q-Calibration"
- This is a fundamental theoretical problem that's getting renewed attention

---

## SUMMARY: TOP 10 AREAS TO WATCH FOR 2025-2026

1. **Diffusion Models for Finance** -- The new generative frontier, displacing GANs and potentially Monte Carlo
2. **Financial Foundation Models** -- Domain-specific FMs for time series, language, and multi-modal financial data
3. **Causal Factor Investing** -- Lopez de Prado's paradigm shift from correlation to causation
4. **Multi-Agent AI Systems** -- Specialist agents collaborating for trading, research, and market making
5. **Post-Quantum Cryptography Migration** -- Urgent infrastructure program with hard deadlines
6. **Agentic AI in Banking** -- From automation to autonomy, 15-20% cost reduction potential
7. **Synthetic Data via Diffusion** -- Regulatory-compliant data generation for training, testing, and stress scenarios
8. **Conformal Prediction** -- Distribution-free uncertainty quantification meeting regulatory demand
9. **Energy Trading Quantification** -- Explosive complexity from renewables, data centers, and nodal markets
10. **Real-World Asset Tokenization** -- TradFi-DeFi convergence at $33B+ and accelerating
