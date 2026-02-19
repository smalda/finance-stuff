# Bank, Asset Manager & Fintech/Vendor Quant Roles: Comprehensive Research

**Research Date:** 2026-02-15
**Purpose:** Understand skill/knowledge requirements for quant roles that DIFFER from buy-side hedge fund quant researcher roles, to inform course restructure.
**Method:** WebSearch only (22 queries across 5 research streams). Findings extracted from search result summaries.

---

## 1. Sell-Side Role Types

### 1.1 Strats (Goldman Sachs Model)

Goldman Sachs' Strats business unit develops quantitative and technological techniques to solve complex business problems. Strats use mathematical and scientific training to create financial products, advise clients on transactions, measure risk, and identify market opportunities.

**Sub-types:**
- **Algo / Core Strats:** Responsible for programming, infrastructure, high-volume coding, and trading platforms
- **Data Strats:** Focus on big data and machine learning
- **Desk Strats:** Work directly with trading desks on pricing, risk, and P&L attribution

**Key requirements:** First-class degree in computer science from a top university; expertise in C++, C#, Java; collaborative teamwork; ability to communicate to both technical and non-technical audiences.

**Interview topics:** Stochastic calculus, PDEs, Monte Carlo simulations, sorting algorithms, CS basics, multithreaded programming, deadlock issues.

*Sources: "Goldman Sachs strat role requirements skills 2025", "bank quant strat vs hedge fund quant researcher differences"*

### 1.2 Desk Quants / Derivatives Pricing Quants

Front-office quants who sit on or near trading desks. They build and maintain derivative pricing models that traders use to price and execute trades.

**Key requirements (JPMorgan model):**
- PhD or equivalent in Mathematics, Mathematical Finance, Sciences, Engineering, or CS
- Deep understanding of options pricing theory and quantitative models for pricing/hedging derivatives
- Excellence in probability theory, stochastic processes, PDEs, and numerical analysis
- Strong software development skills in Python, Excel VBA, C++
- Stochastic modeling (probability, stochastic processes, Ito's formula, derivative pricing)
- Numerical methods (finite difference, Monte Carlo)

**Critical distinction from buy-side:** Desk quants build pricing engines and risk tools FOR traders; they do not generate alpha or manage P&L directly. Work can become routine once models are coded. Focus is on accurate pricing and hedging, not signal discovery.

*Sources: "JPMorgan quantitative analyst derivatives pricing requirements", "sell-side quant career path derivatives vs buy-side 2025"*

### 1.3 Risk Quants

Middle-office roles focused on risk measurement, management, and reporting.

**Key requirements:**
- Derivatives pricing, risk metrics (VaR, CVaR), fixed income, and credit products
- Python and SQL are musts; R or C++ as bonuses
- Communication skills to explain complex models to non-quants
- Critical thinking to anticipate what could go wrong
- Attention to detail (a small bug in a model can cost millions)

*Sources: "sell-side quant derivatives pricing risk skills requirements 2025"*

### 1.4 Model Validation / Model Risk Management (MRM) Quants

Independent review function required by regulators. These quants validate and challenge models built by front-office quants.

**Key requirements:**
- Python, R, SQL, statistical testing, machine learning
- Knowledge of regulatory guidelines: SR 11-7, Basel, CCAR, CECL
- Optional certifications: FRM, PRM, CQF, GARP's AI in Risk
- Technical knowledge and modeling skills necessary to conduct appropriate analysis and critique
- Explicit authority to require changes to models when issues are identified
- Understanding of all possible models a bank might use

**Critical distinction:** MRM quants must be INDEPENDENT of the model builders. Their job is adversarial -- they must find flaws, challenge assumptions, and ensure regulatory compliance. This is a role that essentially does not exist on the buy-side.

*Sources: "model risk management quant SR 11-7 skills requirements"*

### 1.5 Electronic Trading / Algo Quants

Quants who develop algorithms for electronic market-making and execution across asset classes (eFX, eRates, etc.).

**Key requirements:**
- Three-pronged skill set: business/market knowledge, quant/data science skills, stellar coding
- Python, Java, C++, SQL, kdb/q
- Market microstructure knowledge
- Real-time execution capabilities
- Backtesting on historical market data
- More "strat" than "pure quant" -- these are hybrid roles

*Sources: "FX quantitative analyst electronic trading skills"*

### 1.6 XVA Quants

Specialized quants working on valuation adjustments for counterparty risk.

**Key requirements:**
- Detailed knowledge of CVA, DVA, FVA, KVA, MVA and their interactions
- Monte Carlo simulations, PFE calculations
- C++ and Python; DLL/XLL development; QuantLib
- Stochastic calculus, numerical methods, options theory
- VBA, SQL, JSON, SVN/GIT as supplementary
- Understanding of risk management and portfolio valuation techniques (VaR, sensitivities)
- Master's or PhD in Quantitative Finance or numerical subject

**Critical distinction:** XVA is a sell-side specialty that barely exists on the buy-side. It arose from post-2008 regulatory requirements for banks to account for counterparty credit risk in derivatives pricing. This is one of the most technically demanding and fastest-growing areas of sell-side quant work.

*Sources: "XVA quant analyst job description skills CVA DVA FVA 2025"*

### 1.7 Regulatory / Capital Quants (FRTB)

Quants focused on regulatory capital calculations and compliance.

**Key requirements:**
- Understanding of Basel III and FRTB rules (SA and IMA approaches)
- VaR, DRC (Default Risk Charge), Expected Shortfall
- Derivative pricing and exposure to asset classes (IR, FX, Credit, Fixed Income)
- RWA data and processing (CCR and MR RWA)
- Knowledge of SIMM, FRTB, CVA regulations
- Master's in Finance or similar; FRM, CFA, PhD, or CQF certifications
- Typically 4-8 years experience for senior roles

*Sources: "FRTB quant analyst role requirements Basel III bank"*

---

## 2. Key Skill Differences from Buy-Side Hedge Fund Roles

### 2.1 What Banks Need That Hedge Funds Don't

| Dimension | Sell-Side (Banks) | Buy-Side (Hedge Funds) |
|-----------|------------------|----------------------|
| **Primary objective** | Pricing accuracy, risk management, regulatory compliance | Alpha generation, P&L maximization |
| **Model explainability** | Banks want quants who can write easily explainable models due to regulatory constraints | "Black box" models acceptable if they generate returns |
| **Regulatory knowledge** | SR 11-7, Basel III, FRTB, CCAR, CECL, Dodd-Frank are essential | Minimal regulatory burden (mostly Dodd-Frank reporting) |
| **XVA/CVA knowledge** | Critical for derivatives businesses | Rarely relevant |
| **Model validation** | Entire independent function (MRM) required by regulators | Internal risk checks but no formal independent validation mandate |
| **Scope of work** | Narrower, more specialized ("five people doing one HF quant's job") | Broader remit per individual |
| **Career path** | Linear hierarchy: Analyst to MD, time-based promotions | Flatter, performance-based, more entrepreneurial |
| **Compensation structure** | Higher base salaries, smaller bonuses | Lower base, much larger performance bonuses |
| **Math emphasis** | Formal mathematics, favors mathematicians/physicists | More CS-oriented, actuarial science, economics also valued |
| **Client interaction** | Research distributed to clients, structuring for clients | Internal use only |
| **Infrastructure** | Large legacy systems, shared libraries, team-based codebases | More greenfield, individual ownership |

### 2.2 Specific Skill Gaps for an ML/DL Expert Moving to Sell-Side

1. **Derivatives pricing theory:** Options pricing (Black-Scholes, local vol, stochastic vol), Greeks, hedging strategies
2. **Stochastic calculus:** Ito's lemma, SDEs, Girsanov theorem, change of measure
3. **Numerical methods for PDEs:** Finite difference methods, Crank-Nicolson, ADI schemes
4. **Monte Carlo methods:** Variance reduction, path-dependent pricing, LSM for American options
5. **Fixed income mathematics:** Yield curves, bootstrapping, interest rate models (HJM, Hull-White, Vasicek, CIR)
6. **Regulatory frameworks:** Basel III capital requirements, FRTB SA vs IMA, SR 11-7 model governance
7. **XVA suite:** CVA, DVA, FVA, KVA, MVA -- computation, hedging, and capital implications
8. **C++ proficiency:** Most bank pricing libraries are in C++; Python alone is insufficient
9. **Risk metrics:** VaR, Expected Shortfall, stress testing, sensitivity analysis (Greeks)
10. **Financial product knowledge:** Swaps, swaptions, caps/floors, CDS, structured products, repos

*Sources: "bank quant vs hedge fund quant skills comparison", "bank quant strat vs hedge fund quant researcher differences", "sell-side quant career path derivatives vs buy-side 2025", "efinancialcareers bank quant analyst top skills 2025"*

---

## 3. Asset Class Specific Requirements

### 3.1 Rates / Fixed Income

- Interest rate derivatives models: Heath-Jarrow-Morton (HJM), Hull-White, Vasicek, CIR
- Bond pricing: day count conventions, yield-to-maturity, continuous vs discrete compounding
- Spot rates, forward rates, par rates, yield curve construction
- Interest rate derivatives: caps, floors, collars, swaptions using Black-76 model
- Monte Carlo simulation for path-dependent rate products
- C++ object-oriented programming for pricing libraries
- Reference texts: Brigo & Mercurio "Interest Rate Models -- Theory and Practice"

*Sources: "rates quant fixed income derivatives modeling skills"*

### 3.2 Credit / Structured Products

- Quantitative modeling of credit products: TRS, repo, CDS, callable loans
- Python and C++ essential; VBA/C# advantageous
- Financial software: Bloomberg, Excel VBA, risk management platforms
- Derivatives pricing, risk metrics (VaR, CVaR), fixed income, and credit products
- Working on trading floor, independently solving complex desk requirements
- Master's degree in Financial/Applied Mathematics, Physics, or Engineering

*Sources: "credit quant analyst structured products skills requirements"*

### 3.3 FX / Electronic Trading

- Market microstructure specific to FX markets
- Algorithmic development for real-time execution in electronic markets
- Python, Java, C++, SQL, kdb/q
- eTrading roles are hybrid "strat" roles combining quant skills with engineering
- Backtesting frameworks on historical market data
- Understanding of market-making dynamics and order book mechanics

*Sources: "FX quantitative analyst electronic trading skills"*

### 3.4 Commodities / Energy

- Advanced degree (MSc/PhD) in quantitative discipline
- Robust software development in both C++ and Python
- Prior experience with derivatives pricing models (commodities preferred, but FX/equities also valued)
- Monte Carlo methods, PDE solvers, volatility calibration techniques
- Exposure to commodities markets: energy, metals, ags, gas, power, index
- Proficiency in Python, SQL, KDB, Bloomberg BQL
- Close collaboration with traders, structurers, and risk managers

*Sources: "commodities quant analyst energy trading bank requirements"*

### 3.5 Equities (Implied from Multiple Sources)

- Equity derivatives pricing: Black-Scholes framework, local/stochastic vol, jump-diffusion
- Exotic options pricing (barriers, Asians, lookbacks, cliquets)
- Correlation modeling for basket products
- Equity quant developers maintain pricing models which traders use
- Most BB equity derivatives quants are not involved in managing the book directly

*Sources: "bank quant vs hedge fund quant skills comparison", "sell-side quant derivatives pricing risk skills requirements 2025"*

---

## 4. Regulatory & Risk Knowledge

### 4.1 XVA (X-Value Adjustments)

XVA is a family of valuation adjustments that banks must compute for OTC derivatives:

- **CVA (Credit Valuation Adjustment):** Adjustment for counterparty credit risk
- **DVA (Debit Valuation Adjustment):** Adjustment for own credit risk
- **FVA (Funding Valuation Adjustment):** Cost/benefit of funding collateralized hedges for uncollateralized OTC derivatives
- **KVA (Capital Valuation Adjustment):** Cost of regulatory capital
- **MVA (Margin Valuation Adjustment):** Cost of posting initial margin

**Technical requirements:** Monte Carlo simulations, exposure profiles (PFE, EPE, ENE), stochastic calculus, multi-factor models, C++ pricing libraries, QuantLib.

XVA quants deliver analytics to Counterparty Risk Trading desks. They conduct stress testing and scenario analysis for XVA risk exposure.

*Sources: "XVA quant analyst job description skills CVA DVA FVA 2025"*

### 4.2 FRTB (Fundamental Review of the Trading Book)

FRTB is a Basel III framework for bank trading book capital requirements:

- **Standardized Approach (SA):** Prescriptive capital calculation based on sensitivities
- **Internal Models Approach (IMA):** Banks use own models subject to regulatory approval
- **Key metrics:** Expected Shortfall (replacing VaR), Default Risk Charge (DRC)
- **Qualitative issues:** Boundary between banking book and trading book
- **Related frameworks:** SIMM (Standard Initial Margin Model)

**Skills needed:** Understanding of RWA computation, CCR (Counterparty Credit Risk) and MR (Market Risk) RWA, derivative pricing across all asset classes, strong programming and pricing library skills.

*Sources: "FRTB quant analyst role requirements Basel III bank"*

### 4.3 SR 11-7 (Model Risk Management)

Federal Reserve/OCC guidance on model risk management (issued 2011, still the governing standard):

- Requires banks to have independent model validation
- Validators need technical knowledge AND authority to require model changes
- Skills: Python, R, SQL, statistical testing, machine learning
- Knowledge of: Basel, CCAR (Comprehensive Capital Analysis and Review), CECL (Current Expected Credit Losses)
- Certifications valued: FRM, PRM, CQF, GARP's AI in Risk

**Key principle:** The regulation emphasizes that effective model risk management requires professionals with strong technical competencies combined with knowledge of regulatory frameworks and the authority to implement necessary changes.

*Sources: "model risk management quant SR 11-7 skills requirements"*

### 4.4 Additional Regulatory Knowledge

- **Basel III:** Global regulatory framework for bank capital requirements and risk management
- **Dodd-Frank:** US financial reform legislation
- **Stress testing:** CCAR, DFAST (Dodd-Frank Act Stress Test)
- **Risk-weighted assets (RWAs):** Capital adequacy calculations
- **Value at Risk (VaR):** Still widely used despite ES adoption under FRTB

*Sources: "regulatory quant capital markets skills 2025"*

---

## 5. Fintech & Data Vendor Roles

### 5.1 BlackRock (Aladdin Platform)

BlackRock's Aladdin is the dominant institutional risk and portfolio management platform.

**Quantitative Research Analyst (Aladdin Financial Engineering):**
- B.Tech/B.E. or M.Sc. in quantitative discipline (Math, Physics, CS, Finance)
- Strong Mathematics, Statistics, Probability, Linear Algebra
- Data mining, data analytics, data modeling
- Time series forecasting, clustering, statistical and ML approaches
- SQL, Python, R, JavaScript (citizen developer); C++, Java, Scala (OO languages)
- Knowledge of fixed income, equities, credit instruments is a plus
- Market liquidity knowledge is a plus but not required

**Key distinction from pure sell-side:** Aladdin quants build analytics consumed by thousands of institutional clients. The platform orientation means more emphasis on scalability, productionization, and user-facing design than typical bank quant roles.

*Sources: "BlackRock Aladdin quantitative analyst requirements 2025"*

### 5.2 MSCI (Risk Models & Factor Models)

MSCI is the dominant provider of equity factor models and risk analytics used globally.

**Quantitative Researcher requirements:**
- Exceptional quantitative aptitude and critical thinking
- Strong mathematics, linear algebra, statistics
- Python proficiency and scientific methodologies
- Both traditional statistics AND cutting-edge AI techniques
- Advanced degree in financial engineering, economics, mathematics, data science
- 7+ years experience for senior roles
- Full lifecycle of valuation and risk model development: research, design, model fitting, testing
- Excellent communication skills for presenting technical concepts to non-technical stakeholders

**Key distinction:** MSCI quants build models that become INDUSTRY STANDARDS (e.g., Barra risk models). Much more focus on factor modeling, cross-sectional analysis, and risk decomposition than typical bank roles.

*Sources: "MSCI quantitative researcher risk models requirements"*

### 5.3 Bloomberg (Quantitative Analytics)

Bloomberg's Quantitative Analytics team builds modeling analytics across the entire Bloomberg ecosystem.

**Scope of work:**
- Pricing and risk of derivative products across ALL major asset classes
- Market data models
- Counterparty credit risk, XVA, initial margin
- VaR and other market risk metrics
- Climate risk, credit risk, liquidity risk

**Requirements:**
- PhD in quantitative field (Math, Physics, Engineering, Quant Finance)
- 4+ years at VP level or above at a Market Risk modelling team (buy-side or sell-side) or equivalent at a vendor
- Modern C++ and Python libraries
- Novel research AND efficient model delivery

**Tools:** BQuant Desktop -- cloud-based investment research platform for building, testing, and deploying quantitative models.

**Compensation:** $155,000 - $285,000 USD annually plus benefits and bonus.

**Key distinction:** Bloomberg quants must understand the FULL breadth of financial products and risk types, since the terminal serves 300,000+ clients across every segment. Breadth over depth compared to bank desk quants.

*Sources: "Bloomberg quantitative analyst financial engineering role"*

### 5.4 Vanguard (Quantitative Equity Group)

Vanguard's QEG manages systematic equity strategies within a primarily passive asset management firm.

**Approach:**
- Systematic, data-driven investing process
- AI and machine learning for market-beating results
- Fundamentally-based stock selection with quantitative overlay
- Human judgment integrated with quantitative methods (augmentation, not replacement)
- Technology, investment expertise, risk specialists, global footprint
- Focus on keeping costs low

**Key distinction:** Vanguard quants work within an asset manager that is primarily passive/index-oriented. The emphasis is on factor-based equity selection, risk management, and cost efficiency rather than the derivatives-heavy focus of bank quants.

*Sources: "Vanguard quantitative analyst systematic strategies"*

---

## 6. Technology Stack

### 6.1 Programming Languages (by Priority for Sell-Side)

| Language | Use Case | Prevalence |
|----------|----------|------------|
| **C++** | Pricing libraries, low-latency trading, core analytics | Essential for desk quants, XVA, derivatives |
| **Python** | Prototyping, data analysis, scripting, ML, Pandas/NumPy | Universal across all quant roles |
| **Java** | Trading systems, server-side development (Morgan Stanley) | Common for e-trading, infrastructure |
| **SQL** | Database management, regulatory reporting | Required nearly everywhere |
| **R** | Statistical analysis, model validation | Common in MRM and risk |
| **kdb/q** | Time-series databases for tick data | FX, e-trading, commodities |
| **VBA** | Excel-based tools (legacy but persistent) | Still widespread on trading floors |
| **C#** | .NET-based trading systems | Some banks (especially for desktop tools) |
| **MATLAB** | Prototyping, academic-style research | Declining but still used in some risk teams |
| **JavaScript** | Web-based dashboards, Aladdin ecosystem | BlackRock, fintech |

### 6.2 Frameworks & Libraries

- **QuantLib:** Open-source library for quantitative finance (C++/Python)
- **Pandas/NumPy/SciPy:** Python scientific computing stack
- **Bloomberg BQL:** Bloomberg Query Language for data extraction
- **Excel/VBA:** Still the lingua franca of trading floors for ad hoc analysis
- **Hadoop/Spark:** Big data processing for large-scale risk calculations
- **Git/SVN:** Version control (increasingly Git)

### 6.3 Platforms

- **Bloomberg Terminal:** Market data, analytics, trading (300,000+ users)
- **BlackRock Aladdin:** Risk and portfolio management platform
- **Murex:** Trading, risk management, and processing platform for derivatives
- **Calypso:** Cross-asset trading and risk management
- **Summit/Finastra:** Treasury and capital markets platforms

*Sources: "Goldman Sachs strat role requirements skills 2025", "XVA quant analyst job description skills CVA DVA FVA 2025", "Morgan Stanley quantitative developer fixed income skills", "commodities quant analyst energy trading bank requirements"*

---

## 7. Career Paths & Compensation

### 7.1 Sell-Side Career Ladder

Standard bank hierarchy applies:
- **Analyst** (entry level, 0-3 years)
- **Associate / VP** (3-7 years)
- **Senior VP / Director** (7-12 years)
- **Managing Director** (12+ years)

Promotions are more time-based than performance-based. The furthest you can make it as an individual contributor is usually VP; senior roles require team management.

### 7.2 Compensation Benchmarks (from Search Results)

| Role | Base Salary (USD) | Notes |
|------|-------------------|-------|
| GS FICC Mortgage Strat (entry) | $110,000 - $125,000 | Base only |
| MS Quantitative Developer | $150,000 - $250,000 | Total comp $161K-$232K avg |
| Bloomberg Quant Analyst | $155,000 - $285,000 | Plus benefits and bonus |
| Bank quant (entry level) | $100,000 - $150,000 | Per eFinancialCareers |
| Hedge fund quant (entry level) | $200,000 - $300,000 | Significantly higher at start |
| Fintech quant analyst | $120,000 - $180,000 | Per industry surveys |

### 7.3 Key Compensation Dynamics

- Hedge fund researchers receive significantly larger bonuses; sell-side quants receive significantly larger (more stable) base salaries
- The hedge fund space does not view sell-side experience as highly valuable since incentives are entirely different, making bank-to-HF transition difficult
- Senior buy-side compensation can reach millions (with bonus); sell-side MD compensation typically caps lower
- Bank quant compensation is more predictable and less volatile than buy-side

### 7.4 Common Transitions

- Sell-side desk quant --> Buy-side (difficult, different skill emphasis)
- Sell-side quant --> Fintech/vendor (common, well-valued)
- Model validation --> Front-office quant (within bank)
- PhD graduate --> Bank strat/quant (most common entry point)
- Bank quant --> Regulatory/consulting (natural path for risk-focused quants)

*Sources: "bank quant vs hedge fund quant skills comparison", "sell-side quant career path derivatives vs buy-side 2025", "Morgan Stanley quantitative developer fixed income skills", "Bloomberg quantitative analyst financial engineering role"*

---

## 8. Key Findings for Course Design

### 8.1 What an ML/DL Expert Would Need to Learn for These Roles

An ML/deep learning expert already possesses Python fluency, statistical modeling, data pipeline skills, and optimization techniques. The following areas represent the GAPS they would need to fill to be competitive for sell-side, asset manager, and vendor quant roles:

#### Priority 1: Essential Foundations (Must Learn)

1. **Derivatives pricing theory**
   - Black-Scholes-Merton framework, risk-neutral pricing, no-arbitrage principle
   - Greeks (delta, gamma, vega, theta, rho) and hedging strategies
   - Local volatility, stochastic volatility (Heston, SABR), jump-diffusion models
   - Exotic options: barriers, Asians, lookbacks, cliquets, autocallables

2. **Stochastic calculus**
   - Brownian motion, Ito's lemma, stochastic differential equations
   - Girsanov theorem, change of measure, risk-neutral measure
   - Martingale theory and its application to pricing

3. **Numerical methods (beyond standard ML)**
   - Monte Carlo methods: variance reduction, path-dependent pricing, Longstaff-Schwartz
   - PDE methods: finite difference (explicit, implicit, Crank-Nicolson), ADI schemes
   - Tree methods: binomial, trinomial for American options

4. **Fixed income mathematics**
   - Yield curve construction and bootstrapping
   - Interest rate models: Vasicek, CIR, Hull-White, HJM, LMM/BGM
   - Bond math: duration, convexity, day count conventions
   - Interest rate derivatives: swaps, swaptions, caps/floors (Black-76)

5. **C++ programming**
   - Object-oriented design for pricing libraries
   - Performance optimization, memory management
   - Template programming, design patterns for finance
   - Integration with Python (pybind11, SWIG)

#### Priority 2: Role-Differentiating Knowledge

6. **XVA framework**
   - CVA, DVA, FVA, KVA, MVA computation
   - Exposure simulation (PFE, EPE, ENE)
   - Collateral modeling and CSA terms
   - XVA hedging and capital implications

7. **Regulatory frameworks**
   - Basel III capital requirements (CET1, Tier 1, Total Capital)
   - FRTB: SA vs IMA, Expected Shortfall, DRC, RRAO
   - SR 11-7: model governance, validation, documentation standards
   - CCAR/DFAST stress testing
   - SIMM for initial margin

8. **Risk metrics and management**
   - VaR (historical, parametric, Monte Carlo)
   - Expected Shortfall / CVaR
   - Stress testing and scenario analysis
   - Sensitivity analysis and Greek computation at portfolio level

9. **Credit modeling**
   - Credit default swaps (CDS) pricing
   - Structural models (Merton), reduced-form models (Jarrow-Turnbull, Duffie-Singleton)
   - Correlation: copulas, CDO tranche pricing concepts
   - Credit migration and default probability curves

#### Priority 3: Valuable Additions

10. **Financial product knowledge**
    - Swaps (IRS, CCS, TRS), repos, structured notes
    - Securitized products: MBS, ABS, CDO
    - Commodity derivatives: forwards, futures, options, swing options
    - FX derivatives: forwards, options, barriers, TARFs

11. **Market microstructure**
    - Order book dynamics, bid-ask spread modeling
    - Market making algorithms
    - Transaction cost analysis
    - Electronic trading platform design

12. **Model validation methodology**
    - Backtesting frameworks for risk models
    - Statistical tests for model adequacy
    - Documentation standards for regulatory submission
    - Independent replication and benchmarking

### 8.2 Where ML/DL Skills ARE Valued in These Roles

The ML/DL background is not wasted -- it provides competitive advantage in several areas:

- **Data Strats roles** at Goldman Sachs and equivalents explicitly seek ML expertise
- **Model Risk Management** increasingly uses ML techniques and needs people who can validate ML models under SR 11-7
- **BlackRock Aladdin** seeks "cutting-edge AI techniques" alongside traditional statistics
- **MSCI** explicitly wants "both traditional statistics and cutting-edge AI techniques"
- **Electronic trading / algo quants** use ML for signal generation, execution optimization
- **Credit risk modeling** increasingly uses ML for default prediction (e.g., Vanguard's AI for dividend cut forecasting)
- **Regulatory quants** need to understand ML models to validate them -- banks want "easily explainable models" -- creating demand for interpretable ML expertise

### 8.3 Course Module Recommendations

Based on this research, a course for ML/DL practitioners targeting sell-side/AM/vendor roles should include:

| Module | Rationale |
|--------|-----------|
| **Derivatives Pricing Fundamentals** | Core knowledge gap; required for 80%+ of sell-side roles |
| **Stochastic Calculus for Finance** | Mathematical foundation for all pricing; interview essential |
| **Numerical Methods (MC, PDE, Trees)** | Implementation skills for pricing libraries |
| **Fixed Income & Rates** | Largest asset class by notional; most bank quant demand |
| **C++ for Quantitative Finance** | Bank pricing libraries are C++; Python alone insufficient |
| **XVA & Counterparty Risk** | Fastest-growing sell-side specialty; unique to banks |
| **Regulatory Frameworks (Basel/FRTB/SR 11-7)** | Critical differentiator from buy-side; no equivalent in HF world |
| **Credit Derivatives & Structured Products** | Required for credit desks and structured products groups |
| **Risk Metrics & Model Validation** | VaR, ES, stress testing; opens MRM career path |
| **ML in Finance: Interpretability & Regulation** | Leverages existing ML skills while addressing bank-specific explainability requirements |

### 8.4 Key Takeaway

The fundamental difference is this: **buy-side quants optimize for prediction accuracy (alpha), while sell-side quants optimize for pricing accuracy, risk measurement, and regulatory compliance.** An ML/DL expert has the statistical and computational foundation but lacks the financial mathematics, product knowledge, and regulatory awareness that sell-side roles demand. The course should bridge this gap while highlighting where modern ML techniques can be applied within the sell-side framework -- particularly in model validation, e-trading, and the emerging "AI in Risk" regulatory space.

---

## Appendix: Search Queries Used

### Stream 1: Sell-Side Derivatives/Strat Roles
- "Goldman Sachs strat role requirements skills 2025"
- "JPMorgan quantitative analyst derivatives pricing requirements"
- "Morgan Stanley quantitative developer fixed income skills"
- "sell-side quant derivatives pricing risk skills requirements 2025"
- "bank quant strat vs hedge fund quant researcher differences"

### Stream 2: XVA, FRTB, Regulatory Quant
- "XVA quant analyst job description skills CVA DVA FVA 2025"
- "FRTB quant analyst role requirements Basel III bank"
- "model risk management quant SR 11-7 skills requirements"
- "regulatory quant capital markets skills 2025"

### Stream 3: Asset Managers & Fintech
- "BlackRock Aladdin quantitative analyst requirements 2025"
- "MSCI quantitative researcher risk models requirements"
- "Bloomberg quantitative analyst financial engineering role"
- "Vanguard quantitative analyst systematic strategies"
- "fintech quantitative analyst role requirements 2025"

### Stream 4: Aggregator Views
- "efinancialcareers bank quant analyst top skills 2025"
- "sell-side quant career path derivatives vs buy-side 2025"
- "bank quant vs hedge fund quant skills comparison"
- "most in-demand quant skills banking 2025"

### Stream 5: Credit, Rates, FX, Commodities Specific
- "credit quant analyst structured products skills requirements"
- "rates quant fixed income derivatives modeling skills"
- "FX quantitative analyst electronic trading skills"
- "commodities quant analyst energy trading bank requirements"
