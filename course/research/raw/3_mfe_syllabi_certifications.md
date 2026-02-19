# Quantitative Finance Educational Programs: Comprehensive Curriculum Research
## Phase 1 — Open Discovery (Research Date: February 2026)

---

## 1. SUMMARY OF PROGRAMS REVIEWED

### Degree Programs (13 programs)
| # | Program | Duration | Credits/Units | Format |
|---|---------|----------|---------------|--------|
| 1 | CMU MSCF | 2 years (4 mini-semesters + summer) | 25 courses | Full-time |
| 2 | Princeton MFin | 2 years (4 semesters) | 5 core + 11 electives | Full-time |
| 3 | Baruch MFE | 3 semesters (FT) / 5-6 (PT) | 36 credits (12 courses) | FT/PT evening |
| 4 | NYU MFE (Courant) | 3 semesters (FT) | 36 points (9 core + electives) | FT/PT |
| 5 | Columbia MSFE (IEOR) | 3-4 semesters | 36 points (12 courses) | Full-time |
| 6 | Stanford MCF (ICME) | ~1.5 years | 45 units across 5 categories | Full-time |
| 7 | MIT MFin (Sloan) | 12-18 months | Core + restricted electives + action learning | Full-time |
| 8 | Oxford MCF | 10 months (3 terms) | Core + 4 electives + dissertation | Full-time |
| 9 | Imperial MSc Math & Finance | 12 months | 7 core + 5 electives + project | Full-time |
| 10 | ETH Zurich MSc Quant Finance | 2 years | 36 core + 24 elective + 30 thesis credits | Full-time |
| 11 | UCL Financial Mathematics | 12 months | 4 core + 4 optional + thesis | Full-time |
| 12 | UC Berkeley MFE (Haas) | 12 months | 28 units + internship | Full-time |
| 13 | Cornell MFE (ORIE) | 3 semesters | Flexible credit model | Full-time |

### Professional Certifications (4 programs)
| # | Program | Structure |
|---|---------|-----------|
| 1 | CQF (Certificate in Quantitative Finance) | 6 core modules + 2 advanced electives + final project |
| 2 | FRM (Financial Risk Manager) | Part I (100 Q) + Part II (80 Q) |
| 3 | CFA Level III | Portfolio management focus; quant methods integrated throughout |
| 4 | CAIA | Level I + Level II; includes quantitative methods for alternatives |

### Online/Additional Programs
| # | Program | Provider |
|---|---------|----------|
| 1 | WQU MScFE | WorldQuant University (free, online) |
| 2 | ML & Reinforcement Learning in Finance | Coursera / NYU |
| 3 | Investment Mgmt with Python & ML | Coursera / EDHEC |
| 4 | ML for Trading | Coursera / Google Cloud + NYIF |
| 5 | Quant Finance & Risk Modeling | Coursera specialization |
| 6 | Applied Quant Finance & ML | Harvard Extension |

---

## 2. UNIVERSAL TOPICS (Appear in 80%+ of Programs)

These topics appear across virtually ALL degree programs surveyed. They represent the undisputed academic consensus on what a quant needs to know.

### 2.1 Stochastic Calculus / Stochastic Processes
- **Appears in:** CMU, Princeton, Baruch, NYU, Columbia, Stanford, MIT (as "Financial Mathematics"), Oxford, Imperial, ETH, UCL, Berkeley, Cornell
- **Coverage:** Brownian motion, Ito calculus, SDEs, martingale theory, risk-neutral pricing
- **Verdict:** YES, stochastic calculus is STILL universally required. Every single program teaches it as a core/required course.
- **Note:** Programs teach both the derivation AND application of Black-Scholes. The derivation via Ito's lemma and delta hedging is explicitly covered at CMU, NYU, Oxford, Berkeley, and others. This is not just "applied" -- the mathematical foundations remain core.

### 2.2 Derivatives Pricing & Options Theory
- **Appears in:** All 13 programs
- **Coverage:** Black-Scholes model, binomial/trinomial trees, risk-neutral valuation, Greeks, exotic options, hedging strategies
- **CMU:** Separate "Options" course (46973) + Simulation Methods for Option Pricing (46932)
- **NYU:** "Financial Securities and Markets" covers BS formula, binomial trees, arbitrage
- **Oxford:** "Financial Derivatives" as a core Michaelmas term course
- **Berkeley:** "Investments and Derivatives" (MFE 230A) + "Derivatives: Quantitative Methods" (MFE 230D)

### 2.3 Fixed Income / Interest Rate Models
- **Appears in:** CMU, Princeton, NYU, Columbia, Oxford, UCL, Berkeley, Baruch, Imperial, ETH
- **Coverage:** Bond mathematics, term structure models, yield curves, interest rate derivatives, MBS, STRIPS, callable bonds
- **CMU:** Dedicated "Fixed Income" course (46956)
- **NYU:** "Fixed Income: Bonds, Securitized Products, and Derivatives" (elective but widely taken)
- **Oxford:** Core "Fixed Income and Credit" in Hilary term
- **Berkeley:** "Fixed Income Markets" (MFE 230I) — 3-unit core course

### 2.4 Portfolio Theory & Risk Management
- **Appears in:** All 13 programs
- **Coverage:** Markowitz optimization, CAPM, factor models, VaR, risk budgeting, performance attribution
- **CMU:** "MSCF Investments" (46972) — risk measurement, portfolio optimization, asset pricing
- **NYU:** "Risk & Portfolio Management" — risk factors, econometrics, extreme-value theory, VaR
- **Oxford:** Core "Quantitative Risk Management" in Hilary term
- **Berkeley:** "Financial Risk Measurement and Management" (MFE 230H)

### 2.5 Statistical Methods / Econometrics
- **Appears in:** CMU, Princeton, NYU, Columbia, Stanford, MIT, Oxford, UCL, Berkeley, ETH, Cornell
- **Coverage:** Regression, MLE, GMM, hypothesis testing, time series (AR, MA, ARMA, ARIMA, GARCH), multivariate statistics
- **CMU:** "Financial Data Science I" (46921) + "Financial Time Series Analysis" (46929)
- **NYU:** "Machine Learning & Computational Statistics" as Core Level 1
- **Columbia:** "Statistical Analysis and Time Series" (IEOR E4709) — core required
- **Princeton:** "FIN 505/ORF 505: Statistical Analysis of Financial Data" — heavy tail distributions, copulas, regression

### 2.6 Programming / Computational Methods
- **Appears in:** All 13 programs
- **Languages taught:**
  - **Python:** CMU, Princeton, NYU, Oxford, Imperial, Berkeley, MIT, Stanford, Cornell, UCL — UNIVERSAL
  - **C++:** CMU, Oxford (dedicated C++ 1 and C++ 2 courses), Imperial, Baruch, CQF — still widely taught but not universal
  - **R:** Princeton (FIN 505 uses R environment), MIT, some electives at others
- **CMU:** Three-course sequence: Financial Computing I (Python), II (C++), III (Python/NumPy for trading)
- **NYU:** "Computing in Finance" — Python, software development concepts
- **Oxford:** Dedicated "Financial Computing with C++ 1" and "C++ 2" courses
- **MIT:** "Programming for Finance Professionals" — Python/R (pass/fail)

### 2.7 Monte Carlo Simulation / Numerical Methods
- **Appears in:** CMU, NYU, Columbia, Oxford, Berkeley, Baruch, Imperial, ETH, UCL, Stanford
- **Coverage:** Random number generation, variance reduction (antithetic variables, importance sampling, control variates), finite difference methods, PDE solvers, FFT
- **CMU:** "Simulation Methods for Option Pricing" (46932)
- **Columbia:** "Monte Carlo Simulation Methods" (IEOR E4703) — core required
- **NYU:** "Scientific Computing in Finance" — IEEE arithmetic, numerical linear algebra, optimization, Monte Carlo
- **Oxford:** "Numerical Methods" (core) + "Advanced Monte Carlo Methods" (elective)

---

## 3. COMMON TOPICS (Appear in 50-80% of Programs)

### 3.1 Machine Learning (Supervised/Unsupervised)
- **Appears in:** CMU (core), Princeton (core), NYU (core), Columbia (elective concentration), Stanford, Oxford (core - Deep Learning), Imperial (elective), Berkeley (core), MIT (restricted elective), CQF (core modules 4-5)
- **Specific ML topics taught:**
  - **Regression/Classification:** Linear/logistic regression, ridge/lasso, SVM, k-NN — nearly universal
  - **Ensemble methods:** Random forests, boosting, bagging — CMU, NYU, Columbia, CQF
  - **Clustering:** K-means, hierarchical — CMU, NYU, CQF
  - **Dimensionality reduction:** PCA, factor models — most programs
  - **Neural networks / Deep learning:** CMU (ML II), Oxford (core Deep Learning course), Berkeley (Deep Learning for Finance I & II), Imperial, CQF (Module 5)
  - **NLP / Text analysis:** CMU (ML II), NYU (Trends in Financial Data Science), Berkeley (via Deep Learning II - generative AI), CQF (Module 5), Duke
  - **Reinforcement learning:** Stanford (CME 241 — dedicated RL for finance course), CMU (ML II), Berkeley (Deep Learning for Finance I), CQF (Module 5)
- **KEY FINDING:** ML has become CORE at the top programs. At CMU, it is a mandatory 2-course sequence. At NYU, "Machine Learning & Computational Statistics" is Core Level 1. At Oxford, "Deep Learning" is a CORE Hilary term course. This is a significant shift from even 3-4 years ago.

### 3.2 Time Series Analysis
- **Appears in:** CMU (core), NYU (elective), Columbia (core), Princeton, Berkeley, UCL, ETH
- **Coverage:** ARMA/ARIMA, GARCH/ARCH, cointegration, VAR models, regime switching
- **NYU elective:** "Time Series Analysis & Statistical Arbitrage" — covers pairs trading and algorithmic strategies

### 3.3 Optimization Methods
- **Appears in:** Columbia (core — IEOR E4007 "Optimization Models and Methods for FE"), Cornell (core — Optimization Modeling), Stanford, Princeton, Baruch
- **Coverage:** Linear/quadratic programming, convex optimization, dynamic programming, stochastic optimization

### 3.4 Credit Risk / Credit Derivatives
- **Appears in:** Oxford (core — Fixed Income and Credit), NYU (elective — Sell-Side Modeling: XVA, Capital and Credit Derivatives), Columbia, Baruch (MTH 9876 Credit Risk Models), CQF (Module 6), UCL
- **Coverage:** Structural models (Merton), reduced form models, CDS, CDOs, CVA/DVA/FVA, copula models

### 3.5 Stochastic Control / Dynamic Programming
- **Appears in:** Oxford (core), Princeton, Stanford (CME 241), CMU (implicit in computing courses)
- **Coverage:** Hamilton-Jacobi-Bellman equations, optimal stopping, Markov decision processes

### 3.6 Capstone/Practicum Projects
- **Appears in:** CMU, NYU, Columbia, Berkeley, Cornell, MIT, Oxford (dissertation), Imperial (research project), UCL (thesis), ETH (master's thesis)
- **Common formats:**
  - Industry-sponsored projects (Cornell, MIT Finance Lab)
  - Group research with presentation (CMU, NYU)
  - Independent dissertation/thesis (Oxford, Imperial, UCL, ETH)
  - Applied Finance Project (Berkeley MFE 230O)

---

## 4. OCCASIONAL TOPICS (Appear in 20-50% of Programs)

### 4.1 Market Microstructure / Algorithmic Trading
- **Appears in:** NYU (elective), Oxford (elective), Berkeley (elective — High Frequency Finance), Imperial (elective stream), UCL (elective), CMU (Financial Computing III covers trading signals)
- **Coverage:** Limit order books, market making, electronic trading, transaction costs, optimal execution

### 4.2 Volatility Modeling (Advanced)
- **Appears in:** Oxford (elective — Advanced Volatility Modelling), Baruch (MTH 9863 Volatility Filtering and Estimation), CQF (elective), NYU (implied)
- **Coverage:** Stochastic volatility (Heston), local volatility, SABR, rough volatility, volatility surfaces

### 4.3 Asset-Backed Securities / Securitization
- **Appears in:** CMU (Fixed Income covers MBS), NYU (Fixed Income elective), Berkeley (elective — Asset Backed Security Markets)
- **Coverage:** MBS valuation, prepayment models, structured credit, CDOs

### 4.4 Corporate Finance / Accounting
- **Appears in:** MIT (core — Corporate Finance + Corporate Financial Accounting), Columbia (elective IEOR E4402), Princeton (economics core)
- **Note:** Most "pure" MFE programs do NOT require corporate finance. MIT MFin is the exception because it serves a broader finance audience, not just quants.

### 4.5 Behavioral Finance
- **Appears in:** Berkeley (elective), CQF (elective), MIT (implicit in some courses)
- **Coverage:** Cognitive biases, prospect theory, limits to arbitrage, sentiment-based strategies

### 4.6 FX / Currency Markets
- **Appears in:** Berkeley (elective — Currency Markets), NYU (elective — Interest Rate & FX Models), CQF (Module 3 — Equities & Currencies, elective — FX Trading and Hedging)

### 4.7 Commodities / Energy
- **Appears in:** Baruch (MTH 9865 Commodities and Futures Trading), CQF (elective — Energy Trading)

---

## 5. RARE / EMERGING TOPICS (Appear in <20% of Programs)

### 5.1 Decentralized Finance (DeFi) / Blockchain / Crypto
- **Appears in:** NYU (elective — Alternative Data, Cryptocurrencies & Blockchains), Oxford (elective — Decentralised Finance), Berkeley (elective — Decentralized Finance), CQF (elective — Decentralized Finance Technologies), Duke (Crypto Trading mini-course)
- **Status:** Growing rapidly. Oxford added DeFi as an elective. NYU covers Ethereum, Bitcoin mechanics, DeFi protocols. Berkeley has a dedicated DeFi course.

### 5.2 Generative AI / Large Language Models
- **Appears in:** CQF (new elective — "Generative AI and Large Language Models for Quant Finance"), CQF (new elective — "Generative AI Agents in Finance & Beyond"), Berkeley (Deep Learning for Finance II covers generative AI), Duke (LLM-focused course), MIT ("AI and Money" course)
- **Status:** VERY new. Most 2025-2026 additions. This is the bleeding edge of curriculum development.

### 5.3 Quantum Computing for Finance
- **Appears in:** Imperial (elective — Quantum Computing), CQF (elective — Quantum Computing in Finance)
- **Coverage:** Quantum circuits, quantum algorithms for option pricing, portfolio optimization
- **Status:** Very early stage. Only 2 programs offer it.

### 5.4 Climate / ESG Finance
- **Appears in:** MIT (Climate and Social Impact Investing concentration), Columbia (IEOR E4723 — FE for ESG Finance), UCL (elective — Mathematical Climate Finance)
- **Status:** Growing but still rare in pure quant programs. More common in broader finance programs.

### 5.5 Rough Volatility
- **Appears in:** CQF (Advanced Volatility Modeling covers rough path theory), Oxford (research papers but not yet a dedicated course)
- **Status:** Still primarily a research topic, not widely in curricula.

### 5.6 XVA (CVA/DVA/FVA/KVA/MVA)
- **Appears in:** NYU (elective — Trends in Sell-Side Modeling: XVA, Capital and Credit Derivatives), CQF (elective — Counterparty Credit Risk Modeling)
- **Status:** Important in industry but taught as an advanced/elective topic

### 5.7 Financial Crime Detection / RegTech
- **Appears in:** Duke (Spring 2026 — Data Science for Financial Crime Detection)
- **Status:** Very new addition

---

## 6. ML-SPECIFIC CONTENT ACROSS PROGRAMS (Detailed Breakdown)

### ML Content Summary Matrix

| Program | ML Status | Specific ML Courses | DL? | NLP? | RL? |
|---------|-----------|-------------------|-----|------|-----|
| CMU MSCF | **CORE (2 courses)** | ML I (regression, classification), ML II (boosting, clustering, NLP, RL, neural nets) | Yes | Yes | Yes |
| Princeton MFin | **CORE** | FIN 505 covers SVM, CNNs, RNNs; optional ML certificate with 3 additional ML courses | Yes | Partial | Partial |
| Baruch MFE | Elective | MTH 9898 (Big Data), MTH 9899 (ML) | Limited | No | No |
| NYU Courant | **CORE** | "ML & Computational Statistics" (Core L1); "Trends in Financial Data Science" (elective) | Partial | Partial | No |
| Columbia MSFE | Elective (concentration) | IEOR E4525 (ML for FE & OR) — regression, classification, SVM, deep learning | Yes | No | Yes (research) |
| Stanford MCF | Elective | CME 241 (RL for Finance — dedicated); other ML courses via ICME | Partial | No | **Yes (core)** |
| MIT MFin | Restricted elective | "Advanced Data Analytics and ML in Finance", "AI and ML Research in Finance", "Modeling with ML: FinTech", "AI and Money" | Yes | Implied | Implied |
| Oxford MCF | **CORE** | "Deep Learning" is core Hilary term; uses PyTorch | **Yes (core)** | Partial | No |
| Imperial | Elective stream | ML in Finance stream — neural nets, deep learning for computational/statistical problems | Yes | Partial | No |
| ETH Zurich | Core + Elective | "Machine Learning for Finance" listed as core | Partial | No | No |
| UCL | Limited | No dedicated ML course; "Statistical Methods and Data Analytics" covers some | No | No | No |
| Berkeley MFE | **CORE + electives** | "Financial Data Science" (core); "Deep Learning for Finance I" (RL); "Deep Learning for Finance II" (unsupervised + generative AI) | **Yes** | **Yes (GenAI)** | **Yes** |
| Cornell MFE | Elective | ORIE 5256/5257 (ML tools, algorithmic trading with ML) | Partial | No | No |

### ML Growth Assessment (2022-2026)
- **2022-2023:** ML was primarily an ELECTIVE at most programs
- **2024:** Oxford made Deep Learning a CORE course; CMU had ML as core already
- **2025-2026:** NYU's core curriculum includes ML; Berkeley added Deep Learning I & II as electives; CQF added GenAI/LLM elective; Princeton's core FIN 505 now covers neural networks

**Conclusion:** ML content has grown DRAMATICALLY. It has moved from "nice to have elective" to "core requirement" at 5-6 of the top 13 programs in just 2-3 years. Deep learning specifically (not just classical ML) is now core at Oxford and covered extensively at CMU, Berkeley, Princeton, and Imperial.

---

## 7. PROGRESSION PATTERNS (What Is Taught When)

### Typical Curriculum Arc

**Phase 1: Foundations (Semester/Term 1)**
1. Probability Theory & Stochastic Processes
2. Stochastic Calculus (Ito calculus, Brownian motion, SDEs)
3. Basic Programming (Python, sometimes C++)
4. Financial Markets Overview (instruments, no-arbitrage, basic pricing)
5. Statistics / Econometrics Foundations
6. Portfolio Theory (Markowitz, CAPM)

**Phase 2: Core Quantitative Methods (Semester/Term 2)**
1. Derivatives Pricing (Black-Scholes, continuous-time models)
2. Numerical Methods (Monte Carlo, finite differences, PDE methods)
3. Machine Learning I (supervised learning, regression, classification)
4. Fixed Income (bonds, term structure, interest rate models)
5. Advanced Statistics / Time Series (GARCH, cointegration)

**Phase 3: Advanced/Specialized (Semester/Term 3+)**
1. Machine Learning II / Deep Learning (neural nets, NLP, RL)
2. Advanced Risk Management (credit risk, operational risk, XVA)
3. Market Microstructure / Algorithmic Trading
4. Specialized electives (volatility modeling, commodities, FX, DeFi)
5. Capstone Project / Thesis

**Phase 4: Applied/Practicum**
1. Industry internship (summer between years at CMU, Princeton; embedded at Berkeley, Cornell)
2. Capstone/dissertation project
3. Practitioner lecture series

### Program-Specific Progressions

**CMU MSCF (Mini-Semester System):**
- Mini 1: Stochastic Calculus I, Investments, Financial Computing I (Python), Financial Data Science I, Business Communication I
- Mini 2: Stochastic Calculus II, Options, Financial Computing II (C++), Financial Data Science II, ML I
- Mini 3: Simulation Methods, ML II, Time Series Analysis, Practitioner Lectures, Presentations
- Mini 4: Fixed Income, Financial Computing III (Trading), Financial Products & Markets
- Year 2: 7 electives of student's choosing

**NYU Courant:**
- Core Level 1 (Semester 1): Computing in Finance, Financial Securities & Markets, Risk & Portfolio Management, Stochastic Calculus & Dynamic Asset Pricing, ML & Computational Statistics
- Core Level 2 (Semester 2): Scientific Computing in Finance, Continuous Time Finance, Project & Presentation
- Semester 2-3: Electives

**Oxford MCF:**
- Pre-Michaelmas: Intro to PDEs, Probability, Statistics, Python, Financial Markets
- Michaelmas (Term 1): Stochastic Calculus, Financial Derivatives, Numerical Methods, Statistics & Data Analysis, C++ I
- Hilary (Term 2): Fixed Income & Credit, Stochastic Control, Quantitative Risk Management, Deep Learning, + 4 electives, C++ II
- Trinity (Term 3): Dissertation

**Berkeley MFE:**
- Term 1 (Mar-May): Investments & Derivatives, Empirical Methods, Stochastic Calculus, Career Prep
- Term 2 (Jun-Jul): Derivatives Quantitative Methods, Fixed Income, Financial Data Science
- Summer: Internship (10-12 weeks)
- Term 3 (Aug-Oct): Risk Management + Electives
- Term 4 (Jan-Mar): Applied Finance Project + Electives

---

## 8. HOW CERTIFICATIONS COMPARE TO DEGREE PROGRAMS

### CQF vs. Degree Programs

**CQF Coverage:**
- Module 1: Stochastic calculus, Ito calculus (= semester 1 of any MFE)
- Module 2: Portfolio theory, CAPM, ARCH/GARCH, VaR (= risk/portfolio courses)
- Module 3: Black-Scholes, delta hedging (= derivatives courses)
- Module 4: Supervised ML (regression, SVM, k-NN, ensembles)
- Module 5: Unsupervised ML, deep learning, NLP, RL
- Module 6: Fixed income models, credit risk

**CQF Advantages:**
- 24 advanced electives including cutting-edge topics (GenAI/LLMs, quantum computing, DeFi, energy trading)
- Lifelong learning library access
- Can be completed while working (part-time format)

**CQF Gaps vs. Degree Programs:**
- Less depth in pure mathematical foundations
- No formal optimization course
- No capstone project with industry sponsor
- No internship component
- Less programming depth (no C++ focus)
- Less coverage of market microstructure

### FRM vs. Degree Programs

**FRM Part I (Foundations):**
| Topic | Weight | MFE Equivalent |
|-------|--------|----------------|
| Foundations of Risk Management | 20% | Partial overlap with intro courses |
| Quantitative Analysis | 20% | Statistics/econometrics courses |
| Financial Markets and Products | 30% | Financial instruments courses |
| Valuation and Risk Models | 30% | Derivatives + Risk Management courses |

**FRM Part II (Advanced):**
| Topic | Weight | MFE Equivalent |
|-------|--------|----------------|
| Market Risk Measurement | 20% | Risk management courses |
| Credit Risk Measurement | 20% | Credit risk electives |
| Operational Risk & Resilience | 20% | RARELY covered in MFE programs |
| Liquidity & Treasury Risk | 15% | Minimal in most MFEs |
| Investment Management | 15% | Portfolio management courses |
| Current Issues | 10% | Practitioner lecture series |

**FRM vs. MFE Key Differences:**
- FRM is much BROADER in risk coverage (operational risk, liquidity risk, regulatory frameworks)
- FRM covers Basel regulations in depth — MFE programs barely mention it
- MFE programs go DEEPER in mathematical methods (stochastic calculus, PDEs, advanced simulation)
- MFE programs include programming; FRM does not
- MFE programs include ML/AI; FRM does not
- FRM emphasizes stress testing and scenario analysis more than most MFE programs

### CFA Level III vs. Degree Programs

- CFA Level III focuses on portfolio management and wealth planning — broader than quant
- Quantitative methods are INTEGRATED throughout rather than standalone
- Covers performance attribution, risk budgeting at portfolio level
- Less mathematical rigor than MFE programs
- No programming, no ML, no derivatives pricing depth
- Stronger on ethics, client-facing aspects, and regulatory compliance

### CAIA vs. Degree Programs

- CAIA focuses on alternative investments (hedge funds, PE, real assets, structured products)
- Quantitative methods applied specifically to alternative asset classes
- Multi-factor equity pricing, binomial valuation
- Less mathematical depth than MFE
- Unique coverage of alternatives that MFE programs mostly lack

---

## 9. NOTABLE GAPS AND SURPRISES

### Gaps in Academic Programs (Topics Important in Industry but Underrepresented)

1. **Operational Risk:** Heavily tested in FRM (20% of Part II) but almost NEVER taught in MFE programs. This is a significant gap.

2. **Regulatory Frameworks (Basel III/IV, Dodd-Frank):** FRM covers these extensively; MFE programs barely mention them. NYU's XVA elective is the closest.

3. **Software Engineering Best Practices:** Most programs teach coding but NOT software engineering (version control, testing, CI/CD, code review, design patterns). CMU's C++ course is an exception. Programs teach you to write Python scripts but not production-quality code.

4. **Database Systems / Data Engineering:** Almost no program teaches SQL, data pipelines, data infrastructure. CMU's Financial Computing III touches on it.

5. **Cloud Computing / DevOps:** Zero coverage in any program. No AWS/GCP/Azure, no Docker (CQF's Algorithmic Trading II elective is the sole exception with Docker coverage), no Kubernetes.

6. **Real-Time Systems / Low-Latency Programming:** Despite market microstructure courses existing, the actual engineering of low-latency systems is not taught.

7. **Communication / Soft Skills:** MIT and CMU are notable EXCEPTIONS that include business communication courses. Most programs have zero formal communication training.

8. **Accounting / Financial Statements:** MIT requires Corporate Financial Accounting. Almost no other MFE program does. Yet understanding financial statements is critical for fundamental quant strategies.

9. **Alternative Data in Practice:** NYU has an elective on alternative data. Berkeley touches it in Financial Data Science. But detailed coverage of satellite data, web scraping at scale, NLP on SEC filings, etc. is minimal.

10. **Model Validation / Model Risk Management:** SR 11-7 (Fed guidance on model risk) is a major industry concern but barely appears in curricula.

### Surprises

1. **Oxford now requires Deep Learning as a CORE course.** This is a bold move for a traditionally mathematics-heavy program. They use PyTorch and teach it alongside stochastic calculus.

2. **Stochastic calculus is MORE entrenched than ever**, not less. Despite the rise of ML, not a single program has dropped or reduced stochastic calculus. It remains universally required.

3. **C++ is fading** as a universal requirement. Only Oxford and CMU make it mandatory. Most programs have shifted to Python-first with C++ as optional. This contrasts with industry where C++ remains critical for pricing libraries and low-latency trading.

4. **Fixed income is UNIVERSALLY taught** — more than equities. Most programs dedicate an entire course to fixed income (bonds, MBS, term structure) but do NOT have a separate equities-focused course. Equity coverage is embedded in portfolio theory and ML courses.

5. **DeFi has arrived in top programs.** Oxford, NYU, and Berkeley all offer DeFi courses. This was unthinkable 3 years ago.

6. **Princeton offers a formal ML Certificate** within their MFin program — students can earn a Graduate Certificate from the Center for Statistics and Machine Learning alongside their MFin degree.

7. **WorldQuant University offers a FREE MScFE** with substantial ML content (dedicated ML and Deep Learning courses), making quant education far more accessible.

8. **Quantum computing is appearing** in Imperial and CQF but NOWHERE else yet. Still too early for mainstream adoption.

9. **The balance is overwhelmingly derivatives > equities.** Derivatives pricing (options, fixed income derivatives, credit derivatives) receives 2-4 courses in most programs. Pure equity analysis/valuation gets almost no dedicated coursework.

10. **Behavioral finance is almost absent** from quant programs despite its relevance to market anomalies and alpha generation. Only Berkeley offers it as an elective.

---

## 10. NEW 2024-2026 ADDITIONS (Emerging Topics in Curricula)

### Confirmed New Additions

1. **CQF: "Generative AI and Large Language Models for Quant Finance"** (new advanced elective) — covers LLM architecture, NLP, strategic applications of RL, building GenAI applications for finance

2. **CQF: "Generative AI Agents in Finance & Beyond"** (new advanced elective) — technical deep dive into building and deploying AI agents

3. **Berkeley MFE: "Deep Learning for Finance I & II"** — covers RL for finance (DL I) and unsupervised learning + generative AI (DL II)

4. **MIT Sloan: "AI and Money"** (new course) — examines how ML, generative AI, and advanced analytics are redefining asset management, trading, underwriting, customer interactions, finance functions, and compliance

5. **MIT Sloan: "Intensive Hands-On Deep Learning"** (new course)

6. **Duke MIDS: "Large Language Models" for finance** (new course) — Spring 2026

7. **Duke MIDS: "Data Science for Financial Crime Detection"** (new course) — Spring 2026

8. **Duke MIDS: "Crypto Trading"** (new mini-course) — January 2026

9. **Oxford MCF: "Decentralised Finance"** elective — relatively new addition

10. **Imperial: Quantum Computing for Finance** module — recent addition to elective menu

### Topics Trending Upward (Not Yet Widespread)
- Rough volatility models (research → teaching pipeline)
- Neural SDEs and deep BSDE solvers for high-dimensional PDEs
- Deep hedging (replacing traditional delta hedging with neural networks)
- Transformer architectures for financial time series
- Graph neural networks for financial networks
- Causal inference for finance
- Federated learning for financial data privacy
- Climate risk quantification (mathematical models)

---

## APPENDIX A: DETAILED PROGRAM COURSE LISTINGS

### A1. CMU MSCF — Year One Core Courses

| Course # | Title | Timing | Category |
|----------|-------|--------|----------|
| 46901 | Financial Computing I (Python) | Mini 1 | Computing |
| 46902 | Financial Computing II (C++) | Mini 2 | Computing |
| 46903/46950 | Financial Computing III (Trading) | Mini 4 | Computing |
| 46921 | Financial Data Science I | Mini 1 | Data Science |
| 46923 | Financial Data Science II | Mini 2 | Data Science |
| 46926 | Machine Learning I | Mini 2 | Data Science |
| 46927 | Machine Learning II | Mini 3 | Data Science |
| 46929 | Financial Time Series Analysis | Mini 3 | Data Science |
| 46932 | Simulation Methods for Option Pricing | Mini 3 | Finance |
| 46944 | Stochastic Calculus for Finance I | Mini 1 | Finance |
| 46945 | Stochastic Calculus for Finance II | Mini 2 | Finance |
| 46956 | Fixed Income | Mini 4 | Finance |
| 46972 | MSCF Investments | Mini 1 | Finance |
| 46973 | MSCF Options | Mini 2 | Finance |
| 46974 | Financial Products and Markets | Mini 4 | Finance |
| 46906 | Business Communication I | Mini 1 | Professional |
| 46907 | Business Communication II | Mini 2 | Professional |
| 46909 | Practitioner Lecture Series | Mini 3 | Professional |
| 46971 | Presentations for Financial Computation | Mini 3 | Professional |

Year Two: 7 electives chosen by student.

### A2. Princeton MFin — Core Courses

| Course # | Title | Status |
|----------|-------|--------|
| FIN 501/ORF 514 | Asset Pricing I: Pricing Models and Derivatives | Core |
| FIN 503/ORF 515 | Asset Pricing II: Stochastic Calculus and Advanced Derivatives | Core |
| FIN 505/ORF 505 | Statistical Analysis of Financial Data (includes ML, SVM, neural nets) | Core |
| FIN 521 | Fixed Income (alternative to FIN 503) | Core option |
| + 2 additional core | (Economics, Optimization/Probability) | Core |
| 11 electives | From approved lists across departments | Elective |

ML Certificate: Requires 3 CSML courses (core ML + core stats + elective) + seminar + research paper.

### A3. NYU Courant — Mathematics in Finance

**Core Level 1:**
| Course # | Title |
|----------|-------|
| MATH-GA.2041 | Computing in Finance |
| MATH-GA.2791 | Financial Securities and Markets |
| MATH-GA.2751 | Risk & Portfolio Management |
| MATH-GA.2747 | Stochastic Calculus & Dynamic Asset Pricing |
| MATH-GA.2711 | Machine Learning & Computational Statistics |

**Core Level 2:**
| Course # | Title |
|----------|-------|
| MATH-GA.2755 | Project & Presentation (Capstone) |
| MATH-GA.2043 | Scientific Computing |
| MATH-GA.2048 | Scientific Computing in Finance |

**Electives (selected):**
- Time Series Analysis & Statistical Arbitrage
- Trends in Sell-Side Modeling: XVA, Capital and Credit Derivatives
- Alternative Data, Cryptocurrencies & Blockchains
- Fixed Income: Bonds, Securitized Products, and Derivatives
- Active Portfolio Management
- Advanced Risk Management
- Algorithmic Trading & Quantitative Strategies
- Interest Rate & FX Models
- Market Microstructure
- Trends in Financial Data Science

### A4. Columbia MSFE — IEOR

**Core (6 courses, 18 points):**
| Course # | Title |
|----------|-------|
| IEOR E4007 | Optimization Models and Methods for FE |
| IEOR E4701 | Stochastic Models for Financial Engineering |
| IEOR E4703 | Monte Carlo Simulation Methods |
| IEOR E4706 | Foundations of Financial Engineering |
| IEOR E4707 | FE: Continuous Time Models |
| IEOR E4709 | Statistical Analysis and Time Series |

**Seven Concentrations:** Asset Management, Computation & Programming, Computational Finance & Trading Systems, Derivatives, Finance & Economics, FinTech, ML for Financial Engineering

### A5. Oxford MCF — Course Components

**Core (Michaelmas):** Stochastic Calculus, Financial Derivatives, Numerical Methods, Statistics & Financial Data Analysis, Financial Computing with C++ I

**Core (Hilary):** Fixed Income and Credit, Stochastic Control, Quantitative Risk Management, Deep Learning

**Electives (choose 4):** Advanced Monte Carlo Methods, Advanced Volatility Modelling, Advanced Topics in Computational Finance, Asset Pricing, Market Microstructure and Algorithmic Trading, Decentralised Finance

**Computing:** Financial Computing with C++ II (end of Hilary)

**Dissertation:** Trinity Term

### A6. UC Berkeley MFE

**Core Courses:**
| Course # | Title | Units | Term |
|----------|-------|-------|------|
| MFE 230A | Investments and Derivatives | 2 | 1 |
| MFE 230E | Empirical Methods in Finance | 3 | 1 |
| MFE 230Q | Stochastic Calculus with Asset Pricing Applications | 2 | 1 |
| MFE 230D | Derivatives: Quantitative Methods | 2 | 2 |
| MFE 230I | Fixed Income Markets | 3 | 2 |
| MFE 230P | Financial Data Science | 2 | 2 |
| MFE 230H | Financial Risk Measurement and Management | 2 | 3 |
| MFE 230O | Applied Finance Project (Capstone) | 3 | 4 |

**Electives (selected):**
- Deep Learning for Finance I (RL methods)
- Deep Learning for Finance II (unsupervised + generative AI)
- High Frequency Finance (market microstructure)
- Behavioral Finance
- Dynamic Asset Management
- Decentralized Finance
- Asset Backed Security Markets
- Currency Markets
- Equity Markets
- Financial Innovation with Data Science Applications

### A7. MIT MFin

**Core Courses:**
- Foundations of Modern Finance
- Financial Mathematics OR Advanced Mathematical Methods for FE
- Programming for Finance Professionals (Python/R, pass/fail)
- Corporate Financial Accounting
- Finance Ethics & Regulation
- Corporate Finance
- Financial Markets
- Analytics of Finance OR Advanced Analytics of Finance
- Communications

**Action Learning (min 1):** Proseminar in Capital Markets, Proseminar in Corporate Finance, Finance Lab

**Restricted Electives (selected):**
- Financial Engineering
- Advanced Data Analytics and ML in Finance
- AI and Money
- AI and ML Research in Finance
- Options and Futures Markets
- Fixed Income Securities and Derivatives
- Practice of Finance: Crypto Finance
- Climate and Social Impact Investing
- Modeling with ML: Financial Technology

**Concentrations:** Capital Markets, Corporate Finance, Financial Engineering, Climate & Social Impact Finance, FinTech

### A8. CQF Full Curriculum

**Core Modules:**
1. Building Blocks of Quantitative Finance (Ito calculus, SDEs, Fokker-Planck)
2. Quantitative Risk & Return (Markowitz, CAPM, ARCH, VaR)
3. Equities & Currencies (Black-Scholes, delta hedging)
4. Data Science & Machine Learning I (supervised learning: regression, SVM, k-NN, ensembles)
5. Data Science & Machine Learning II (unsupervised, deep learning, NLP, RL)
6. Fixed Income & Credit (interest rate models, structural/reduced form credit models, copulas)

**Advanced Electives (24 available, choose 2):**
- Advanced Ensemble Modeling
- Advanced Portfolio Management
- Advanced Machine Learning I (deep sequential modeling, TensorFlow/Keras)
- Advanced Machine Learning II (ML lifecycle, experiment tracking)
- Advanced Risk Management (market, credit, climate, pandemic risk)
- Advanced Volatility Modeling (stochastic vol, rough volatility, jump diffusion)
- Algorithmic Trading I (data science workflow, APIs)
- Algorithmic Trading II (Docker, databases, backtesting, live execution)
- Behavioral Finance for Quants
- Counterparty Credit Risk Modeling (CVA, DVA, FVA)
- Decentralized Finance Technologies (blockchain, DeFi, smart contracts)
- Energy Trading
- FX Trading and Hedging
- Generative AI Agents in Finance & Beyond
- Generative AI and Large Language Models for Quant Finance
- Modeling using C++
- Quantum Computing in Finance
- Numerical Methods
- R for Data Science & Machine Learning
- Risk Budgeting

### A9. FRM Exam Topics

**Part I (100 questions, 4 hours):**
- Foundations of Risk Management (20%)
- Quantitative Analysis (20%) — probability, statistics, regression, time series
- Financial Markets and Products (30%) — derivatives, fixed income, equities, FX
- Valuation and Risk Models (30%) — VaR, stress testing, scenario analysis, BS model

**Part II (80 questions, 4 hours):**
- Market Risk Measurement & Management (20%)
- Credit Risk Measurement & Management (20%)
- Operational Risk & Resilience (20%)
- Liquidity & Treasury Risk (15%)
- Risk Management & Investment Management (15%)
- Current Issues in Financial Markets (10%)

---

## APPENDIX B: KEY QUESTIONS ANSWERED

### Is stochastic calculus still universally required?
**YES.** Every single one of the 13 degree programs requires stochastic calculus as a core course. It has not been displaced by machine learning. If anything, programs have ADDED ML on top of stochastic calculus rather than replacing it.

### Do programs still teach Black-Scholes derivation, or just application?
**Both.** Programs teach the full derivation via Ito's lemma, delta hedging arguments, and the risk-neutral pricing framework. Then they teach applications, extensions (stochastic volatility, jump diffusion), and numerical implementation. The derivation is not considered optional — it is foundational to understanding continuous-time finance.

### How much programming/engineering is in the curriculum?
**Substantial programming, minimal engineering.**
- Python is taught in ALL programs (100%)
- C++ is required in 4-5 programs (CMU, Oxford, Imperial, CQF option, Baruch implicit)
- R appears in a few (Princeton, MIT)
- MATLAB has essentially disappeared from curricula
- However: Software engineering (testing, version control, deployment, design patterns) is barely covered. Only CQF's Algorithmic Trading II (Docker, databases) approaches this.

### What's the fixed income vs. equities vs. derivatives balance?
- **Derivatives:** 2-4 dedicated courses (options, Monte Carlo, continuous-time models) — DOMINANT
- **Fixed Income:** 1-2 dedicated courses (bonds, term structure, MBS) — STRONG
- **Equities:** 0-1 dedicated courses (usually embedded in portfolio theory or ML courses) — WEAKEST
- The emphasis is heavily on derivatives and fixed income pricing. Pure equity analysis/stock selection gets very little dedicated attention in MFE programs.

### How much has ML content grown in the last 2-3 years?
**Enormously.** Specific evidence:
- 2022: ~3-4 programs had ML as core. Most offered it as an elective.
- 2024: Oxford made Deep Learning a CORE course. CMU already had 2 required ML courses.
- 2025-2026: NYU, Princeton, Berkeley all have ML in core curriculum. CQF has 2 full ML modules (out of 6 total). New electives in GenAI/LLMs appearing at CQF, MIT, Berkeley, Duke.
- Deep learning has gone from "exotic" to "expected" in just 3 years.
- Reinforcement learning for finance (Stanford CME 241, Berkeley DL I, CQF Module 5) is the next frontier becoming mainstream.

---

## APPENDIX C: COURSERA/ONLINE PLATFORM COVERAGE

### Coursera Specializations

**1. Machine Learning and Reinforcement Learning in Finance (NYU)**
- Guided Tour of Machine Learning in Finance
- Fundamentals of Machine Learning in Finance
- Reinforcement Learning in Finance
- Overview of Advanced Methods of Reinforcement Learning in Finance

**2. Investment Management with Python and Machine Learning (EDHEC)**
- Introduction to Portfolio Construction and Analysis with Python
- Advanced Portfolio Construction and Analysis with Python
- Python and Machine Learning for Asset Management
- Advanced Portfolio Construction with Python

**3. Machine Learning for Trading (Google Cloud + NYIF)**
- Introduction to Trading, Machine Learning & GCP
- Using Machine Learning in Trading and Finance
- Reinforcement Learning for Trading Strategies

**4. Quantitative Finance & Risk Modeling (Coursera)**
- 16 project-based courses covering VaR, stress testing, risk prediction

**5. WorldQuant University MScFE (Free)**
Complete 2-year program:
- MScFE 560: Financial Markets
- MScFE 600: Financial Data
- MScFE 610: Financial Econometrics
- MScFE 620: Derivative Pricing
- MScFE 622: Stochastic Modeling
- MScFE 632: Machine Learning in Finance
- MScFE 642: Deep Learning for Finance
- MScFE 652: Portfolio Management
- MScFE 660: Risk Management
- MScFE 690/692: Capstone

---

## APPENDIX D: TOPIC FREQUENCY HEAT MAP

Topic frequency across the 13 degree programs + 4 certifications:

| Topic | Count (/13 programs) | % | Category |
|-------|---------------------|---|----------|
| Stochastic Calculus | 13/13 | 100% | Universal |
| Derivatives/Options Pricing | 13/13 | 100% | Universal |
| Portfolio Theory/CAPM | 13/13 | 100% | Universal |
| Programming (Python) | 13/13 | 100% | Universal |
| Statistics/Econometrics | 13/13 | 100% | Universal |
| Monte Carlo / Numerical Methods | 12/13 | 92% | Universal |
| Fixed Income | 11/13 | 85% | Universal |
| Risk Management (market risk) | 11/13 | 85% | Universal |
| Machine Learning (any) | 11/13 | 85% | Universal |
| Time Series Analysis | 10/13 | 77% | Common |
| C++ Programming | 6/13 | 46% | Occasional |
| Optimization Methods | 7/13 | 54% | Common |
| Credit Risk/Derivatives | 8/13 | 62% | Common |
| Market Microstructure | 6/13 | 46% | Occasional |
| Algorithmic Trading | 6/13 | 46% | Occasional |
| Deep Learning (neural nets) | 7/13 | 54% | Common |
| NLP for Finance | 4/13 | 31% | Occasional |
| Reinforcement Learning | 5/13 | 38% | Occasional |
| Capstone/Thesis Project | 12/13 | 92% | Universal |
| Stochastic Control | 4/13 | 31% | Occasional |
| Volatility Modeling (advanced) | 4/13 | 31% | Occasional |
| DeFi/Blockchain/Crypto | 4/13 | 31% | Occasional |
| Behavioral Finance | 2/13 | 15% | Rare |
| ESG/Climate Finance | 3/13 | 23% | Occasional |
| Quantum Computing | 2/13 | 15% | Rare |
| Generative AI/LLMs | 3/13 | 23% | Emerging |
| Corporate Finance | 2/13 | 15% | Rare (in MFE) |
| Operational Risk | 1/13 | 8% | Rare |
| Commodities/Energy | 2/13 | 15% | Rare |
| FX Markets | 3/13 | 23% | Occasional |
