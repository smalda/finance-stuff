# Course Outline Specification

**Purpose:** This document is the instruction set for writing the actual course outline. It defines all ambiguous terms, resolves contradictions between requirements, and establishes the rules the outline must follow.

**Input documents:** RESEARCH_SYNTHESIS.md (topic tiers, career paths), COURSE_PLAN.md (current state), COURSE_RESTRUCTURE_REQUIREMENTS.md (research methodology).

---

## Part 1: Definitions

### 1.1 Coverage Depth Tiers

**COVERED** — The topic is the primary focus of one or more weeks.
- Full lecture treatment (conceptual + applied)
- Seminar exercises with working implementations
- Homework requiring students to build and evaluate
- Students achieve **working knowledge** to **mastery**
- Example: "Cross-sectional return prediction with gradient boosting" — students build a feature matrix, train XGBoost, evaluate with IC, construct a long-short portfolio

**TOUCHED** — The topic appears as a significant component (roughly 1/3 to 1/2) of a week.
- Dedicated lecture section (15–40 min equivalent)
- At least one seminar exercise with partial implementation
- May appear in homework as a sub-task
- Students achieve **conceptual awareness** to **working knowledge**
- Example: "Fixed income basics" — within a broader financial products week, students download yield curve data from FRED, bootstrap zero-coupon rates, compute duration/convexity, see how bond portfolios differ from equity. Not a standalone week, but enough to hold a conversation and build intuition from real data.

**MENTIONED** — The topic appears in lecture context with enough depth that students understand what it is, why it matters, and which career paths require it.
- A dedicated section in lecture (5–15 min equivalent, or a focused sidebar/box)
- Possibly a brief demo or worked example, but no student implementation expected
- Pointers to 2–3 resources for self-study
- Students achieve **conceptual awareness**
- Example: "XVA (CVA/DVA/FVA)" — in a derivatives-related week, a sidebar explains: "Banks compute these valuation adjustments on top of the 'clean' price. Here's what each one means intuitively, here's why the sell-side cares, here's where to learn more."

**MENTIONED ABSTRACTLY** — Brief paragraph or table row. Students learn the area exists and has significance.
- 1–3 paragraphs in lecture notes, or a row in a comparison table
- No implementation, no dedicated exercises
- "This area exists, it matters for [career paths], here's the key idea in one sentence, here's one resource"
- Students achieve **awareness** (they could name the area and say one sentence about why it matters)
- Example: "Quantum computing for finance" — "HSBC improved bond predictions 34% with quantum-hybrid methods. Goldman achieved 25x faster risk processing. Still pilot-mode. See [resource] if curious."

### 1.2 Integration Modes for Supporting Topics

**WOVEN INTO THE COURSE** — A topic that is NOT the focus of any week but appears naturally across multiple weeks as part of the applied work.
- No dedicated lecture, section, or week
- Appears through: assignment standards, implementation patterns, grading criteria, discussion prompts, tooling choices
- Students absorb it cumulatively rather than through a discrete teaching unit
- The outline must specify WHERE and HOW each woven-in topic manifests (which weeks, through what mechanism)

**Concrete examples of weaving:**
| Woven-in topic | How it appears |
|---|---|
| Production code quality | Every homework requires clean, documented code with proper abstractions. Grading criteria include code quality. No separate lecture on software engineering. |
| Communication | Homework deliverables include written analysis sections. Seminars include discussion/presentation components. |
| SQL | Data-loading exercises use SQL where appropriate (e.g., querying WRDS-style databases). |
| Python best practices | Assignments require proper use of NumPy/Pandas idioms, but never taught as a topic. |
| Data engineering / ETL | Every applied week starts with messy data. Data cleaning is the DEFAULT, not a topic. Pipeline construction is part of homework requirements. |
| Research methodology / data snooping | Every modeling week requires proper temporal CV, purged splits, honest evaluation. The methodology is taught THROUGH the application, not as a standalone lecture. |

**EXPLAINED AS PREREQUISITE** — A hard topic from math/programming/infrastructure that is necessary to understand a finance topic, but is not the course's focus.
- Appears as a "Background" or "Prerequisites Refresher" section within a week's lecture
- Provides enough to understand the application, explicitly NOT a comprehensive treatment
- Format: "Here's the minimum you need to follow this week. For a complete treatment, see [textbook chapters]."
- Goal: students can WORK WITH the concept in financial context, not derive it from scratch
- These sections should be brief (10–15 min of lecture time maximum)

**Examples of prerequisite explanations:**
| Prerequisite topic | Where it appears | What to explain | What to explicitly skip |
|---|---|---|---|
| Stochastic calculus | Before derivatives pricing | Ito's lemma intuitively, BS derivation at high level, risk-neutral pricing concept | SDE convergence proofs, measure-theoretic foundation, Girsanov theorem derivation |
| Convex optimization | Before portfolio construction | Markowitz as QP, how solvers work conceptually, what "efficient frontier" means as an optimization surface | KKT conditions, duality theory, convergence analysis |
| Bayesian inference basics | Before Bayesian DL | Prior → likelihood → posterior intuitively, conjugate priors for linear regression, connection to Ridge | Variational inference derivation, MCMC convergence diagnostics |

---

## Part 2: Contradictions Identified and Resolved

### Contradiction 1: MUST KNOW topics in "excluded" categories

**The tension:** Point 3 states the course gives no separate screen time to math, programming, data/infrastructure, or soft skills. But the research synthesis classifies several topics in these categories as MUST KNOW:
- Probability & Statistics (math) — MUST KNOW
- Time-Series Analysis / ARMA / GARCH / cointegration (math) — MUST KNOW
- Hypothesis Testing & Experiment Design (methodology) — MUST KNOW
- Python (programming) — MUST KNOW
- Production Code Quality (engineering) — MUST KNOW
- Working with Large-Scale Financial Datasets (data/infra) — MUST KNOW
- Communication (soft skill) — MUST KNOW
- Research Methodology / Avoiding Data Snooping (methodology) — MUST KNOW

**Resolution:** Apply a two-part test to each:

1. **Is the topic generic or finance-specific in practice?**
   - Generic → WOVEN IN (no screen time)
   - Finance-specific → qualifies for screen time

2. **Does the ML expert already have it?**
   - Yes → WOVEN IN (assume competence, reinforce through application)
   - No → needs explicit teaching

**Applying this test:**

| Topic | Generic or finance-specific? | ML expert has it? | Verdict |
|---|---|---|---|
| Probability & Statistics | Generic | YES | WOVEN IN. The course uses probability fluently. Finance-specific applications (IC, information ratio, deflated Sharpe) appear within finance topics. |
| Time-Series Analysis (ARMA, GARCH, frac diff) | **Finance-specific.** GARCH is THE industry vol model. Fractional differentiation is a finance-specific technique. These are not "statistics" — they are the canonical tools of financial time series. | NO (ML experts rarely know GARCH or frac diff) | **GETS SCREEN TIME** as financial domain knowledge. |
| Hypothesis Testing & Experiment Design | Borderline. The generic version is known; the finance-specific version (multiple testing correction, deflated Sharpe, backtest overfitting) is not. | Partially | Finance-specific aspects COVERED within backtesting/methodology weeks. Generic aspects WOVEN IN. |
| Python | Generic | YES | WOVEN IN. |
| Production Code Quality | Generic | Partially | WOVEN IN through homework standards and grading criteria. |
| Working with Large-Scale Financial Datasets | Finance-specific in practice (survivorship bias, point-in-time, corporate actions, etc.) | NO | **GETS SCREEN TIME** as part of "how financial data works" — a core finance topic. |
| Communication | Generic | Variable | WOVEN IN through written analysis requirements. |
| Research Methodology / Data Snooping | **Finance-specific.** Lopez de Prado's labeling, purged CV, and backtest discipline are unique to financial ML. | NO | **GETS SCREEN TIME.** This is financial ML methodology, not generic research methods. |

**The principle:** If a topic becomes genuinely finance-specific in application AND the ML expert doesn't already have it, it qualifies for screen time regardless of its academic home department. GARCH is "statistics" in a textbook but "how the industry models volatility" in practice. Purged CV is "methodology" in a textbook but "how to not fool yourself in financial backtesting" in practice.

### Contradiction 2: GOOD TO KNOW sell-side vs. NICHE topics

**The tension:** Point 4 says GOOD TO KNOW sell-side topics should be "mentioned so students understand the area exists and is important" and NICHE topics should be "mentioned like GOOD TO KNOW, but maybe more abstractly." These sound nearly identical.

**Resolution:** The difference is:
- **GOOD TO KNOW sell-side → MENTIONED:** Gets a dedicated section (5–15 min). Students learn the what, why, and who. Resources provided. Appears as a "sidebar" within a relevant week (e.g., credit risk as a sidebar in a risk week, XVA as a sidebar in a derivatives week).
- **NICHE → MENTIONED ABSTRACTLY:** Gets 1–3 paragraphs or a table row. "This exists, here's why someone cares, one resource." Appears in summary tables or "What's Beyond This Course" sections.

The distinguishing question: "Would a student entering a job interview at a buy-side firm benefit from being able to discuss this topic for 2 minutes?" If yes → MENTIONED. If the answer is "only if they're interviewing at a very specific type of firm" → MENTIONED ABSTRACTLY.

### Contradiction 3: "Infinite weeks" vs. "optimize for total learnt material"

**The tension:** Point 5 says to write with infinite weeks (correct progression, no time constraint) but also says each successive week is less likely to be studied and we should optimize for total learnt material.

**Resolution:** This is not a contradiction — it's a weighted ordering problem. The outline should:
1. Include ALL topics that meet the coverage criteria from Point 4 (MUST KNOW covered, SHOULD KNOW touched, etc.)
2. Order them by a composite score: (industry relevance × career-path breadth) balanced against (prerequisite correctness)
3. Mark explicit cutoff points: "Essential core above this line" / "Strongly recommended" / "Specialized electives"
4. After the outline is complete, the user decides where to cut — but the outline is designed so that ANY prefix (first N weeks) is a coherent course

This means the outline may have more than 18 weeks. That's by design.

### Contradiction 4: Implementability mandate vs. domain knowledge content

**The tension:** Point 8 says every week should be practical and data-driven, but some financial domain knowledge (derivatives theory, market structure concepts) is inherently conceptual.

**Resolution:** The rule is: **always prefer a data-driven or model-driven narrative over plain text.** Even "conceptual" finance topics should be taught through data and computation wherever possible.

| Seemingly conceptual topic | Data-driven approach |
|---|---|
| How markets work | Download order book snapshots, compute bid-ask spreads, build a data pipeline |
| Portfolio theory | Implement efficient frontier, construct factor models from actual returns |
| Derivatives intuition | Implement BS in PyTorch, compute Greeks via autograd, compare to market data |
| Fixed income basics | Download yield curve from FRED, bootstrap zero-coupon rates, compute duration |
| Market microstructure | Visualize LOB data, compute OFI, verify the Cont et al. linear relationship |
| Regime detection | Fit HMM to real market data, compare in-sample regimes to known events |

The implementability grade reflects how much of the week is hands-on:
- **HIGH:** ≥70% of the week is implemented in code. Most applied ML weeks.
- **MEDIUM:** 40–70% implemented. Domain knowledge weeks with computational exercises.
- **LOW:** <40% implemented. Mostly conceptual with some demo/exploration code.

**LOW implementability weeks should be RARE and should always be combined with a more implementable topic in the same week** to maintain the course's practical character.

### Contradiction 5: "Some weeks can include multiple topics" vs. clear topic mapping

**The tension (mild):** Point 9b says weeks can be multi-topic. But Point 6 says the outline should explicitly state which topics are covered and to what extent.

**Resolution:** No real contradiction — just a bookkeeping requirement. Multi-topic weeks must list EACH topic with its individual coverage depth. Example: "Week N: [Topic A — COVERED] + [Topic B — TOUCHED] + [Topic C — MENTIONED]."

---

## Part 3: Career Track Weighting

### 3.1 Priority Tiers for Career Paths

Based on: (a) relevance to ML/DL experts (from research synthesis), (b) user's stated buy-side preference, (c) accessibility of the transition.

**Priority 1 — Primary targets (buy-side, highest ML relevance):**
- ML Engineer at Fund (HIGHEST relevance — near-direct skill transfer)
- NLP/AI Engineer (VERY HIGH — fastest-growing role type)
- Data Scientist at Fund (VERY HIGH — most accessible entry point)
- Quant Researcher, Buy-Side (HIGH — the "core" transition target, broadest scope)

**Priority 2 — Strong secondary targets (buy-side or accessible):**
- Execution Quant (HIGH — underappreciated, less competition, ML-driven)
- Quant Developer / Research Engineer (HIGH for those with engineering backgrounds)
- Investment Engineer (HIGH — Bridgewater-specific but representative of a broader archetype)
- Fintech/Vendor Quant (MODERATE-HIGH — Bloomberg, MSCI, BlackRock)

**Priority 3 — Sell-side and specialized:**
- Model Validation / MRM Quant (MODERATE-HIGH — growing need for ML+regulatory combo)
- Quant Trader (MODERATE — requires probabilistic intuition ML backgrounds don't naturally provide)
- Risk Quant (MODERATE — increasingly uses ML, but further from pure ML background)

**Priority 4 — Deep sell-side (far from ML background):**
- Desk Quant / Strat (MODERATE-LOW — requires stochastic calculus + C++ + derivatives depth)
- XVA Quant (LOW — most technically demanding sell-side specialty, niche)

### 3.2 Weighting Rules

| Rule | Application |
|---|---|
| Topics required by Priority 1 roles → highest coverage | COVERED |
| Topics required by Priority 2 AND Priority 1 → stay at highest coverage | COVERED |
| Topics required by Priority 2 only (not Priority 1) → reduced | TOUCHED |
| Topics required by Priority 3 only → | MENTIONED |
| Topics required by Priority 4 only → | MENTIONED ABSTRACTLY or SKIP |
| Topics required universally (all tiers) → | COVERED, placed FIRST in the outline |

### 3.3 The Weighting Analysis the Outline Must Include

The outline document must contain a dedicated section analyzing:
1. Which career paths the course primarily serves (and why)
2. Which career paths get partial coverage (and what's missing for full preparation)
3. Which career paths are explicitly out of scope (and where those students should look instead)
4. For each major career path, which weeks are most relevant (a "reading guide by career goal")

---

## Part 4: Ordering Principles

### 4.1 Dual-Pressure Balancing

The outline orders weeks by balancing two competing pressures:

**Pressure A: Prerequisite correctness** — You can't model what you don't understand. Financial domain knowledge must come before its ML application.

**Pressure B: Industry relevance density** — More students will complete early weeks than late ones. Front-load content with the highest expected career value.

### 4.2 Resolution: Banded Ordering

Within each "band" of roughly equal industry relevance, order by prerequisites. Between bands, order by relevance.

**Band 1 — Essential foundation + highest-value application:**
Universal topics + content relevant to ALL Priority 1 roles. This is where the most careers open up. Every student should complete this band.

**Band 2 — High-value applications + key domain knowledge:**
Content relevant to MOST Priority 1 roles and many Priority 2 roles. Broadly valuable.

**Band 3 — Important specializations:**
Content relevant to SPECIFIC Priority 1/2 roles or to Priority 3 roles. More specialized but widely valued.

**Band 4 — Electives and frontier:**
Content for specific career paths, emerging areas, sell-side topics. Valuable to those who need it.

### 4.3 The Prefix Property

The outline must be designed so that ANY prefix (first N weeks) forms a coherent, self-contained learning experience. A student who stops at week 8 should have a complete (if incomplete) foundation, not a half-built edifice.

This means:
- No week should introduce something that REQUIRES a later week to make sense
- Each week should leave the student with a usable skill or framework, not a cliffhanger
- The outline should avoid "Part 1 of 2" structures where possible — if a topic needs two weeks, both should be in the same band

---

## Part 5: What the Outline Document Must Contain

### Per-Week Content

For each week, the outline states:

1. **Week number and title**
2. **Band** — which priority band (1/2/3/4) this week belongs to
3. **Topics** — explicit list with individual coverage depth (COVERED / TOUCHED / MENTIONED / MENTIONED ABSTRACTLY)
4. **Career paths served** — which career paths from the Career Paths Map this week is most relevant to, using the priority tier labels
5. **Research synthesis grounding** — which tier(s) from the Master Topic Classification this week addresses (MUST KNOW / SHOULD KNOW / GOOD TO KNOW / etc.), with specific topic names
6. **Implementability grade** — HIGH / MEDIUM / LOW with 1-sentence justification
7. **Prerequisites** — which prior weeks are required
8. **Prerequisite explanations needed** — any hard topics from math/programming/infra that need a "Background" section within this week, with a note on what to explain and what to skip
9. **Multi-topic note** — if the week covers multiple partially-related topics, state how they connect (or explicitly note they're bundled for efficiency)

### Global Sections

The outline document must also include:

1. **Preamble** — explains the ordering logic, career track weighting, and the band structure
2. **Career Track Analysis** — the full analysis from §3.3 above
3. **Cutoff markers** — explicit markers at natural break points between bands
4. **Coverage matrix** — a table mapping every MUST KNOW and SHOULD KNOW topic to which week covers it and at what depth. Rows = topics from research synthesis. Columns = week number + depth.
5. **Gap list** — topics that are NOT covered at all, with justification for each exclusion
6. **Woven-in topic manifest** — for each woven-in topic (from §1.2), which weeks it appears in and through what mechanism
7. **"What This Course Does Not Prepare You For" section** — honest statement of which career paths and topics are beyond scope, with pointers to resources

### What the Outline Document Must NOT Contain

- Lecture/seminar/homework breakdowns (that's for individual week READMEs)
- Tone, narration, or style guidance (that's in writing_guidelines.md)
- Specific code libraries or tools (that's for implementation planning)
- Paper lists or reading assignments (that's for week READMEs)
- Implementation details of any kind — only WHAT is taught, not HOW

---

## Part 6: Topic Disposition Table

This table maps every topic from the research synthesis to its treatment in the outline, applying the rules from Parts 1–4. The outline must honor these dispositions.

### MUST KNOW Topics — All must be COVERED (or WOVEN IN per §Contradiction 1)

| Topic | Disposition | Rationale |
|---|---|---|
| Probability & Statistics | WOVEN IN | ML expert already has this. Finance-specific applications appear within finance topics. |
| Time-Series Analysis (ARMA, GARCH, cointegration) | COVERED | Finance-specific. ML expert doesn't know GARCH/frac diff. |
| How Markets Work (order books, bid-ask, microstructure basics) | COVERED | Finance-specific. The #1 gap for ML-to-finance transitions. |
| Portfolio Construction & Risk Management | COVERED | Finance-specific. Required to understand what "alpha" means. |
| Backtesting Methodology (overfitting, look-ahead bias, transaction costs) | COVERED | Finance-specific methodology. Lopez de Prado's core contribution. |
| Transaction Cost Modeling | COVERED (within backtesting/portfolio weeks) | Finance-specific. "Backtests that ignore TC are worthless." |
| ML for Alpha/Signal Generation | COVERED | The core applied topic of the course. |
| Financial Feature Engineering | COVERED | Finance-specific. Practitioners stress this over model architecture. |
| Hypothesis Testing & Experiment Design | COVERED (finance-specific parts within methodology weeks) + WOVEN IN (generic parts) | Split treatment per Contradiction 1. |
| Python | WOVEN IN | ML expert already has this. |
| Production Code Quality | WOVEN IN | Enforced through homework standards. |
| Git / Version Control | WOVEN IN | ML expert already has this. |
| Working with Large-Scale Financial Datasets | COVERED (within markets/data week) | Finance-specific (survivorship bias, point-in-time, corporate actions). |
| Communication | WOVEN IN | Through written analysis requirements. |
| Research Methodology / Data Snooping | COVERED (finance-specific parts) | Financial ML methodology is finance content. |

### SHOULD KNOW Topics — All must be at least TOUCHED

| Topic | Disposition | Rationale |
|---|---|---|
| Derivatives / Options Pricing | TOUCHED to COVERED | Huge gap in current course. ~30% of buy-side listings. 100% of MFE programs. Critical for quant traders and vol researchers. Should get significant treatment with data-driven approach. |
| Factor Models (Fama-French, Barra) | COVERED | Central to the cross-sectional prediction framework that IS the course. |
| Fixed Income (yield curves, duration, interest rate models) | TOUCHED | "Half of global capital markets." MFE programs teach it universally. Can be data-driven (FRED yield curve data). |
| Stochastic Calculus | EXPLAINED AS PREREQUISITE | Required for derivatives. Not taught as standalone. Intuition-only treatment before derivatives week. |
| Optimization Methods | WOVEN IN (within portfolio construction) | Markowitz QP, solver usage. Not a standalone topic. |
| NLP/LLM for Financial Text | COVERED | Fastest-growing area. High demand, few candidates. Direct ML skill transfer. |
| Deep Learning for Financial Time Series | COVERED | Core applied topic. |
| Volatility Modeling (stochastic vol, IV surfaces) | TOUCHED | Connects to derivatives and time-series weeks. |
| Alternative Data | TOUCHED | Growing. Can be woven into NLP and feature engineering weeks. |
| Causal Inference / Causal Factor Investing | TOUCHED to COVERED | "Rapidly growing differentiator." Lopez de Prado's current research agenda. |
| Bayesian Statistics | TOUCHED (within Bayesian DL week) | Growing in importance. Natural fit with uncertainty quantification. |
| Risk Metrics (VaR, Expected Shortfall) | TOUCHED (within portfolio/risk week) | Part of the portfolio/risk coverage. |
| Position Sizing / Kelly Criterion | TOUCHED (within uncertainty/risk week) | Connects uncertainty quantification to practical decision-making. |
| Interpretable/Explainable ML for Finance | TOUCHED | Growing regulatory requirement. Can be woven into methodology weeks. |
| Data Engineering / ETL Pipelines | WOVEN IN | "70-80% of the job." Appears through messy-data-by-default homework design. |
| PyTorch / TensorFlow | WOVEN IN | ML expert already has this. Used as tool, not taught. |
| SQL | WOVEN IN | Through data exercises. |
| C++ | MENTIONED | Important for some roles but cannot be taught in this course. Acknowledge importance, point to resources. |
| Linux Environment | WOVEN IN | Through tooling. |
| Numerical Methods (Monte Carlo, PDE, trees) | TOUCHED (within derivatives week) | Required for derivatives pricing context. |
| Cross-Functional Collaboration | WOVEN IN | Through team project components. |

### GOOD TO KNOW (Buy-Side) Topics — Must be at least TOUCHED

| Topic | Disposition | Rationale |
|---|---|---|
| Market Microstructure (advanced) | TOUCHED to COVERED | Important for execution quants. Can be highly data-driven. |
| Statistical Arbitrage / Pairs Trading | TOUCHED | Historically important. Good implementability (cointegration on real data). |
| Regime Detection / HMMs | TOUCHED | Practically important. Good implementability (fit HMM to market data). |
| RL for Finance | TOUCHED to COVERED | ML experts already know RL. Finance-specific formulation is the content. |
| Bayesian DL & Uncertainty Quantification | TOUCHED to COVERED | Natural extension of existing skills. High practical value (position sizing). |
| Deep Hedging | TOUCHED (within derivatives week) | Growing. Connects DL skills to derivatives domain. |
| Diffusion Models for Finance | MENTIONED to TOUCHED | Rapidly growing. NeurIPS 2025 Best Paper. |
| Financial Foundation Models | TOUCHED (within time-series DL week) | Emerging. "Do they work?" debate is the content. |
| Synthetic Data Generation | MENTIONED | Growing but can be briefly covered within generative models discussion. |
| DeFi/Tokenization | MENTIONED to TOUCHED | 31% of MFE programs now offer it. TradFi-DeFi convergence is real. |
| GNNs for Finance | TOUCHED | Niche enough to be interesting. Good implementability. |
| GPU Computing / Distributed Training | WOVEN IN | Through assignments that benefit from GPU. |
| ESG/Climate Risk | MENTIONED | Cooling from peak but still relevant context. |
| Credit Risk / Default Prediction | MENTIONED | Growing ("quant credit takes center stage" per QuantMinds 2025). |

### GOOD TO KNOW (Sell-Side) Topics — Must be at least MENTIONED

| Topic | Disposition | Rationale |
|---|---|---|
| Regulatory Frameworks (Basel III, FRTB, SR 11-7) | MENTIONED | Essential for sell-side. Irrelevant for buy-side target audience. |
| Stochastic Control / HJB | MENTIONED | Relevant for execution and academic track. Brief conceptual mention. |

### NICHE Topics — MENTIONED ABSTRACTLY

| Topic | Disposition | Rationale |
|---|---|---|
| XVA (CVA/DVA/FVA) | MENTIONED ABSTRACTLY | Sell-side only. Sidebar within derivatives discussion. |
| Model Risk Management (SR 11-7) | MENTIONED ABSTRACTLY | Growing but bank-specific. |
| Commodities / Energy Markets | MENTIONED ABSTRACTLY | Specialized. Brief mention in market landscape. |
| FX Markets | MENTIONED ABSTRACTLY | Specialized. Brief mention in market landscape. |
| Behavioral Finance | SKIP | Minimal direct demand. |
| Conformal Prediction | MENTIONED ABSTRACTLY | Emerging. Brief note in uncertainty section. |
| Multi-Agent RL for Market Making | MENTIONED ABSTRACTLY | Near production but still niche. |
| Geopolitical Risk Modeling | MENTIONED ABSTRACTLY | Emerging. Brief note. |
| Agentic AI for Finance | MENTIONED | Dominant ICAIF 2025 topic but "agentic humility" reality check needed. Slightly above NICHE because of industry attention. |
| Rough Volatility Models | MENTIONED ABSTRACTLY | Research frontier. Brief note in vol discussion. |

### DECLINING Topics — Brief awareness mention or SKIP

| Topic | Disposition | Rationale |
|---|---|---|
| KDB/Q | MENTIONED ABSTRACTLY | "You may encounter this." Being replaced by Polars/DuckDB. |
| Pure RL for Trading | N/A — absorbed into hybrid RL coverage | Teach the hybrid approach, not pure RL. |
| GANs for Financial Data | MENTIONED (as historical context) | Being replaced by diffusion models. Brief mention. |
| R Programming | SKIP | Python covers all use cases. |
| MATLAB | SKIP | Essentially disappeared from curricula. |
| VBA | SKIP | Legacy. |

---

## Part 7: Non-Obvious Findings Integration

The research synthesis identified 12 non-obvious findings (Section 6). Each must be integrated into the outline as a recurring theme or specific teaching moment:

| Finding | Integration mode |
|---|---|
| 1. Finance knowledge NOT required at top firms | Preamble context. Reassures students that the course is sufficient preparation for entry. |
| 2. Simple regression often beats DL in production | Core teaching moment in DL weeks. Compare simple vs. complex honestly. |
| 3. Signals decay 5-10% per year | Recurring theme in every signal-generation week. Alpha decay is not a topic — it's a lens. |
| 4. Data cleaning is 70-80% of the job | WOVEN IN. Every applied week starts with messy data. Pipeline quality is graded. |
| 5. Competition is physics/math PhDs, not ML engineers | Preamble context. Motivates why financial mathematical vocabulary matters. |
| 6. Interview topics haven't changed much | Woven into probability/statistics applications. Frame concepts through interview lens where natural. |
| 7. Risk management = alpha generation in importance | Risk thinking integrated into every strategy week, not relegated to one week. |
| 8. Sisyphus paradigm (solo researchers fail) | Motivates understanding the full pipeline. Each week shows where it fits in the investment process. |
| 9. AI coding assistants make experienced devs slower on complex tasks | Brief mention in methodology context. Use assistants thoughtfully. |
| 10. C++ demand surging | MENTIONED. Acknowledge importance, point to resources. |
| 11. Kaggle as credible entry signal | Assignments designed in competition-like format where appropriate. |
| 12. DeFi-TradFi convergence | Justifies including DeFi content. |

---

## Part 8: Summary of Rules

1. **Coverage depth** follows the four-tier system (COVERED → TOUCHED → MENTIONED → MENTIONED ABSTRACTLY) as defined in §1.1
2. **Supporting topics** (math, programming, data/infra, soft skills) are WOVEN IN unless they are finance-specific AND the ML expert doesn't already have them, per §Contradiction 1
3. **Career track weighting** follows Priority 1–4 tiers. Buy-side + universal topics come first.
4. **Ordering** uses banded structure: within a band, order by prerequisites; between bands, order by industry relevance density
5. **Prefix property**: any first-N-weeks subset is a coherent learning experience
6. **Implementability**: every week should be data-driven or model-driven. LOW implementability weeks are combined with more implementable content
7. **No time restriction**: the outline includes all topics meeting the criteria. Cutoff decided after outline is complete
8. **Content only**: no tone, narration, style, code libraries, or paper lists
9. **Multi-topic weeks** are allowed and encouraged for efficiency; each topic lists its individual coverage depth
10. **Non-obvious findings** from research are integrated as recurring themes, not standalone topics
