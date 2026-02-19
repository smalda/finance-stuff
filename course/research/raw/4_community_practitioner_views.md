# Phase 1: Open Discovery -- Raw Findings
## What Practitioners and the Quant Community Say About Essential Skills for ML/DL Experts Breaking Into Quantitative Finance (2025-2026)

**Research Date:** February 15, 2026
**Scope:** Community forums, practitioner blogs, podcasts, recent books, interview resources

---

## 1. Summary of Sources Reviewed

### Community Forums
- **Reddit r/quant** -- Career advice threads, ML transition discussions, "what I wish I knew" posts
- **QuantNet** -- Career advice forum, AMA threads from buy-side quant researchers at top hedge funds (Jane Street, Two Sigma, AQR), day-to-day reality discussions, program ranking discussions
- **Wall Street Oasis** -- Quant research forum, day-in-the-life threads, quant PM Q&As, career path advice
- **Hacker News** -- Thread from experienced quant finance hiring manager on what they look for (item 8699260); thread from practitioner who "got into quant finance 12 years ago with the mistaken idea" about ML (item 14944613); thread from professional quant disagreeing with common ML assumptions (item 20723576)
- **Wilmott** -- Forum discussions on maths sweet spot, practitioner advice threads

### Practitioner Blogs and Newsletters
- **Kris Abdelmessih (Moontower/Substack)** -- Career advice for quants, trading career guidance, emphasis on coachability and probabilistic thinking
- **Cliff Asness / AQR ("Cliff's Perspective")** -- Major shift: became an "AI believer" in 2025; ML now powers ~20% of AQR's flagship fund signals; emphasis on balance between economic intuition and data
- **Alpha Architect** -- Research on factor performance, volatility premium, financial regulation and AI, private equity analysis
- **Quantocracy** -- Aggregated topics: delta hedging, market regime detection, position sizing (Kelly criterion), pragmatic asset allocation, prediction market inefficiencies
- **Matt Levine (Bloomberg/Money Stuff)** -- Podcast with Giuseppe Paleologo on what makes a good quant researcher; coverage of AI integration into quant funds; prediction markets
- **Marcos Lopez de Prado** -- 8 lectures (2023-2025) heavily emphasizing causal inference, the replication crisis in finance, and why standard ML approaches fail; "Causal Factor Investing" (2023); ongoing work at ADIA
- **Emanuel Derman** -- Perspectives on quant education at Columbia; advocates for practitioner-taught courses; notes the fundamental difference between physics models (objective) and finance models (subjective)
- **Dr. Derek Snow (ML-Quant / PyQuant News)** -- Weekly newsletter covering deep hedging, ML for risk management, multi-agent AI systems for equity research
- **Giuseppe Paleologo (Balyasny)** -- "The Elements of Quantitative Investing" (2025); podcasts on factor model limitations, multi-manager hedge fund dynamics

### Podcasts Reviewed
- **Money Stuff: The Podcast** (Matt Levine + Katie Greifeld) -- Episode with Paleologo on quant research roles
- **Flirting with Models** -- Episode with Paleologo on multi-manager hedge funds and "thinking deeply about simple things"
- **Top Traders Unplugged** -- Leading quant trading podcast with hedge fund managers and researchers
- **QuantSpeak (CQF Institute)** -- Expert insights from quants for quants
- **Talking Tuesdays with Fancy Quant** -- Quantitative topics including career development
- **The Quantopian Podcast** -- ML reshaping portfolio construction, model uncertainty

### Books and Academic Resources
- **Recent books (2024-2026):**
  - "Deep Learning in Quantitative Finance" (Wiley, Andrew Green) -- feedforward networks, volatility models, credit curve mapping, realistic data generation, hedging
  - "Deep Learning for Quant Finance: Transformers, LSTMs, and Reinforcement Learning" (Victor Trex)
  - "The Elements of Quantitative Investing" (Wiley, Giuseppe Paleologo, 2025) -- complete investment lifecycle from strategy to execution
  - "Rust for Quant Finance: The 2025 Advanced Guide" (Van Der Post)
  - "Contemporary Issues in Quantitative Finance" (Routledge, Ahmet Can Inci) -- ETFs, Monte Carlo, AI, big data analytics
  - "Bayesian Machine Learning in Quantitative Finance" (Springer, June 2025)
  - "Deep Learning: Foundations and Concepts" (Bishop & Bishop, 2025) -- transformers, normalizing flows, diffusion models
- **Canonical references still heavily cited:**
  - Lopez de Prado: "Advances in Financial Machine Learning" (2018), "Machine Learning for Asset Managers" (2020), "Causal Factor Investing" (2023)
  - Hull: "Options, Futures, and Other Derivatives"
  - Wilmott: "Paul Wilmott on Quantitative Finance"
  - Joshi: "The Concepts and Practice of Mathematical Finance"
  - Natenberg: "Option Volatility and Pricing"
  - Taleb: "Dynamic Hedging"

### Interview Resources
- "A Practical Guide to Quantitative Finance Interviews" (the "Green Book" by Xinfeng Zhou)
- "Heard on the Street" (Timothy Falcon Crack, 15th edition)
- OpenQuant interview question bank
- QuantBlueprint 12-week bootcamp (500+ real interview questions)
- Princeton Career Development quantitative interview guide
- GitHub repos: quant-prep, ml-quant-interview-prep

### Academic Surveys
- "From Deep Learning to LLMs: A Survey of AI in Quantitative Investment" (arXiv, March 2025)
- "The New Quant: A Survey of Large Language Models in Financial Prediction and Trading" (arXiv, October 2025)
- "LLMs for Quantitative Investment Research: A Practitioner's Guide" (SSRN, 2025)
- CFA Institute: "The Factor Mirage: How Quant Models Go Wrong" (October 2025)

---

## 2. Community Consensus: What Matters Most

### The Foundational Trifecta (Universal Agreement)
Every source -- forums, practitioners, hiring managers, bloggers -- converges on three pillars:
1. **Mathematics** (probability, statistics, linear algebra, calculus, stochastic calculus)
2. **Programming** (Python as lingua franca; C++ for performance-critical roles; SQL for data)
3. **Finance domain knowledge** (market microstructure, derivatives, risk management)

### What Actually Separates Candidates (Practitioner Consensus)
The community strongly agrees that beyond the trifecta, the following differentiate:

- **Statistical depth over ML breadth**: Probability and statistics are "mother tongue" level requirements. Knowing when your model breaks matters more than knowing many models. (HN hiring manager thread, QuantNet AMAs, Lopez de Prado lectures)
- **Ability to think under uncertainty**: Probabilistic reasoning, expected value thinking, calibration -- not just knowing Bayes' theorem but living it. (Kris Abdelmessih, Jane Street interviews)
- **Research taste**: Knowing which questions to ask, which data to trust, which results to be skeptical of. "Having a research sample and discussing it has the highest signal-to-noise ratio in hiring." (QuantNet AMA, Paleologo)
- **Communication skills**: Explaining what models do, where they break, and why assumptions hold. Repeatedly cited as the skill gap that "catches out" technically brilliant candidates. (Multiple sources)
- **Coachability and intellectual humility**: Kris Abdelmessih emphasizes this as the #1 predictor. Even brilliant hires at SIG need humbling. "Defensive egos that scramble feedback messages indicate poor fit."

### The Hierarchy of Skills (Synthesized Across Sources)
1. **Non-negotiable**: Probability, statistics, linear algebra, Python
2. **Strongly expected**: Stochastic calculus (derivatives roles), time series analysis, regression (all roles), basic ML (random forests, PCA, cross-validation)
3. **Differentiating**: Causal inference, Bayesian statistics, market microstructure, research methodology
4. **Role-specific**: C++ (quant dev/HFT), deep learning (ML researcher roles), reinforcement learning (execution/portfolio optimization)
5. **Emerging value**: NLP/LLM skills for textual alpha, Rust (growing but niche), MLOps/production engineering

---

## 3. Common Advice for ML/DL to Quant Transitions

### What ML Engineers Have Going for Them
- Strong programming skills (often stronger than traditional quants)
- Experience with large codebases and production systems
- Understanding of ML model lifecycle, deployment, monitoring
- Data pipeline engineering skills

### What ML Engineers Typically Lack (and Must Develop)
- **Financial domain knowledge**: "It's all well and good being the best mathematician and programmer on the globe, but if you can't tell your stock from your bond, or your bank from your fund, you'll find it harder to pass HR screenings." (QuantStart)
- **Statistical depth**: Many "data scientists" use libraries without understanding underlying assumptions and failure modes. "If you used SVM, talk about your kernel selection methods." (HN hiring manager)
- **Understanding of signal-to-noise reality**: "R-squared of 1% ARE something to write home about" in finance. Financial markets have extremely high noise and weak signals -- fundamentally different from domains where ML excels. (HN practitioner, murbard2)
- **Market microstructure**: Transaction costs, slippage, liquidity constraints, market impact -- the mechanics that destroy backtested returns
- **Stochastic calculus and probability theory**: Even if you think you won't need it, it comes up in interviews and in understanding derivatives pricing

### Specific Transition Advice from Practitioners
1. **Don't assume your ML toolkit transfers directly.** Geoffrey Hinton's framework: ML is for complex structure with low noise; statistics is for simple structure with high noise. Finance is the latter. (HN practitioner)
2. **Start with hypothesis-driven research, not data mining.** "Understanding the latent market dynamics you are trying to capitalize on, finding new data feeds that provide valuable information, carefully building out a model to test your hypothesis." (HN practitioner)
3. **Learn the seven deadly sins of ML in finance** (Lopez de Prado): Sisyphus paradigm, integer differentiation, inefficient sampling, wrong labeling, weighting non-IID samples, cross-validation leakage, backtest overfitting.
4. **Target the role that fits your strengths.** ML engineers with strong software engineering may find quant developer roles more accessible initially, with pathways to ML researcher/quant researcher roles.
5. **Build financial intuition.** Read Natenberg, Hull, Taleb. Understand what an order book is, how market making works, what drives volatility.
6. **Self-study is non-negotiable.** "For SWEs committed to making the transition to quant research roles, self-learning is indispensable." (QuantStart)
7. **Consider alternative data roles as an entry point.** Firms hiring "data scientists" for alternative data focus on Python (NumPy, Pandas, Scikit-Learn) -- skills ML engineers already have.

---

## 4. Overrated vs. Underrated Skills (Per Practitioners)

### OVERRATED (According to the Community)

| Skill/Trait | Why It's Overrated | Source |
|---|---|---|
| **Deep learning / neural networks** | "Definitely not the end-all-be-all solution" at prop trading firms. Some top market makers don't use DL at all. At one major firm, "none of the deep learning made money except simple regression." | HN, QuantNet, ML-Quant |
| **Fancy ML model architectures** | Quants start with clear hypotheses and simple models; ML practitioners overcomplicate. "The core failure mode of AI in finance is not insufficient learning capacity, but a mismatch between stationary assumptions and a competitive, reflexive environment." | LLMQuant Substack, practitioners |
| **PhD requirement** | "The notion that you need a Master's/PhD is absurd. The average undergrad quant is likely to have better outcomes than the average PhD quant." (WSO) | Wall Street Oasis |
| **Backtesting Sharpe ratios** | Most backtests are misleading. "With four parameters I can fit an elephant." Multiple testing without adjustment inflates false discovery rates dramatically. | Lopez de Prado, Quantpedia |
| **Knowing many programming languages** | Depth in Python + one systems language (C++ or Rust) beats knowing 5 languages superficially | Multiple sources |
| **LLM agents for trading** | 2024 was "agentic hype"; 2025 saw "agentic humility." Autonomy expands the action space faster than understanding. | LLMQuant 2025 retrospective |

### UNDERRATED (According to the Community)

| Skill/Trait | Why It's Underrated | Source |
|---|---|---|
| **Basic financial markets knowledge** | "Routinely catches out prospective quants at interview." | QuantStart, HN, recruiters |
| **Data cleaning and engineering** | 70-80% of a quant's time is spent on data cleaning, yet rarely emphasized in curricula | Practitioners, Quora, QuantInsti |
| **Causal inference** | "Factor investing's predicament will not be resolved with more data or more complex methods; what is most needed is causal reasoning." Lopez de Prado's dominant theme 2023-2025. | Lopez de Prado, CFA Institute |
| **Market microstructure** | "Candidates from Big Tech often lack understanding of market microstructure, risk parameters, or regulatory constraints, making them less effective." | Selby Jennings, practitioner sources |
| **Communication / soft skills** | "Many technically gifted candidates fall short here." Essential for explaining model limitations to PMs and risk managers. | Multiple sources |
| **Research methodology / experimental design** | Knowing how to design a proper out-of-sample test, control for multiple comparisons, avoid data snooping. Much more important than model selection. | Lopez de Prado, Quantpedia |
| **Transaction costs / execution** | Backtests that ignore realistic transaction costs, market impact, and slippage are worthless | HN practitioner, practitioners |
| **Risk management** | "Should be considered of equal importance to alpha generation, as opposed to a secondary function." | CQF, Quant PM discussions |
| **Bayesian statistics** | Growing rapidly in importance; natural framework for updating beliefs with new data, integrating alternative data, and quantifying uncertainty | Springer 2025 book, practitioners |
| **Coachability** | "Even brilliant hires at elite firms like SIG need humbling." | Kris Abdelmessih |

---

## 5. Day-to-Day Reality vs. Academic Expectations

### What People Think Quants Do
- Build sophisticated mathematical models all day
- Develop cutting-edge ML algorithms
- Make brilliant market predictions
- Work with clean data and elegant theories

### What Quants Actually Do

**Quant Researcher (day-to-day):**
- Read new research in ML and quantitative finance, think about how to apply it
- Work with team to determine which approaches to investigate further
- Initial research is very independent; collaboration increases when something promising emerges
- Post-trade analysis: understanding why trades succeeded or failed
- Debugging models when they break (sometimes sleepless nights)
- Hours: typically 9am-7pm, with crunch periods
- "Most of a quant researcher's day is spent reading" -- not coding

**Quant Trader (day-to-day):**
- Monitor market movements during trading hours
- Execute trades and adjust strategies
- Many quant traders "don't even code at all" and are more macro oriented
- After market close: think about projects, analyze the day
- A Citadel Securities quant: works on model improvements, applies them 90 min before clocking off, runs overnight experiments

**Quant Developer (day-to-day):**
- 6AM: Check automated data tasks and overnight cron jobs, fix failed scripts
- Morning: Infrastructure maintenance, fixing API changes, debugging data pipelines
- Afternoon: New data source integration, feature development, unit testing
- "A mixture of diagnosis and repair of abnormalities in the infrastructure" and "development of new requested features"
- Tools: Python, unit testing frameworks, brokerage APIs, automated monitoring
- Problems: undocumented API changes, buggy data points, pricing anomalies

### The Unglamorous Truth
- **Data cleaning dominates**: 70-80% of time on data preparation, not model building
- **Infrastructure matters enormously**: Failed cron jobs, broken APIs, data quality issues consume daily attention
- **Signal strength is tiny**: R-squared of 1% is noteworthy. Most signals are ephemeral.
- **Models break regularly**: Monitoring, debugging, and patching are continuous
- **Simplicity usually wins**: "At one major investment firm, none of the deep learning made money except simple regression"
- **It's a team sport**: No single researcher builds complete strategies alone (Lopez de Prado's "Sisyphus paradigm")

---

## 6. Field Changes and Trends (2023-2026)

### Major Shifts

**1. AI Goes from Skepticism to Selective Adoption (2023-2025)**
- Cliff Asness went from AI skeptic to "AI believer" -- ML now powers ~20% of AQR's flagship fund signals
- AQR using ML to parse unstructured data (earnings calls), assign weights to trading signals
- But Asness maintains: "It's not magic" -- it's a more sophisticated analytical capability
- Key application: NLP for textual alpha, not predicting price from price

**2. The Pivot from Models to Systems (2025)**
- "2025: The Year Quant Finance Stopped Chasing Models and Started Shipping Systems"
- "AI did not conquer markets in 2025. It conquered workflows."
- Competitive advantage shifted from model novelty to integration, operational robustness, governance
- Skills that matter now: "systems engineering, deployment, and operational resilience"

**3. Causal Inference Becomes Central (2023-2025)**
- Lopez de Prado's dominant research agenda: moving from association to causation
- "The Factor Mirage" (CFA Institute, 2025): most factor models are causally misspecified
- "Factor investing's predicament will not be resolved with more data or more complex methods; what is most needed is causal reasoning"
- Growing recognition that standard econometric textbook approach (include anything correlated) is fundamentally flawed

**4. Alpha Decay Accelerates (2024-2025)**
- Predictive signals losing 5-10% effectiveness annually in developed markets
- Dozens of participants rushing to execute on the same signal simultaneously
- Shared datasets, overlapping strategies, common talent pools create correlated failure
- 2025 quant stumbles: crowding and deleveraging triggered cascading losses across firms
- "Signals that once provided a clear advantage are now widely recognized"

**5. LLMs Enter (and Get Reality-Checked) (2024-2025)**
- 2024: "agentic hype" -- LLM-based trading agents look compelling on paper
- 2025: "agentic humility" -- autonomy expands action space faster than understanding
- Practical LLM uses: NLP for sentiment extraction, earnings call analysis, regulatory filing parsing
- LLM hallucination in finance is "potentially hazardous" -- accuracy requirements are non-negotiable
- Systemic risk concern: herd behavior when many agents learn from similar templates
- New role emerging: "LLM Quant Analyst" -- fast-growing but niche

**6. Cross-Functional Teams Replace Siloed Roles (2024-2025)**
- Quant firms creating "agile, cross-functional squads" of researchers, engineers, ML experts, PMs
- The "new generation of AI-first quant roles demands a blend of statistical rigour, software engineering, ML fluency, and domain-specific knowledge"
- Hiring for versatility over narrow specialization

**7. C++ Developer Demand Surges (2024-2025)**
- UK median salary jumped to 170K GBP, up 17.24% YoY
- Number of permanent roles nearly doubled (33 vs 18 prior year)
- Low-latency and HFT driving demand; 93.94% of C++ quant job ads mention low-latency
- Rust gaining traction in crypto/digital assets but C++ dominance continues for traditional finance

**8. Non-Stationarity Acknowledged as Core Challenge**
- "The most intellectually honest development of 2025 was renewed focus on non-stationarity"
- "Markets adapt faster than models do"
- "The core failure mode of AI in finance is a mismatch between stationary assumptions and a competitive, reflexive environment"

**9. Synthetic Data Under Scrutiny**
- "Synthetic data without governance is just a more subtle form of overfitting"
- Requires same discipline as other data: validation, versioning, explicit limitations

### What's Growing
- NLP / textual alpha from alternative data
- Causal inference methods
- Production ML / MLOps for finance
- Bayesian approaches to uncertainty quantification
- C++ low-latency development
- Cross-asset systematic strategies
- Risk management as first-class concern
- Reinforcement learning for execution (academic momentum, limited production use)

### What's Shrinking or Stable
- Traditional stochastic calculus-only roles (stable but evolving)
- Pure derivatives pricing quants (bank-side contracting somewhat)
- Standalone ML model development without production awareness
- Strategies that only work in backtests

---

## 7. Interview Preparation Topics

### What Firms Actually Test (Synthesis Across Sources)

**Universal Core (All Quant Roles):**
- Probability puzzles and brain teasers (especially Jane Street, Citadel, SIG)
- Mental math under pressure
- Combinatorics problems
- Linear algebra applications
- Statistics: hypothesis testing, regression interpretation, distribution properties
- Coding challenges: algorithm design, data structures (Python or C++)

**Role-Specific:**

| Role | Key Interview Topics |
|---|---|
| **Quant Trading** | Probability (heavy), expected value, market making scenarios, option pricing intuition, game theory, betting strategy |
| **Quant Research** | Statistics (deep), ML fundamentals (regression, random forests, PCA, cross-validation), research methodology, time series, stochastic processes, research presentation |
| **Quant Developer** | C++ (low-latency, memory management, multithreading), Python (NumPy, Pandas), system design, algorithm optimization, data structures |
| **ML Researcher** | Deep learning architectures, PyTorch/TensorFlow, kernel methods, feature engineering, model evaluation, NLP, reinforcement learning |

**What's Growing in Interviews (2024-2025):**
- Machine learning questions: "really growing" -- linear regression interpretation, model evaluation metrics, bias-variance tradeoff
- Practical ML: "If you used SVM, talk about your kernel selection methods"
- Research sample presentation: "has the highest signal-to-noise ratio in hiring"
- Causal inference questions (emerging at some firms)

**Key Resources for Preparation:**
- "A Practical Guide to Quantitative Finance Interviews" (Green Book) -- covers brain teasers, calculus, linear algebra, probability, stochastic processes, finance, programming
- "Heard on the Street" (15th edition) -- 185+ questions from actual interviews covering quant/logic, financial economics, derivatives, statistics
- OpenQuant.co question bank
- QuantBlueprint bootcamp (500+ real questions)
- Princeton Career Development guide

**Interview Format:**
- Often starts with 30-minute online assessment
- Preliminary interview: 45-60 minutes
- Typically 5-15 minutes per question set
- Multiple rounds at top firms
- Some firms (Jane Street, SIG) have notoriously difficult probability-heavy interviews
- Research roles often include a take-home project or research presentation

---

## 8. Recommended Resources from Practitioners

### Books Most Frequently Cited by Practitioners

**For ML/DL Practitioners Entering Finance:**
1. Lopez de Prado: "Advances in Financial Machine Learning" -- the most-cited ML-in-finance book by practitioners
2. Lopez de Prado: "Machine Learning for Asset Managers" -- shorter, more focused
3. Lopez de Prado: "Causal Factor Investing" -- the emerging frontier
4. Hull: "Options, Futures, and Other Derivatives" -- foundational finance knowledge
5. Natenberg: "Option Volatility and Pricing" -- practical derivatives understanding
6. Paleologo: "The Elements of Quantitative Investing" (2025) -- complete practitioner guide
7. Andrew Green: "Deep Learning in Quantitative Finance" (Wiley) -- DL applied properly to finance

**For Career Understanding:**
- Derman: "My Life as a Quant"
- Patterson: "The Quants" / Mallaby: "More Money Than God"
- Bernstein: "Against the Gods"

**For Interview Prep:**
- Zhou: "A Practical Guide to Quantitative Finance Interviews" (Green Book)
- Crack: "Heard on the Street"

**For Algorithmic Trading:**
- Chan: "Quantitative Trading" (intro) and "Algorithmic Trading" (advanced)
- Narang: "Inside the Black Box"

### Newsletters and Blogs
- Moontower (Kris Abdelmessih) -- options, volatility, career
- ML-Quant / Dr. Derek Snow -- ML in quant finance weekly
- PyQuant News -- Python for quant finance
- Quantocracy -- aggregator of quant blog posts
- LLMQuant Substack -- AI/LLM in quant finance
- Matt Levine's Money Stuff -- broader financial markets context

### Podcasts
- Top Traders Unplugged -- leading quant trading interviews
- Flirting with Models -- quantitative investing
- Money Stuff: The Podcast -- finance with quant perspective
- QuantSpeak -- quant-to-quant insights

---

## 9. Debates and Controversies

### Active Debates in the Community (2024-2026)

**1. Does ML Actually Work in Finance?**
- **Skeptics**: "Machine learning is used a lot less than people think" (HN hiring manager). R-squared of 1% is noteworthy. Simple regression often beats complex models. Finance is high-noise, weak-signal -- the opposite of where ML excels.
- **Proponents**: AQR now has 20% of flagship signals from ML. NLP for textual alpha works. The key is applying ML to the right sub-problems (data parsing, signal weighting) not end-to-end prediction.
- **Emerging consensus**: ML works for specific tasks (NLP, signal combination, execution optimization) but not as a magic prediction engine. The "model" was never the bottleneck -- data, infrastructure, and research methodology are.

**2. Causal vs. Associational Methods**
- Lopez de Prado's crusade: most factor investing is causally confused
- "Econometric textbooks teach that regressions should include any variable associated with returns regardless of its role in the causal mechanism -- a methodological error"
- Counter-argument: in practice, causal methods are still immature and difficult to apply to financial data
- Growing acceptance that at minimum, researchers must think causally even if tools are imperfect

**3. PhD Necessity**
- Some sources: Renaissance only hires PhDs; Two Sigma strongly biases toward PhDs
- WSO contrarian view: "The average undergrad quant is likely to have better outcomes than the average PhD quant"
- Practical consensus: PhD helps for research roles at top firms, but is not strictly required; demonstrated ability matters more than credential

**4. Markets Becoming More or Less Efficient?**
- Cliff Asness: markets have become "less efficient over the past three decades, rather than more so"
- Counter-view: alpha decay data suggests competitive efficiency is increasing for quantitative signals
- Resolution: may be that markets are less efficient for fundamental mispricings but more efficient for quantitative signals (crowding)

**5. LLMs: Transformative or Overhyped?**
- Hype camp: LLMs can parse qualitative data that quants couldn't before; game-changer for fundamental analysis
- Skeptic camp: hallucination risk is unacceptable in finance; herd behavior risk; "synthetic data without governance is just a more subtle form of overfitting"
- Practical middle ground: LLMs useful for workflow acceleration and NLP tasks, not for autonomous trading decisions

**6. The Talent Pipeline Problem**
- "Is the talent pool keeping up?" -- firms struggling to find candidates who combine ML, statistics, finance domain knowledge, and software engineering
- Big Tech candidates have engineering but lack market intuition
- Academic candidates have theory but lack production skills
- Very few people bridge ML expertise with financial domain knowledge -- "a relatively rare skill combination"

**7. Stochastic Calculus: Still Relevant?**
- Still foundational for derivatives pricing, still tested in interviews
- But for stat arb / ML researcher roles, decreasing in relative importance vs. statistics, ML, and causal inference
- Consensus: you need to know it, but the era where stochastic calculus was the core skill is fading for non-derivatives roles

**8. Private Assets Criticism (Asness)**
- Cliff Asness: private equity allocations failed to protect from volatility; private credit is "essentially high-fee public credit that performed worse after fees"
- Implications for quants: liquid systematic strategies gaining favor vs. illiquid alternatives

---

## 10. Surprising or Non-Obvious Findings

### Things That Might Surprise ML/DL Practitioners

1. **"Simple regression" often beats deep learning in production.** At one major investment firm, "none of the deep learning made money except simple regression." This is not apocryphal -- it reflects the signal-to-noise reality of financial markets. Complex models overfit to noise.

2. **Data cleaning is 70-80% of the job.** Despite all the modeling theory, the bulk of quant work is data engineering -- getting disparate datasets aligned, handling API changes, fixing anomalies. This is rarely mentioned in textbooks or course descriptions.

3. **Your biggest competition isn't other ML engineers -- it's physics and math PhDs.** Top quant firms recruit heavily from pure math, physics, and statistics programs. These candidates often have deeper mathematical foundations even if their programming is weaker.

4. **Coachability might matter more than raw intelligence.** Kris Abdelmessih (21 years as a volatility trader) identifies coachability -- not IQ or technical skill -- as the #1 predictor of success. "If you need hand-holding to identify quality content, you're already disadvantaged."

5. **The "Sisyphus paradigm" explains why solo ML researchers fail.** Lopez de Prado argues that building investment strategies requires specialized teams with divided subtasks. Individual researchers attempting complete strategies alone face insurmountable complexity. "Every successful quantitative firm applies the meta-strategy paradigm."

6. **Interview questions haven't changed as much as you'd think.** Despite all the ML hype, probability puzzles, brain teasers, and statistical reasoning still dominate interviews. ML questions are growing but supplement rather than replace the classics.

7. **Risk management is as important as alpha generation** -- but almost nobody talks about it in educational contexts. PMs who understand risk allocation, position sizing, and drawdown management are more valuable than those who can only generate signals.

8. **Signals decay 5-10% per year.** In highly electronic markets, the half-life of a quantitative signal is shockingly short. This means continuous research and signal generation is the core job -- not building one great model and walking away.

9. **The 2025 pivot: from models to systems.** The competitive advantage in quant finance has shifted from "who has the best model" to "who has the best production infrastructure." AI conquered workflows before it conquered markets.

10. **Rust is coming, but not fast enough to matter yet for most.** While Rust adoption is growing (especially in crypto), the overwhelming majority of quant finance infrastructure remains C++ and Python. Learning Rust is optionally valuable but not a priority for career switchers.

11. **Causal inference is the next frontier.** Lopez de Prado has made this his primary research agenda. The "factor mirage" -- factor models that appear valid but are causally misspecified -- is now a recognized problem. This field is wide open and under-explored relative to its importance.

12. **Firms that hire fast, win.** "Candidates are often receiving and accepting offers in under a week." The talent market is extremely competitive, and speed of hiring process is a differentiator for firms, not just candidates.

13. **Some quant traders don't code at all.** Despite the stereotype, many quant traders are "more macro oriented" and focus on market intuition, risk management, and execution rather than model building.

14. **"If you're going to figure it out, you just kind of *are*."** (Kris Abdelmessih) Talent finds appropriate mentors and resources organically. If the foundational reading material feels like a chore rather than engaging, the career probably isn't right. This is the most honest filter.

---

## Source Index

### Community Forum Sources
- [QuantNet: Day-to-Day Quant Job Reality](https://quantnet.com/threads/realistically-what-do-quants-do-as-their-day-to-day-job.56257/)
- [QuantNet: Buy-Side Quant Researcher AMA](https://quantnet.com/threads/im-a-buy-side-quant-researcher-at-a-top-hedge-fund-jane-street-two-sigma-aqr-etc-ama.61401/)
- [QuantNet: Senior Buy-Side QR AMA](https://quantnet.com/threads/im-a-senior-buy-side-quant-researcher-ama.53587/)
- [QuantNet: Study Programme for QR Interviews](https://quantnet.com/threads/study-programme-for-quant-researcher-interviews.50152/)
- [HN: Hiring Quants and Programmers in Finance](https://news.ycombinator.com/item?id=8699260)
- [HN: Quant Finance 12 Years On -- ML Misconceptions](https://news.ycombinator.com/item?id=14944613)
- [WSO: So You Want to Be a Quant](https://www.wallstreetoasis.com/forum/hedge-fund/so-you-want-to-be-a-quant)
- [WSO: Q&A Quantitative Trading Advice](https://www.wallstreetoasis.com/forum/trading/q&a-quantitative-trading-advice-20212024-recruit-multiple-offers)
- [WSO: Day in the Life -- Quant Trader](https://www.wallstreetoasis.com/forum/hedge-fund/day-in-the-life-quant-trader)
- [WSO: Quant Research vs Quant Trading](https://www.wallstreetoasis.com/forum/hedge-fund/quant-research-vs-quant-trading)

### Practitioner Blog / Newsletter Sources
- [Moontower: Career Advice for Quants](https://medium.com/@moontower/career-advice-for-quants-3a4b6920b936)
- [Moontower: So You're Interested in Trading](https://moontower.substack.com/p/so-youre-interested-in-trading)
- [AQR: Cliff's Perspectives](https://www.aqr.com/Insights/Perspectives)
- [Cliff Asness on AI in Finance](https://acquirersmultiple.com/2025/11/cliff-asness-on-quant-investing-and-ais-role-in-finance/)
- [AQR Bets on ML -- Bloomberg](https://www.bloomberg.com/news/articles/2025-04-23/aqr-bets-on-machine-learning-as-cliff-asness-becomes-ai-believer)
- [Alpha Architect Blog](https://alphaarchitect.com/blog/)
- [Quantocracy](https://quantocracy.com/)
- [ML-Quant / Derek Snow Substack](https://blog.ml-quant.com/)
- [PyQuant News](https://www.pyquantnews.com)
- [LLMQuant: 2025 Quant Finance Retrospective](https://llmquant.substack.com/p/2025-the-year-quant-finance-stopped)

### Lopez de Prado Sources
- [Lectures Page](https://www.quantresearch.org/Lectures.htm)
- [Publications Page](https://www.quantresearch.org/Publications.htm)
- [Causal Factor Investing (Quantitative Finance)](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2354849)
- [The Factor Mirage -- CFA Institute](https://blogs.cfainstitute.org/investor/2025/10/30/the-factor-mirage-how-quant-models-go-wrong/)

### Book Sources
- [Deep Learning in Quantitative Finance (Wiley)](https://www.wiley.com/en-us/Deep+Learning+in+Quantitative+Finance-p-00066021)
- [The Elements of Quantitative Investing (Paleologo)](https://www.amazon.com/Elements-Quantitative-Investing-Wiley-Finance/dp/139426545X)
- [Bayesian ML in Quantitative Finance (Springer)](https://link.springer.com/book/10.1007/978-3-031-88431-3)
- [Rust for Quant Finance 2025](https://www.amazon.com/Rust-Quant-Finance-High-Performance-Computing-ebook/dp/B0FPVTKW7C)
- [QuantStart Reading List](https://www.quantstart.com/articles/Quantitative-Finance-Reading-List/)
- [QuantNet Master Reading List](https://quantnet.com/threads/master-reading-list-for-quants-mfe-financial-engineering-students.535/)
- [OpenQuant: Best Books for Quant Finance](https://openquant.co/blog/best-books-for-quant-finance)

### Interview Preparation Sources
- [OpenQuant Interview Questions](https://openquant.co/questions)
- [Statistics and ML for Quant Interviews](https://openquant.co/blog/statistics-and-ml-concepts-for-quant-finance-interview)
- [Princeton Quantitative Interview Preparation](https://careerdevelopment.princeton.edu/guides/interviews/quantitative-interview-preparation)
- [QuantBlueprint Bootcamp](https://www.quantblueprint.com/quant-bootcamp)
- [GitHub: ML-Quant Interview Prep](https://github.com/meagmohit/ml-quant-interview-prep)

### Academic / Research Sources
- [AI in Quantitative Investment Survey (arXiv)](https://arxiv.org/html/2503.21422v1)
- [LLMs in Financial Prediction Survey (arXiv)](https://arxiv.org/html/2510.05533v1)
- [LLMs for Quantitative Investment -- Practitioner's Guide (SSRN)](https://papers.ssrn.com/sol3/Delivery.cfm/5934015.pdf?abstractid=5934015)
- [Why ML Funds Fail (Quantpedia)](https://quantpedia.com/why-machine-learning-funds-fail/)
- [The Great Quant Convergence (KX)](https://kx.com/blog/the-great-quant-convergence-hedge-funds-kx/)
- [Quant Unwind Manual (Resonanz Capital)](https://resonanzcapital.com/insights/crowding-deleveraging-a-manual-for-the-next-quant-unwind)

### Career and Industry Sources
- [QuantStart: Engineering to Quant Transition](https://www.quantstart.com/articles/engineering-to-quant-finance-how-to-make-the-transition/)
- [QuantStart: Quant Career Skills](https://www.quantstart.com/articles/quant-finance-career-skills-what-are-employers-looking-for/)
- [QuantStart: Day in the Life of a Quant Developer](https://www.quantstart.com/articles/A-Day-in-the-Life-of-a-Quantitative-Developer/)
- [Selby Jennings: Quant Talent Trends 2025](https://www.selbyjennings.com/en-us/industry-insights/hiring-advice/what-s-influencing-quant-talent-in-2025)
- [Selby Jennings: Key Hiring Trends 2025](https://www.selbyjennings.com/en-us/industry-insights/hiring-advice/key-trends-impacting-quant-hiring-in-2025)
- [Harrington Starr: Quants and the AI Arms Race](https://www.harringtonstarr.com/resources/blog/quants-and-the-ai-arms-race--is-the-talent-pool-keeping-up-/)
- [C++ for Quants: Job Market](https://cppforquants.com/c-quantitative-developers-a-skyrocketing-job-market/)
- [Medium: Breaking into Quant Finance 2025](https://medium.com/@anil.oezer/breaking-into-quant-finance-in-2025-why-its-still-one-of-the-toughest-games-in-town-36a5f16d9e06)
- [CQF Careers Guide 2024](https://cloudtechssolution.com/wp-content/uploads/2025/03/navigating-the-future-of-en.pdf)

### Podcast Sources
- [Money Stuff: Gappy Paleologo Episode](https://podcasts.apple.com/us/podcast/gappy-paleologo/id1739582836?i=1000715731351)
- [Flirting with Models: Paleologo on Multi-Manager HFs](https://open.spotify.com/episode/6NbQI7GFMUU0e8x1njTcP8)
- [Top Traders Unplugged](https://player.fm/podcasts/Quantitative-Finance)
- [QuantSpeak](https://cqfinstitute.org/resources/podcasts/)
