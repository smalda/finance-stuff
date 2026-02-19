# ML for Quantitative Finance

An 18-week course teaching **ML/DL experts** quantitative finance — not the other way around. You already know neural architectures, training methodology, and standard ML. This course teaches the financial domain knowledge, evaluation frameworks, and industry-specific adaptations you need to apply that expertise in quant roles.

## Course Structure

The course is organized in three bands of increasing specialization. Any first-N-weeks prefix forms a self-contained learning experience.

### Band 1: Essential Core (Weeks 1-6)

The complete alpha pipeline — data, features, model, backtest, portfolio.

| Week | Topic | Focus |
|------|-------|-------|
| 1 | [How Markets Work & Financial Data](course/week01_markets_data/) | Exchanges, order types, bid-ask spreads, survivorship bias, data pipelines |
| 2 | [Financial Time Series & Volatility](course/week02_time_series_volatility/) | Returns, stationarity, ARMA, GARCH family, realized volatility, fractional differentiation |
| 3 | [Factor Models & Cross-Sectional Analysis](course/week03_factor_models/) | CAPM, Fama-French, Barra, Fama-MacBeth, financial feature engineering |
| 4 | [ML for Alpha](course/week04_ml_alpha/) | Gradient boosting + neural nets for cross-sectional prediction, IC/rank IC, signal-to-portfolio |
| 5 | [Backtesting & Transaction Costs](course/week05_backtesting_txcosts/) | Purged CV, deflated Sharpe, market impact, the seven sins of backtesting |
| 6 | Portfolio Construction & Risk Management | Mean-variance, risk parity, Black-Litterman, HRP, VaR/ES, Kelly criterion |

### Band 2: High-Value Applications (Weeks 7-12)

Major applied ML domains + essential knowledge beyond equities.

| Week | Topic | Focus |
|------|-------|-------|
| 7 | NLP & LLMs for Financial Alpha | Earnings call parsing, FinBERT vs. LLM embeddings, textual + numerical alpha models |
| 8 | Deep Learning for Financial Time Series | LSTM/GRU for vol forecasting, attention, financial foundation models (Kronos, FinCast) |
| 9 | Derivatives Pricing & the Greeks | Black-Scholes, the Greeks, implied vol surfaces, SABR, Monte Carlo pricing |
| 10 | Causal Inference & Modern Factor Investing | DAGs, collider bias, causal factor investing, SHAP for financial models |
| 11 | Bayesian Methods & Uncertainty | MC dropout, deep ensembles, calibration, uncertainty-aware position sizing |
| 12 | Fixed Income & Interest Rates | Bonds, yield curves, duration/convexity, Vasicek/CIR, credit risk intro |

### Band 3: Specializations (Weeks 13-18)

Advanced domains for specific career tracks.

| Week | Topic | Focus |
|------|-------|-------|
| 13 | Market Microstructure & Execution | LOB dynamics, market impact (Kyle, Almgren-Chriss), OFI, execution algorithms |
| 14 | Statistical Arbitrage & Regime Detection | Cointegration, Ornstein-Uhlenbeck, pairs trading, HMMs, changepoint detection |
| 15 | Reinforcement Learning for Finance | Financial MDPs, reward shaping, hybrid RL, why pure RL fails in finance |
| 16 | Deep Hedging & Neural Derivatives Pricing | Neural surrogates for pricing, hedging under transaction costs, neural calibration |
| 17 | Generative Models & GNNs for Finance | Diffusion models for synthetic data, LOB simulation, graph-based stock prediction |
| 18 | DeFi, Emerging Markets & Frontier Topics | AMMs, tokenized assets, ESG/climate, agentic AI, quantum computing |

## Notebooks

Weeks 1-5 have complete lecture, seminar, and homework notebooks:

| Week | Lecture | Seminar | Homework |
|------|---------|---------|----------|
| 1 — Markets & Data | [lecture.ipynb](course/week01_markets_data/lecture.ipynb) | [seminar.ipynb](course/week01_markets_data/seminar.ipynb) | [hw.ipynb](course/week01_markets_data/hw.ipynb) |
| 2 — Time Series & Volatility | [lecture.ipynb](course/week02_time_series_volatility/lecture.ipynb) | [seminar.ipynb](course/week02_time_series_volatility/seminar.ipynb) | [hw.ipynb](course/week02_time_series_volatility/hw.ipynb) |
| 3 — Factor Models | [lecture.ipynb](course/week03_factor_models/lecture.ipynb) | [seminar.ipynb](course/week03_factor_models/seminar.ipynb) | [hw.ipynb](course/week03_factor_models/hw.ipynb) |
| 4 — ML for Alpha | [lecture.ipynb](course/week04_ml_alpha/lecture.ipynb) | [seminar.ipynb](course/week04_ml_alpha/seminar.ipynb) | [hw.ipynb](course/week04_ml_alpha/hw.ipynb) |
| 5 — Backtesting & Tx Costs | [lecture.ipynb](course/week05_backtesting_txcosts/lecture.ipynb) | [seminar.ipynb](course/week05_backtesting_txcosts/seminar.ipynb) | [hw.ipynb](course/week05_backtesting_txcosts/hw.ipynb) |

## Repo Structure

```
finance_stuff/
├── course/
│   ├── COURSE_OUTLINE.md              # Week-by-week outline with topic depths, career mappings
│   ├── curriculum_state.md            # Cumulative record of what prior weeks taught
│   ├── guides/                        # Pipeline guide files (one per step + shared references)
│   ├── research/                      # Outline research: synthesis + 5 raw agent reports
│   ├── shared/                        # Cross-week infrastructure
│   │   ├── data.py                    #   Data cache layer (48 datasets, ~143 MB)
│   │   ├── metrics.py                 #   IC, rank IC, ICIR, hit rate, R²_OOS
│   │   ├── temporal.py                #   Walk-forward, expanding, purged CV splitters
│   │   ├── evaluation.py              #   Rolling cross-sectional prediction harness
│   │   ├── backtesting.py             #   Quantile portfolios, long-short, Sharpe, drawdowns
│   │   ├── dl_training.py             #   NN fit/predict, SequenceDataset, device support
│   │   └── [domain modules]           #   portfolio, derivatives, microstructure, regime, nlp, etc.
│   └── weekNN_topic/                  # One folder per week
│       ├── research_notes.md          #   Step 1: domain knowledge
│       ├── blueprint.md               #   Step 2: ideal teaching plan
│       ├── expectations.md            #   Step 3: data reality + acceptance criteria
│       ├── code/                      #   Step 4: verified Python (lecture/, seminar/, hw/)
│       ├── observations.md            #   Step 5: fresh-eyes review of code output
│       ├── narrative_brief.md         #   Step 6: reconciled teaching narrative
│       ├── lecture.ipynb              #   Step 7: final notebook
│       ├── seminar.ipynb              #   Step 7: final notebook
│       └── hw.ipynb                   #   Step 7: final notebook
├── legacy/                            # Previous course versions (see Appendix C)
│   ├── first_draft_weeks/             #   Gen 1: 18 weeks of directly-drafted notebooks
│   ├── old_guides/                    #   Gen 2: 6-file guide system (~3,000 lines)
│   ├── old_pipeline_weeks/            #   Gen 2: weeks 3-4 partially rebuilt
│   └── QUALITY_AUDIT_W1_W4.md         #   Audit that motivated the pipeline redesign
├── nb_builder.py                      # Converts build scripts to .ipynb notebooks
├── pyproject.toml / poetry.lock       # Python 3.12 dependencies (Poetry)
└── CLAUDE.md                          # Instructions for the AI authoring system
```

Each week folder accumulates artifacts as it moves through the pipeline. Weeks 3-5 have the full set (research through notebooks). Weeks 1-2 have research, code, and notebooks but predate the blueprint/expectations/consolidation steps. Weeks 6-18 are scaffolded but not yet built.

The `shared/` layer provides reusable infrastructure so week code stays focused on pedagogy rather than data plumbing. The data cache covers OHLCV prices (456 tickers), Fama-French 3/5/6 + Carhart factors, Ken French sorted portfolios, FRED macro series (24 series including full yield curve, VIX, credit spreads), fundamentals (455 tickers), and crypto (8 tokens).

## Setup

Requires Python 3.12. Dependencies managed with [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Notebooks download data on first run via `course/shared/data.py`, which caches everything locally.

---

## Appendix A: The Build Pipeline

Each week is built through a 7-step pipeline. Every step produces one immutable artifact — nothing flows backward.

| Step | Agent | Produces | What it does |
|------|-------|----------|--------------|
| 1 | Research | `research_notes.md` | Gathers current domain knowledge: papers, tools, libraries, best practices |
| 2 | Blueprint | `blueprint.md` | Designs the ideal teaching week: narrative arc, sections, exercises, career connections |
| 3 | Expectations | `expectations.md` | Assesses data reality and sets acceptance criteria for each code file |
| 4 | Code | `code/` + `run_log.txt` | Plans, implements, and verifies all Python files (lecture, seminar, homework) |
| 5 | Observation | `observations.md` | Fresh-eyes review of plots and numerical output in a clean context |
| 6 | Consolidation | `narrative_brief.md` | Reconciles the ideal plan (blueprint) with actual results (observations) |
| 7 | Notebooks | `.ipynb` files | Converts code + narrative into three final notebooks (lecture, seminar, hw) |

Approval gates sit between each step. In supervised mode, a human reviews each artifact before the pipeline advances. In autonomous mode, the orchestrator evaluates gate criteria and proceeds unless it hits a hard stop.

Step 4 (code) is the most complex — it runs in three sub-phases: a planning pass that maps blueprint sections to Python files, parallel file agents that implement each file independently, and a verification pass that runs every file sequentially to catch cross-file issues.

Step 6 also has two sub-steps that run after the main consolidation: flag resolution (fixing issues without full reruns) and a brief audit (adversarial review checking for inflated claims or unjustified conclusions).

The pipeline is orchestrated from a single Claude Code session that launches specialized Task agents for each step. Each agent reads at most 2-3 guide files — keeping context focused and preventing the instruction drift that plagued earlier approaches.

## Appendix B: How the Course Outline Was Researched

The course outline was built from a structured research effort designed to answer one question: *what should an ML/DL expert learn to break into quantitative finance?*

**Methodology.** Five specialized research agents each investigated a different slice of the landscape:

1. **Quant fund jobs** — 45-50 job listings across 23 firms (Jane Street, Citadel, Two Sigma, D.E. Shaw, HRT, XTX, G-Research, etc.). Extracted required vs. preferred skills, day-to-day responsibilities, asset class mentions, tools named.
2. **Bank/AM roles** — Sell-side banks (Goldman, JPMorgan, Morgan Stanley), asset managers (BlackRock, MSCI, Vanguard), and vendors (Bloomberg). Mapped how these roles differ from buy-side.
3. **MFE syllabi & certifications** — 13 degree programs (CMU, Princeton, Baruch, NYU, Columbia, Stanford, MIT, Oxford, Imperial, ETH, UCL, Berkeley, Cornell) + 4 certifications (CQF, FRM, CFA, CAIA) + 6 online programs. Identified what academia considers essential.
4. **Community/practitioner views** — Reddit r/quant, QuantNet, Wilmott forums, practitioner blogs (Kris Abdelmessih, Cliff Asness/AQR, Lopez de Prado), 15-20 recent books. Captured what practitioners say actually matters vs. what job listings claim.
5. **Emerging trends** — Conference programs (ICAIF, NeurIPS/ICML finance workshops, QuantMinds), frontier research, industry reports. Identified what's genuinely new in 2025-2026 vs. what's hype.

**Synthesis.** The five raw reports (~170-200 sources total) were synthesized into a single reference document that classifies every discovered topic into tiers: MUST KNOW (80%+ of roles), SHOULD KNOW (40-80%), GOOD TO KNOW (15-40%), NICHE (<15%), or DECLINING. This classification drove the band structure — Band 1 covers all MUST KNOW topics, Band 2 reaches all SHOULD KNOW, Band 3 addresses GOOD TO KNOW.

**Key finding that shaped the restructure.** The research showed that top firms (Jane Street, Two Sigma, RenTech, G-Research, D.E. Shaw) explicitly do NOT require finance knowledge — they teach domain on the job but can't teach mathematical maturity or research methodology. This meant the course should build frameworks and intuition, not attempt encyclopedic coverage. The old structure's worst mistake was the opposite: spending weeks on ML architecture teaching (which the audience already knows) while leaving derivatives, fixed income, execution, and causal inference completely uncovered.

The full synthesis and all five raw research reports are in [`course/research/`](course/research/).

## Appendix C: Project History

This course was created entirely by one person using Claude (Anthropic's AI) as the primary authoring tool. The process went through three distinct generations.

### Generation 1: First Drafts

The initial 18-week structure was drafted directly as Jupyter notebooks — one lecture, seminar, and homework per week. These first drafts covered weeks 1-18 but with a different topic ordering that front-loaded ML architecture teaching (feedforward nets in week 7, LSTMs in week 8, GNNs in week 12) rather than financial domain knowledge. The old structure had major gaps: no derivatives pricing, no fixed income, no execution algorithms, no stat arb, no causal inference. Meanwhile it spent full weeks teaching neural network fundamentals to an audience assumed to already know them.

### Generation 2: Old Pipeline + Restructure

A structured build pipeline was introduced (research, code verification, notebook creation) with 6 guide files (~3,000 lines). Weeks 3-4 were partially rebuilt under this system. A quality audit of weeks 1-4 identified systemic issues: insufficient workspace cells, code cells exceeding line limits, voice inconsistencies, and content overlap between notebooks. Worst of all was the fact that code cells weren't viable - code didn't run, guidelines weren't followed, etc.

Separately, the course outline was restructured from scratch (see Appendix B). The synthesis identified that the old structure's biggest problem was teaching ML fundamentals to ML experts while skipping essential financial topics. The new 20-week outline (18 core + 2 optional) reorganized content into four career-priority bands, front-loading industry relevance.

### Generation 3: Current Pipeline (v2)

The old 6-file guide system was replaced with the 7-step pipeline described in Appendix A. The redesign addressed specific failures: massive duplication across guides (~80% restated content), broken encapsulation (notebook instructions inside the code verification guide), no consolidation step to reconcile planned vs. actual results, and unreliable plot verification in long contexts.

Weeks 1-2 were built under an earlier variant of this process. Weeks 3-5 were built under the full v2 pipeline. Weeks 6-18 are scaffolded but not yet built.
