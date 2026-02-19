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

## Setup

Requires Python 3.12. Dependencies managed with [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Notebooks download data on first run via the shared data layer (`course/shared/data.py`), which caches ~143 MB across 48 datasets (OHLCV prices, Fama-French factors, FRED macro series, fundamentals, crypto).

---

## Appendix: How This Course Was Built

This course was created entirely by one person using Claude (Anthropic's AI) as the primary authoring tool. The process went through three distinct generations.

### Generation 1: First Drafts

The initial 18-week structure was drafted directly as Jupyter notebooks — one lecture, seminar, and homework per week. These first drafts covered weeks 1-18 but with a different topic ordering that front-loaded ML architecture teaching (feedforward nets in week 7, LSTMs in week 8, GNNs in week 12) rather than financial domain knowledge. The old structure had major gaps: no derivatives pricing, no fixed income, no execution algorithms, no stat arb, no causal inference. Meanwhile it spent full weeks teaching neural network fundamentals to an audience assumed to already know them.

### Generation 2: Old Pipeline + Restructure

A structured build pipeline was introduced (research, code verification, notebook creation) with 6 guide files (~3,000 lines). Weeks 3-4 were partially rebuilt under this system. A quality audit of weeks 1-4 identified systemic issues: insufficient workspace cells, code cells exceeding line limits, voice inconsistencies, and content overlap between notebooks.

Separately, the course outline was restructured from scratch. Five specialized research agents investigated: quant fund job requirements, bank/AM roles, MFE syllabi, practitioner community views, and emerging trends. The synthesis identified that the old structure's biggest problem was teaching ML fundamentals to ML experts while skipping essential financial topics. The new 20-week outline (18 core + 2 optional) reorganized content into four career-priority bands, front-loading industry relevance.

### Generation 3: Current Pipeline (v2)

The old 6-file guide system was replaced with a 7-step pipeline where each step produces one immutable artifact. The redesign addressed specific failures: massive duplication across guides (~80% restated content), broken encapsulation (notebook instructions inside the code verification guide), no consolidation step to reconcile planned vs. actual results, and unreliable plot verification in long contexts. The current pipeline: research, blueprint, expectations, code, observation review, consolidation, notebooks — with approval gates between steps and a shared data infrastructure layer caching 48 datasets.

Weeks 1-2 were built under an earlier variant of this process. Weeks 3-5 were built under the full v2 pipeline. Weeks 6-18 are scaffolded but not yet built.
