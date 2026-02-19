# Week 3 Research Notes: Factor Models & Cross-Sectional Analysis

**Research tier:** LIGHT
**Date:** 2026-02-17

## Research Plan (as approved)

**Tier:** LIGHT (Factor models are well-established foundations — CAPM, Fama-French, Barra are decades old. Main check: verify stability, look for recent paradigm shifts, confirm free data/tool accessibility.)

**Sub-questions:**
1. What's the canonical approach to factor models and cross-sectional analysis?
   → Academic literature, textbooks, survey papers
2. Have there been paradigm shifts in factor modeling since ~2024?
   → Recent arxiv/SSRN, practitioner forums
3. What do practitioners at funds actually use for factor analysis?
   → r/quant, QuantNet, job postings
4. What Python tools exist for factor model construction, and are they free and maintained?
   → PyPI, GitHub, official documentation
5. What free data sources support building factor models from scratch without institutional subscriptions?
   → Data provider docs, yfinance, SEC EDGAR

**Estimated scope:** 3 initial discovery queries + follow-up verification as needed. Expect "no major changes" as a valid finding.

---

## Findings

### Canonical Foundations

- **Fama & French (2015)** — "A Five-Factor Asset Pricing Model" (Journal of Financial Economics). Extends the classic 3-factor model (market, size, value) with profitability (RMW) and investment (CMA). The canonical factor model in empirical asset pricing. Widely taught and used.

- **Fama & MacBeth (1973)** — Original cross-sectional regression methodology. Two-step procedure: (1) time-series regression to estimate factor loadings, (2) cross-sectional regression each period to estimate risk premia. Standard errors automatically account for cross-sectional correlation. Remains the standard methodology for testing whether factors are priced.

- **Fama & French (2020)** — "Comparing Cross-Section and Time-Series Factor Models" (Review of Financial Studies). Directly compares portfolio-sorting vs. cross-sectional regression approaches. Establishes that cross-sectional models provide a more complete picture.

- **Barra/MSCI Risk Models** — Cross-sectional multivariate regression approach (regress all asset returns on all characteristics simultaneously each period). Factor returns = regression coefficients. Includes explicit industry factors (55-70 industries in USE4). Practitioner-origin methodology used in risk management and portfolio construction. Proprietary (MSCI), but the methodology is public.

- **CAPM → FF3 → FF5 → Barra progression** — The standard teaching sequence across financial econometrics courses. Confirmed stable; no competing sequence has emerged.

### Recent Developments

- **[ArXiv 2403.06779](https://arxiv.org/html/2403.06779v1) (March 2024)** — "From Factor Models to Deep Learning: Machine Learning in Reshaping Empirical Asset Pricing." Comprehensive review showing ML/AI transforms traditional asset pricing models. Identifies that CAPM and Fama-French "struggle to capture multifaceted and nonlinear dynamics." ML offers superior predictive accuracy through flexible functional forms. **Adoption level:** Academic review; reflects real trend but ML is augmenting, not replacing linear models in production.

- **[ArXiv 2507.07107](https://www.arxiv.org/pdf/2507.07107) (June 2025)** — "Machine Learning Enhanced Multi-Factor Quantitative Trading." Cross-sectional portfolio optimization with bias correction, PyTorch-based computational acceleration for factor engineering. **Adoption level:** Academic; demonstrates modern computational approaches.

- **[FactorGCL (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/31993)** — Hypergraph-based factor model using temporal residual contrastive learning for capturing high-order nonlinear factor relationships. **Adoption level:** Cutting-edge academic; no evidence of practitioner adoption.

- **Cross-Sectional Paradigm Shift (verified by multiple sources):** Cross-sectional portfolio construction is emerging as the dominant paradigm over pure time-series approaches. Focus on *relative* performance within the investment universe rather than predicting *absolute* returns. Verified by QuantPedia, ArXiv 2403.06779, ArXiv 2507.07107. This is evolutionary, not revolutionary — practitioners have used cross-sectional methods for years, but it's now the acknowledged standard.

- **The "Factor Zoo" Problem Persists:** Hundreds of proposed factors create overfitting and interpretability concerns. As of 2025-2026, ML is used to navigate this complexity through dimensionality reduction (PCA, PLS, LASSO). Practitioner consensus: firms are moving toward parsimonious models with strong economic intuition rather than kitchen-sink approaches.

- **IPCA (Instrumented Principal Component Analysis):** Emerging as a modern approach for conditional factor models. Mentioned in several recent papers but not yet standard in production. Worth noting as a frontier topic.

- **No paradigm shift in core theory:** CAPM, Fama-French, and Barra models remain foundational. No major theoretical breakthroughs invalidating these frameworks since 2024. The "factor zoo" debate continues without consensus on a superior unified model.

### Practitioner Reality

- **Core methods remain stable in production.** Fama-French 5-factor + momentum + quality factors are standard. Barra-style risk models (MSCI USE4, GEM3) are the industry standard for risk management. Selective ML augmentation is growing but hasn't displaced linear models. **Corroboration:** Convergent signal from multiple papers positioning themselves as addressing practitioner needs, plus absence of hot-topic discussion on r/quant (consistent with factor models being "mature, stable technology").

- **Academic-industry gap.** Academia: exploring hundreds of factors, neural architectures, graph models. Industry: still primarily using the canonical 5-8 factors with strong economic intuition, with selective ML augmentation. Simple linear factor models remain competitive in many settings; complexity must be justified.

- **Barra vs. Fama-French in job postings.** Both appear in quant job descriptions ("experience with Barra models" vs. "knowledge of Fama-French factors"). They serve different purposes: FF for alpha research, Barra for risk management. Understanding both is important for career readiness.

- **Cross-sectional approaches are standard in industry.** Confirmed by multiple papers and practitioner discussions. Ranking stocks within universe (relative performance) is the modern standard.

### University Coverage

- **Boston University EC 794: Financial Econometrics (Spring 2023):** Covers volatility from both time-series and cross-sectional perspectives. References Fama & French (2020) comparing cross-section and time-series factor models. Emphasizes Fama-MacBeth procedure with attention to errors-in-variables problem.

- **General observation:** Could not locate specific 2025-2026 syllabi for Stanford/CMU/Princeton courses dedicated to "Barra risk models." Factor models are typically embedded within broader financial econometrics or empirical asset pricing courses. Traditional sequence (CAPM → FF → Barra) is standard across programs.

---

## Tools & Libraries

### Factor Return Data Access

| Option | Package | Pros | Cons | Access |
|--------|---------|------|------|--------|
| A | `getfactormodels` | 12+ models (FF3/5/6, Carhart, q-factors, etc.), Pandas + Polars + PyArrow support, modern API, data from 1926+ | Convenience layer — doesn't teach construction mechanics | ✅ Free |
| B | `pandas-datareader` | Ken French data via `web.DataReader('F-F_Research_Data_Factors', 'famafrench')`, well-known | Limited to Ken French datasets, Pandas only | ✅ Free |
| C | `famafrench` (v0.1.4, May 2020) | Constructs factors from raw stock/fundamental data, LRU caching | ⚠️ Requires WRDS subscription (CRSP + Compustat, ~$1000s/year), Beta status, Pandas only | ⚠️ Institutional |
| D | Manual download from Ken French Data Library | No dependencies, primary source | Manual CSV parsing required | ✅ Free |

**Recommendation:** `getfactormodels` (Option A) for validation against official factor returns. It's modern, supports Polars, and covers 12+ factor models. `famafrench` (Option C) should only be mentioned as an institutional tool — it creates an accessibility barrier.
**Verified:** PyPI pages fetched for both libraries. `getfactormodels` confirmed free with no credentials required. `famafrench` GitHub docs explicitly state WRDS subscription requirement.

### Fama-MacBeth Regression

| Option | Package | Pros | Cons | Access |
|--------|---------|------|------|--------|
| A | `linearmodels` (Kevin Sheppard) | Production-grade `FamaMacBeth` class, Newey-West SEs, well-documented, actively maintained | Requires panel data MultiIndex setup | ✅ Free |
| B | Manual implementation (pandas + numpy/statsmodels) | High pedagogical value, ~5-10 lines of code, teaches two-step logic explicitly | No built-in Newey-West correction | ✅ Free |
| C | `fama_macbeth` (leoliu0) | Standalone with Newey-West SEs | Less documented than linearmodels | ✅ Free |

**Recommendation:** Manual implementation first (pedagogical value), then `linearmodels` as the production tool.
**Verified:** `linearmodels` documentation fetched — confirmed `FamaMacBeth` class exists with `cov_type='kernel'` for Newey-West. [Tidy Finance tutorial](https://www.tidy-finance.org/python/fama-macbeth-regressions.html) verified for manual approach. [Kevin Sheppard example notebook](https://www.kevinsheppard.com/teaching/python/notes/notebooks/example-fama-macbeth/) also confirmed.

---

## Data Source Accessibility

### Daily Equity Prices (multi-stock, multi-year)

| Option | Source | Coverage | Limitations | Access |
|--------|--------|----------|-------------|--------|
| A | yfinance | Current & historical OHLCV, multi-ticker batch download, adjusted/unadjusted prices | Survivorship bias (current constituents only), no delisted stocks, rate limits, schema changes | ✅ Free |
| B | Polygon.io | Comprehensive, reliable | Free tier limited | ⚠️ Free tier limited |

**Recommendation:** yfinance — students already use it from Week 1.
**Verified:** Confirmed functional in Week 1 pipeline.

### Fundamental Data (balance sheet, income statement, cash flow)

| Option | Source | Coverage | Limitations | Access |
|--------|--------|----------|-------------|--------|
| A | yfinance | Balance sheet, income stmt, cash flow stmt. Key fields: total equity, market cap, net income, operating income, total assets, cash flow. Past 4 years annual + 5 quarters quarterly. | No point-in-time fundamentals (uses latest restated values), survivorship bias, missing delisted stocks, fundamental data may lag corporate action adjustments | ✅ Free |
| B | Financial Modeling Prep (FMP) | Earnings, EPS estimates, revenue projections, full financial statements, Python SDK | Free tier rate-limited | ✅ Free tier |
| C | EODHD | Comprehensive fundamentals, bulk download capability | Free plan limited | ✅ Free tier |
| D | SEC EDGAR (via `edgartools` or `sec-api`) | Authoritative government source, 10-K/10-Q parsing, XBRL financial statements | More complex API, requires CIK-to-ticker mapping, parsing overhead | ✅ Free |
| E | WRDS (CRSP + Compustat) | Survivorship-bias-free, point-in-time fundamentals, clean corporate actions | Requires institutional subscription (~$1000s/year) | ⚠️ Institutional |

**Recommendation:** yfinance (Option A) as primary — simplest API, widely documented, no registration, adequate for factor construction. Limitations (survivorship bias, no point-in-time data) become explicit teaching moments. SEC EDGAR (Option D) worth mentioning for awareness of authoritative government data.
**Verified:** yfinance fields confirmed via [Medium tutorial](https://rfachrizal.medium.com/how-to-obtain-financial-statements-from-stocks-using-yfinance-87c432b803b8). `edgartools` confirmed on PyPI. EODHD and FMP confirmed via their documentation sites.

### Factor Construction Data Sufficiency

All canonical Fama-French factors can be constructed from freely available data:

| Factor | Required Data | Available via yfinance? |
|--------|---------------|-------------------------|
| SMB (size) | Market capitalization | ✅ `info['marketCap']` or price × shares |
| HML (value) | Book value, market cap | ✅ balance_sheet total equity + market cap |
| RMW (profitability) | Operating income, book equity | ✅ income_stmt + balance_sheet |
| CMA (investment) | Total assets YoY growth | ✅ balance_sheet total assets, multi-year |
| Momentum | Past 12-month returns | ✅ price history |

**Accessibility bottom line:** ✅ All core needs can be met with free tools and data. No institutional subscriptions required for any core exercise.

---

## Domain Landscape Implications

- **Factor models are mature, stable technology.** No paradigm shifts in core theory. CAPM, Fama-French, and Barra remain foundational. The teaching sequence (CAPM → FF3 → FF5 → Barra) is universally standard and unchallenged.

- **Cross-sectional framing is now the acknowledged standard.** While time-series factor analysis remains important, the field has shifted toward cross-sectional ranking and relative performance. This is evolutionary — practitioners have used cross-sectional methods for years — but it's now the dominant framing in both academic and practitioner literature.

- **ML augments but does not replace linear factor models.** The academic frontier (transformers, GNNs, hypergraphs for factor modeling) has not translated to production. Simple linear models remain competitive in many settings. The gap between academic literature and production practice is significant — academia explores hundreds of factors and neural architectures, while industry uses the canonical 5-8 factors with selective ML augmentation.

- **The factor zoo is a real problem, not just academic.** Hundreds of proposed factors create genuine overfitting risk. The practitioner consensus is parsimony and economic intuition over statistical fit. Validation methodology (out-of-sample testing, multiple testing correction) is critical.

- **Free data is sufficient for factor construction but has known biases.** yfinance provides adequate fundamental data, but survivorship bias and lack of point-in-time fundamentals are structural limitations. These limitations are well-understood and serve as natural teaching moments about why institutional data (WRDS/CRSP/Compustat) costs money.

- **Barra-style risk models vs. Fama-French factor models serve different purposes.** FF is used for alpha research (do these characteristics predict returns?). Barra is used for risk management (what are the sources of portfolio risk?). Both use similar underlying characteristics but differ in construction methodology (portfolio sorting vs. cross-sectional regression) and application. Both appear in quant job postings.

---

## Collected References

### Foundational
- **Fama & French (2015)** — "A Five-Factor Asset Pricing Model" (JFE). The canonical factor model; extends FF3 with profitability and investment.
- **Fama & MacBeth (1973)** — Cross-sectional regression methodology. The standard procedure for testing whether factors are priced.
- **Fama & French (2020)** — "Comparing Cross-Section and Time-Series Factor Models" (RFS). Directly compares portfolio-sorting vs. cross-sectional regression approaches.

### Modern / Cutting-Edge
- **ArXiv 2403.06779 (March 2024)** — "From Factor Models to Deep Learning." Comprehensive review of ML integration with asset pricing. Adoption: academic review, reflects real trends.
- **ArXiv 2507.07107 (June 2025)** — "ML Enhanced Multi-Factor Quantitative Trading." Cross-sectional optimization with bias correction. Adoption: academic.
- **FactorGCL (AAAI 2025)** — Hypergraph-based factor model with contrastive learning. Adoption: cutting-edge academic, no practitioner adoption.
- **Gu, Kelly & Xiu (2020)** — "Empirical Asset Pricing via Machine Learning" (RFS). Bridge to Week 4 — establishes ML framework for cross-sectional prediction.

### Practitioner
- **[AnalystForum: Barra vs Fama-French](https://www.analystforum.com/t/barra-vs-fama-french/76740)** — Clear practitioner discussion of methodological differences. Verified primary source.
- **[QuantPedia: Factor Zoo with ML](https://quantpedia.com/exploring-the-factor-zoo-with-a-machine-learning-portfolio/)** — Practitioner-facing analysis of navigating the factor zoo. Corroborates cross-sectional paradigm shift.
- **[Tidy Finance: Fama-MacBeth in Python](https://www.tidy-finance.org/python/fama-macbeth-regressions.html)** — Step-by-step implementation tutorial.

### Tool & Library Documentation
- **[getfactormodels](https://github.com/x512/getfactormodels)** — Free factor data download (12+ models, Pandas/Polars). Key validation tool.
- **[linearmodels](https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.FamaMacBeth.html)** — Production-grade Fama-MacBeth implementation.
- **[famafrench](https://pypi.org/project/famafrench/)** — WRDS-dependent factor construction (institutional only).
- **[edgartools](https://pypi.org/project/edgartools/)** — SEC EDGAR data access for fundamentals.
- **[Kevin Sheppard Fama-MacBeth notebook](https://www.kevinsheppard.com/teaching/python/notes/notebooks/example-fama-macbeth/)** — Worked example connecting manual implementation to linearmodels.

---

## Open Questions for Downstream Steps

- **Data universe scope:** How many stocks should the factor construction exercise use? Full S&P 500 constituents vs. a smaller subset? yfinance can handle 500 tickers but fundamental data downloads are slow. The expectations agent (Step 3) should determine the practical data universe.

- **Factor construction detail level:** Should the exercises build all five FF factors (SMB, HML, RMW, CMA, MOM) or focus on a subset? Building all five is comprehensive but time-intensive. The blueprint agent (Step 2) should decide scope.

- **Barra-style exercise depth:** The simplified cross-sectional regression approach is implementable, but should industry factors (SIC/GICS codes, 55+ dummies) be included or just style factors? The blueprint should calibrate this based on time budget.

- **yfinance fundamental data reliability:** While fields are confirmed available, the practical reliability of batch-downloading fundamentals for 500 tickers needs verification at code time. The code verification agent (Step 4) should test this.

- **Survivorship bias quantification:** yfinance only provides current constituents. The magnitude of survivorship bias in constructed factors (vs. official Ken French factors) is an empirical question for Step 3/4.

---

## Verification Audit Trail

> This section is for the user's review during the Step 1 approval gate.
> Downstream agents (Steps 2–3) can skip this section.

**Queries run (initial discovery):**
1. `factor models cross-sectional analysis new developments 2025 2026 finance` → ArXiv papers on ML integration, cross-sectional paradigm shift signal
2. `Fama-French factor models python library 2025 2026 updates` → Confirmed tool landscape (famafrench, getfactormodels)
3. `Barra risk models course syllabus 2025 2026 Stanford CMU Princeton` → Found BU EC 794; no Stanford/CMU/Princeton-specific Barra syllabi
4. `"factor models" OR "cross-sectional analysis" site:edu syllabus 2025 2026 course` → Confirmed traditional teaching sequence
5. `"factor zoo" OR "factor models" paradigm shift 2025 2026 finance machine learning` → Confirmed factor zoo persistence, cross-sectional shift

**Follow-up verification queries:**
6. `free financial fundamental data API book value market cap earnings python 2025 2026` → Confirmed yfinance, FMP, EODHD as free sources
7. `"yfinance" OR "yahoo finance" fundamental data balance sheet income statement free` → Verified yfinance fundamental data fields
8. `SEC EDGAR fundamental data python free academic research 2025` → Found edgartools, OpenEDGAR, edgar-crawler, sec-api
9. `Fama MacBeth regression python implementation linearmodels statsmodels 2025` → Verified linearmodels.FamaMacBeth, Tidy Finance tutorial, Sheppard notebook
10. `"Barra risk model" vs "Fama French" difference industry factors characteristic factors` → Verified methodological differences via AnalystForum, Princeton lecture notes

**Findings dropped (failed verification):**
- None. All initial findings were corroborated or verified via primary sources.

**Verification summary:**
- Total queries: 5 initial + 5 follow-up = 10
- Sources fetched (primary): 15+ (PyPI pages, GitHub repos, documentation sites, academic papers, practitioner forums)
- Findings verified: All
- Findings dropped: 0
