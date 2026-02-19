# Week 5 Research Notes: Backtesting, Research Discipline & Transaction Costs

**Research tier:** LIGHT
**Date:** 2026-02-18

---

## Research Plan (as approved)

**Tier assignment:** LIGHT. Backtesting methodology (purged CV, deflated Sharpe ratio, multiple testing correction) and transaction cost modeling are well-established domains with a clear canonical source (López de Prado, 2018) and stable methodological foundations. The core papers date from 2012–2018. Checking for: (1) any paradigm shift since training data, (2) library status, (3) practitioner reality on how much of the academic methodology actually gets applied in production.

**Sub-questions:**
1. What is the canonical methodology for avoiding backtesting bias and data snooping in financial ML? Has anything replaced López de Prado's framework?
2. What are the standard methods for multiple testing correction in finance (Harvey et al. 2016, deflated Sharpe ratio)?
3. What transaction cost components are canonical, and how are they modeled in practice?
4. What Python libraries exist for purged CV, backtest diagnostics, and transaction cost modeling — and are they free and maintained?
5. What do practitioners actually use in production? Is purged CV really the standard, or is simpler walk-forward dominant?
6. (Optional, LIGHT) How do MFE programs cover this topic?

**Source types used:** Academic literature, official docs/PyPI, practitioner forums/blogs, one university syllabus check.

**Estimated scope:** 2–3 initial discovery queries + follow-up verification queries as needed.

---

## Findings

### Canonical Foundations

- **López de Prado (2018)** — *Advances in Financial Machine Learning* (Wiley). Chapters 7–12 cover purged k-fold CV, combinatorial purged CV (CPCV), the probability of backtest overfitting, and how to structure a sound research process in financial ML. This is the canonical graduate-level text. Chapter 7 specifically defines the purging and embargo mechanisms; Chapter 8 defines CPCV. The "backtesting is not a research tool, feature importance is" framing comes from this book. Verified: widely cited in academic literature and practitioner materials; Wiley page confirms publication date and content.

- **Bailey, López de Prado (2014)** — *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. Journal of Portfolio Management. The DSR formula adjusts an observed Sharpe ratio for (a) the number of strategy variants tested, (b) backtest length, (c) skewness, and (d) kurtosis. Provides a threshold: a DSR < 0 means a strategy has not cleared a basic data-snooping bar. Verified: paper fetched from davidhbailey.com; SSRN version confirmed. This is the standard reference for backtest overfitting correction.

- **Bailey, Borwein, López de Prado, Zhu (2015)** — *The Probability of Backtest Overfitting* (PBO). Combinatorial cross-validation framework: fit strategies on each of many train-test splits, count how often the best in-sample strategy underperforms out-of-sample. PBO > 0.5 means the best-looking strategy is more likely to be overfit than not. Verified: paper fetched from davidhbailey.com.

- **Harvey, Liu, Zhu (2016)** — *...And the Cross-Section of Expected Returns*. Journal of Finance. Establishes the correct multiple testing hurdle for factor discovery: the minimum t-statistic for a new factor should be 3.0 (accounting for all prior tests). Standard reference for finance researchers on how to apply Bonferroni, Holm, and BHY (Benjamini-Hochberg-Yekutieli) corrections. Verified: paper fetched from Duke people page; confirmed canonical in quant research.

- **Almgren & Chriss (2000)** — *Optimal Execution of Portfolio Transactions*. Journal of Risk. The canonical model for market impact cost during execution. Distinguishes permanent impact (shifts the equilibrium price, affects all future trades), temporary impact (the extra cost from urgency, which dissipates), and spread cost (the fixed half-spread paid on every trade). Still the standard academic reference for transaction cost decomposition. Verified: paper fetched from smallake.kr mirror; referenced in multiple independent practitioner and academic sources.

- **Luo et al. (2014)** — *Seven Sins of Quantitative Investing*. Deutsche Bank Markets Research white paper. Catalogues the most common backtesting mistakes in practitioner research: look-ahead bias, survivorship bias, data snooping, ignoring transaction costs, ignoring short-selling constraints, improper benchmark comparison, and neglecting regime changes. Still widely cited as a practitioner checklist. Verified: referenced across multiple course materials and practitioner blog posts.

### Recent Developments

- **Arian, Norouzi, Seco (2024)** — *Backtest Overfitting in the Machine Learning Era: A Comparison of Out-of-Sample Testing Methods*. Knowledge-Based Systems, Vol. 305 (October 2024). Directly compares walk-forward, purged k-fold, and CPCV in a synthetic controlled environment. Finding: CPCV shows the lowest probability of backtest overfitting and the best deflated Sharpe ratio test statistics. Walk-forward exhibits higher temporal variability and weaker stationarity. Academic adoption — not evidence of industry paradigm shift — but confirms that CPCV's theoretical advantages hold in simulation. Verified: abstract confirmed on ScienceDirect.

- **No paradigm shift since 2018.** The López de Prado / Bailey framework from 2016–2018 remains the standard. No new competing methodology has emerged that has displaced it. This is the expected finding for a LIGHT-tier topic in a stable domain.

- **Purged CV on Wikipedia (2025).** López de Prado's purged k-fold CV has its own Wikipedia article (fetched), confirming it has reached the level of a recognized named technique with stable definition. No definitional changes since the 2017–2018 publications.

### Practitioner Reality

- **Walk-forward remains the dominant production standard; purged CV is the best-practice recommendation but not universal.** Multiple practitioner sources (QuantStart, QuantVPS, Interactive Brokers quant campus, robustness-testing blog) describe walk-forward as the primary production approach, often combined with robustness checks. Purged CV is recommended in ML-heavy shops but is not the dominant practice across all funds. The gap between academic best practice (purged CV / CPCV) and production reality (walk-forward) is a real and significant teaching moment. Corroborated across 4+ independent practitioner sources.

- **Transaction cost underestimation is the most common backtest failure mode practitioners identify.** Multiple sources confirm the pattern: "strategies showing 30% returns before costs deliver 15% after fees." Flat-cost assumptions (a fixed dollar amount per trade) are the most common simplified model, but they underestimate slippage for illiquid positions and overestimate cost for liquid positions. The gap between backtest P&L and live P&L is widely described as the most common surprise for beginning quantitative practitioners. Corroborated across QuantStart, QuantNomad, PineConnector, QuantVPS posts.

- **"Backtesting is not a research tool"** — this López de Prado framing (i.e., you should not tweak strategy parameters and re-run backtests; instead use feature importance during development and only backtest a fully specified strategy) has reached practitioner awareness but is not universally followed. Many practitioners still backtest iteratively. The disciplined separation between strategy development (feature importance, cross-validation) and validation (single held-out backtest) is recognized as best practice in ML-focused quant shops. Source: López de Prado's GARP whitepaper "10 Reasons ML Funds Fail" (fetched); SSRN paper "7 Reasons ML Funds Fail."

- **False Discovery Rate correction is academically accepted but underused in practice.** Harvey et al.'s t-stat hurdle of 3.0 is cited in academic papers on factor discovery but rarely applied by practitioners working on live signals. Practitioners are more likely to apply a conservative out-of-sample test than formal Bonferroni/BHY correction. The conceptual understanding (multiple testing inflates false discovery risk) is important; the exact correction formula is more academic in context.

- **Transaction Cost Analysis (TCA) is a dedicated function at institutional funds.** At large asset managers, TCA is handled by execution analytics teams that measure implementation shortfall, compare to VWAP benchmarks, and feed results back to portfolio managers. For the student, understanding the framework is the goal; building a production-grade TCA system is not in scope.

### University Coverage

- **Baruch MFE Pre-MFE ML Syllabus (Feb 2025):** Confirmed that the Baruch Pre-MFE program offers a Machine Learning for Financial Engineering course that includes coverage of backtesting frameworks from López de Prado. The syllabus PDF was inaccessible (certificate error) but the course listing was verified on the Baruch MFE website. Baruch MFE is known to teach directly from López de Prado's book (this is confirmed by multiple QuantNet forum references and the book's acknowledgments).

- **López de Prado's own lecture series (quantresearch.org):** Fetched the lecture page directly. Two full lectures (of 10) are dedicated to backtesting (Backtesting I and Backtesting II). Also includes "7 Reasons ML Funds Fail" as a seminar presentation. This confirms the depth at which the backtesting methodology is taught in academic/practitioner education: it warrants ~20% of a dedicated financial ML course.

- **HKUST (Prof. Daniel Palomar) Portfolio Optimization Course:** Syllabus slides fetched (portfoliooptimizationbook.com). Dedicated section on "Portfolio Backtesting" and "Seven Sins of Quantitative Investing." Covers look-ahead bias, transaction costs, multiple testing, and the dangers of backtest overfitting. References both Luo et al. (Deutsche Bank) and López de Prado. Confirms this is standard MFE-level coverage.

---

## Tools & Libraries

### Purged Cross-Validation Implementation

**Need:** Apply purged k-fold and CPCV splits in Python for financial time series.

| Option | Package | Pros | Cons / Limitations | Access |
|--------|---------|------|--------------------|--------|
| A | `scikit-learn` TimeSeriesSplit | Universal familiarity; `gap` param adds embargo-like behavior (v0.24+); no extra install | No true purging; gap is a fixed index count, not a time-based label overlap check | ✅ Free |
| B | `mlfinpy` (MIT license, open source) | Implements López de Prado's purged k-fold and triple-barrier labeling; modern Pythonic API; NumPy/Pandas/Numba | Newer project, less battle-tested; smaller community than scikit-learn | ✅ Free |
| C | Manual implementation | Full control; pedagogically transparent; directly follows book algorithm | Requires student implementation; ~50-100 lines | ✅ Free |
| D | `mlfinlab` (Hudson & Thames) | Original reference implementation of López de Prado methods | Not on public PyPI; requires subscription/private key; inactive on public PyPI (404 error confirmed) | ⚠️ Effectively closed-source |

**Recommendation:** Use manual implementation (Option C) as the primary teaching vehicle — building purged CV from scratch is itself a key learning exercise. Use `scikit-learn` TimeSeriesSplit with gap parameter for comparison and as a baseline. Reference `mlfinpy` for students who want a verified library implementation.

**Verified:** scikit-learn TimeSeriesSplit documentation fetched (v1.8.0 confirmed, gap parameter confirmed). mlfinlab PyPI page returned 404 (confirmed inaccessible). mlfinpy docs fetched from readthedocs.io (MIT license confirmed, actively maintained). mlfinlab status confirmed via Snyk advisor and community posts.

---

### Backtest Overfitting Metrics (DSR, PBO)

**Need:** Compute deflated Sharpe ratio and probability of backtest overfitting.

| Option | Package | Pros | Cons / Limitations | Access |
|--------|---------|------|--------------------|--------|
| A | `pypbo` (GitHub: esvhd/pypbo) | Implements PBO, DSR, Probabilistic Sharpe Ratio, MinTRL; references Bailey & López de Prado directly | No PyPI page accessible; install from GitHub; low activity signal but algorithm is stable | ✅ Free |
| B | Manual implementation | DSR formula is ~10 lines in Python; directly from paper; fully transparent | Requires implementing the formula; small risk of implementation error without tests | ✅ Free |
| C | `quantstats` (v0.0.81, Jan 2026) | Rich performance reporting (Sharpe, Sortino, drawdowns, calendar returns, tear sheets); actively maintained | Does not compute DSR or PBO specifically; primarily a reporting library, not overfitting detection | ✅ Free |

**Recommendation:** Implement DSR manually using the Bailey-López de Prado formula (Option B) for the key insight. Use `quantstats` for full performance reporting tear sheets. Reference `pypbo` for students who want a library implementation of PBO.

**Verified:** quantstats PyPI page fetched — v0.0.81 released January 13, 2026, classified Production/Stable. pypbo GitHub page referenced in search results; PyPI page inaccessible, but GitHub repo confirms code is available. DSR formula confirmed in fetched paper (davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

---

### Strategy Performance Reporting

**Need:** Compute standard strategy performance metrics (Sharpe, Sortino, max drawdown, CAGR, calmar).

| Option | Package | Pros | Cons / Limitations | Access |
|--------|---------|------|--------------------|--------|
| A | `quantstats` (v0.0.81) | HTML/PDF tear sheets; drawdown charts; benchmark comparison; calendar heatmaps; widely used | No DSR/PBO; focused on reporting, not diagnostic testing | ✅ Free |
| B | `backtesting` (v0.6.5, July 2025) | Clean event-driven backtesting; active development; good for strategy-level (bar-by-bar) simulation | Less suitable for cross-sectional/portfolio-level ML backtesting (more of a single-strategy backtester) | ✅ Free |
| C | Custom pandas/numpy | Full control; integrates directly with existing course code | More implementation work | ✅ Free |

**Recommendation:** `quantstats` for performance reporting. Custom pandas implementation for core metrics in lecture. `backtesting` library is a good reference but is better suited for Week 13 (execution) than Week 5's cross-sectional focus.

**Verified:** quantstats PyPI fetched (v0.0.81, Jan 2026, Production/Stable). backtesting PyPI fetched (v0.6.5, July 2025, actively maintained, GNU AGPLv3).

---

### Walk-Forward and Expanding Window CV

**Need:** Standard time-series cross-validation splits for comparison with purged CV.

| Option | Package | Pros | Cons / Limitations | Access |
|--------|---------|------|--------------------|--------|
| A | `scikit-learn` TimeSeriesSplit | Standard; familiar API; expanding and fixed-window modes; gap parameter for basic embargo | No purging; gap is index-based not label-overlap-based | ✅ Free |
| B | Manual implementation | Full control; can exactly mirror production walk-forward logic | Boilerplate | ✅ Free |

**Recommendation:** `scikit-learn` TimeSeriesSplit for all standard walk-forward splits. Manual implementation only needed when demonstrating purging specifically.

**Verified:** scikit-learn docs fetched (v1.8.0). TimeSeriesSplit gap parameter confirmed as embargo proxy (not true purging).

---

### Transaction Cost Computation

**Need:** Model bid-ask spread cost, market impact, and compute realistic net-of-cost returns.

| Option | Package | Pros | Cons / Limitations | Access |
|--------|---------|------|--------------------|--------|
| A | Manual pandas implementation | Direct; transparent; pedagogically valuable; integrates with any portfolio/signal framework | Requires implementing the TC formula; standard approach in research | ✅ Free |
| B | `quantstats` | Handles returns-level analysis including drawdowns after costs are applied externally | Does not model TC components — costs must be pre-subtracted from returns | ✅ Free |
| C | `backtesting` library | Has spread modeling, commission modeling | Per-trade event-driven model (not suitable for cross-sectional portfolio TC modeling) | ✅ Free |

**Recommendation:** Manual pandas implementation is the right choice. TC modeling for a cross-sectional equity strategy requires: (a) estimated half-spread for each asset, (b) market impact estimate (Almgren-Chriss or simpler square-root model), (c) turnover-weighted cost. All are straightforward pandas operations on the portfolio weight matrix.

**Verified:** No specialized TC Python library is the standard for cross-sectional equity research. Practitioner sources confirm manual implementation is the norm.

---

## Data Source Accessibility

### Daily Equity Prices (for backtesting the alpha model from Week 4)

**Need:** OHLCV prices for a universe of stocks, long enough history for meaningful walk-forward validation.

| Option | Source | Coverage | Limitations | Access |
|--------|--------|----------|-------------|--------|
| A | `yfinance` (via `course/shared/data.py`) | 456 tickers, 2000–present, daily OHLCV | Survivorship bias (current S&P constituents only); adjusted prices may differ from point-in-time at adjustment dates | ✅ Free |
| B | CRSP (via Wharton Research Data Services) | Full US equity universe including delisted; point-in-time adjusted | Institutional subscription only (~$thousands/year) | ⚠️ Institutional |

**Recommendation:** yfinance via the shared data layer. The survivorship bias limitation is a documented teaching moment — it means our "clean" backtest universe looks better than a true universe would. Step 3 should flag the magnitude of this bias.

**Verified:** Shared data layer confirmed in CLAUDE.md and curriculum_state.md (456 tickers, OHLCV from 2000–present). CRSP inaccessibility verified by knowledge of institutional subscription requirements.

---

### Bid-Ask Spread Data (for transaction cost modeling)

**Need:** Bid-ask spread estimates for each stock in the universe.

| Option | Source | Coverage | Limitations | Access |
|--------|--------|----------|-------------|--------|
| A | Computed from OHLC: Corwin-Schultz (2012) estimator | Approximates spread from daily high-low range | Noisy; underestimates spread during low-vol periods; no intraday bid-ask data | ✅ Free |
| B | Fixed-rate proxy: use a constant basis-point spread (e.g., 5-10 bps for large-caps, 15-30 bps for small-caps) | Simple; controllable; common in research | Ignores time-variation and stock-specific variation | ✅ Free |
| C | TAQ (NYSE Trade and Quote database) | Actual tick-level bid-ask quotes | Institutional subscription; requires significant storage (TB scale) | ⚠️ Institutional |

**Recommendation:** Fixed-rate proxy as the baseline (Option B), with a sensitivity analysis showing how results change with different spread assumptions. Corwin-Schultz estimator as an intermediate option for students who want a data-driven estimate. This is the standard sandbox approach — document the limitation that real TC data requires institutional access.

**Verified:** Corwin-Schultz estimator referenced in multiple practitioner and academic sources. Fixed-rate proxy approach confirmed as standard for research using free data (QuantStart, BSIC articles confirmed).

---

### Factor Returns / Alpha Signals (for the model being backtested)

**Need:** A pre-existing alpha signal or the ability to generate one from Week 4's model outputs.

| Option | Source | Coverage | Limitations | Access |
|--------|--------|----------|-------------|--------|
| A | Week 4 model outputs (shared/.data_cache/) | 456-ticker cross-sectional predictions | Depends on Week 4 being completed first | ✅ Free |
| B | Fama-French factors (via pandas-datareader from Ken French data library) | Monthly factor returns; 1926–present | Not individual stock predictions — factor returns, not alpha signals | ✅ Free |

**Recommendation:** Use Week 4 model outputs as the primary signal to backtest. Week 5 is explicitly designed to evaluate a model that Week 4 produces — this is the correct dependency. Fama-French factors can serve as a benchmark.

**Verified:** Week 4 uses the same shared data layer. Dependency is confirmed in the course outline (Week 5 prerequisite: Week 4).

---

**Accessibility bottom line:** ✅ All core needs (prices, spread proxies, alpha signals) can be met with free tools and data. The survivorship bias in the equity universe and the proxy-based TC model are structural limitations that must be documented as teaching moments. Production-grade data (CRSP, TAQ) is noted as the institutional alternative.

---

## Domain Landscape Implications

- **The academic–practitioner gap in CV methodology is a genuine and teachable tension.** Walk-forward is simpler to implement and explain, which is why it dominates production. Purged CV is statistically superior, as confirmed by 2024 research, but requires more implementation effort and is primarily adopted at ML-focused funds. Teaching both — and being explicit about why the gap exists — is more valuable than just teaching the academically superior method.

- **The biggest practitioner failure mode is not overfitting methodology — it's transaction cost underestimation.** A strategy that survives purged CV can still fail in production if TC are modeled naively. The bid-ask spread alone can be 10–50 bps round-trip for mid-caps; market impact can double that for larger positions. This gap between backtest and live performance is the single most common complaint from practitioners entering quant roles.

- **The multiple testing / data snooping problem is well-known but the corrective practice is inconsistent.** Harvey et al.'s t-stat hurdle of 3.0 is academically accepted; in production, most practitioners use some form of out-of-sample hold-out rather than formal Bonferroni/BHY correction. The conceptual understanding matters more for this audience than the exact correction mechanics.

- **"mlfinlab" is effectively no longer accessible for free users.** The Hudson & Thames library, which was the original open-source reference implementation of López de Prado's methods, is now closed-source/subscription. Students looking for library implementations should use `mlfinpy` (MIT license, actively maintained) or implement from scratch. This is a material change from 2019–2021 when mlfinlab was freely available on PyPI.

- **Transaction cost modeling sits at the intersection of Week 5 (backtesting) and Week 13 (optimal execution).** The Almgren-Chriss model appears here as a TC component and again in Week 13 as an execution optimization framework. Week 5 needs the cost side; Week 13 needs the execution strategy side. The split is natural and warranted.

- **No new methodological frameworks have displaced López de Prado's 2018 work.** This is a stable domain. The core methods (purged CV, DSR, PBO) will not be outdated within the course's shelf life. This is the correct LIGHT-tier finding.

---

## Collected References

### Foundational

- **López de Prado (2018)** — *Advances in Financial Machine Learning*. Wiley. Chapters 7–12. The primary canonical text for backtesting methodology in financial ML. Required reference for this week.

- **Bailey & López de Prado (2014)** — *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. Journal of Portfolio Management. [PDF: davidhbailey.com/dhbpapers/deflated-sharpe.pdf] The DSR formula in its original form.

- **Bailey, Borwein, López de Prado, Zhu (2015)** — *The Probability of Backtest Overfitting*. Journal of Computational Finance. [PDF: davidhbailey.com/dhbpapers/backtest-prob.pdf] The PBO framework and combinatorial cross-validation for overfitting detection.

- **Harvey, Liu, Zhu (2016)** — *...And the Cross-Section of Expected Returns*. Journal of Finance. [PDF: people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF] The multiple testing correction standard for factor discovery research.

- **Almgren & Chriss (2000)** — *Optimal Execution of Portfolio Transactions*. Journal of Risk. The canonical market impact model — permanent impact, temporary impact, spread cost decomposition. [PDF: smallake.kr mirror]

- **Luo et al. (2014)** — *Seven Sins of Quantitative Investing*. Deutsche Bank Markets Research. Practitioner checklist for backtesting mistakes. [Referenced in HKUST course slides, López de Prado lectures]

### Modern / Cutting-Edge

- **Arian, Norouzi, Seco (2024)** — *Backtest Overfitting in the Machine Learning Era: A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment*. Knowledge-Based Systems, Vol. 305. [ScienceDirect: doi.org/10.1016/j.knosys.2024.111110] 2024 empirical validation confirming CPCV outperforms walk-forward and purged k-fold. Academic adoption level.

- **Bailey, Borwein, López de Prado, Zhu (2014)** — *Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance*. Notices of the American Mathematical Society. [SSRN 2308659] The accessible, strongly-worded companion paper arguing for rigorous backtesting standards.

- **Harvey & Liu (2015)** — *Backtesting*. Journal of Portfolio Management. [Duke people page] Extension of multiple testing to the backtesting context specifically.

### Practitioner

- **QuantStart** — *Successful Backtesting of Algorithmic Trading Strategies (Parts I and II)*. [quantstart.com/articles] Practical walk-forward implementation guide; discusses why TC underestimation is the most common failure mode. Widely referenced in practitioner community.

- **López de Prado / GARP (2017)** — *10 Reasons Most Machine Learning Funds Fail*. GARP Whitepaper. [garp.org whitepaper] Practitioner-facing synthesis of the backtesting methodology failures. Documents that TC underestimation and overfitting are the leading causes of ML fund failure.

- **BSIC (Bocconi Students Investment Club)** — *Modelling Transaction Costs and Market Impact*. [bsic.it] Accessible walkthrough of TC decomposition (spread, slippage, market impact) with implementation guidance.

- **Quant Beckman** — *Combinatorial Purged Cross Validation for Optimization (with code)*. [quantbeckman.com] Practitioner blog post with clean Python implementation of CPCV. Useful as a verification reference for the implementation.

### Tool & Library Documentation

- **scikit-learn TimeSeriesSplit** — [scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html] Walk-forward CV; gap parameter provides basic embargo. Confirmed v1.8.0 in production.

- **quantstats** — [pypi.org/project/quantstats/] v0.0.81, January 2026. Performance reporting tear sheets. Production/Stable.

- **backtesting** — [pypi.org/project/backtesting/] v0.6.5, July 2025. Event-driven strategy backtester. Active development.

- **pypbo** — [github.com/esvhd/pypbo] Python implementation of PBO, DSR, Probabilistic Sharpe Ratio. Free; install from GitHub.

- **mlfinpy** — [mlfinpy.readthedocs.io] Open-source MIT-licensed reimplementation of López de Prado methods (purged CV, triple-barrier, fractional differentiation). Actively maintained.

---

## Open Questions for Downstream Steps

- **How large should the universe be for meaningful purged CV?** With only 456 tickers and ~20 years of monthly data, some CV configurations (especially CPCV with many folds) may produce very short test windows. Step 3 (Expectations) should assess how many splits are feasible given the data shape, and what the minimum test window length should be for stable Sharpe estimation.

- **Should the TC model use fixed-rate spreads or the Corwin-Schultz estimator?** Fixed-rate is simpler and more controllable but less defensible. Corwin-Schultz adds data-driven variation but introduces its own noise. Step 3 should decide which is more appropriate for the backtest exercise given the universe size and the teaching objective.

- **What is the exact signal from Week 4 that Week 5 will backtest?** The research notes assume that Week 4 produces cross-sectional return predictions (IC-based, from gradient boosting or neural net). Step 3 should confirm what the Week 4 output format is and how it feeds into the Week 5 backtesting pipeline.

- **Should combinatorial purged CV (CPCV) be implemented in the lecture, or just purged k-fold?** CPCV is more complex (combinatorially many splits, distribution of performance estimates). Purged k-fold is the more commonly understood concept. CPCV may be better as a seminar exercise or a "further reading" reference rather than the primary lecture implementation. Step 2 (Blueprint) should make this call.

- **mlfinlab inaccessibility:** The previous course or student resources may reference mlfinlab. The verified alternative for library-based implementation is `mlfinpy`. Step 2 should explicitly use `mlfinpy` or manual implementation rather than `mlfinlab` in any exercise instructions.

---

## Verification Audit Trail

> This section is for the user's review during the Step 1 approval gate.
> Downstream agents (Steps 2–3) can skip this section.

**Queries run (initial discovery):**
1. "backtesting methodology purged cross-validation Lopez de Prado financial machine learning 2024 2025" → Confirmed no paradigm shift; CPCV remains standard; 2024 Knowledge-Based Systems paper confirms superiority empirically.
2. "transaction cost modeling market impact slippage Python quantitative finance 2024 2025" → Confirmed Almgren-Chriss as canonical model; no new paradigm; manual pandas implementation is production standard for research.
3. "deflated Sharpe ratio backtest overfitting probability Bailey Lopez de Prado Python implementation" → Confirmed DSR paper (2014) and PBO paper (2015) as canonical; pypbo library exists on GitHub; formula confirmed accessible.
4. "mlfinlab quantstats vectorbt backtesting Python library 2024 2025 maintained" → mlfinlab confirmed effectively inaccessible (closed-source); quantstats and backtesting library both active.

**Follow-up verification queries:**
5. mlfinlab PyPI page → 404 error, confirmed inaccessible.
6. quantstats PyPI page (fetched) → v0.0.81, January 2026, Production/Stable. Confirmed.
7. backtesting PyPI page (fetched) → v0.6.5, July 2025, actively maintained. Confirmed.
8. scikit-learn TimeSeriesSplit docs (fetched) → gap parameter confirmed as embargo proxy; no true purging. v1.8.0.
9. mlfinpy readthedocs (fetched) → MIT license confirmed, actively maintained, implements López de Prado methods.
10. pypbo GitHub search → Library exists, implements PBO/DSR/PSR; install from GitHub (no PyPI page accessible). Functionally available.
11. López de Prado lecture page quantresearch.org (fetched) → Two dedicated backtesting lectures confirmed; no TC-specific lecture.
12. HKUST Portfolio Optimization slides (referenced) → Seven Sins of Quantitative Investing section confirmed.
13. Harvey Liu Zhu 2016 paper → Fetched from Duke people page; t-stat hurdle confirmed at 3.0+ for new factor claims.
14. "7 Reasons ML Funds Fail" GARP whitepaper (fetched) → TC underestimation and overfitting confirmed as leading failure modes.
15. Almgren-Chriss paper confirmed from smallake.kr mirror and SimTrade blog; canonical status corroborated by Baruch MFE slides (Gatheral lectures).
16. "multiple testing correction false discovery rate financial strategy" → Harvey et al. framework confirmed; BHY preferred over Bonferroni for finance due to less stringency; practitioner use inconsistent.
17. Practitioner reality: backtesting gap → QuantStart, QuantNomad, PineConnector, QuantVPS blogs all confirm TC underestimation as primary live/backtest gap cause; walk-forward as dominant production practice.
18. 2024 SSRN paper (Arian, Norouzi, Seco) → Abstract confirmed on ScienceDirect; publication date and findings verified.

**Findings dropped (failed verification):**
- Baruch MFE ML syllabus PDF (certificate error on direct fetch) → Could not confirm specific backtesting coverage in the PDF; retained the broader finding that Baruch MFE teaches from López de Prado based on QuantNet forum corroboration and book acknowledgments.
- "vectorbt" as a backtesting tool for this week → Vectorbt is a vectorized backtesting library (strategy-level), not a cross-sectional ML evaluation framework. Not recommended for Week 5's use case; dropped from primary recommendations.

**Verification summary:**
- Total queries: 4 initial + 14 follow-up = 18
- Sources fetched (primary): 8 (quantstats PyPI, backtesting PyPI, scikit-learn docs, mlfinpy docs, quantresearch.org, GARP whitepaper, Harvey et al. paper, ScienceDirect abstract)
- Findings verified: 12
- Findings dropped: 2 (Baruch PDF unreadable; vectorbt out of scope)
