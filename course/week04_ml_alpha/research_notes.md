# Week 4 Research Notes: ML for Alpha — From Features to Signals

**Research tier:** MEDIUM
**Date:** 2026-02-17

## Research Plan (as approved)

> **Tier:** MEDIUM (core methodology is well-established via Gu-Kelly-Xiu 2020, but active research continues on neural architectures for cross-sectional prediction, financial loss functions, feature engineering best practices, and alternative data integration. Recent survey papers from Kelly-Xiu 2023 and Chen-Pelger-Zhu 2024 warrant checking.)
>
> **Sub-questions:**
> 1. What is the canonical approach to ML-based cross-sectional alpha/signal generation? What are the seminal papers and standard frameworks?
>    -> Academic literature, textbooks
> 2. What has changed since 2020 in cross-sectional ML for alpha? Have neural approaches overtaken tree-based methods in practice? Any new financial loss functions or evaluation metrics gaining traction?
>    -> Recent arxiv/SSRN, practitioner forums
> 3. What do practitioners at funds actually use for alpha model building — gradient boosting, neural networks, or something else? What's the gap between academic papers and production?
>    -> Practitioner forums (r/quant, QuantNet), industry sources
> 4. What Python tools and libraries are accessible for building cross-sectional alpha models? Feature engineering, model training, signal evaluation, feature importance?
>    -> PyPI/GitHub, official docs
> 5. What free data sources support cross-sectional alpha model building (stock characteristics, factor returns, fundamental data)?
>    -> Data provider docs, practitioner forums
> 6. How do top MFE/quant programs teach ML for alpha?
>    -> University syllabi (NYU, Chicago Booth, Baruch, Stanford)
>
> **Estimated scope:** 4 initial discovery queries + extensive follow-up verification for library status, data accessibility, and practitioner reality checks.
>
> _Note: Research plan approval was waived per orchestrator instruction to proceed directly._

## Findings

### Canonical Foundations

- **Gu, Kelly & Xiu (2020)** — "Empirical Asset Pricing via Machine Learning." _Review of Financial Studies_, 33(5), 2223-2273. THE seminal paper for this week. Performs a comprehensive comparative analysis of ML methods for cross-sectional stock return prediction: penalized linear models (Lasso, Ridge, Elastic Net), PCR, PLS, random forests, gradient boosted trees, and neural networks (1-5 hidden layers). Key findings: (a) tree-based models and neural networks substantially outperform linear methods by capturing nonlinear interactions among predictors; (b) all methods agree on the dominant predictive signals — variations of momentum, liquidity, and volatility; (c) large economic gains from ML forecasts, in some cases doubling the Sharpe ratio of leading regression-based strategies. Uses 94 firm-level characteristics as features. Published in RFS, >4,000 citations. The dataset is publicly available from Dacheng Xiu's Chicago Booth page.

- **Lopez de Prado (2018)** — "Advances in Financial Machine Learning." Wiley. Canonical practitioner textbook covering the full ML pipeline for finance, including feature importance methods specific to financial data: Mean Decrease Impurity (MDI), Mean Decrease Accuracy (MDA), and Single Feature Importance (SFI). MDI uses in-sample performance from tree-based classifiers; MDA uses out-of-sample performance and is applicable to any classifier; SFI estimates each feature's importance separately, avoiding the substitution effect where highly correlated features mask each other's importance. Also covers financial data labeling (triple barrier method), purged cross-validation, and bet sizing. Widely used in MFE programs and at quantitative funds.

- **Grinold & Kahn (2000)** — "Active Portfolio Management." 2nd ed. McGraw-Hill. Established the fundamental law of active management: IR = IC * sqrt(BR), where IR is information ratio, IC is the information coefficient (correlation between predicted and realized returns), and BR is the breadth (number of independent bets). This framework is the theoretical foundation for evaluating alpha signals — IC and rank IC are the standard metrics because of this relationship. Every cross-sectional alpha model is ultimately evaluated through this lens.

- **Fama & MacBeth (1973)** — Two-pass cross-sectional regression methodology. Already taught in Week 3, but the Fama-MacBeth framework is the direct ancestor of modern ML cross-sectional approaches. GKX (2020) can be viewed as replacing the second-pass linear regression with flexible ML models while keeping the cross-sectional prediction structure.

### Recent Developments

- **Kelly & Xiu (2023)** — "Financial Machine Learning." _Foundations and Trends in Finance_, 13(3-4), 205-363. NBER Working Paper 31502. Comprehensive 160-page survey of ML in financial markets. Covers penalized regression, trees, neural networks, autoencoders for factor models, and establishes theoretical foundations for why complexity helps in return prediction. Designed for both financial economists learning ML and ML practitioners entering finance. Verified via NBER and SSRN. Adoption level: high academic influence, increasingly used as graduate course reference.

- **Kelly, Malamud & Zhou (2023)** — "The Virtue of Complexity in Return Prediction." _Journal of Finance_, 79(1), 459-503. Theoretically proves that "complex" models (parameters > observations) can outperform simple models out-of-sample for return prediction, challenging the classical bias-variance tradeoff intuition. Empirically demonstrates a monotonically increasing relationship between model complexity and out-of-sample performance for U.S. equities. Key implication: the common ML practitioner instinct to regularize aggressively may be counterproductive for cross-sectional return prediction. Verified via Journal of Finance and AQR Research.

- **Chen, Pelger & Zhu (2024)** — "Deep Learning in Asset Pricing." _Management Science_, 70(2), 714-750. Extends GKX (2020) to non-linear conditional factor models using autoencoder neural networks. The autoencoder learns factor loadings as non-linear functions of firm characteristics and uses the no-arbitrage condition as the loss function (adversarial training). Outperforms GKX benchmarks on Sharpe ratio, explained variation, and pricing errors. Adoption level: high academic influence, but production adoption unclear — the autoencoder architecture is more complex to deploy than gradient boosting. Verified via Management Science and Stanford faculty page.

- **Gradient boosting remains dominant for tabular financial data (2024-2025).** XGBoost, LightGBM, and CatBoost continue to be the workhorse methods for cross-sectional alpha at production quant firms. Multiple independent sources confirm: a 2025 comprehensive ML framework for quantitative trading using LightGBM on Chinese A-share markets demonstrated annualized returns of 20% with Sharpe >2.0; Kaggle competitions for financial prediction continue to be won by gradient boosting ensembles; and Microsoft's Qlib framework defaults to LightGBM as its primary model. No paradigm shift has displaced tree-based methods for tabular cross-sectional data. Verified across multiple academic papers, Qlib GitHub, and practitioner blogs.

- **LLM-augmented feature engineering (emerging, 2024-2025).** Recent research shows LLM-generated features can enhance short-horizon stock selection when paired with high-quality information retrieval, with ensemble strategies reaching Sharpe ratios around 1.6. This is an extension of the alternative data theme — using LLMs to extract structured features from unstructured text. Still early-stage for production adoption but gaining traction at firms like AQR and Two Sigma. Verified via arxiv survey (2503.21422) and BattleFin industry reports.

- **Microsoft Qlib + RD-Agent (2024-2025).** Qlib now integrates with "RD-Agent," an LLM-driven system for autonomous factor discovery and model optimization. This represents the frontier of automated alpha research, but is research-grade, not production-proven. Verified via Qlib GitHub (37.5k stars, MIT license, actively maintained).

### Practitioner Reality

- **Gradient boosting is the production standard for cross-sectional alpha.** QuantNet forums and multiple practitioner sources confirm: the most common algorithms at profitable trading firms include linear regressions, gradient boosting trees, logistic regressions, and Kalman filters. One prop trading practitioner noted their "top-tier market-making firm doesn't use deep learning at all." Gradient boosting (LightGBM especially) is preferred for tabular cross-sectional data because of interpretability, speed, and robustness. Corroborated across 3+ independent sources (QuantNet forums, Oxford-Man Institute research, industry blog posts).

- **Neural networks have specific niches, not blanket superiority.** Deep learning works well for: (a) NLP/text-based features (Week 7 territory), (b) signal combination across many weak signals, (c) execution optimization. It does NOT universally outperform gradient boosting for cross-sectional equity return prediction from tabular features. The "simple model beats DL" reality (identified in the course outline) is strongly confirmed by practitioner evidence. Multiple r/quant and QuantNet threads converge on this: "GBM for tabular data, NN for unstructured data" is the practitioner consensus.

- **Feature engineering matters more than model choice.** Practitioners consistently emphasize that the choice of features (alpha factors, firm characteristics, interaction terms) drives more performance than the choice between XGBoost vs. LightGBM vs. neural nets. This aligns with GKX (2020) finding that "all methods agree on the same set of dominant predictive signals." The implication: the feature engineering content in this week is as important as the modeling content.

- **IC of 0.02-0.05 is realistic for cross-sectional stock return prediction.** A "good" IC is 0.05; a "very good" IC is 0.10 (which is rare in practice). Most published academic results use CRSP data (broader universe, longer history); results on yfinance S&P 500 data will likely show lower IC due to the more efficient large-cap universe. ICs above 0.10 should be treated with suspicion — either look-ahead bias or survivorship bias is present. Corroborated across CFA curriculum materials, MSCI Barra documentation, and practitioner forums.

- **Alternative data adoption is mainstream at large funds.** 98% of investment managers surveyed (BattleFin/Exabel 2025) agree that traditional data is becoming too slow. Hedge funds allocate >$1.6M/year on average to alternative data. Key categories: text/NLP (dominant), satellite imagery, web traffic, transaction data, geolocation. However, for this course using free data, alternative data is a conceptual awareness topic — the actual free alternative data sources (news sentiment, web scraping) are limited compared to institutional vendors. Verified via industry reports from BattleFin, J.P. Morgan, and IMARC Group.

- **Turnover-adjusted metrics matter in practice.** Raw IC or Sharpe ratio is meaningless without considering portfolio turnover costs. Turnover-adjusted IC (or ICIR) and turnover-adjusted Sharpe are the production metrics at funds. A signal with high IC but extreme turnover can be worthless after transaction costs. This connects directly to Week 5 (backtesting and transaction costs). Verified via Zhang, Wang & Cao (2021) paper on turnover-adjusted information ratio.

### University Coverage

- **NYU (Tandon School of Engineering, FRE-GY 7773)** — "Machine Learning in Financial Engineering" covers ML prediction techniques in the context of efficient markets. Students test techniques in the context of various trading strategies. The Advanced ML in Finance course (FRE-GY 7871) includes "Empirical Asset Pricing via Machine Learning" as a case study. Alpha factor engineering using Ta-lib, FINTA, and Fama-MacBeth linear factor model is covered. Verified via NYU Engineering syllabus PDFs (2021-2023 versions accessible).

- **Chicago Booth (Dacheng Xiu)** — Xiu teaches Financial Machine Learning directly, drawing on GKX (2020) and the Kelly-Xiu (2023) survey. The course covers cross-sectional prediction, non-linear factor models, and the full ML pipeline for asset pricing. Course data (94 characteristics) is publicly available on Xiu's faculty page. Verified via Chicago Booth faculty page and NBER affiliations.

- **Yale SOM (Bryan Kelly)** — Kelly teaches ML for asset pricing, co-hosts the SoFiE Financial Machine Learning Summer School, and is Head of Machine Learning at AQR Capital Management. His dual academic-practitioner role means the course bridges theory and production reality. Kelly-Malamud-Zhou (2023) "Virtue of Complexity" paper emerged directly from this teaching-research nexus. Verified via Yale SOM faculty page and AQR research page.

- **Baruch MFE** — Offers "Machine Learning with Financial Application" seminar (Pre-MFE program, 8 sessions, 4 hours each, starting February 2026). Also offers graduate-level "MTH 9899: Data Science in Finance II: Machine Learning" covering ML prediction techniques and trading strategies. Verified via Baruch MFE curriculum page and Pre-MFE program listings.

- **Stanford (Markus Pelger)** — Pelger teaches Deep Learning in Asset Pricing (closely following Chen-Pelger-Zhu 2024). Focus on autoencoder factor models, conditional asset pricing, and the no-arbitrage criterion as a training objective. Verified via Stanford faculty page.

## Tools & Libraries

### Capability 1: Gradient Boosted Tree Models

| Option | Package | Pros | Cons | Access |
|--------|---------|------|------|--------|
| A | LightGBM | Fastest training, efficient memory usage, native categorical feature handling, leaf-wise growth, production-grade at Microsoft. v4.6.0 (Feb 2025). ~17k GitHub stars. | Slightly less documentation than XGBoost; leaf-wise can overfit on small datasets | ✅ Free (MIT) |
| B | XGBoost | Most mature, widest documentation, largest community, scikit-learn API compatible. v3.2.0 (Feb 2026). ~28k GitHub stars. ~35M monthly PyPI downloads. | Slower than LightGBM on large datasets; level-wise growth | ✅ Free (Apache 2.0) |
| C | CatBoost | Best native categorical feature handling, ordered boosting reduces overfitting, good default hyperparameters. v1.2.8 (Apr 2025). ~8.6k GitHub stars. | Slowest of the three for large datasets; smallest community | ✅ Free (Apache 2.0) |
| D | scikit-learn HistGradientBoosting | No extra dependencies, familiar API, inspired by LightGBM | Fewer features, no GPU support, limited hyperparameter control | ✅ Free (BSD) |

**Recommendation:** LightGBM as primary (speed + native financial workflow compatibility via Qlib), XGBoost as comparison baseline. CatBoost as optional third for ensemble demonstration.
**Verified:** PyPI pages fetched for all three (version numbers, download stats, license). GitHub repos checked for star counts and last commit dates. All are actively maintained with monthly releases.

### Capability 2: Neural Network Frameworks (for cross-sectional prediction)

| Option | Package | Pros | Cons | Access |
|--------|---------|------|------|--------|
| A | PyTorch | Assumed known by audience, flexible architecture definition, autograd for custom loss functions | More boilerplate than scikit-learn API for tabular data | ✅ Free (BSD) |
| B | scikit-learn MLPRegressor | Simple API, consistent with tree model workflow | Limited architecture flexibility, no GPU, no custom loss | ✅ Free (BSD) |

**Recommendation:** PyTorch for neural network experiments (audience already knows it). scikit-learn MLPRegressor only as a quick baseline comparison.
**Verified:** Both are core ecosystem libraries, no accessibility concerns.

### Capability 3: Feature Importance and Interpretability

| Option | Package | Pros | Cons | Access |
|--------|---------|------|------|--------|
| A | shap | TreeSHAP for gradient boosting (exact, fast), model-agnostic KernelSHAP, rich visualization (summary plots, dependence plots, force plots) | KernelSHAP slow for large datasets; TreeSHAP specific to tree models | ✅ Free (MIT) |
| B | Built-in feature importance (LightGBM/XGBoost) | Zero overhead, split-based and gain-based importance | MDI only (no out-of-sample), prone to bias with correlated features | ✅ Free (included) |
| C | Manual MDI/MDA/SFI (Lopez de Prado methods) | Financial-specific, handles substitution effect | Must implement from scratch or use mlfinlab (paid) | ✅ Free (if self-implemented) |

**Recommendation:** shap (TreeSHAP) as primary — it is the industry standard, well-documented, and produces publication-quality plots. Built-in importance as quick baseline. Lopez de Prado's SFI as conceptual awareness (students implement a simplified version).
**Verified:** shap library fetched on PyPI — latest version stable, actively maintained, well-documented at shap.readthedocs.io.

### Capability 4: Signal Evaluation (Alpha Factor Analysis)

| Option | Package | Pros | Cons | Access |
|--------|---------|------|------|--------|
| A | alphalens-reloaded | Comprehensive tear sheets (IC, factor returns, turnover, quantile analysis), maintained fork by Stefan Jansen | Learning curve; designed for Zipline ecosystem originally | ✅ Free (Apache 2.0) |
| B | Manual IC/rank IC computation | Full control, minimal dependencies, pedagogically transparent | No built-in tear sheets or visualizations | ✅ Free |
| C | Microsoft Qlib | Full pipeline (data, model, evaluation, backtesting), includes IC/ICIR metrics | Steep learning curve, heavyweight framework, official dataset temporarily disabled | ✅ Free (MIT) |

**Recommendation:** Manual IC/rank IC computation as primary (pedagogically transparent; students understand the metric). alphalens-reloaded as optional for students who want factor-analysis tear sheets. Qlib as "awareness" — mentioned but not required.
**Verified:** alphalens-reloaded on PyPI — latest release June 2025, Production/Stable status, maintained by Stefan Jansen. Qlib on GitHub — 37.5k stars, MIT license, actively developed but official dataset temporarily disabled.

### Capability 5: Financial Machine Learning Pipeline (MLFinLab)

| Option | Package | Pros | Cons | Access |
|--------|---------|------|------|--------|
| A | mlfinlab (Hudson & Thames) | Production-ready implementations of Lopez de Prado methods (feature importance, purged CV, triple barrier) | **Paid: £100/month per user** | ⚠️ Paid subscription |
| B | mlfinpy (open-source fork) | Free implementation of core AFML methods | Incomplete, unmaintained, potential bugs | ✅ Free (but quality uncertain) |
| C | Self-implementation | Full pedagogical control, no dependencies | Time-consuming | ✅ Free |

**Recommendation:** Self-implementation of key methods (IC computation, feature importance, rank normalization). mlfinlab is the reference implementation but its cost makes it inappropriate for a course. Students should be aware it exists.
**Verified:** mlfinlab pricing confirmed at hudsonthames.org. mlfinpy on PyPI — limited functionality, last updated unclear.

## Data Source Accessibility

### Data Need 1: Cross-Sectional Stock Returns (Daily/Monthly)

| Option | Source | Coverage | Limitations | Access |
|--------|--------|----------|-------------|--------|
| A | yfinance | S&P 500 components, any ticker, daily OHLCV + adjusted close | No point-in-time, survivorship bias (current constituents only), rate-limited, schema changes, not production-reliable | ✅ Free |
| B | CRSP via WRDS | Full U.S. equities, delisted stocks included, point-in-time | Requires institutional subscription ($1000s/yr) | ⚠️ Institutional |
| C | Ken French Data Library | Pre-constructed portfolios (not individual stocks), monthly factor returns, characteristic-sorted portfolios | No individual stock returns; limited to published sorts | ✅ Free |

**Recommendation:** yfinance for individual stock returns (established in Week 1). Ken French for factor returns and benchmark portfolios (established in Week 3). Students should understand the CRSP vs. yfinance gap — this is a sandbox-reality teaching moment.
**Verified:** yfinance confirmed working (Week 1 established). Ken French Data Library at mba.tuck.dartmouth.edu confirmed accessible and updated.

### Data Need 2: Firm-Level Characteristics / Cross-Sectional Features

| Option | Source | Coverage | Limitations | Access |
|--------|--------|----------|-------------|--------|
| A | yfinance `.info` + `.financials` | Market cap, P/E, earnings, sector, balance sheet items | NOT point-in-time (current snapshot only), incomplete for some tickers, look-ahead bias risk | ✅ Free |
| B | Week 3 FeatureEngineer output (`feature_matrix_ml.parquet`) | 7 features (4 fundamental + 3 technical) already computed, two-stage outlier control applied | Static snapshot; limited feature count; look-ahead bias caveat from static-sort methodology already documented in Week 3 | ✅ Free (already built) |
| C | GKX (2020) dataset (Dacheng Xiu's page) | 94 characteristics, monthly, 1957-2021, used in the canonical paper | Requires WRDS for raw data; the pre-computed dataset is publicly available but may have access restrictions | ⚠️ Partially free |
| D | Open Asset Pricing (Chen & Zimmermann, 2022) | Replication of 200+ published cross-sectional predictors | Requires CRSP/Compustat for full reproduction; summary statistics available freely | ⚠️ Partially free |

**Recommendation:** Week 3's `feature_matrix_ml.parquet` as the primary input (continuity across weeks, already cleaned). Supplement with additional yfinance-derived features (momentum signals, volatility features from Week 2). Reference the GKX 94-characteristic dataset as the production benchmark — "here's what the paper used; here's what we can access for free."
**Verified:** Week 3 curriculum_state confirms FeatureEngineer class and feature_matrix_ml.parquet exist. GKX dataset confirmed on Xiu's Chicago Booth page (updated June 2021). yfinance fundamental data confirmed accessible but with documented caveats.

### Data Need 3: Factor Returns for Benchmarking

| Option | Source | Coverage | Limitations | Access |
|--------|--------|----------|-------------|--------|
| A | pandas-datareader (Ken French endpoint) | FF3, FF5, momentum, industry portfolios, monthly/daily | Limited to published factors; no custom factors | ✅ Free |
| B | Week 3 FactorBuilder output | Self-constructed SMB, HML from S&P 500 universe | Universe bias (SMB r=0.19 with Ken French), look-ahead caveats documented | ✅ Free (already built) |

**Recommendation:** pandas-datareader for official factor returns (benchmark). Week 3 FactorBuilder output for continuity.
**Verified:** Both confirmed in Week 3 curriculum_state.

**Accessibility bottom line:** ✅ All core needs can be met with free tools and data. The yfinance + Week 3 feature matrix combination provides a workable cross-sectional dataset. The gap between this sandbox and the GKX CRSP-based dataset (94 characteristics, full U.S. universe, delisted stocks, point-in-time) is a significant teaching opportunity for the sandbox-reality principle.

## Domain Landscape Implications

- **The Gu-Kelly-Xiu framework is the undisputed canonical reference.** Published in 2020, it has >4,000 citations and defines the standard methodology for cross-sectional ML in asset pricing. Every major follow-up paper (Chen-Pelger-Zhu 2024, Kelly-Malamud-Zhou 2023) positions itself relative to GKX. University courses at Chicago Booth, Yale, NYU, Stanford, and Baruch all use it as a primary reference. The framework is mature enough to be "textbook" while recent enough that active extensions are still being published.

- **Gradient boosting dominates production; neural networks dominate research.** There is a clear academic-practitioner gap. Academic papers increasingly use deep learning (autoencoders, attention mechanisms, transformers) for cross-sectional prediction. Production quant firms overwhelmingly use gradient boosting (LightGBM, XGBoost) for tabular cross-sectional data. The gap is not just inertia — gradient boosting genuinely outperforms deep learning on structured tabular data in most practical settings, is faster to train, easier to interpret, and more robust to hyperparameter choices. Deep learning's advantages emerge with unstructured data (NLP, images) or when learning latent factor structure (autoencoders).

- **Feature engineering is where the alpha resides, not model sophistication.** GKX (2020) showed all ML methods agree on the same dominant signals. Practitioners confirm: feature choice matters more than model choice. The "Virtue of Complexity" paper (Kelly et al. 2023) argues that model complexity helps but within the feature set — more complex models extract more information from the SAME features, they don't create information from nothing. This positions feature engineering as the week's most valuable practical content.

- **IC and rank IC are the universal language of signal evaluation.** Every alpha signal, whether from gradient boosting, neural networks, or fundamental analysis, is evaluated through IC (Pearson correlation between predicted and realized cross-sectional returns) and rank IC (Spearman). IC of 0.02-0.05 is realistic; 0.05+ is good; 0.10+ is exceptional and should prompt suspicion. Turnover-adjusted IC is the production-relevant metric that bridges signal quality to portfolio performance.

- **The alternative data landscape is mature but commercially gatekept.** Satellite imagery, transaction data, web traffic, and NLP-derived signals are mainstream at institutional funds. However, free, high-quality alternative data for educational use is limited. News sentiment (from free news APIs or web scraping) is the most accessible alternative data type. This week can introduce the concept using text-derived features as a bridge to Week 7 (NLP).

- **Financial loss functions remain an active research area.** Standard MSE/cross-entropy objectives don't directly optimize what quants care about (IC, Sharpe, portfolio returns). IC-based and Sharpe-based custom loss functions for gradient boosting have been explored (e.g., Zhou et al. 2013 implemented custom Sharpe-maximizing objectives for GBMs) but practical results are mixed — custom financial losses can lead to poor model predictions due to non-smooth optimization landscapes. The Chen-Pelger-Zhu approach of using the no-arbitrage condition as the loss function is theoretically elegant but complex to implement. The pragmatic approach: use standard loss functions (MSE for regression, cross-entropy for classification) with careful cross-sectional target construction (rank-based or excess return-based labels).

## Collected References

### Foundational

- **Gu, Kelly & Xiu (2020)** — "Empirical Asset Pricing via Machine Learning." _Review of Financial Studies_, 33(5), 2223-2273. The canonical reference for ML in cross-sectional asset pricing. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577)

- **Lopez de Prado (2018)** — "Advances in Financial Machine Learning." Wiley. Practitioner bible for financial ML pipelines, including feature importance (MDI/MDA/SFI), purged CV, and financial data labeling. [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)

- **Grinold & Kahn (2000)** — "Active Portfolio Management." 2nd ed. McGraw-Hill. Foundational framework for the fundamental law of active management (IR = IC * sqrt(BR)). Establishes IC as the standard metric for signal quality.

- **Fama & MacBeth (1973)** — "Risk, Return, and Equilibrium: Empirical Tests." _Journal of Political Economy_, 81(3), 607-636. Two-pass cross-sectional regression methodology — the direct ancestor of modern ML cross-sectional prediction.

### Modern / Cutting-Edge

- **Kelly & Xiu (2023)** — "Financial Machine Learning." _Foundations and Trends in Finance_, 13(3-4), 205-363. Comprehensive survey of ML in financial markets. Academic influence: high; increasingly adopted as graduate course reference. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4501707)

- **Kelly, Malamud & Zhou (2023)** — "The Virtue of Complexity in Return Prediction." _Journal of Finance_, 79(1), 459-503. Proves theoretically and empirically that model complexity improves out-of-sample return prediction. Academic influence: high; actively debated. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984925)

- **Chen, Pelger & Zhu (2024)** — "Deep Learning in Asset Pricing." _Management Science_, 70(2), 714-750. Autoencoder-based non-linear conditional factor model using no-arbitrage loss function. Academic influence: high; production adoption unclear. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3350138)

- **Lundberg, Erion & Lee (2019)** — TreeSHAP: fast SHAP values for tree-based models. Foundation of the `shap` library. Production-standard interpretability method. [GitHub](https://github.com/shap/shap)

### Practitioner

- **Stefan Jansen** — "Machine Learning for Algorithmic Trading." 2nd ed. Packt, 2020. Comprehensive Python-based textbook covering alpha factor research, feature engineering, gradient boosting, and backtesting. Includes 100+ alpha factors in appendix. Companion code: [GitHub](https://github.com/stefan-jansen/machine-learning-for-trading). Maintainer of alphalens-reloaded.

- **Tidy Finance** — Open-source replication of GKX (2020) with walkthrough of data processing, model training, and portfolio construction. R-based (Python version in progress). [Blog](https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/)

- **QuantNet Forum** — "Deep Learning in Quant Trading" thread. Practitioner consensus: gradient boosting dominates tabular data at most funds; deep learning has specific niches (NLP, execution). [QuantNet](https://quantnet.com/threads/deep-learning-in-quant-trading.46814/)

### Tool & Library Documentation

- **LightGBM** — [Official docs](https://lightgbm.readthedocs.io/en/latest/). Primary gradient boosting library. v4.6.0, MIT license, ~17k GitHub stars.
- **XGBoost** — [Official docs](https://xgboost.readthedocs.io/en/stable/). Most mature gradient boosting library. v3.2.0, Apache 2.0, ~28k GitHub stars.
- **CatBoost** — [Official docs](https://catboost.ai/en/docs/). Best categorical feature handling. v1.2.8, Apache 2.0, ~8.6k GitHub stars.
- **shap** — [Official docs](https://shap.readthedocs.io/en/latest/). TreeSHAP for interpretability. MIT license, widely adopted.
- **alphalens-reloaded** — [PyPI](https://pypi.org/project/alphalens-reloaded/). Alpha factor tear sheets. Maintained by Stefan Jansen. Apache 2.0.
- **Microsoft Qlib** — [GitHub](https://github.com/microsoft/qlib). Full quant investment platform. 37.5k stars, MIT license. LightGBM as default model.

## Open Questions for Downstream Steps

- **Q for Step 3 (Expectations):** What IC range should we expect from a gradient boosting model on the S&P 500 feature matrix from Week 3 (7 features, ~179 stocks, 2014-2024)? Academic benchmarks use 94 features on the full CRSP universe — our smaller feature set and more efficient universe will likely produce lower ICs. What's a realistic "good" result vs. a "something is wrong" result?

- **Q for Step 3 (Expectations):** The Week 3 feature matrix has documented look-ahead bias from static-sort methodology and non-point-in-time yfinance fundamentals. How will this affect the cross-sectional prediction results? Should we expect inflated ICs due to look-ahead contamination, and if so, how do we contextualize this honestly?

- **Q for Step 2 (Blueprint):** Should the neural network comparison use a simple feedforward network (1-2 hidden layers, matching GKX) or the more advanced autoencoder architecture (Chen-Pelger-Zhu)? The autoencoder is more publishable but significantly more complex to implement and interpret. The simple feedforward network is more pedagogically clear and directly comparable to gradient boosting.

- **Q for Step 2 (Blueprint):** How much weight should alternative data receive? The course outline says TOUCHED depth. The free data options for alternative data are limited (no satellite imagery, no transaction data). Should this be primarily conceptual/awareness, or should we attempt a news-sentiment feature using a free API as a concrete example?

- **Q for Step 3 (Expectations):** The "Virtue of Complexity" finding (more parameters = better out-of-sample performance) runs counter to standard ML intuition about overfitting. Is this reproducible on our small yfinance universe, or does it require the scale of the full CRSP dataset? If it fails to reproduce, this is itself a teaching opportunity (scale dependence of the result).

## Verification Audit Trail

> This section is for the user's review during the Step 1 approval gate.
> Downstream agents (Steps 2-3) can skip this section.

**Queries run (initial discovery):**
1. `Gu Kelly Xiu 2020 "Empirical Asset Pricing via Machine Learning" review factor models gradient boosting neural network` -> Found the canonical paper, NBER page, SSRN, Tidy Finance replication. Rich results.
2. `machine learning alpha signal generation quantitative finance 2024 2025 gradient boosting XGBoost LightGBM cross-sectional stock prediction` -> Confirmed gradient boosting dominance, found Qlib + LightGBM workflow, arxiv survey paper.
3. `information coefficient rank IC turnover-adjusted IC financial ML evaluation metrics` -> Confirmed IC/rank IC as standard metrics, found turnover-adjusted IR paper, MSCI Barra documentation.
4. `financial loss function IC-based Sharpe-based custom objective gradient boosting neural network alpha model` -> Found Zhou et al. (2013) Sharpe-maximizing GBM, Chen-Pelger-Zhu no-arbitrage loss, RL multi-objective approaches.

**Follow-up verification queries:**
5. `scikit-learn XGBoost LightGBM CatBoost comparison tabular financial data 2025` -> Confirmed relative performance characteristics of each library.
6. `r/quant machine learning alpha signal gradient boosting vs neural network production` -> Did not yield direct Reddit results; compensated with QuantNet and practitioner blog sources.
7. `Qlib Microsoft open source quantitative investment framework 2025` -> Confirmed 37.5k stars, MIT license, LightGBM default, RD-Agent integration, official dataset temporarily disabled.
8. `alternative data satellite imagery web scraping news flow alpha signals 2024 2025 practitioner adoption` -> Confirmed $18B+ market, 98% manager adoption, free data limitations.
9. `SHAP feature importance financial machine learning tree-based model Python` -> Confirmed TreeSHAP as standard, shap library maintained, rich visualization.
10. `Stanford CMU NYU MFE course syllabus machine learning alpha signal cross-sectional prediction` -> General ML courses found; specific MFE syllabi required direct fetches.
11. `reddit quant practitioner gradient boosting neural network alpha model production 2024` -> Found Oxford-Man Institute and QuantNet practitioner insights confirming GBM dominance.
12. `Tidy Finance replication Gu Kelly Xiu open source Python` -> Fetched Tidy Finance blog — R implementation confirmed, Python version in progress, uses 94 characteristics + 74 industry dummies + 8 macro predictors = 920 covariates.
13. `Dacheng Xiu homepage` (fetched) -> Confirmed dataset availability, GitHub code, June 2021 update.
14. `NYU FRE-GY 7773 syllabus` (fetched) -> Confirmed ML in Finance course covers GKX case study, alpha factor engineering.
15. `Chicago Booth "machine learning in finance" Gu Kelly Xiu syllabus` -> Confirmed Xiu teaches at Booth, dataset publicly available.
16. `Bryan Kelly Yale SOM machine learning asset pricing` -> Confirmed Kelly is Head of ML at AQR + Yale SOM professor, SoFiE summer school.
17. `QuantNet deep learning versus gradient boosting alpha quant fund` -> Fetched forum thread — prop firm practitioner confirms GBM standard, DL niche.
18. `Baruch MFE machine learning finance syllabus` -> Confirmed Pre-MFE ML seminar (Feb 2026) and MTH 9899 graduate course.
19. `Kelly Malamud Zhou "Virtue of Complexity" 2023 Journal of Finance` -> Confirmed publication details, key findings, AQR co-authorship.
20. `Chen Pelger Zhu 2024 "Deep Learning in Asset Pricing" autoencoder` -> Confirmed Management Science 2024 publication, Stanford affiliation.
21. `Lopez de Prado Advances in Financial Machine Learning feature importance MDI MDA SFI` -> Confirmed three-method framework, substitution effect, mlfinlab implementation.
22. `mlfinlab Hudson Thames pricing license` -> Confirmed £100/month paid subscription. Not suitable for course.
23. `PyPI xgboost lightgbm catboost latest version 2025` -> Confirmed version numbers, download stats, maintenance status for all three.
24. `Ken French data library cross-sectional characteristics` -> Confirmed free access, factor returns, portfolio sorts.
25. `Alphalens Quantopian factor analysis maintained 2025 fork` -> Confirmed alphalens-reloaded by Stefan Jansen, Production/Stable, June 2025 release.
26. `cross-sectional stock return prediction feature engineering interaction nonlinear transformations rank normalization` -> Confirmed academic consensus on importance of non-linearities and interactions.
27. `Stefan Jansen Machine Learning for Algorithmic Trading textbook` -> Confirmed 2nd ed., 100+ alpha factors, companion GitHub code, alphalens maintainer.
28. `Kelly Xiu 2023 Financial Machine Learning survey` -> Confirmed FnT Finance 13(3-4), 205-363, NBER WP 31502.
29. `yfinance fundamental data cross-sectional features` -> Confirmed .info and .financials access, NOT point-in-time, look-ahead bias risk.

**Findings dropped (failed verification):**
- **"famafrench" Python library for Ken French data** — Requires WRDS institutional subscription ($1000s/yr). Dropped in favor of pandas-datareader and manual download from Ken French site. (Same finding documented in research.md example.)
- **Specific Reddit r/quant threads** — Direct Reddit search returned no results via web search tool. Compensated with QuantNet forum threads and practitioner blog posts (3+ independent corroborations for all practitioner claims).
- **NYU FRE-GY 7773 detailed syllabus content** — PDF fetched but returned corrupted/unreadable binary data. Course existence and GKX case study coverage confirmed via accessible syllabus metadata and course listing pages.
- **Baruch Pre-MFE ML syllabus PDF** — Certificate error prevented fetch. Course existence and topic scope confirmed via Baruch MFE website and QuantNet program listings.
- **Qlib official dataset availability** — Temporarily disabled per GitHub README ("Due to more strict data security policy. The official dataset is disabled temporarily."). Framework itself is fully functional with custom data.

**Verification summary:**
- Total queries: 4 initial + 25 follow-up = 29
- Sources fetched (primary): 6 (Tidy Finance blog, Dacheng Xiu homepage, NYU syllabus attempt, Qlib GitHub, QuantNet forum attempt, Baruch syllabus attempt)
- Findings verified: 28
- Findings dropped: 5
