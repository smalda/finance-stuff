# Week 4: ML for Alpha — From Features to Signals

**Band:** 1 (Essential Core)
**Implementability:** HIGH

---

## Learning Objectives

1. **Build** a cross-sectional return prediction model using gradient boosting on a standardized feature matrix, following the Gu-Kelly-Xiu framework.
2. **Evaluate** alpha signals using information coefficient (IC), rank IC, and turnover-adjusted IC, and explain why these metrics — not accuracy or MSE — are the industry standard via the fundamental law of active management.
3. **Compare** gradient boosting vs. feedforward neural network performance on the same cross-sectional data, diagnosing where and why tree-based models dominate tabular financial features.
4. **Engineer** interaction features and non-linear transformations from raw firm characteristics, and measure their marginal contribution to signal quality using SHAP and permutation importance.
5. **Construct** a long-short portfolio from model predictions and assess whether the signal survives basic transaction cost adjustments.
6. **Distinguish** sandbox limitations (S&P 500 universe, 7-10 features, yfinance) from production reality (CRSP full universe, 94+ characteristics, point-in-time data) and articulate how each gap affects measured signal strength.
7. **Identify** the role of alternative data as a feature source for alpha models and explain the institutional landscape — what exists, what it costs, and why NLP-derived features are the most accessible entry point.

---

## Prerequisites

- **Required prior weeks:** Week 2, Week 3
- **Assumed from curriculum_state:**
  - Cross-sectional feature engineering: `FeatureEngineer` class, `feature_matrix_ml.parquet` with 7 features (4 fundamental + 3 technical), two-stage outlier control, rank normalization (Week 3)
  - Factor model context: CAPM, FF3, FF5 regressions, Fama-MacBeth procedure, factor loading interpretation, cross-sectional R-squared ~8-15% as baseline (Week 3)
  - Factor construction and validation: `FactorBuilder` class, double-sort methodology, factor zoo and multiple testing awareness (Week 3)
  - Financial time series: stationarity testing, GARCH volatility, realized volatility computation, fractional differentiation (Week 2)
  - Data handling: yfinance, adjusted prices, return computation, Parquet I/O, data quality checks, survivorship/look-ahead bias awareness (Week 1)
- **Assumed ML knowledge:** Gradient boosting (XGBoost/LightGBM), feedforward neural networks, cross-validation, hyperparameter tuning, overfitting diagnostics, SHAP values, feature importance methods, custom loss functions in PyTorch
- **Prerequisite explanations in this week:** None required (ML methodology assumed known; financial evaluation metrics — IC, rank IC, the fundamental law — are NEW finance content taught in Section 2)

---

## Opening Hook

In 2020, a team at Chicago Booth fed 94 firm characteristics into every ML model they could find — from Ridge regression to five-layer neural networks — and discovered something that should unsettle every ML engineer: all the models agreed. Not on the predictions, but on which features mattered. Momentum, liquidity, volatility — the same handful of signals dominated regardless of model complexity. The implication is disquieting: in cross-sectional equity prediction, the features ARE the alpha. Your model is just the lens.

---

## Narrative Arc

**The setup.** You are an ML expert. You know that model architecture matters — that a well-tuned neural network should crush a gradient boosting model given enough data, that feature interactions are best learned end-to-end, that more parameters mean more capacity to capture signal. You have spent years optimizing architectures, loss functions, and training procedures. Now someone hands you a spreadsheet of 94 columns — book-to-market, momentum, earnings yield — and asks you to predict which stocks will outperform next month. This should be straightforward.

**The complication.** It is not. Cross-sectional return prediction breaks three pillars of ML intuition. First, signal-to-noise is catastrophically low — an information coefficient of 0.05 makes you a star; your best ImageNet model gets 0.99. Second, gradient boosting stubbornly matches or beats neural networks on this tabular data, not because the networks are wrong but because the signal lives in a few dominant features that trees extract efficiently. Third, the features you engineer matter more than the model you choose — a mediocre model on great features beats a brilliant model on mediocre features. You are not in a regime where architectural cleverness wins. You are in a regime where domain knowledge about which features to construct, how to normalize them cross-sectionally, and how to evaluate them with IC rather than accuracy determines everything.

**The resolution.** By week's end, the ML expert has been reshaped into a signal researcher. They understand that cross-sectional prediction is not a Kaggle competition where the leaderboard rewards model complexity — it is a signal extraction problem where the fundamental law of active management (IR = IC x sqrt(BR)) dictates that breadth and consistency matter more than peak accuracy. They can build a cross-sectional alpha model, evaluate it with the metrics that funds actually use, construct a long-short portfolio from its predictions, and — critically — articulate exactly where their sandbox results diverge from institutional reality and why.

---

## Lecture Outline

### Section 1: The Cross-Sectional Prediction Problem → `s1_cross_sectional_setup.py`

- **Hook:** Every month, a quant researcher lines up 3,000 stocks in a row and asks one question: which ones will be in the top decile next month? That single question — cross-sectional return prediction — is the economic engine behind most systematic equity funds.
- **Core concepts:** Cross-sectional vs. time-series prediction, the prediction target (forward returns, rank of forward returns, excess returns), why cross-sectional standardization matters, the Gu-Kelly-Xiu framework as the canonical structure
- **Key demonstration:** Load the Week 3 feature matrix; show the cross-sectional structure — features at time t, forward returns at t+1; visualize the feature-return relationship for a single characteristic (e.g., momentum) as a cross-sectional scatter with rank normalization
- **Key formulas:** Cross-sectional rank normalization: r_i,t = rank(x_i,t) / N_t; Forward return target: y_i,t+1 = R_i,t+1 - R_bar_t+1 (excess cross-sectional return)
- **"So what?":** This framing — features today, returns tomorrow, repeated every month — is the skeleton of every ML alpha model at every systematic fund. Everything in this week is a variation on filling in the bones.
- **Bridge:** We have features and a target. Before we build a model, we need to know how to measure whether its predictions are worth anything.

### Section 2: The Language of Signal Quality → `s2_signal_evaluation.py`

- **Hook:** Your model has an R-squared of 0.001. In any other ML domain, you would delete it. In cross-sectional equity prediction, you might have just found a signal worth hundreds of millions of dollars.
- **Core concepts:** Information coefficient (IC) as Pearson correlation between predicted and realized cross-sectional returns, rank IC as Spearman variant (more robust to outliers), the fundamental law of active management (IR = IC x sqrt(BR)), why IC of 0.02-0.05 is realistic and IC > 0.10 warrants suspicion, turnover-adjusted IC, IC information ratio (ICIR = mean(IC) / std(IC)) as stability measure
- **Key demonstration:** Compute IC and rank IC from the Week 3 Fama-MacBeth predicted cross-sectional returns; show the IC time series and its instability; compute ICIR; show how a small but stable IC translates to economic value via the fundamental law
- **Key formulas:** IC_t = corr(y_hat_i,t, y_i,t) across stocks i; IR = IC x sqrt(BR) (fundamental law of active management, Grinold & Kahn 2000); ICIR = mean(IC_t) / std(IC_t)
- **"So what?":** IC is the universal currency of the alpha industry. Every signal — whether from gradient boosting, a neural network, or a fundamental analyst's intuition — is reduced to this single number. Understanding it is non-negotiable.
- **Bridge:** Now we know how to measure signals. Time to build one.

### Section 3: Gradient Boosting for Cross-Sectional Alpha → `s3_gradient_boosting_alpha.py`

- **Hook:** Ask a quant at a top fund what model they use for cross-sectional equity prediction, and the honest answer is usually three words: LightGBM or XGBoost. Not a transformer. Not a diffusion model. A gradient boosted tree from 2016.
- **Core concepts:** Why gradient boosting dominates tabular cross-sectional data (automatic feature interaction, robustness to outliers, fast training, interpretable), temporal train/test split (purged, no shuffle — connecting to Week 5's deeper treatment), rolling-window prediction, LightGBM configuration for financial cross-sections
- **Key demonstration:** Train LightGBM on the feature matrix to predict cross-sectional returns; evaluate with IC and rank IC over the test period; show the IC time series — months of positive IC, months of negative IC, the inherent instability of alpha signals
- **Key formulas:** LightGBM objective: L = sum(l(y_i, y_hat_i)) + Omega(f) (standard, but note the financial target construction matters more than the loss)
- **"So what?":** This is the production workhorse. If you join a systematic equity fund tomorrow, your first model will very likely be some variant of this. Understanding why it works here — and why your deep learning instincts to replace it are often wrong — is the meta-lesson.
- **Bridge:** The obvious question: can a neural network do better?

### Section 4: Neural Networks vs. Trees — The Honest Comparison → `s4_neural_vs_trees.py`

- **Hook:** Kelly, Malamud, and Zhou proved in 2023 that more complex models should outperform simpler ones for return prediction — theoretically. In practice, on our data, the answer is less satisfying.
- **Core concepts:** Feedforward neural network for cross-sectional prediction (1-2 hidden layers, matching GKX architecture), why the "virtue of complexity" result requires scale (94 features, full universe) that our sandbox lacks, the practitioner consensus ("GBM for tabular, NN for unstructured"), when neural networks DO add value in finance (NLP features, signal combination, latent factor learning)
- **Key demonstration:** Train a simple feedforward network on the same feature matrix; compare IC and rank IC head-to-head with the gradient boosting model; show where they agree and where they diverge; discuss scale dependence — why the result might differ with 94 features on 3,000 stocks
- **Sidebar:** Chen-Pelger-Zhu autoencoder factor model (MENTIONED) — the no-arbitrage loss function and conditional factor loadings; a frontier approach that bridges to Week 10 (causal/interpretable models)
- **Key formulas:** Feedforward prediction: y_hat = f_theta(x) where f is a 2-layer ReLU network; GKX used networks up to 5 hidden layers with 32 neurons each
- **"So what?":** The honest takeaway is not "trees always win" — it is "trees win HERE, on THIS data, at THIS scale." Knowing when to reach for deep learning (unstructured data, latent structure, massive feature sets) vs. when to stick with gradient boosting (tabular, moderate features, production constraints) is a career-defining judgment call.
- **Bridge:** Both models agree on something: the features matter more than the model. Time to examine what they are actually learning.

### Section 5: Feature Engineering and Importance → `s5_feature_engineering.py`

- **Hook:** GKX tested 94 characteristics. All their models — from Ridge to five-layer networks — agreed on the same top features. The model was the lens; the features were the light.
- **Core concepts:** Interaction features (e.g., momentum x volatility), non-linear transformations (log, polynomial, rank interactions), cross-sectional rank normalization as the standard preprocessing step, why feature engineering in finance is domain-driven (not automated), the substitution effect in correlated features
- **Key demonstration:** Extend the Week 3 feature matrix with interaction and non-linear terms; retrain the gradient boosting model; compare IC before and after feature expansion; use SHAP (TreeSHAP) to visualize which features and interactions drive predictions
- **Key formulas:** SHAP value: phi_j = sum over S of [|S|!(p-|S|-1)!/p!] x [f(S union {j}) - f(S)] (Shapley, conceptual; TreeSHAP computes exactly for trees)
- **"So what?":** At a fund, the feature engineering team often has more headcount than the modeling team. This is not an accident. Understanding which features to build — and how to verify they contribute — is where domain knowledge creates economic value that pure ML skill cannot.
- **Bridge:** We have a model and we have features. But a signal in isolation is just a number. To make money, it must become a portfolio.

### Section 6: From Signal to Portfolio → `s6_signal_to_portfolio.py`

- **Hook:** Your model says AAPL will be in the top decile next month. So you buy AAPL. But what do you sell? And how much? The gap between "good signal" and "profitable strategy" is where most quant careers are actually spent.
- **Core concepts:** Decile portfolio construction from model predictions (long top decile, short bottom decile), cumulative long-short returns, Sharpe ratio of the signal portfolio, why signal strength and portfolio profitability are not the same thing, basic transaction cost drag (turnover x estimated spread), connection to Week 5 (full backtesting) and Week 6 (portfolio optimization)
- **Key demonstration:** Sort stocks into deciles by model prediction each month; compute long-short portfolio returns; plot cumulative performance; compute Sharpe ratio; estimate turnover and apply a simple transaction cost haircut; show how the raw signal degrades after costs
- **Key formulas:** Long-short return: R_LS,t = R_top_decile,t - R_bottom_decile,t; Turnover: T_t = sum(|w_i,t - w_i,t-1|); Cost-adjusted return: R_net,t = R_LS,t - T_t x c (where c = estimated one-way cost)
- **"So what?":** This is the first time in the course that a student sees the complete arc: features → model → signal → portfolio → returns. It is a preview of the full pipeline that Week 5 (backtesting) and Week 6 (portfolio construction) will formalize. The signal-to-portfolio gap is why quant funds have portfolio management teams, not just research teams.
- **Bridge:** We have built everything with standard financial features. But the industry's fastest-growing edge comes from data that does not live in a Bloomberg terminal.

### Section 7: Alternative Data as Alpha Features → `s7_alternative_data.py`

- **Hook:** In 2025, 98% of institutional investors agreed that traditional financial data is "becoming too slow." The average hedge fund now spends over $1.6 million per year on satellite imagery, web traffic, transaction records, and parsed text. The question is no longer whether alternative data works — it is whether you can afford not to use it.
- **Core concepts:** Alternative data taxonomy (text/NLP, satellite, geolocation, transaction, web traffic), the institutional landscape (vendors, costs, data moats), why NLP-derived features are the most accessible category for ML engineers, how alternative data features enter the same cross-sectional framework as traditional features, the free-data constraint and what it means for education vs. production
- **Key demonstration:** Conceptual — diagram/discussion. Show the alternative data ecosystem as a taxonomy; illustrate how a sentiment score enters the feature matrix alongside momentum and earnings yield; discuss what is freely available (news headlines, SEC filings) vs. what costs institutional money (satellite, credit card transaction data)
- **Sidebar:** LLM-augmented feature engineering (MENTIONED) — recent research on LLM-generated features for short-horizon prediction; bridge to Week 7 (NLP for Financial Alpha)
- **Key formulas:** None (conceptual section)
- **"So what?":** Alternative data is the fastest-growing competitive frontier in systematic investing. For an ML engineer entering finance, the ability to process unstructured data into cross-sectional features is the single most in-demand skill after core alpha modeling. This section maps the landscape; Week 7 builds the tools.
- **Bridge:** This week built the core alpha pipeline — features, model, signal, portfolio. Next week tears it apart, asking: how do you know you have not fooled yourself?

---

## Narrative Hooks

1. **"The $0.05 correlation"** — An IC of 0.05 sounds pathetic by ML standards. But applied to 3,000 stocks monthly for 20 years, it can generate hundreds of millions in cumulative alpha. The fundamental law of active management explains why: breadth amplifies weak signals. *Suggested placement: Section 2, after introducing IC.*

2. **"The 94-feature consensus"** — GKX (2020) tested every ML model from Ridge to deep neural networks on 94 firm characteristics. The punchline: all models agreed on which features mattered — momentum, liquidity, and short-term reversal dominated regardless of model complexity. The features are the alpha; the model is just the lens. *Suggested placement: Section 5, opening.*

3. **"The $1.6 million data budget"** — The average hedge fund spends $1.6 million per year on alternative data (BattleFin/Exabel 2025). Satellite imagery of parking lots, credit card transaction volumes, parsed earnings call transcripts — the arms race has moved from model sophistication to data acquisition. For an ML engineer, this is actually good news: your NLP and computer vision skills ARE the competitive edge. *Suggested placement: Section 7, as the motivating statistic.*

4. **"The gradient boosting confession"** — Ask a quant at a top-tier market-making firm what model they use, and the honest answer is often: "We don't use deep learning at all." Gradient boosting on tabular data, tuned carefully, remains the production standard at firms trading billions. The ML engineer's instinct to reach for a transformer is, in this domain, usually wrong. *Suggested placement: Section 3, opening hook reinforcement.*

---

## Seminar Outline

### Exercise 1: The IC Autopsy

- **Task type:** Guided Discovery
- **The question:** Your gradient boosting model produces monthly ICs. In which market regimes (high-volatility months vs. low-volatility months) does the signal strengthen, weaken, or reverse? What does this tell you about the stability of cross-sectional alpha?
- **Expected insight:** IC is significantly more volatile in high-volatility regimes; the signal may even reverse sign during market stress. This instability — not the average IC — is what makes alpha modeling hard and why ICIR (IC stability) matters as much as IC level.

### Exercise 2: The Feature Knockout Experiment

- **Task type:** Guided Discovery
- **The question:** Systematically remove one feature at a time from the gradient boosting model and retrain. Which single feature removal causes the largest IC drop? Now remove the top two features simultaneously. Does the IC drop more or less than the sum of the individual drops? What does this reveal about feature substitution?
- **Expected insight:** Correlated features exhibit substitution — removing one barely hurts because another compensates. Removing both simultaneously causes a disproportionate drop. This is the substitution effect that makes standard feature importance misleading and motivates SHAP-based analysis.

### Exercise 3: The Complexity Ladder

- **Task type:** Skill Building
- **The question:** Train a sequence of models on the same feature matrix: (1) OLS linear regression, (2) Ridge regression, (3) LightGBM with depth=3, (4) LightGBM with depth=8, (5) feedforward neural network. Compute IC, rank IC, and ICIR for each. Does complexity monotonically help, or is there a plateau or reversal?
- **Expected insight:** On a small feature set and efficient universe (S&P 500), the complexity gains are modest or plateau after gradient boosting. This contrasts with the Kelly-Malamud-Zhou (2023) "Virtue of Complexity" result, which requires a larger feature set and broader universe — a sandbox-vs-production teaching moment about scale dependence.

### Exercise 4: The Turnover Tax

- **Task type:** Guided Discovery
- **The question:** Compute the monthly portfolio turnover of the long-short decile strategy from your gradient boosting model. Apply transaction costs at three levels (5 bps, 20 bps, 50 bps one-way). At what cost level does the signal's Sharpe ratio cross zero? How does constraining maximum monthly turnover affect net performance?
- **Expected insight:** Even modest transaction costs dramatically erode alpha from high-turnover signals. Students discover that signal strength (IC) and portfolio profitability (net Sharpe) can diverge sharply, motivating why turnover-adjusted metrics are the production standard and previewing Week 5's deeper treatment.

---

## Homework Outline

### Deliverable 1: Cross-Sectional Alpha Engine

- **Task type:** Construction
- **Mission framing:** Build a complete, reusable `AlphaModelPipeline` class that takes a feature matrix and produces evaluated alpha signals. This is the first artifact in the course that connects features to portfolio performance end-to-end — the skeleton that Weeks 5 and 6 will build upon. By the end, you will have a system that trains, predicts, evaluates, and reports with a single method call.
- **Scope:** A class that encapsulates: (a) temporal train/test splitting with a purge gap, (b) rolling-window model training (LightGBM), (c) IC/rank IC/ICIR computation per window, (d) long-short decile portfolio construction, (e) a summary report with signal and portfolio statistics. Extensible to accept any scikit-learn-compatible model.

### Deliverable 2: Feature Engineering Lab

- **Task type:** Investigation
- **Mission framing:** The Week 3 feature matrix has 7 features. Can you do better? This deliverable challenges you to expand the feature set with domain-motivated interaction terms and non-linear transformations, rigorously measure whether the additions improve signal quality, and produce a ranked importance analysis using SHAP. The discovery: which features actually matter, and which are noise masquerading as signal.
- **Scope:** Extend the feature matrix to 15-25 features (interactions, transformations, volatility-based features from Week 2); retrain the AlphaModelPipeline; compare IC/ICIR before and after expansion; produce SHAP summary plots and a written feature importance ranking with economic interpretation.
- **Requires:** Deliverable 1

### Deliverable 3: The Model Comparison Report

- **Task type:** Investigation
- **Mission framing:** Your fund's CIO asks: "Should we switch from gradient boosting to neural networks?" This deliverable produces the analysis that answers that question. You will run a rigorous head-to-head comparison across model families, evaluate on the metrics that matter (IC, ICIR, turnover, net Sharpe), and write a one-page recommendation with honest caveats about the sandbox-vs-production gap.
- **Scope:** Compare at least four model families (linear, Ridge, LightGBM, feedforward NN) on the expanded feature matrix; evaluate on IC, rank IC, ICIR, Sharpe, turnover, and net Sharpe after costs; include a "sandbox vs. production" section articulating how results might differ with institutional data (94+ features, 3,000+ stocks, point-in-time); write a structured recommendation (1 page max).
- **Requires:** Deliverables 1 and 2

---

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Cross-sectional prediction setup | Demo: feature-return scatter, target construction (S1) | — | — |
| IC / rank IC / ICIR | Demo: compute from Fama-MacBeth predictions (S2) | — | — |
| Fundamental law of active management | Demo: IC to IR derivation (S2) | — | — |
| Gradient boosting for alpha | Demo: train LightGBM, IC time series (S3) | — | — |
| Neural net vs. tree comparison | Demo: head-to-head IC comparison (S4) | Ex 3: complexity ladder across 5 model types | — |
| SHAP / feature importance | Demo: TreeSHAP visualization (S5) | Ex 2: feature knockout + substitution effect | — |
| Interaction / non-linear features | Demo: extend feature matrix, IC lift (S5) | — | D2: engineer 15-25 features, measure IC gain |
| Signal-to-portfolio construction | Demo: decile sort, long-short returns (S6) | Ex 4: turnover cost analysis | — |
| IC regime dependence | — | Ex 1: IC autopsy by volatility regime | — |
| Turnover-adjusted evaluation | — | Ex 4: cost breakeven analysis | D3: net Sharpe in model comparison |
| Alternative data landscape | Demo: taxonomy + ecosystem diagram (S7) | — | — |
| End-to-end alpha pipeline | — | — | D1: AlphaModelPipeline class |
| Model comparison methodology | — | — | D3: structured CIO recommendation |
| Sandbox vs. production gap | Demo: contextualization throughout | — | D3: explicit sandbox-vs-production section |

---

## Career Connections

- **ML Engineer at a systematic equity fund (e.g., Two Sigma, WorldQuant):** Your daily work is building and maintaining the cross-sectional alpha models taught this week — training LightGBM on hundreds of features, evaluating signals via IC dashboards, and deploying model updates to production pipelines that generate trade lists overnight.
- **Quantitative Researcher at a multi-manager pod shop (e.g., Millennium, Citadel):** You own the full pipeline from feature engineering to portfolio construction. Your performance is measured by net Sharpe after costs — exactly the metric this week's homework produces. Feature discovery is your competitive edge; model architecture is secondary.
- **Data Scientist at an asset manager (e.g., BlackRock Systematic, AQR):** You build factor-based and ML-based signal libraries consumed by portfolio managers. Your tools are SHAP plots, IC tear sheets, and turnover analyses — the exact outputs of this week's lecture and seminar.
- **NLP/AI Engineer at a hedge fund (e.g., Point72, Man Group AHL):** You transform unstructured text (earnings calls, SEC filings, news) into features that enter the same cross-sectional framework taught here. Section 7's alternative data landscape is your career roadmap; Week 7 is your deep dive.

---

## Data Sources

- yfinance — daily OHLCV, adjusted prices, fundamental snapshots
- Ken French Data Library (via pandas-datareader) — factor returns, benchmark portfolios
- Week 3 artifacts — `feature_matrix_ml.parquet`, `FactorBuilder` output
- LightGBM — gradient boosting models
- XGBoost — comparison baseline
- PyTorch — feedforward neural networks
- shap — TreeSHAP feature importance and visualization
- arch — volatility features (from Week 2)

---

## Key Papers & References

### Core

- **Gu, Kelly & Xiu (2020)** — "Empirical Asset Pricing via Machine Learning." _Review of Financial Studies_. The canonical reference: 94 characteristics, comprehensive model comparison, tree-based and neural approaches outperform linear. Why you'd read this: it defines the framework this entire week implements.
- **Grinold & Kahn (2000)** — "Active Portfolio Management." McGraw-Hill. Established IR = IC x sqrt(BR). Why you'd read this: every signal evaluation in the industry traces back to this framework.
- **Lopez de Prado (2018)** — "Advances in Financial Machine Learning." Wiley. MDI/MDA/SFI feature importance, purged CV, financial data labeling. Why you'd read this: the practitioner bible for financial ML pipelines.
- **Kelly & Xiu (2023)** — "Financial Machine Learning." _Foundations and Trends in Finance_. 160-page survey of the field. Why you'd read this: the most comprehensive graduate-level reference for ML in asset pricing.

### Advanced

- **Kelly, Malamud & Zhou (2023)** — "The Virtue of Complexity in Return Prediction." _Journal of Finance_. Proves more parameters can improve out-of-sample prediction for returns. Why you'd read this: it challenges standard bias-variance intuition and explains when scale makes complexity pay off.
- **Chen, Pelger & Zhu (2024)** — "Deep Learning in Asset Pricing." _Management Science_. Autoencoder-based conditional factor models with no-arbitrage loss. Why you'd read this: the frontier of neural network approaches to cross-sectional prediction.
- **Jansen (2020)** — "Machine Learning for Algorithmic Trading." Packt. 100+ alpha factors with Python code. Why you'd read this: the most hands-on practitioner guide for feature engineering.

---

## Bridge to Next Week

You now have a cross-sectional alpha model that produces signals and constructs portfolios. But how do you know the results are real? Week 5 takes this pipeline and stress-tests it — purged cross-validation, combinatorial purged CV, deflated Sharpe ratio, backtest overfitting quantification, and transaction cost modeling that goes far beyond our simple spread estimate. The question shifts from "can I build a signal?" to "should I believe it?"
