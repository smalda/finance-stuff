# Week 4: ML for Alpha — From Features to Signals

**Band:** 1
**Implementability:** HIGH

## Learning Objectives

After this week, students will be able to:

1. Implement the Gu-Kelly-Xiu cross-sectional prediction framework end-to-end: take Week 3's firm characteristics as features, train gradient boosting and neural network models, and generate stock-level alpha signals
2. Evaluate alpha signals using the IC-based metrics that industry practitioners actually use — information coefficient (IC), rank IC, IC information ratio (mean IC / std IC), and percentage of positive-IC months — and explain why IC = 0.05 is good, not "only 5%"
3. Build a long-short equity portfolio from model predictions (long the top decile, short the bottom decile) and compute its Sharpe ratio, turnover, and cumulative return
4. Train models with financial loss functions (IC-based, Sharpe-based) instead of MSE, and demonstrate empirically that rank-order losses outperform magnitude-based losses for cross-sectional prediction
5. Extend Week 3's feature matrix with interaction features, non-linear transformations, and feature importance analysis (permutation importance) to understand which characteristics drive alpha
6. Construct a rigorous model comparison framework: linear baseline (from Week 3) versus XGBoost versus LightGBM versus feedforward neural network, evaluated on identical temporal train/test splits with no look-ahead bias
7. Explain the landscape of alternative data (satellite imagery, web traffic, news sentiment, credit card transactions), articulate why most sources are paid, and identify text-based alternative data as the bridge to Week 7's NLP deep dive
8. Calibrate realistic performance expectations for ML alpha models on free data: IC in [0.03, 0.15], long-short Sharpe in [0.5, 1.5], and out-of-sample degradation of 30-50% as the norm

## Prerequisites

- **Required:** Week 2 (return computation, stationarity, panel data structure, time-series regression mechanics), Week 3 (cross-sectional feature matrix — size, book-to-market, profitability, investment, momentum — portfolio-sorting methodology, Fama-MacBeth intuition, out-of-sample validation concepts, the factor zoo problem). Week 3's homework `FactorBuilder` class produces the feature matrix that Week 4 rebuilds. Students should have a working pipeline that downloads fundamentals from yfinance, computes monthly firm characteristics, and outputs a stock-month panel with at least 5 characteristics.
- **Assumed ML knowledge:** Gradient boosting (XGBoost/LightGBM APIs, tree-based feature importance, regularization via max_depth/num_leaves), feedforward neural networks (PyTorch or equivalent — hidden layers, batch normalization, dropout, custom loss functions), scikit-learn's train/test split and cross-validation APIs, hyperparameter tuning concepts (GridSearchCV, early stopping). Students do NOT need to be taught how gradient boosting or neural nets work — they need to learn HOW THESE TOOLS BEHAVE ON FINANCIAL DATA and WHY standard ML evaluation metrics are wrong for this domain.
- **Prerequisite explanations in this week:** IC-based evaluation framework (15 min, Lecture Section 2): What is the information coefficient, why it replaces R-squared for cross-sectional prediction, the relationship between IC and the Information Ratio via the Fundamental Law of Active Management (IR ≈ IC * sqrt(breadth)), and why IC = 0.05-0.10 is economically significant despite sounding small. This is the evaluation paradigm shift — from "how much variance do I explain?" (R-squared) to "how well do I rank stocks?" (IC). Also: brief intro to financial loss functions (10 min, Lecture Section 4) — why MSE is suboptimal when you care about rank order, and how to implement differentiable IC and Sharpe approximations as training objectives.

## Narrative Arc

Week 3 ended with a powerful but frustrating conclusion: factor models explain cross-sectional returns, but they're LINEAR. You built a feature matrix with five firm characteristics, ran Fama-MacBeth regressions, and found statistically significant risk premia. But linear models assume that each characteristic's effect on returns is independent and additive — that the value premium is the same for small-caps and large-caps, that momentum works identically for profitable and unprofitable firms. Real markets are not that simple. The factor zoo paper showed that hundreds of proposed linear factors are noise. The question that opens Week 4 is: **can machine learning learn the non-linear interactions among firm characteristics that linear models miss, and does this actually translate into better alpha signals?**

The answer is nuanced, and the nuance IS the lesson. Week 4 is organized around the Gu-Kelly-Xiu (2020) framework — the landmark paper that systematically compared linear regression, gradient boosting, and neural networks for cross-sectional return prediction using firm characteristics as features. Their finding: tree-based models and neural networks roughly double the out-of-sample R-squared relative to linear regression (R-squared of 3-5% versus less than 1%). That sounds transformative until you translate it into the metrics that practitioners actually use: an IC improvement from approximately 0.04 to 0.07 — small in absolute terms, but economically significant because you're making this bet across hundreds of stocks every month. The "aha moment" of this week is recalibrating what "good" means in financial ML. An IC of 0.07 does not sound impressive to someone used to 95% accuracy in image classification. But applied to 500 stocks monthly, it compounds into a long-short portfolio with a Sharpe ratio above 1.0 — a result that would get you hired at a systematic fund.

The week resolves a second tension: ML models are more powerful than linear regressions, but they are also more dangerous. Overfitting is the dominant failure mode — a model that achieves IC = 0.20 in-sample will almost certainly degrade to IC = 0.05 out-of-sample (and possibly to zero). Week 4 introduces this problem through honest evaluation: temporal train/test splits (never shuffle cross-sectional data through time), in-sample versus out-of-sample IC comparison, and the discipline of always benchmarking against a linear baseline. If your XGBoost model doesn't beat linear regression by at least IC = 0.02 out-of-sample, the added complexity isn't justified. Week 5 will formalize this discipline with purged cross-validation and deflated Sharpe ratios, but Week 4 plants the seed: in financial ML, baselines are sacred, and honest evaluation is the difference between a research insight and an expensive hallucination.

## Lecture Outline

### Section 1: From Factor Models to ML — The Cross-Sectional Prediction Framework

- **Hook:** "Last week you ran Fama-MacBeth regressions and found that size, value, and momentum predict cross-sectional returns. But Fama-MacBeth is a linear model — it assumes the value premium is the same for a $500M mid-cap and a $500B mega-cap. In 2020, Gu, Kelly, and Xiu published the definitive study testing whether ML can do better. They compared six methods on 30,000 US stocks over 60 years. The headline: gradient boosting and neural networks roughly doubled the predictive power of linear regression. But the absolute numbers are humbling — R-squared went from less than 1% to 3-5%. Welcome to financial ML, where small numbers move billions of dollars."
- **Core concepts:** The cross-sectional prediction paradigm: at each month t, use firm characteristics known at month t-1 (size, book-to-market, momentum, profitability, investment — the exact features from Week 3) to predict the cross-section of stock returns at month t. This is NOT time-series forecasting (predicting one stock's future price from its past). It is ranking prediction — which stocks will outperform this month, relative to the universe. The Gu-Kelly-Xiu model hierarchy: (1) OLS — the linear baseline, equivalent to Fama-MacBeth with all characteristics simultaneously, (2) Elastic net — regularized linear, handles multicollinearity, (3) Gradient boosted trees — captures non-linear relationships and interactions automatically, (4) Feedforward neural network (3-5 hidden layers with batch normalization and dropout) — learns complex functional forms from data. The key insight from Gu-Kelly-Xiu: trees and neural nets improve prediction primarily by learning INTERACTION EFFECTS (value works differently for small-caps than large-caps) and NON-LINEAR TRANSFORMATIONS (extreme momentum has a different effect than moderate momentum). They do NOT discover fundamentally new factors — the same firm characteristics from Week 3 are the inputs. ML's value is in how it combines them, not what it combines.
- **Key demonstration:** Build the same feature matrix that Week 3's `FactorBuilder` produced (500 stocks, 10 years of monthly data, 5 characteristics: size, book-to-market, profitability, investment, momentum). Show the DataFrame: rows are stock-months, columns are characteristics plus next-month return (the target). Fit three models on the first 7 years: (1) OLS regression (using statsmodels or sklearn LinearRegression), (2) XGBoost with default hyperparameters (max_depth=4, n_estimators=100, learning_rate=0.1), (3) A 2-layer feedforward neural network (64-32 hidden units, ReLU, batch normalization, dropout=0.1, trained for 50 epochs with Adam). Generate cross-sectional predictions for the held-out 3 years. For now, evaluate with R-squared to connect to Week 3's metrics — then immediately show why R-squared is inadequate (Section 2 will introduce IC). Print a table: Model, In-Sample R-squared, Out-of-Sample R-squared. Expected results: OLS R-squared approximately 0.5-1.0%, XGBoost approximately 1.5-3.0%, Neural Net approximately 1.0-2.5%. All numbers are TINY — and that's the point.
- **Bridge:** "R-squared of 3% looks useless. But R-squared is the wrong metric for cross-sectional prediction. You don't need to predict the magnitude of returns — you need to predict the RANK ORDER. The metric that captures this is the Information Coefficient, and it changes everything."
- **Acceptance criteria:**
  - Feature matrix loaded with >= 400 stocks and >= 80 months of data
  - Three models trained: OLS, XGBoost, neural network
  - Temporal train/test split used (first 70% of months for training, last 30% for testing — NO random shuffling across time)
  - In-sample R-squared: OLS in [0.002, 0.020], XGBoost in [0.010, 0.080], Neural net in [0.005, 0.060]
  - Out-of-sample R-squared: OLS in [0.001, 0.015], XGBoost in [0.005, 0.040], Neural net in [0.003, 0.030]
  - Comparison table printed with all three models

### Section 2: The IC Framework — How the Industry Evaluates Alpha Signals

- **Hook:** "If you walked into a Monday morning meeting at Two Sigma and reported your model's R-squared, you'd get blank stares. The language of alpha signal evaluation is IC — Information Coefficient. It's the cross-sectional correlation between your predictions and realized returns, computed separately for each month and then aggregated over time. IC captures what matters for a long-short portfolio: can you rank stocks correctly? An IC of 0.05 sounds pathetic. Applied to 500 stocks monthly, it generates a Sharpe ratio above 0.7. That's a career."
- **Core concepts:** IC (Information Coefficient): at each month t, compute the Pearson correlation between model predictions and realized returns across all stocks. This gives one IC value per month — a time series of ICs. The key summary statistics: (1) Mean IC — the average predictive power, (2) IC standard deviation — how volatile the signal is, (3) IC Information Ratio = mean IC / std IC — the signal's consistency (analogous to Sharpe for the IC time series itself), (4) Percentage of positive-IC months — how often the signal is directionally correct (> 55% is meaningful, > 60% is strong). Rank IC (Spearman): use Spearman rank correlation instead of Pearson. More robust to outliers — a single stock with +300% return won't distort the metric. In practice, Rank IC is the default at most systematic funds. The Fundamental Law of Active Management: IR ≈ IC × sqrt(breadth), where IR is the Information Ratio (annualized return / tracking error of the active portfolio) and breadth is the number of independent bets per year. For monthly rebalancing across 500 stocks, breadth ≈ 500 × 12 = 6000 (though effective breadth is lower due to correlation). This means even IC = 0.05 translates to IR ≈ 0.05 × sqrt(6000) ≈ 3.9 — an extraordinarily high Information Ratio. The caveat: the formula assumes independent bets, which stocks are not (they're correlated). Effective IR is lower. But the directional insight holds: small IC × large breadth = economically significant signal. Calibrating expectations: IC = 0.03-0.05 is marginal but potentially tradeable. IC = 0.05-0.10 is good — most single equity factors live here. IC = 0.10-0.15 is excellent — multi-factor ML models can achieve this. IC > 0.20 is almost certainly overfit or look-ahead-biased. IC > 0.30 is "snake oil" — run away.
- **Key demonstration:** Take the three models from Section 1 and re-evaluate them with IC metrics. For each model and each out-of-sample month, compute: (a) Pearson IC, (b) Spearman Rank IC. Produce a time-series plot of monthly IC for all three models overlaid. Compute and display a summary table: Model, Mean IC, IC Std, IC IR (mean/std), % Positive IC Months, Mean Rank IC. Expected results: OLS Mean IC approximately 0.02-0.05, XGBoost Mean IC approximately 0.04-0.08, Neural Net Mean IC approximately 0.03-0.07. Then compute the long-short portfolio: at each month, sort stocks by model prediction, go long top decile (top 10%), short bottom decile. Compute monthly long-short returns, annualize, and report the Sharpe ratio. Show that even with modest IC, the long-short portfolio generates meaningful risk-adjusted returns.
- **Bridge:** "IC tells you HOW GOOD your signal is. The next question: HOW DO YOU MAKE IT BETTER? The answer isn't more data or bigger models — it's smarter features and financial loss functions."
- **Acceptance criteria:**
  - Monthly IC computed for all three models across all out-of-sample months
  - Both Pearson IC and Spearman Rank IC computed
  - Time-series IC plot shows visible variation across months and visible differences across models
  - IC summary table: Mean IC in [0.01, 0.15] for all models, XGBoost mean IC >= OLS mean IC (ML should improve on baseline)
  - % positive IC months > 50% for at least the XGBoost model
  - Long-short portfolio constructed: long top decile, short bottom decile, equal-weighted within each leg
  - Long-short Sharpe ratio computed and in [0.3, 2.0] for at least one model
  - Cumulative return plot of the long-short portfolio produced

### Section 3: Feature Engineering for Alpha — Interactions, Transformations, and Importance

- **Hook:** "Week 3 gave you five raw firm characteristics: size, value, momentum, profitability, investment. A linear model treats them independently. But in practice, momentum works differently for small-caps (where it's stronger) than large-caps (where it's weaker). Value interacts with profitability — cheap AND profitable is a much stronger signal than cheap alone. ML models can learn these interactions automatically, but you can also engineer them explicitly. The question is: which approach works better, and how do you know which features actually matter?"
- **Core concepts:** Three categories of feature enhancement: (1) Interaction features — products of pairs of characteristics (size × value, momentum × volatility, profitability × investment). These capture conditional relationships: "momentum conditional on size." Gradient boosting learns these automatically via tree splits, but explicit interactions can still improve neural networks and make the signal more interpretable. (2) Non-linear transformations — rank-transform each characteristic to [0, 1] (eliminates outlier effects), z-score within each cross-section (standardizes each month independently), quintile buckets (converts continuous to categorical — sometimes more robust for sparse data). The most impactful: cross-sectional rank transformation. It eliminates the problem of time-varying scale (market caps grew 10x over 20 years; ranks didn't). (3) Feature importance — permutation importance (shuffle one feature, measure IC drop) and built-in tree-based importance (XGBoost's feature_importance_). These answer: which features does the model actually use? If momentum and profitability contribute 60% of total importance and investment contributes 2%, that tells you where to focus research effort. SHAP values provide richer decomposition (feature contribution per prediction) but are computationally expensive — mentioned here, covered in depth in Week 10 (Causal Inference & Interpretable ML).
- **Key demonstration:** Start with Week 3's 5 raw characteristics. Add 5 interaction features (size × value, size × momentum, value × profitability, momentum × volatility [use 60-day realized vol from Week 2], profitability × investment). Add rank-transformed versions of all 5 raw characteristics. Total features: 5 raw + 5 interactions + 5 ranked = 15. Re-train XGBoost on the expanded feature set. Compare IC to the 5-feature model from Section 1. Compute permutation importance for all 15 features: for each feature, shuffle it across stocks within each month, re-predict, and measure the IC drop. Plot a horizontal bar chart of feature importances. Expected result: momentum and value are likely the top features, interactions add marginal improvement (IC gain of 0.01-0.02), and some interactions have near-zero importance (can be pruned).
- **Bridge:** "Better features improve the signal. But there's a more fundamental lever: the loss function. Standard ML models minimize MSE — prediction error in MAGNITUDE. For cross-sectional prediction, you care about RANK ORDER, not magnitude. Financial loss functions align the training objective with the evaluation metric."
- **Acceptance criteria:**
  - Expanded feature matrix has >= 15 features (5 raw + 5 interactions + 5 transformations)
  - XGBoost re-trained on expanded features using same temporal split
  - IC comparison: expanded-feature model IC >= 5-feature model IC (improvement or at least no degradation)
  - Permutation importance computed for all 15 features
  - Feature importance bar chart produced with features sorted by importance
  - Top 3 features account for >= 40% of total importance (signal is concentrated, not diffuse)
  - At least one interaction feature has importance > 0 (interactions contribute)
  - Rank-transformed features have different importance than raw features (transformation matters)

### Section 4: Financial Loss Functions — Training for What Matters

- **Hook:** "Your XGBoost model minimizes squared error. But you're evaluated on IC — rank-order correlation. These aren't the same thing. A model that perfectly predicts the magnitude of returns gets a perfect IC. But a model that perfectly ranks stocks (top performer predicted highest, bottom performer predicted lowest) also gets a perfect IC — even if the predicted magnitudes are completely wrong. For long-short portfolios, rank order is all that matters. Can we train the model to maximize IC directly?"
- **Core concepts:** Three loss functions for financial ML: (1) MSE (standard) — minimizes (predicted return - actual return)^2. Cares about magnitude. Penalizes large errors more than small ones. The default in every ML library. (2) IC-based loss — maximizes the cross-sectional correlation between predictions and actuals within each training batch. Implementation: compute the negative Pearson (or Spearman) correlation as the loss and backpropagate. Works naturally with neural networks (differentiable). For XGBoost, approximate via a pairwise ranking loss (LambdaMART-style). (3) Sharpe-based loss — directly maximizes the Sharpe ratio of the resulting long-short portfolio. Implementation: at each training step, form a long-short portfolio from predictions, compute its Sharpe ratio, and maximize it. More computationally expensive but directly aligned with the economic objective. Practical trade-offs: IC-based loss is the most common in practice because it's simple and well-behaved (smooth optimization landscape). Sharpe-based loss can overfit to the training period's volatility regime. MSE is a reasonable default — it correlates with IC in practice because models that explain more variance tend to rank better too. The empirical finding: IC-based loss typically improves Rank IC by 0.01-0.03 over MSE on cross-sectional data. The improvement is modest but consistent.
- **Key demonstration:** Train three feedforward neural networks (identical architecture: 3 hidden layers, 64-32-16, batch norm, dropout=0.1, Adam optimizer) with three different loss functions: (1) MSE loss, (2) Negative Pearson IC loss (compute IC within each batch, negate it so the optimizer minimizes), (3) Differentiable Sharpe loss (compute batch Sharpe of long-short portfolio from predictions). Train all three for 100 epochs on the same training data. Evaluate all three on the held-out period using Rank IC and long-short Sharpe. Display a comparison table: Loss Function, Mean Rank IC, Long-Short Sharpe, Training Time. Expected result: IC-based loss produces the highest Rank IC, Sharpe-based loss produces the highest portfolio Sharpe (but may overfit), MSE is a close third. The differences are modest (0.01-0.03 in IC) but statistically meaningful over many months.
- **Bridge:** "We've improved features and loss functions. But every improvement must be measured against a baseline — and the baseline is linear regression from Week 3. Before we declare victory for ML, we need an honest head-to-head comparison."
- **Acceptance criteria:**
  - Three neural networks trained with three different loss functions (MSE, IC, Sharpe)
  - Identical architecture across all three (only the loss differs)
  - Evaluation on the same held-out test period
  - IC-based loss model has Rank IC >= MSE loss model Rank IC (IC loss should help, or at least not hurt)
  - Comparison table includes: Loss Function, Mean Rank IC, IC IR, Long-Short Sharpe, Training Time
  - All three models have Mean Rank IC > 0 (all learn something)
  - Prose explains WHY IC-based loss helps: "rank-order objectives align training with evaluation"

### Section 5: The Honest Comparison — ML vs. Linear Baselines

- **Hook:** "Here's the uncomfortable truth that every quant fund knows and few academics acknowledge: simple linear regression beats complex ML models more often than you'd expect. Not always. Not on every dataset. But often enough that no serious researcher ships an ML model without first proving it beats the linear baseline. If your XGBoost can't improve on OLS by at least IC = 0.02 out-of-sample, the complexity isn't justified — and in production, complexity has real costs: longer training, harder debugging, more overfitting surface area."
- **Core concepts:** The honest comparison framework: (1) Identical data — same features, same universe, same temporal split, (2) Identical evaluation — same IC metrics, same long-short portfolio construction, same out-of-sample period, (3) Multiple baselines — equal-weight portfolio (no prediction at all), linear regression (Week 3's approach), elastic net (regularized linear), (4) ML models — XGBoost, LightGBM, feedforward neural network, (5) Overfitting diagnostic — compute in-sample IC and out-of-sample IC for each model; the ratio (OOS IC / IS IC) measures generalization. Trees and neural nets will show much higher IS IC than OLS (they can memorize) — the question is whether they maintain an advantage OOS. The recurring theme (appears in Weeks 4, 8, 15): simple models often beat complex ones in financial ML production. The reason: financial data has a very low signal-to-noise ratio (IC = 0.05 means 99.75% noise) and limited effective sample size (monthly data × 20 years = only 240 time periods). In this regime, complex models overfit easily.
- **Key demonstration:** Run the full comparison. Train 6 models on the same data: (1) Equal-weight benchmark (no prediction — just buy everything equally), (2) OLS, (3) Elastic net (alpha tuned by time-series CV on training data), (4) XGBoost (max_depth=4, n_estimators=200, learning_rate=0.05), (5) LightGBM (num_leaves=31, n_estimators=200, learning_rate=0.05), (6) 3-layer neural network (64-32-16, MSE loss). Evaluate all on the same 3-year out-of-sample period. Produce a comprehensive comparison table: Model, IS Mean IC, OOS Mean IC, OOS Rank IC, OOS IC IR, OOS % Positive IC Months, Long-Short Sharpe (OOS), OOS/IS IC Ratio (generalization). Expected pattern: XGBoost and LightGBM OOS IC approximately 0.05-0.08, OLS approximately 0.03-0.05, Neural net approximately 0.04-0.07. The ML improvement over OLS is real but modest (0.02-0.03 in IC). The generalization ratio is worse for ML (0.4-0.6) than for OLS (0.7-0.9).
- **Bridge:** "ML provides a genuine edge over linear models — but a modest one, and only with discipline. Before we close, let's briefly look at the future: alternative data sources that go beyond firm characteristics."
- **Acceptance criteria:**
  - Six models trained and evaluated on identical data and temporal split
  - Equal-weight benchmark included (long-short Sharpe should be near 0 — no prediction means no alpha)
  - OOS Mean IC: at least one ML model exceeds OLS by >= 0.01 (ML adds value)
  - OOS/IS IC ratio: ML models have lower ratio than OLS (more overfitting, as expected)
  - Comprehensive comparison table with all specified columns
  - Long-short Sharpe: at least one model achieves Sharpe >= 0.5 OOS
  - Generalization discussion: "XGBoost achieves the highest OOS IC but also shows the largest IS-to-OOS degradation, confirming that the capacity to memorize is a double-edged sword"

### Section 6: Alternative Data — A Landscape Tour (TOUCHED)

- **Hook:** "Everything we've done this week uses publicly available accounting data — balance sheets, income statements, market prices. Every hedge fund on earth has access to the same data. The competitive frontier in alpha research has shifted to alternative data: satellite images of parking lots (to predict retail earnings before they're announced), credit card transaction flows (to track consumer spending in real time), web traffic patterns (to forecast product demand), and — most importantly for ML practitioners — natural language from earnings calls, news articles, and social media. Alternative data is a $7 billion market growing at 30% annually. Most of it is expensive. But text data is increasingly accessible, and it's where NLP/AI engineers are making the biggest impact in quant finance."
- **Core concepts:** The alternative data taxonomy: (1) Textual — news articles, earnings call transcripts, SEC filings, social media, analyst reports. The MOST accessible alternative data for ML practitioners. FinBERT and LLM embeddings can extract sentiment, topics, and named entities. Week 7 covers this in depth. (2) Geospatial/satellite — parking lot fill rates, oil storage levels, agricultural crop health. High barriers to entry (expensive imagery, domain-specific processing). Used primarily by specialized quant funds (RS Metrics, Orbital Insight). (3) Transactional — credit card spending, point-of-sale data, app download counts. Very expensive ($100K-$1M/year). Providers: Second Measure, Earnest Research. (4) Web/digital — web traffic (SimilarWeb), app usage (Sensor Tower), job postings (Burning Glass), patent filings. Medium cost; some free proxies exist. (5) Sensor/IoT — shipping container movements (MarineTraffic — free for basic data), flight tracking (FlightAware), energy grid data. Data access challenges: most alternative data is PAID (tens of thousands to millions of dollars per year), requires specialized infrastructure (large storage, NLP pipelines, image processing), and has a short research half-life (once a dataset is widely adopted, its alpha decays). The free data ecosystem: yfinance (prices + fundamentals, Week 3-4), FRED (macroeconomic data), SEC EDGAR (filings — free, but raw text requires NLP processing), Google Trends (free proxy for attention), Wikipedia page views (free proxy for retail investor interest). Forward pointer: Week 7 will build a complete NLP pipeline for earnings call sentiment analysis and combine textual features with the numerical features from this week.
- **Key demonstration:** Show a BRIEF, conceptual example — do NOT build a full alternative data pipeline (that's Week 7). Download Google Trends data for 5 ticker symbols (using the `pytrends` library or a pre-cached CSV). Show the time series of search interest. Compute the cross-sectional correlation between search interest changes and next-month returns for the 5 stocks across 3 years. The IC will be noisy and near zero — this is the point: a single free alternative data proxy, without NLP sophistication, is weak. The alpha in alternative data comes from (a) proprietary sources with genuine information (paid), (b) sophisticated NLP processing of textual data (Week 7), or (c) combining many weak signals (ensemble approach). End with a clear message: alternative data is an alpha SOURCE (a type of feature input), not an alpha MODEL. The modeling framework from Sections 1-5 of this lecture applies identically whether your features come from balance sheets or satellite imagery.
- **Bridge:** This section is intentionally brief. The takeaway is landscape awareness, not implementation depth.
- **Acceptance criteria:**
  - Alternative data taxonomy presented with at least 5 categories and 2+ examples per category
  - Cost barriers mentioned explicitly ($100K+ for most institutional sources)
  - At least one free alternative data proxy demonstrated (Google Trends or equivalent)
  - IC between the free proxy and returns is near zero or very weak (honest demonstration — free proxies are weak)
  - Forward pointer to Week 7 (NLP for alpha) stated explicitly
  - Total section length: 10-15 minutes, not more

### Closing

- **Summary table:**

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| Cross-sectional prediction | Predict RELATIVE returns across stocks at each time period | The modern paradigm — replaces time-series price prediction |
| Information Coefficient (IC) | Cross-sectional correlation between predictions and returns | THE evaluation metric for alpha signals — replaces R-squared |
| Rank IC (Spearman) | IC computed on ranks instead of raw values | More robust to outliers; the industry default |
| IC Information Ratio | Mean IC / Std IC | Measures signal consistency, not just average strength |
| Fundamental Law | IR ≈ IC × sqrt(breadth) | Small IC × many stocks = economically significant alpha |
| Financial loss functions | IC-based, Sharpe-based training objectives | Align what the model optimizes with what you actually care about |
| Feature interactions | Products of characteristics (size × value) | Capture conditional relationships linear models miss |
| Permutation importance | Shuffle one feature, measure IC drop | Identifies which characteristics drive the model's predictions |
| Long-short portfolio | Long top decile, short bottom decile, by model prediction | Translates a statistical signal into a tradeable strategy |
| Alternative data | Non-traditional data sources (text, satellite, transactions) | The competitive frontier — most alpha from traditional data is crowded |

- **Career connections:**
  - A quant researcher at a stat-arb fund (Two Sigma, Citadel, DE Shaw) runs exactly this pipeline every week: update the feature matrix with Friday's data, retrain XGBoost and neural net models, generate Monday's alpha signals, compute IC against last month's predictions, and flag any signals where IC has decayed below threshold. The Sections 1-5 workflow is the literal research process.
  - An ML engineer at a systematic fund maintains the feature engineering pipeline (Section 3), model training infrastructure (Sections 1, 4), and evaluation dashboards (Section 2). The IC monitoring system — tracking rolling IC, alerting when signals degrade — is 30% of the ML engineer's daily work.
  - A data scientist evaluating alternative data sources (Section 6) uses the IC framework from Section 2 as the standard test: does this new data source (satellite imagery, credit card data) produce a signal with IC > 0.03 after controlling for traditional factors? If yes, it's worth the subscription cost. If no, move on.
  - An NLP/AI engineer (forward pointer to Week 7) will apply the exact same IC-based evaluation from this week to textual features extracted from earnings calls and news. Week 4's evaluation framework is the bridge that connects ANY feature source to the alpha pipeline.

- **Bridge to Week 5:** "You now have ML models that generate alpha signals evaluated by IC. But we used a single temporal train/test split — the simplest possible validation. How confident are you that the OOS IC of 0.06 isn't just lucky? Week 5 introduces rigorous backtesting: purged cross-validation (which accounts for autocorrelation in financial returns), the deflated Sharpe ratio (which corrects for multiple testing), and transaction cost modeling (because a high-IC signal with 200% monthly turnover is worthless after costs). Week 4 built the signal. Week 5 will stress-test it."

- **Suggested reading with annotations:**
  - **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning," *Review of Financial Studies*, 33(5), 2223-2273** — THE reference for this week. Compares 6 ML methods on 30,000 US stocks over 60 years. Read Sections 1-3 and the tables. The empirical benchmark that all subsequent financial ML papers cite.
  - **Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns," *Review of Financial Studies*, 29(1), 5-68** — The factor zoo paper from Week 3, now re-read with ML eyes: the 316 factors they catalog are potential features for ML models, and the multiple-testing problem they identify is the overfitting risk from Week 4.
  - **Lopez de Prado (2018), *Advances in Financial Machine Learning*, Chapters 6-7** — Chapter 6: ensemble methods for finance. Chapter 7: cross-validation in finance (preview of Week 5). Practitioners' perspective on why standard ML CV fails in finance.
  - **Dixon, Halperin & Bilokon (2020), *Machine Learning in Finance: From Theory to Practice*, Chapters 3-4** — Comprehensive textbook treatment of supervised learning for alpha. More rigorous than Lopez de Prado on the statistical theory.
  - **Chinco, Clark-Joseph & Ye (2019), "Sparse Signals in the Cross-Section of Returns," *Journal of Finance*, 74(1), 449-492** — Shows LASSO identifies predictive signals in high-dimensional feature spaces. Demonstrates that sparsity (most features don't matter) is a real property of financial data, not just a modeling assumption.
  - **Freyberger, Neuhierl & Weber (2020), "Dissecting Characteristics Nonparametrically," *Review of Financial Studies*, 33(5), 2326-2377** — Uses non-parametric methods (trees) to uncover non-linear characteristic-return relationships. Independent confirmation of Gu-Kelly-Xiu's main finding.

## Seminar Outline

### Exercise 1: XGBoost vs. LightGBM — Does the Gradient Boosting Implementation Matter?

- **Task type:** Skill Building
- **The question:** The lecture used XGBoost as the primary tree-based model. LightGBM (Microsoft) uses a different tree-growth strategy (leaf-wise vs. level-wise) and is 2-5x faster. Does this implementation difference affect alpha signal quality? Is there a speed-accuracy tradeoff that matters for production?
- **Data needed:** Week 3's feature matrix (500 stocks, >= 80 months, 5 characteristics), extended with 5 interaction features from the lecture (Section 3). Use a 70/30 temporal train/test split.
- **Tasks:**
    1. Train an XGBoost model with controlled hyperparameters: max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1
    2. Train a LightGBM model with approximately equivalent hyperparameters: num_leaves=15 (roughly equivalent to max_depth=4), n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1
    3. Evaluate both on the same held-out period: compute Mean IC, Rank IC, IC IR, % Positive IC Months, Long-Short Sharpe, and training time
    4. Vary one hyperparameter — n_estimators in {100, 200, 300, 500, 1000} — and plot the IC vs. training time tradeoff curve for both models. At what point does IC plateau for each?
    5. Extract and compare feature importances from both models. Do they agree on which features matter most?
    6. Produce a summary: "For our universe and feature set, [XGBoost/LightGBM] achieves IC of [X] in [Y] seconds. [The other] achieves IC of [X'] in [Y'] seconds. The speed difference is [Z]x while the IC difference is [W]."
- **Expected insight:** XGBoost and LightGBM produce very similar IC values (within 0.01 of each other) but LightGBM is typically 2-5x faster. Feature importances agree on the top 3 features but may differ on lower-ranked features. The IC vs. n_estimators curve plateaus around 200-500 trees for both — adding 1000 trees adds training time without improving OOS IC (and may slightly degrade it due to overfitting). The practical conclusion: for research iteration (where you retrain daily), LightGBM's speed advantage matters. For a one-time model, either works. Students learn that implementation choices matter for workflow efficiency, not statistical performance.
- **Acceptance criteria:**
  - Both models trained with matched hyperparameters
  - IC comparison: |XGBoost Mean IC - LightGBM Mean IC| < 0.02 (they should be close)
  - LightGBM training time < XGBoost training time for the same n_estimators (expected 2-5x speedup)
  - IC vs. n_estimators plot shows diminishing returns beyond 200-500 trees for both models
  - Feature importance comparison: top 3 features are the same for both models (or a clear explanation of why they differ)
  - Summary statement quantifies both the speed difference and the IC difference

### Exercise 2: The Noise Feature Trap — How Overfitting Hides in Feature Engineering

- **Task type:** Guided Discovery
- **The question:** You built 15 features in the lecture (5 raw + 5 interactions + 5 ranked). But what if some of those features are pure noise? How does adding noise features affect in-sample IC, out-of-sample IC, and feature importance? Can you detect which features are noise?
- **Data needed:** The same feature matrix as Exercise 1 (500 stocks, >= 80 months, 10 features: 5 raw characteristics + 5 interactions)
- **Tasks:**
    1. Train XGBoost on the 10 real features and record IS IC and OOS IC
    2. Add 5 "noise features": for each month, generate 5 columns of random Gaussian noise, independent of returns. Label them noise_1 through noise_5. Now you have 15 features (10 real + 5 noise).
    3. Re-train XGBoost on all 15 features. Record IS IC and OOS IC. Compare to the 10-feature model.
    4. Add 20 MORE noise features (total: 10 real + 25 noise = 35 features). Re-train. Record IS IC and OOS IC.
    5. For each model (10, 15, 35 features), compute permutation importance. Plot the feature importances for all features in the 35-feature model. Can you visually distinguish real features from noise features?
    6. Compute the "overfitting ratio": IS IC / OOS IC for each model. Plot: number of noise features (x-axis) vs. overfitting ratio (y-axis)
- **Expected insight:** In-sample IC INCREASES as you add noise features — the model finds spurious patterns in noise that happen to correlate with returns in the training set. Out-of-sample IC DECREASES or stays flat — the noise features don't help (and can actively hurt by consuming model capacity). The overfitting ratio rises from approximately 1.3 (10 features) to approximately 2.0+ (35 features). Permutation importance correctly assigns near-zero importance to noise features in most cases, but with 25 noise features, 1-2 may appear to have non-negligible importance by chance. The lesson: feature engineering is powerful but dangerous. Every new feature is a potential overfitting vector. The discipline of checking OOS performance AFTER adding features (not just IS performance) is essential. This previews Week 5's rigorous methodology.
- **Acceptance criteria:**
  - IS IC increases monotonically (or at least non-decreasingly) with the number of noise features: IS IC (35 features) >= IS IC (10 features)
  - OOS IC does NOT increase with noise features: OOS IC (35 features) <= OOS IC (10 features) + 0.01 (allowing small numerical noise)
  - Overfitting ratio (IS IC / OOS IC) increases with number of noise features
  - Permutation importance for noise features: average importance of noise features < average importance of real features
  - At least 4 of 5 noise features in the 15-feature model have importance in the bottom half of all features
  - Feature importance plot clearly labels real vs. noise features
  - Summary: "Adding 25 noise features increased IS IC by [X] but decreased OOS IC by [Y], demonstrating that in-sample improvement from noise is entirely spurious"

### Exercise 3: Temporal Stability of Alpha Signals — When Do Models Break?

- **Task type:** Investigation
- **The question:** You trained a model on 2012-2018 and tested it on 2019-2023. But does the model's predictive power degrade uniformly over time, or does it break during specific periods (COVID crash, 2022 rate hikes, momentum crashes)? Understanding WHEN your model fails is as important as knowing its average IC.
- **Data needed:** Feature matrix (500 stocks, January 2010 through December 2023, 10 features). Train on 2010-2017. Test on 2018-2023 (6 years of monthly OOS data).
- **Tasks:**
    1. Train XGBoost on 2010-2017 data. Generate monthly predictions for 2018-2023
    2. Compute monthly Rank IC for the entire OOS period. Plot the time series of monthly IC with a 6-month rolling average overlaid
    3. Identify the 5 worst IC months and the 5 best IC months. What was happening in markets during those months? (Use SPY returns as context: if SPY dropped 10%+ in a month, it was a crisis.)
    4. Compute rolling 12-month IC (average IC over the trailing 12 months) and plot it over time. Is there a visible decay trend? Does the model's average IC in 2022-2023 differ from 2018-2019?
    5. **Layer 2 (Smarter):** Split the OOS period into "calm" months (absolute SPY return < 3%) and "crisis" months (absolute SPY return > 5%). Compute mean IC separately for each regime. Does the model perform differently in calm vs. volatile markets?
    6. Compute the IC decay rate: fit a linear regression of rolling 12-month IC on time (in months). The slope estimates the rate of alpha decay. Is it statistically significant?
- **Expected insight:** IC is HIGHLY volatile across months — ranging from -0.10 to +0.15 in a typical OOS period. The rolling average is more stable but shows a mild downward trend (alpha decay). The worst IC months often coincide with market crises (March 2020, rate shock months in 2022) — cross-sectional models that learned pre-crisis patterns struggle when market regimes shift. The calm-vs-crisis comparison reveals that IC is typically 0.02-0.04 higher in calm markets than in crisis markets — factor relationships are more stable when markets behave "normally." The alpha decay rate is approximately -0.002 to -0.005 IC per year (about 5-10% annualized degradation), consistent with industry findings. This is a REALISTIC demonstration that alpha signals degrade over time and require periodic retraining.
- **Acceptance criteria:**
  - Monthly IC computed for all OOS months (>= 60 months)
  - Time-series IC plot with 6-month rolling average produced
  - 5 worst and 5 best IC months identified with market context (SPY return that month)
  - Rolling 12-month IC plotted; any downward trend visually identifiable or its absence noted
  - Layer 2: calm vs. crisis IC comparison computed with explicit regime definitions
  - Mean IC (calm) > Mean IC (crisis) by at least 0.01 (model performs better in calm markets)
  - IC decay regression: slope, t-stat, and R-squared reported
  - Summary: "The model shows [statistically significant / insignificant] alpha decay of approximately [X] IC per year. Performance is [Y] IC higher in calm markets than in crisis months."

### Exercise 4: Neural Networks for Cross-Sectional Prediction — Architecture Search

- **Task type:** Skill Building
- **The question:** The lecture used a simple 3-layer feedforward network. In standard ML, deeper and wider networks often perform better. Does this hold for cross-sectional return prediction, where signal-to-noise is extremely low and effective sample sizes are small?
- **Data needed:** Same feature matrix as previous exercises (500 stocks, 80+ months, 10 features). 70/30 temporal split.
- **Tasks:**
    1. Define 5 neural network architectures of increasing complexity: (a) Shallow: 1 hidden layer, 32 units, (b) Medium: 2 layers, 64-32, (c) Deep: 3 layers, 128-64-32, (d) Wide: 2 layers, 256-128, (e) Deep+Wide: 4 layers, 256-128-64-32. All use ReLU, batch normalization, dropout=0.1, Adam optimizer, MSE loss.
    2. Train each architecture for 200 epochs with early stopping (patience=20, monitor validation IC on the last 20% of training data)
    3. Record for each: OOS Mean IC, OOS Rank IC, IS IC, number of trainable parameters, training time, epoch at which early stopping triggered
    4. Plot two figures: (a) OOS Rank IC vs. number of parameters (does bigger = better?), (b) IS IC vs. OOS IC for each architecture (which architectures overfit most?)
    5. Compare the best neural network to XGBoost from Exercise 1 on the same data and metrics
- **Expected insight:** For cross-sectional return prediction on 500 stocks with 10 features, larger neural networks do NOT consistently improve OOS IC. The shallow (32-unit) and medium (64-32) networks often match or beat the deep/wide architectures OOS, despite having much worse IS IC. The deep+wide architecture (256-128-64-32) achieves the highest IS IC but shows the worst IS-to-OOS degradation — it memorizes training patterns that don't generalize. Early stopping triggers EARLIER for smaller networks (they converge faster) and LATER for larger networks (they keep finding patterns in training data, but these patterns are noise). XGBoost typically beats all neural network architectures on tabular cross-sectional data with this many features — consistent with the well-documented finding that tree-based models dominate neural nets on tabular data (the "tabular data" problem in ML). Neural nets will find their advantage in Weeks 7-8 (text data, time-series data), where the data structure favors them.
- **Acceptance criteria:**
  - All 5 architectures trained and evaluated on identical data
  - OOS Rank IC vs. parameters plot shows no clear monotonic improvement with model size (bigger is NOT always better)
  - IS IC increases with model size (larger models fit training data better)
  - IS-to-OOS ratio is worst for the largest architecture (most overfitting)
  - Early stopping epoch reported for all architectures; smaller architectures stop earlier
  - XGBoost comparison: XGBoost OOS Rank IC >= best neural net OOS Rank IC - 0.01 (trees likely win or tie on tabular data)
  - Summary: "On tabular cross-sectional data with 10 features and 500 stocks, a 2-layer network with 64-32 units achieves the best OOS IC. Increasing to 4 layers with 256-128-64-32 units improves IS IC by [X] but degrades OOS IC by [Y]."

## Homework Outline

### Mission Framing

You are building your first end-to-end ML alpha research pipeline. The head of research at your fund has given you a clear mandate: "Take the factor data from Week 3, apply ML to generate alpha signals, and show me — with honest evaluation — whether ML beats the linear baseline. I don't want to see an overfit IC of 0.25 that disappears out-of-sample. I want to see a realistic IC of 0.06 that I can trust, with a proper model comparison, feature importance analysis, and a long-short portfolio that I'd be willing to paper-trade."

This homework differs from the seminar in three ways: (1) SCALE — you'll build a full pipeline that handles data loading, feature engineering, model training, evaluation, and portfolio construction in a single modular workflow, not isolated experiments. (2) RIGOR — you'll implement a proper multi-model comparison with standardized evaluation, not one-off model fits. (3) PRODUCTION DISCIPLINE — your code must be modular (separate functions for each stage), reproducible (fixed random seeds), and documented (docstrings explaining every design choice). The pipeline you build here will be stress-tested in Week 5 (backtesting) and used as an input to Week 6 (portfolio construction).

The deliverables form a coherent narrative: first build the prediction engine (Deliverable 1), then evaluate it honestly (Deliverable 2), then extract insights about what drives alpha (Deliverable 3), and finally assess whether the edge is real or illusory (Deliverable 4).

### Deliverables

1. **An `AlphaModelPipeline` class**
   - **Task type:** Construction
   - Build a reusable class that encapsulates the full alpha research workflow:
     - **Data ingestion:** Accept a feature matrix (DataFrame with stock-month rows, characteristic columns, and a next-period-return target column). Validate inputs: check for look-ahead bias (features at time t should not use return at time t), check for missing values (report percentage, apply median imputation per cross-section per month), check for sufficient universe size (warn if fewer than 100 stocks in any month).
     - **Feature engineering:** Accept raw characteristics from Week 3 and optionally expand them with: (a) interaction features (all pairwise products of raw features), (b) cross-sectional rank transformations (rank each feature within each month to [0, 1]), (c) cross-sectional z-score normalization. Return an expanded feature matrix with configurable feature sets.
     - **Model training:** Support at least 4 model types: OLS (sklearn LinearRegression), Elastic Net (sklearn ElasticNet with configurable alpha), XGBoost (configurable hyperparameters), and a feedforward neural network (configurable architecture, loss function choice: MSE or IC-based). Use temporal train/test splits with a configurable train window and test window. Set random seeds for reproducibility.
     - **Signal evaluation:** For each model, compute the full IC evaluation suite: monthly IC (Pearson), monthly Rank IC (Spearman), Mean IC, IC Std, IC IR, % positive IC months, cumulative IC time series. Also construct a long-short portfolio (configurable decile: top/bottom 10%, 20%, or 30%) and compute: monthly returns, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, turnover.
     - **Model comparison:** Return a comparison DataFrame with one row per model and columns for all evaluation metrics. Support plotting: IC time series (all models overlaid), cumulative long-short returns (all models overlaid), feature importance bar charts.
   - **Acceptance criteria:**
     - Class accepts a feature matrix with >= 400 stocks and >= 80 months
     - Feature engineering produces >= 15 features from 5 raw characteristics (with interactions and ranks)
     - All 4 model types train without error: OLS, Elastic Net, XGBoost, Neural Net
     - IC evaluation suite computed for each model: Mean IC, Rank IC, IC IR, % positive months all present
     - Long-short portfolio computed with configurable decile sizes
     - Comparison DataFrame has one row per model, >= 8 metric columns
     - All random seeds set — re-running the pipeline produces identical results
     - Input validation catches: (a) future-looking features, (b) excessive missing values (> 50%), (c) insufficient universe size (< 50 stocks in any month)
     - Pipeline runs end-to-end in < 10 minutes on a single CPU (reasonable performance)

2. **A multi-model comparison report**
   - **Task type:** Skill Building
   - Using your `AlphaModelPipeline`, run the full comparison on S&P 500 data:
     - **Universe:** S&P 500 constituents, monthly data, 2010-2023
     - **Features:** 5 raw characteristics from Week 3 (size, book-to-market, profitability, investment, momentum) + 10 interaction features + 5 rank-transformed features = 20 features
     - **Models:** (1) Equal-weight benchmark, (2) OLS, (3) Elastic Net, (4) XGBoost, (5) LightGBM, (6) 3-layer neural network with MSE loss, (7) 3-layer neural network with IC-based loss
     - **Temporal split:** Train on 2010-2018, test on 2019-2023 (5 years OOS)
     - **Output:** Comparison table, IC time-series plot (all 7 models), cumulative long-short return plot (all 7 models), one-paragraph interpretation per model
   - **Acceptance criteria:**
     - All 7 models trained and evaluated on the same data
     - OOS evaluation period is >= 48 months (4+ years)
     - At least one ML model (XGBoost, LightGBM, or neural net) has OOS Mean IC > OLS Mean IC by >= 0.01
     - ML model with IC-based loss has OOS Rank IC >= ML model with MSE loss Rank IC (IC loss should help)
     - Equal-weight benchmark has Long-Short Sharpe near 0 (no alpha from random selection)
     - Best ML model achieves Long-Short Sharpe >= 0.5 OOS
     - IC time-series plot shows all 7 models with distinguishable lines
     - Cumulative return plot shows separation between ML models and linear baseline
     - Interpretation paragraphs are specific: "XGBoost achieves mean IC of [X] vs. OLS mean IC of [Y], an improvement of [Z]. This translates to a long-short Sharpe improvement from [A] to [B]."

3. **A feature importance and interaction analysis**
   - **Task type:** Investigation
   - Investigate what the ML models actually learn — which features matter, and do interactions provide genuine alpha beyond raw characteristics?
   - **Layer 1 (Baseline):** For the best-performing model from Deliverable 2, compute permutation importance for all 20 features. Rank features by importance. Plot a horizontal bar chart. Identify the top 5 features and bottom 5 features.
   - **Layer 2 (Smarter):** Compare the importance rankings from XGBoost (built-in gain-based importance), XGBoost (permutation importance), and the neural network (permutation importance). Do the three methods agree on the top features? Where do they disagree, and why?
   - Run an ablation study: train XGBoost on (a) 5 raw features only, (b) 5 raw + 5 ranked features, (c) 5 raw + 10 interactions, (d) all 20 features. Compare OOS IC across the four configurations. Does the feature set that produces the best OOS IC also have the highest feature importance concentration (top 3 features account for the largest share)?
   - **Acceptance criteria:**
     - Permutation importance computed for all 20 features for at least one model
     - Feature importance bar chart produced with features sorted by importance
     - Top 5 features and bottom 5 features identified and discussed
     - Three importance methods compared: XGBoost gain, XGBoost permutation, neural net permutation
     - Agreement/disagreement between methods explicitly stated: "All three methods rank [feature X] as most important. They disagree on [feature Y]: XGBoost gain ranks it #3 while neural net permutation ranks it #8."
     - Ablation study: 4 feature-set configurations trained and evaluated
     - OOS IC comparison across ablation configurations presented in a table
     - Discussion: "Interaction features improve OOS IC by [X] / do not improve OOS IC, suggesting that [XGBoost/neural nets] [can/cannot] learn these interactions from raw features alone"

4. **An out-of-sample reality check**
   - **Task type:** Investigation
   - This deliverable answers the hardest question: is the ML edge real, or is it an artifact of our specific train/test split?
   - **Layer 1 (Baseline):** Take the best ML model (likely XGBoost) and the OLS baseline. Implement walk-forward evaluation: train on the first 60 months, predict month 61. Retrain on months 1-61, predict month 62. Continue through the end of the sample. This gives you a TRUE out-of-sample IC for every month (no single train/test split to cherry-pick). Compare walk-forward IC to the single-split IC from Deliverable 2. Are they consistent?
   - **Layer 2 (Smarter):** Compute the "ML improvement" at each month: IC_XGBoost(t) - IC_OLS(t). Plot this difference over time. Is the improvement consistent, or does ML only beat OLS in certain periods? Compute the t-statistic on the mean difference: is the improvement statistically significant (t > 2)?
   - Report: ML model vs. OLS, walk-forward Mean IC, walk-forward IC IR, walk-forward Long-Short Sharpe, IS/OOS degradation ratio (comparing the walk-forward OOS IC to a single full-sample IS IC). Interpret honestly: "The ML model achieves walk-forward Sharpe of [X] vs. OLS walk-forward Sharpe of [Y]. The improvement of [Z] is [statistically significant / not significant] (t = [W]). The IS/OOS degradation is [A]%, consistent with the 30-50% degradation that research literature predicts."
   - **Acceptance criteria:**
     - Walk-forward evaluation implemented with expanding training window
     - Walk-forward IC computed for >= 48 months (at least 4 years of OOS predictions)
     - Walk-forward Mean IC is within 0.03 of single-split OOS Mean IC (consistency check)
     - ML improvement time series (IC_XGBoost - IC_OLS) plotted with clear labeling
     - t-statistic on mean ML improvement computed and reported
     - IS/OOS degradation ratio: OOS IC is 50-80% of IS IC (30-50% degradation is normal)
     - Final report includes: walk-forward Mean IC, IC IR, Long-Short Sharpe, degradation ratio for both ML and OLS
     - Honest interpretation: if the ML improvement is not statistically significant, SAY SO. "The ML model outperforms OLS by IC = [X] on average, but this improvement is not statistically significant (t = [Y], p = [Z]). With free data and survivorship bias, we cannot conclusively claim ML superiority — a finding consistent with Gu-Kelly-Xiu's observation that ML's edge is real but modest."

### Expected Discoveries

- The `AlphaModelPipeline` reveals that the ML engineering is not the hard part — data cleaning and feature alignment dominate the effort. Students will spend more time ensuring that features at time t use only information available at t-1 than on model architecture choices. This mirrors real quant research, where data quality consumes 70-80% of the work.
- The multi-model comparison will show that XGBoost and LightGBM produce nearly identical OOS IC (within 0.01), that the neural network with IC-based loss edges out the MSE-loss version by 0.01-0.02 in Rank IC, and that ALL ML models beat OLS by a modest but consistent margin (IC improvement of 0.02-0.04). The equal-weight benchmark confirms that prediction skill exists — but it's small.
- The feature importance analysis will reveal that 2-3 characteristics (typically momentum and value, or momentum and profitability) dominate model predictions, accounting for 50-70% of total importance. Interaction features add marginal value — XGBoost can learn most interactions automatically via tree splits. Rank-transformed features have different importance patterns than raw features, confirming that preprocessing choices affect what the model learns.
- The walk-forward reality check will show IS/OOS degradation of 30-50%, consistent with academic findings. The ML improvement over OLS may or may not be statistically significant at conventional thresholds — this is an honest finding, not a failure. Students who achieve a t-statistic of 1.5-2.5 on the ML improvement should understand that this represents REAL but MARGINAL alpha, which is exactly what the quant industry operates on. The Sharpe ratio of the best long-short portfolio will be 0.5-1.5 — respectable but not spectacular, because the data is free (survivorship-biased, not point-in-time) and the feature set is canonical (no proprietary alpha). With institutional data (CRSP/Compustat) and proprietary features, these numbers would improve.

## Key Papers & References

- **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning," *Review of Financial Studies*, 33(5), 2223-2273** — The organizing framework for this week. Compares linear, tree, and neural net models for cross-sectional return prediction. The benchmark that all subsequent financial ML papers must beat.
- **Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns," *Review of Financial Studies*, 29(1), 5-68** — The factor zoo paper. 316 proposed factors, most don't replicate. Establishes the overfitting risk context for Week 4's ML models. Recommends t > 3.0 for new factor discovery.
- **Lopez de Prado (2018), *Advances in Financial Machine Learning*, Chapters 6-7** — Ensemble methods and cross-validation for financial ML. The practitioner's guide to avoiding overfitting in temporal data. Preview of Week 5's purged CV methodology.
- **Fama & French (1992), "The Cross-Section of Expected Stock Returns," *Journal of Financial Economics*, 32(2), 427-465** — The original cross-sectional study showing size and value predict returns. Foundation for Week 3's features and Week 4's ML targets.
- **Chinco, Clark-Joseph & Ye (2019), "Sparse Signals in the Cross-Section of Returns," *Journal of Finance*, 74(1), 449-492** — LASSO for high-dimensional financial feature selection. Demonstrates that most features are noise — consistent with Week 4's feature importance findings.
- **Freyberger, Neuhierl & Weber (2020), "Dissecting Characteristics Nonparametrically," *Review of Financial Studies*, 33(5), 2326-2377** — Non-parametric (tree-based) decomposition of characteristic-return relationships. Independent confirmation that non-linear methods outperform linear ones for cross-sectional prediction.
- **Dixon, Halperin & Bilokon (2020), *Machine Learning in Finance: From Theory to Practice*, Chapters 3-4** — Textbook treatment of supervised learning for alpha generation. More mathematically rigorous than Lopez de Prado. Good reference for neural network architecture choices.
- **Kolanovic & Krishnamachari (2017), "Big Data and AI Strategies," J.P. Morgan Report** — Sell-side overview of alternative data adoption in quant finance. Useful for Section 6's landscape context.

## Career Connections

- **Quant researcher (buy-side):** A quant researcher at a stat-arb fund (Two Sigma, Citadel, DE Shaw) runs exactly the pipeline from this week's homework every Monday. The weekly cycle: pull last week's data, update the feature matrix, retrain XGBoost and neural net models, generate this week's alpha signals, compute IC vs. predictions from last week, flag any signal where rolling 6-month IC has dropped below 0.03. The `AlphaModelPipeline` from Deliverable 1 is a simplified version of the production system. A researcher who can build this pipeline from scratch — and articulate why the IC of 0.06 is trustworthy — is demonstrating the exact skill set these firms hire for.

- **ML engineer at a systematic fund:** ML engineers at Point72, Millennium, or Balyasny maintain the infrastructure that runs researchers' models at scale: distributed feature computation across 10,000+ stocks, daily model retraining on GPU clusters, real-time IC monitoring dashboards, and automated alerts when a signal degrades. The engineering behind Deliverable 1's pipeline — modular design, reproducibility, input validation — is 40% of the ML engineer role. The other 60% is productionizing it (model versioning, A/B testing new features, latency optimization).

- **Data scientist exploring alternative data:** A data scientist at a fund evaluating alternative data sources (Section 6) uses Week 4's IC framework as the standard acceptance test: "Does this satellite imagery dataset produce a cross-sectional signal with Rank IC > 0.03 after controlling for momentum, value, and size?" The IC evaluation suite from Deliverable 2 is literally the tool used to approve or reject $200K/year data subscriptions.

- **NLP/AI engineer (bridge to Week 7):** The alternative data section (Section 6) is a preview of the NLP/AI engineer's primary domain. In Week 7, students will build a full NLP pipeline: extract sentiment features from earnings call transcripts using FinBERT, combine them with the numerical features from this week, and evaluate the combined model using the same IC framework. The IC improvement from adding textual features on top of fundamental features is typically 0.02-0.04 — modest but highly valued because it's ORTHOGONAL to traditional factors (a new source of alpha, not a repackaging of existing signals).

## Data Sources

- **yfinance (free, no API key):** All price and fundamental data for the S&P 500 universe. Used for: daily adjusted close prices, market capitalization (from `ticker.info` or price x shares outstanding), balance sheet data (total equity, total assets), income statement data (operating income, net income). The feature matrix uses the same characteristics that Week 3's `FactorBuilder` computed — size, book-to-market, profitability, investment, momentum — rebuilt here as a self-contained pipeline. Limitations: survivorship bias (current S&P 500 constituents only), fundamental data is not point-in-time (latest reported, not when-reported), rate limits on bulk fundamental data downloads (use retries and caching). These limitations are pedagogically valuable — they explain why institutional data (CRSP/Compustat at ~$10K/year) exists.
- **getfactormodels (free, no API key):** Ken French factor returns for benchmarking. Used to validate that the feature matrix's characteristics produce IC patterns consistent with known factor premia (e.g., momentum IC should be positive on average). Also useful for computing factor-adjusted alpha: "Does the ML model add value BEYOND what Fama-French factors already explain?"
- **pytrends or pre-cached CSV (free):** Google Trends data for the alternative data demonstration in Lecture Section 6. Optional — if `pytrends` is unreliable or rate-limited, use a pre-cached CSV of search interest for 5 tickers over 3 years.
- **Approximate data sizes:** 500 stocks x 13 years x 12 months = 78,000 stock-month rows. With 20 features + target = 21 columns. CSV: approximately 35 MB. Parquet: approximately 8 MB. The feature matrix should be pre-computed and cached (from Week 3 pipeline) to avoid re-downloading fundamentals. Total yfinance download time if starting from scratch: 10-15 minutes. With caching: < 1 minute to load.
- **Key libraries:** `xgboost` (v3.0+), `lightgbm` (latest stable), `scikit-learn` (v1.3+), `torch` or `tensorflow` (for neural network implementation), `alphalens` (v0.4.0 — maintenance mode but functional, for IC analysis validation), `matplotlib` / `seaborn` (plotting), `numpy` / `pandas` (computation). All free and pip-installable.
