# Week 5 — Tree-Based Methods & Gradient Boosting

> **If you learn only one ML method for finance, learn this one. XGBoost and LightGBM dominate production quant finance the way Postgres dominates backend engineering — quietly, reliably, everywhere.**

## Prerequisites
Week 4 is essential: the cross-sectional prediction framework, feature matrix construction, expanding-window CV, and the IC metric. Students should have a working feature matrix for 200+ stocks, understand quantile portfolio analysis, and have Ridge/Lasso results as baselines. Familiarity with decision trees and random forests from general ML is assumed; we build rapidly from there.

## The Big Idea

Last week, you built a linear model that predicts stock returns with an IC of about 0.02-0.04. That's genuine signal, but it leaves something on the table. The Gu-Kelly-Xiu paper showed that the gap between linear models and nonlinear models (trees, neural nets) comes from a specific place: interaction effects. The most important one is the momentum × volatility interaction — stocks with high momentum and low volatility outperform differently from stocks with high momentum and high volatility. A linear model treats momentum and volatility as additive. The market does not.

Gradient-boosted trees — XGBoost and LightGBM specifically — are the single most widely used ML model class in production quantitative finance. They won the Kaggle Two Sigma Financial Modeling challenge. They won the Jane Street Market Prediction competition. They power signal generation at firms managing hundreds of billions of dollars. There are deep reasons for this dominance: trees handle missing values natively (financial data is full of gaps), they capture nonlinear interactions without specification (you don't need to engineer feature crosses), they provide built-in feature importance (critical for model trust), and they're fast to train and tune (critical for daily or weekly retraining).

But trees are also dangerous in finance, in a way that's less commonly discussed. Their flexibility makes them aggressive overfitters when the signal-to-noise ratio is low — which in financial data, it always is. A tree with max_depth=10 can memorize the training set perfectly and produce beautiful in-sample results that are pure fiction. The entire art of using trees for finance is in the regularization: shallow trees, high learning rates, aggressive subsampling, and early stopping. You're not looking for the best-fitting model — you're looking for the model that barely fits, because in this domain, the signal is so weak that "barely fitting" is exactly the right amount.

This week, you'll extend your Week 4 pipeline with Random Forest, XGBoost, and LightGBM. You'll tune them with Optuna using expanding-window CV. And you'll use SHAP values to see inside the model — to understand exactly which features and feature interactions drive predictions. By the end, you'll have the model that most quant funds actually use in production, and you'll understand why.

## Lecture Arc

### Opening Hook

"In 2017, Two Sigma — a quantitative hedge fund managing $60 billion — hosted a Kaggle competition called 'Two Sigma Financial Modeling Challenge.' The prize: $100,000 and a recruitment pipeline. Over 2,000 teams competed. The winning solutions? Gradient-boosted trees. Not neural networks, not transformers, not anything exotic — XGBoost and LightGBM, carefully regularized, carefully tuned, applied to tabular financial features. In 2020, Jane Street hosted their own competition on Kaggle. Gradient-boosted trees dominated again. This isn't a coincidence. Trees have structural advantages for tabular financial data that no other model class matches. Today we're going to understand why — and build one that beats everything you've built so far."

### Section 1: From Decision Trees to Random Forests to Gradient Boosting
**Narrative arc:** A rapid but precise tour through the evolution from a single decision tree to gradient boosted ensembles. The setup: decision trees are interpretable but fragile. The tension: bagging (random forests) reduces variance but doesn't reduce bias. The resolution: boosting reduces bias iteratively, producing the most powerful tabular models in existence.

**Key concepts:**
- Decision tree: recursive binary splits, overfits easily
- Random Forest: bagging (bootstrap aggregation) + random feature subsets
- Gradient Boosting: sequential learning — each tree corrects the previous tree's errors
- The bias-variance lens: bagging reduces variance, boosting reduces bias
- XGBoost: regularized gradient boosting with second-order approximation
- LightGBM: histogram-based, leaf-wise growth, faster than XGBoost on large data
- CatBoost: handles categorical features natively (less relevant for our tabular features)

**The hook:** "A single decision tree is like a junior analyst making a flowchart: 'If momentum > 10% and volatility < 20%, buy.' It's easy to understand but usually wrong. A random forest is like polling 500 junior analysts, each seeing different data and features — the consensus is better than any individual. Gradient boosting is like having 500 analysts work in sequence: the first makes predictions, the second focuses on the first's mistakes, the third focuses on the remaining mistakes, and so on. By the 500th analyst, you've squeezed out every learnable pattern in the data. The risk is obvious — you might also squeeze out the noise and mistake it for signal."

**Key formulas:**
"Gradient boosting builds the prediction function additively:

$$\hat{y}_i^{(m)} = \hat{y}_i^{(m-1)} + \eta \cdot h_m(\mathbf{x}_i)$$

where $h_m$ is the $m$th tree, fitted to the residuals of the previous ensemble, and $\eta$ is the learning rate (typically 0.01-0.1).

XGBoost's objective at step $m$:

$$\mathcal{L}^{(m)} = \sum_{i=1}^{N} \left[ g_i h_m(\mathbf{x}_i) + \frac{1}{2} h_i h_m(\mathbf{x}_i)^2 \right] + \Omega(h_m)$$

where $g_i$ and $h_i$ are the first and second derivatives of the loss with respect to the prediction (gradient and Hessian), and $\Omega$ is the regularization term:

$$\Omega(h_m) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

$T$ is the number of leaves, $w_j$ is the leaf weight, $\gamma$ penalizes tree complexity, and $\lambda$ penalizes leaf weights. The second-order approximation is what makes XGBoost faster and more accurate than original gradient boosting — it's Newton's method instead of gradient descent."

**Code moment:** Show the three models trained on the same data:

```python
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Random Forest
rf = RandomForestRegressor(n_estimators=500, max_depth=5,
                            min_samples_leaf=50, n_jobs=-1)

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=1000, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    early_stopping_rounds=50
)

# LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0
)
```

Compare training times. LightGBM will be 2-5x faster than XGBoost, which will be 5-10x faster than Random Forest. On 200 stocks × 200 months × 20 features, training should take seconds for LightGBM.

**"So what?":** "Speed matters because we retrain monthly in expanding-window CV. A model that takes 10 minutes to train means the full backtest takes hours. LightGBM's speed is a genuine competitive advantage — it lets you iterate faster on features and hyperparameters."

### Section 2: Why Trees Dominate Tabular Financial Data
**Narrative arc:** We explain the structural advantages of trees for financial data specifically, connecting abstract model properties to concrete financial data characteristics.

**Key concepts:**
- Native handling of missing values (financial data is full of gaps)
- No need for feature scaling or normalization (trees split on rank, not magnitude)
- Automatic capture of nonlinear interactions (the momentum × volatility effect)
- Robustness to outliers (tree splits are rank-based)
- Built-in feature importance (for trust and interpretability)
- Fast inference (important for live trading systems)

**The hook:** "Financial data has a specific pathology that makes it uniquely suited to trees: it's tabular, heterogeneous, full of missing values, and the relationships between features are nonlinear and interactive. Neural networks need clean inputs, careful normalization, and large datasets to learn these interactions from scratch. Trees discover them by design — every split is a potential interaction. When XGBoost splits first on momentum, then on volatility, it's learned the momentum × volatility interaction without you ever specifying it."

**Key formulas:** No new formulas — this section is conceptual, connecting the tree structure to financial data properties.

**Code moment:** Demonstrate two advantages concretely. First, missing values: create a feature matrix with 20% NaN values and show that XGBoost handles them without imputation (it learns optimal default directions for missing values). Compare to Ridge, which requires imputation and is sensitive to the imputation method. Second, nonlinear interactions: train a tree on momentum and volatility, then show the partial dependence surface — it won't be a plane (as it would be for linear models) but a curved surface that shows the interaction.

**"So what?":** "These aren't theoretical advantages — they're practical ones. In production, financial data arrives with missing values, outliers, and nonstationary distributions. A model that handles these natively, without preprocessing, is more robust and requires less engineering effort. This is why trees are the default at most quant funds, even when neural nets have slightly higher IC on clean benchmark data."

### Section 3: XGBoost vs. LightGBM vs. CatBoost — When to Use Which
**Narrative arc:** A practical comparison of the three major gradient boosting libraries. The setup: they all implement the same algorithm. The tension: the implementations differ in ways that matter for financial data. The resolution: LightGBM is the default for most cases; XGBoost is the fallback when LightGBM doesn't converge; CatBoost is useful when you have categorical features.

**Key concepts:**
- XGBoost: level-wise tree growth, exact split finding, most regularization options
- LightGBM: leaf-wise growth (best-first), histogram binning, faster and often more accurate
- CatBoost: ordered boosting (avoids target leakage), native categorical support
- Level-wise vs. leaf-wise growth: leaf-wise finds better splits but overfits more easily
- When CatBoost's categorical handling matters: sector codes, month indicators

**The hook:** "LightGBM was created by Microsoft in 2017, and within a year it had become the default at most quant funds. The reason is simple: it's 2-5x faster than XGBoost and often slightly more accurate. The speed comes from histogram-based split finding (256 bins instead of exact splits) and leaf-wise tree growth (grow the most promising leaf first, not all leaves at the same depth). The accuracy comes from the same leaf-wise growth — it finds better splits. The risk is overfitting, which we handle with max_depth and min_child_samples."

**Code moment:** Train XGBoost, LightGBM, and CatBoost on the same data with similar hyperparameters. Compare: training time, out-of-sample IC, and IC standard deviation across expanding windows. LightGBM will likely be fastest and within 0.001 of the best IC.

**"So what?":** "Use LightGBM as your default. Switch to XGBoost if you need more regularization options or if LightGBM overfits on small datasets. Use CatBoost if you have meaningful categorical features (sector codes, exchange indicators). For most cross-sectional stock prediction tasks, LightGBM is the right choice."

### Section 4: Hyperparameter Tuning for Financial Data
**Narrative arc:** This is the craft section — the difference between a tree model that works and one that hallucinates. The setup: trees have many hyperparameters. The tension: in low-signal-to-noise financial data, most hyperparameter settings overfit. The resolution: aggressive regularization, early stopping, and systematic tuning with Optuna.

**Key concepts:**
- max_depth: the most important regularizer. 3-6 for financial data (not 10-15 as in other domains)
- learning_rate: 0.01-0.1. Lower = more robust, but needs more trees
- n_estimators + early_stopping: let the model decide when to stop
- subsample and colsample_bytree: random subsampling of rows and features per tree
- min_child_weight / min_samples_leaf: prevent trees from fitting to tiny subsets
- reg_alpha (L1) and reg_lambda (L2): leaf weight regularization
- Optuna for systematic tuning with expanding-window CV

**The hook:** "Here's a dirty secret of applied ML in finance: most of the performance difference between a good model and a bad model comes from hyperparameter tuning, not architecture. A poorly tuned XGBoost with max_depth=10 and learning_rate=0.3 will overfit spectacularly and produce an out-of-sample IC near zero. The same XGBoost with max_depth=4 and learning_rate=0.05 will produce IC of 0.03-0.05. Same algorithm. Same data. Same features. The difference is entirely in the regularization. In financial data, where signal-to-noise is 0.01, the model's job is to barely fit — and barely fitting requires careful tuning."

**Key formulas:**
"The typical hyperparameter search space for financial data:

| Parameter | Range | Typical Best | Why |
|-----------|-------|-------------|-----|
| max_depth | 3-6 | 4 | Prevents memorizing noise |
| learning_rate | 0.01-0.1 | 0.05 | Slow learning = better generalization |
| n_estimators | 500-5000 | Use early stopping | Let validation loss decide |
| subsample | 0.6-0.9 | 0.8 | Row subsampling = less overfitting |
| colsample_bytree | 0.6-0.9 | 0.8 | Feature subsampling = decorrelates trees |
| min_child_weight | 10-100 | 50 | Prevents fitting to small clusters |
| reg_alpha | 0-1.0 | 0.1 | L1 regularization on leaf weights |
| reg_lambda | 0.1-10 | 1.0 | L2 regularization on leaf weights |

Note: these are much more conservative than default settings. The defaults are designed for domains with higher signal-to-noise ratios."

**Code moment:** Use Optuna to tune XGBoost with expanding-window CV:

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
    }
    # Evaluate with expanding-window CV
    ics = expanding_window_cv(params)
    return np.mean(ics)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

Show the optimization history plot (IC vs. trial number) and the hyperparameter importance plot. Students should see that max_depth and learning_rate are the two most important hyperparameters — getting these right accounts for 70%+ of the performance variation.

**"So what?":** "Hyperparameter tuning is not optional in financial ML. The difference between default XGBoost parameters (max_depth=6, learning_rate=0.3) and optimally tuned parameters is often the difference between an IC of 0 and an IC of 0.04. Spend your time on tuning, not on architecture."

### Section 5: Feature Importance and SHAP Values
**Narrative arc:** We move from "does the model work?" to "why does the model work?" The setup: you have a black-box model with IC of 0.04. The tension: can you trust it? Should your firm risk $100 million on it? The resolution: SHAP values decompose every prediction into feature contributions, making the model transparent.

**Key concepts:**
- Gain-based feature importance: how much each feature reduces the loss. Misleading for correlated features
- Permutation importance: how much accuracy drops when a feature is shuffled. Better but slow
- SHAP values: Shapley values from game theory applied to feature attribution. The gold standard
- SHAP for individual predictions vs. global importance
- SHAP interaction values: directly measuring the momentum × volatility interaction
- Partial dependence plots: visualizing the marginal effect of each feature

**The hook:** "Gain-based feature importance — the default in XGBoost — is dangerously misleading. If you have two highly correlated features (say, 20-day and 60-day momentum), the tree will arbitrarily split on one or the other, and the importance will be spread randomly between them. One will look important and the other will look useless, even though they carry the same signal. SHAP values solve this by computing the marginal contribution of each feature, averaged over all possible feature orderings. It's computationally expensive (O(2^K) in theory, O(K log K) in practice for trees), but it gives you honest answers."

**Key formulas:**
"The SHAP value for feature j in prediction i:

$$\phi_j(i) = \sum_{S \subseteq K \setminus \{j\}} \frac{|S|!(|K|-|S|-1)!}{|K|!} [f(S \cup \{j\}) - f(S)]$$

This is the average marginal contribution of feature j across all possible coalitions of features. For tree models, TreeSHAP computes this exactly in O(L × D) time per prediction, where L is the number of leaves and D is the max depth.

Key properties:
1. **Additivity:** $\sum_j \phi_j(i) = f(\mathbf{x}_i) - E[f(\mathbf{x})]$ (the SHAP values sum to the prediction)
2. **Consistency:** if a feature's contribution increases in a new model, its SHAP value increases
3. **Missingness:** features not in the model have SHAP value = 0"

**Code moment:** Compute SHAP values for the best XGBoost model:

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Global importance: mean |SHAP| per feature
shap.summary_plot(shap_values, X_test)

# Interaction between momentum and volatility
shap_interaction = explainer.shap_interaction_values(X_test)
shap.dependence_plot(('mom_12m', 'vol_20d'), shap_interaction, X_test)
```

Three plots: (1) The SHAP summary plot showing global feature importance. (2) A SHAP dependence plot for momentum, colored by volatility — this should show the interaction effect. (3) A waterfall plot for a single prediction, showing how each feature pushes the prediction up or down.

**What they'll see:** Momentum (12m skip-1) will be the most important feature. Volatility and reversal will be next. The interaction plot will show that high momentum + low volatility → strong positive SHAP, while high momentum + high volatility → weaker positive SHAP. This is the nonlinear interaction that linear models miss.

**"So what?":** "SHAP values aren't just an interpretability tool — they're a sanity check. If your model's top feature is something you don't expect (say, a technical indicator that shouldn't predict returns), you probably have a data leak or an artifact. In production, portfolio managers won't trade a model they can't explain. SHAP makes the model explainable."

### Section 6: The Overfitting Problem — In-Sample vs. Out-of-Sample
**Narrative arc:** A brutally honest section about the primary danger of using powerful models on noisy data. The setup: your XGBoost model has an in-sample R-squared of 15%. The tension: the out-of-sample R-squared is 0.3%. The resolution: the gap is expected — but you need to understand it, monitor it, and manage it.

**Key concepts:**
- In-sample vs. out-of-sample IC: the gap as a measure of overfitting
- Rolling in-sample and out-of-sample IC plots over time
- Does the gap widen? If so, the model is becoming more overfit as data grows (unusual but possible)
- The test for genuine signal: is the out-of-sample IC statistically significantly different from zero?
- Model complexity vs. OOS performance: the Goldilocks zone

**The hook:** "Here's a test I run on every model I build: plot in-sample IC and out-of-sample IC over time, on the same graph. If the in-sample line is at 0.15 and the out-of-sample line is at 0.03, you're fitting 5x more noise than signal. That ratio tells you how much of your model's apparent skill is real. If the gap is growing over time, your model is memorizing recent patterns that don't generalize. If the gap is stable, you've found a consistent level of signal. For most financial data, a 3:1 to 5:1 ratio of in-sample to out-of-sample IC is normal. Above 10:1, something is wrong."

**Code moment:** For the best XGBoost model, plot in-sample IC and out-of-sample IC for each expanding-window step. Show them on the same axes. The gap should be stable and the out-of-sample line should be above zero (statistically).

**"So what?":** "Overfitting in financial ML is not a theoretical concern — it's the primary practical challenge. A model with in-sample IC of 0.20 and OOS IC of 0.02 is mostly noise. But that 0.02 might still be profitable if it's real. The danger is mistaking the 0.20 for the truth. Always evaluate on out-of-sample data. Always."

### Section 7: Model Combination — Averaging Predictions
**Narrative arc:** A practical section on the simplest ensembling technique: averaging predictions across different models. The setup: you now have OLS, Ridge, Lasso, Random Forest, XGBoost, and LightGBM. The tension: which one should you deploy? The resolution: deploy all of them — the average is usually better than any individual.

**Key concepts:**
- Simple averaging: mean of predictions from multiple models
- Weighted averaging: weight by validation-set IC
- Why averaging works: different models make uncorrelated errors (diversification for predictions)
- Diminishing returns: the third model helps less than the second
- The Gu-Kelly-Xiu finding: ensemble of all 8 models outperforms any single model

**The hook:** "Here's the simplest way to improve any financial ML system: average the predictions of your three best models. That's it. In the Gu-Kelly-Xiu study, the ensemble of all 8 models had higher out-of-sample R-squared than the best individual model. This is diversification applied to predictions — the same principle that works for portfolios works for models. Different models make different mistakes, and averaging cancels some of the mistakes."

**Code moment:**

```python
# Simple ensemble: average predictions from top 3 models
preds_ensemble = (preds_ridge + preds_xgb + preds_lgb) / 3

# Weighted ensemble: weight by validation IC
w_ridge = 0.2  # proportional to validation IC
w_xgb = 0.4
w_lgb = 0.4
preds_weighted = w_ridge * preds_ridge + w_xgb * preds_xgb + w_lgb * preds_lgb
```

Compare ensemble IC against individual model ICs. The ensemble should beat every individual model by 5-15%.

**"So what?":** "Model ensembling is free alpha. It costs nothing (the models are already trained), adds almost no latency (averaging is trivial), and consistently improves performance. Every production system at every major quant fund uses some form of model averaging. Don't deploy a single model when you can average three."

### Closing Bridge

"You now have the model that most production quant funds actually use: gradient-boosted trees, tuned with Optuna, explained with SHAP, combined via simple averaging. Your out-of-sample IC should be higher than Week 4's linear models — probably 0.03-0.06 compared to 0.02-0.04. The improvement comes from nonlinear interactions: momentum × volatility, size × reversal, and other cross-effects that trees discover automatically.

But we've been sloppy about something important. We've been labeling our target as 'next month's return' — but what does that even mean? If a stock goes up 5% then crashes back down within the month, is that a 'success'? And we've been using expanding-window CV, which is better than shuffled CV, but still doesn't account for the fact that our labels overlap in time (a 5-day return label at time t shares 4 days with the label at time t+1). Next week, we confront these issues head-on with Lopez de Prado's financial ML methodology: triple-barrier labeling, purged cross-validation, and meta-labeling. It's the difference between 'doing ML in finance' and 'doing financial ML.'"

## Seminar Exercises

### Exercise 1: Trees vs. Linear Models — The Fair Comparison
**The question we're answering:** How much does nonlinearity actually help for cross-sectional stock prediction?

**Setup narrative:** "We're about to answer the question that launched a thousand Kaggle notebooks: do trees beat linear models for stock prediction? The answer is yes — but the margin matters. If trees beat Ridge by 0.001 in IC, the extra complexity isn't worth it. If they beat it by 0.02, it is."

**What they build:** Using the Week 4 feature matrix, train Random Forest, XGBoost, and LightGBM alongside Ridge and Lasso. Use expanding-window CV for all models. Report: IC, R-squared, quintile spread, and portfolio Sharpe for each model.

**What they'll see:** XGBoost and LightGBM will beat Ridge by 0.01-0.02 in IC — a meaningful but not enormous improvement. Random Forest will be intermediate. The quintile spread will be more monotonic for trees. Portfolio Sharpe will improve by 0.2-0.5 points.

**The insight:** "The improvement from trees over linear models is consistent but modest — maybe 30-50% higher IC. Most of the signal comes from the same features (momentum, reversal, volatility). Trees add value primarily through interactions, not through better handling of the linear effects."

### Exercise 2: The Overfitting Frontier — How Deep Is Too Deep?
**The question we're answering:** At what point does tree depth stop adding signal and start fitting noise, and can you see the transition happen?

**Setup narrative:** "The lecture showed that optimal max_depth for financial data is 3-5. But why? You're going to trace the overfitting frontier directly: train XGBoost at max_depth 1 through 12, and plot in-sample vs. out-of-sample IC at each depth. The crossover point — where more depth hurts OOS performance — tells you the complexity budget that financial data can support."

**What they build:** Fix all hyperparameters except max_depth. Train XGBoost for max_depth = 1, 2, 3, ..., 12 using expanding-window CV. For each depth: record in-sample IC, out-of-sample IC, and the ratio (IS/OOS). Plot all three. Then repeat the experiment for two different market regimes: a calm period (e.g., 2017-2019) and a volatile period (e.g., 2020-2022). Compare the optimal depth across regimes.

**What they'll see:** In-sample IC will monotonically increase with depth (more capacity = more fitting). OOS IC will peak around depth 3-5 and then decline. The IS/OOS ratio will increase steeply beyond depth 5 — from 3:1 to 10:1 or worse. During volatile periods, the optimal depth is even shallower (2-3), because the signal-to-noise ratio drops further. The overfitting frontier is regime-dependent.

**The insight:** "The gap between in-sample and out-of-sample performance is your overfitting diagnostic. When the IS/OOS ratio exceeds 5:1, you're spending most of your model capacity on noise. The optimal depth isn't a fixed number — it depends on the signal-to-noise ratio of the current regime. This is why production systems re-tune periodically."

### Exercise 3: SHAP Stability — Do Feature Importances Shift Across Regimes?
**The question we're answering:** Is the model learning the same features in bull markets as it does in bear markets, or does its internal logic shift with the regime?

**Setup narrative:** "The lecture showed SHAP values for a single test period. But if the model's feature usage is different in 2017 (calm bull) vs. 2020 (COVID crash) vs. 2022 (rate-hike bear), then you're effectively running three different strategies under one hood. You need to know when the model changes its mind about what matters."

**What they build:** Using the tuned XGBoost from Exercise 1, compute SHAP values for four distinct test periods: (a) 2017-2018 (low-vol bull), (b) 2020 (COVID crisis), (c) 2021 (recovery/meme-stock era), (d) 2022 (rate-hike bear). For each period: generate a SHAP summary plot and record the top-5 feature ranking. Then build a "SHAP drift heatmap" — a matrix of (feature × period) showing mean |SHAP| values, color-coded to reveal shifts.

**What they'll see:** Momentum will be dominant in trending regimes (2017-2018, 2021) but less important during regime breaks (2020 COVID crash — momentum crashed spectacularly). Volatility and reversal features will gain importance during crisis periods. The SHAP drift heatmap will show that the model's internal feature weighting is not constant — it adapts to what's working. Size and value features will show the most instability.

**The insight:** "SHAP stability analysis is a regime detection tool in disguise. When SHAP importances shift dramatically, the market's factor structure is changing — and your model is adapting (or struggling to adapt). At production quant funds, a sudden SHAP shift triggers a model review: is the model correctly adapting to a new regime, or is it fitting noise in a transitional period? Monitoring SHAP over time is as important as computing it once."

### Exercise 4: Partial Dependence Plots — Visualizing Nonlinearity
**The question we're answering:** What does the nonlinear relationship between features and returns actually look like?

**Setup narrative:** "Linear models assume straight lines. Trees don't. Let's see what shape the real relationship takes."

**What they build:** Generate partial dependence plots (PDPs) for the top 3 features using sklearn or SHAP. For momentum: the PDP should show a positive but diminishing marginal effect — very high momentum has a weaker effect than moderate momentum (this is the momentum crash effect). For volatility: the PDP should show a negative relationship — higher volatility → lower predicted returns (the low-volatility anomaly).

**What they'll see:** The momentum PDP will not be linear — it will flatten or even curve downward for extreme momentum values (very high past returns tend to reverse). The volatility PDP will show a strongly negative effect that's roughly linear but with curvature at the extremes. The size PDP will show a negative effect (small stocks have higher predicted returns) but it will be noisy.

**The insight:** "These nonlinearities are why trees beat linear models. A linear model assumes the relationship between momentum and expected return is a straight line. The actual relationship is concave — strong momentum helps, but extreme momentum is less helpful (and may even hurt). Trees capture this automatically."

## Homework: "Cross-Sectional Alpha Model v2 (Trees)"

### Mission Framing

You're upgrading your Week 4 alpha model with the most powerful tabular model class in finance. The pipeline is the same — features, expanding-window CV, quantile portfolio analysis — but the model is different. Where Ridge and Lasso saw linear relationships, XGBoost and LightGBM will find interactions and nonlinearities. Where Ridge assumed monotonic effects, trees will discover that the momentum signal flattens at extremes.

But power comes with danger. Trees overfit more aggressively than linear models, especially in low-signal environments. Your job is not just to build a better model — it's to prove it's better honestly, using proper temporal evaluation, and to understand why it's better through SHAP analysis. A 0.01 improvement in IC that you understand and can explain is worth more than a 0.03 improvement that you can't.

This homework also introduces model combination: you'll average predictions from your top 3 models and see if the ensemble beats every individual model. Spoiler: it almost always does.

### Deliverables

1. **Extend your Week 4 pipeline with Random Forest, XGBoost, and LightGBM.** Use the same feature matrix, same expanding-window CV, same evaluation metrics. This is a drop-in replacement — the only thing that changes is the model.

2. **Tune hyperparameters with Optuna (at least 50 trials) using expanding-window CV.** The search space should include: max_depth (3-6), learning_rate (0.01-0.1), n_estimators (500-5000 with early stopping), subsample (0.6-0.9), colsample_bytree (0.6-0.9), min_child_weight (10-100), reg_alpha (0.001-1.0), reg_lambda (0.1-10). Report the best hyperparameters and Optuna's importance analysis.

3. **Compare all models on out-of-sample IC and portfolio Sharpe.** Create a comprehensive comparison table: OLS, Ridge, Lasso, Elastic Net, Random Forest, XGBoost, LightGBM. Report: mean IC, IC standard deviation, IC t-statistic, portfolio Sharpe (gross and net of 10 bps costs), max drawdown, and turnover.

4. **Compute SHAP values for your best tree model.** Create: (a) summary plot (global feature importance), (b) dependence plots for the top 3 features, (c) at least one interaction analysis (momentum × volatility). Discuss whether the model's feature usage makes economic sense.

5. **Test for overfitting: plot in-sample vs. out-of-sample IC over time.** Do the two lines track each other? Does the gap widen over time? Is the OOS IC statistically significantly different from zero (t-test)?

6. **Stretch goal: model combination.** Average predictions from your top 3 models (choose by validation IC). Report the ensemble IC and Sharpe. Does it beat every individual model?

### What They'll Discover

- XGBoost/LightGBM will outperform Ridge by approximately 0.01-0.02 in OOS IC. This sounds small but translates to a meaningful improvement in portfolio Sharpe — roughly 0.2-0.5 points. The improvement is consistent across time, not concentrated in specific periods.

- The optimal max_depth will be 3-5 — much shallower than what you'd use for a typical ML task. This is because the signal-to-noise ratio in financial data is so low that deep trees overfit rapidly. The "right" model complexity for finance is surprisingly low.

- SHAP will confirm that momentum (12m skip-1) is the dominant feature, and that it interacts with volatility in a specific way: momentum works better for low-volatility stocks. This is a known effect — Barroso and Santa-Clara (2015) called it "momentum risk-managed by volatility." Your model rediscovered a published anomaly.

- The model ensemble (average of top 3 models) will almost certainly beat every individual model. The improvement will be 5-15% in IC. This is the diversification effect applied to predictions — different models make different errors, and averaging cancels some of the noise.

- The in-sample vs. out-of-sample IC gap will be 3:1 to 5:1 for trees (e.g., in-sample IC of 0.15, OOS IC of 0.04). For Ridge, the gap will be smaller (2:1 to 3:1). This confirms that trees overfit more, but their higher capacity still results in better OOS performance. The gap is the "price" you pay for the nonlinear capabilities.

### Deliverable
A complete Jupyter notebook containing: the full model comparison table, Optuna tuning results, SHAP analysis, overfitting analysis, and (stretch) ensemble results. Code should extend the Week 4 codebase — same features, same CV, just more models.

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Decision trees → Random Forest → Gradient Boosting | Demo: train RF, XGBoost, LightGBM on same data, compare training time | Exercise 1: systematic fair comparison of trees vs. linear models on expanding-window CV | At scale: full model comparison table (7 models) on 200+ stocks |
| XGBoost vs. LightGBM vs. CatBoost | Demo: compare all three on same data with similar hyperparameters | Not covered (done in lecture) | At scale: include all three in model comparison |
| Hyperparameter tuning (Optuna) | Demo: Optuna search with expanding-window objective | Exercise 2: trace overfitting frontier across max_depth 1-12, compare across regimes | Build: tune with Optuna (50+ trials), report best params and importance |
| SHAP values & feature interactions | Demo: summary plot, dependence plot, waterfall for single test period | Exercise 3: compute SHAP across 4 market regimes, build drift heatmap | At scale: SHAP analysis for best model, momentum × volatility interaction |
| Partial dependence plots (nonlinearity) | Demo: mention in SHAP section as complementary tool | Exercise 4: PDP for top 3 features, discover nonlinear shapes | Integrate: use PDPs to interpret model behavior |
| Overfitting diagnostics (IS vs. OOS IC) | Demo: concept and plotting approach | Exercise 2: directly measure IS/OOS gap at each tree depth | Build: plot IS vs. OOS IC over time, test OOS IC significance |
| Model combination / ensembling | Demo: simple and weighted averaging of predictions | Not covered (done in homework) | Build: average top-3 models, report ensemble IC improvement |
| Trees' structural advantages for finance | Demo: missing value handling, no scaling needed, interaction capture | Not covered (done in lecture) | Reference: leverage in pipeline (no imputation for tree models) |

## Key Stories & Facts to Weave In

1. **Two Sigma's Kaggle competition (2017).** Two Sigma, managing $60 billion, hosted a Kaggle competition for predicting financial returns. Over 2,000 teams participated. The top solutions were dominated by XGBoost and LightGBM ensembles. Neural networks were used by some top teams but primarily as part of stacking ensembles with tree models. The winning insight wasn't a novel architecture — it was careful feature engineering, aggressive regularization, and proper temporal validation.

2. **Jane Street's Kaggle competition (2020-2021).** Jane Street, one of the largest quantitative trading firms, hosted a competition on Kaggle with a real trading dataset. Again, gradient-boosted trees dominated the leaderboard. The top solutions used LightGBM with 5-10 model ensembles, carefully tuned hyperparameters, and feature interactions. First place used a neural network, but positions 2-10 were dominated by tree ensembles. The Jane Street competition is notable because the data was generated from real trading activity, not synthetic data.

3. **LightGBM's origin at Microsoft (2017).** LightGBM was created by Guolin Ke and collaborators at Microsoft Research. It introduced two key innovations: Gradient-based One-Side Sampling (GOSS), which focuses on data points with large gradients, and Exclusive Feature Bundling (EFB), which bundles mutually exclusive features. These made it 10-20x faster than XGBoost on large datasets while maintaining or improving accuracy. Within a year of release, it became the default at most quant funds.

4. **The momentum × volatility interaction.** Barroso and Santa-Clara published "Momentum Has Its Moments" in 2015, showing that momentum strategy crashes can be predicted by volatility. When volatility is high, momentum is unreliable and often crashes. When volatility is low, momentum is strong and consistent. Your tree model will learn this interaction automatically — it's the most important nonlinear effect in the cross-section, and it's the primary reason trees outperform linear models for stock prediction.

5. **SHAP's Nobel-adjacent history.** SHAP values are based on Shapley values, developed by Lloyd Shapley in 1953 for cooperative game theory. Shapley won the Nobel Prize in Economics in 2012 (sharing it with Alvin Roth) for this work. Scott Lundberg adapted Shapley values for ML feature attribution in 2017, creating the SHAP framework. For tree models, Lundberg's TreeSHAP algorithm computes exact Shapley values in polynomial time — a remarkable theoretical result that makes SHAP practical for production use.

6. **Tianqin Chen and XGBoost's origin.** XGBoost was created by Tianqi Chen at the University of Washington and open-sourced in 2014. It won 17 out of 29 challenges posted on Kaggle's ML competition platform in 2015. The key innovation was the second-order (Newton's method) approximation to the gradient boosting objective, plus L1/L2 regularization on leaf weights. Before XGBoost, gradient boosting was implemented via sklearn's GradientBoostingClassifier, which was slow and had fewer regularization options.

7. **Feature importance in production.** At quant funds like AQR and WorldQuant, SHAP-based feature importance is part of the model monitoring pipeline. If a model's SHAP values shift dramatically — say, momentum drops from the top feature to the 10th — it signals a regime change that requires investigation. The shift might mean momentum has stopped working (as it did in 2009's "momentum crash"), or it might mean the data pipeline has a bug. Either way, the SHAP analysis catches it.

8. **The low-volatility anomaly.** One of the most puzzling findings in finance: low-volatility stocks outperform high-volatility stocks. This contradicts basic risk-return theory (higher risk should earn higher return). Baker, Bradley & Wurgler (2011) showed that a portfolio of the lowest-volatility quintile of stocks outperforms the highest-volatility quintile by about 2% per year, with lower drawdowns. Your SHAP analysis will show this — volatility's SHAP will be negative, meaning the model learns that high volatility predicts lower returns.

## Cross-References
- **Builds on:** The feature matrix and expanding-window CV from Week 4 (reused directly). The IC metric and quantile portfolio analysis from Weeks 3-4. The transaction cost model from Week 3.
- **Sets up:** The tree-based models and SHAP analysis from this week become the standard baseline for all remaining weeks. In Week 6, we'll apply proper financial labeling (triple-barrier) and purged CV to these same models — fixing the methodological issues we've been glossing over. In Week 7, neural networks compete against the trees built here. In Week 9, foundation model embeddings are fed into the XGBoost/LightGBM from this week (the "hybrid approach"). The model ensemble technique introduced here is used throughout.
- **Recurring thread:** The "simple vs. sophisticated" thread continues. Trees beat linear models, but the improvement is modest — maybe 30-50% in IC. The biggest gains come from features and methodology, not from model architecture. This pattern repeats in Week 7 (NNs vs. trees: modest improvement) and Week 9 (foundation models vs. XGBoost: depends on the setting).

## Suggested Reading
- **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning"** — Sections on tree-based methods. Focus on the model comparison tables and the feature interaction analysis. This paper is the benchmark for understanding when trees outperform linear models and why.
- **Stefan Jansen, *Machine Learning for Algorithmic Trading*, Chapters 10-12** — Practical implementation of gradient boosting for finance with complete Python code. Covers XGBoost, LightGBM, and hyperparameter tuning.
- **Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions" (SHAP paper)** — The paper that introduced SHAP values. Readable and important for understanding the theory behind TreeSHAP. The key insight: SHAP is the only feature attribution method that satisfies consistency, missingness, and additivity simultaneously.
