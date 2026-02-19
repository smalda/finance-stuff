# Code Plan — Week 4: ML for Alpha — From Features to Signals

## Criteria Map

| # | Criterion (from expectations.md) | Assertion range | Target file | Notes |
|---|---|---|---|---|
| **S1-1** | Feature matrix loads with shape ~(23192, 7), MultiIndex (date, ticker), 130 unique months, 177–179 unique tickers per month | shape[0] in [22000, 24000], shape[1]==7, months==130, tickers_per_month in [177, 179] | lecture/s1_cross_sectional_setup.py | — |
| **S1-2** | Forward 1-month returns computed for 129 months (2014-04 to 2024-12); no future data from 2025+ | 129 months, max date <= 2024-12-31 | lecture/s1_cross_sectional_setup.py | — |
| **S1-3** | Cross-sectional excess returns (demeaned per month) have mean ~0 within each cross-section | \|mean\| < 1e-10 per month | lecture/s1_cross_sectional_setup.py | — |
| **S1-4** | Cross-sectional scatter: momentum_z vs forward return Spearman correlation in [-0.02, 0.10] for most months | median monthly spearman in [-0.02, 0.10] | lecture/s1_cross_sectional_setup.py | — |
| **S1-5** | DATA QUALITY block printed | verified by presence in stdout | lecture/s1_cross_sectional_setup.py | — |
| **S1-6** | Survivorship bias acknowledged | text in structured output | lecture/s1_cross_sectional_setup.py | — |
| **S2-1** | IC time series computed for 60–129 months; each IC is cross-sectional correlation across ~179 stocks | len(ic_series) in [60, 129] | lecture/s2_signal_evaluation.py | — |
| **S2-2** | Mean IC (Fama-MacBeth linear model): range [-0.01, 0.06] | [-0.01, 0.06] | lecture/s2_signal_evaluation.py | — |
| **S2-3** | Rank IC and Pearson IC both computed; rank IC within +/-0.02 of Pearson IC | \|rank_ic - pearson_ic\| <= 0.02 | lecture/s2_signal_evaluation.py | — |
| **S2-4** | ICIR computed; expected range [0.05, 0.50] | [0.05, 0.50] | lecture/s2_signal_evaluation.py | — |
| **S2-5** | IC t-statistic and p-value computed (implementing ic_summary logic from scratch) | finite t-stat and p-value | lecture/s2_signal_evaluation.py | First-intro: implement from scratch |
| **S2-6** | Fundamental law demonstrated: IR = IC x sqrt(BR) with BR = 179 | computation present | lecture/s2_signal_evaluation.py | — |
| **S2-7** | pct_positive reported; range [0.50, 0.70] | [0.50, 0.70] | lecture/s2_signal_evaluation.py | — |
| **S3-1** | Walk-forward produces predictions for 69 OOS months (2019-04 to 2024-12), ~179 stocks/month | n_oos_months == 69 (allow [65, 72]) | lecture/s3_gradient_boosting_alpha.py | — |
| **S3-2** | Mean OOS IC (Pearson): range [0.005, 0.06] | [0.005, 0.06] | lecture/s3_gradient_boosting_alpha.py | — |
| **S3-3** | Mean OOS rank IC: range [0.005, 0.06], within +/-0.01 of Pearson IC | [0.005, 0.06] | lecture/s3_gradient_boosting_alpha.py | — |
| **S3-4** | IC t-statistic reported with p-value | finite, if IC >= 0.02 expect t > 1.96 | lecture/s3_gradient_boosting_alpha.py | — |
| **S3-5** | ICIR: range [0.05, 0.60] | [0.05, 0.60] | lecture/s3_gradient_boosting_alpha.py | — |
| **S3-6** | pct_positive: range [0.50, 0.72] | [0.50, 0.72] | lecture/s3_gradient_boosting_alpha.py | — |
| **S3-7** | prediction_quality() check: spread_ratio > 0.05 | spread_ratio > 0.05 | lecture/s3_gradient_boosting_alpha.py | First-intro: implement from scratch |
| **S3-8** | Overfitting check: if train IC > 2x OOS IC, print OVERFIT | conditional check | lecture/s3_gradient_boosting_alpha.py | — |
| **S3-9** | DATA QUALITY block printed before training | present in stdout | lecture/s3_gradient_boosting_alpha.py | — |
| **S3-10** | Chosen hyperparameters and early stopping rounds reported | present in structured output | lecture/s3_gradient_boosting_alpha.py | — |
| **S3-11** | Baseline comparison: naive IC series + paired test | paired test present | lecture/s3_gradient_boosting_alpha.py | First-intro: implement vs_naive_baseline from scratch |
| **S4-1** | NN walk-forward produces predictions for 69 OOS months, ~179 stocks/month | n_oos in [65, 72] | lecture/s4_neural_vs_trees.py | — |
| **S4-2** | Mean OOS IC (NN): range [-0.01, 0.06] | [-0.01, 0.06] | lecture/s4_neural_vs_trees.py | — |
| **S4-3** | prediction_quality() check: spread_ratio > 0.03 | spread_ratio > 0.03 | lecture/s4_neural_vs_trees.py | — |
| **S4-4** | \|GBM IC - NN IC\| reported; expected [0.00, 0.03] | [0.00, 0.03] | lecture/s4_neural_vs_trees.py | comparative: needs S3 output |
| **S4-5** | Paired t-test for GBM vs NN IC series | t-stat and p-value reported | lecture/s4_neural_vs_trees.py | comparative: needs S3 output |
| **S4-6** | Training stability: mean and std of stopping epoch reported | present in output | lecture/s4_neural_vs_trees.py | — |
| **S4-7** | Overfitting check: if train IC > 3x OOS IC, print OVERFIT | conditional check | lecture/s4_neural_vs_trees.py | — |
| **S5-1** | Expanded feature matrix has 10–12 columns; no NaN from interactions | n_cols in [10, 12] | lecture/s5_feature_engineering.py | — |
| **S5-2** | Mean OOS IC (expanded model): range [0.005, 0.07] | [0.005, 0.07] | lecture/s5_feature_engineering.py | — |
| **S5-3** | IC change (expanded minus baseline): range [-0.015, +0.025] | [-0.015, 0.025] | lecture/s5_feature_engineering.py | comparative: needs S3 output |
| **S5-4** | SHAP summary plot produced; top-5 features by mean \|SHAP\| reported | present in output | lecture/s5_feature_engineering.py | — |
| **S5-5** | SHAP dominated by 2–4 original features | top-2 are original features | lecture/s5_feature_engineering.py | — |
| **S5-6** | Permutation importance (OOS) for top-5; rank corr with SHAP > 0.5 | rank_corr > 0.5 | lecture/s5_feature_engineering.py | — |
| **S6-1** | Decile portfolios for 69 OOS months: 10 groups, ~18 stocks/decile | n_months in [60, 69], groups == 10 | lecture/s6_signal_to_portfolio.py | consumes S3 predictions |
| **S6-2** | Monotonic return pattern: top > bottom (if IC > 0) | top_decile_mean > bottom_decile_mean | lecture/s6_signal_to_portfolio.py | — |
| **S6-3** | Long-short monthly return series: 69 obs; mean in [-0.005, +0.015] | mean in [-0.005, 0.015] | lecture/s6_signal_to_portfolio.py | — |
| **S6-4** | Gross annualized Sharpe: range [-0.5, 1.5] | [-0.5, 1.5] | lecture/s6_signal_to_portfolio.py | — |
| **S6-5** | Monthly one-way turnover: range [0.20, 0.80] | [0.20, 0.80] | lecture/s6_signal_to_portfolio.py | — |
| **S6-6** | Net returns at 10 bps: net Sharpe < gross Sharpe | net_sharpe < gross_sharpe | lecture/s6_signal_to_portfolio.py | — |
| **S6-7** | Cumulative return plot produced for gross and net | plot exists | lecture/s6_signal_to_portfolio.py | — |
| **S6-8** | Max drawdown reported: range [-0.50, 0.00] | [-0.50, 0.00] | lecture/s6_signal_to_portfolio.py | — |
| **S6-9** | Skewness and excess kurtosis reported alongside Sharpe | present in output | lecture/s6_signal_to_portfolio.py | — |
| **S7-1** | Alt data taxonomy: >=5 categories | n_categories >= 5 | lecture/s7_alternative_data.py | conceptual, no ML |
| **S7-2** | Institutional cost context: BattleFin/Exabel $1.6M reference | text present | lecture/s7_alternative_data.py | — |
| **S7-3** | Bridge to Week 7 articulated | text present | lecture/s7_alternative_data.py | — |
| **S7-4** | No claims about IC from alt data without citation | editorial check | lecture/s7_alternative_data.py | — |
| **EX1-1** | VIX regime classification: 69 OOS months split into ~34 high-vol + ~35 low-vol | split sizes in [28, 41] each | seminar/ex1_ic_autopsy.py | consumes S3 IC series |
| **EX1-2** | Mean IC per regime: both in [-0.05, 0.10] | [-0.05, 0.10] | seminar/ex1_ic_autopsy.py | — |
| **EX1-3** | IC std in high-vol > IC std in low-vol | std_high > std_low | seminar/ex1_ic_autopsy.py | — |
| **EX1-4** | Regime contrast reported: \|IC_high - IC_low\| | value reported | seminar/ex1_ic_autopsy.py | — |
| **EX1-5** | Two-sample t-test: t-stat and p-value reported | present | seminar/ex1_ic_autopsy.py | — |
| **EX1-6** | IC time series plot with VIX regime shading | plot exists | seminar/ex1_ic_autopsy.py | — |
| **EX1-7** | pct_positive per regime: range [0.35, 0.75] each | [0.35, 0.75] | seminar/ex1_ic_autopsy.py | — |
| **EX2-1** | 7 LOO models trained; mean OOS IC each in [-0.01, 0.06] | all in range | seminar/ex2_feature_knockout.py | independent from S3 (retrains own models) |
| **EX2-2** | Largest single-feature IC drop: range [0.001, 0.03] | [0.001, 0.03] | seminar/ex2_feature_knockout.py | — |
| **EX2-3** | Top-2 correlated features removed simultaneously | computation present | seminar/ex2_feature_knockout.py | — |
| **EX2-4** | Substitution ratio (joint drop / sum individual drops): range [1.0, 3.0] | [1.0, 3.0]; if < 1.0 report honestly | seminar/ex2_feature_knockout.py | — |
| **EX2-5** | Feature correlation matrix (7x7) printed | present in output | seminar/ex2_feature_knockout.py | — |
| **EX3-1** | All 5 models produce IC series of length 69 | len in [65, 72] each | seminar/ex3_complexity_ladder.py | — |
| **EX3-2** | Mean OOS IC per model in range [-0.01, 0.06] (per-model ranges specified) | see expectations | seminar/ex3_complexity_ladder.py | — |
| **EX3-3** | Complexity-performance: IC does NOT monotonically increase with complexity | pattern check | seminar/ex3_complexity_ladder.py | — |
| **EX3-4** | ICIR per model: range [0.0, 0.60] | [0.0, 0.60] | seminar/ex3_complexity_ladder.py | — |
| **EX3-5** | Paired t-tests for adjacent complexity levels: t-stat and p-value | all reported | seminar/ex3_complexity_ladder.py | — |
| **EX3-6** | Summary table: model, mean IC, std IC, ICIR, pct_positive, t-stat, p-value | present | seminar/ex3_complexity_ladder.py | — |
| **EX3-7** | prediction_quality() check per model: spread_ratio > 0.03 | all > 0.03 | seminar/ex3_complexity_ladder.py | — |
| **EX4-1** | Monthly one-way turnover series: 68 values; mean in [0.20, 0.80] | len in [60, 68], mean in [0.20, 0.80] | seminar/ex4_turnover_tax.py | consumes S3 predictions |
| **EX4-2** | Annualized one-way turnover reported: range [2.4, 9.6] | [2.4, 9.6] | seminar/ex4_turnover_tax.py | — |
| **EX4-3** | Net Sharpe at 5 bps: reduction < 0.3 from gross | sharpe_diff < 0.3 | seminar/ex4_turnover_tax.py | — |
| **EX4-4** | Net Sharpe at 20 bps: reduction between 0.2 and 1.0 from gross | diff in [0.2, 1.0] | seminar/ex4_turnover_tax.py | — |
| **EX4-5** | Net Sharpe at 50 bps: expected negative or near zero | net_sharpe_50 <= 0.2 | seminar/ex4_turnover_tax.py | — |
| **EX4-6** | Breakeven cost level computed | value reported | seminar/ex4_turnover_tax.py | — |
| **EX4-7** | HIGH TURNOVER warning if monthly > 0.50 | conditional print | seminar/ex4_turnover_tax.py | — |
| **EX4-8** | Plot: net Sharpe as function of cost level (0–100 bps) | plot exists | seminar/ex4_turnover_tax.py | — |
| **D1-1** | AlphaModelPipeline class with __init__, fit_predict, summary | class structure verified | hw/d1_alpha_engine.py | — |
| **D1-2** | Accepts sklearn-compatible models (LinearRegression, Ridge, LGBMRegressor) | tested with >=2 model types | hw/d1_alpha_engine.py | — |
| **D1-3** | fit_predict returns IC series (DataFrame) and prediction series (MultiIndex) | return types verified | hw/d1_alpha_engine.py | — |
| **D1-4** | summary() returns dict with: mean_ic, std_ic, icir, pct_positive, t_stat, p_value, mean_rank_ic, sharpe_gross, sharpe_net, mean_turnover, max_drawdown, n_oos_months | all keys present | hw/d1_alpha_engine.py | — |
| **D1-5** | Default run (LightGBM, 60-mo window): mean IC in [0.005, 0.06] | [0.005, 0.06] | hw/d1_alpha_engine.py | — |
| **D1-6** | Temporal integrity: no future data leakage | structural check | hw/d1_alpha_engine.py | — |
| **D1-7** | Purge gap correctly implemented (>= 1 month) | structural check | hw/d1_alpha_engine.py | — |
| **D1-8** | DATA QUALITY block printed | present | hw/d1_alpha_engine.py | — |
| **D1-9** | Handles missing values (LightGBM natively or imputation within train window) | no crash with NaN | hw/d1_alpha_engine.py | — |
| **D1-10** | Long-short portfolio integrated: decile sort, cumulative returns, turnover, net returns | computation present | hw/d1_alpha_engine.py | — |
| **D2-1** | Expanded feature matrix: 15–25 features | n_features in [15, 25] | hw/d2_feature_lab.py | — |
| **D2-2** | At least 3 new price-derived features | count >= 3 | hw/d2_feature_lab.py | — |
| **D2-3** | At least 2 interaction features | count >= 2 | hw/d2_feature_lab.py | — |
| **D2-4** | At least 2 new fundamental features with +90 day PIT lag | count >= 2, lag applied | hw/d2_feature_lab.py | — |
| **D2-5** | Cross-sectional rank normalization on all new features | per-month rank to [0,1] | hw/d2_feature_lab.py | — |
| **D2-6** | Feature correlation matrix: no pair with \|corr\| > 0.95 | max_corr < 0.95 | hw/d2_feature_lab.py | — |
| **D2-7** | Baseline IC vs expanded IC: change in [-0.02, +0.03] | [-0.02, 0.03] | hw/d2_feature_lab.py | comparative: needs D1 baseline |
| **D2-8** | Paired t-test on IC series (baseline vs expanded) | present | hw/d2_feature_lab.py | — |
| **D2-9** | SHAP summary plot for expanded model: top-10 reported | present | hw/d2_feature_lab.py | — |
| **D2-10** | Feature importance ranking with economic interpretation (top-5) | present | hw/d2_feature_lab.py | — |
| **D2-11** | PIT contamination check: IC with PIT-clean only vs all | comparison present | hw/d2_feature_lab.py | — |
| **D3-1** | At least 4 model families compared (OLS, Ridge, LightGBM, NN) | count >= 4 | hw/d3_model_comparison.py | — |
| **D3-2** | Summary table: mean IC, std IC, ICIR, pct_positive, t_stat, p_value, mean rank IC, gross Sharpe, net Sharpe, turnover, max drawdown | all columns present | hw/d3_model_comparison.py | — |
| **D3-3** | All mean ICs in range [-0.01, 0.07] | [-0.01, 0.07] | hw/d3_model_comparison.py | — |
| **D3-4** | Pairwise paired t-tests: Ridge vs OLS, LightGBM vs Ridge, NN vs LightGBM | at least 3 pairs | hw/d3_model_comparison.py | — |
| **D3-5** | Multiple testing awareness: N_trials reported | present | hw/d3_model_comparison.py | — |
| **D3-6** | Sandbox vs production section (feature gap, universe gap, PIT, survivorship) | present | hw/d3_model_comparison.py | — |
| **D3-7** | CIO recommendation paragraph | present | hw/d3_model_comparison.py | — |
| **D3-8** | Net Sharpe comparison at 10 bps | present | hw/d3_model_comparison.py | — |

## Execution Waves

- **Wave 0:** `data_setup.py`
  Loads Week 3 feature matrix, shared prices, FF3 factors, FRED VIX. Computes forward monthly returns, monthly feature panel with target. Caches everything to `.cache/`.

- **Wave 1:** `s1_cross_sectional_setup.py`, `s2_signal_evaluation.py`
  Both depend only on data_setup outputs. S1 demonstrates the cross-sectional structure; S2 computes IC from Fama-MacBeth linear predictions. Independent of each other.

- **Wave 2:** `s3_gradient_boosting_alpha.py`, `ex2_feature_knockout.py`, `ex3_complexity_ladder.py`
  S3 trains the primary GBM model. Ex2 and Ex3 are independent walk-forward exercises that train their own models from the feature matrix (they do NOT depend on S3's predictions — they run their own walk-forward loops). S3 caches predictions for downstream use.

- **Wave 3:** `s4_neural_vs_trees.py`, `s5_feature_engineering.py`, `s6_signal_to_portfolio.py`, `ex1_ic_autopsy.py`, `ex4_turnover_tax.py`, `s7_alternative_data.py`
  - S4 needs S3's IC series for head-to-head comparison.
  - S5 needs S3's best hyperparameters (or re-tunes) and S3's baseline IC for comparison.
  - S6 needs S3's predictions for portfolio construction.
  - Ex1 needs S3's IC series for regime analysis.
  - Ex4 needs S3's predictions (or S6 turnover output) for cost analysis.
  - S7 is conceptual (no data dependency) but grouped here for balance.

- **Wave 4:** `d1_alpha_engine.py`
  Builds the AlphaModelPipeline class. Self-contained but uses same data. Grouped after lecture/seminar to ensure patterns are established.

- **Wave 5:** `d2_feature_lab.py`
  Uses D1's AlphaModelPipeline for evaluation. Expands feature set. Needs D1.

- **Wave 6:** `d3_model_comparison.py`
  Uses D1's AlphaModelPipeline + D2's expanded feature matrix. Needs both D1 and D2.

## Per-File Implementation Strategy

### data_setup.py (Wave 0)
- **Runtime:** fast (<1 min — all data from shared cache + local parquet read)
- **Shared infra:** `shared/data.py` (load_sp500_prices, load_ff_factors, load_fred_series), Week 3 feature matrix (read-only)
- **Caches for downstream:** `feature_matrix.parquet` (copy of W3 matrix), `forward_returns.parquet`, `monthly_panel.parquet` (features + target merged), `vix_monthly.parquet`, `ff3_monthly.parquet`
- **Exports:** CACHE_DIR, PLOT_DIR, TICKERS, START, END, TRAIN_WINDOW, PURGE_GAP, N_OOS_MONTHS, load functions

### lecture/s1_cross_sectional_setup.py (Wave 1)
- **Runtime:** fast (<1 min)
- **Shared infra:** none beyond data_setup
- **Visualization:** scatter plot of momentum_z vs forward return for a single representative month; histogram of cross-sectional returns for one month
- **Criteria:** S1-1 through S1-6

### lecture/s2_signal_evaluation.py (Wave 1)
- **Runtime:** fast (<1 min)
- **Shared infra:** none (first-intro: implements IC computation from scratch)
- **Visualization:** IC time series as bar chart (one bar per month, color by sign); ICIR decomposition
- **Criteria:** S2-1 through S2-7
- **Notes:** Implements ic_summary() from scratch as first-introduction content. Re-derives Fama-MacBeth cross-sectional predictions by running per-month OLS (excess returns on features) and using fitted values — does NOT import from Week 3.

### lecture/s3_gradient_boosting_alpha.py (Wave 2)
- **Runtime:** medium (2–5 min — 69 walk-forward windows x LightGBM with HP search)
- **Device:** cpu
- **Shared infra:** `shared/temporal.walk_forward_splits`, `shared/temporal.PurgedWalkForwardCV` (for HP search inside training window)
- **HP search:** GridSearchCV with PurgedWalkForwardCV(n_splits=3) — learning_rate [0.01, 0.05, 0.1], num_leaves [15, 31, 63], n_estimators with early stopping (max 500, patience 50). Search done on first training window; re-used for subsequent windows for speed.
- **Early stopping:** 50 rounds on temporal validation holdout (last 12 months of training window)
- **Visualization:** bar chart of monthly IC (one bar per month, color by sign); cumulative IC line chart
- **Caches for downstream:** `gbm_predictions.parquet` (MultiIndex predictions), `gbm_ic_series.parquet` (monthly IC), `gbm_best_params.json` (chosen HPs)
- **Criteria:** S3-1 through S3-11
- **Notes:** First-intro for prediction_quality() spread-ratio check (from scratch) and vs_naive_baseline() paired test (from scratch). Naive baseline = previous month's return as prediction.

### lecture/s4_neural_vs_trees.py (Wave 3)
- **Runtime:** slow (5–12 min — 69 walk-forward windows x NN training, ~5-15s per window)
- **Device:** mps (downshift: cpu)
- **Shared infra:** `shared/dl_training.fit_nn`, `shared/dl_training.predict_nn`, `shared/temporal.walk_forward_splits`
- **HP search:** Small grid — learning_rate [1e-4, 1e-3, 1e-2], hidden_size [16, 32, 64], dropout [0.1, 0.3, 0.5]. Search on first window using temporal CV within training data. Re-use best HPs for all windows.
- **Early stopping:** patience=10 epochs on temporal validation (last 12 months of training window), max 50 epochs
- **Visualization:** side-by-side bar chart of GBM vs NN IC statistics; overlaid monthly IC line chart (both models)
- **Caches for downstream:** `nn_ic_series.parquet`, `nn_predictions.parquet`
- **Criteria:** S4-1 through S4-7
- **Pedagogical alternatives:**
  - **Alternative A:** sklearn MLPRegressor with default parameters
    Trigger: PyTorch NN produces degenerate predictions (spread_ratio < 0.03) after tuning
    Expected: MLPRegressor with simpler optimization may produce non-degenerate but weak predictions
  - **Alternative B:** Reduce to single hidden layer with 16 neurons
    Trigger: training unstable (>30% of windows diverge or NaN losses)
    Expected: simpler architecture stabilizes training

### lecture/s5_feature_engineering.py (Wave 3)
- **Runtime:** medium (2–5 min — retrain GBM walk-forward with expanded features + SHAP computation)
- **Device:** cpu
- **Shared infra:** `shared/temporal.walk_forward_splits`, `shap.TreeExplainer` (external), `sklearn.inspection.permutation_importance`
- **HP search:** Reuse S3 best hyperparameters (loaded from gbm_best_params.json)
- **Visualization:** SHAP summary plot (beeswarm) for last OOS cross-section; bar chart of SHAP mean |value| by feature
- **Criteria:** S5-1 through S5-6
- **Notes:** Extends the 7-feature matrix with 3–5 interaction/non-linear terms to get 10–12 total. Interactions: momentum_z x volatility_z, earnings_yield_z x pb_ratio_z. Non-linear: momentum_z^2, |reversal_z|. Computed from existing z-scored features (no new data downloads needed).

### lecture/s6_signal_to_portfolio.py (Wave 3)
- **Runtime:** fast (<1 min)
- **Shared infra:** `shared/backtesting.quantile_portfolios`, `shared/backtesting.long_short_returns`, `shared/backtesting.portfolio_turnover`, `shared/backtesting.sharpe_ratio`, `shared/backtesting.net_returns`, `shared/backtesting.max_drawdown`, `shared/backtesting.cumulative_returns`
- **Visualization:** line chart of cumulative gross + net long-short returns; bar chart of decile mean returns (monotonicity check)
- **Criteria:** S6-1 through S6-9
- **Notes:** Consumes S3 predictions (gbm_predictions.parquet). Touches Tier 3 evaluation (transaction costs, return distributions) — applies relevant rigor rules.

### lecture/s7_alternative_data.py (Wave 3)
- **Runtime:** fast (<30 sec)
- **No data, no ML** — conceptual taxonomy section
- **Criteria:** S7-1 through S7-4
- **Notes:** Prints structured taxonomy. No plots required by expectations (taxonomy can be printed as formatted text). No assertions on numerical values.

### seminar/ex1_ic_autopsy.py (Wave 3)
- **Runtime:** fast (<1 min)
- **Shared infra:** `shared/data.load_fred_series` (VIX), `scipy.stats.ttest_ind`
- **Visualization:** IC time series line chart with VIX regime shading (high-vol months highlighted in a different color)
- **Criteria:** EX1-1 through EX1-7
- **Notes:** Consumes S3 IC series (gbm_ic_series.parquet). Loads FRED VIX and resamples to monthly.

### seminar/ex2_feature_knockout.py (Wave 2)
- **Runtime:** slow (8–15 min — 7 LOO models x 69 windows + 1 leave-two-out)
- **Device:** cpu
- **Shared infra:** `shared/temporal.walk_forward_splits`
- **HP search:** Reuse S3-compatible HPs (pre-set in code; does NOT re-tune per knockout as per expectations)
- **Visualization:** bar chart of IC drop per feature (7 bars); grouped bar chart comparing individual drops vs joint drop for top-2
- **Criteria:** EX2-1 through EX2-5
- **Pedagogical alternatives:**
  - **Alternative A:** Permutation importance instead of retraining
    Trigger: retraining 7 LOO models x 69 windows takes >45 minutes
    Expected: faster but captures different effect (marginal vs structural importance)

### seminar/ex3_complexity_ladder.py (Wave 2)
- **Runtime:** slow (10–25 min — 5 models x 69 walk-forward windows)
- **Device:** mps for NN (downshift: cpu)
- **Shared infra:** `shared/temporal.walk_forward_splits`, `shared/dl_training.fit_nn`, `shared/dl_training.predict_nn`
- **HP search:**
  - OLS: none
  - Ridge: alpha log-spaced [1e-4, 1e-2, 1, 100, 1e4] with temporal CV
  - LightGBM depth=3: learning_rate [0.01, 0.05, 0.1], early stopping
  - LightGBM depth=8: same as depth=3
  - NN: reuse S4 architecture (1-2 hidden layers, 32 neurons, ReLU, dropout 0.3)
- **Visualization:** grouped bar chart of IC, ICIR, pct_positive across 5 models; summary table printed
- **Criteria:** EX3-1 through EX3-7

### seminar/ex4_turnover_tax.py (Wave 3)
- **Runtime:** fast (<1 min)
- **Shared infra:** `shared/backtesting.portfolio_turnover`, `shared/backtesting.sharpe_ratio`, `shared/backtesting.net_returns`, `shared/backtesting.long_short_returns`
- **Visualization:** line chart of net Sharpe as function of cost level (0–100 bps); bar chart of Sharpe at 0/5/20/50 bps
- **Criteria:** EX4-1 through EX4-8
- **Notes:** Consumes S3 predictions (gbm_predictions.parquet) and forward returns from data_setup.

### hw/d1_alpha_engine.py (Wave 4)
- **Runtime:** medium (3–5 min — full walk-forward with LightGBM as default test)
- **Device:** cpu
- **Shared infra:** `shared/temporal.walk_forward_splits`, `shared/backtesting.*`, `shared/metrics.ic_summary` (after first-intro in S2/S3, D1 can import)
- **HP search:** Built into the class (configurable grid, temporal CV inside train window)
- **Caches for downstream:** `alpha_pipeline_test.parquet` (verification run output)
- **Criteria:** D1-1 through D1-10
- **Notes:** This is a Construction task. The class must be reusable by D2 and D3.

### hw/d2_feature_lab.py (Wave 5)
- **Runtime:** slow (5–10 min — feature construction + walk-forward via D1 pipeline)
- **Device:** cpu
- **Shared infra:** `shared/data.load_sp500_ohlcv`, `shared/data.load_sp500_fundamentals`, `shared/data.load_ff_factors`, `shared/data.load_ff_portfolios`, `shap.TreeExplainer`
- **Visualization:** SHAP summary beeswarm plot for expanded model; bar chart comparing baseline vs expanded IC; feature correlation heatmap
- **Criteria:** D2-1 through D2-11
- **Notes:** Constructs 15–25 features from price-derived + fundamental + interaction sources. Uses D1 AlphaModelPipeline for evaluation.

### hw/d3_model_comparison.py (Wave 6)
- **Runtime:** slow (10–20 min — 4 models x walk-forward via D1 pipeline)
- **Device:** mps for NN (downshift: cpu)
- **Shared infra:** D1 AlphaModelPipeline, `shared/dl_training.*`, `shared/metrics.deflated_sharpe_ratio`
- **Visualization:** grouped bar chart comparing IC/ICIR/Sharpe across models; summary table; cumulative return overlay for all models
- **Criteria:** D3-1 through D3-8
- **Notes:** Consumes D2 expanded feature matrix. Runs D1 pipeline with 4+ model families.

## Cross-File Data Flow

```
data_setup.py
  → feature_matrix.parquet (copy of W3)
  → forward_returns.parquet
  → monthly_panel.parquet (features + target)
  → vix_monthly.parquet
  → ff3_monthly.parquet

s3_gradient_boosting_alpha.py
  → gbm_predictions.parquet    → consumed by: s4, s5, s6, ex1, ex4
  → gbm_ic_series.parquet      → consumed by: s4, ex1
  → gbm_best_params.json       → consumed by: s5, ex2

s4_neural_vs_trees.py
  → nn_ic_series.parquet       → informational (not consumed downstream)
  → nn_predictions.parquet     → informational

d1_alpha_engine.py
  → (class definition, no cache — imported by d2 and d3)

d2_feature_lab.py
  → expanded_features.parquet  → consumed by: d3
  → expanded_ic_series.parquet → consumed by: d3

d3_model_comparison.py
  → (terminal — no downstream consumers)
```

## Open Questions (from expectations.md) — Mapped to Files

1. **Will GBM IC on S&P 500 with 7 features be distinguishable from zero?**
   → Investigated by: S3 (primary), Ex3, D1, D3
   → If t < 1.96: reframe around methodology demonstration

2. **Will the neural network produce non-degenerate predictions?**
   → Investigated by: S4 (primary), Ex3
   → Alternatives A and B documented above

3. **What will the SHAP feature ranking look like — fundamentals vs technicals?**
   → Investigated by: S5 (primary), D2
   → PIT contamination check in D2 isolates the effect
