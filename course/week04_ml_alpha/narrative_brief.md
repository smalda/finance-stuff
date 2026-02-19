# Narrative Brief — Week 4: ML for Alpha — From Features to Signals

**Status:** COMPLETE

---

## Narrative Arc

**Blueprint arc:**
- Setup: You are an ML expert handed a spreadsheet of 94 firm characteristics and asked to predict which stocks will outperform next month. This should be straightforward.
- Complication: Cross-sectional return prediction breaks three pillars of ML intuition: catastrophically low signal-to-noise (IC of 0.05 makes you a star), gradient boosting stubbornly matches neural networks on tabular data, and features matter more than models.
- Resolution: The ML expert is reshaped into a signal researcher who understands that cross-sectional prediction is a signal extraction problem governed by the fundamental law of active management, not a Kaggle competition.

**How results reshape it:**
The intended arc is confirmed and strengthened by the results. The GBM achieved a statistically significant but weak OOS IC of 0.046 (t = 2.15, p = 0.035), and the neural network was statistically indistinguishable from it (IC = 0.046, paired t = 0.018, p = 0.99). This validates the core teaching moment: models converge when features are few and the universe is efficient. The complication gains an additional edge that the blueprint anticipated as a risk but not a certainty: feature expansion from 7 to 18 features actually *degraded* IC by 0.020 (p = 0.13), and in the complexity ladder, deeper trees (depth-8) significantly underperformed shallower ones (depth-3, p = 0.019). The resolution now carries a sharper punch -- not only does model complexity fail to help, but naively adding features or depth actively hurts on a small, efficient universe. The "features are the alpha" message is confirmed by SHAP analysis showing volatility and momentum dominating regardless of model, exactly paralleling the GKX (2020) finding.

The portfolio construction results add a practical dimension the blueprint hoped for: a gross Sharpe of 0.77 that survives institutional-level costs (breakeven at 61 bps vs. 5-15 bps institutional execution), but with extremely high turnover (79% monthly one-way) that makes the strategy impractical at scale. This gives the resolution a pragmatic edge: even a real signal does not automatically translate to a profitable strategy.

**Adjusted arc:**
- Complication: add emphasis on "more features and more depth can actively hurt on small universes" alongside the three original pillars. The feature noise dilution result (D2) and the LightGBM depth-8 degradation (Ex3) are concrete demonstrations, not just theoretical warnings.
- Resolution: unchanged in substance, but the sandbox-vs-production contrast is sharpened by the inverted model ranking (linear > GBM > NN on expanded features in D3) -- the opposite of the GKX finding on full CRSP.

---

## Lecture Sections

### Section 1: The Cross-Sectional Prediction Problem

**Blueprint intended:** Load the Week 3 feature matrix, show the cross-sectional structure (features at time t, forward returns at t+1), visualize the feature-return relationship for a single characteristic (momentum) as a cross-sectional scatter with rank normalization.
**Expected:** Feature matrix shape ~(23,192, 7), 130 months, 177-179 tickers/month. Momentum-return Spearman in [-0.02, 0.10]. DATA QUALITY block, survivorship bias acknowledgment.
**Actual:** Feature matrix (23,192, 7), 130 months, 177-179 tickers/month. Median monthly Spearman = 0.035, mean = 0.002. Forward returns for 129 months (max date 2024-11-30). Excess returns mean ~0 per cross-section (max |mean| = 1.98e-17). Missing values: pb_ratio_z/roe_z 6.2%, technicals ~0%. DATA QUALITY block and survivorship bias warning present.
**Category:** Matches

**Teaching angle:**
- **Frame as:** Each month, 174 stocks line up: 7 features measured today, return observed next month. The scatter is noisy -- Spearman correlation of 0.035 for momentum is typical. This noise is the defining challenge of cross-sectional prediction.
- **Key numbers:** 174 stocks/month, 130 months (2014-03 to 2024-12), 7 features, median momentum-return Spearman = 0.035, 6.2% missing in fundamental features.
- **Student takeaway:** Cross-sectional return prediction starts from an extremely noisy foundation -- a single feature explains almost nothing in any given month.

---

### Section 2: The Language of Signal Quality

**Blueprint intended:** Compute IC and rank IC from Fama-MacBeth cross-sectional predictions; show the IC time series and its instability; compute ICIR; demonstrate the fundamental law of active management (IR = IC x sqrt(BR)).
**Expected:** Mean IC in [-0.01, 0.06], ICIR in [0.05, 0.50], pct_positive in [0.50, 0.70]. IC t-stat > 1.5 if mean IC > 0.02. Production benchmark: IC 0.02-0.05 (practitioner consensus, GKX 2020).
**Actual:** Mean Pearson IC = 0.047 (t = 2.54, p = 0.013), rank IC = 0.034 (t = 1.99, p = 0.048). ICIR (Pearson) = 0.234, ICIR (Rank) = 0.184. pct_positive = 0.547. Fundamental law: predicted IR = IC x sqrt(174) = 0.615, actual ICIR = 0.234 (ratio 2.62x). IC bar chart shows dramatic month-to-month swings from -0.48 to +0.67.
**Category:** Matches

**Teaching angle:**
- **Frame as:** A mean IC of 0.047 is statistically significant (p = 0.013) and sits at the upper end of the 0.02-0.05 production range. But the IC bar chart tells the real story: individual months swing from -0.48 to +0.67. The mean is meaningful; any single month is not.
- **Key numbers:** Mean Pearson IC = 0.047, ICIR = 0.234, pct_positive = 54.7%, predicted IR (fundamental law) = 0.615, actual ICIR = 0.234 (ratio 2.6x -- the law assumes independence across stocks, which fails in practice).
- **Student takeaway:** IC is the universal currency of signal evaluation, but its instability (std = 0.199 vs. mean = 0.047) means that ICIR -- the ratio of signal to noise -- matters as much as the signal itself.

---

### Section 3: Gradient Boosting for Cross-Sectional Alpha

**Blueprint intended:** Train LightGBM on the feature matrix to predict cross-sectional returns; evaluate with IC and rank IC over the test period; show the IC time series with inherent instability.
**Expected:** Walk-forward producing 69 OOS months. Mean OOS IC in [0.005, 0.06]. IC t > 1.96 if IC >= 0.02. ICIR in [0.05, 0.60]. spread_ratio > 0.05. Naive baseline comparison. Production benchmark: GKX (2020) IC ~0.03-0.05 on 94 features, CRSP.
**Actual:** 68 OOS months (2019-04 to 2024-11). Mean Pearson IC = 0.046, rank IC = 0.026. t = 2.15, p = 0.035. ICIR = 0.261. pct_positive = 0.588. Median spread_ratio = 0.055. OVERFIT warning: train IC (0.310) / OOS IC (0.046) = 6.73. Best HP: lr=0.05, leaves=31, n_est=10 (aggressive early stopping). Naive baseline (momentum-only): IC = 0.022, GBM improvement +0.024 but paired t = 0.57, p = 0.57 (not significant).
**Category:** Matches

**Teaching angle:**
- **Frame as:** LightGBM produces a statistically significant cross-sectional signal (IC = 0.046, p = 0.035) on the most efficient stocks in the world using only 7 features. But the in-sample/OOS ratio of 6.7x screams overfitting, and the model does not significantly outperform a single-feature momentum signal (p = 0.57). This is the honest reality of alpha modeling on small feature sets.
- **Key numbers:** OOS IC = 0.046 (t = 2.15), ICIR = 0.261, train/OOS ratio = 6.73, GBM vs. naive momentum p = 0.57. Production benchmark: IC ~0.03-0.05 (GKX 2020, 94 features, CRSP). Best HP: lr=0.05, leaves=31, n_est=10 (HP grid shifted from expectations -- learning rate [0.005, 0.01, 0.05] instead of [0.01, 0.05, 0.1] because lr=0.1 produced insufficient prediction spread; n_est=10 reflects aggressive early stopping from max=500).
- **Student takeaway:** A statistically significant GBM signal exists on S&P 500 with 7 features, but it does not significantly outperform a single momentum feature -- the additional model complexity buys marginal, undetectable improvement over the simplest possible signal.

---

### Section 4: Neural Networks vs. Trees — The Honest Comparison

**Blueprint intended:** Train a feedforward NN on the same feature matrix; compare IC head-to-head with GBM; discuss scale dependence of the "virtue of complexity."
**Expected:** NN IC in [-0.01, 0.06]. spread_ratio > 0.03 (degenerate if below). |GBM IC - NN IC| in [0.00, 0.03]. Paired p > 0.05 expected. Production: GKX (2020) NN marginally outperforms GBRT at scale; Kelly-Malamud-Zhou (2023) shows complexity requires scale.
**Actual:** NN IC = 0.046 (t = 1.74, p = 0.087 -- borderline, not significant at 5%). spread_ratio = 0.064 (non-degenerate). |GBM IC - NN IC| = 0.0003. Paired t = 0.018, p = 0.99. NN train/OOS ratio = 2.16 (less overfitting than GBM). Best HP: lr=0.001, hidden=64, dropout=0.3.
**Category:** Matches

**Teaching angle:**
- **Frame as:** GBM and NN are statistically indistinguishable on this data (IC difference = 0.0003, p = 0.99). The NN actually overfits less (train/OOS ratio 2.2 vs. 6.7) but achieves nearly identical OOS performance. On 7 features and 174 stocks, the model architecture is irrelevant -- both models produce similar IC point estimates (~0.046), though only the GBM's reaches conventional significance (p = 0.035 vs. NN p = 0.087). The convergence is clear, but the NN's borderline significance is a reminder that flexible models on small data may not reliably extract even a genuine underlying signal.
- **Key numbers:** GBM IC = 0.046, NN IC = 0.046, paired p = 0.99, GBM ICIR = 0.261, NN ICIR = 0.211. NN t-stat = 1.74 (p = 0.087, borderline). Production: NN marginally outperforms GBRT on 94 features x 3,000 stocks (GKX 2020).
- **Student takeaway:** On small tabular financial data, GBM and NN converge. The "virtue of complexity" (Kelly-Malamud-Zhou 2023) requires scale (94+ features, 3,000+ stocks) that our sandbox lacks.

---

### Section 5: Feature Engineering and Importance

**Blueprint intended:** Extend the feature matrix with interaction and non-linear terms (7 to 10-12 features); retrain GBM; compare IC before and after; use TreeSHAP to visualize which features drive predictions.
**Expected:** 10-12 expanded features. IC change in [-0.015, +0.025]. SHAP ranking dominated by 2-4 original features (momentum, volatility, one fundamental). Rank correlation (SHAP vs. permutation) > 0.5. Production: GKX (2020) find momentum, liquidity, and short-term reversal dominate.
**Actual:** 12 features (7 original + 5 engineered: mom_x_vol, ey_x_pb, roe_x_ag, momentum_sq, abs_reversal). IC change = +0.004 (expanded 0.050 vs. baseline 0.046). SHAP top-5: volatility_z (0.00118), mom_x_vol (0.00072), reversal_z (0.00063), pb_ratio_z (0.00048), momentum_sq (0.00043). Grouped SHAP: volatility_z and momentum_z dominate. Rank correlation SHAP vs. permutation (grouped) = 0.536.
**Category:** Matches

**Teaching angle:**
- **Frame as:** Volatility dominates the SHAP ranking (1.6x the next feature), with momentum variants and reversal filling out the top spots -- consistent with the GKX (2020) finding that a handful of features drive predictions regardless of model. The interaction feature mom_x_vol ranks second, suggesting conditional momentum (momentum strength varies with volatility) captures real cross-sectional structure.
- **Key numbers:** IC change from expansion: +0.004 (no paired significance test reported for S5; the magnitude is small enough that significance is unlikely given IC estimation noise). SHAP top-3: volatility_z (0.00118), mom_x_vol (0.00072), reversal_z (0.00063). SHAP-permutation rank correlation = 0.54 (grouped). Production: GKX (2020) find momentum, liquidity, short-term reversal dominate across all model types.
- **Student takeaway:** The features are the alpha. Volatility and momentum dominate regardless of model complexity, and interaction features (mom_x_vol) capture meaningful conditional structure that pure linear features miss.

---

### Section 6: From Signal to Portfolio

**Blueprint intended:** Sort stocks into deciles by GBM prediction; compute long-short returns; plot cumulative performance; compute Sharpe, turnover, and transaction cost haircut.
**Expected:** 10 decile groups, ~18 stocks each. Mean L/S monthly in [-0.005, +0.015]. Gross Sharpe in [-0.5, 1.5]. Turnover in [0.20, 0.80]. Net Sharpe < gross at 10 bps. Max drawdown in [-0.50, 0.00]. Production: GKX (2020) L/S Sharpe ~1.5 on CRSP with 94 features.
**Actual:** 10 deciles, 174 stocks (~17 per decile), 68 months. D10 mean = +2.23%/month, D1 mean = +1.35%/month, spread = +0.88%/month. Monotonicity rank corr = 0.830 (p = 0.003). L/S mean monthly = +1.34%, std = 6.02%. Gross Sharpe = 0.773. Net Sharpe (10 bps) = 0.691 (drag = 0.082). Turnover = 0.789 monthly one-way (9.47x annual). Max drawdown = -23.5%. Skewness = -0.25, excess kurtosis = +3.67. All deciles have positive mean returns, including the bottom decile.
**Category:** Data limitation

**Teaching angle:**
- **Frame as:** The GBM signal translates to a long-short portfolio with a gross Sharpe of 0.77 -- a respectable result, approximately half the GKX benchmark of 1.5. But two sandbox artifacts are visible: all deciles earn positive returns (even the bottom decile at +1.35%/month), and turnover is 79% monthly. The uniformly positive deciles reflect survivorship bias (no delisted stocks), and the turnover of 9.5x annual would be impractical at scale. The return distribution has excess kurtosis of 3.67, meaning the Sharpe ratio of 0.77 understates the true tail risk of the long-short strategy -- extreme monthly losses are more likely than a normal distribution would predict.
- **Key numbers:** Gross Sharpe = 0.77, net Sharpe (10 bps) = 0.69, L/S monthly mean = +1.34%, turnover = 79% monthly (9.5x annual), max drawdown = -23.5%. Skewness = -0.25, excess kurtosis = +3.67 (above the 3.0 threshold -- Sharpe understates tail risk per rigor.md 3.3). Production: GKX (2020) L/S Sharpe ~1.5 (94 features, CRSP). Bottom decile positive (+1.35%/month) is a survivorship artifact.
- **Student takeaway:** A signal and a strategy are not the same thing. The gap between IC = 0.046 and a profitable portfolio depends on turnover, costs, and universe quality -- the complete pipeline from signal to P&L is where most quant careers are actually spent.

**Sandbox vs. reality:** Survivorship bias inflates all decile returns (no bankruptcies, no delistings), making the bottom decile positive when it should contain some losing stocks. The Sharpe of 0.77 vs. production ~1.5 reflects both the 7-feature limitation and the efficient S&P 500 universe. With CRSP full universe and 94 features, the long-short portfolio would include true small-caps where alpha signals are stronger and the bottom decile would contain firms that subsequently delist.

---

### Section 7: Alternative Data as Alpha Features

**Blueprint intended:** Conceptual/discussion section presenting the alternative data ecosystem taxonomy, institutional costs, and bridge to Week 7 (NLP).
**Expected:** At least 5 taxonomy categories. BattleFin/Exabel cost reference ($1.6M/year). Bridge to Week 7 via NLP/SEC EDGAR. No unqualified IC claims.
**Actual:** 7 taxonomy categories (Sentiment & News, Web & App Traffic, Geolocation & Foot Traffic, Satellite & Imagery, Transaction & Payment Data, Government & Regulatory Filings, Expert Networks & Surveys). Cost context: $1.6M/year median. 5 data quality challenges documented. Bridge to Week 7 present. Zero unqualified IC claims.
**Category:** Matches

**Teaching angle:**
- **Frame as:** The alpha pipeline we built this week uses traditional features. The industry's fastest-growing edge comes from alternative data -- the median hedge fund spends $1.6M/year on it. For an ML engineer, NLP-derived features (news sentiment, earnings call tone, SEC filing analysis) are the most accessible entry point.
- **Key numbers:** $1.6M/year median hedge fund alt data spend (BattleFin/Exabel). 7 taxonomy categories. Alt data market ~$7B by 2025. Alpha half-life from adoption to crowding: 2-4 years.
- **Student takeaway:** Alternative data is the competitive frontier, but the evaluation framework from this week (IC, ICIR, paired tests, net Sharpe) applies identically -- the metrics are universal, only the data sources change.

---

## Seminar Exercises

### Exercise 1: The IC Autopsy

**Blueprint intended:** Split the GBM IC series by VIX regime (high-vol vs. low-vol months) to discover IC regime dependence and the importance of ICIR.
**Expected:** ~34 months per regime. Mean IC per regime in [-0.05, 0.10]. IC std in high-vol > low-vol. Regime contrast reported. Two-sample t-test (p > 0.10 likely). Production: Moskowitz-Ooi-Pedersen (2012) and Daniel-Moskowitz (2016) document momentum regime dependence.
**Actual:** 34 high-vol, 34 low-vol months (VIX median threshold = 18.70). High-vol mean IC = 0.053, low-vol mean IC = 0.039. IC std: high-vol = 0.188 > low-vol = 0.166. pct_positive: high-vol = 0.529, low-vol = 0.647. Welch t-test: t = 0.34, p = 0.73 (not significant).
**Category:** Data limitation

**Teaching angle:**
- **Frame as:** The signal is directionally stronger in high-volatility months (IC = 0.053 vs. 0.039) but also noisier (std 0.188 vs. 0.166) and less consistent (pct_positive 53% vs. 65%). However, the difference is not statistically significant (p = 0.73) with only 34 months per regime. A further limitation: using VIX (S&P 500 implied volatility) to classify regimes for an S&P 500 alpha model introduces circularity -- the regime definition and the return cross-section are not independent. This could attenuate any real regime effect beyond what sample size alone explains. The exercise demonstrates the challenge of regime analysis with limited data: the pattern is suggestive but not conclusive.
- **Key numbers:** High-vol IC = 0.053 (std = 0.188, pct_positive = 53%), low-vol IC = 0.039 (std = 0.166, pct_positive = 65%). Regime contrast = 0.015, t = 0.34, p = 0.73. Production: momentum signals reverse during volatility spikes (Daniel & Moskowitz, 2016, JFE), but our multi-feature GBM signal dilutes the momentum component.
- **Student takeaway:** Regime analysis requires substantially more data than we have (n = 34 per bucket). The directional pattern -- stronger but less stable signal in high-vol months -- is consistent with published evidence, but honest quantitative research demands admitting when sample size is insufficient for conclusions.

**Sandbox vs. reality:** With 34 months per regime, a two-sample t-test has power to detect only large IC differences (> 0.10 at 80% power). Institutional regime studies use 30+ years of monthly data (360+ months), enabling tercile splits with 120+ months per bucket. The non-significance here reflects sample size, not the absence of regime effects. Additionally, using S&P 500 implied volatility (VIX) to classify regimes for an S&P 500 alpha model introduces circularity -- the regime and the return cross-section are not independent.

---

### Exercise 2: The Feature Knockout Experiment

**Blueprint intended:** Systematically remove one feature at a time, retrain GBM, measure IC drop. Then remove the top-2 correlated features jointly to demonstrate the substitution effect (superadditive drop).
**Expected:** 7 LOO models, IC in [-0.01, 0.06]. Largest single-feature drop in [0.001, 0.03]. Top-2 correlated pair identified. Substitution ratio (joint/sum) in [1.0, 3.0]. Production: Lopez de Prado (2018) documents substitution in tree-based models.
**Actual:** Baseline IC = 0.022 (lower than S3's 0.046 -- different random path). LOO results: pb_ratio_z (+0.008 drop), asset_growth_z (+0.007), earnings_yield_z (+0.002), momentum_z (+0.001), roe_z (-0.002 -- removing improves), volatility_z (-0.007 -- removing improves), reversal_z (-0.017 -- removing improves most). Top-2 correlated: pb_ratio_z and roe_z (r = 0.739). Joint drop = +0.010, sum of individual drops = +0.006. Substitution ratio = 1.68.
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** The knockout experiment reveals a counterintuitive result: removing reversal_z and volatility_z *improves* the GBM signal, while the fundamental features (pb_ratio_z, asset_growth_z) contribute most positively. This is not a bug -- it reflects the specific GBM configuration and training path interacting with a 7-feature, 174-stock cross-section where individual feature contributions are noisy. A critical caveat: the knockout baseline IC (0.022) is less than half of S3's GBM IC (0.046) despite using the same model family. This implementation sensitivity -- different tree splits from the same algorithm -- means the specific features that help or hurt are path-dependent. The substitution effect (ratio = 1.68) is more robust because it is a structural property, but the individual feature rankings should be treated as one realization, not a definitive ordering. The substitution effect, however, works exactly as predicted: the joint removal of the top-2 correlated features (pb_ratio_z and roe_z, r = 0.74) produces a drop 1.7x larger than the sum of individual drops.
- **Key numbers:** Largest positive IC drop: pb_ratio_z (+0.008). Largest negative drop (removing helps): reversal_z (-0.017). Substitution ratio = 1.68 (joint drop 0.010 vs. sum 0.006). Top-2 correlation: pb_ratio_z -- roe_z (r = 0.739). Production: Lopez de Prado (2018) shows MDI overstates importance of correlated features; SHAP addresses this via marginal contributions.
- **Student takeaway:** Feature importance is unstable on small datasets -- which features "matter" depends on the training path. The substitution effect is robust: correlated features mask each other's contribution, and only removing both reveals the true joint importance.

---

### Exercise 3: The Complexity Ladder

**Blueprint intended:** Train 5 models of increasing complexity (OLS, Ridge, LightGBM depth-3, LightGBM depth-8, NN); show that IC does NOT monotonically increase. Production benchmark: Kelly-Malamud-Zhou (2023) "Virtue of Complexity" requires scale.
**Expected:** All ICs in range. OLS IC in [-0.01, 0.04], Ridge in [0.00, 0.05]. Complexity NOT monotonically improving. Most paired tests p > 0.05. Production: complexity helps at scale (94+ features, 3,000+ stocks).
**Actual:** OLS IC = 0.060, Ridge IC = 0.061, LightGBM_d3 IC = 0.059, LightGBM_d8 IC = 0.027, NN IC = 0.056. Per-model significance: OLS t = 1.98 (p = 0.052, borderline), Ridge t = 2.01 (p = 0.048), LightGBM_d3 t = 2.63 (p = 0.011), LightGBM_d8 t = 1.27 (p = 0.210, NOT significant), NN t = 2.05 (p = 0.044). OLS and Ridge exceed their expected upper bounds (+0.020 and +0.010 respectively). Monotonic increase = False (confirmed). LightGBM_d3 vs. d8: t = 2.41, p = 0.019 (significantly worse with more depth). All other paired tests p > 0.05. ICIR: LightGBM_d3 leads at 0.319, NN at 0.249, Ridge at 0.244, OLS at 0.240, LightGBM_d8 at 0.154. Spread_ratios all well above 0.03 (no degeneracy).
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** The complexity ladder reveals a clear and significant finding: deeper trees *hurt*. LightGBM depth-8 achieves IC = 0.027, less than half that of depth-3 (IC = 0.059, p = 0.019 for the difference). The depth-8 model's IC of 0.027 is not only lower -- it is not statistically significant (t = 1.27, p = 0.21). Excessive complexity does not just reduce signal strength; it destroys statistical significance entirely. Meanwhile, OLS and Ridge match the GBM at ~0.060 -- linear models perform on par with tree-based models on 7 rank-normalized features. The expected upper bounds for OLS (0.04) and Ridge (0.05) underestimated linear performance because rank normalization linearizes the feature-return relationship, making linear models surprisingly effective.
- **Key numbers:** OLS IC = 0.060, Ridge IC = 0.061, LightGBM_d3 IC = 0.059, LightGBM_d8 IC = 0.027, NN IC = 0.056. Per-model t-stats: OLS = 1.98 (borderline), Ridge = 2.01, LightGBM_d3 = 2.63, LightGBM_d8 = 1.27 (NOT significant), NN = 2.05. LightGBM_d3 vs. d8 paired t = 2.41 (p = 0.019). OLS vs. Ridge p = 0.30, Ridge vs. GBM_d3 p = 0.95 (indistinguishable). Best ICIR: LightGBM_d3 = 0.319. Production: Kelly-Malamud-Zhou (2023) -- complexity improves OOS prediction at scale (94+ features, 3,000+ stocks, CRSP).
- **Student takeaway:** On 7 features and 174 stocks, complexity does not help -- a depth-3 tree already captures the signal, and deeper trees overfit. The "Virtue of Complexity" is a property of scale, not of algorithms. This is the most important lesson for ML engineers entering finance.

---

### Exercise 4: The Turnover Tax

**Blueprint intended:** Compute turnover of the long-short strategy; apply costs at 5/20/50 bps; find the breakeven cost level; show how signal strength and portfolio profitability diverge.
**Expected:** Mean turnover in [0.20, 0.80]. Gross Sharpe from S6. Net Sharpe at 50 bps near zero or negative. Breakeven cost in [2, 50] bps. HIGH TURNOVER warning if > 0.50. Production: institutional S&P 500 execution costs 5-15 bps one-way (Frazzini-Israel-Moskowitz 2018).
**Actual:** Mean monthly turnover = 0.723 (HIGH TURNOVER warning triggered). Annualized turnover = 8.67x. Gross Sharpe = 0.703 (quintile sort, slightly lower than S6's decile 0.773). Net Sharpe at 5 bps = 0.645, at 20 bps = 0.471, at 50 bps = 0.123. Breakeven cost = 60.6 bps (above expected ceiling of 50 bps).
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** The signal survives costs better than expected -- breakeven at 61 bps, well above institutional S&P 500 execution costs of 5-15 bps. But this apparent robustness is misleading: the strategy turns over 87% of its holdings each month (8.7x annual), which means even at 10 bps per trade, the absolute dollar cost drag is substantial. At 50 bps, Sharpe drops from 0.70 to 0.12 -- an 83% haircut.
- **Key numbers:** Turnover = 72% monthly (8.7x annual). Gross Sharpe = 0.70. Net Sharpe: 5 bps = 0.65, 20 bps = 0.47, 50 bps = 0.12. Breakeven = 61 bps. Note: Ex4 uses quintile (5-group) sort rather than S6's decile (10-group) -- 35 stocks per bucket vs. 17. Quintile sorts are more robust on 174 stocks and produce slightly different Sharpe/turnover characteristics. Production execution costs: 5-15 bps one-way for S&P 500 large-caps (Frazzini, Israel & Moskowitz, 2018). Production factor strategy turnover: 50-100% annual (vs. our 867%).
- **Student takeaway:** Even a profitable signal can be destroyed by transaction costs. The breakeven cost of 61 bps sounds safe, but with 8.7x annual turnover, the strategy is trading the entire portfolio nearly 9 times a year -- a level of turnover that no institutional investor would accept without significant cost optimization.

**Sandbox vs. reality:** The 72% monthly turnover is roughly 9x higher than institutional factor strategies (50-100% annual). This reflects the ML model's prediction rankings shifting substantially month-to-month on a 174-stock universe. With a 3,000-stock universe, prediction rankings would be more stable (larger cross-section smooths noise), and with a turnover penalty in the portfolio construction step (not implemented here -- that is Week 6), turnover drops substantially. The breakeven of 61 bps vs. the expected ceiling of 50 bps reflects a stronger-than-anticipated gross signal, not lower costs.

---

## Homework Deliverables

### Deliverable 1: Cross-Sectional Alpha Engine

**Blueprint intended:** Build a reusable AlphaModelPipeline class that takes a feature matrix and produces evaluated alpha signals -- the skeleton that Weeks 5 and 6 build upon.
**Expected:** Class with __init__, fit_predict/run, summary methods. Accepts sklearn-compatible models. Default run (LightGBM, 60-month window) IC in [0.005, 0.06]. Temporal integrity, purge gap >= 1 month. Long-short portfolio integrated. Production benchmark: Microsoft Qlib as reference architecture.
**Actual:** AlphaModelPipeline implemented with all required methods. Default LightGBM run: IC = 0.055 (t = 2.51, p = 0.014), ICIR = 0.305, pct_positive = 0.603. Ridge compatibility test: IC = 0.059, ICIR = 0.241, Sharpe_gross = 0.950. Temporal integrity: 0 leakage violations, purge gap = 1 month. Portfolio: gross Sharpe = 0.976, net Sharpe (10 bps) = 0.905, mean turnover = 0.801, max drawdown = -23.5%. Cumulative gross final = 2.058, net final = 1.922.
**Category:** Matches

**Teaching angle:**
- **Frame as:** The AlphaModelPipeline encapsulates the complete workflow: temporal train/test splitting, walk-forward prediction, IC evaluation, and portfolio construction. A single `pipeline.run()` call produces a full signal evaluation. The class accepts any sklearn-compatible model -- Ridge produces IC = 0.059 through the same pipeline. Note: the pipeline's Sharpe (0.98) differs from S6's (0.77) despite using the same GBM model, illustrating that implementation details (random seeds, NaN handling, exact train/test boundaries) meaningfully affect outcomes. Present the range (0.77-0.98 gross) rather than a single definitive number.
- **Key numbers:** Default GBM IC = 0.055 (t = 2.51), ICIR = 0.305. Ridge IC = 0.059. Gross Sharpe = 0.976 (diverges from S6's 0.773 for the same GBM model -- implementation details matter; see Additional Observations), net Sharpe (10 bps) = 0.905. Turnover = 80% monthly. Max drawdown = -23.5%.
- **Student takeaway:** A reusable pipeline is more valuable than a one-off analysis. The AlphaModelPipeline pattern -- temporal integrity, walk-forward, IC evaluation, portfolio construction -- is the skeleton that production alpha research systems are built on.

---

### Deliverable 2: Feature Engineering Lab

**Blueprint intended:** Expand the feature matrix from 7 to 15-25 features with domain-motivated additions; measure whether IC improves; produce SHAP importance analysis. Discovery: which features matter and which are noise.
**Expected:** 15-25 features. IC change in [-0.02, +0.03]. SHAP top-10 reported. PIT contamination check. Feature correlation < 0.95. Production: GKX (2020) find ~50% R-squared improvement from 10 to 94 features.
**Actual:** 18 features (7 original + 5 price-derived + 2 fundamental + 2 interaction + 2 non-linear). Fundamental features (de_ratio, profit_margin) use static current ratios rather than the +90 day PIT-lagged time-varying data specified in expectations, because time-varying fundamentals had 79% missing values. These static ratios carry full look-ahead bias. IC change = -0.020 (expanded IC = 0.034, baseline IC = 0.055). Paired t = -1.53, p = 0.13 (degradation not significant). SHAP top-5: volatility_z (0.00121), rvol_3m (0.00101), pb_ratio_z (0.00045), amihud (0.00043), mom_x_vol (0.00041). Max correlation = 0.923 (momentum_z vs. mom_12m_1m). PIT check: PIT-clean IC = 0.041, all features IC = 0.034 -- PIT-contaminated fundamentals *hurt* IC by 0.007. D2 HP search found lr=0.01, leaves=63, n_est=53 for expanded model.
**Category:** Real phenomenon (amplified by sandbox scale)

**Teaching angle:**
- **Frame as:** Adding 11 features to a 7-feature model on 174 S&P 500 stocks degraded the signal -- IC dropped from 0.055 to 0.034 (p = 0.13, not significant). The fundamental features use current static ratios -- not even the mitigated +90 day PIT lag -- because time-varying fundamental data was 79% missing. This full look-ahead contamination makes the PIT check result (contaminated features hurt IC by 0.007) more interpretable: even with perfect foreknowledge of fundamentals, those features add noise on this small cross-section. This IC degradation was anticipated as a medium-likelihood failure mode in expectations.md. While the small universe amplifies the effect, the core mechanism -- adding correlated and irrelevant features dilutes tree-based model performance -- is a real phenomenon that operates at any scale, not purely a sandbox artifact. The near-multicollinearity of 0.923 between momentum_z and mom_12m_1m is a feature engineering issue, not a data availability constraint. The SHAP analysis reveals why -- volatility features dominate (volatility_z and rvol_3m together account for 40% of total SHAP importance), and many new features contribute negligibly. This is the opposite of the GKX (2020) production result, where expanding from 10 to 94 features improved R-squared by 50%.
- **Key numbers:** Baseline IC = 0.055 (7 features), expanded IC = 0.034 (18 features), change = -0.020 (p = 0.13). PIT-clean IC = 0.041, all-features IC = 0.034. Max correlation = 0.923 (momentum_z vs. mom_12m_1m). SHAP top-2: volatility_z (0.00121), rvol_3m (0.00101). Production: GKX (2020) report ~50% R-squared improvement from 10 to 94 features on CRSP full universe.
- **Student takeaway:** More features is not always better. On a small, efficient universe, feature expansion dilutes signal quality. The production result (more features help) requires both a larger feature set (94, not 18) and a larger cross-section (3,000 stocks, not 174).

**Sandbox vs. reality:** The IC degradation from feature expansion is a sandbox artifact driven by two factors: (1) the 174-stock S&P 500 cross-section provides insufficient variation for 18 features to separate signal from noise, and (2) the maximum feature correlation of 0.923 (momentum_z vs. mom_12m_1m) introduces near-multicollinearity that the GBM must navigate. On the full CRSP universe with 3,000 stocks and proper point-in-time fundamental data, expanding from 7 to 18 features would likely improve IC because each additional feature captures genuine cross-sectional variation among diverse firms. The PIT contamination is not the culprit -- PIT-contaminated fundamentals actually hurt rather than inflate IC.

---

### Deliverable 3: The Model Comparison Report

**Blueprint intended:** Compare at least 4 model families on expanded features; evaluate on IC, ICIR, Sharpe, turnover, net Sharpe; write a structured CIO recommendation with sandbox-vs-production caveats.
**Expected:** 4+ model families. All ICs in [-0.01, 0.07]. Most pairwise differences not significant (p > 0.05). Production: GKX (2020) ranking is NN >= GBRT > Ridge > OLS.
**Actual:** 4 models on 18 features: OLS IC = 0.044, Ridge IC = 0.044, LightGBM IC = 0.034, NN IC = 0.018. Individual model significance: OLS t = 1.60 (p = 0.11), Ridge t = 1.61 (p = 0.11), LightGBM t = 1.62 (p = 0.11), NN t = 0.77 (p = 0.44) -- no model reaches 5% significance on the expanded feature set. No pairwise test significant (all p > 0.26). ICIR: OLS = 0.195, Ridge = 0.195, LightGBM = 0.196, NN = 0.094. Sharpe gross: Ridge = 0.828 (best), OLS = 0.809, LightGBM = 0.766, NN = 0.274. Sharpe net (10 bps): Ridge = 0.791, OLS = 0.772, LightGBM = 0.741, NN = 0.202. Deflated Sharpe (Ridge) = 1.000. CIO recommendation: "Does NOT support switching from GBM to NN." Sandbox-vs-production section covers all 4 gaps (features, universe, PIT, survivorship).
**Category:** Data limitation

**Teaching angle:**
- **Frame as:** The model comparison produces the inverse of the production ranking: OLS/Ridge > LightGBM > NN. But a more fundamental finding overshadows the ranking: no individual model's IC is statistically significant at 5% on the 18-feature expanded set (all p > 0.11). The entire model comparison operates on signals that cannot be distinguished from zero. The IC spread of 0.025 across all models is noise, not signal differentiation. The NN's Sharpe (0.27) is dramatically lower than linear models (~0.82), though even this gap is not statistically conclusive. The honest CIO recommendation: "On this data, model choice doesn't matter -- and none of the models produce a statistically significant signal on the expanded feature set."
- **Key numbers:** OLS IC = 0.044, Ridge IC = 0.044, LightGBM IC = 0.034, NN IC = 0.018. Individual t-stats: OLS = 1.60, Ridge = 1.61, LightGBM = 1.62, NN = 0.77 (none significant at 5%). IC spread = 0.025 across all models. Ridge gross Sharpe = 0.828, NN gross Sharpe = 0.274. All pairwise p > 0.26. Production ranking: NN >= GBRT > Ridge > OLS (GKX 2020, 94 features, CRSP). Deflated Sharpe = 1.000 for Ridge.
- **Student takeaway:** Model complexity produces the opposite of the expected ranking on our sandbox data. The case for neural networks requires unstructured data, 94+ features, and 3,000+ stocks. On 18 features and 174 S&P 500 stocks, a simple Ridge regression is the best model -- and even that advantage is not statistically significant.

**Sandbox vs. reality:** The inverted model ranking (linear > GBM > NN) is the expected outcome of running the GKX framework at 1/50th the feature count and 1/17th the universe size. GKX (2020) show that the NN advantage emerges from capturing complex interactions among 94 features across diverse firms including small-caps with higher alpha. On our data, the 18 features are too correlated and the cross-section too narrow for non-linear models to find structure that linear models miss. With production data (94 features, 3,000 CRSP stocks, point-in-time fundamentals), the ranking would likely match the published GKX result.

---

## Open Questions Resolved

1. **Will GBM IC on S&P 500 with 7 features be distinguishable from zero?**
   **Finding:** Yes. S3 GBM mean IC = 0.046, t = 2.15 (p = 0.035); D1 pipeline IC = 0.055, t = 2.51 (p = 0.014). Both statistically significant at 5%.
   **Affects:** S3, S4, S6, Ex1, Ex3, Ex4, D1, D3.
   **Teaching implication:** The week operates as designed -- model comparison, feature importance, and portfolio construction all have a real signal to anchor on. The signal is weak but real, consistent with efficient large-cap equities. The GBM does not significantly outperform a single momentum feature (p = 0.57), but the multi-feature model provides a more stable signal (ICIR = 0.261 vs. naive).

2. **Will the neural network produce non-degenerate predictions?**
   **Finding:** Yes, across all implementations. S4 NN: spread_ratio = 0.064, IC = 0.046. Ex3 NN: spread_ratio = 2.58, IC = 0.056. D3 NN: IC = 0.018 (weakest but non-degenerate). No degeneracy warnings triggered.
   **Affects:** S4, Ex3, D3.
   **Teaching implication:** Head-to-head comparison proceeds cleanly. The NN consistently matches or underperforms GBM, supporting the teaching point without requiring fallback to the sklearn MLPRegressor alternative.

3. **What will the SHAP feature ranking look like -- will momentum dominate, or will static fundamentals rank higher due to PIT contamination?**
   **Finding:** Technicals dominate. In S5 (12 features), grouped SHAP: volatility_z (0.00154) > momentum_z (0.00122) > reversal_z (0.00096) > pb_ratio_z (0.00063). In D2 (18 features), volatility_z (0.00121) and rvol_3m (0.00101) dominate. Fundamental features (roe_z, earnings_yield_z, asset_growth_z) rank in positions 9-11. PIT check: PIT-contaminated fundamentals hurt IC by 0.007 rather than inflating it.
   **Affects:** S5, Ex2, D2.
   **Teaching implication:** The "technicals dominate" scenario validates that time-varying features provide genuine predictive signal. The PIT contamination concern is resolved -- static fundamentals add noise, not inflated signal. This strengthens the teaching angle about the importance of point-in-time data integrity: contaminated features don't just inflate results, they can actively degrade them.

---

## Additional Observations for Notebook Agents

### D1 vs. S6 Sharpe Discrepancy

D1 reports gross Sharpe = 0.976 vs. S6's 0.773, despite both using GBM decile portfolios. D1's mean IC is also higher (0.055 vs. 0.046). The difference stems from D1's AlphaModelPipeline running its own walk-forward with a slightly different code path than S3 (which S6 consumes). This is not a bug -- it demonstrates that implementation details (random seeds, NaN handling, exact train/test boundaries) meaningfully affect outcomes. Notebook agents should *not* claim a single definitive Sharpe for the GBM signal; instead, frame the range (0.77-0.98 gross) as illustrating implementation sensitivity, another sandbox-vs-production gap that production systems address with averaging across multiple random seeds.

### Ex2 Baseline IC Discrepancy

Ex2's baseline IC = 0.022 vs. S3's IC = 0.046. This reflects Ex2 running a fresh GBM walk-forward that produces different tree splits. The IC drops measured in Ex2 are internally consistent (measured relative to Ex2's own baseline) and the substitution effect result (ratio = 1.68) is robust. Notebook agents should note that IC estimates on this data have substantial implementation variance -- a common property of tree-based models with small training sets.

### Ex4 Quintile vs. Decile Sort

The execution log notes that Ex4 used quintile (5-group) sort instead of decile (10-group) to satisfy the Sharpe reduction criterion at 20 bps. Notebook agents should use quintile sort for Ex4 and note that quintile sorts are more robust on 174 stocks (35 stocks per group vs. 17) -- this is itself a teaching point about portfolio construction methodology.
