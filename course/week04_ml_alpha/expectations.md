# Week 4: ML for Alpha — From Features to Signals — Expectations

**Band:** 1 (Essential Core)
**Implementability:** HIGH

**Rigor:** Tier 2 — ML Training Discipline.
- Teaches: signal significance (implement `ic_summary()` logic from scratch, then compare against `shared/metrics.py`).
- Teaches: prediction quality (implement spread-ratio check from scratch).
- Teaches: baseline significance (implement paired IC comparison from scratch).
- Teaches: feature preprocessing (cross-sectional rank normalization applied before model training — reinforce Week 3's FeatureEngineer pattern in an ML context).

---

## Data Plan

### Universe

- **Option A: Week 3 feature matrix as-is (179 tickers, S&P 500 subset).**
  The feature matrix from `course/week03_factor_models/code/.cache/feature_matrix_ml.parquet` contains 179 tickers with 7 pre-computed, z-scored cross-sectional features (pb_ratio_z, roe_z, asset_growth_z, earnings_yield_z, momentum_z, reversal_z, volatility_z). Provides immediate continuity with Week 3. The 179-ticker count arises from the completeness filter applied during Week 3's FeatureEngineer pipeline.
  - *Statistical power:* 179 stocks per cross-section is adequate for decile sorts (18 stocks per decile) and for cross-sectional IC computation. Cross-sectional power is moderate — enough for pedagogical demonstrations but below the ~3,000-stock cross-sections used in GKX (2020).
  - *Limitation:* S&P 500 is the most efficient segment of the U.S. equity market. Alpha signals are weaker in large-caps because they are more heavily researched. ICs will be lower than full-universe benchmarks.

- **Option B: Expanded universe (~300 tickers from shared cache, with reconstructed features).**
  The shared price cache contains 264+ tickers with >90% monthly coverage over 2014–2024 (verified via code probe). Expanding the universe to ~300 tickers would increase cross-sectional power (30 stocks per decile) and strengthen statistical conclusions. However, the Week 3 FeatureEngineer would need to be re-run on the expanded universe to produce new features, and fundamental features (pb_ratio, roe, asset_growth, earnings_yield) require yfinance fundamental data that has ~4 years of history — the same look-ahead-contaminated static fundamentals as Week 3.
  - *Statistical power:* ~300 stocks per cross-section improves decile sort reliability and reduces noise in cross-sectional correlations.
  - *Cost:* Requires re-running Week 3's FeatureEngineer on a larger universe during data_setup.py. The feature construction logic must be faithfully replicated, including two-stage outlier control (percentile clip + z-cap at +/-3) and rank normalization. Risk of subtle divergence from Week 3's exact methodology.

- **Option C: Full shared cache universe (~450 tickers, with reconstructed features).**
  Using all 456 tickers surviving the completeness filter maximizes cross-sectional power (~45 stocks per decile). This approaches a respectable universe size for cross-sectional ML. However, ~50% of these tickers are post-2005 IPOs with limited fundamental history, and yfinance fundamental coverage drops significantly outside the S&P 500 core.
  - *Statistical power:* Best cross-sectional power available with free data.
  - *Cost:* Fundamental data gaps become severe — many mid-cap tickers will have incomplete or missing fundamentals, requiring imputation or dropping features. The fundamental features that give Week 4 its continuity with Week 3 would be weakened.

- **Recommendation: Option A (179 tickers) as the primary universe, with Option B explored in the homework (D2: Feature Engineering Lab) when students expand the feature set.**
  Rationale: (1) Direct continuity with Week 3's feature_matrix_ml.parquet — the code agent loads it directly rather than reconstructing it, eliminating replication risk. (2) 179 stocks is sufficient for the core pedagogical goals: demonstrating IC evaluation, comparing models, and constructing long-short portfolios. (3) The homework naturally motivates universe expansion when students extend the feature set with price-only features (momentum variants, volatility measures, Amihud illiquidity) that don't require fundamentals and can cover 300+ tickers from the shared price cache. (4) The sandbox-vs-production gap between 179 S&P 500 stocks and GKX's ~3,000 CRSP stocks is itself a teaching point.
- **Risk:** The efficient S&P 500 universe may produce ICs that are statistically significant but economically small, making model comparisons harder to distinguish. If all models produce IC ~ 0.02 with overlapping confidence intervals, the "complexity ladder" (Seminar Ex3) and model comparison (Homework D3) may show no meaningful differentiation. Contingency: widen acceptance criteria to include sign-consistency (pct_positive IC) and ICIR as differentiators even when absolute IC levels converge.

### Date Range

- **Option A: 2014-03 to 2024-12 (130 months) — matches Week 3 feature matrix.**
  The existing feature_matrix_ml.parquet covers this exact range. With a 60-month training window + 1-month purge gap, the OOS period is 2019-04 to 2024-12 (69 months). This period includes COVID volatility (2020), the 2021–2022 growth-to-value rotation, and the 2022 rate shock — a diverse regime sample.
  - *Statistical power:* 69 OOS months with IC mean ~0.03 and std ~0.08 yields t-stat ~3.1 — sufficient for 5% significance (verified via calculation). For a 36-month train window, OOS expands to 93 months (t ~3.6).
  - *Limitation:* The training period starts in 2014, a post-GFC bull market. The model never trains on a recession or prolonged downturn, which limits generalizability.

- **Option B: 2010-01 to 2024-12 (180 months) — extended range using shared price cache.**
  Extending backward to 2010 provides 180 months total. With a 60-month train window, OOS expands to 119 months (2015-02 to 2024-12). The training data now includes the 2010–2013 recovery period. However, fundamental features (pb_ratio, roe, earnings_yield, asset_growth) cannot be extended — yfinance fundamentals cover ~2021–2025 only. The feature matrix would need to be reconstructed with only technical features (momentum, reversal, volatility) for the 2010–2014 period, creating an inconsistent feature set across time.
  - *Statistical power:* 119 OOS months gives t ~4.1 — excellent power.
  - *Cost:* Feature matrix reconstruction required. Fundamental features either missing or static-extrapolated for 2010–2020, introducing severe look-ahead contamination. Alternatively, using only 3 technical features for the full range would undermine the Week 3 continuity story.

- **Recommendation: Option A (2014-03 to 2024-12, 130 months).**
  Rationale: (1) Direct inheritance from Week 3 — no reconstruction required. (2) 69 OOS months with 60-month training provides adequate statistical power for IC significance testing (t > 1.96 at IC = 0.03). (3) The feature matrix consistency across all 7 features is preserved. (4) The 2019–2024 OOS window includes multiple distinct regimes (COVID crash, recovery, inflation, rate hikes) — pedagogically rich. (5) If the 60-month train window proves too restrictive (e.g., for the complexity ladder exercise where multiple models are compared), the code agent can use a 36-month window to gain 93 OOS months while still having 3 years of training data — adequate for tree models and simple neural networks.
- **Risk:** The 2014–2019 training period is exclusively bull market. Models may overfit to momentum-driven regime and underperform during the 2020 crash or 2022 value rotation in OOS. This is pedagogically valuable (regime dependence is the insight of Ex1) but could produce confusing IC reversals. Contingency: document the regime composition of train vs. test in expectations, and use IC sub-period analysis as a feature, not a bug.

### Frequency

- **Monthly cross-sections (primary).** All cross-sectional prediction and IC evaluation operates on monthly frequency — features measured at month-end t, forward returns measured from t to t+1. This matches the GKX framework and is the industry standard for cross-sectional equity models.
- **Daily price data (inputs to features).** Raw daily prices from the shared cache are used to compute monthly features (momentum = cumulative return over trailing 12-2 months, reversal = prior 1-month return, volatility = std of daily returns over trailing 21 days). Data_setup.py computes these from daily data and outputs monthly features.
- **Risk:** Monthly frequency means 130 data points per time-series dimension and ~179 per cross-sectional dimension. The cross-sectional dimension is adequate; the time-series dimension limits the number of OOS evaluation periods. No mitigation needed — this is the standard tradeoff for monthly cross-sectional models.

### Data Sources

**Primary data sources used across sections:**

| Source | API / Loader | What | Sections |
|--------|-------------|------|----------|
| Week 3 feature matrix | `pd.read_parquet(feature_matrix_ml.parquet)` | 7 z-scored features, 179 tickers, 130 months | S1–S6, Ex1–Ex4, D1–D3 |
| Shared equity prices | `load_sp500_prices('2014-01-01', '2024-12-31')` | Daily adjusted close for forward return computation + additional features | S1–S6, Ex1–Ex4, D1–D3 |
| Shared OHLCV | `load_sp500_ohlcv('2012-01-01', '2024-12-31', fields=['Close','Volume'])` | Volume for Amihud illiquidity feature (D2 feature expansion) | S5, D2 |
| Shared FF3 factors | `load_ff_factors(model='3', frequency='M')` | RF column for excess return computation; Mkt-RF for beta feature | S1, D2 |
| Shared fundamentals | `load_sp500_fundamentals(tickers, pit_lag_days=90)` | Additional fundamental features for D2 expansion | D2 |
| FRED VIX | `load_fred_series(['VIXCLS'])` | Monthly VIX for regime classification (high-vol vs. low-vol months) | Ex1 |
| Ken French momentum deciles | `load_ff_portfolios('10_momentum', frequency='M')` | PIT-clean benchmark for momentum-sorted returns | S5, D2 |

### Interaction Analysis

**Universe x features x date range.** The 179-ticker universe with 7 features over 130 months produces a panel of ~23,200 stock-months (verified: 23,192 actual observations in the feature matrix). With ~6.2% missing values concentrated in the fundamental features (pb_ratio_z and roe_z, both 6.2% missing), complete-case analysis reduces to ~21,740 observations (~167 stocks x 130 months). This is adequate for gradient boosting (which handles missing values natively) and for neural networks (after imputation or dropping).

**Train window x OOS length x model count.** With 60-month train windows on 130 months, each walk-forward iteration trains on ~167 stocks x 60 months = ~10,000 observations. The complexity ladder exercise (Ex3) trains 5 models (OLS, Ridge, LightGBM depth=3, LightGBM depth=8, feedforward NN) — each requiring a full walk-forward loop of 69 iterations. Total: 5 x 69 = 345 model fits. LightGBM fits are fast (~seconds each); the neural network is the bottleneck (~5-15 seconds per window with 1-2 hidden layers, ~10K training observations). Estimated total runtime for Ex3: ~15-30 minutes. Manageable, but the code agent should implement early stopping and avoid excessive hyperparameter grids.

**Feature expansion x fundamental data.** Deliverable D2 (Feature Engineering Lab) expands from 7 to 15–25 features. Price-derived features (multi-horizon momentum, realized vol at multiple windows, Amihud illiquidity, max daily return, zero-trading-day fraction) require only daily price/volume data — available for all 179 tickers with no coverage issues. Fundamental-derived additions (D/E ratio, free cash flow yield, R&D intensity, revenue growth) use yfinance fundamentals — limited to ~4 annual periods (2021–2025), PIT-contaminated, and ~15–20% missing for some fields. The fundamental features will have more missing data than the price-derived features, creating an asymmetric feature quality landscape that the code agent must document.

**Model training x feature count.** Gradient boosting handles 15–25 features on ~10K training observations easily. The feedforward neural network (1–2 hidden layers, 32–64 neurons, matching GKX architecture) on 25 features is also computationally trivial. No scaling concerns for the feature expansion.

**VIX regime classification x IC series.** Ex1 splits the monthly IC series by VIX regime (above/below median, or terciles). With 69 OOS months, each regime bucket gets ~23–35 months. A t-test on IC within a regime bucket of 23 months has low power (at IC=0.03, std=0.08: t=1.8 — marginally significant). The exercise should focus on regime-contrast (difference in IC between high-vol and low-vol) rather than absolute significance within each regime.

**Fundamental PIT contamination x alpha signal strength.** The 4 fundamental features in the Week 3 matrix are computed from yfinance data that is not point-in-time: fiscal period end dates are used, and restated values are silently applied. The +90 day PIT lag mitigation (available via `load_sp500_fundamentals(pit_lag_days=90)`) was not applied during Week 3's feature construction. This means the existing feature matrix contains PIT-contaminated fundamental features. The code agent should (a) document this contamination, (b) compare model IC with and without the fundamental features to bound the PIT effect, and (c) note that the 3 technical features (momentum, reversal, volatility) are PIT-clean.

### Failure Modes & Contingencies

| Risk | Likelihood | Affected Sections | Contingency |
|------|-----------|-------------------|-------------|
| All models produce indistinguishable IC (efficient universe kills differentiation) | Medium | S3–S4, Ex3, D3 | Widen criteria to include ICIR, pct_positive, sub-period IC, and turnover as differentiators. Report paired t-test p-values — insignificant differences are themselves the teaching point about scale dependence. |
| Neural network fails to converge or produces degenerate predictions (constant output) | Medium | S4, Ex3, D3 | Use `prediction_quality()` check; fall back to sklearn MLPRegressor with default parameters as baseline; reduce hidden layer size to match small feature set. Degenerate NN is itself a teaching point: "this is what happens when the signal-to-noise ratio is too low for flexible models." |
| Feature matrix NaN pattern breaks walk-forward (some months have <100 complete observations) | Low | All predictive sections | Code probe shows minimum 166 complete observations per month. LightGBM handles NaN natively. For neural network, apply cross-sectional median imputation within each training window (no look-ahead). |
| VIX regime split produces too few months per bucket for IC comparison | Medium | Ex1 | Use terciles instead of median split (top/bottom 23 months each); or use rolling VIX (average of trailing 3 months) to smooth regime classification. Report the limitation: "n=23 per regime is insufficient for separate significance testing; focus on the regime-contrast." |
| Turnover is too low to demonstrate cost erosion (all models rank stocks similarly month-to-month) | Low | S6, Ex4 | Rebalancing monthly with ML signals typically produces 80–200% annual one-way turnover. If turnover is unexpectedly low (<30% annually), increase the cost assumption (50 bps) or compare monthly vs. quarterly rebalancing frequency. |
| Feature expansion (D2) adds noise features that reduce IC | Medium | S5, D2 | This is expected and pedagogically valuable: demonstrate that not all feature engineering helps. The acceptance criteria should allow for no improvement or marginal improvement. SHAP analysis reveals which additions contributed vs. which were noise. |
| Long-short portfolio Sharpe is negative before costs | Medium | S6, Ex4, D3 | A negative or near-zero Sharpe is a realistic outcome on an efficient S&P 500 universe with limited features. Frame honestly: "This is what happens with 7 features on the most researched stocks in the world. GKX needed 94 features and 3,000 stocks." Do not treat as failure. |
| yfinance API throttling during fundamental data download for D2 feature expansion | Low | D2 | Fundamentals are pre-cached in shared layer (`load_sp500_fundamentals`). The shared cache handles retry logic. If specific tickers fail, drop them — the exercise works with any subset of the 179 tickers that has fundamental coverage (~165 tickers per probe). |

### Known Constraints

- **Survivorship bias:** The 179-ticker universe is today's S&P 500 projected backward to 2014. Companies that were delisted, went bankrupt, or were removed from the index during 2014–2024 are missing. This inflates backtest returns by an estimated 1–4% annually and may inflate alpha model IC by creating an artificially homogeneous cross-section. All sections should note: "Universe: current S&P 500 — subject to survivorship bias." Compare self-built factor sorts against Ken French portfolios (which use full NYSE/AMEX/NASDAQ with proper PIT methodology) as a calibration check.

- **Point-in-time contamination of fundamental features:** The 4 fundamental features (pb_ratio_z, roe_z, asset_growth_z, earnings_yield_z) in the Week 3 feature matrix are derived from yfinance data indexed by fiscal period end dates, not filing dates. Reporting lag bias (~55 days median, tail to 90+ days) and restatement bias (silently updated to latest revision) contaminate these features. The Week 3 FeatureEngineer applied static (non-time-varying) fundamentals across the full 2014–2024 window, compounding the look-ahead issue. The +90 day PIT lag was NOT applied during Week 3's feature construction. Affected sections: all predictive sections (S3–S6, Ex1–Ex4, D1–D3). Mitigation: (a) document contamination explicitly, (b) isolate PIT effect by comparing model IC with all 7 features vs. only 3 technical features (momentum, reversal, volatility — which are PIT-clean), (c) note that IC from fundamental features may be inflated and should not be compared directly to production benchmarks.

- **Static fundamental features:** Week 3's FeatureEngineer applies a single point-in-time snapshot of fundamentals uniformly across all months in 2014–2024. This means the cross-sectional rankings by pb_ratio, roe, etc., do not change over time — a stock's fundamental rank in March 2014 is the same as in December 2024. This is a severe form of look-ahead bias beyond the standard PIT issue. The technical features (momentum, reversal, volatility) do vary monthly and are computed correctly from trailing price data.

- **Small feature set vs. production:** The 7-feature matrix is dramatically smaller than the 94 characteristics used in GKX (2020) or the 100+ factors typical at production funds. This limits model differentiation (trees and NNs converge on similar signal extraction from few features) and reduces the expected IC relative to published benchmarks. The Kelly-Malamud-Zhou (2023) "Virtue of Complexity" result specifically requires a larger feature set and broader universe to materialize — it is unlikely to reproduce on our data.

- **Large-cap efficiency:** S&P 500 constituents are the most heavily researched, most liquid stocks in the world. Alpha signals are weaker in large-caps than in small/mid-caps or the full market universe. Published ICs of 0.03–0.05 on CRSP data will likely translate to ICs of 0.01–0.04 on our S&P 500 subset, with less differentiation between model families.

- **Limited fundamental history:** yfinance returns ~4–5 annual periods of financial statement data (typically 2021–2025). Historical fundamental features for 2014–2020 cannot be computed from yfinance — they require Compustat or Refinitiv. This prevents time-varying fundamental feature construction and forces the static-sort methodology documented above.

- **No point-in-time alternative data:** Section 7 (Alternative Data) is conceptual/taxonomic only — no free alternative data source provides production-quality features for our universe and date range. SEC EDGAR filings are available via the shared data layer, but extracting structured alpha features from filing text is Week 7 territory. The section requires no data and produces no code output.

- **FRED VIX as regime proxy:** VIX (VIXCLS) is used in Ex1 to classify months as high-vol or low-vol. VIX reflects S&P 500 implied volatility, not realized volatility, and is itself a mean-reverting process. Using VIX level as a regime indicator is standard in practice but somewhat circular when the universe is S&P 500 constituents — the regime and the return cross-section are not independent. Document this limitation; the exercise is about IC regime dependence, not about VIX as a predictor.

### Statistical Implications

The recommended configuration (179 stocks x 130 months, 60-month train window, 69 OOS months) provides:

- **Cross-sectional power:** 179 stocks per month is sufficient for decile portfolio construction (18 stocks per decile, adequate for long-short return estimation), cross-sectional IC computation (n=179 per cross-section), and SHAP analysis. It is insufficient for ventile (5% quantile) sorts — these require 200+ stocks.
- **Time-series power for IC significance:** At IC = 0.03 with std = 0.08, the 69-month OOS sample yields t = 3.11 (p < 0.01). Even if the true IC is as low as 0.02, t = 2.07 (p = 0.04) — still marginally significant at 5%. Signal significance is achievable.
- **Statistical power for model comparison:** Paired t-tests on IC series of length 69 can detect an IC improvement of ~0.02 at 80% power (assuming IC std = 0.08). Differences smaller than 0.02 will not be statistically significant. This means the complexity ladder and model comparison exercises will likely find that model differences are statistically insignificant — which is the intended teaching point about scale dependence.
- **Panel for model training:** Each walk-forward window contains ~167 stocks x 60 months = ~10,000 training observations with 7 features. This is adequate for Ridge regression, LightGBM, and small neural networks (1–2 hidden layers, 32 neurons). It is insufficient for the deep architectures (3–5 hidden layers) used in GKX — another sandbox-vs-production gap.
- **Sub-period analysis:** The 69-month OOS period splits into two ~34-month halves. Within each half, IC significance at 5% requires IC > 0.027 (at std = 0.08). Sub-period significance testing is possible but has moderate power.

### Data Allocation Summary

| Section / Exercise | Data Source | Universe Subset | Specifics |
|--------------------|-----------|-----------------|-----------|
| S1: Cross-Sectional Setup | Week 3 feature matrix | 179 tickers | Load feature matrix + compute forward monthly returns from shared prices (2014-2024) |
| S2: Signal Evaluation | Week 3 feature matrix + FF3 | 179 tickers | Compute IC/rank IC from Fama-MacBeth predictions (Week 3 artifact or re-derived); FF3 RF for excess returns |
| S3: Gradient Boosting Alpha | Week 3 feature matrix + shared prices | 179 tickers | Walk-forward LightGBM, 60-month train window, monthly forward returns |
| S4: Neural Nets vs. Trees | Same as S3 | 179 tickers | Add feedforward NN on same data; head-to-head IC comparison |
| S5: Feature Engineering | Week 3 feature matrix + shared OHLCV + FF3 | 179 tickers | Extend features with interaction terms (momentum x volatility, etc.); retrain GBM; TreeSHAP analysis |
| S6: Signal to Portfolio | S3 predictions | 179 tickers | Decile sort, long-short returns, turnover, transaction cost haircut |
| S7: Alternative Data | None (conceptual) | N/A | Taxonomy diagram and discussion — no data or code |
| Ex1: IC Autopsy | S3 IC series + FRED VIX | 179 tickers + VIX | Split IC series by VIX regime (high/low); compare IC stability |
| Ex2: Feature Knockout | Week 3 feature matrix | 179 tickers | Leave-one-out and leave-two-out GBM retraining; IC comparison |
| Ex3: Complexity Ladder | Week 3 feature matrix | 179 tickers | 5 models (OLS, Ridge, LightGBM x2, NN) on same data; IC/ICIR comparison |
| Ex4: Turnover Tax | S3 or D1 portfolio output | 179 tickers | Turnover computation; cost at 5/20/50 bps; breakeven analysis |
| D1: Alpha Engine | Week 3 feature matrix + shared prices | 179 tickers | AlphaModelPipeline class with walk-forward, IC, portfolio construction |
| D2: Feature Lab | Shared prices + OHLCV + fundamentals (PIT-lagged) + FF3 | 179–300 tickers | Expand to 15–25 features; price-derived (momentum variants, vol, Amihud) + fundamental (D/E, FCF yield, etc.); retrain via D1 pipeline |
| D3: Model Comparison | D2 feature matrix | Same as D2 | 4+ model families; IC/ICIR/Sharpe/turnover/net Sharpe; sandbox-vs-production writeup |

---

## Per-Section Expectations

### Lecture Sections

### Section 1: The Cross-Sectional Prediction Problem

**Data:** Week 3 feature matrix (`feature_matrix_ml.parquet`): 179 tickers, 7 z-scored features, monthly from 2014-03 to 2024-12. Shared equity prices (`load_sp500_prices('2014-01-01', '2024-12-31')`) for forward monthly return computation. FF3 monthly factors (`load_ff_factors(model='3', frequency='M')`) for the risk-free rate (RF column) needed to compute excess returns.

**Acceptance criteria:**
- Feature matrix loads with shape ~(23,192, 7), MultiIndex (date, ticker), 130 unique months, 177–179 unique tickers per month.
- Forward 1-month returns computed for 129 months (2014-04 to 2024-12); no future data from 2025+.
- Cross-sectional excess returns (demeaned per month) have mean ~0 within each cross-section (|mean| < 1e-10 per month).
- Cross-sectional scatter plot of a single feature (momentum_z) vs. forward return shows a weak but non-zero positive relationship: Spearman correlation in range [-0.02, 0.10] for most months.
- DATA QUALITY block printed: panel shape, missing value percentage (expect ~6% in pb_ratio_z and roe_z, ~0% in technical features), universe coverage per month (expect 177–179 stocks).
- Survivorship bias acknowledged in structured output.

**Production benchmark:** GKX (2020) use 94 firm characteristics for ~3,000 CRSP stocks, 1957–2016, monthly. Our feature matrix is a 7-feature, 179-stock sandbox subset of this framework. The cross-sectional scatter is expected to be noisier and the feature-return relationships weaker than what GKX observe on the full universe.

---

### Section 2: The Language of Signal Quality

**Data:** Week 3 feature matrix (same as S1). Fama-MacBeth cross-sectional predicted returns — either loaded from Week 3 artifacts if available, or re-derived by running per-month OLS of forward returns on features and using the fitted values as cross-sectional predictions. FF3 monthly factors for excess return computation.

**Acceptance criteria:**
- IC (Pearson) time series computed for 60–129 months (depending on whether walk-forward is used or all months with predictions are included); each IC value is a cross-sectional correlation across ~179 stocks.
- Mean IC (Fama-MacBeth linear model): range [-0.01, 0.06]. A linear Fama-MacBeth predictor on 7 features for S&P 500 stocks is expected to show weak but positive cross-sectional predictive power.
- Rank IC (Spearman) and Pearson IC are both computed; rank IC is within +/-0.02 of Pearson IC (expected to be similar for rank-normalized features).
- ICIR = mean(IC) / std(IC) computed and reported; expected range [0.05, 0.50]. ICIR > 0.5 would be strong for a linear model on this data.
- IC t-statistic and p-value computed (implementing `ic_summary()` logic from scratch); if mean IC > 0.02, expect t > 1.5 on 60+ months.
- Fundamental law of active management demonstrated: IR = IC x sqrt(BR) computation with BR = 179 (stocks per month) and IC from above; show how weak IC translates to non-trivial IR given breadth.
- pct_positive (fraction of months with IC > 0) reported; expected range [0.50, 0.70] for a weakly positive signal.

**Production benchmark:** Practitioner consensus (CFA curriculum, MSCI Barra documentation): IC of 0.02–0.05 is realistic for cross-sectional equity return prediction. ICIR > 0.5 is considered a "good" signal. GKX (2020) report OOS R-squared of ~0.4% monthly for gradient boosting on CRSP, which corresponds approximately to IC of 0.03–0.05.

---

### Section 3: Gradient Boosting for Cross-Sectional Alpha

**Data:** Week 3 feature matrix (7 features, 179 tickers, 130 months). Shared equity prices for forward 1-month return target. Features at month t predict returns from t to t+1.

**ML methodology:**
- Validation: walk-forward with fixed 60-month training window, 1-month purge gap.
- Train window: 60 months (~10,000 stock-month observations per window).
- Hyperparameter search: yes — search `learning_rate` ([0.01, 0.05, 0.1]), `num_leaves` ([15, 31, 63]), `n_estimators` with early stopping (max 500, patience 50) on a temporal validation holdout (last 12 months of training window). Use time-series CV within training window (3-fold temporal split). Justify grid from Ke et al. (2017) defaults and small dataset considerations.
- Prediction horizon: 1-month forward returns. Standard for cross-sectional equity models per GKX (2020).

**Signal viability:** Moderate — 69 OOS months. At IC ~0.03 with std ~0.08, t ~3.1 — significance likely at 5% level. However, if the true IC is closer to 0.01 (plausible on efficient S&P 500), t ~0.9 — significance not guaranteed.

**Acceptance criteria:**
- Walk-forward produces predictions for 69 OOS months (2019-04 to 2024-12), ~179 stocks per month.
- Mean OOS IC (Pearson): range [0.005, 0.06]. Values below 0.005 suggest the model learned nothing; values above 0.06 on 7 features and S&P 500 warrant investigation of look-ahead bias.
- Mean OOS rank IC: range [0.005, 0.06], expected within +/-0.01 of Pearson IC.
- IC t-statistic reported with p-value. If mean IC >= 0.02, expect t > 1.96 (significant at 5%).
- If mean IC < 0.02 and t < 1.96: print `WEAK SIGNAL` warning. This is an acceptable outcome — document as sandbox limitation.
- ICIR (mean IC / std IC): range [0.05, 0.60].
- pct_positive: range [0.50, 0.72].
- `prediction_quality()` check on last OOS cross-section: spread_ratio > 0.05 (model is not predicting the unconditional mean).
- Overfitting check: if mean train IC > 2x mean OOS IC, print `OVERFIT` warning.
- DATA QUALITY block printed before training.
- Chosen hyperparameters and early stopping rounds reported in structured output.
- Baseline comparison: naive (previous-month return as prediction) IC series computed alongside. Paired test via `vs_naive_baseline()` logic: report mean improvement, t-stat, and p-value.

**Production benchmark:** GKX (2020) report OOS R-squared of ~0.40% monthly for gradient boosted regression trees (GBRT) on 94 features, full CRSP universe (1957–2016). This translates to IC approximately 0.03–0.05. Our sandbox with 7 features and 179 S&P 500 stocks should produce lower IC — the feature set is 13x smaller and the universe is the most efficient segment of the market. Tidy Finance GKX replication reports similar magnitudes. Kelly & Xiu (2023) survey corroborates IC of 0.02–0.05 as the realistic range for cross-sectional equity prediction with ML.

---

### Section 4: Neural Networks vs. Trees — The Honest Comparison

**Data:** Same as S3 — Week 3 feature matrix (7 features, 179 tickers, 130 months), shared equity prices for forward returns.

**ML methodology:**
- Validation: walk-forward with fixed 60-month training window, 1-month purge gap (identical to S3 for fair comparison).
- Train window: 60 months. Last 12 months of training window held out as validation set for early stopping.
- Hyperparameter search: yes — search `learning_rate` ([1e-4, 1e-3, 1e-2]), `hidden_size` ([16, 32, 64]), `dropout` ([0.1, 0.3, 0.5]). Small grid with temporal CV. Architecture: 1–2 hidden layers with ReLU activation, matching GKX NN1/NN2 specification.
- Prediction horizon: 1-month forward returns.

**Signal viability:** Moderate — 69 OOS months, same as S3. The NN may produce slightly lower or comparable IC to GBM on this small feature set. Significance depends on IC magnitude.

**Acceptance criteria:**
- NN walk-forward produces predictions for 69 OOS months, ~179 stocks per month.
- Mean OOS IC (NN): range [-0.01, 0.06]. Neural networks on 7 tabular features with ~10K training observations may underperform GBM. Negative mean IC is possible if the network overfits or fails to learn.
- `prediction_quality()` check: spread_ratio > 0.03. If spread_ratio < 0.03, print `DEGENERATE PREDICTIONS` warning — the network is predicting near-constant values.
- Head-to-head comparison: |GBM IC - NN IC| reported. Expected range [0.00, 0.03]. If GBM IC > NN IC (expected on small tabular data), the difference is likely statistically insignificant — report paired t-test (via `vs_naive_baseline()` logic on the two IC series).
- Paired t-test for GBM vs. NN IC series: report t-stat and p-value. Expected: p > 0.05 (not significant), confirming that model differences are small on this data.
- Training stability: report mean and std of stopping epoch across windows. If >50% of windows hit max epochs without early stopping, regularization may be insufficient.
- Overfitting check: if mean train IC > 3x mean OOS IC, print `OVERFIT` warning.

**Pedagogical alternatives:**
- **Primary:** Feedforward NN with 1–2 hidden layers (32 neurons per layer), walk-forward identical to S3.
- **Alternative A:** sklearn MLPRegressor with default parameters.
  Trigger: if PyTorch NN produces degenerate predictions (spread_ratio < 0.03) after tuning.
  Expected effect: MLPRegressor with simpler optimization may produce non-degenerate but weak predictions, making the comparison viable.
- **Alternative B:** Reduce NN to single hidden layer with 16 neurons.
  Trigger: if training is unstable (>30% of windows diverge or produce NaN losses).
  Expected effect: simpler architecture stabilizes training at the cost of capacity.

**Production benchmark:** GKX (2020) report OOS R-squared of ~0.40% for NN1 (single hidden layer) and ~0.44% for NN3 (three hidden layers) on CRSP with 94 features — marginally outperforming GBRT. On our 7-feature sandbox, this advantage is unlikely to materialize. Kelly, Malamud & Zhou (2023) demonstrate that the "Virtue of Complexity" requires scale (many features, large universe) — on our data, complexity may hurt or show no benefit. Research notes confirm practitioner consensus: "GBM for tabular data, NN for unstructured data" (QuantNet forums, Oxford-Man Institute).

---

### Section 5: Feature Engineering and Importance

**Data:** Week 3 feature matrix (7 original features). Shared equity prices and OHLCV data (`load_sp500_ohlcv('2012-01-01', '2024-12-31', fields=['Close','Volume'])`) for computing additional price-derived features. Interaction features computed from existing features: momentum_z x volatility_z, earnings_yield_z x pb_ratio_z, etc. Non-linear transformations: momentum_z^2, |reversal_z|. Expand from 7 to ~10–12 features for the lecture demonstration (the full 15–25 expansion is D2 homework territory).

**ML methodology:**
- Validation: walk-forward, 60-month train window, 1-month purge gap (reuse S3 pipeline).
- Train window: 60 months.
- Hyperparameter search: reuse S3 best hyperparameters (or re-tune on expanded features if time permits).
- Prediction horizon: 1-month forward returns.

**Signal viability:** Moderate — 69 OOS months. The IC change from feature expansion is expected to be small (+/-0.01), which is unlikely to reach statistical significance by itself. The paired comparison (before vs. after expansion) is the relevant test.

**Acceptance criteria:**
- Expanded feature matrix has 10–12 columns; no NaN introduced by interaction terms (interactions of non-NaN features produce non-NaN results).
- Mean OOS IC (expanded model): range [0.005, 0.07]. Feature expansion may improve, match, or slightly reduce IC relative to the 7-feature baseline.
- IC change (expanded minus baseline): range [-0.015, +0.025]. Small or zero improvement is expected and pedagogically valid — features are more correlated on a small universe.
- SHAP summary plot produced via TreeSHAP for the last OOS cross-section: top-5 features by mean |SHAP value| reported in structured output.
- SHAP feature ranking is dominated by 2–4 original features (momentum, volatility, and one fundamental feature expected to lead). Interaction features typically have lower importance — verify this pattern.
- Permutation importance (OOS) computed for top-5 features and reported alongside SHAP rankings; rank correlation between SHAP and permutation importance > 0.5 (methods should broadly agree).

**Production benchmark:** GKX (2020) find that all ML methods agree on the same dominant predictive signals: momentum (various horizons), liquidity, and short-term reversal. These 3 categories account for the majority of predictive power regardless of model complexity. Lopez de Prado (2018) documents the substitution effect: highly correlated features mask each other's importance in standard MDI — SHAP addresses this by computing marginal contributions.

---

### Section 6: From Signal to Portfolio

**Data:** S3 GBM predictions (69 OOS months, ~179 stocks per month). Shared equity prices for realized forward returns. `quantile_portfolios()`, `long_short_returns()`, `portfolio_turnover()`, `sharpe_ratio()`, `net_returns()` from `course/shared/backtesting.py`.

**Acceptance criteria:**
- Decile portfolios constructed for 69 OOS months: 10 groups, ~18 stocks per decile.
- Monotonic return pattern across deciles: top decile mean return > bottom decile mean return (if IC > 0). Monotonicity is not required to be strict across all 10 deciles — a broadly upward pattern suffices.
- Long-short monthly return series: 69 observations. Mean long-short return: range [-0.005, +0.015] per month. A mean near zero or slightly negative is realistic for 7 features on S&P 500.
- Gross annualized Sharpe ratio of long-short portfolio: range [-0.5, 1.5]. Both positive and negative Sharpe are acceptable outcomes on this data.
- Monthly one-way turnover: range [0.20, 0.80]. ML-based decile strategies typically have high turnover because prediction rankings shift month-to-month.
- Net returns computed at 10 bps one-way cost (default for S&P 500 large-caps): net Sharpe < gross Sharpe.
- If gross Sharpe > 0 but net Sharpe < 0: this is a teaching moment, not a failure. Print the cost drag explicitly: `Cost drag = mean_turnover x 2 x 10 bps = X bps/month`.
- Cumulative return plot produced for both gross and net long-short strategies.
- Max drawdown of long-short portfolio reported: range [-0.50, 0.00]. Deep drawdowns are expected for dollar-neutral strategies during crises (e.g., COVID 2020).
- Skewness and excess kurtosis of long-short returns reported alongside Sharpe (per rigor.md Tier 3 requirement — S6 touches strategy evaluation).

**Production benchmark:** GKX (2020) report that their gradient boosting decile strategy produces annualized long-short Sharpe of ~1.5 on CRSP (1957–2016) with 94 features. Our sandbox Sharpe should be substantially lower — 7 features on S&P 500 is a much weaker setup. A Sharpe of 0.3–0.8 gross would be a strong result; negative is realistic.

---

### Section 7: Alternative Data as Alpha Features

**Data:** None. This is a conceptual/discussion section. No data downloads, no model training, no code output beyond taxonomy diagrams or illustrative figures.

**Acceptance criteria:**
- Alternative data taxonomy presented: at minimum 5 categories (text/NLP, satellite/imagery, geolocation/foot traffic, transaction/credit card, web traffic/app usage).
- Institutional cost context: reference the BattleFin/Exabel (2025) survey statistic — $1.6M average annual hedge fund spending on alternative data; 98% of institutional investors surveyed agree traditional data is "becoming too slow."
- Bridge to Week 7 articulated: NLP-derived features identified as the most accessible alternative data category for ML engineers, with SEC EDGAR filings as the free data entry point.
- No claims about specific IC improvements from alternative data without citation.

---

## Seminar Exercises

### Exercise 1: The IC Autopsy

**Data:** S3 GBM IC series (69 monthly OOS IC values). FRED VIX monthly (`load_fred_series(['VIXCLS'], '2019-01-01', '2025-01-01')`) resampled to month-end to match IC dates. VIX median used to split months into high-volatility and low-volatility regimes.

**Acceptance criteria:**
- VIX regime classification: 69 OOS months split into ~34 high-vol and ~35 low-vol months (or terciles: ~23 each for top/bottom).
- Mean IC reported separately for each regime: both values in range [-0.05, 0.10].
- IC standard deviation in high-vol regime > IC standard deviation in low-vol regime (expected: alpha signals are less stable in turbulent markets).
- Regime contrast: |mean IC(high-vol) - mean IC(low-vol)| reported. No specific direction required — the signal may strengthen or weaken in high-vol months depending on feature composition.
- Two-sample t-test for IC difference between regimes: report t-stat and p-value. With n ~ 34 per group, the test has limited power — p > 0.10 is likely. Document this: "n = 34 per regime is insufficient for reliable significance testing; the directional pattern is more informative than the p-value."
- IC time series plot with VIX regime shading (high-vol months highlighted).
- pct_positive reported per regime: expected range [0.35, 0.75] for each.

**Production benchmark:** Practitioner experience and academic evidence (Moskowitz, Ooi & Pedersen, 2012, "Time Series Momentum," JFE) confirm that cross-sectional signals exhibit regime dependence. Momentum signals tend to reverse during volatility spikes (momentum crashes — Daniel & Moskowitz, 2016, "Momentum Crashes," JFE). Our exercise may or may not reproduce this pattern depending on which features dominate the GBM predictions.

---

### Exercise 2: The Feature Knockout Experiment

**Data:** Week 3 feature matrix (7 features, 179 tickers, 130 months). Same walk-forward setup as S3 (60-month window, 1-month purge, 69 OOS months).

**ML methodology:**
- Validation: walk-forward, 60-month train window, 1-month purge gap.
- Train window: 60 months.
- Hyperparameter search: reuse S3 best hyperparameters (do NOT re-tune per knockout — the point is to isolate the feature effect, not the HP effect).
- Prediction horizon: 1-month forward returns.

**Signal viability:** Moderate — 69 OOS months per knockout variant. The IC *difference* between full model and knockout model is the target. With 69 paired observations, a mean IC difference of 0.01 at std 0.05 gives t = 1.4 — marginally detectable. Larger effects (0.02+) are detectable.

**Acceptance criteria:**
- 7 leave-one-out (LOO) models trained, each dropping one feature. Mean OOS IC reported for each: all in range [-0.01, 0.06].
- IC drop per feature (full model IC minus LOO IC): reported for all 7 features. The feature with the largest IC drop is the most individually important.
- Largest single-feature IC drop: range [0.001, 0.03]. On 7 correlated features, the substitution effect means single-feature drops are small.
- Top-2 correlated features identified (from the original 7-feature correlation matrix); both removed simultaneously.
- IC drop from removing top-2 simultaneously: > sum of individual drops (superadditivity). Expected ratio: (joint drop) / (sum of individual drops) in range [1.0, 3.0]. This is the substitution effect.
- If ratio < 1.0 (subadditive): features are redundant but not substitutes in the model's splitting logic. This is a valid alternative outcome — report it honestly.
- Feature correlation matrix (7x7) printed in structured output to support the substitution interpretation.

**Pedagogical alternatives:**
- **Primary:** Leave-one-out + leave-two-out for the top correlated pair.
- **Alternative A:** Use permutation importance (shuffle one feature's values at prediction time) instead of retraining.
  Trigger: if retraining 7 LOO models x 69 windows is too slow (>45 minutes).
  Expected effect: permutation importance is faster (no retraining) but captures a different effect (marginal vs. structural importance).

**Production benchmark:** Lopez de Prado (2018) documents the substitution effect as a fundamental property of correlated features in tree-based models: MDI overstates the importance of correlated features because each can substitute for the other. SFI (Single Feature Importance) addresses this by evaluating each feature in isolation. GKX (2020) find that momentum, liquidity, and short-term reversal account for the majority of predictive power regardless of model complexity — the remaining features are substitutes.

---

### Exercise 3: The Complexity Ladder

**Data:** Week 3 feature matrix (7 features, 179 tickers, 130 months). Same walk-forward setup.

**ML methodology:**
- Validation: walk-forward, 60-month train window, 1-month purge gap, for all 5 models.
- Train window: 60 months.
- Hyperparameter search:
  - OLS: none (no hyperparameters).
  - Ridge: search `alpha` via log-spaced grid [1e-4, 1e-2, 1, 100, 1e4] with temporal CV inside training window.
  - LightGBM (depth=3): `max_depth=3` fixed; search `learning_rate` [0.01, 0.05, 0.1], `n_estimators` with early stopping.
  - LightGBM (depth=8): `max_depth=8` fixed; same search as depth=3.
  - Feedforward NN: reuse S4 architecture (1–2 hidden layers, 32 neurons, ReLU, dropout).
- Prediction horizon: 1-month forward returns.

**Signal viability:** Moderate — 69 OOS months per model. Individual model significance is plausible (t > 1.96 at IC = 0.03). Cross-model differences are unlikely to be statistically significant (paired t-test requires IC delta > 0.02 for significance at n=69).

**Acceptance criteria:**
- All 5 models produce IC series of length 69.
- Mean OOS IC per model, all in range [-0.01, 0.06]:
  - OLS: range [-0.01, 0.04]. Linear model with no regularization may overfit or produce weak signal.
  - Ridge: range [0.00, 0.05]. Regularization should stabilize predictions.
  - LightGBM depth=3: range [0.005, 0.06]. Shallow trees capture main effects.
  - LightGBM depth=8: range [0.005, 0.06]. Deeper trees capture interactions.
  - NN: range [-0.01, 0.06]. May match or underperform tree models.
- Complexity-performance relationship: IC does NOT monotonically increase with complexity. Expected pattern: plateau or slight reversal after LightGBM. The Kelly-Malamud-Zhou (2023) "Virtue of Complexity" result requires scale that our sandbox lacks.
- ICIR computed per model: range [0.0, 0.60]. ICIR may differ from IC ranking (a model with lower mean IC but higher ICIR is more stable).
- Paired t-tests for adjacent complexity levels (OLS vs. Ridge, Ridge vs. GBM-3, GBM-3 vs. GBM-8, GBM-8 vs. NN): report t-stat and p-value for each pair. Expected: most or all p > 0.05, confirming that model differences are not significant on this data.
- Summary table: model, mean IC, std IC, ICIR, pct_positive, t-stat, p-value.
- `prediction_quality()` check for each model: spread_ratio > 0.03 for all (degenerate predictions flagged).

**Production benchmark:** Kelly, Malamud & Zhou (2023) prove theoretically that complexity improves OOS prediction and demonstrate it empirically on CRSP with 94+ features and 3,000+ stocks. GKX (2020) show NN marginally outperforming GBM on the same full-scale setup. On our sandbox (7 features, 179 stocks), complexity gains are expected to be negligible or negative — this is the scale-dependence teaching point.

---

### Exercise 4: The Turnover Tax

**Data:** S3 GBM predictions (or D1 pipeline output) — 69 months of predictions for 179 stocks. Long-short decile portfolio returns and turnover from S6 (or recomputed). `portfolio_turnover()`, `net_returns()`, `sharpe_ratio()` from `course/shared/backtesting.py`.

**Acceptance criteria:**
- Monthly one-way turnover series: 68 values (first month has no prior for comparison). Mean turnover: range [0.20, 0.80].
- Annualized one-way turnover (mean monthly x 12): reported. Expected range [2.4, 9.6] (i.e., the portfolio turns over 2–10 times per year).
- Gross Sharpe ratio: from S6 output.
- Net Sharpe at 5 bps one-way: Sharpe reduction < 0.3 from gross (small cost, small impact).
- Net Sharpe at 20 bps one-way: Sharpe reduction between 0.2 and 1.0 from gross.
- Net Sharpe at 50 bps one-way: expected to be negative or near zero. This is the teaching point — even modest costs kill high-turnover signals.
- Breakeven cost level computed: the one-way cost (in bps) at which net Sharpe = 0. If gross Sharpe <= 0, breakeven is 0 bps — report: "Signal is unprofitable before costs."
- If gross Sharpe > 0: breakeven cost in range [2, 50] bps. Values below 5 bps indicate an extremely fragile signal.
- High turnover warning: if mean monthly one-way turnover > 0.50, print `HIGH TURNOVER` warning with cost drag estimate.
- Plot: net Sharpe as a function of cost level (0 to 100 bps) — monotonically decreasing curve.

**Production benchmark:** Large-cap S&P 500 execution costs are approximately 5–15 bps one-way for institutional investors (Frazzini, Israel & Moskowitz, 2018, "Trading Costs," SSRN). A signal that breaks even at 10 bps is borderline viable in production; one that breaks even at 5 bps is not. High-capacity factor strategies (value, momentum) have institutional turnovers of 50–100% annually, much lower than typical ML-based strategies.

---

## Homework Deliverables

### Deliverable 1: Cross-Sectional Alpha Engine

**Data:** Week 3 feature matrix (7 features, 179 tickers, 130 months). Shared equity prices for forward return computation. FF3 monthly factors for risk-free rate.

**ML methodology:**
- Validation: walk-forward with configurable train window (default 60 months), configurable purge gap (default 1 month). The class must accept both walk-forward and expanding window modes.
- Train window: configurable, default 60 months.
- Hyperparameter search: the class must support HP search via temporal CV within the training window. Default: LightGBM with the S3 grid.
- Prediction horizon: 1-month forward returns (configurable).

**Signal viability:** Moderate — 69 OOS months with default settings. Same assessment as S3.

**Acceptance criteria:**
- `AlphaModelPipeline` class implemented with at minimum: `__init__()`, `fit_predict()` (or `run()`), and `summary()` methods.
- Accepts any scikit-learn-compatible model (must work with at least `LinearRegression`, `Ridge`, and `LGBMRegressor`).
- `fit_predict()` runs the full walk-forward loop: returns IC series (DataFrame with columns [date, ic_pearson, ic_rank]) and prediction series (MultiIndex (date, ticker)).
- `summary()` returns a dict with at minimum: mean_ic, std_ic, icir, pct_positive, t_stat, p_value, mean_rank_ic, sharpe_gross, sharpe_net (at 10 bps), mean_turnover, max_drawdown, n_oos_months.
- Default run (LightGBM, 60-month window) produces mean IC in range [0.005, 0.06] — consistent with S3 results.
- Temporal integrity verified: for every prediction date t, all training data has date < t - purge_gap. No future data leakage.
- Purge gap correctly implemented: gap of at least 1 month between last training date and prediction date.
- DATA QUALITY block printed at start of run.
- Handles missing values: either passes them to LightGBM natively or imputes within each training window (cross-sectional median per feature, no temporal look-ahead).
- Long-short portfolio construction integrated: decile sort, cumulative returns, turnover, net returns at configurable cost.
- Class is reusable: D2 and D3 call it with different feature matrices and models.

**Production benchmark:** Microsoft Qlib framework provides a similar pipeline (data, model, evaluation, backtesting) as an open-source reference. The student's `AlphaModelPipeline` is a simplified, pedagogically transparent version of this production pattern.

---

### Deliverable 2: Feature Engineering Lab

**Data:** Week 3 feature matrix (7 original features) as baseline. Shared equity prices (`load_sp500_prices('2012-01-01', '2024-12-31')`) for multi-horizon momentum and additional price-derived features. Shared OHLCV (`load_sp500_ohlcv('2012-01-01', '2024-12-31', fields=['Close', 'Volume'])`) for Amihud illiquidity and volume-based features. Shared fundamentals (`load_sp500_fundamentals(tickers, pit_lag_days=90)`) for additional fundamental ratios (D/E, FCF yield, etc.). FF3 monthly factors for market beta computation. Ken French momentum decile portfolios (`load_ff_portfolios('10_momentum', frequency='M')`) for PIT-clean benchmark comparison.

**ML methodology:**
- Validation: walk-forward via D1 `AlphaModelPipeline`, 60-month train window, 1-month purge gap.
- Train window: 60 months.
- Hyperparameter search: re-tune LightGBM on expanded feature set (larger feature set may warrant different tree parameters).
- Prediction horizon: 1-month forward returns.

**Signal viability:** Moderate — 69 OOS months. IC improvement from feature expansion is expected to be small (+/-0.01). The paired comparison (7-feature baseline vs. expanded) has limited power to detect small improvements.

**Acceptance criteria:**
- Expanded feature matrix: 15–25 features. Must include at least:
  - 3+ new price-derived features (e.g., multi-horizon momentum [3m, 6m, 12m-1m], realized volatility at multiple horizons, Amihud illiquidity).
  - 2+ interaction features (e.g., momentum x volatility, value x quality).
  - 2+ new fundamental features (e.g., D/E ratio, FCF yield) with +90 day PIT lag applied. PIT mitigation stated in structured output.
- Cross-sectional rank normalization applied to all new features (per-month rank transform to [0, 1]).
- Feature correlation matrix printed: no two features with |corr| > 0.95 (near-duplicates should be dropped).
- Baseline IC (7 features) vs. expanded IC (15–25 features) comparison: IC change in range [-0.02, +0.03].
- Paired t-test on IC series (baseline vs. expanded): report t-stat and p-value. Expected: p > 0.05 in most cases (improvement not significant). Insignificant improvement is a valid result.
- SHAP summary plot for expanded model: top-10 features by mean |SHAP value|. Report in structured output.
- Feature importance ranking with economic interpretation: for each top-5 feature, a 1-sentence explanation of why it should (or should not) predict cross-sectional returns.
- PIT contamination check: compare IC using only PIT-clean features (price-derived) vs. all features. Report the difference — if IC is substantially higher with fundamental features included, PIT contamination is a likely contributor.
- If IC drops after expansion: document this as "feature noise dilution" — adding irrelevant features can hurt tree-based models by increasing the search space.

**Production benchmark:** GKX (2020) use 94 firm characteristics and find that expanding the feature set from ~10 to 94 improves OOS R-squared by approximately 50% (from ~0.3% to ~0.4% monthly). Jansen (2020) documents 100+ alpha factors with Python code. Our expansion from 7 to 15–25 features is a small fraction of the production feature space; marginal gains are expected to be modest.

---

### Deliverable 3: The Model Comparison Report

**Data:** D2 expanded feature matrix (15–25 features, 179+ tickers, 130 months). D1 `AlphaModelPipeline` class. Shared equity prices for forward returns. Transaction cost assumptions: 10 bps one-way (default for S&P 500 large-caps).

**ML methodology:**
- Validation: walk-forward via D1 pipeline, 60-month train window, 1-month purge gap, identical for all models.
- Train window: 60 months.
- Hyperparameter search: per-model:
  - OLS: none.
  - Ridge: alpha search [1e-4, 1e-2, 1, 100, 1e4].
  - LightGBM: search learning_rate, num_leaves, n_estimators with early stopping.
  - Feedforward NN: search learning_rate, hidden_size, dropout (reuse S4 grid).
- Prediction horizon: 1-month forward returns.

**Signal viability:** Moderate — 69 OOS months. Individual model significance is plausible. Cross-model differences are unlikely to be statistically significant (see Statistical Implications in Data Plan).

**Acceptance criteria:**
- At least 4 model families compared: OLS (or linear), Ridge, LightGBM, feedforward NN.
- Summary table for all models with: mean IC, std IC, ICIR, pct_positive, t_stat, p_value, mean rank IC, gross Sharpe, net Sharpe (10 bps), mean monthly turnover, max drawdown.
- All mean ICs in range [-0.01, 0.07].
- Pairwise paired t-tests: at minimum, (Ridge vs. OLS), (LightGBM vs. Ridge), (NN vs. LightGBM). Report t-stat and p-value for each.
- Expected: most pairwise differences are NOT statistically significant (p > 0.05). This is the central finding — on this data, model complexity does not significantly improve signal quality.
- If any model produces mean IC > 0.07 or mean IC < -0.03: investigate for bugs or data leakage.
- Multiple testing awareness: total number of model configurations tested reported (expected 4–6). If >6 configurations tested, note the multiple testing concern and compute deflated Sharpe ratio for the best model.
- Sandbox vs. production section: structured writeup that addresses at minimum (a) feature set gap (7–25 vs. 94+), (b) universe gap (179 S&P 500 vs. 3,000+ CRSP), (c) PIT contamination in fundamental features, (d) survivorship bias. Each gap must include a directional statement about how results would likely differ with institutional data.
- Recommendation: a 1-paragraph "CIO recommendation" that honestly assesses whether the model comparison supports switching from GBM to neural networks. Expected honest answer: "No — on this data, at this scale, the difference is not significant. The case for neural networks requires unstructured data or a substantially larger feature set."
- Net Sharpe comparison: at 10 bps one-way cost, at least one model should show positive net Sharpe OR all models should show negative net Sharpe with honest framing about sandbox limitations.

**Production benchmark:** GKX (2020) find that gradient boosted trees and neural networks (NN3, NN5) outperform linear models on full CRSP with 94 features. The performance ranking is: NN3 >= GBRT > Ridge > OLS > Lasso (by OOS R-squared). Chen, Pelger & Zhu (2024) show autoencoders outperform all GKX models. The magnitude of improvement from linear to nonlinear is approximately 30–50% in OOS R-squared on the full setup. On our sandbox, this improvement is expected to be much smaller (0–15%) or not detectable.

---

## Open Questions

1. **Will GBM IC on S&P 500 with 7 features be distinguishable from zero?**
   **Why unknowable:** The combination of an efficient large-cap universe, small feature set, static fundamental features with PIT contamination, and a 69-month OOS window creates genuine uncertainty about whether a statistically significant signal exists. Prior probes give t ~3.1 at IC=0.03, but the true IC could be 0.01 (t ~0.9).
   **Affects:** S3, S4, Ex3, D1, D3.
   **If IC is significant (t > 1.96):** The week operates as designed — model comparison, feature importance, and portfolio construction all have a real signal to work with.
   **If IC is not significant (t < 1.96):** Reframe around methodology demonstration. The pipeline and evaluation framework are correct even when the signal is weak. Use pct_positive and sign consistency as secondary evidence of directional learning.

2. **Will the neural network produce non-degenerate predictions?**
   **Why unknowable:** A feedforward NN with 1–2 hidden layers trained on ~10K stock-month observations with 7 features and monthly returns as target has a realistic chance of predicting near-constant values (the unconditional mean). Whether this happens depends on the specific HP configuration, initialization, and data realization.
   **Affects:** S4, Ex3, D3.
   **If non-degenerate:** Head-to-head comparison with GBM proceeds as planned.
   **If degenerate:** Document as a teaching point about signal-to-noise ratio limits for flexible models. Fall back to sklearn MLPRegressor or reduced architecture.

3. **What will the SHAP feature ranking look like — will momentum dominate, or will static fundamentals rank higher due to PIT contamination?**
   **Why unknowable:** The static fundamental features have perfect cross-sectional look-ahead, which could inflate their apparent importance. Whether this manifests as higher SHAP values depends on how the tree exploits the static ranking vs. the time-varying technical features.
   **Affects:** S5, Ex2, D2.
   **If fundamentals dominate:** Teaching point about PIT contamination inflating apparent feature importance. Compare with PIT-clean-only model (3 technical features) to isolate the effect.
   **If technicals dominate:** Validates that time-varying features provide genuine predictive signal even in the presence of contaminated alternatives.
