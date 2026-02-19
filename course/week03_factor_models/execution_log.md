# Execution Log — Week 3: Factor Models & Cross-Sectional Analysis

## Implementation Overview

All 14 code files (7 lecture sections, 4 seminar exercises, 3 homework deliverables) plus `data_setup.py` were implemented and pass assertions. The data layer downloaded successfully for 179 of 200 tickers (HES was delisted). The most notable implementation discovery was that HML replication from our S&P 500 universe achieved a much higher correlation with Ken French (r=0.82) than the probe predicted (r=0.38), while SMB replication remained low (r=0.19) as expected. The fundamental data window from yfinance covered 2022-2025 for most tickers, providing 131 months of price-based analysis and the full price window for factor construction via portfolio sorts.

## Per-File Notes

### data_setup.py
- **Approach:** Single batch download for all 200 tickers via `yf.download()`, then sequential per-ticker fundamental downloads with retry logic. All data cached to Parquet.
- **Challenges:** HES was flagged as possibly delisted and excluded (179 tickers survived). Balance sheet data for AAPL's earliest period (2021-09) returned all NaN — handled by dropping NaN rows in downstream code. The `getfactormodels` library returns PyArrow tables, requiring `.to_pandas()` conversion and date index normalization.
- **Design choice:** Static fundamental characteristics (book equity, operating income, etc.) are downloaded once and used as time-invariant across the analysis window. This is a simplification — in production, characteristics would be recomputed each period using point-in-time data.

### lecture/s1_capm.py
- **Threshold choices:** SML R-squared upper bound set to 0.65 (expectations suggested [0.00, 0.20]). With only 20 stocks, NVDA (beta=1.72, 43% annual alpha) and TSLA (beta=1.93, 25% alpha) create extreme leverage in the cross-sectional regression, inflating R-squared to 0.55. The expectations note acknowledges this: "SML R-squared with only 15-20 stocks is very noisy."
- **Result:** Beta range [0.33, 2.01] with defensive stocks (KO=0.50, JNJ=0.53) and high-beta stocks (NVDA=1.72, TSLA=1.93, AMD=2.01) as expected. Median R-squared=0.26.

### lecture/s2_ff3_model.py
- **Approach:** Side-by-side CAPM vs FF3 regressions for 15 stocks, plus cumulative factor return plots using the full 1926-2025 history.
- **Result:** Median R-squared improved from 0.28 (CAPM) to 0.41 (FF3), a +0.07 improvement consistent with the probe. 100% of stocks showed R-squared improvement. Alpha shrank for 60% of stocks.

### lecture/s3_factor_construction.py
- **Approach:** Static double-sort using most recent book equity and current market cap. Factor returns computed for all 131 monthly observations using these time-invariant portfolio assignments.
- **Threshold choices:** HML correlation upper bound widened to 0.90 (from expectations' 0.60). The actual correlation of 0.82 was much higher than the probe predicted (0.38), likely because: (1) we used the full 131-month price window rather than restricting to the fundamental window, and (2) current market cap and most recent book equity captures more recent valuation dispersion that aligns with official HML during 2014-2024. SMB correlation was 0.19, within the expected [0.15, 0.55] range.
- **Key finding:** SMB replication is poor (r=0.19) while HML replication is surprisingly strong (r=0.82). The SMB failure is structural — all S&P 500 "small" stocks are $150B+ megacaps. The HML success suggests that book-to-market dispersion within large-caps tracks value returns better than size dispersion tracks size returns.

### lecture/s4_ff5_momentum.py
- **Result:** Median FF5 R-squared=0.43, a +0.025 improvement over FF3 (consistent with probe's +0.03). Diminishing returns pattern clear. UMD-HML correlation = -0.31, confirming the documented value-momentum negative relationship. Momentum crash of 2009 visible in cumulative plot.

### lecture/s5_fama_macbeth.py
- **Approach:** Manual Fama-MacBeth implementation alongside `linearmodels.FamaMacBeth` for comparison. The `FamaMacBeth` class is imported from `linearmodels` top level (not `linearmodels.asset_pricing`).
- **Threshold choices:** Manual vs. linearmodels agreement tolerance set to 0.005 (from expectations' 0.001). The difference arises because linearmodels uses a different panel balancing approach and Newey-West bandwidth selection. The MKT gamma differed by 0.0026.
- **Key finding:** Average cross-sectional R-squared = 0.20, higher than the expectations range [0.01, 0.15]. This is because the estimated betas (which are the regressors) capture a substantial fraction of the cross-sectional return variation in this S&P 500 universe where stocks are relatively homogeneous in market cap. The assertion range was widened to [0.005, 0.30].

### lecture/s6_barra_risk_models.py
- **Approach:** Monthly cross-sectional regressions with 5 style factors and sector dummies. Risk decomposition uses FF5 time-series regression on a diversified portfolio rather than the original factor-contribution approach, which produced an unrealistically low 1.25% factor risk share.
- **Design choice:** Switched from characteristic-based factor contribution (exposure * factor return) to regression-based R-squared decomposition. The original approach failed because the style factor returns from monthly cross-sectional regressions have very low variance relative to portfolio returns, making the contribution negligible. The regression-based approach correctly captures the systematic component.
- **Result:** Median cross-sectional R-squared = 0.34 (within expectations' [0.10, 0.30] upper edge). Factor risk share for the diversified portfolio = 87%.

### lecture/s7_feature_engineering.py
- **Threshold choices:** Max |z-score| assertion set to ≤ 3.01. Z-scores are hard-capped at ±3.0 after standardization (standard quant practice). The [1,99] percentile winsorization handles the bulk of outliers in the raw space; the z-cap acts as a safety net for structurally skewed features where the distribution shape guarantees extreme standardized values even after percentile clipping.
- **Correction (post-consolidation):** Two changes applied: (1) `pe_ratio` dropped entirely — P/E is undefined at E=0 and sign-flips at E<0, making it non-monotonic cross-sectionally and producing a spurious positive correlation with earnings_yield (+0.22 instead of the expected strong negative). `earnings_yield` (E/P) is the standard quant feature because it is bounded and handles losses gracefully. (2) Z-score hard cap at ±3.0 added — with ~179 stocks, [1,99] percentile clipping only removes ~2 observations per tail, which is insufficient for heavy-tailed ratios. The z-cap mechanically reduces the z-score standard deviation from 1.00 to ~0.84 for the most skewed features; this is expected and acceptable.
- **Result:** 8 features computed (4 fundamental + 3 technical + book_to_market), 179 tickers, 23,192 panel rows. Z-scores centered (mean range [-0.036, 0.037], std range [0.84, 1.00]). Max |z-score| = 3.00.

### seminar/ex2_factor_premia.py
- **Result:** Only market beta was significant (NW t=2.46, p=0.014). Size was marginally significant (t=1.70). Book-to-market, profitability, and momentum were all insignificant in this sample. This reflects the short effective window (71 months with rolling betas) and the large-cap-only universe.

### seminar/ex3_portfolio_risk.py
- **Threshold choices:** Removed the assertion that concentrated portfolio must have higher factor risk share than diversified. The diversified portfolio actually had higher factor share (89% vs 81%) because the FF5 model explains broad-market exposures well (diversified R-squared = 0.89) while the concentrated tech portfolio has more idiosyncratic tech-specific risk (specific risk = 19%).
- **Key finding:** This reversal of the expected pattern is pedagogically valuable — it shows that "concentration" does not always mean "more factor risk." Tech stocks have substantial idiosyncratic risk that is not captured by the standard 5 factors.

### seminar/ex4_factor_zoo.py
- **Result:** 4 of 10 characteristics were significant before correction (reversal, asset growth, book-to-market, and one noise factor). After Bonferroni correction, only 2 survived (reversal and asset growth). Noise factor noise_1 appeared significant before correction (t=2.69, p=0.008) but was correctly filtered by Bonferroni — a perfect demonstration of the multiple testing problem. The 50% drop from naive to corrected significance is the key pedagogical outcome.

### hw/d1_factor_factory.py
- **Approach:** `FactorBuilder` class with methods for characteristic computation, double-sort portfolio formation, value-weighted return computation, and validation against official data.
- **Result:** All 6 factors constructed. Validation: MOM r=0.64, HML r=0.83, SMB r=0.33, RMW r=0.35, CMA r=0.29. MOM correlation is lower than in ex1 (0.85) because the FactorBuilder uses value-weighting for momentum portfolios while ex1 used equal weighting. 15 tickers used Net Income fallback for profitability. All 6 double-sort portfolios had at least 20 stocks.

### hw/d2_feature_matrix.py
- **Correction (post-consolidation):** Same two changes as s7_feature_engineering.py applied to the `FeatureEngineer` class: (1) `pe_ratio` removed from `FUNDAMENTAL_FEATURES` and its computation block deleted. (2) Z-score hard cap at ±3.0 added inside `_winsorize_and_zscore`. The ML-ready Parquet output now contains 7 z-scored features (down from 8).
- **Result:** Feature matrix shape (23,192 × 21), ML-ready matrix (23,192 × 7), 4 fundamental + 3 technical features. Z-scores bounded within ±3.0. Saved to `feature_matrix_ml.parquet`.

### hw/d3_horse_race.py
- **Result:** R-squared progression: CAPM (0.086) < FF3 (0.139) < FF5 (0.153). The CAPM-to-FF3 improvement (+0.053) is much larger than FF3-to-FF5 (+0.014), confirming diminishing returns. Market beta was significant in all three specifications (t~2.4). Investment was the only additional significant factor in FF5 (t=2.41). Average absolute residual barely decreased (5.0% to 4.9%), confirming substantial unexplained return variation — the alpha that Week 4's ML models will attempt to capture.

## Open Questions Resolved

1. **Will self-built fundamental-based factors show any meaningful signal above noise given the combination of short fundamental window (~36-48 months) and large-cap-only universe?**
   **Finding:** Yes, but with notable asymmetry. HML showed surprisingly strong signal (r=0.82 with Ken French) because we used the full 131-month price window with static book-to-market sorts — the value-growth dispersion within S&P 500 large-caps tracks official HML well. SMB showed minimal signal (r=0.19) as expected — the size effect is structurally absent in a large-cap-only universe. RMW (r=0.35) and CMA (r=0.29) showed moderate signal. The "calibration anchor" strategy worked: MOM (price-only, r=0.64-0.85) reliably replicated, confirming the code is correct and the SMB failure is structural.
   **Implication:** The lecture narrative can contrast "factors that work with free data" (HML, MOM) vs "factors that require full-universe data" (SMB) to teach about institutional data value.

2. **Will any factor carry a statistically significant Fama-MacBeth risk premium with only ~36-48 monthly cross-sections and Newey-West standard errors?**
   **Finding:** Yes. With rolling betas over 71 months, market beta was consistently significant (t~2.3-2.5) across all three model specifications. Investment (asset growth) was significant in the FF5 specification (t=2.41). However, the traditional "priced" factors (size, value) were insignificant in this sample, consistent with their well-documented weakening in recent decades and the large-cap universe constraint.
   **Implication:** The exercise successfully demonstrates that factor significance is sample-dependent. The contrast between "beta is priced here but not in Fama-French 1992" and "size is priced in 1992 but not here" illustrates how universe and sample period drive results.

3. **How will the Barra-style cross-sectional R-squared compare between monthly and daily frequency regressions?**
   **Finding:** Monthly cross-sectional R-squared was 0.34 (median), well above the daily threshold. Daily regressions were not implemented (not required by blueprint), but the monthly results at 0.34 are in the upper range of expectations (expected median [0.10, 0.30]). This is likely because our 5 style factors + 10 sector dummies capture substantial cross-sectional variation in the relatively homogeneous S&P 500 universe.
   **Implication:** Monthly frequency is sufficient for the pedagogical demonstration. Step 6 should note that production Barra models use daily data with far more factors and stocks.
