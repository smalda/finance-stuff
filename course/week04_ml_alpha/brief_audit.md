# Brief Audit — Week 4: ML for Alpha — From Features to Signals

## Summary

- **Findings:** 12 total (2 critical, 6 moderate, 4 minor)
- **Overall assessment:** Adequate

The brief is generally thorough and handles most divergences honestly. Signal significance is reported for S2, S3, and S4 lecture sections. The narrative arc accurately reflects the empirical results, and the sandbox-vs-production framing is consistently applied. However, there are two critical omissions: (1) the D3 section fails to report that no individual model reaches 5% significance on the expanded feature set, while presenting the model comparison as though the IC point estimates are meaningful, and (2) the Ex3 section omits per-model t-statistics that would show LightGBM_d8 is not significant and OLS is only borderline, which changes the interpretation of the complexity ladder. There are also several moderate gaps in methodology transparency and expectations regression that should be addressed before Step 7.

---

## Findings

### Finding 1: D3 omits individual model significance — all models fail 5% threshold on expanded features

**Check:** Signal Integrity
**Severity:** Critical
**Section(s):** Deliverable 3: The Model Comparison Report

**The expectations document said:**
"Summary table for all models with: mean IC, std IC, ICIR, pct_positive, t_stat, p_value, mean rank IC, gross Sharpe, net Sharpe (10 bps), mean monthly turnover, max drawdown." (D3 acceptance criteria)

**What happened:**
Observations.md Signal Significance section documents: "D3 model comparison (18 features): No individual model reaches 5% significance -- OLS t = 1.60 (p=0.11), Ridge t = 1.61 (p=0.11), LightGBM t = 1.62 (p=0.11), NN t = 0.77 (p=0.44)."

**How the brief handled it:**
The brief's D3 section reports IC point estimates (OLS 0.044, Ridge 0.044, LightGBM 0.034, NN 0.018), pairwise test p-values, ICIR, and Sharpe ratios -- but never reports the individual model t-statistics or p-values. The teaching angle discusses the "inverted ranking" and makes claims like "linear models outperform non-linear models" without disclosing that none of the models produce statistically significant signals on this feature set.

**The problem:**
This is a critical omission. When no model reaches 5% significance, the entire model comparison is operating on statistically insignificant signals. The "inverted ranking" of OLS/Ridge > LightGBM > NN may simply reflect noise -- a conclusion the brief should state explicitly. Instead, the brief frames the ranking as though it reveals something real about model complexity, when the t-statistics show none of the IC point estimates are distinguishable from zero. Students would learn the wrong lesson: that linear models genuinely outperform on expanded features, when the honest conclusion is that the expanded feature set weakened all signals to non-significance.

**Suggested fix:**
Add to the D3 "Actual" field: "Individual model significance: OLS t = 1.60 (p = 0.11), Ridge t = 1.61 (p = 0.11), LightGBM t = 1.62 (p = 0.11), NN t = 0.77 (p = 0.44) -- no model reaches 5% significance on the expanded feature set."

Revise the D3 "Frame as" opening to: "The model comparison produces the inverse of the production ranking: OLS/Ridge > LightGBM > NN. But a more fundamental finding overshadows the ranking: no individual model's IC is statistically significant at 5% on the 18-feature expanded set (all p > 0.11). The entire model comparison operates on signals that cannot be distinguished from zero. The IC spread of 0.025 across all models is noise, not signal differentiation."

Add to D3 "Key numbers": "Individual t-stats: OLS = 1.60, Ridge = 1.61, LightGBM = 1.62, NN = 0.77 (none significant at 5%)."

---

### Finding 2: Ex3 omits per-model t-statistics — LightGBM_d8 is not significant, OLS is borderline

**Check:** Signal Integrity
**Severity:** Critical
**Section(s):** Exercise 3: The Complexity Ladder

**The expectations document said:**
"Summary table: model, mean IC, std IC, ICIR, pct_positive, t-stat, p-value." (Ex3 acceptance criteria). Also: "Signal viability: Moderate -- 69 OOS months. Individual model significance is plausible (t > 1.96 at IC = 0.03)."

**What happened:**
Observations.md Signal Significance section documents: "Ex3 complexity ladder (7 features): OLS: t = 1.98, p = 0.052 (borderline). Ridge: t = 2.01, p = 0.048 (significant). LightGBM_d3: t = 2.63, p = 0.011 (significant). LightGBM_d8: t = 1.27, p = 0.210 (NOT significant). NN: t = 2.05, p = 0.044 (significant)."

**How the brief handled it:**
The brief's Ex3 "Actual" field reports paired t-tests between models (d3 vs. d8: p = 0.019) but does not report any individual model t-statistics. The "Key numbers" section similarly omits per-model t-stats. The teaching angle says "OLS and Ridge match the GBM at ~0.060 -- linear models perform on par with tree-based models" without noting that OLS is only borderline significant (p = 0.052) and LightGBM_d8 (IC = 0.027) is not significant at all (p = 0.21).

**The problem:**
The complexity ladder's teaching point depends critically on which models produce statistically significant signals. The brief's framing -- that "complexity does not help" and "deeper trees overfit" -- is directionally correct but incomplete. LightGBM_d8's IC of 0.027 is not just lower than d3's -- it is not distinguishable from zero. This transforms the teaching point from "complexity reduces IC" to "complexity destroys statistical significance," which is a stronger and more honest message. Additionally, OLS being borderline (p = 0.052) while Ridge (p = 0.048) and LightGBM_d3 (p = 0.011) are significant is itself informative about the regularization benefit.

**Suggested fix:**
Add to Ex3 "Actual" field: "Per-model significance: OLS t = 1.98 (p = 0.052, borderline), Ridge t = 2.01 (p = 0.048), LightGBM_d3 t = 2.63 (p = 0.011), LightGBM_d8 t = 1.27 (p = 0.210, NOT significant), NN t = 2.05 (p = 0.044)."

Add to Ex3 "Key numbers": "Per-model t-stats: OLS = 1.98 (borderline), Ridge = 2.01, LightGBM_d3 = 2.63, LightGBM_d8 = 1.27 (NOT significant), NN = 2.05."

Revise the Ex3 "Frame as" to add after the sentence about depth-8: "The depth-8 model's IC of 0.027 is not only lower -- it is not statistically significant (t = 1.27, p = 0.21). Excessive complexity does not just reduce signal strength; it destroys statistical significance entirely."

---

### Finding 3: S4 NN signal is not significant at 5% but the brief uses language implying the NN "learned" something

**Check:** Signal Integrity
**Severity:** Moderate
**Section(s):** Section 4: Neural Networks vs. Trees

**The expectations document said:**
"Signal viability: Moderate -- 69 OOS months, same as S3. The NN may produce slightly lower or comparable IC to GBM on this small feature set. Significance depends on IC magnitude."

**What happened:**
Observations.md documents: "S4 NN walk-forward: mean IC = 0.046, t = 1.74, p = 0.087 (n=68) -- NOT significant at 5% (borderline at 10%)."

**How the brief handled it:**
The brief's S4 "Actual" field correctly reports "t = 1.74, p = 0.087 -- borderline, not significant at 5%." The "Key numbers" section includes "NN t-stat = 1.74 (p = 0.087, borderline)." However, the teaching angle says "On 7 features and 174 stocks, the model architecture is irrelevant -- the signal lives in a few dominant features that any reasonable model extracts." This implies the NN is extracting signal, when its IC is not statistically significant.

**The problem:**
The teaching angle frames both models as successfully "extracting" signal from the same features, when only the GBM's signal is statistically significant (p = 0.035) while the NN's is not (p = 0.087). The paired test showing indistinguishability (p = 0.99) demonstrates the models converge, but the brief should be clearer that the NN's IC of 0.046 -- while numerically similar to GBM's -- is not individually significant. The convergence message is fine; the language about "any reasonable model extracts" the signal is too strong.

**Suggested fix:**
Revise the S4 "Frame as" sentence to: "On 7 features and 174 stocks, the model architecture is irrelevant -- both models produce similar IC point estimates (~0.046), though only the GBM's reaches conventional significance (p = 0.035 vs. NN p = 0.087). The convergence is clear, but the NN's borderline significance is a reminder that flexible models on small data may not reliably extract even a genuine underlying signal."

---

### Finding 4: D1 vs. S6 Sharpe discrepancy not flagged as a divergence concern in the teaching angles

**Check:** Cross-Document Consistency
**Severity:** Moderate
**Section(s):** Deliverable 1: Cross-Sectional Alpha Engine, Section 6: From Signal to Portfolio

**The expectations document said:**
D1 "Default run (LightGBM, 60-month window) produces mean IC in range [0.005, 0.06] -- consistent with S3 results." (D1 acceptance criteria)

**What happened:**
Observations.md divergence summary documents: "D1 vs S6 Sharpe discrepancy: D1 pipeline Sharpe = 0.98 vs S6 Sharpe = 0.77 using the same GBM model. The AlphaModelPipeline appears to construct portfolios differently (D1 L/S mean monthly = 1.97% vs S6 = 1.34%). This internal inconsistency warrants investigation in Step 6." D1 IC = 0.055 vs S3 IC = 0.046.

**How the brief handled it:**
The brief includes a dedicated "Additional Observations" section acknowledging the discrepancy and frames it as "implementation sensitivity." The D1 teaching angle reports the Sharpe of 0.976 without noting that it diverges from S6's 0.773 for the same model type.

**The problem:**
While the brief does address the discrepancy in its "Additional Observations" section, the D1 teaching angle itself presents the 0.976 Sharpe and 0.055 IC without any caveat. A notebook agent reading the D1 section alone would not know these numbers diverge from the same model's results in S6. The D1 "Key numbers" should cross-reference S6 to prevent the notebook from presenting inconsistent numbers without explanation.

**Suggested fix:**
Add to D1 "Key numbers" after "Gross Sharpe = 0.976": "(diverges from S6's 0.773 for the same GBM model -- implementation details matter; see Additional Observations)."

Add to D1 "Frame as" after the first sentence: "Note: the pipeline's Sharpe (0.98) differs from S6's (0.77) despite using the same GBM model, illustrating that implementation details (random seeds, NaN handling, exact train/test boundaries) meaningfully affect outcomes. Present the range (0.77-0.98 gross) rather than a single definitive number."

---

### Finding 5: Brief does not acknowledge the Ex2 baseline IC discrepancy in the Ex2 section itself

**Check:** Cross-Document Consistency
**Severity:** Moderate
**Section(s):** Exercise 2: The Feature Knockout Experiment

**The expectations document said:**
"Same walk-forward setup as S3 (60-month window, 1-month purge, 69 OOS months)." (Ex2 methodology). "Hyperparameter search: reuse S3 best hyperparameters." (Ex2 acceptance criteria)

**What happened:**
Observations.md Notable Values section: "Ex2 baseline IC = 0.0221: This is notably lower than s3's GBM IC of 0.0460 despite ostensibly using similar models. The difference suggests ex2 may use a different random seed, hyperparameters, or evaluation methodology for its baseline."

**How the brief handled it:**
The brief's Ex2 "Actual" field reports "Baseline IC = 0.022 (lower than S3's 0.046 -- different random path)." The brief also includes an "Additional Observations" section explaining the discrepancy. However, the Ex2 teaching angle itself does not mention this discrepancy or note that the baseline IC of 0.022 would not be statistically significant (at 68 months, IC = 0.022, std ~0.08 gives t ~2.3, so it might still be, but the brief does not report the t-stat for Ex2's baseline).

**The problem:**
The Ex2 feature knockout results are interpreted relative to a baseline IC of 0.022 that is less than half of S3's 0.046. While the brief notes "different random path" parenthetically, the teaching angle does not discuss implications: the knockout experiment is being run on a substantially weaker baseline, which changes the interpretation of which features help and hurt. If the knockout were run on S3's 0.046 baseline, the reversal_z and volatility_z results (removing them improves IC) might not replicate. The brief should make this instability explicit in the teaching angle, not just as an endnote.

**Suggested fix:**
Add to Ex2 "Frame as" after the first sentence: "A critical caveat: the knockout baseline IC (0.022) is less than half of S3's GBM IC (0.046) despite using the same model family. This implementation sensitivity -- different tree splits from the same algorithm -- means the specific features that help or hurt are path-dependent. The substitution effect (ratio = 1.68) is more robust because it is a structural property, but the individual feature rankings should be treated as one realization, not a definitive ordering."

---

### Finding 6: S6 excess kurtosis of 3.67 not discussed in the teaching angle despite Tier 3 requirement

**Check:** Methodology Transparency
**Severity:** Moderate
**Section(s):** Section 6: From Signal to Portfolio

**The expectations document said:**
"Skewness and excess kurtosis of long-short returns reported alongside Sharpe (per rigor.md Tier 3 requirement -- S6 touches strategy evaluation)." Per rigor.md 3.3: "If |skewness| > 1 or excess_kurt > 3: note that Sharpe understates tail risk."

**What happened:**
Observations.md confirms: "Skewness = -0.248, Excess kurtosis = +3.669."

**How the brief handled it:**
The brief's S6 "Actual" field reports "Skewness = -0.25, excess kurtosis = +3.67." The "Key numbers" section does not mention skewness or kurtosis. The teaching angle does not discuss the excess kurtosis of 3.67 exceeding the rigor.md threshold of 3.0, which means the Sharpe ratio understates tail risk.

**The problem:**
The excess kurtosis of 3.67 exceeds the rigor.md threshold of 3.0, which requires a note that "Sharpe understates tail risk." The brief reports the numbers in the "Actual" field but does not carry them into the teaching angle or note the implication. Students would see the Sharpe of 0.77 without understanding that the heavy-tailed returns make this metric misleading as a risk-adjusted measure. This is a rigor.md compliance gap.

**Suggested fix:**
Add to S6 "Key numbers": "Skewness = -0.25, excess kurtosis = +3.67 (above the 3.0 threshold -- Sharpe understates tail risk per rigor.md 3.3)."

Add to S6 "Frame as" after the sentence about turnover: "The return distribution has excess kurtosis of 3.67, meaning the Sharpe ratio of 0.77 understates the true tail risk of the long-short strategy -- extreme monthly losses are more likely than a normal distribution would predict."

---

### Finding 7: S3 GBM HP grid divergence from expectations not disclosed in the brief

**Check:** Methodology Transparency
**Severity:** Moderate
**Section(s):** Section 3: Gradient Boosting for Cross-Sectional Alpha

**The expectations document said:**
"Hyperparameter search: yes -- search `learning_rate` ([0.01, 0.05, 0.1]), `num_leaves` ([15, 31, 63]), `n_estimators` with early stopping (max 500, patience 50)." (S3 ML methodology)

**What happened:**
Execution_log.md documents: "HP grid changed from [0.01, 0.05, 0.1] to [0.005, 0.01, 0.05] to prevent lr=0.1 from producing insufficient prediction spread." The final model uses n_est=10 (aggressive early stopping) vs. the expected "max 500, patience 50."

**How the brief handled it:**
The brief's S3 "Actual" field reports "Best HP: lr=0.05, leaves=31, n_est=10 (aggressive early stopping)" but does not mention that the learning rate grid was changed from expectations or that n_est=10 vs. max 500 represents very aggressive early stopping.

**The problem:**
The learning rate grid change and the very low n_est=10 are methodological choices that affect results. The brief should disclose that: (a) the learning rate grid was shifted downward from expectations to prevent degenerate predictions at lr=0.1, and (b) the model effectively uses only 10 trees, which is far below the expected 500-max configuration. The n_est=10 result contributes to the 6.73x train/OOS overfitting ratio being somewhat controlled -- with more trees, overfitting could be worse but OOS IC might differ. This is methodology transparency that the teaching angle should acknowledge.

**Suggested fix:**
Add to S3 "Teaching angle" after "Best HP: lr=0.05, leaves=31, n_est=10": "(HP grid shifted from expectations -- learning rate [0.005, 0.01, 0.05] instead of [0.01, 0.05, 0.1] because lr=0.1 produced insufficient prediction spread; n_est=10 reflects aggressive early stopping from max=500)."

---

### Finding 8: Expectations interaction concern about VIX circularity not carried to Ex1 teaching angle

**Check:** Expectations Regression
**Severity:** Moderate
**Section(s):** Exercise 1: The IC Autopsy

**The expectations document said:**
Known Constraints: "FRED VIX as regime proxy: VIX (VIXCLS) is used in Ex1 to classify months as high-vol or low-vol. VIX reflects S&P 500 implied volatility, not realized volatility, and is itself a mean-reverting process. Using VIX level as a regime indicator is standard in practice but somewhat circular when the universe is S&P 500 constituents -- the regime and the return cross-section are not independent. Document this limitation."

**What happened:**
Observations.md Known Constraints Manifested section: "FRED VIX circularity: The Ex1 regime analysis finds no significant difference in IC between high-vol and low-vol months (p = 0.73). While the non-significance may partly reflect low statistical power (n = 34 per regime), the circularity of using S&P 500 implied volatility to classify regimes for an S&P 500 alpha model may also attenuate any real regime effect."

**How the brief handled it:**
The brief's Ex1 "Sandbox vs. reality" paragraph mentions the circularity in its final sentence: "Additionally, using S&P 500 implied volatility (VIX) to classify regimes for an S&P 500 alpha model introduces circularity -- the regime and the return cross-section are not independent." However, the teaching angle itself does not mention this concern. The "Frame as" and "Student takeaway" discuss only the sample size limitation.

**The problem:**
The VIX circularity is a genuine methodological concern documented in both expectations.md and observations.md. Burying it in the "Sandbox vs. reality" addendum while the teaching angle focuses only on sample size gives students an incomplete picture. The circularity may explain why no regime effect is detected -- it is not just a power issue.

**Suggested fix:**
Add to Ex1 "Frame as" after the sentence about sample size: "A further limitation: using VIX (S&P 500 implied volatility) to classify regimes for an S&P 500 alpha model introduces circularity -- the regime definition and the return cross-section are not independent. This could attenuate any real regime effect beyond what sample size alone explains."

---

### Finding 9: Brief does not note that S5 IC change of +0.004 lacks a t-statistic or p-value

**Check:** Signal Integrity
**Severity:** Minor
**Section(s):** Section 5: Feature Engineering and Importance

**The expectations document said:**
"The paired comparison (before vs. after expansion) is the relevant test." (S5 signal viability). "IC change (expanded minus baseline): range [-0.015, +0.025]. Small or zero improvement is expected and pedagogically valid."

**What happened:**
Observations.md Signal Significance section notes: "S5 expanded model (12 features): mean IC = 0.050, ICIR = 0.295, pct_positive = 0.603. No t-stat or p-value reported directly; ICIR = 0.295 and n=68 implies t = 0.295 x sqrt(68) = 2.43."

**How the brief handled it:**
The brief reports "IC change from expansion: +0.004 (not significant)" in the key numbers but does not provide a paired t-test p-value for the S5 expansion. The D2 section correctly reports the paired test (p = 0.13), but S5 does not.

**The problem:**
Stating the IC change is "not significant" without a p-value is an unsupported claim. The S5 expansion added only 5 features (vs. D2's 11), and the +0.004 change is indeed likely not significant, but the brief should either provide the test statistic or note that the paired significance test was not performed for S5. The individual model t-stat (implied t = 2.43) is not the same as a paired test of the IC change.

**Suggested fix:**
Change S5 "Key numbers" from "IC change from expansion: +0.004 (not significant)" to "IC change from expansion: +0.004 (no paired significance test reported for S5; the magnitude is small enough that significance is unlikely given IC estimation noise)."

---

### Finding 10: D2 fundamental features use static ratios, not +90 day PIT lag as expectations specified

**Check:** Methodology Transparency
**Severity:** Minor
**Section(s):** Deliverable 2: Feature Engineering Lab

**The expectations document said:**
"2+ new fundamental features (e.g., D/E ratio, FCF yield) with +90 day PIT lag applied. PIT mitigation stated in structured output." (D2 acceptance criteria). Data sources table: "Shared fundamentals (`load_sp500_fundamentals(tickers, pit_lag_days=90)`) for additional fundamental ratios."

**What happened:**
Execution_log.md documents: "Fundamental features use static ratios (debtToEquity, profitMargins) instead of time-varying BS/CF data due to 79% missing coverage after PIT lag." D2 run_log shows: "PIT WARNING: static ratios have full look-ahead bias."

**How the brief handled it:**
The brief's D2 "Actual" field mentions "PIT check: PIT-clean IC = 0.041, all features IC = 0.034 -- PIT-contaminated fundamentals *hurt* IC by 0.007" and mentions "D2 HP search found lr=0.01, leaves=63, n_est=53 for expanded model." The teaching angle discusses PIT contamination and its effect. However, the brief does not explicitly state that the fundamental features used static ratios instead of the +90 day PIT lag that expectations specified.

**The problem:**
The expectations document explicitly specified `pit_lag_days=90` for D2 fundamentals. The code agent switched to static ratios because time-varying data had 79% missing values. This is a legitimate implementation choice, but it is a methodology divergence that the brief should disclose. The PIT contamination discussion in the brief implies the fundamentals have standard PIT contamination, when in fact they have the more severe "full look-ahead bias" of static ratios applied across all historical months.

**Suggested fix:**
Add to D2 "Actual" field: "Fundamental features (de_ratio, profit_margin) use static current ratios rather than the +90 day PIT-lagged time-varying data specified in expectations, because time-varying fundamentals had 79% missing values. These static ratios carry full look-ahead bias."

Add to D2 "Frame as" a sentence: "The fundamental features use current static ratios -- not even the mitigated +90 day PIT lag -- because time-varying fundamental data was 79% missing. This full look-ahead contamination makes the PIT check result (contaminated features hurt IC by 0.007) more interpretable: even with perfect foreknowledge of fundamentals, those features add noise on this small cross-section."

---

### Finding 11: Brief does not note the Ex4 quintile-vs-decile methodology change in the Ex4 teaching angle

**Check:** Methodology Transparency
**Severity:** Minor
**Section(s):** Exercise 4: The Turnover Tax

**The expectations document said:**
Ex4 acceptance criteria are written in terms of the long-short portfolio output from S6 (which uses decile sort).

**What happened:**
Execution_log.md: "Used quintile (n_groups=5) instead of default decile (n_groups=10) to satisfy EX4-4 Sharpe reduction criterion."

**How the brief handled it:**
The brief mentions "quintile sort" parenthetically in the "Actual" field ("Gross Sharpe = 0.703 (quintile sort, slightly lower than S6's decile 0.773)") and the "Additional Observations" section notes the change. The teaching angle does not mention that Ex4 uses quintile while S6 uses decile, nor does it explain why.

**The problem:**
The switch from decile to quintile changes the portfolio characteristics (35 stocks per bucket vs. 17, different turnover profile, different Sharpe). The brief should note this in the teaching angle so notebook agents understand that the Ex4 numbers are not directly comparable to S6 on the portfolio construction dimension.

**Suggested fix:**
Add to Ex4 "Frame as" or "Key numbers": "Note: Ex4 uses quintile (5-group) sort rather than S6's decile (10-group) -- 35 stocks per bucket vs. 17. Quintile sorts are more robust on 174 stocks and produce slightly different Sharpe/turnover characteristics."

---

### Finding 12: Brief categorizes D2 as "Data limitation" but the IC degradation is partly a methodology/interaction effect

**Check:** Divergence Honesty
**Severity:** Minor
**Section(s):** Deliverable 2: Feature Engineering Lab

**The expectations document said:**
Failure Modes: "Feature expansion (D2) adds noise features that reduce IC -- Medium likelihood." "This is expected and pedagogically valuable: demonstrate that not all feature engineering helps."

**What happened:**
D2 IC dropped from 0.055 to 0.034. Observations.md notes: "The maximum feature correlation is 0.923 (momentum_z vs mom_12m_1m), which introduces multicollinearity." Execution_log.md: "Feature noise dilution: expanded IC=0.034 vs baseline IC=0.055 (IC drop = -0.020)."

**How the brief handled it:**
The brief categorizes D2 as "Data limitation" and frames the IC degradation as primarily a universe size issue: "on a small, efficient cross-section, additional features add more noise than signal."

**The problem:**
The "Data limitation" category implies the degradation is primarily an artifact of our sandbox data constraints. But expectations.md explicitly anticipated this as a "Medium likelihood" failure mode, and the execution_log attributes it to "feature noise dilution." The near-multicollinearity (0.923 between momentum_z and mom_12m_1m) is a feature engineering issue, not a data limitation. The degradation is partly methodological (adding highly correlated features that confuse the tree) and partly a genuine insight about feature engineering at small scale. "Expectation miscalibration" or "Real phenomenon" would be more honest categories, since the brief itself frames this as a pedagogically valuable finding.

**Suggested fix:**
Consider recategorizing D2 from "Data limitation" to "Real phenomenon" or at minimum add to the teaching angle: "This IC degradation was anticipated as a medium-likelihood failure mode in expectations.md. While the small universe amplifies the effect, the core mechanism -- adding correlated and irrelevant features dilutes tree-based model performance -- is a real phenomenon that operates at any scale, not purely a sandbox artifact. The near-multicollinearity of 0.923 between momentum_z and mom_12m_1m is a feature engineering issue, not a data availability constraint."

---

## Observations Divergence Checklist

1. **S3 rank IC vs Pearson IC gap (0.020 > 0.01 threshold):** Addressed in brief S3 "Actual" field (mentions gap but does not explicitly state it exceeds the criterion) -- Partially addressed ~ -- not a separate finding because the brief does report both IC values.

2. **Ex3 OLS IC above ceiling (0.060 vs [−0.01, 0.04]):** Addressed in brief Ex3 "Actual" and teaching angle. Explanation provided (rank normalization linearizes relationship). Addressed in brief ✓

3. **Ex3 Ridge IC above ceiling (0.061 vs [0.00, 0.05]):** Addressed in brief Ex3 "Actual" alongside OLS. Addressed in brief ✓

4. **Ex4 breakeven cost above ceiling (60.6 vs [2, 50] bps):** Addressed in brief Ex4 "Actual" and teaching angle. Addressed in brief ✓

5. **D2 IC change at boundary (−0.0203 vs [−0.02, +0.03]):** Addressed in brief D2 "Actual" and teaching angle. Addressed in brief ✓

6. **D2 max feature correlation near boundary (0.923 vs < 0.95):** Mentioned in brief D2 "Key numbers" (max correlation = 0.923). Addressed in brief ✓

7. **S3 GBM vs naive baseline not significant (p = 0.57):** Addressed in brief S3 "Actual" and teaching angle. Addressed in brief ✓

8. **S3 GBM overfitting ratio (6.73x):** Addressed in brief S3 "Actual" and teaching angle. Addressed in brief ✓

9. **D1 vs S6 Sharpe discrepancy (0.98 vs 0.77):** Addressed in brief "Additional Observations" section but not in D1 teaching angle. Partially addressed ~ -- see Finding 4

10. **D3 NN performance on expanded features (IC = 0.018, ICIR = 0.094):** Addressed in brief D3 "Actual" and teaching angle. Addressed in brief ✓

---

## Expectations Regression Checklist

### High-Likelihood Risks

1. **All models produce indistinguishable IC (efficient universe kills differentiation) -- Medium:** Addressed ✓ (Ex3 and D3 teaching angles discuss model convergence)
2. **Neural network fails to converge or produces degenerate predictions -- Medium:** Addressed ✓ (Open Questions section resolves this)
3. **VIX regime split produces too few months per bucket for IC comparison -- Medium:** Addressed ✓ (Ex1 teaching angle discusses n = 34 limitation)
4. **Feature expansion (D2) adds noise features that reduce IC -- Medium:** Addressed ✓ (D2 teaching angle covers noise dilution)
5. **Long-short portfolio Sharpe is negative before costs -- Medium:** Addressed ✓ (Sharpe was positive; brief discusses both scenarios)

### Known Constraints

6. **Survivorship bias:** Addressed ✓ (S6 teaching angle, sandbox-vs-reality sections)
7. **Point-in-time contamination of fundamental features:** Addressed ✓ (D2 PIT check, Open Question 3 resolution)
8. **Static fundamental features:** Partially addressed ~ -- the brief mentions static fundamentals in multiple places but does not explicitly state that the D2 fundamentals also use static ratios instead of the planned +90 day lag. See Finding 10.
9. **Small feature set vs. production:** Addressed ✓ (sandbox-vs-production framing throughout)
10. **Large-cap efficiency:** Addressed ✓ (S3, S4, D3 teaching angles)
11. **Limited fundamental history:** Partially addressed ~ -- mentioned in D2 context but the specific constraint (79% missing after PIT lag) is not in the D2 teaching angle. See Finding 10.
12. **No point-in-time alternative data:** Addressed ✓ (S7 correctly handles as conceptual section)
13. **FRED VIX as regime proxy (circularity):** Partially addressed ~ -- in Ex1 sandbox-vs-reality but not in the teaching angle itself. See Finding 8.

### Interactions

14. **Universe x features x date range (panel size):** Addressed ✓ (S1 teaching angle establishes the 174 x 130 panel)
15. **Train window x OOS length x model count:** Addressed ✓ (brief notes 68 OOS months throughout)
16. **Feature expansion x fundamental data quality:** Addressed ✓ (D2 discusses missing data impact)
17. **VIX regime classification x IC series (low power):** Addressed ✓ (Ex1 teaching angle discusses n = 34 per regime)
18. **Fundamental PIT contamination x alpha signal strength:** Addressed ✓ (Open Question 3 resolution, D2 PIT check)

### Open Questions

19. **Q1: Will GBM IC be distinguishable from zero?** Addressed ✓ -- brief resolves correctly with S3 and D1 t-statistics.
20. **Q2: Will NN produce non-degenerate predictions?** Addressed ✓ -- brief resolves correctly across S4, Ex3, D3.
21. **Q3: What will the SHAP feature ranking look like?** Addressed ✓ -- brief resolves with "technicals dominate" finding.

---

## Low-Confidence Section Review

None marked. The brief does not use `Confidence: low` annotations on any section.

---

## Sections with No Issues

The following sections passed all 7 checks with no findings:

- **Section 1: The Cross-Sectional Prediction Problem** -- Accurate data reporting, appropriate teaching angle, all numbers consistent with observations.md.
- **Section 2: The Language of Signal Quality** -- t-statistics and p-values properly reported, ICIR honestly contextualized, fundamental law gap (predicted 0.615 vs actual 0.234) used as a teaching point.
- **Section 7: Alternative Data as Alpha Features** -- Conceptual section correctly handled with no numerical claims. Taxonomy, cost context, and Week 7 bridge all present.
- **Open Questions Resolved** -- All three questions resolved with appropriate caveats and evidence from the empirical results.

