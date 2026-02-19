# Brief Audit — Week 3: Factor Models & Risk Decomposition

## Summary

- **Findings:** 10 total (0 critical, 5 moderate, 5 minor)
- **Overall assessment:** Adequate

The brief is thorough and handles the most important divergences honestly — particularly the static-sort / look-ahead bias issue for HML, which is disclosed extensively with appropriate caveats. The narrative arc adjustments are well-motivated and evidence-grounded. The brief earns "Adequate" rather than "Conservative" because of several moderate omissions: the point-in-time violation constraint is never mentioned despite being flagged in expectations.md across 10 sections; the S6 risk decomposition methodology switch is disclosed but understates how dramatically the original approach failed; a noise factor surviving Benjamini-Hochberg correction is not flagged; and the Fama-MacBeth open question is resolved under more favorable conditions than originally posed without sufficient disclosure.

---

## Findings

### Finding 1: Point-in-time violation constraint silently dropped

**Check:** Expectations Regression
**Severity:** Moderate
**Section(s):** S3, S6, S7, Ex1, D1, D2 (all fundamental-dependent sections)

**The expectations document said:**
"Point-in-time violation: yfinance fundamental data is 'as-reported' but not point-in-time — we cannot verify exact reporting dates, introducing potential look-ahead bias. In production (CRSP/Compustat), point-in-time databases track when each data point was actually available to investors. Our factor construction may use fundamentals before they were publicly reported. This affects S3, S6, S7, Ex1, D1, D2." (Known Constraints section)

**What happened:**
The execution_log confirms: "Static fundamental characteristics (book equity, operating income, etc.) are downloaded once and used as time-invariant across the analysis window. This is a simplification — in production, characteristics would be recomputed each period using point-in-time data." The point-in-time violation is compounded by the static sort, since fundamentals from ~2025 are applied backward to 2014 returns.

**How the brief handled it:**
The brief never mentions point-in-time violation. The static-sort discussion covers look-ahead bias from applying recent classifications backward, but this is a different (though related) concern. Point-in-time bias means that even within the fundamental window (~2021-2025), we cannot verify that a balance sheet filed in March 2022 was actually used after its filing date vs. during the prior quarter when it was not yet public. The brief's extensive look-ahead discussion subsumes part of this but not the reporting-date uncertainty.

**The problem:**
Point-in-time violation is a core concept in quantitative finance that the expectations document flagged as a Known Constraint affecting 10 sections/exercises/deliverables. Students learning factor construction should understand that production implementations use point-in-time databases precisely because "as-reported" data introduces forward-looking bias even when the analysis window matches the fundamental window. Omitting this concept means students learn about universe bias and data depth limitations but miss a third major data quality concern that every practitioner encounters.

**Suggested fix:**
Add a sentence to the Section 3 "Sandbox vs. reality" field and the D1 "Sandbox vs. reality" field:
> "A third source of bias: yfinance fundamental data is 'as-reported' but not point-in-time. We cannot verify exact reporting dates, so our factor sorts may use fundamentals before they were publicly available. In production, CRSP/Compustat point-in-time databases track when each data point became known to investors, preventing this contamination."

Also add to the "Key Caveats Established" in the curriculum_state entry:
> "- yfinance fundamental data is not point-in-time: reporting dates are unknown, introducing potential look-ahead bias beyond the static-sort issue"

---

### Finding 2: S6 risk decomposition methodology switch understated

**Check:** Methodology Transparency
**Severity:** Moderate
**Section(s):** S6, Ex3

**The expectations document said:**
Section 6 should use Barra-style cross-sectional factor contribution for risk decomposition. "Risk decomposition for a sample equal-weight portfolio: common factor risk should explain 40-80% of total portfolio return variance." (S6 acceptance criteria). Specific risk "20-60%." (S6 acceptance criteria).

**What happened:**
Execution_log: "Risk decomposition uses FF5 time-series regression on a diversified portfolio rather than the original factor-contribution approach, which produced an unrealistically low 1.25% factor risk share." The original Barra-style approach failed dramatically (1.25% factor risk), prompting a switch to FF5 regression (86.6% factor risk).

**How the brief handled it:**
S6 Sandbox vs. reality: "The high R-squared and factor risk share partly reflect the risk decomposition methodology: the code uses FF5 time-series regression for portfolio risk decomposition rather than a pure Barra cross-sectional factor contribution approach." Ex3 Sandbox vs. reality: "The code uses FF5 time-series regression for risk decomposition rather than a Barra-style cross-sectional factor contribution approach."

**The problem:**
The brief discloses the methodology switch but omits the critical detail that the original Barra approach produced 1.25% factor risk share — a number so low it was deemed unrealistic and abandoned. This is pedagogically important because it reveals that Barra-style factor contribution and FF5 regression answer different questions and can produce wildly different risk decompositions for the same portfolio. Students should know that the "87% factor risk" number comes from the FF5 regression approach and that the Barra approach (which is what the section claims to demonstrate) produced a dramatically different answer. Without this context, students may conflate two distinct risk decomposition methodologies and think Barra models naturally produce 87% factor risk shares.

**Suggested fix:**
Add to S6 teaching angle or Sandbox vs. reality:
> "An important implementation note: the original Barra-style factor contribution approach (exposure times factor return) produced a factor risk share of only 1.25% — far too low to be useful. This happened because the style factor returns from monthly cross-sectional regressions have very low variance relative to portfolio returns. The code switched to FF5 time-series regression, which correctly captures the systematic component but is a different methodology. The 87% factor risk share reflects the FF5 regression approach, not the Barra cross-sectional factor contribution. Production Barra models avoid this problem by using daily data with thousands of stocks, which stabilizes factor return estimates."

---

### Finding 3: Noise factor surviving Benjamini-Hochberg not disclosed in Ex4

**Check:** Conservative Framing
**Severity:** Moderate
**Section(s):** Ex4

**The expectations document said:**
"After Benjamini-Hochberg correction at FDR = 0.05: 1-4 characteristics survive. BH is less conservative than Bonferroni, so more signals may survive, but the noise characteristics should still be filtered out." (Ex4 acceptance criteria)

**What happened:**
Observations report: "bh_significant=4" and the 4 naive significant characteristics are reversal, asset_growth, book_to_market, and noise_1. Since BH also gives 4, noise_1 (a pure noise factor) survives BH correction. Observations report: "Noise factor noise_1 appeared significant before correction (t=2.69, p=0.008)."

**How the brief handled it:**
Brief Ex4 teaching angle: "Without correction, 4 of our 10 characteristics appear significant — including one that is pure random noise (noise_1, t = 2.69). After Bonferroni correction, only 2 survive, and the noise factor is correctly filtered out." Key numbers: "BH: 4 survive." The brief focuses on Bonferroni filtering out noise but does not mention that BH fails to filter noise_1.

**The problem:**
The expectations document predicted that noise factors "should still be filtered out" by BH. The actual result contradicts this prediction — BH at FDR=5% retains all 4 naive significant factors including noise_1. This is a pedagogically important failure of BH in a small-scale multiple testing scenario and directly relevant to the factor zoo lesson. The brief's narrative implies that correction methods reliably filter noise, but the data shows BH does not accomplish this here. Students should learn that BH controls the false discovery *rate*, not individual false discoveries, and that with only 10 tests a noise factor can survive.

**Suggested fix:**
Add to Ex4 teaching angle after the Bonferroni discussion:
> "A subtlety: Benjamini-Hochberg at FDR=5% retains all 4 naive significant factors — including the noise factor. BH controls the expected *proportion* of false discoveries, not individual false positives. With only 10 tests, a noise factor at p=0.008 can survive BH even though it is pure noise. This illustrates why BH is insufficient for the factor zoo: controlling the false discovery rate at 5% still allows individual false discoveries through. Bonferroni and the HLZ threshold (t > 3.0) are more conservative and correctly reject noise_1."

Update key numbers to: "BH (FDR=5%): 4 survive (including noise_1 — BH fails to filter the false discovery)."

---

### Finding 4: Open Question 2 resolved under more favorable conditions than posed

**Check:** Open Question Integrity
**Severity:** Moderate
**Section(s):** Ex2, D3, Open Questions Resolved

**The expectations document said:**
"Will any factor carry a statistically significant Fama-MacBeth risk premium with only ~36-48 monthly cross-sections and Newey-West standard errors? Why unknowable: Statistical power depends on the true effect size relative to noise, and with T~40, even a real premium of 0.5% monthly may not reach significance at 5% with Newey-West correction." (Open Question 2)

**What happened:**
The actual test used 71 monthly cross-sections (after rolling-beta estimation consumes ~60 months), not 36-48 as the question specified. 71 months provides substantially more statistical power than 40 — the standard error of the mean scales as sigma/sqrt(T), so going from T=40 to T=71 reduces standard errors by ~25%.

**How the brief handled it:**
Brief Open Question 2 resolution: "Yes. Market beta is consistently significant (t = 2.3-2.5)...The effective window is 71 months (after rolling-beta estimation), not 36-48."

**The problem:**
The brief acknowledges that the effective window is 71 months, which is good. However, it does not note that the question was specifically about the ~36-48 month scenario (which corresponds to the fundamental window), and that the 71-month window represents substantially more favorable conditions. The brief resolves the question as "Yes" without quantifying how much the extra statistical power contributed. The question's "If zero factors are significant" contingency (expectations.md) was never tested because the scenario it anticipated (fundamental-window-limited cross-sections) did not materialize for the rolling-beta approach.

**Suggested fix:**
Add a sentence to Open Question 2 resolution:
> "Note: the question asked about ~36-48 monthly cross-sections (the fundamental window), but the actual test used 71 months because rolling betas consume the first ~60 months of the price window, leaving 71 months of cross-sectional data. This nearly doubles the statistical power relative to the question's premise (standard errors ~25% smaller at T=71 vs. T=40). Whether beta would remain significant with only T=40 cross-sections is untested — the expectations document's contingency ('if zero factors are significant') remains an open possibility for the fundamental-window scenario."

---

### Finding 5: D1 MOM correlation discrepancy between Ex1 and D1 unexplained

**Check:** Cross-Document Consistency
**Severity:** Moderate
**Section(s):** D1, Ex1

**The expectations document said:**
MOM correlation should be in [0.50, 0.85] and serve as the "calibration anchor" — proof that the code works correctly (Ex1, D1 acceptance criteria).

**What happened:**
Observations cross-file consistency item 3: "MOM correlation with Ken French — Reported in two places: ex1_replicate_ff (0.8460) and d1_factor_factory (0.6395). The seminar Exercise 1 reports a substantially higher momentum correlation (0.85) than the homework D1 (0.64). This is a notable divergence between the two implementations." Execution_log D1: "MOM correlation is lower than in ex1 (0.85) because the FactorBuilder uses value-weighting for momentum portfolios while ex1 used equal weighting."

**How the brief handled it:**
Brief D1 reports MOM r=0.64. Brief Ex1 reports MOM r=0.85. The brief does not explain why the same factor (momentum) has such different correlations across the two implementations. The execution_log's explanation (equal-weight vs. value-weight) is not mentioned in the brief.

**The problem:**
If MOM is the "calibration anchor" (proof the code is correct), then a 0.21 correlation gap between two implementations of the same factor undermines the anchor's credibility. Students who notice MOM r=0.85 in Ex1 and MOM r=0.64 in D1 will reasonably wonder whether one implementation has a bug. The explanation — equal-weighting vs. value-weighting — is itself pedagogically important: it demonstrates that "the momentum factor" is not a unique object but depends on portfolio construction choices. The brief should disclose this.

**Suggested fix:**
Add to D1 teaching angle or methodological note:
> "Note: MOM correlation is 0.64 here vs. 0.85 in Exercise 1. The difference is not a bug — Exercise 1 uses equal-weighted portfolios for momentum, while the FactorBuilder uses value-weighted portfolios (matching the Fama-French methodology). Value-weighting concentrates the portfolio in mega-cap stocks, whose momentum signals are noisier relative to the broad market. This 0.21 correlation gap from a single weighting choice illustrates how 'the momentum factor' is not a unique object — construction methodology matters."

---

### Finding 6: S1 SML R-squared of 0.55 does not sufficiently address small-sample leverage

**Check:** Conservative Framing
**Severity:** Minor
**Section(s):** S1

**The expectations document said:**
"SML R-squared with only 15-20 stocks is very noisy — the key visual is the scatter's poor fit, not the exact R-squared." (S1 acceptance criteria)

**What happened:**
SML R-squared = 0.55 with 20 stocks. Observations: "High-beta stocks (TSLA, AMD, NVDA) cluster at high returns but with wide vertical spread." "NVDA is the highest-return point (~63%, beta ~1.72)."

**How the brief handled it:**
Brief S1: "Key numbers: ours = SML R-squared 0.55, slope 22.5%...20-stock sample with 3 stocks (NVDA, TSLA, AMD) contributing extreme leverage." Student takeaway: "With 20 stocks and one dominant sector (tech), small samples magnify period-specific effects."

**The problem:**
The brief mentions leverage from 3 stocks but does not explicitly connect this to the R-squared inflation. With only 20 data points, 3 extreme leverage points (NVDA at 43% alpha, TSLA at 25%, AMD at 22%) can mechanically inflate R-squared from near-zero to 0.55. The brief frames R-squared=0.55 as a "Real phenomenon" (sample period effect), but the expectations document itself warned this metric is "very noisy" with 15-20 stocks. A more conservative framing would distinguish the period effect (steep SML slope) from the small-sample artifact (high R-squared from leverage points).

**Suggested fix:**
Add to S1 teaching angle:
> "The R-squared of 0.55 is inflated by small-sample leverage: with only 20 stocks, three extreme points (NVDA, TSLA, AMD) dominate the regression. Remove those three stocks and the R-squared would drop substantially. The steep slope is a genuine period effect; the high R-squared is partly a small-sample artifact. The expectations document warned that 'SML R-squared with only 15-20 stocks is very noisy' — treat it as illustrative, not as evidence that CAPM works in this decade."

---

### Finding 7: S6 negative momentum factor return not discussed

**Check:** Expectations Regression
**Severity:** Minor
**Section(s):** S6

**The expectations document said:**
"Factor returns for momentum: monthly mean in [0.000, +0.010]. Momentum typically carries a positive premium even in cross-sectional regressions." (S6 acceptance criteria)

**What happened:**
Observations S6: "Momentum factor monthly mean = -0.001026." This is below the expected lower bound of 0.000.

**How the brief handled it:**
Brief S6 key numbers: "momentum factor monthly mean = -0.10%." The brief reports the number but does not note that it falls below the expected range or discuss why momentum carries a negative premium in cross-sectional regressions for this sample.

**The problem:**
The negative momentum premium is consistent with the broader theme of momentum weakness in this sample (Ex2: momentum NW t=0.81; Ex4: momentum t=0.84). The brief discusses momentum's weakness extensively in Ex2 and Ex4 but does not connect the S6 negative momentum factor return to this pattern. This is a minor gap — the number is reported, and the broader context exists elsewhere in the brief — but the S6 section misses an opportunity to reinforce the momentum-weakness theme.

**Suggested fix:**
Add to S6 teaching angle:
> "Note that the momentum factor return is slightly negative (-0.10% monthly), below the expected positive range. This is consistent with the broader weakness of momentum in our 2014-2024 sample, as we will see in Exercise 2 and Exercise 4."

---

### Finding 8: Reversal t-statistic of 36.58 needs stronger methodological caution

**Check:** Conservative Framing
**Severity:** Minor
**Section(s):** Ex4

**The expectations document said:**
No specific prediction for reversal's t-statistic magnitude. However, observations.md Notable Value #5 warns: "A t-statistic this large in a cross-sectional asset pricing test is unusual and may reflect a near-mechanical relationship (short-term reversal is measured from recent returns, and returns on the left-hand-side of the regression also include recent-period returns)."

**What happened:**
Reversal t = 36.58, an order of magnitude larger than any other characteristic.

**How the brief handled it:**
Brief Ex4: "The reversal t-statistic of 36.58 is suspiciously extreme and likely reflects a near-mechanical relationship: short-term reversal is computed from recent returns that partially overlap with the dependent variable."

**The problem:**
The brief flags the issue but then includes reversal as one of the two Bonferroni survivors and as evidence that the correction "works." If reversal's significance is near-mechanical (not a genuine predictive signal), then using it as a legitimate survivor overstates the correction's ability to identify real signals. The brief should be more explicit that reversal may be an artifact and that the only genuinely informative Bonferroni survivor is asset_growth. This matters for the "factor zoo" lesson: students should learn that surviving multiple testing correction is necessary but not sufficient — you also need to verify the signal is not mechanical.

**Suggested fix:**
Add to Ex4 student takeaway:
> "Surviving multiple testing correction is necessary but not sufficient. Reversal survives Bonferroni with t=36.58, but this likely reflects a mechanical overlap between the reversal signal (last month's return) and the dependent variable (this month's return, which includes mean reversion). In practice, you would investigate whether a surviving factor's signal is genuinely predictive or mechanically induced before declaring it a real discovery."

---

### Finding 9: Curriculum_state entry claims "death of value" was covered but uses term loosely

**Check:** Cross-Document Consistency
**Severity:** Minor
**Section(s):** Curriculum_state entry

**The expectations document said:**
S2 acceptance criteria: "HML should show positive long-run returns but with notable post-2007 weakness ('death of value')."

**What happened:**
Observations S2: "HML shows a notable decline from roughly 2007 onward, losing about half its cumulative value from peak." HML cumulative 2014-2024 = 0.77 (lost 23%).

**How the brief handled it:**
The brief's curriculum_state entry uses the phrase "'death of value' post-2007" in the Concepts Taught section. The S2 teaching angle mentions "HML cumulative return 2014-2024 = 0.77 (23% loss)" but does not use the phrase "death of value." The brief's narrative arc mentions it in passing.

**The problem:**
The curriculum_state entry lists "death of value" as a concept taught, but the brief's actual S2 teaching content presents HML's decline as a data point without the "death of value" label or its theoretical context. The term "death of value" specifically refers to the academic debate about whether the value premium has permanently disappeared (see Fama & French 2021 "The Value Premium" and related literature). If the concept is listed as "taught," the notebook should actually teach it — including the debate about whether it's a structural shift or a cyclical trough. Otherwise, the curriculum_state entry over-promises what students learn.

**Suggested fix:**
Either (a) add a sentence to S2 teaching angle explaining the "death of value" debate: "HML's 23% loss over 2014-2024 connects to the broader 'death of value' debate — whether the value premium has permanently disappeared due to crowding and arbitrage, or is cyclically depressed and will revert. This remains unresolved." Or (b) soften the curriculum_state entry from "'death of value' post-2007" to "HML post-2007 weakness."

---

### Finding 10: Brief does not flag that z-score std below 0.90 violates expectations for D2

**Check:** Cross-Document Consistency
**Severity:** Minor
**Section(s):** S7, D2

**The expectations document said:**
"Cross-sectional z-scores: within each month, each feature should have mean ~0 and std ~1...Verify round-trip: mean in [-0.05, +0.05], std in [0.90, 1.10] after standardization." (S7 acceptance criteria). D2 acceptance criteria: "Z-score mean per month [-0.05, +0.05] and std in [0.90, 1.10]."

**What happened:**
Observations: "z_std_range=[0.8432, 1.0000]." The lower bound of 0.84 is below the expected minimum of 0.90. Observations verdict: "marginal (lower bound 0.84 slightly below 0.90; the z-cap at +/-3 mechanically compresses std for heavy-tailed features)."

**How the brief handled it:**
Brief S7 key numbers: "Z-std range [0.84, 1.00]." Brief D2 key numbers: "Z-std range [0.84, 1.00]." Both report the numbers without noting they fall below the expected minimum.

**The problem:**
The z-score std of 0.84 is a known and expected consequence of the z-cap at +/-3 (the brief explains this well in the "What was wrong" section). However, the brief should note in the key numbers or teaching angle that the expectations document's std range of [0.90, 1.10] is technically violated and explain why this is acceptable. Without this note, the notebook agent may compare the numbers against expectations and be confused.

**Suggested fix:**
Add to S7 or D2 key numbers:
> "Z-std range [0.84, 1.00] — below the expectations document's [0.90, 1.10] range because the z-cap at +/-3 mechanically compresses standard deviation for heavy-tailed features. This is the expected cost of the two-stage outlier approach and is acceptable."

---

## Observations Divergence Checklist

Every item from observations.md Phase 2 Divergence Summary:

1. **SML is steep and positive (R-squared=0.55), not flat as CAPM literature predicts.** Addressed in brief S1. See Finding 6 for small-sample leverage understatement.
2. **HML replication far exceeds expectations (r=0.82 vs. expected max 0.60).** Addressed in brief S3, Ex1, D1 with extensive static-sort caveat. No issue.
3. **Fundamental-based factor return series span 131 months, not the expected 36-48.** Addressed in brief S3, Ex1, D1 via static-sort methodological notes. No issue.
4. **Beta is significantly priced (Ex2: t=2.46; D3: t=2.51), contradicting the "beta anomaly."** Addressed in brief Ex2, D3. No issue.
5. **Momentum is insignificant in cross-sectional tests (Ex2: t=0.81; Ex4: t=0.84).** Addressed in brief Ex2, Ex4. No issue.
6. **Diversified portfolio has higher factor risk share (89%) than concentrated tech portfolio (81%).** Addressed in brief Ex3. No issue.
7. **Max z-score of 7.32 — RESOLVED.** Addressed in brief S7, D2. No issue.
8. **Manual vs. linearmodels Fama-MacBeth gammas disagree by >0.001 for MKT and SMB.** Addressed in brief S5. No issue.
9. **Alpha shrinkage from FF3-to-FF5 is only 40%, below the 55% threshold.** Addressed in brief S4. No issue.
10. **pe_ratio-earnings_yield correlation positive — RESOLVED.** Addressed in brief S7, D2. No issue.

All 10 items addressed.

## Expectations Regression Checklist

### High-Likelihood Risks

| Risk | Status | Notes |
|------|--------|-------|
| Self-built SMB has very low correlation (~0.3-0.4) with Ken French SMB due to large-cap-only universe | Addressed | Extensively discussed in S3, Ex1, D1 |
| Self-built HML also has low correlation (~0.3-0.5) with Ken French HML | Addressed | Actual was higher (0.82); static-sort caveat provided |
| yfinance fundamental data covers only ~4-5 years (~2021-2025) | Addressed | Discussed via static-sort and 131-month analysis |
| Banks/financials missing Operating Income (~4% of tickers) | Partially addressed ~ | Mentioned in D1 (15 tickers Net Income fallback) but not discussed in S3, S7, or curriculum caveats |
| Panel data for Fama-MacBeth is unbalanced | Partially addressed ~ | Ex2 mentions 71 months and 179 stocks but does not discuss panel unbalancedness or NaN handling |
| Fama-MacBeth gamma estimates noisy with ~36-48 cross-sections | Partially addressed ~ | Brief discusses sample period effects but not the specific power concern from short window |

### Interactions

| Interaction | Status | Notes |
|-------------|--------|-------|
| Interaction 1: Large-cap universe x fundamental-based factor construction = structurally weak factors | Addressed | Core theme of S3, Ex1, D1 |
| Interaction 2: 200-ticker fundamental download x rate-limit exposure | Not addressed (N/A) | Not relevant to teaching narrative; acceptable omission |
| Interaction 3: Full price window vs. fundamental window = inconsistent date ranges | Partially addressed ~ | Brief notes 131 vs 48 months in various sections but does not explicitly warn about student confusion from inconsistent windows |
| Interaction 4: Panel data indexing for Fama-MacBeth | Not addressed (N/A) | Implementation detail; acceptable omission |

### Known Constraints

| Constraint | Status | Notes |
|------------|--------|-------|
| Survivorship bias (yfinance only has current listings) | Addressed | S1 Sandbox vs. reality |
| Universe bias (large-cap only, kills the size effect) | Addressed | Core theme throughout |
| Point-in-time violation | Silently dropped -- see Finding 1 | Never mentioned in brief |
| Fundamental data depth (~4-5 annual periods) | Addressed | Via static-sort discussion |
| Financial sector accounting differences | Partially addressed ~ | D1 mentions 15 tickers; not discussed in S3 or S7 |
| No delisted stock data | Addressed | Via survivorship bias discussion |
| getfactormodels returns PyArrow format | Not addressed (N/A) | Implementation detail; acceptable omission |

### Open Questions

| Question | Status | Notes |
|----------|--------|-------|
| OQ1: Will self-built factors show signal given short window + large-cap universe? | Addressed | Brief honestly states "not tested as originally framed" |
| OQ2: Will any factor carry significant Fama-MacBeth premium with ~36-48 cross-sections? | Partially addressed ~ — see Finding 4 | Resolved under 71-month conditions, not 36-48 as posed |
| OQ3: How will Barra R-squared compare monthly vs. daily? | Addressed | Brief notes daily not implemented |

## Sections with No Issues

The following sections passed all 6 checks with no findings:

- **Section 2: FF3 Model** — All acceptance criteria addressed, R-squared improvement and alpha shrinkage numbers consistent, cumulative factor plots described accurately, no methodology divergences, no conservative framing issues.
- **Section 4: FF5 + Momentum** — Alpha shrinkage below 55% honestly reported, momentum crash well-described, HML-CMA redundancy noted, diminishing returns clearly framed.
- **Section 5: Fama-MacBeth** — Manual vs. linearmodels disagreement disclosed, NW t-statistic increase explained, cross-sectional R-squared reported with production benchmark, sample-period effects noted.
- **Exercise 2: Factor Premia** — Beta significance and momentum insignificance honestly framed as sample-period effects, production benchmarks cited, Sandbox vs. reality provides appropriate context.
- **Deliverable 2: Feature Matrix** — Corrections thoroughly documented, two-stage outlier control well-explained, P/E pathology as teaching moment is well-framed.
- **Deliverable 3: Horse Race** — R-squared progression accurately reported, diminishing returns clearly demonstrated, residual alpha bridge to Week 4 well-constructed, production benchmarks cited.
