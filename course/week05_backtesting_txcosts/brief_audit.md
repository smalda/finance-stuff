# Brief Audit — Week 5: Backtesting, Research Discipline & Transaction Costs

## Summary

- **Findings:** 6 total (0 critical, 2 moderate, 4 minor)
- **Viability:** 13 Viable, 0 Reframe, 0 Redesign, 0 Remove
- **Overall assessment:** Adequate

The brief is honest about the week's major limitations — statistically insignificant IC, fragile DSR verdicts, and frequency-dependent purging effects. It correctly identifies the short OOS window (68 vs. expected 132 months) and the heavy-tail penalty on MinTRL as the dominant drivers of expectation miscalibrations. Two moderate issues exist: (1) the MinTRL values reported in S5 (174 months) and D3 (77.5 months) are not reconciled, which would confuse notebook agents and students; (2) the brief uses "genuine predictive power" language in S3 for models that do not clear t = 1.96, overstating the statistical evidence that PBO alone provides. Four minor issues involve undisclosed methodology deviations and inconsistent t-statistic reporting. No critical findings — no section would mislead students about the fundamental reliability of a result.

---

## Findings

### Finding 1: MinTRL inconsistency between S5 and D3

**Check:** Cross-Document Consistency
**Severity:** Moderate
**Section(s):** S5 (Deflated Sharpe Ratio), D3 (Responsible Backtest Report)

**The expectations document said:**
S5: "MinTRL in [6, 30] months at SR=0.7 with normal returns" (§Section 5 Acceptance Criteria). D3 L3: "MinTRL formula applied to the winning model's observed Sharpe (net, purged). MinTRL printed: `MINTRL_95pct={d} months`" (§Deliverable 3 Acceptance Criteria).

**What happened:**
S5 reports MinTRL = 174 months at M=10, 95% confidence, using gross monthly SR = 0.253 with skewness = -0.261 and excess kurtosis = 4.23 (from s5_deflated_sharpe.log). D3 reports MinTRL = 77.5 months at 95% confidence (from d3_responsible_report.log). The observations note this discrepancy at Cross-File Consistency point 7 and Notable Values points 6 and 11, stating D3 "uses different input parameters."

**How the brief handled it:**
S5 key numbers: "MinTRL = 174 months at M=10, 95% confidence." D3 key numbers: "MinTRL = 77.5 months, observed = 67 months." D3 teaching angle: "MinTRL of 77.5 months exceeds the observed 67-month track record by 10.5 months." The two values appear in separate sections with no cross-reference or reconciliation.

**The problem:**
A notebook agent writing S5 would present "174 months (14.5 years) to confirm this Sharpe." A notebook agent writing D3 would present "77.5 months (~6.5 years)." A student encountering both would see a 2.2x discrepancy in the same week with no explanation. The likely cause: S5 uses M=10 (the trial penalty inflates MinTRL), while D3 appears to use M=1 or a different M value — but the brief does not specify D3's M parameter. Additionally, S5 uses gross monthly SR (0.253) while D3 uses net monthly SR (derived from annualized 0.676 ≈ monthly 0.195), which should produce a *higher* MinTRL in D3, not lower — reinforcing that the trial parameter must differ. Without disclosure, the numbers appear contradictory.

**Suggested fix:**
In D3's key numbers, add the M parameter: "MinTRL = 77.5 months at M=1, 95% confidence (no trial penalty); at M=10, MinTRL = 174 months (from S5, using gross returns)." In D3's teaching angle, add: "The MinTRL of 77.5 months assumes this is the only strategy evaluated (M=1). Under the M=10 trial penalty applied in Section 5, the requirement extends to 174 months — reinforcing that honest accounting of the search process is not just an academic exercise but a 2x multiplier on the required track record."

---

### Finding 2: "Genuine predictive power" language for statistically insignificant signals

**Check:** Signal Integrity / Conservative Framing
**Severity:** Moderate
**Section(s):** S3 (CPCV & Multiple Testing)

**The expectations document said:**
S3 Signal viability: "Low — same 132-month series as Section 2. CPCV does not change the signal; it changes the evaluation framework. Interpret PBO with awareness that low n_paths (15) means each PBO estimate has high variance." (§Section 3 Signal Viability). The rigor.md §2.4 mandates that when IC t < 1.96, the signal should be characterized as weak.

**What happened:**
Observations Signal Significance table: GBM t = 1.387 (p = 0.170), NN t = 1.027 (p = 0.308), Ridge t = -0.058 (p = 0.954). None of the three models clear t = 1.96. PBO = 0.267 is computed from only 15 CPCV paths across 3 models.

**How the brief handled it:**
S3 teaching angle: "PBO = 0.267 means the in-sample winner ranks above the OOS median in 73% of partitions — a signal of genuine, if weak, predictive power." S3 key numbers: "PBO = 0.267 (ours), production benchmark for genuine alpha: 0.25-0.35." The Open Questions section describes PBO = 0.267 as "in the 'genuine alpha' range." The brief correctly reports all three t-statistics and notes none clear t = 1.96 or t = 3.0.

**The problem:**
"Genuine predictive power" is too strong for models that fail t = 1.96. PBO and IC t-statistics measure different things: PBO measures whether the IS winner maintains its rank OOS, while the IC t-stat measures whether the signal is distinguishable from zero. A model can maintain its rank (low PBO) while producing a signal indistinguishable from noise (low t-stat) — this simply means it's consistently the "best of three weak models." The brief reports the t-stats correctly but then frames the PBO result using language that implies the signal is real when the statistical evidence does not support that conclusion. Additionally, with only 15 paths and 3 models, the PBO estimate has high sampling variance (as the expectations explicitly note), making the point estimate of 0.267 unreliable. The "production benchmark for genuine alpha: 0.25-0.35" framing further anchors the reader to interpret the PBO as confirming real signal.

**Suggested fix:**
Replace "a signal of genuine, if weak, predictive power" with: "a positive PBO result, indicating the in-sample winner tends to maintain relative ranking out-of-sample — though with only 15 paths and 3 models, this estimate has high sampling variance. Importantly, PBO < 0.5 does not confirm statistical significance: none of the three models clears t = 1.96, so the absolute signal remains indistinguishable from zero at conventional confidence levels. The PBO tells us the GBM model is persistently the least-bad of the three, not that its signal is statistically real."

Replace the key numbers citation "production benchmark for genuine alpha: 0.25-0.35" with: "production benchmark: genuine alpha 0.25-0.35, noise ~0.50 (Bailey et al., 2015) — our 0.267 falls in the 'genuine alpha' sub-range, but with 15 paths and IC t < 1.96, the PBO alone does not confirm the signal's statistical reality."

---

### Finding 3: S4 fixed spread regime deviation not disclosed

**Check:** Methodology Transparency
**Severity:** Minor
**Section(s):** S4 (Transaction Cost Decomposition)

**The expectations document said:**
"Three cost regimes: (a) zero cost, (b) fixed 10 bps one-way half-spread for all stocks, (c) market-cap-tiered spread" (§Section 4 Data). The acceptance criterion: "Spread cost (regime b, fixed 10 bps): monthly spread drag = turnover × 2 × 0.001. At 50% one-way turnover: drag ≈ 0.10% per month = 1.2% per year." (§Section 4 Acceptance Criteria).

**What happened:**
Execution log deviation #3: "Spread levels changed from plan's 10 bps / 5-15-25 bps to 5 bps / 10-20-30 bps to ensure correct equity curve ordering." The code uses 5 bps as the fixed regime, not 10 bps.

**How the brief handled it:**
S4 teaching angle: "even a modest 5 bps half-spread costs 1.68% per year in spread alone." S4 key numbers: "Annual spread cost = 1.68% at 5 bps." The brief reports the 5 bps value without noting it differs from the expectations' 10 bps specification.

**The problem:**
The brief presents the 5 bps fixed spread as if it were the plan. At the expectations' specified 10 bps, the spread drag would be approximately double (~3.36%/year at 139.9% turnover), making the TC teaching point even more dramatic. The deviation is pedagogically defensible (ensures equity curve ordering), but should be disclosed so the notebook agent knows the fixed regime is at the low end of the expectations' range.

**Suggested fix:**
Add to S4 teaching angle: "The fixed spread regime uses 5 bps (reduced from the expectations' 10 bps to ensure correct three-regime equity curve ordering). At the expectations' original 10 bps, spread drag would approximately double to ~3.4%/year — reinforcing that even small spread assumptions are material at this turnover level."

---

### Finding 4: S5 fixed-SR surface methodology not disclosed

**Check:** Methodology Transparency
**Severity:** Minor
**Section(s):** S5 (Deflated Sharpe Ratio)

**The expectations document said:**
"The DSR surface uses rolling sub-windows of 24, 36, 48, 60, 84, and 120 months (trailing windows ending at the most recent period)" (§Section 5 Data). The probe computed DSR using sub-window Sharpe values.

**What happened:**
Execution log deviation #4: "Uses fixed full-series monthly SR (0.253) with varying n_obs rather than sub-window Sharpe. Sub-window Sharpe broke monotone-in-T assertions because recent favorable returns produced higher Sharpe at short windows."

**How the brief handled it:**
S5 teaching angle describes the surface as "DSR surface spanning [0.159, 1.000]" and mentions that SR is "held constant" ("DSR surface computed over T=[24,36,48,60,84,120] x M=[1,5,10,20,50], holding monthly SR constant at 0.253"). The "holding SR constant" phrasing is present in the Actual field but not explained in the teaching angle or key numbers.

**The problem:**
Holding SR constant while varying T isolates the effect of track record length and trial count — pedagogically clean. But it differs from the expectations' design of using trailing sub-windows (which would let SR vary with T). The brief should disclose that this is a deliberate simplification and why: sub-window SR broke monotonicity because recent periods had higher returns. A student or notebook agent should understand that the surface shows the *statistical* effect of track record length, not the *empirical* effect (which would also capture regime variation).

**Suggested fix:**
Add to S5 teaching angle: "The surface holds the monthly SR constant at 0.253 (the full-period estimate) rather than recomputing SR per sub-window. This isolates the pure statistical effect of track record length and trial count. Using sub-window SR would conflate the statistical power effect with regime-dependent SR variation — in our data, recent favorable returns produced higher SR at short windows, which would break the monotone-in-T property that the surface is designed to teach."

---

### Finding 5: IC t-statistics not consistently in key numbers

**Check:** Signal Integrity
**Severity:** Minor
**Section(s):** S2 (Purged Cross-Validation), Ex2 (Purging Comparison), D1 (Purged CV Engine)

**The expectations document said:**
Rigor.md §2.4: "For any rolling IC or OOS R²: report t-statistic and p-value alongside the point estimate." §2.6: "Every model comparison must test statistical significance."

**What happened:**
Observations Signal Significance table reports t-stats only for S3 and D3 (the sections that explicitly test multiple models). S2, Ex2, and D1 report IC values with the delta t-stat (t = -0.030 for the WF-vs-purged difference) but do not report the t-stat for the absolute IC level (WF IC = 0.0203 or purged IC = 0.0213).

**How the brief handled it:**
S2 key numbers: "WF IC = 0.0203, purged IC = 0.0213, delta = -0.0011 (not statistically significant, t = -0.030)." S3 key numbers include model-level t-stats. D3 includes t-stats. S2, Ex2, and D1 do not include IC-level t-stats in key numbers.

**The problem:**
Rigor.md requires t-statistics alongside IC point estimates. S2 reports "IC = 0.0203" without its own t-stat, only reporting the t-stat for the delta. While the expectations explicitly anticipated that IC t-stats would be below 1.96 (and S2 notes "Signal viability confirmed low"), the brief's key numbers for S2, Ex2, and D1 do not follow the rigor.md format of "IC = X (t=Y, p=Z, n=N)." A notebook agent following the key numbers would reproduce the IC omission.

**Suggested fix:**
For S2, Ex2, and D1 key numbers, add the absolute IC t-statistics from the S3 or D3 signal significance analysis. For example, S2 key numbers should read: "WF IC = 0.0203 (t ≈ 1.4, p ≈ 0.17 per S3 GBM analysis, n = 68 months — statistically insignificant), purged IC = 0.0213, delta = -0.0011 (t = -0.030)." This maintains rigor.md compliance without changing the teaching angle.

---

### Finding 6: Survivorship × DSR interaction not quantified

**Check:** Expectations Regression
**Severity:** Minor
**Section(s):** S5 (Deflated Sharpe Ratio), S6 (Responsible Backtest)

**The expectations document said:**
Interaction 5: "The survivorship-biased universe inflates gross returns, which inflates the observed Sharpe, which inflates the DSR... The code agent must note this in structured output: `Universe: current S&P 500 — survivorship inflates Sharpe by approximately 0.1–0.3 (annual) before DSR correction.`" (§Interaction Analysis).

**What happened:**
The data_setup output includes a survivorship bias note. S1 quantifies the survivorship premium at 3.07% (annualized return inflation). S5 and S6 use the gross Sharpe (0.87-0.88) that includes this inflation. The observations confirm survivorship inflates gross returns (Known Constraints Manifested point 1).

**How the brief handled it:**
The narrative arc mentions "survivorship-biased S&P 500 universe." S1 reports the premium. S5 and S6 use the gross Sharpe without noting survivorship inflation's effect on the DSR surface or the responsible-vs-naive gap. The "Sandbox vs. reality" for S5 does not mention survivorship.

**The problem:**
The expectations explicitly flagged that survivorship inflates Sharpe by 0.1-0.3 annual units, which would shift the DSR surface. If the true (survivorship-corrected) Sharpe is 0.1-0.3 units lower, the DSR surface would be more pessimistic and the MinTRL longer. The brief does not quantify this interaction — the DSR results are presented on the survivorship-inflated Sharpe without noting the survivorship contribution. This is partially addressed by the general survivorship disclosure in the narrative arc, but not specifically in the DSR sections where it matters most.

**Suggested fix:**
Add to S5's "Sandbox vs. reality" or teaching angle: "The gross Sharpe of 0.88 includes survivorship inflation estimated at 0.1-0.3 units annually (per S1 and expectations Interaction 5). If the true Sharpe is 0.1-0.3 units lower, the DSR surface shifts substantially toward more pessimistic values and the MinTRL would increase further. The DSR results presented here are an upper bound on the strategy's statistical significance." Similarly, S6's "Sandbox vs. reality" could note: "The 0.302 Sharpe gap between naive and responsible evaluation understates the full gap because the responsible evaluation does not correct for survivorship inflation of the gross Sharpe."

---

## Viability Assessment

| Section | Verdict |
|---------|---------|
| S1: The Seven Sins | Viable |
| S2: Purged Cross-Validation | Viable |
| S3: CPCV & Multiple Testing | Viable |
| S4: Transaction Cost Decomposition | Viable |
| S5: The Deflated Sharpe Ratio | Viable |
| S6: The Responsible Backtest | Viable |
| Ex1: The Look-Ahead Bug Hunt | Viable |
| Ex2: Purging vs. Walk-Forward | Viable |
| Ex3: TC Sensitivity Analysis | Viable |
| Ex4: DSR Calibration | Viable |
| D1: Build a Purged CV Engine | Viable |
| D2: Transaction Cost Accounting Pipeline | Viable |
| D3: The Responsible Backtest Report | Viable |

All 13 sections are Viable. After applying the suggested finding fixes, every section teaches what the blueprint intended. The purging sections (S2, Ex2, D1) have shifted emphasis from "purging dramatically changes IC" to "purging's effect is frequency-dependent and near-zero on monthly data with 1-month labels" — but this reframing is already captured in the brief and is a valid teaching point. The DSR sections (S5, Ex4) produce more dramatic results than expected due to heavy tails — this strengthens rather than weakens the teaching point. The TC sections (S4, Ex3, D2) show higher turnover than expected, which amplifies the cost teaching point.

---

## Observations Divergence Checklist

| # | Divergence Item | Status |
|---|----------------|--------|
| 1 | OOS period length (68 months vs. expected 132 months) | Addressed in brief ✓ — mentioned in narrative arc ("68 OOS months"), S5 Actual, S6 Actual, D3 Actual, and Open Questions section |
| 2 | Purging effect near zero (IC delta = -0.0011 vs. expected 0.005-0.030) | Addressed in brief ✓ — thoroughly explained in S2, Ex2, D1 teaching angles as structural frequency-dependence |
| 3 | Turnover substantially higher than expected (139.9% vs. expected 30-100%) | Addressed in brief ✓ — S4 and D2 frame this as the teaching point; narrative arc revised to emphasize TC dominance |
| 4 | MinTRL dramatically higher than expected (174 months vs. expected 6-30 months) | Addressed in brief ✓ — S5 explains excess kurtosis penalty; but see Finding 1 re: D3 inconsistency |
| 5 | DSR at full OOS below expected (0.41-0.68 at M=10 vs. expected >= 0.95) | Addressed in brief ✓ — S6 reports NO-DEPLOY, D3 reports marginal DEPLOY, fragility explicitly flagged |
| 6 | Model count: 3 variants instead of 4 | Addressed in brief ✓ — Open Question 2 explains Ridge substitution for OLS |
| 7 | Signal B OOS IC marginally below threshold (0.0173 vs. expected >= 0.02) | Addressed in brief ✓ — Ex1 Actual notes the gap is 0.0027, within noise |

---

## Expectations Regression Checklist

### High-Likelihood Risks

| Risk | Status |
|------|--------|
| Survivorship bias inflates returns by 1-4% annually | Addressed ✓ — S1 quantifies at 3.07%, narrative arc notes bias |
| Week 4 cache dependency (Medium likelihood) | Addressed ✓ — OQ2 confirms cache available, synthetic fallback not needed |
| DSR demo pedagogically flat if full OOS used (High — certain) | Addressed ✓ — S5 uses sub-windows with fixed SR; Ex4 uses net SR |
| HIGH TURNOVER warning expected (Medium-High) | Addressed ✓ — S4, D2 report 139.9% with appropriate framing |
| Monthly IC statistical power weak — t < 1.96 expected | Addressed ✓ — S2 notes "signal viability low," S3 reports 0/3 pass t=1.96 |
| Corwin-Schultz negative values (Medium) | Addressed ✓ — code used fixed-rate proxy instead; CS was optional path |
| Non-stationarity and regime shifts across sub-periods | Addressed ✓ — S6 reports 27% IC decline across sub-periods |
| Heavy tails inflate MinTRL and depress DSR | Addressed ✓ — S5 explains kurtosis penalty; D3 notes fragility |
| quantstats API dependency (Low) | Addressed ✓ — D3 confirms quantstats installed and producing tearsheet |
| No TAQ data for spread estimation | Addressed ✓ — S4 uses fixed-rate proxy, notes TAQ in sandbox comparison |
| No intraday data for execution modeling | Partially addressed ~ — S4 sandbox section mentions institutional scale but doesn't explicitly note the no-intraday limitation |

### Interactions

| Interaction | Status |
|-------------|--------|
| 1: Week 4 cache × S1 independence | Addressed ✓ — S1 uses fresh signals independent of Week 4 |
| 2: Monthly frequency × CPCV structure | Addressed ✓ — S3 reports 15 paths at k=6 |
| 3: Daily OHLCV × TC model | Addressed ✓ — S4 and D2 use daily OHLCV for spread/impact |
| 4: DSR surface × OOS window length | Addressed ✓ — S5 uses sub-windows per design |
| 5: Survivorship bias × DSR demonstration | Partially addressed ~ — survivorship noted in narrative arc but not quantified in DSR sections — see Finding 6 |
| 6: TC sensitivity grid × turnover | Addressed ✓ — Ex3 reports feasibility frontier with 140% turnover |

### Open Questions

| Open Question | Status |
|---------------|--------|
| OQ1: Whether Week 4 long-short portfolio gross Sharpe above/below 0.5 | Addressed ✓ — resolved: gross Sharpe = 0.876 over 68 OOS months; OOS start is 2019-04 not 2015 noted |
| OQ2: Whether all three Week 4 model variants available | Addressed ✓ — resolved: GBM, NN from cache; Ridge synthesized inline |

---

## Low-Confidence Section Review

**D3: The Responsible Backtest Report — Categorization justified ✓**

The brief marks D3 as `Confidence: low` because "The DSR@M=10 = 0.504 barely clears the 0.50 DEPLOY threshold. A slightly different cost assumption, data sample, or SR convention would flip the verdict to NO-DEPLOY. The S6 version of DSR reports 0.414 — firmly NO-DEPLOY."

Assessment: This is the correct use of the low-confidence flag. The DSR of 0.504 is within rounding error of the 0.50 threshold, and the S6 computation (using a different but equally defensible cost specification) produces 0.414 — a NO-DEPLOY verdict. The brief explicitly states the fragility in both the Confidence note and the teaching angle. The categorization of "Matches" is technically correct (all acceptance criteria pass), but the DEPLOY verdict should not be presented as definitive. The existing framing ("razor-thin pass," "fragility of this verdict is itself pedagogically interesting") is adequate and honest.

No additional action needed beyond Finding 1's suggested reconciliation of the MinTRL discrepancy, which would further reinforce the fragility message.

---

## Sections with No Issues

The following sections passed all 7 checks with no findings:

- **S1: The Seven Sins** — Look-ahead bug demonstration (IC = 1.0 vs -0.013) and survivorship simulation (3.07% premium) are clean, dramatic, and honestly framed. The simulated delisting approach (replacing the plan's natural survivorship comparison because all 449 tickers have 100% coverage) is disclosed in the Actual field.

- **S6: The Responsible Backtest** — The 0.302 Sharpe gap (34% reduction) and NO-DEPLOY DSR verdict (0.414) are the central quantitative results. Both are honestly presented with production benchmarks. The sub-period IC split (0.054 → 0.039, 27% decline) is below the SIGNAL DECAY threshold and correctly reported.

- **Ex1: The Look-Ahead Bug Hunt** — Three-signal fingerprint works as designed. Signal A ratio = 64.4x, Signal B ratio = 1.8x, Signal C ratio = 1.6x. The reframing of Signal B as "model-universe coupling" is disclosed. Signal B OOS IC marginally below threshold (0.0173 vs 0.02) but the diagnostic pattern remains unambiguous.

- **Ex3: Transaction Cost Sensitivity Analysis** — 9x6 grid with clear feasibility frontier. Breakeven spread = 15 bps at 0% turnover reduction. 46/54 cells feasible. All acceptance criteria pass.

- **Ex4: DSR Calibration** — Dramatic surface from 0.045 to 1.000 using net SR. Crossover at M=5 for T <= 24 months. Non-normality (excess kurtosis = 4.23) correctly incorporated.

- **D1: Build a Purged CV Engine** — Non-negotiable correctness test passes (zero leaking samples). IC direction reversal explained as structural. Visual diagnostic uses weekly frequency to make purge/embargo zones visible.

- **D2: Transaction Cost Accounting Pipeline** — Correctness test passes (synthetic 2-asset portfolio). Three-regime ordering maintained. HIGH TURNOVER warning appropriately prominent. Cost decomposition identifies March 2020 as highest-cost month (impact-dominated).
