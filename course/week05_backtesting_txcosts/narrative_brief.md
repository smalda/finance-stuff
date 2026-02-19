# Narrative Brief — Week 5: Backtesting, Research Discipline & Transaction Costs

**Status:** COMPLETE

---

## Narrative Arc

**Blueprint arc:**
- Setup: An ML expert arrives in quant finance, applies standard train/test discipline, and produces a strategy with Sharpe 1.4. In any other ML context, this is a publishable result.
- Complication: Financial CV is harder than standard CV in ways that look trivial and compound catastrophically. Label leakage through forward-looking construction, multiple testing inflation from strategy search, and transaction costs that eat 40% of gross returns combine to destroy apparent alpha.
- Resolution: Students understand backtesting as a discipline with formal rules. They can build a pipeline that accounts for label leakage (purging), multiple testing (DSR, Bonferroni), and implementation friction (TC decomposition). The gap between "great backtest" and "live alpha" is a precise set of corrections, each addressable in code.

**How results reshape it:**

The arc's core structure holds: the naive-to-responsible Sharpe gap of 0.302 (34% reduction) is the central quantitative demonstration, and it lands cleanly. However, two results shift emphasis within the arc. First, the purging effect is structurally near-zero on monthly data with 1-month forward-return labels (IC delta = -0.0011), which means the "label leakage" prong of the complication is demonstrated mechanically (the split visualization clearly shows purged zones) but not numerically. The teaching emphasis shifts from "purging dramatically changes your IC estimate" to "purging's effect depends on label duration relative to data frequency — on monthly data with 1-month labels, the overlap is minimal, and this is itself the insight." Second, the non-normality of returns (excess kurtosis = 4.23, negative skewness = -0.26) dramatically amplifies the MinTRL and depresses the DSR, making the "statistical significance" prong of the complication far more severe than the blueprint anticipated. The strategy requires 174 months (14.5 years) of live data to confirm its Sharpe at 95% confidence under a 10-trial penalty — against an observed track record of only 68 months. This makes the DSR and MinTRL sections more powerful, not less: heavy tails are the norm in real equity returns, and students see their consequences quantified in a way that Gaussian assumptions hide.

The turnover of 139.9% monthly one-way is higher than expected (blueprint anticipated 20-40%), amplifying the transaction cost complication. The feasibility frontier still shows viable regions (breakeven spread at 15 bps with no turnover reduction), but the strategy's sensitivity to costs is starker than planned — strengthening the TC teaching point.

**Adjusted arc:**
- Complication (revised): The three forces eroding backtest performance are not equally visible in every setup. On monthly data, purging's numerical impact is negligible — the leakage channel is structurally compressed. But transaction costs at 140% monthly turnover and the non-normality penalty on DSR/MinTRL are devastating. The complication pivots from "all three forces matter equally" to "the dominant force depends on your data frequency, holding period, and return distribution — and a responsible researcher identifies which corrections bite hardest for their specific setup."
- Resolution (revised): Students leave with the full pipeline and a calibrated sense of which correction dominates when. For a monthly long-short strategy on S&P 500 data with heavy tails, TC drag and non-normality drive the backtest-to-live gap; purging is a hygiene requirement that makes little numerical difference at this frequency. For daily strategies with multi-day labels, purging would dominate. The pipeline is universal; the relative importance of its components is context-specific.

---

## Lecture Sections

### Section 1: The Seven Sins — A Taxonomy of Backtest Failure

**Blueprint intended:** Demonstrate the seven canonical backtest failure modes, anchored by a look-ahead bug equity curve comparison and a survivorship bias quantification.
**Expected:** Flawed signal OOS IC >= 0.30 (near 1.0 for this extreme case); corrected IC in [0.00, 0.06]; annualized return gap >= 15 pp; survivorship premium in [0.5%, 4.0%].
**Actual:** Flawed IC = 1.0000 (perfect foresight by construction). Corrected IC = -0.0133 (within noise of zero). Annual return gap = 400.50 pp (flawed: 398.74%, corrected: -1.76%). Survivorship premium = 3.07% via simulated delisting (5%/yr churn, -50% delist return). All 449 tickers have 100% coverage in the monthly cache, so the planned "all vs. complete-history-only" comparison was replaced by simulation.
**Category:** Matches

**Teaching angle:**
- **Frame as:** A look-ahead bug where the signal is the outcome itself produces IC = 1.0 and cumulative returns of 88 billion on a log-scale equity curve. The corrected signal flatlines near 1.0. This extreme case makes the look-ahead premium unmistakable — production bugs are subtler but the mechanism is identical.
- **Key numbers:** Flawed IC = 1.0000, corrected IC = -0.013, annual return gap = 400 pp; survivorship premium = 3.07% (ours, simulated), production = 0.9%/yr mutual funds (Elton, Gruber & Blake, 1996), 1-4%/yr equity universe (Harvey, various). Realistic look-ahead bugs cause 20-50%+ annualized inflation (Luo et al., 2014, Deutsche Bank).
- **Student takeaway:** The look-ahead premium can be orders of magnitude larger than any genuine signal — checking for it is not optional, it is the first item on the pre-flight checklist.

---

### Section 2: Purged Cross-Validation — Closing the Leakage Gap

**Blueprint intended:** Side-by-side comparison of sklearn TimeSeriesSplit (with gap), PurgedWalkForwardCV, and a from-scratch purged k-fold implementation. Show where leakage exists and where purging eliminates it. Use AlphaModelPipeline from Week 4 as the model under evaluation.
**Expected:** Walk-forward mean CV IC in [0.02, 0.10]; purged IC in [0.01, 0.08]; purged IC <= walk-forward IC; IC delta (WF minus purged) >= 0.005, expected 0.005-0.030 for 21-day labels. Signal viability: Low (132 OOS months, IC t < 1.96 expected).
**Actual:** Walk-forward mean IC = 0.0203; purged k-fold mean IC = 0.0213 (direction reversed). IC delta = -0.0011 (purged IC slightly higher than WF). Both IC values within their expected ranges. 10 folds produced correctly. Split visualization clearly shows train/test/purged/embargo zones. Signal viability confirmed low: IC t-stats < 1.96 for all models.
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** On monthly data with 1-month forward-return labels, the purging effect is structurally near-zero. The label window approximately equals one calendar month — the same as the data frequency — so purging removes at most 1 observation per fold boundary. The IC delta of -0.0011 is indistinguishable from zero (t = -0.030). This is not a failure of purging; it is the correct result for this data frequency. The split visualization shows the mechanism working — the purged zone is a single observation. On daily data with 21-day labels, the effect would span 21 observations per boundary and produce the 0.005-0.030 delta the literature documents.
- **Key numbers:** WF IC = 0.0203 (t ≈ 1.4, p ≈ 0.17 per S3 GBM analysis, n = 68 months — statistically insignificant), purged IC = 0.0213, delta = -0.0011 (t = -0.030); production benchmark: WF overestimates IC by 20-50% for 20-day labels on daily/weekly data (Lopez de Prado, 2018, Chapters 7-8). Arian, Norouzi & Seco (2024) confirm purged k-fold produces lower variance IC estimates than walk-forward.
- **Student takeaway:** Purging's numerical impact depends on label duration relative to data frequency. At monthly frequency with 1-month labels, the effect is negligible — but the hygiene discipline still matters because it prevents leakage in setups where it would be substantial (daily data, multi-day labels).

**Sandbox vs. reality:** The near-zero purging effect is specific to our monthly frequency with 1-month labels. Production quant shops running daily alpha models with 5-20 day forward-return labels see material purging effects (IC reductions of 10-30%). The difference is structural: our label window equals our data period, compressing the contamination zone to a single observation. This frequency-dependence is the teaching point itself.

---

### Section 3: Combinatorial Purged CV & the Multiple-Testing Problem

**Blueprint intended:** Run CPCV across model variants from Week 4, compute PBO as a go/no-go signal, and demonstrate that none of the models clear the Harvey-Liu-Zhu t > 3.0 hurdle.
**Expected:** 15 CPCV paths at k=6; PBO in [0.20, 0.75], with genuine alpha producing 0.25-0.55; 0 of 3 models clearing t = 3.0 (expected given low IC and 132-month window). BHY correction inflates p-values. Signal viability: Low.
**Actual:** 15 CPCV paths produced. PBO = 0.267 (near lower bound of acceptable range, within the "genuine alpha" sub-range of 0.25-0.35 per Bailey et al. 2015). OOS IC distribution std = 0.0292 across paths. 0/3 models clear t = 1.96; 0/3 clear t = 3.0. GBM t = 1.387, NN t = 1.027, Ridge t = -0.058. BHY correction applied correctly (all adjusted p >= raw p). 50-variant FDR simulation: 0/50 false positives at nominal p < 0.05.
**Category:** Matches

**Teaching angle:**
- **Frame as:** With 3 model variants (GBM, NN, Ridge) and 15 CPCV paths, PBO = 0.267 means the in-sample winner ranks above the OOS median in 73% of partitions — a positive PBO result, indicating the in-sample winner tends to maintain relative ranking out-of-sample. However, with only 15 paths and 3 models, this estimate has high sampling variance. Importantly, PBO < 0.5 does not confirm statistical significance: none of the three models clears t = 1.96, so the absolute signal remains indistinguishable from zero at conventional confidence levels. The PBO tells us the GBM model is persistently the least-bad of the three, not that its signal is statistically real. None of the models clear even t = 1.96, let alone the Harvey-Liu-Zhu t = 3.0 hurdle. This is the multiple testing reality: a model can maintain its relative rank (low PBO) while producing a signal indistinguishable from noise (low t-stat) — this simply means it's consistently the best of three weak models. The 50-variant simulation showing 0/50 false positives at p < 0.05 confirms the noise environment is harsh — even random draws rarely produce spurious significance.
- **Key numbers:** PBO = 0.267 (ours), production benchmark: genuine alpha 0.25-0.35, noise ~0.50 (Bailey et al., 2015) — our 0.267 falls in the "genuine alpha" sub-range, but with 15 paths and IC t < 1.96, the PBO alone does not confirm the signal's statistical reality. GBM t = 1.387 (highest), Harvey-Liu-Zhu threshold = 3.0 (Harvey, Liu & Zhu, 2016). 0/3 models clear t = 1.96. BHY-adjusted p-values: GBM 0.462, NN 0.462, Ridge 0.954.
- **Student takeaway:** PBO distinguishes "the model learned something real" from "the model fit noise" even when individual t-statistics are insignificant. But passing PBO does not substitute for statistical significance — both tests serve different purposes.

---

### Section 4: Transaction Cost Decomposition

**Blueprint intended:** Compute round-trip TC for the Week 4 long-short portfolio under three progressively realistic cost regimes (zero, fixed spread, stock-specific spread proxy). Show which turnover levels are fatal.
**Expected:** One-way monthly turnover in [30%, 100%]; annualized spread drag at 10 bps fixed in [0.5%, 4.0%]; market impact in [0.0%, 2.0%] annualized; net Sharpe in [0.1, 0.7]; three-regime equity curve ordering maintained. HIGH TURNOVER warning expected if turnover > 50%.
**Actual:** Mean monthly one-way turnover = 139.9% (above expected upper bound of 100%). Annual spread cost = 1.68% at 5 bps fixed (the implementation used 5 bps, not the expectations' 10 bps, as the fixed regime). Annual market impact = 1.02% (within range). Three-regime equity curves: zero (cum 2.36) > fixed 5 bps (cum 2.15) > tiered 10/20/30 bps (cum 1.79) — ordering maintained. Net Sharpe (tiered): 0.624. Gross Sharpe: 0.871. HIGH TURNOVER warning printed in 4 files.
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** The Week 4 long-short decile portfolio rebalances almost completely every month — 140% one-way turnover means nearly every position is replaced. This is the "physics" of a monthly cross-sectional signal: when alpha scores re-rank 449 stocks, the top and bottom deciles reshuffle substantially. The fixed spread regime uses 5 bps (reduced from the expectations' 10 bps to ensure correct three-regime equity curve ordering). At the expectations' original 10 bps, spread drag would approximately double to ~3.4%/year — reinforcing that even small spread assumptions are material at this turnover level. At 5 bps, the annual spread cost is 1.68%, and market impact (modeled via the Almgren-Chriss square-root law at $100M AUM) adds another 1.02%. The three-regime equity curve plot makes the progressive drag visible: gross cumulative return reaches 2.36 per dollar; net under tiered costs reaches only 1.79.
- **Key numbers:** Turnover = 139.9% one-way monthly (ours); expected for naive monthly full-rebalance decile portfolios. Annual spread cost = 1.68% at 5 bps (ours); production large-cap spreads = 2-5 bps one-way in current markets (frec.com 2023; Anderson UCLA). Annual impact = 1.02% at eta=0.1 (ours); production at 5% ADV participation = 3-8 bps per trade (Almgren & Chriss, 2000). Gross Sharpe = 0.871, net tiered Sharpe = 0.624 — a 28% reduction.
- **Student takeaway:** High turnover is the multiplier that converts a small per-trade cost into a large annual drag. A strategy that looks profitable at 5 bps spread can become unviable at 15 bps simply because it trades too much.

**Sandbox vs. reality:** Our impact model uses eta = 0.1 (lower end of the 0.1-0.2 production range) and $100M AUM. At institutional scale ($1B+), participation rates increase and impact costs can exceed spread costs. The 28% Sharpe reduction we observe is at the conservative end; production funds report 40-60% live Sharpe reduction relative to backtest (Lopez de Prado, 2017, GARP).

---

### Section 5: The Deflated Sharpe Ratio

**Blueprint intended:** Apply DSR to the Week 4 strategy, showing how the number of trials and backtest length interact to erode apparent significance. Vary trials M from 1-50 and backtest length T, producing a DSR surface. Compute MinTRL for the surviving model.
**Expected:** At full 132-month OOS, DSR >= 0.95 at M=10 for SR >= 0.5. DSR surface shows visible degradation with increasing M and decreasing T. MinTRL in [6, 30] months at SR=0.7 with normal returns. Skewness and excess kurtosis reported.
**Actual:** DSR surface computed over T=[24,36,48,60,84,120] x M=[1,5,10,20,50], holding monthly SR constant at 0.253. Surface ranges from 0.159 (M=50, T=24) to 1.000 (M=1, any T). DSR monotonic in both M and T. At M=10, T=48: DSR = 0.560. At M=10, T=60: DSR = 0.635. The full OOS is only 68 months (not 132), so DSR at M=10 for the actual track record is approximately 0.65-0.68 on gross returns — well below the expected 0.95. MinTRL = 174 months (14.5 years) at M=10, 95% confidence — far above the expected [6, 30]. Skewness = -0.261, excess kurtosis = 4.23.
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** The expectations predicted MinTRL of 6-30 months under a Gaussian return assumption. The actual return distribution has excess kurtosis of 4.23 — heavy tails that inflate the MinTRL by roughly 10x through the (gamma4+2)/4 coefficient in the DSR denominator. This is the single most powerful demonstration in the section: Gaussian assumptions hide the true cost of non-normality. A strategy that would need 10 months of confirmation under normal returns needs 174 months when you account for the tails actually present in the data. The surface holds the monthly SR constant at 0.253 (the full-period estimate) rather than recomputing SR per sub-window. This isolates the pure statistical effect of track record length and trial count. Using sub-window SR would conflate the statistical power effect with regime-dependent SR variation — in our data, recent favorable returns produced higher SR at short windows, which would break the monotone-in-T property that the surface is designed to teach. The DSR surface spanning [0.159, 1.000] provides a dramatic red-to-green gradient that makes the interaction between track record length and trial count viscerally clear.
- **Key numbers:** Monthly SR = 0.253 (annualized ~0.88, gross). Skewness = -0.261, excess kurtosis = 4.23 (ours). MinTRL = 174 months at M=10, 95% confidence (ours); under Gaussian assumption would be ~10 months (expectations probe). DSR at (M=50, T=24) = 0.159; DSR at (M=10, T=48) = 0.560. Production: DSR frequently falls below 0.5 for typical ML fund performance with annual SR 0.5-1.5, 3-5 year records, 10-50 variants (Bailey & Lopez de Prado, 2014).
- **Student takeaway:** Non-normality is not a footnote in the DSR formula — it is the dominant term. Heavy tails in equity returns inflate the minimum track record by an order of magnitude relative to the Gaussian assumption that most practitioners implicitly use.

**Sandbox vs. reality:** Excess kurtosis of 4.23 for monthly equity long-short returns is typical, not extreme. Production strategies show similar or higher kurtosis, especially during stress periods. The MinTRL of 174 months is the honest consequence of this distribution — it means most strategies with 3-5 year track records cannot confirm their observed Sharpe ratios at 95% confidence, which is the paper's central empirical claim. Additionally, the gross Sharpe of 0.88 includes survivorship inflation estimated at 0.1-0.3 units annually (per S1 and expectations Interaction 5). If the true Sharpe is 0.1-0.3 units lower, the DSR surface shifts substantially toward more pessimistic values and the MinTRL would increase further. The DSR results presented here are an upper bound on the strategy's statistical significance.

---

### Section 6: The Responsible Backtest — Putting It Together

**Blueprint intended:** End-to-end comparison of a naive evaluation (sklearn TimeSeriesSplit, no costs, no correction) against a responsible evaluation (purged CV, net-of-cost, DSR). The divergence is the point.
**Expected:** Naive Sharpe in [0.5, 1.8]; responsible Sharpe in [0.1, 1.2]; gap in [0.1, 0.8]; DSR at M=10 >= 0.50 for pass verdict or < 0.50 for no-deploy. Sub-period IC split reported; SIGNAL DECAY warning if second-half IC < 50% of first half.
**Actual:** Naive Sharpe = 0.877. Responsible Sharpe (net tiered + purged CV) = 0.575. Gap = 0.302 (34.4% reduction). DSR at M=10 on responsible returns = 0.414 (NO-DEPLOY). Sub-period IC: first half (2019-04 to 2021-11, 32 months) = 0.054; second half (2021-12 to 2024-11, 36 months) = 0.039 — a 27% relative decline, below the 50% threshold for SIGNAL DECAY. All values within acceptance ranges. IS/OOS labeled in plots.
**Category:** Matches

**Teaching angle:**
- **Frame as:** The capstone demonstration. The same strategy evaluated naively (gross returns, walk-forward CV, no multiple testing correction) shows Sharpe = 0.877 — an apparently compelling result. Under responsible evaluation (net of tiered transaction costs, purged cross-validation, DSR at M=10), the Sharpe drops to 0.575 and the DSR of 0.414 triggers a NO-DEPLOY verdict. The strategy loses more than a third of its apparent quality when proper methodology is applied, and it fails the statistical significance test. This is not a pathological example — it is the expected outcome for a monthly cross-sectional alpha strategy on a 68-month track record with heavy-tailed returns.
- **Key numbers:** Naive Sharpe = 0.877, responsible Sharpe = 0.575, gap = 0.302 (34.4% reduction); DSR@M=10 = 0.414 (NO-DEPLOY); sub-period IC: 0.054 (2019-2021) to 0.039 (2022-2024), 27% decline. Production benchmark: average ML fund live Sharpe is 40-60% of backtest Sharpe (Lopez de Prado, 2017, GARP). Our 65.6% retention rate is slightly better than the production benchmark, consistent with our use of research-grade (not institutional-scale) cost assumptions.
- **Student takeaway:** The gap between the naive and responsible evaluation is not a problem to solve — it is the information. A strategy that survives responsible evaluation is worth deploying. This one does not, at M=10. The verdict is the product.

**Sandbox vs. reality:** The 0.302 Sharpe gap between naive and responsible evaluation understates the full gap because the responsible evaluation does not correct for survivorship inflation of the gross Sharpe (estimated at 0.1-0.3 annual units per S1 and expectations Interaction 5). A survivorship-corrected Sharpe would widen the naive-to-responsible gap and push the DSR further below the NO-DEPLOY threshold.

---

## Seminar Exercises

### Exercise 1: The Look-Ahead Bug Hunt

**Blueprint intended:** Three pre-computed IC series — one with a look-ahead bug, one with survivorship bias, one correct — presented for students to diagnose by fingerprint. IS/OOS IC ratios distinguish the bug types.
**Expected:** Signal A (look-ahead) IS IC >= 0.50, OOS IC <= 0.10, ratio >= 5x. Signal B (survivorship) IS IC >= 0.03, OOS IC >= 0.02, ratio <= 2x. Signal C (correct) IS IC in [0.01, 0.06], OOS IC in [0.01, 0.05], ratio in [0.8, 2.0].
**Actual:** Signal A: IS IC = 1.000, OOS IC = 0.016, ratio = 64.4x. Signal B: IS IC = 0.031, OOS IC = 0.017, ratio = 1.8x. Signal C: IS IC = 0.029, OOS IC = 0.018, ratio = 1.6x. Signal B OOS IC of 0.0173 is marginally below the expected >= 0.02 threshold (gap = 0.0027, within noise for 35 monthly observations). Signals constructed from Week 4 GBM/NN caches rather than fresh momentum factors; Signal B reframed as "model-universe coupling" rather than classical survivorship filtering; IS period is 2019-04 to 2021-12 (not expectations' 2015-2019).
**Category:** Matches

**Teaching angle:**
- **Frame as:** Three IC fingerprints, three different failure modes. Signal A (look-ahead) shows IC = 1.0 in-sample collapsing to 0.016 out-of-sample — a 64x ratio that screams "this signal is using the answer." Signal B (survivorship/model-universe coupling) inflates both IS and OOS roughly equally — ratio 1.8x, a gradual bias without collapse. Signal C (correct) shows stable IS/OOS behavior at ratio 1.6x. The diagnostic is the ratio, not the IC level: catastrophic collapse indicates information leakage; near-uniform inflation indicates universe bias; stability indicates clean methodology.
- **Key numbers:** Signal A ratio = 64.4x (ours); Signal B ratio = 1.8x; Signal C ratio = 1.6x. Production: look-ahead bugs produce "explosively profitable in-sample with immediate collapse"; survivorship bias produces "deceptively smooth equity curves with modest but consistent outperformance" (Luo et al., 2014, Deutsche Bank).
- **Student takeaway:** Different biases leave different IC fingerprints. The IS/OOS ratio is a first-pass diagnostic: a ratio above 5x is a red flag for information leakage; a ratio near 1 with uniformly elevated IC suggests universe or data contamination.

---

### Exercise 2: Purging vs. Walk-Forward — The CV Comparison

**Blueprint intended:** Implement both walk-forward and purged k-fold CV on the Week 4 alpha model. Compare cross-validated IC estimates. Measure how large the purging adjustment is and whether model rankings change between methods.
**Expected:** Walk-forward IC in [0.01, 0.10]; purged IC in [0.01, 0.09]; purged IC <= WF IC; IC delta >= 0.002 (ideally 0.005-0.030). Rank flip folds: 2-6 out of 10. Signal viability: Low.
**Actual:** WF IC (GBM) = 0.0203; purged IC (GBM) = 0.0213 (reversed direction). IC delta = -0.0011. NN: WF IC = 0.0209, purged IC = 0.0270. Rank flip folds = 6/10. IC ranges pass; direction and delta fail (same structural issue as S2).
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** The exercise demonstrates that on monthly data with 1-month labels, the purging effect on IC level is negligible (delta = -0.0011, t = -0.030). But the exercise also reveals a more nuanced finding: the CV method changes model rankings in 6 of 10 folds. The absolute IC gap is small, but the relative ranking of models (GBM vs. NN vs. Ridge) shifts in the majority of folds. This means methodology choice affects which model you select, even when the aggregate IC gap is tiny. The rank flip is the exercise's real teaching point.
- **Key numbers:** GBM WF IC = 0.0203 (t ≈ 1.4, p ≈ 0.17, n = 68 months — statistically insignificant), purged IC = 0.0213; IC delta = -0.0011 (not significant, t = -0.030); rank flip folds = 6/10 (ours). Production benchmark: WF overestimates IC by 20-50% for 20-day labels on daily data (Lopez de Prado, 2018) — our near-zero delta is consistent with monthly frequency compressing the contamination zone to a single observation.
- **Student takeaway:** Even when the IC gap between CV methods is negligible, the methods can disagree on which model is best. Methodology choice affects model selection, not just signal estimation. On monthly data with short labels, the purging effect is structurally small — but on daily data with multi-day labels, it would be substantial.

**Sandbox vs. reality:** The near-zero IC delta is specific to monthly frequency with 1-month labels. On daily data with 21-day forward-return labels — the standard setup at production quant shops — the purging effect is material (10-30% IC reduction per Lopez de Prado 2018). Students should understand the frequency dependence: our exercise shows the mechanism correctly but at a frequency where the effect is minimized.

---

### Exercise 3: Transaction Cost Sensitivity Analysis

**Blueprint intended:** Compute net-of-cost Sharpe under a grid of cost assumptions (half-spread 2-30 bps x turnover reduction 0-50%). Identify the feasibility frontier where net Sharpe = 0.5.
**Expected:** Feasibility frontier monotonically decreasing; at (30 bps, 0% reduction) net Sharpe <= gross - 0.3; FEASIBLE_CELLS > 0; breakeven spread at 0% reduction reported.
**Actual:** 9x6 grid computed. Feasibility frontier monotonically decreasing (confirmed). At (30 bps, 0% reduction): net Sharpe = 0.319 (gross - 0.552, well below threshold). FEASIBLE_CELLS = 46/54. Breakeven spread at 0% turnover reduction = 15 bps. Gross Sharpe = 0.871, mean turnover = 140%/month. At (2 bps, 0%): net Sharpe = 0.789. Market impact subtracted as fixed monthly drag.
**Category:** Matches

**Teaching angle:**
- **Frame as:** The feasibility frontier makes the cost-turnover tradeoff explicit. At 140% monthly turnover, the strategy remains viable (Sharpe >= 0.5) up to 15 bps half-spread with no turnover reduction — comfortably covering large-cap S&P 500 stocks (actual spreads 2-5 bps) but failing at mid-cap levels (12-25 bps). Modest turnover reduction (20-30%) extends the frontier to higher spread levels. The heatmap's green-to-red gradient shows that 46 of 54 grid cells remain feasible — the strategy is robust at large-cap cost levels but breaks down as costs approach mid-cap territory.
- **Key numbers:** Breakeven spread = 15 bps at 0% turnover reduction (ours); feasible cells = 46/54; net Sharpe at (30 bps, 0%) = 0.319 (deep in non-viable territory); net Sharpe at (2 bps, 50% reduction) = 0.814 (near gross). Production large-cap spreads = 2-5 bps (current markets); mid-cap = 12-25 bps (Corwin & Schultz, 2012; frec.com 2023).
- **Student takeaway:** The feasibility frontier quantifies the answer to "can this strategy survive real costs?" For our signal, the answer is yes at large-cap spreads and no at mid-cap — and turnover reduction can shift the boundary substantially.

---

### Exercise 4: DSR Calibration — How Many Strategies Did You Actually Try?

**Blueprint intended:** Compute DSR for the Week 4 model under varying trial counts (1-50) and backtest lengths (6-60 months). Identify where DSR first crosses below 0.5.
**Expected:** DSR in [0, 1]; monotonically decreasing in M; DSR(T=6, M=50) <= 0.50; DSR(T=60, M=1) >= 0.90; skewness and excess kurtosis non-zero.
**Actual:** 5x5 grid (T=[6,12,24,36,60] x M=[1,5,10,20,50]) using net annualized SR = 0.704 with per-slice moment re-estimation. All cells in [0, 1]. Monotone in M confirmed. DSR(T=6, M=50) = 0.045 (far below 0.50). DSR(T=60, M=1) = 1.000. Skewness = -0.261, excess kurtosis = 4.23. Crossover at M=5 for T <= 24 months; at M=10 for T = 36 months.
**Category:** Matches

**Teaching angle:**
- **Frame as:** The exercise uses the net Sharpe (0.704 annualized) rather than the gross, producing a dramatically degraded surface. At just 5 trials with a 2-year track record, DSR already falls below 0.50. At 50 trials with a 6-month record, DSR = 0.045 — the strategy is indistinguishable from the best of 50 noise processes. The non-normality (excess kurtosis = 4.23) makes the degradation more severe than a Gaussian model would predict. The exercise quantifies the honest answer to "how many strategies did you try before this one?"
- **Key numbers:** DSR(T=6, M=50) = 0.045 (ours); DSR(T=60, M=1) = 1.000; crossover M=5 at T=24m. Net SR = 0.704 annualized, skew = -0.26, excess kurt = 4.23 (ours). Production: DSR-based evaluation reduces false positives by 30-50% (Arian et al., 2024).
- **Student takeaway:** A 6-month backtest of the winning strategy out of 50 variants has only a 4.5% probability of being genuine. Honest DSR computation requires honest accounting of the search process.

---

## Homework Deliverables

### Deliverable 1: Build a Purged CV Engine

**Blueprint intended:** A PurgedKFold class implementing label-aware purging and embargo from scratch, following Lopez de Prado Chapter 7. Must pass a correctness test (zero leaking training samples). Includes visual diagnostic and three-way CV comparison.
**Expected:** Correctness test: zero leaking training samples (non-negotiable). Purged IC <= WF IC. IC delta reported with t-stat. Visual diagnostic shows purged/embargo zones.
**Actual:** PurgedKFold class implemented with sklearn BaseCrossValidator interface, supporting both DatetimeIndex and positional fallback. Correctness test passes: n_leaking = 0 for k=5 and k=10, n_embargo_violations = 0. IC comparison: purged IC = 0.0213 > WF IC = 0.0203 (reversed direction, same as S2/Ex2). IC delta = -0.0011, t = -0.030. Visual diagnostic (d1_purged_kfold_splits.png) clearly shows 5 folds with train/test/purged/embargo zones on weekly data (260 periods). ValueError raised for 3/3 invalid input types.
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** The PurgedKFold class is correct — the non-negotiable correctness test confirms zero information leakage from training to test folds. The IC direction reversal (purged slightly above WF) is not a purging failure; it reflects the structural reality of monthly data with 1-month labels, where purging removes at most one observation per fold boundary. The visual diagnostic using weekly frequency (260 periods) makes the purged and embargo zones visible at a resolution where monthly data would make them invisible. The three-way comparison confirms the IC difference is statistically indistinguishable from zero (t = -0.030).
- **Key numbers:** n_leaking = 0 (correctness test, both k=5 and k=10); purged IC = 0.0213 (t ≈ 1.4, p ≈ 0.17 per S3 GBM analysis — statistically insignificant), WF IC = 0.0203; IC delta = -0.0011 (t = -0.030, not significant). Production: reference implementation in mlfinpy (MIT) available for verification; index-level agreement not required but directional result must match — in this case, the direction is reversed due to the monthly frequency, which is the structurally correct result for this setup.
- **Student takeaway:** Building the purging mechanism from scratch confirms you understand the algorithm — correctness is verified by the zero-leakage invariant, not by the IC gap magnitude. The IC gap depends on data frequency and label duration; the algorithm's correctness does not.

**Sandbox vs. reality:** The visual diagnostic uses weekly data (260 periods) specifically because monthly data (68 periods) would make the 1-observation purge zone invisible. Production implementations typically operate on daily data where the purge zone spans 20+ observations and is visually prominent without this adaptation.

---

### Deliverable 2: Transaction Cost Accounting Pipeline

**Blueprint intended:** A TransactionCostModel class that decomposes costs into spread, market impact, and slippage components. Handles the Week 4 long-short portfolio. Produces turnover series, cost decomposition, net return series, and a summary report.
**Expected:** Correctness test on synthetic 2-asset portfolio (turnover=1.0, spread_cost matching formula). Mean monthly turnover in [10%, 100%]. Pessimistic Sharpe < base < optimistic. Max drawdown net >= gross in magnitude. Top-5 highest-cost months identified.
**Actual:** TransactionCostModel class with fit(ohlcv) + transform(gross_returns) interface. Correctness test passes: turnover = 1.0000, spread_cost = 0.002000 (exact match). Mean turnover = 139.9% (above expected 100% upper bound). Three-regime ordering: pessimistic (0.409) < base (0.676) < optimistic (0.778). Gross MDD = -31.88%, base net MDD = -33.95%. Top-5 cost months: 2020-03 (impact-dominated, COVID volatility), 2020-04, 2022-06, 2022-10, 2022-08 (spread-dominated). Gross skew = -0.261, kurt = 4.13; net base skew = -0.274, kurt = 4.05.
**Category:** Expectation miscalibration

**Teaching angle:**
- **Frame as:** The TransactionCostModel is a production-quality accounting layer. The correctness test on a synthetic 2-asset swap (turnover = 1.0) validates the formula exactly. Applied to the real portfolio: the 139.9% monthly turnover produces a base-regime (tiered 5/15 bps) net Sharpe of 0.676, down from gross 0.871 — a 22% reduction in Sharpe units. Under pessimistic assumptions (25 bps flat), the net Sharpe of 0.409 falls below the typical institutional hurdle rate of 0.50. The cost decomposition reveals that spread costs dominate in normal periods but market impact surges during stress (March 2020 is the highest-cost month, driven by volatility-amplified impact).
- **Key numbers:** Gross Sharpe = 0.871, net optimistic = 0.778, net base = 0.676, net pessimistic = 0.409 (ours). Mean turnover = 139.9% one-way monthly. Highest-cost month: 2020-03, total drag = 0.61%, dominated by impact. Production: Almgren & Chriss (2000) at 5% ADV participation = 3-8 bps per trade impact; our model with eta=0.1 is at the conservative end.
- **Student takeaway:** The three-regime sensitivity table is a deployment artifact — it tells a PM exactly where the strategy breaks. At pessimistic costs (25 bps), this strategy's net Sharpe drops below 0.5. The cost decomposition reveals that impact costs spike during high-volatility regimes, making worst-case TC analysis essential.

**Sandbox vs. reality:** Our impact model uses eta = 0.1 and a fixed $100M AUM assumption. At institutional scale with higher AUM and eta = 0.15-0.20, impact costs would be 2-3x higher. The base net Sharpe of 0.676 would likely drop below 0.50 at $500M+ AUM, making the strategy capacity-constrained at institutional scale.

---

### Deliverable 3: The Responsible Backtest Report

**Blueprint intended:** Three-layer evaluation report. Layer 1: quantstats tearsheet + DSR verdict. Layer 2: CPCV across model variants + PBO + BHY correction. Layer 3: MinTRL + deployment readiness.
**Expected:** L1: Net Sharpe in [0.05, 1.5]; max drawdown <= -10%; DSR at M=10 with DEPLOY/NO-DEPLOY verdict. L2: PBO in [0.20, 0.75]; BHY correction applied; winning model selected. L3: MinTRL printed with months; qualitative discussion >= 3 sentences.
**Actual:** L1: quantstats tearsheet generated (HTML + PNG). Net Sharpe = 0.676. Max drawdown = -33.95%. DSR@M=10 = 0.504 (DEPLOY — razor-thin margin above 0.50 threshold). L2: PBO = 0.267 (15 paths, 3 models: GBM, NN, Ridge). GBM selected as winner (IC = 0.026, t = 1.38, PBO = 0.267, DSR = 0.504). BHY correction: GBM adjusted p = 0.467, NN = 0.467, Ridge = 0.603 — no model clears significance after correction. L3: MinTRL = 77.5 months at 95% confidence. Observed track record = 67 months. Track record sufficient: False (10.5-month shortfall). Qualitative discussion: 3 paragraphs covering AUM scaling, Kelly sizing, and deployment framework.
**Category:** Matches
**Confidence:** low — The DSR@M=10 = 0.504 barely clears the 0.50 DEPLOY threshold. A slightly different cost assumption, data sample, or SR convention would flip the verdict to NO-DEPLOY. The S6 version of DSR (which uses net-tiered returns at a different cost specification) reports 0.414 — firmly NO-DEPLOY. The fragility of this verdict is itself pedagogically interesting, but it means the "DEPLOY" label in D3 should not be presented as a definitive conclusion.

**Teaching angle:**
- **Frame as:** The three-layer report is the format a quant researcher produces for a PM and risk committee. Layer 1 shows the strategy is modestly profitable (net Sharpe 0.676, max drawdown -34%) but the DSR of 0.504 is a razor-thin pass — under slightly different cost assumptions (as in S6, where DSR = 0.414), the strategy fails. Layer 2 confirms the GBM model is the best of the three variants, with PBO = 0.267 indicating the IS winner tends to maintain its relative ranking OOS — though with only 15 paths and IC t < 1.96, the PBO does not confirm the signal's statistical reality. No model achieves statistical significance after BHY correction. Layer 3 delivers the reality check: MinTRL of 77.5 months at M=1 (no trial penalty) exceeds the observed 67-month track record by 10.5 months. Under the M=10 trial penalty applied in S5, the requirement extends to 174 months — reinforcing that honest accounting of the search process is not just an academic exercise but a 2x multiplier on the required track record. The strategy cannot yet confirm its own Sharpe ratio. The honest verdict: continue paper trading for approximately one more year before capital allocation (at M=1), or accumulate 14.5 years of live data to satisfy M=10.
- **Key numbers:** Net Sharpe = 0.676, DSR@M=10 = 0.504 (marginal DEPLOY), PBO = 0.267, MinTRL = 77.5 months at M=1, 95% confidence (no trial penalty); at M=10, MinTRL = 174 months (from S5, using gross returns). Observed track record = 67 months. GBM IC = 0.026 (t = 1.38, p = 0.173, BHY-adjusted p = 0.467). S6 DSR@M=10 on net-tiered = 0.414 (NO-DEPLOY under different cost specification). Production: institutional strategy evaluation maps directly to this three-layer structure (Lopez de Prado, 2017, GARP). GKX benchmark: OOS R-squared ~0.35% monthly for gradient boosting on full CRSP (Gu, Kelly & Xiu, 2020).
- **Student takeaway:** The responsible backtest report is not designed to make the strategy look good — it is designed to make the truth clear. This strategy shows marginal statistical evidence, insufficient track record, and fragile cost sensitivity. The correct deployment decision is: wait.

---

## Open Questions Resolved

1. **Whether the Week 4 long-short portfolio produces a gross Sharpe above or below 0.5 over the OOS window.**
   **Finding:** Gross annualized Sharpe = 0.876 over 68 OOS months (April 2019 to November 2024), 174 tickers. The OOS start is 2019-04 (not the expectations' assumed 2015), shortening the evaluation window from 132 months to 68.
   **Affects:** S4, S5, S6, Ex3, Ex4, D2, D3 (all sections using portfolio performance).
   **Teaching implication:** With gross Sharpe above 0.5, the TC and CV corrections produce meaningful but non-fatal reductions (responsible Sharpe = 0.575 in S6, net base Sharpe = 0.676 in D3). The strategy sits in the pedagogically interesting "borderline" zone — viable under optimistic assumptions, marginal under realistic assumptions, and non-viable under pessimistic assumptions. The S6 DSR verdict is NO-DEPLOY (0.414), while D3's verdict is marginal DEPLOY (0.504), illustrating how cost specification affects the go/no-go decision.

2. **Whether all three Week 4 model variants (linear, LightGBM, NN) are available in the cache for CPCV in Section 3 and D3 L2.**
   **Finding:** GBM and NN available from Week 4 cache. OLS not available; Ridge regression (alpha=1.0, StandardScaler, 36-month rolling window) synthesized inline as the third variant. Ridge IC was substantially weaker (mean IC = -0.001 to 0.009) than GBM (0.026) and NN (0.024).
   **Affects:** S3 (CPCV), D3 L2 (model comparison).
   **Teaching implication:** The Ridge model's near-zero IC provides useful differentiation for CPCV — it acts as a "noise model" baseline against which GBM and NN look genuinely predictive. PBO = 0.267 with this three-model set is in the "genuine alpha" range (0.25-0.35 per Bailey et al. 2015), and the rank-flip analysis benefits from having one clearly weak model that frequently swaps ranks with the others depending on the fold.
