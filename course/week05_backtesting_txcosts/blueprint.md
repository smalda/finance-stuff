# Week 5: Backtesting, Research Discipline & Transaction Costs

**Band:** 1 — Essential Core
**Implementability:** HIGH
**Difficulty:** ABOVE AVERAGE — This week is denser than a typical week. It covers three tightly coupled but individually substantial topics (backtesting methodology, multiple testing correction, transaction cost modeling), each with its own mathematical apparatus. Students should expect to spend more time here than on Weeks 1–4. The payoff is proportional: this material is the single most referenced skill set in quant researcher interviews.

---

## Learning Objectives

By the end of all three notebooks, students will be able to:

1. Explain look-ahead bias, purging, and embargoing — and implement purged k-fold CV from scratch following the López de Prado algorithm.
2. Distinguish walk-forward, purged k-fold, and combinatorial purged CV (CPCV) — and articulate when each is appropriate and why the gap between them matters for production.
3. Compute the deflated Sharpe ratio (DSR) and probability of backtest overfitting (PBO) for a candidate strategy, and interpret the results as a go/no-go signal.
4. Apply multiple testing corrections (Bonferroni, Benjamini-Hochberg) to a set of candidate signals and quantify the false discovery rate.
5. Decompose transaction costs into their three canonical components — bid-ask spread, market impact, and slippage — and compute the net-of-cost portfolio return series.
6. Measure and explain the backtest-to-live performance gap for a strategy that survives naive evaluation but degrades when costs and proper CV are applied.
7. Build a rigorous backtesting pipeline that integrates purged CV, deflated Sharpe evaluation, and transaction cost accounting in end-to-end form.

---

## Prerequisites

- **Required prior weeks:** Week 1 (data mechanics, look-ahead bias concept), Week 2 (stationarity, time-series structure), Week 3 (factor models, multiple testing correction concepts introduced there), Week 4 (alpha model output — the `AlphaModelPipeline`, IC/ICIR metrics, decile portfolio construction)
- **Assumed from curriculum_state:** IC and rank IC computation (`shared.metrics.ic_summary`, `rank_ic`); long-short decile portfolio construction (`shared.backtesting.quantile_portfolios`, `long_short_returns`); basic turnover and transaction cost estimation (`shared.backtesting.net_returns` with `cost_bps`, introduced in Week 4); Bonferroni and Benjamini-Hochberg correction mechanics (introduced in Week 3); survivorship bias definition and look-ahead bias concept (Week 1); adjusted close prices, return computation (Week 1)
- **Week 4 code artifacts consumed by this week:**
  - `AlphaModelPipeline` class (`week04/code/hw/d1_alpha_engine.py`) — the primary integration point; produces `.predictions_`, `.ic_series_`, `.summary()` with gross/net Sharpe, turnover, drawdown
  - Model comparison output (`week04/code/hw/d3_model_comparison.py`) — OLS, Ridge, LightGBM, and Neural Net variants with `NNRegressor` sklearn wrapper; this gives Week 5 four (not three) model variants to run CPCV across
  - Expanded feature set (`week04/code/.cache/expanded_features.parquet`) — 15+ features from D2 feature lab
  - Walk-forward splits (`shared.temporal.walk_forward_splits`, `PurgedWalkForwardCV`) — already implemented with `purge_gap` parameter
  - `CombinatorialPurgedCV` (`shared.temporal`) — already implemented; Week 5 uses it, does not rebuild it
  - `deflated_sharpe_ratio` (`shared.metrics`) — already implemented; Week 5 teaches the theory and applies the utility
- **Assumed ML knowledge:** Cross-validation mechanics (k-fold, train/test splits); the bias-variance tradeoff; overfitting and generalization; the concept of information leakage in supervised learning
- **Prerequisite explanations in this week:**
  - None required — all supporting concepts have been established in Weeks 1–4. This week applies them in a new, higher-stakes context.

---

## Opening Hook

In 2016, a study in the *Notices of the American Mathematical Society* showed that a researcher testing 45 independent trading strategies on the same dataset has an expected probability greater than 50% of finding one that looks spectacular purely by chance — before accounting for parameter tuning. The strategy that looks best in your backtest is, more likely than not, the one that fit the noise the hardest. The question is not whether your backtest has a high Sharpe ratio. The question is: has it cleared the bar that accounts for how many times you looked?

---

## Narrative Arc

**The setup.** An ML expert arriving in quant finance sees backtesting as a solved problem. They know train/test splits, they know k-fold cross-validation, they know not to shuffle time-series data. They apply these techniques and get a strategy with a Sharpe ratio of 1.4. In any other ML context, that would be a publishable result. They present it. Then a senior researcher asks: "What's the deflated Sharpe? Did you purge? How many variants did you try before that one? What happens when you subtract real transaction costs?"

**The complication.** Financial time-series cross-validation is harder than standard CV in ways that look trivial and compound catastrophically. The label at time T is computed from returns between T and T+k — which means the training samples immediately adjacent to a test fold contain information from the future that leaked through the label construction. Standard k-fold, even time-aware k-fold, does not remove this leakage; purging does. Add to this: every additional backtest you run on the same dataset is another draw from the same noisy distribution. The strategy that survived 100 variants of parameter search did not outperform because it is good — it outperformed because it fit the noise for that particular 10-year window. The deflated Sharpe ratio and PBO quantify exactly how much of your observed performance disappears once you account for the search. And even a strategy that survives all of this evaporates in live trading when the bid-ask spread and market impact eat 40% of the gross return.

**The resolution.** By week's end, students understand backtesting not as a final verification step but as a discipline with formal rules. They can build a pipeline that accounts for label leakage (purging), multiple testing (DSR, Bonferroni), and implementation friction (TC decomposition). They leave knowing that the gap between "great backtest" and "live alpha" is not a mystery — it is a precise set of corrections, each one addressable in code.

---

## Lecture Outline

### Section 1: The Seven Sins — A Taxonomy of Backtest Failure → `s1_seven_sins.py`

- **Hook:** A Deutsche Bank quant team catalogued the most common ways systematic strategies fail after a promising backtest. Every sin on the list is a form of optimism that the data didn't earn — and every one is preventable with a checklist.
- **Core concepts:** Look-ahead bias (using T+k information at T); survivorship bias (universe bias in backtested returns, quantified from Week 1); data snooping (in-sample search masquerading as discovery); ignoring transaction costs; ignoring short-selling constraints; improper benchmark comparison; regime neglect
- **Note on regime neglect:** This sin is identified here but not resolved in this week. The tools to detect and condition on regimes arrive in Week 14 (Regime Detection & Adaptive Models). Explicitly flag this to students: "You now know this failure mode exists. You do not yet have the tools to fix it. That's intentional — regime-conditional backtesting requires regime detection machinery that we haven't built yet."
- **Key demonstration:** Walk through a deliberately flawed backtest — a momentum strategy coded with a classic look-ahead bug (using current close rather than previous close for signal formation). Show the equity curve of the flawed version vs. the corrected version; the difference is the look-ahead premium.
- **Key formulas:** Look-ahead return bias: signal uses return from t to t+1 to predict return from t to t+1 (trivially explained as the outcome itself). Survivorship return inflation: E[R | survive] − E[R | full universe].
- **"So what?":** Every sin in this taxonomy has appeared in published papers and live fund strategies. The taxonomy is a pre-flight checklist. Students who cannot name these seven failure modes will repeat them.
- **Bridge:** Now that we know the failure modes, we need the tools to prevent them — starting with the most important: a sound cross-validation framework.

---

### Section 2: Purged Cross-Validation — Closing the Leakage Gap → `s2_purged_cv.py`

- **Hook:** Standard time-series CV respects order but not labels. When your label is a 21-day forward return, the training sample at T-1 already "knows" most of what the test sample at T will discover. Purging removes contaminated observations. Embargoing blocks the information-rich boundary region. Without both, your CV score is an optimistic fiction.
- **Core concepts:** Label overlap and information leakage in financial labels; the purging rule (remove all training observations whose label interval overlaps the test fold's date range); the embargo (remove the k-day buffer after each test fold); k-fold variant — walk-forward vs. expanding window vs. purged k-fold; the difference between a gap parameter (sklearn TimeSeriesSplit) and true label-aware purging
- **Key demonstration:** Side-by-side CV score comparison: sklearn TimeSeriesSplit (with gap), `shared.temporal.PurgedWalkForwardCV` (already used in Week 4 with `purge_gap=1`), and a from-scratch purged k-fold implementation that makes the purging logic explicit. Plot the train/test splits visually — show where leakage exists and where purging eliminates it. Use `AlphaModelPipeline` from `week04/code/hw/d1_alpha_engine.py` as the model under evaluation.
- **Key formulas:** Purging condition: remove observation i from training set if t_i ∈ [t_j − L, t_j] where t_j is a test sample and L is the label look-forward window. Embargo: additionally remove observations in [t_j, t_j + embargo_days].
- **"So what?":** The difference in cross-validated IC between leaky and purged CV is not just a theoretical concern — it determines whether you deploy a strategy. Leaky CV can show a positive IC for a strategy that has zero true predictive power.
- **Sidebar:** mlfinlab closure (MENTIONED) — the original reference implementation (Hudson & Thames) is now closed-source/subscription; use `mlfinpy` (MIT, actively maintained) or implement from scratch as done here.
- **Bridge:** Purged CV tells us how to split correctly. But even with correct splits, we can still overfit if we search long enough over enough variants. The next section quantifies that search bias.

---

### Section 3: Combinatorial Purged CV & the Multiple-Testing Problem → `s3_cpcv_multiple_testing.py`

- **Hook:** You found the best of 50 strategy variants. Walk-forward says Sharpe 1.4. What the walk-forward doesn't tell you: if those 50 variants were all drawn from the same noisy distribution, the expected maximum Sharpe across 50 independent noise processes is substantially above 1.0. Your "best" strategy may have won a beauty contest in noise.
- **Core concepts:** Combinatorial purged CV (CPCV): exhaustive combinatorial train/test partitioning that generates a distribution of performance outcomes, not a single point estimate; probability of backtest overfitting (PBO): the fraction of CPCV splits where the best in-sample variant underperforms out-of-sample; multiple testing in factor research — the Harvey-Liu-Zhu t-statistic hurdle of 3.0; Bonferroni vs. Benjamini-Hochberg correction (the BHY variant for dependent tests in finance)
- **Key demonstration:** Use the four model variants from `week04/code/hw/d3_model_comparison.py` (OLS, Ridge, LightGBM, Neural Net via `NNRegressor`) and vary feature subsets (baseline 7 features vs. expanded 15+ from `week04/code/.cache/expanded_features.parquet`) to create a realistic pool of strategy variants. Run `shared.temporal.CombinatorialPurgedCV` across all variants. Show the distribution of out-of-sample returns for the in-sample winner — illustrate PBO as the fraction of the distribution that falls below zero.
- **Key formulas:** PBO = fraction of CPCV splits where the IS winner underperforms the median OOS rank. Harvey-Liu-Zhu threshold: minimum t-stat for a new factor claim = 3.0 (accounting for cumulative factor discovery). Bonferroni-adjusted p-value: p_adj = p × M (number of tests). BHY step-up: sorted p-values with dependency correction for positive correlation.
- **"So what?":** CPCV is the quantitative answer to "how do I know my strategy is real?" PBO > 0.5 is a no-deploy signal. Most published academic factors fail the Harvey-Liu-Zhu hurdle when tested on the same data used to discover them.
- **Sidebar:** "Backtesting is not a research tool" (MENTIONED) — López de Prado's framing: use feature importance and purged CV during development; run the backtest once on a fully specified, locked strategy. Iterative backtesting is strategy mining with extra steps.
- **Bridge:** We can now evaluate whether a strategy's performance is statistically real. The next question is: even if it's real gross, does it survive the cost of trading?

---

### Section 4: Transaction Cost Decomposition → `s4_transaction_costs.py`

- **Hook:** A strategy showing 30% gross annual return commonly survives backtesting. The same strategy delivering 14% net return after costs does not match the fund's hurdle rate. The gap — 16 percentage points — is not slippage. It's physics: the market impact of moving money, the bid-ask tax on every trade, and the drift between your signal's price and your fill price.
- **Core concepts:** Three-component TC decomposition: (1) bid-ask spread cost — the half-spread paid on each buy and sell, expressed in basis points; (2) market impact — permanent impact (price shift that persists) and temporary impact (urgency premium that dissipates), modeled via the square-root law; (3) slippage — the difference between the signal price and the execution price, driven by latency and order routing; turnover as the multiplier; the Almgren-Chriss framework at a conceptual level
- **Key demonstration:** Compute round-trip TC for the Week 4 long-short portfolio (built via `shared.backtesting.long_short_returns` and `portfolio_turnover` in `week04/code/lecture/s6_signal_to_portfolio.py`) under three progressively realistic cost regimes: (a) zero cost (naive), (b) a fixed spread assumption at `COST_BPS=10` (the Week 4 default from `data_setup.py`), (c) stock-specific spread proxy using market-cap tiers from `shared/data.py` fundamentals. Show the equity curve for all three; show which turnover levels are fatal. Week 4's `shared.backtesting.net_returns(gross, turnover, cost_bps)` handles regime (b); this section extends it to stock-level costs.
- **Key formulas:** Round-trip spread cost: 2 × half_spread × |Δweight|. Market impact (square-root law): impact = σ × η × sqrt(participation_rate), where η is the impact coefficient. Net return: R_net = R_gross − turnover × cost_per_unit_turnover.
- **"So what?":** TC is the most common cause of live-to-backtest performance gap, confirmed by practitioners across multiple independent sources. A strategy that doesn't survive realistic costs at its observed turnover level should not be deployed — the backtest is irrelevant.
- **Sidebar:** TAQ and institutional TC data (MENTIONED ABSTRACTLY) — production-grade TC modeling uses actual quote data (NYSE TAQ) and measured fills. Our proxy approach (fixed spread tiers + square-root impact) is standard for research with free data.
- **Bridge:** We can decompose costs analytically. But we need to integrate this into a complete evaluation — where gross performance, CV methodology, and cost modeling all feed into a single verdict.

---

### Section 5: The Deflated Sharpe Ratio → `s5_deflated_sharpe.py`

- **Hook:** Your strategy has a Sharpe ratio of 1.4. How impressive is that? It depends entirely on how long the backtest is, how non-normal the returns are, and how many strategies you tried before that one. The deflated Sharpe ratio adjusts for all three — and often reveals that the "impressive" number is indistinguishable from a lucky draw from noise.
- **Core concepts:** The probabilistic Sharpe ratio (PSR): the probability that the true Sharpe ratio exceeds a benchmark threshold, given observed return length, skewness, and kurtosis; the deflated Sharpe ratio (DSR): the PSR adjusted for the number of strategy variants tested (the trials penalty); minimum track record length (MinTRL): the minimum backtest length required to assert a given Sharpe with confidence; non-normality adjustment via skewness and excess kurtosis; the DSR < 0 cutoff as a disqualification threshold
- **Intuition scaffolding (present before the formula):** Frame DSR as a signal-to-noise ratio. The numerator is the observed Sharpe minus the expected maximum Sharpe from pure noise (which grows with the number of trials). The denominator captures estimation uncertainty — it inflates when returns are non-normal (fat tails, skew) and when the backtest is short. Students should see this as: "DSR answers: is my Sharpe high enough to survive the combined drag of search breadth, short history, and non-Gaussian returns?" Build the formula term-by-term from this frame — do not present it as a monolithic expression. `shared.metrics.deflated_sharpe_ratio()` already implements this; students should understand the formula before using the utility.
- **Key demonstration:** Apply `shared.metrics.deflated_sharpe_ratio(observed_sr, n_trials, n_obs, skew, excess_kurt)` to the Week 4 LightGBM model's backtest performance (from `AlphaModelPipeline.summary()["sharpe_net"]`). Vary the number of "trials" (candidate models tested) across a wide range and show the DSR surface — illustrating how quickly apparent performance becomes statistically insignificant as the search space grows.
- **Key formulas:** PSR(SR*) = Φ[(SR_hat − SR*) × sqrt(T−1) / sqrt(1 − γ₃ × SR_hat + (γ₄+2)/4 × SR_hat²)], where γ₃ = skewness, γ₄ = excess kurtosis (normal = 0). Note: original paper uses kurtosis κ with (κ−1)/4; since excess kurtosis γ₄ = κ−3, this becomes (γ₄+2)/4. DSR = PSR adjusted for M trials: SR*_adjusted = SR_max × (1 − Euler-Mascheroni constant + erf(log M / sqrt(2))) applied per Bailey-López de Prado (2014).
- **"So what?":** DSR makes the multiple-testing penalty concrete and quantitative. A 3-year backtest on a strategy with 20 variants tested should have DSR evaluated at the 20-trial penalty — not the naive single-trial Sharpe. Most "live alpha" announcements don't report DSR, which is why most of them fail.
- **Bridge:** We now have the full toolkit — CV methodology, TC accounting, statistical correction for search. The final section shows what a responsible, integrated backtest looks like from start to finish.

---

### Section 6: The Responsible Backtest — Putting It Together → `s6_responsible_backtest.py`

- **Hook:** Most backtests you will encounter in the wild violate at least three of the principles covered this week. A responsible backtest is not just "more conservative" — it is a fundamentally different artifact, built to estimate live performance rather than to impress a committee.
- **Core concepts:** The backtest discipline pipeline: data hygiene check → purged CV → feature importance (during development) → single final backtest on locked strategy → TC adjustment → DSR / PBO evaluation → honest reporting; the academic-practitioner gap: walk-forward vs. purged CV in production; "the 7 reasons ML funds fail" as a closing framework; what a quant researcher's evaluation report looks like at a real fund
- **Key demonstration:** Run the complete pipeline end-to-end on the Week 4 alpha model (instantiate `AlphaModelPipeline` from `week04/code/hw/d1_alpha_engine.py` with the LightGBM configuration): purged CV via `PurgedWalkForwardCV` → net-of-cost equity curve via the extended TC model → DSR via `shared.metrics.deflated_sharpe_ratio` → final go/no-go verdict. Contrast the naive evaluation (sklearn `TimeSeriesSplit`, no costs, no correction) against the responsible evaluation — show both equity curves and DSRs side by side. The divergence is the point.
- **Key formulas:** None new — this section integrates all prior formulas into a single pass.
- **"So what?":** The gap between the naive result and the responsible result is not a problem to solve — it is the information. If the strategy survives responsible evaluation, it is worth deploying. If it doesn't, the backtest just saved the fund a year of live losses.
- **Bridge:** Week 6 moves to portfolio construction and risk management. The rigorous evaluation framework from this week flows directly into portfolio-level risk budgeting — because a strategy that survives backtesting still needs to be sized correctly in a portfolio context.

---

## Narrative Hooks

1. **The Charlatanism Paper** (Section 3): In 2014, Bailey, Borwein, López de Prado, and Zhu published a paper in the *Notices of the American Mathematical Society* titled "Pseudo-Mathematics and Financial Charlatanism." The abstract: "we demonstrate that financial charlatanism is a mathematical possibility." They showed that even honest researchers testing enough variants on a fixed dataset will produce spurious results with high probability. This is not fraud — it is the expected output of a scientific process applied to a small, noisy signal. The paper launched the modern backtesting reform movement in quantitative finance. **Suggested placement:** Introduction to Section 3 or as opening of the lecture notebook's multiple testing section.

2. **The 30→14 Collapse** (Section 4): Practitioners across multiple independent sources describe the same pattern: "We showed the PM a 30% gross return backtest. After accounting for spread, impact, and slippage at our actual AUM, the net return was 14% — below our hurdle rate." The strategy was not wrong. The backtest was. This pattern is the single most common cause of disappointment for first-year quant analysts. **Suggested placement:** Section 4 opening hook, or as the motivating example in the transaction cost section of the seminar notebook.

3. **The t=3.0 Bar** (Section 3): In 2016, Harvey, Liu, and Zhu established that given the number of factors published through 2012, any new factor claim needs a t-statistic of at least 3.0 to be taken seriously — not the traditional 1.96 (p < 0.05). As of 2024, with hundreds of additional published factors, some researchers argue the bar should be even higher. Most published factors in academic finance do not clear 3.0. **Suggested placement:** Section 3 sidebar on the factor zoo, connecting to Week 3's multiple testing introduction.

4. **The Walk-Forward Majority** (Section 2/6): Despite the statistical superiority of purged k-fold CV (confirmed in a 2024 controlled study in *Knowledge-Based Systems*), the majority of production quant shops still use walk-forward analysis as their primary validation tool. Not because it's better — because it's simpler to explain to PMs, runs faster, and produces single-path equity curves that are easier to interpret. This is the academic-practitioner gap in quantitative finance. Understanding why the gap exists teaches more than simply adopting the academically superior method. **Suggested placement:** Section 2 closing or Section 6 framing.

---

## Seminar Outline

### Exercise 1: The Look-Ahead Bug Hunt

- **Task type:** Guided Discovery
- **The question:** Given three versions of a momentum signal computation — one with a look-ahead bug, one with a survivorship bias, one correct — can you identify each bug by comparing in-sample vs. out-of-sample IC degradation patterns? What is the IC premium attributable to each bias?
- **Expected insight:** The look-ahead bug produces suspiciously high in-sample IC that collapses out-of-sample. Survivorship bias inflates both in-sample and out-of-sample IC roughly equally (the contaminated universe is used throughout). The correct version shows lower IC everywhere but more stable in-sample vs. out-of-sample behavior. Students discover that different biases have different fingerprints in the IC degradation pattern.

---

### Exercise 2: Purging vs. Walk-Forward — The CV Comparison

- **Task type:** Guided Discovery
- **The question:** Implement both walk-forward CV and purged k-fold CV on the Week 4 alpha model. Compare cross-validated IC estimates. How large is the purging adjustment? Does the IC ranking of candidate models change between the two methods?
- **Expected insight:** Purged CV produces lower IC estimates than walk-forward on the same model and data — the magnitude of the difference is the leakage premium that walk-forward naively attributed to signal quality. In some folds, purging changes the rank ordering of models, meaning the model that looks best under walk-forward may not be the model that looks best under purged CV. This is the practical consequence of methodology choice.

---

### Exercise 3: Transaction Cost Sensitivity Analysis

- **Task type:** Guided Discovery
- **The question:** For the Week 4 long-short portfolio, compute net-of-cost Sharpe under a grid of cost assumptions: half-spread from 2 to 30 bps, turnover reduction (filter low-conviction trades) from 0% to 50%. Identify the feasibility frontier — the combinations of spread and turnover where the strategy remains viable at a Sharpe threshold.
- **Expected insight:** The strategy's viability is non-linearly sensitive to turnover: modest turnover reduction (e.g., 20%) can restore profitability that a high spread assumption destroys. High-turnover signals are simply not viable beyond a cost threshold, regardless of gross IC. The feasibility frontier makes the tradeoff explicit and quantitative.

---

### Exercise 4: DSR Calibration — How Many Strategies Did You Actually Try?

- **Task type:** Guided Discovery
- **The question:** Compute the deflated Sharpe ratio for the Week 4 model's backtest results under varying assumed "trials" counts: 1, 5, 10, 20, 50. At what trial count does the DSR first cross below 0? How does backtest length (6 months, 1 year, 3 years, 5 years) interact with the trials penalty?
- **Expected insight:** The DSR degrades rapidly with the number of trials — a result that comfortably clears the bar at 1 trial may fall below zero by 10–15 trials. Short backtests are especially vulnerable: a 6-month backtest that looks compelling under no-search assumption is essentially uninformative at 20 trials. Students discover that honest DSR computation requires honest accounting of the search process, not just the final model.

---

## Homework Outline

### Deliverable 1: Build a Purged CV Engine

- **Task type:** Construction
- **Mission framing:** Every backtest you will run for the rest of the course — and in your career — depends on having a correct temporal splitting mechanism. This deliverable builds that infrastructure from first principles. The purging algorithm is short, the stakes are high, and the implementation must be provably correct.
- **Scope:** A `PurgedKFold` class that accepts a DataFrame with DatetimeIndex and a label look-forward window (in trading days), implements the purging and embargo logic from López de Prado Chapter 7, and produces train/test split indices. This is distinct from `shared.temporal.PurgedWalkForwardCV` (which does walk-forward with a purge gap) — `PurgedKFold` implements true k-fold splitting with label-aware purging, allowing non-sequential test folds. Must pass a correctness test: given a synthetic DataFrame with a known look-ahead label, the test fold must contain zero training samples whose label interval overlaps it. Includes a visual diagnostic plot showing the purged splits for a sample dataset. Includes a three-way comparison of sklearn's `TimeSeriesSplit` (with gap parameter) vs. `shared.temporal.PurgedWalkForwardCV` (Week 4's approach) vs. `PurgedKFold` CV scores on the Week 4 alpha model via `AlphaModelPipeline`.
- **Requires:** Week 4 `AlphaModelPipeline` (`week04/code/hw/d1_alpha_engine.py`), `shared.temporal` splitters

---

### Deliverable 2: Transaction Cost Accounting Pipeline

- **Task type:** Construction
- **Mission framing:** A backtest without transaction costs is a wish. This deliverable builds the accounting layer that converts gross portfolio returns into net returns, decomposed by cost component. It must handle the Week 4 long-short portfolio and generalize to arbitrary portfolio weight matrices.
- **Scope:** A `TransactionCostModel` class that accepts a DataFrame of portfolio weights (stocks × dates), a spread assumption (either fixed bps or a stock-level vector derived from market-cap tier via `shared.data.load_sp500_fundamentals` market cap data), and a market impact parameter. Produces: (1) turnover series, (2) spread cost series, (3) impact cost series, (4) net return series, (5) a cost decomposition summary report. This extends Week 4's simple `shared.backtesting.net_returns(gross, turnover, cost_bps)` — which applies a flat per-unit cost — into a three-component decomposition with stock-level granularity. Must demonstrate: compute gross vs. net Sharpe for the Week 4 long-short portfolio (using `AlphaModelPipeline.predictions_` to derive weights); sensitivity table across three spread regimes (optimistic / base / pessimistic); identification of highest-cost periods and their drivers.
- **Requires:** Week 4 `AlphaModelPipeline` output (`week04/code/hw/d1_alpha_engine.py`), `shared.backtesting` portfolio utilities

---

### Deliverable 3: The Responsible Backtest Report

- **Task type:** Investigation (Layered)
- **Mission framing:** A real fund researcher produces a backtest report that a PM and a risk committee can interrogate. This deliverable builds that report for the Week 4 alpha model — integrating everything from Deliverables 1 and 2 plus statistical evaluation of overfitting risk. The goal is not to make the strategy look good. The goal is to make the truth clear.
- **Scope:**
  - **Layer 1 (Baseline):** Run the full pipeline: purged CV (using D1's `PurgedKFold`) → net-of-cost equity curve (using D2's `TransactionCostModel`) → quantstats tear sheet with Sharpe, Sortino, max drawdown (`shared.backtesting.max_drawdown`), calmar ratio, monthly returns heatmap. Report the DSR via `shared.metrics.deflated_sharpe_ratio` assuming a trials count of 10. If DSR < 0, document this as the verdict. If DSR > 0, continue to Layer 2.
  - **Layer 2 (Smarter Evaluation):** Run `shared.temporal.CombinatorialPurgedCV` across the 4 model variants from `week04/code/hw/d3_model_comparison.py` (OLS, Ridge, LightGBM, Neural Net via `NNRegressor`). Compute PBO. Report which model survives PBO < 0.5 and has the highest net-of-cost Sharpe under purged CV. Apply BHY multiple testing correction to the four IC estimates. The final report states the winning model with full statistical justification.
  - **Layer 3 (Frontier):** Implement MinTRL (minimum track record length) for the surviving model — compute how many additional months of live trading would be needed to confirm the observed Sharpe with 95% confidence. Then make it quantitative: given MinTRL and the fund's cost of capital (assume 5% annual hurdle), compute the maximum capital a rational allocator would commit during the confirmation period under a simple Kelly-fraction argument — i.e., the position size that limits expected loss to X% of AUM if the strategy turns out to be noise. Present MinTRL, confirmation-period capital limit, and break-even live Sharpe as a three-number "deployment readiness summary" — the format a PM would actually see on a strategy evaluation memo.
- **Requires:** Deliverables 1 and 2; Week 4 model outputs (`AlphaModelPipeline`, `d3_model_comparison.py` variants); `shared.metrics.deflated_sharpe_ratio`; `shared.temporal.CombinatorialPurgedCV`

---

## Concept Matrix

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Seven sins taxonomy | Demo: look-ahead bug equity curve | Ex 1: identify bug by IC fingerprint | — |
| Purging & embargo | Demo: visual split comparison | Ex 2: compare CV score gap | D1: build PurgedKFold class |
| Walk-forward vs. purged CV | Demo: methodology gap | Ex 2: IC rank flip across methods | D3: use purged CV throughout |
| CPCV | Demo: CPCV distribution of performance | — | D3 L2: run across model variants |
| Multiple testing correction (Bonferroni, BHY) | Demo: factor discovery t-stat hurdle | — | D3 L2: BHY correction on IC estimates |
| Deflated Sharpe ratio | Demo: DSR surface vs. trial count | Ex 4: DSR calibration | D3 L1: compute DSR |
| Probability of backtest overfitting | Demo: PBO from CPCV simulation | — | D3 L2: PBO decision signal |
| TC decomposition (spread, impact, slippage) | Demo: three-regime cost comparison | Ex 3: feasibility frontier | D2: build TransactionCostModel |
| Turnover and net Sharpe | Demo: turnover as multiplier | Ex 3: turnover reduction tradeoff | D2: gross vs. net Sharpe |
| Market impact (square-root law) | Demo: impact cost as participation rate function | — | D2: implement in TC model |
| Minimum track record length | Demo: MinTRL formula applied to Week 4 result | — | D3 L3: live confirmation horizon |
| Academic-practitioner gap | Demo: walk-forward dominance in production | Ex 2: experience the gap firsthand | D3: use responsible methodology throughout |

---

## Career Connections

**Quantitative Researcher at a multi-strategy hedge fund:** Reviews backtest reports for new signals daily. The first question asked of any new strategy is: "What's the gross and net Sharpe, and what's the DSR?" Reports without TC accounting or overfitting correction are rejected at first review. Weekly IC monitoring tracks whether live performance matches backtest predictions; persistent gaps trigger a look-ahead audit.

**ML Engineer at a systematic fund (Point72, Two Sigma, D.E. Shaw):** Owns the backtesting infrastructure. Implements purged CV splits, walk-forward engines, and transaction cost modules as production-grade library code. Runs continuous validation pipelines that flag model degradation. Responsible for ensuring that the evaluation framework used by researchers cannot accidentally produce leaky results — this is a systems problem as much as a methodology problem.

**Execution Quant / Portfolio Analytics:** Runs Transaction Cost Analysis (TCA) — the post-trade measurement of actual implementation shortfall vs. backtest assumption. Compares the strategy's live fill prices to VWAP and arrival price benchmarks. Feeds measured TC back to the research team as calibration data for future TC models. The TC model in this week's homework is the pre-trade version of what execution quants measure post-trade.

**Model Risk / Validation Analyst at an asset manager:** Audits backtests produced by the research team using exactly the framework from this week — checking for look-ahead bias, documentation of the search process, TC realism, and statistical significance under multiple testing correction. SR 11-7 model governance frameworks require exactly this kind of structured validation for models used in investment decisions.

---

## Data Sources

- `yfinance` (via `course/shared/data.py`) — OHLCV prices for the backtesting universe
- Week 4 `AlphaModelPipeline` output — cross-sectional predictions to be evaluated
- `quantstats` — performance reporting tear sheets
- `scikit-learn` — `TimeSeriesSplit` for baseline walk-forward CV comparison
- `scipy` — statistical functions for DSR (normal CDF, special functions)
- `mlfinpy` — reference implementation of purged CV (for verification, not primary code)
- Ken French Data Library factors (via `course/shared/data.py`) — benchmark returns for performance attribution

---

## Key Papers & References

### Core

- **López de Prado (2018)** — *Advances in Financial Machine Learning*, Wiley, Chapters 7–12. The primary canonical text. Defines purged k-fold CV, CPCV, the distinction between development-phase CV and validation-phase backtesting, and the research discipline framework. Required reading for any ML quant researcher.
- **Bailey & López de Prado (2014)** — *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. Journal of Portfolio Management. The DSR formula and the MinTRL calculation. The standard reference for adjusting backtested Sharpe ratios for the number of variants tested.
- **Bailey, Borwein, López de Prado, Zhu (2015)** — *The Probability of Backtest Overfitting*. Journal of Computational Finance. The CPCV framework and PBO statistic. Use when you need a full distribution of out-of-sample performance estimates, not just a single walk-forward path.
- **Harvey, Liu, Zhu (2016)** — *...And the Cross-Section of Expected Returns*. Journal of Finance. Establishes the t-stat hurdle of 3.0 for factor discovery claims, given the multiple testing inflation from prior factor research. Essential context for anyone claiming to have found a new factor.
- **Almgren & Chriss (2000)** — *Optimal Execution of Portfolio Transactions*. Journal of Risk. The canonical model for transaction cost decomposition — permanent impact, temporary impact, and spread cost. Returns in Week 13 as the basis for execution optimization.

### Advanced

- **Arian, Norouzi, Seco (2024)** — *Backtest Overfitting in the Machine Learning Era: A Comparison of Out-of-Sample Testing Methods*. Knowledge-Based Systems, Vol. 305. 2024 empirical confirmation that CPCV outperforms walk-forward and purged k-fold in a controlled synthetic environment. The strongest current academic validation of the López de Prado framework.
- **Bailey, Borwein, López de Prado, Zhu (2014)** — *Pseudo-Mathematics and Financial Charlatanism*. Notices of the American Mathematical Society. The accessible, polemical companion paper. Motivates the entire backtesting reform project in vivid language. Read for the framing, not the formulas.
- **Harvey & Liu (2015)** — *Backtesting*. Journal of Portfolio Management. Extension of the multiple testing problem specifically to the backtesting context.

---

## Bridge to Next Week

Week 5 establishes how to evaluate a strategy honestly. Week 6 asks the next question: given a strategy that has survived rigorous evaluation, how should it be sized and combined with other strategies? Portfolio construction and risk management are the natural downstream of a trustworthy signal.

**Forward reference — regime conditioning (Week 14):** The Seven Sins taxonomy identifies regime neglect as a failure mode, but this week does not resolve it. Regime-conditional backtesting — where CV splits respect regime boundaries and performance is evaluated per-regime — requires the Hidden Markov Model and change-point detection machinery from Week 14. Students should understand this as an open item: the backtesting pipeline built here is rigorous within a stationary assumption; Week 14 relaxes that assumption.
