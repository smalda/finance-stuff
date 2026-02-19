# Rigor Standard — Shared Reference

> **Consumers:** Code agent (Step 4, primary), Expectations agent (Step 3),
> Verification agent (Step 4C), Brief audit agent (Step 6¾),
> Shared export agent (Step 8).
> This is the quality standard for all code in the course. It is a set of
> rules and patterns: "When you encounter situation X, do Y."
>
> **Why this exists:** The course teaches ML practitioners how to apply ML
> in finance. The code is exemplary — students learn habits from it.
> Cutting corners teaches bad habits, even when the number falls in range.

---

## How This Guide Works

**Three tiers of progressive rigor.** Each week's expectations.md assigns
a tier. Higher tiers include all lower-tier rules.

| Tier | When it applies | What it adds |
|------|----------------|--------------|
| **1: Data Foundations** | All weeks | Temporal integrity, bias awareness, data quality, basic evaluation |
| **2: ML Discipline** | Weeks that train models | HP selection, training discipline, signal significance, feature preprocessing |
| **3: Finance Evaluation** | Weeks that evaluate strategies or production signals | Transaction costs, sub-period checks, return distributions, multiple testing |

**In expectations.md, the Rigor field looks like:**

    **Rigor:** Tier 2. Teaches: signal significance, prediction quality.

- The tier number sets which rules apply.
- "Teaches: X" means this week introduces X — implement from scratch.
  In later weeks, import from shared/.
- If a tier rule is naturally inapplicable (e.g., PIT when no
  fundamentals are used), skip it silently — no annotation needed.

**Not every code file involves ML.** Data downloads, visualizations, and
descriptive statistics don't need training discipline. Within a Tier 2
or 3 week, non-ML sections use only Tier 1 rules. Use judgment.

---

## Rigor Schedule

| Week | Band | Content | Tier | First introductions (Teaches) |
|------|------|---------|------|-------------------------------|
| 1 | 1 | Markets & Data | 1 | Survivorship bias, data quality reporting |
| 2 | 1 | Time Series & Volatility | 1 | — |
| 3 | 1 | Factor Models | 1 | — |
| 4 | 1 | ML for Alpha | 2 | Signal significance, prediction quality, baseline significance, feature preprocessing |
| 5 | 1 | Backtesting & Research | 3 | Transaction costs, sub-period degradation, multiple testing, return distributions |
| 6 | 1 | Portfolio Construction | 3 | — |
| 7 | 2 | NLP & LLMs | 3 | — |
| 8 | 2 | DL for Time Series | 3 | — |
| 9 | 2 | Derivatives Pricing | 1 | — (numerical methods, not ML prediction) |
| 10 | 2 | Causal Inference | 3 | — |
| 11 | 2 | Bayesian Methods | 3 | Calibration (ECE) |
| 12 | 2 | Fixed Income | 1 | — |
| 13 | 3 | Microstructure | 3 | — |
| 14 | 3 | Stat Arb & Regimes | 3 | — |
| 15 | 3 | RL for Finance | 3 | — |
| 16 | 3 | Deep Hedging | 3 | — |
| 17 | 3 | Generative & GNN | 3 | — |
| 18 | 3 | DeFi & Frontier | 1 | — (survey week, mostly conceptual) |

This schedule is a default. The expectations agent may override if a
week's content warrants it (e.g., if a Tier 1 week adds an ML prediction
component, bump the relevant section to Tier 2). Overrides must be justified.

---

## Tier 1: Data Foundations

Applies to ALL weeks. These rules govern data handling, bias prevention,
and basic evaluation — even when no ML model is trained.

---

### 1.1 Temporal splitting

Financial data has a time axis. Standard ML random splits destroy
temporal structure and create look-ahead bias.

**1. Default to walk-forward or expanding-window validation for any task
that predicts future values.**
- Walk-forward: train on [t₀, t₁], predict t₁₊₁, slide forward.
- Expanding: train on [t₀, t₁], predict t₁₊₁, expand t₁ forward.
- expectations.md specifies which.

**2. Random splits are acceptable ONLY for pure cross-sectional tasks
with no temporal prediction** (e.g., classifying stocks by sector from
a single-date snapshot). If in doubt, use temporal splits.

**3. Purging and embargo.** When the target uses forward-looking returns,
purge overlapping observations from training. For multi-day targets, add
embargo equal to target horizon. If expectations.md doesn't specify,
apply 1-period purge for any forward-return target.

**4. Fit preprocessing on training data only.** Scalers, PCA, any
transformation: fit on train, transform both. Per-month cross-sectional
z-scoring is acceptable pre-split (each month's stats are independent).
Document this reasoning when it applies.

---

### 1.2 Look-ahead bias checkpoints

At every data-touching step, verify no future information leaks into
current predictions. Common sources:
- End-of-period data not available at prediction time
- Target period included in feature computation
- Full-sample statistics used for normalization

When look-ahead bias is known and unavoidable (e.g., static fundamentals
per Known Constraints), acknowledge it in structured output and note that
results are an upper bound.

---

### 1.3 Data leakage taxonomy

Beyond look-ahead bias, four additional leakage categories require checks.

**1. Feature selection leakage.** If features are screened based on
correlation with the target, do this INSIDE the training window — not on
the full dataset. Feature importance or correlation-based selection must
be performed within each fold's training set.

**2. Survivorship bias.** Note in structured output if the universe is
subject to survivorship bias:
```
Universe: current S&P 500 tickers — subject to survivorship bias.
```
When results are surprisingly strong (IC > 0.07 for monthly equity
returns), investigate whether survivorship is a contributing factor.

**3. Data timing.** Verify all features are available at prediction time.
For quarterly data, ensure the quarter is complete before the prediction
date. For any lagged feature, confirm the lag is applied correctly.

**4. Point-in-time fundamentals.** When using data indexed by fiscal
period end (not publication date):

- Apply a conservative reporting lag: shift all fundamental dates forward
  by at least 90 days before merging with price data. This covers ~95%
  of 10-K filing delays.
- For sections where fundamentals are the dominant feature: run the
  analysis twice (no lag and +90 day lag) and report both. If IC drops
  substantially, the signal may be partly explained by look-ahead.
- State the mitigation in structured output:
  ```
  Point-in-time: +90 day reporting lag applied to all fundamental features.
  Restatement bias: not addressable with structured free data — residual PIT bias exists.
  ```
- See `data.md` for which data sources have PIT contamination and which
  alternatives are available.
- See Appendix A below for background on the two biases.

---

### 1.4 Data quality reporting

Before model training (or before the main analysis in non-ML sections),
print a data quality summary in structured output:

```
DATA QUALITY
  Panel: {N_stocks} stocks × {T_periods} periods
  Missing values: {pct_missing:.1%} overall
    Worst feature: {feature_name} ({pct:.1%} missing)
  Universe coverage: min {min_stocks} stocks ({date}),
    max {max_stocks} stocks ({date})
  ⚠ periods with <80% of max coverage: {count}
  Zero-return stocks (>30% zeros): {count} tickers [{list}]
```

This enables Steps 5 and 6 to distinguish data problems from model
problems. If `⚠ periods with <80%` appears, it may indicate universe
attrition or download failures.

---

### 1.5 IS/OOS labeling

In-sample vs out-of-sample must always be explicitly labeled.
- In plots: title or legend must distinguish IS vs OOS.
- In structured output: separate lines for IS and OOS metrics.
- Never present a single metric without specifying which it is.

---

### 1.6 Baseline comparison

Every model's value is relative to a simpler alternative. No model result
is presented in isolation. Common baselines: buy-and-hold, equal-weight,
linear model, naive forecast (previous period's value). expectations.md
specifies which baselines. If unspecified, include at least one naive
baseline and one linear model.

---

### 1.7 Statistical context

Report standard errors or confidence intervals when sample size allows.
At minimum: always report the number of observations underlying any metric.
- "IC = 0.03 (n=69 months)" is interpretable.
- "IC = 0.03" is not.

For time-series metrics (rolling IC, rolling Sharpe): report both mean
and standard deviation.

---

## Tier 2: ML Training Discipline

Applies when a section trains a model, fits a pipeline, or evaluates
predictive performance. Includes all Tier 1 rules.

---

### 2.1 Hyperparameter selection

Every model has hyperparameters. Using defaults without justification is
not acceptable.

**1. Minimum bar: a documented search.**
- Tree models (LightGBM, XGBoost): search at least `learning_rate`,
  `max_depth`/`num_leaves`, `n_estimators`. Small grid (3-5 values per
  param) with time-series CV.
- Neural nets: search at least `learning_rate`, `hidden_size`, `dropout`.
  Small grid or manual schedule with early stopping.
- Linear models (Ridge, Lasso): search regularization `alpha`.
  Log-spaced grid (e.g., `np.logspace(-4, 4, 9)`).

**2. Grid values must be justified.** Start from literature defaults or
data-driven reasoning. Example: "LightGBM `learning_rate` searched over
[0.01, 0.05, 0.1] — Ke et al. (2017) recommend 0.1 as default; lower
values included because dataset is small."

**3. Selection via time-series CV, not a single validation set.** Use the
same temporal strategy as final evaluation. If cost is prohibitive, use
coarser grid — do NOT fall back to random splits.

**4. "Default parameters" acceptable only with explicit citation.**
"LightGBM defaults are well-calibrated for medium-scale tabular data
(Ke et al., 2017); with 7 features and ~22K obs, defaults are appropriate."

**5. HP search tools.**
- sklearn-compatible models: `GridSearchCV`/`RandomizedSearchCV` with
  temporal CV splitter.
- DL, NLP fine-tuning, RL: Optuna (`optuna.create_study()`).
- Rule of thumb: sklearn API → GridSearchCV. Otherwise → Optuna.

**6. Document everything.** In RESULTS block: chosen HPs and selection
metric. In execution_log.md: what was searched, chosen, and why.

---

### 2.2 Training discipline

**1. Reproducibility.** Set random seeds for all randomness sources.
Single seed constant near top of file: `SEED = 42`.

**2. Early stopping is mandatory for iterative models.**
- Gradient boosting: `early_stopping_rounds` with held-out temporal
  validation set. Report rounds used vs. maximum.
- Neural nets: validation loss monitoring with patience. Report stopping
  epoch vs. maximum.
- Monitoring set must be temporal (most recent slice of training window).

**3. Regularization present and justified for neural networks.**
- At minimum: dropout OR weight decay. Both if >1 hidden layer.
- Justify: "Dropout = 0.3 because feature space is small (7 features)."
- Tree models: note which params serve as regularizers (max_depth,
  min_child_samples).

**4. Flag overfitting explicitly.** If train metric >> OOS metric (e.g.,
train R² > 2× test R²), print:
```
  ⚠ OVERFIT: train_r2={train:.4f} vs test_r2={test:.4f}
```
Informational — doesn't fail assertions — but must be visible in
run_log.txt so Steps 5 and 6 can address it.

---

### 2.3 Feature preprocessing

**1. Outlier treatment.** Cross-sectional features must be treated for
extreme values before model training. Preferred: rank-transform to
uniform [0, 1] per period (robust, distribution-free, subsumes
winsorization). Alternative: winsorize at 1st/99th percentile per period.
Document treatment in structured output. If deliberately skipped, note why.

**2. Cross-sectional normalization.** Features must be normalized within
each time period before modeling. Preferred: rank-transform per period
(same operation as outlier treatment — one step handles both).
Alternative: z-score per period (pair with winsorization). Per-period
normalization does not create temporal look-ahead. Document the method.

---

### 2.4 Signal significance

For any rolling IC or OOS R²: report t-statistic and p-value alongside
the point estimate.
- "IC = 0.03 (t=2.4, p=0.02, n=69)" — interpretable.
- "IC = 0.03 (n=69)" — NOT sufficient.

Use `ic_summary()` from `course/shared/metrics.py`. If t < 1.96, print:
```
  ⚠ WEAK SIGNAL: mean IC = {ic:.4f}, t = {t:.2f}, p = {p:.3f}
```
The `rolling_predict()` harness in `course/shared/evaluation.py` emits
this automatically. For custom loops, add the check manually.

---

### 2.5 Prediction quality

After generating predictions, verify the model differentiates stocks.
Use `prediction_quality()` from `course/shared/metrics.py`.
- If spread ratio < 0.10: model predicts unconditional mean. Print:
  ```
  ⚠ DEGENERATE PREDICTIONS: spread_ratio = {ratio:.3f}
  ```
- If unique_ratio < 0.05: tree has too few leaves. Note it.
- `rolling_predict()` checks automatically on last cross-section.

---

### 2.6 Baseline significance

Every model comparison must test statistical significance — not just
report "A is better than B."

Use `vs_naive_baseline()` from `course/shared/metrics.py` for paired
tests on aligned IC series.

Report with significance: "Ridge IC = 0.04 vs. naive IC = 0.03,
Δ = +0.01 (paired t = 0.8, p = 0.42 — not significant)."

An insignificant improvement is a legitimate result — it shows complexity
doesn't always help. But it must be framed as insignificant.

---

### 2.7 Multiple complementary metrics

No single number tells the full story.
- Return prediction: IC + ICIR + hit rate (or R² + MAE + direction).
- Classification: accuracy + precision + recall (or AUC + calibration).

expectations.md specifies the primary metric. Code should also compute
1-2 supporting metrics in structured output.

---

### 2.8 Comparison fairness — controlled invariants

When comparing two methods (models, validation schemes, feature sets,
cost assumptions), **vary only the variable under study. Hold everything
else constant.** A comparison that changes multiple things simultaneously
is uninterpretable — any observed difference could be caused by any of
the changes.

**Invariants to hold constant across method comparisons:**

| Invariant | What it means | Common violation |
|-----------|--------------|-----------------|
| Training data | Same rows used for fitting | One method trains on pre-test data only; the other trains on all non-test data — conflates algorithm effect with training set size |
| Test data | Same rows used for evaluation | Different fold boundaries produce different test sets — conflates method with test-set difficulty |
| Features | Same feature matrix | Adding/removing features between methods — conflates model with feature engineering |
| Preprocessing | Same transformations | Rank-transforming for one model but not another — conflates model with input distribution |
| Random seed | Same stochastic initialization | Different seeds add noise to the comparison |
| HP budget | Comparable search effort | Tuning one model extensively while using defaults for another — conflates model quality with tuning effort |

**When the comparison inherently requires different invariants:**
Some comparisons cannot hold everything constant — that's the point.
Walk-forward vs. k-fold produces different train/test partitions by
design. When a comparison must violate an invariant:

1. **State explicitly what differs and why.** In structured output:
   ```
   COMPARISON: walk-forward vs. purged k-fold
     Varied: validation scheme (train/test partition structure)
     Held constant: features, preprocessing, model, HP, seed
     Note: training set sizes differ by design — WF uses only
     prior data, PKF uses all non-test non-purged data.
   ```
2. **Acknowledge the confound.** If IC differs, note that the difference
   reflects both the purging effect AND the training set difference.
   Do not attribute the full delta to purging alone.
3. **Mitigate when possible.** If the comparison can be made fairer
   without defeating its purpose (e.g., ensuring both methods see the
   same total number of training observations via resampling), do so
   and report both the naive and controlled comparison.

**This rule applies to:**
- Model vs. model comparisons (Ridge vs. GBM vs. NN)
- Validation scheme comparisons (walk-forward vs. purged k-fold vs. CPCV)
- Feature set comparisons (with vs. without a feature group)
- Pre/post comparisons (gross vs. net returns, with vs. without TC)
- Any exercise where the teaching point depends on a delta between methods

---

## Tier 3: Finance-Aware Evaluation

Applies when evaluating strategies, production-relevant signals, or
portfolio-level performance. Includes all Tier 1 and Tier 2 rules.

---

### 3.1 Transaction costs and turnover

**1. Gross and net.** Any strategy metric (Sharpe, return, drawdown) must
be reported both gross and net of estimated transaction costs. Use the
cost from expectations.md (typically 5-50 bps one-way). Default: 10 bps
for S&P 500 large-caps.

**2. Turnover reporting.** For strategy sections that compute portfolio
turnover: report one-way turnover alongside returns. If one-way turnover
exceeds 50% per period, print:
```
  ⚠ HIGH TURNOVER: {turnover:.0%} one-way — cost drag ≈ {drag:.2%}/period at {cost_bps} bps
```

**3.** A strategy profitable gross but unprofitable net is a teaching
moment, not a failure. Frame it accordingly.

---

### 3.2 Sub-period performance degradation

Split OOS IC series into two halves. Report IC and t-stat separately for
each. If second-half IC < 50% of first-half IC, print:
```
  ⚠ SIGNAL DECAY: IC 1st half = {:.4f} (t={:.2f}), 2nd half = {:.4f} (t={:.2f})
```
This reveals regime dependence and signal non-stationarity. Walk-forward
IC naturally shows this through the time series — also plot it.

---

### 3.3 Return distribution characterization

For any strategy-level analysis reporting Sharpe ratio: also report
skewness and excess kurtosis of the return series.
- If |skewness| > 1 or excess_kurt > 3: note that Sharpe understates
  tail risk.
- When computing `deflated_sharpe_ratio()`, pass actual skew and kurtosis
  — the zero defaults assume normality.
- Minimum reporting:
  ```
  Sharpe = {:.2f}, skew = {:.2f}, excess_kurt = {:.2f}
  ```

---

### 3.4 Prediction horizon justification

When predicting future returns, justify the chosen target horizon.
Shorter horizons: higher IC but more microstructure noise. Longer: less
predictable but cleaner signal. If fixed by curriculum, note: "Horizon
fixed by curriculum; in production, test multiple horizons."

---

### 3.5 Multiple testing awareness

If multiple models or feature subsets are tested during development, log
the total number of hypotheses tested. When reporting the best model's
metrics, note N_trials. Use `deflated_sharpe_ratio()` from
`course/shared/metrics.py` when comparing strategy Sharpe ratios across
multiple configurations.

---

### 3.6 Annualization discipline

Verify `periods_per_year` matches actual return frequency when computing
annualized metrics. Default in `sharpe_ratio()` is 12 (monthly).
- Daily returns → 252
- Weekly → 52
- Monthly → 12

Mismatched frequency inflates or deflates Sharpe by up to 3.5x.

---

### 3.7 Non-stationarity awareness

Financial relationships change over time. When results vary across
sub-periods (IC positive in one half, negative in another), note this.
Don't just report the average — the variation IS the insight.

---

### 3.8 Low signal-to-noise awareness

Cross-sectional IC of 0.02-0.05 is meaningful for equity returns. Do not
over-tune to chase higher IC — that's overfitting. If IC > 0.10 on
monthly equity returns with free data, be suspicious — investigate
look-ahead or data leakage. Reference: Gu-Kelly-Xiu (2020) report OOS
R² ~0.4% monthly (IC ~0.03-0.05) using 94 features on full CRSP.

---

## Advanced Checks (Optional)

These checks are not part of any tier. Use them when expectations.md
specifically requests them or when the analysis naturally warrants them.

**IC autocorrelation.** If IC series has lag-1 autocorrelation > 0.2, the
t-stat from `ic_summary()` overstates significance. Report lag-1 rho in
structured output: `IC lag-1 autocorrelation = {rho:.2f}`. For
interpretation: effective n ≈ N × (1-ρ)/(1+ρ).

**Calibration.** For models outputting probabilities: compute ECE via
`calibration_error()` from `course/shared/metrics.py`. Plot calibration
curve. Relevant primarily for Week 11 (Bayesian methods).

**Ensemble disagreement.** For sections training multiple models: compute
pairwise prediction disagreement on last cross-section:
`mean(|pred_A - pred_B|) / std(actuals)`. If > 0.15, note in output.

**Conditional performance by regime.** Split IC by market condition
(high-vol vs low-vol, or bull vs bear). Note if signal is regime-dependent.

**Portfolio concentration.** For multi-asset portfolios: compute Herfindahl
index of weights. If > 0.15, the portfolio is concentrated — note it.

**Rolling parameter stability.** For linear models in walk-forward: track
top-3 coefficient signs across windows. If any flips in >30% of windows:
```
  ⚠ UNSTABLE COEF: {feature} flips sign in {pct:.0%} of windows
```

**Feature importance on OOS data.** For tree and neural models: compute
importance on last OOS cross-section (permutation importance preferred).
Print top-5 features + importance scores in structured output.

---

## Anti-Patterns

Common agent failure modes. **Scan this table before writing any code file.**

| # | What the agent does | Why it's wrong | What to do instead |
|---|---|---|---|
| 1 | Uses model defaults without comment | Students learn defaults are always fine | Search or cite — §2.1 rule 4 |
| 2 | `scaler.fit_transform(all_data)` before split | Leaks test distribution into training | `scaler.fit(train)` then `scaler.transform(test)` |
| 3 | Single random 80/20 split on time-series data | Destroys temporal structure, inflates metrics | Walk-forward or expanding window |
| 4 | Neural net without early stopping | Trains to convergence = trains to memorization | Patience + temporal validation monitoring |
| 5 | "Model A is better" from one metric | Incomplete, possibly misleading | Multiple metrics + significance test |
| 6 | Reports IC = 0.03 without sample size | Uninterpretable | Always report n: "IC = 0.03 (n=69)" |
| 7 | Reports IC without t-stat | Can't tell if signal is real | Use `ic_summary()` — report t-stat and p-value |
| 8 | "Outperforms baseline" without paired test | Could be noise | `vs_naive_baseline()` for paired t-test |
| 9 | Model outputs near-constant predictions | Predicting the mean — learned nothing | Check `prediction_quality()` spread ratio |
| 10 | Trains and evaluates on full dataset | No OOS evaluation | Always separate IS/OOS, always label |
| 11 | Ignores transaction costs for strategy | Gross Sharpe 1.5, net Sharpe -0.2 | Report both gross and net |
| 12 | Daily returns to `sharpe_ratio()` with default `periods_per_year=12` | Sharpe inflated ~3.5x | Match frequency: daily→252, monthly→12, weekly→52 |
| 13 | Raw cross-sectional features fed to model | Scale differences dominate — market cap in billions vs P/E in single digits | Rank-transform or z-score per period |
| 14 | No data quality report before modeling | Steps 5/6 can't distinguish data vs model problems | Print DATA QUALITY block (§1.4) |
| 15 | Sharpe reported without skew/kurtosis | Misleading for non-normal returns | Report skew and excess_kurt alongside Sharpe |
| 16 | Fundamentals at fiscal period end date | Using data before it was publicly available | Apply +90 day reporting lag before merging with prices — see §1.3 rule 4 |
| 17 | Features screened on full dataset | Selection leakage | Screen within each fold's training set |
| 18 | Does not note survivorship bias | Results silently inflated | State universe bias in structured output |

---

## Shared Infrastructure & First-Introduction Principle

**Shared infra:** Read `course/shared/` modules before writing utility
code. Walk-forward splitters, IC computation, evaluation harnesses live
there to avoid per-week reimplementation.

**First-introduction rule:** When a week first introduces a concept
(e.g., Week 4 introduces signal significance), implement from scratch —
the implementation IS the teaching. In later weeks, import from shared/.
The expectations.md "Teaches" annotation marks first introductions.

---

## Scope & Boundaries

This guide covers ML engineering quality. It does NOT cover:

| Excluded | Where it lives |
|---|---|
| What tasks to assign | task_design.md |
| What metrics to expect | expectations.md |
| Code file format, CELL markers, verification blocks | code_verification.md |
| Teaching strategy, narrative, prose | Steps 6 and 7 |

---

## Appendix A: Point-in-Time Fundamentals

> Reference material for §1.3 rule 4. This appendix explains the two
> biases and the defensive strategies when data is PIT-contaminated.
> For which specific data sources are affected and what alternatives
> exist, see `data.md`.

### Two distinct biases

1. **Reporting-lag bias.** Fundamental data indexed by fiscal period end
   (e.g., Q4 ending Dec 31) creates look-ahead when the filing arrives
   weeks or months later. Median 10-K filing delay for S&P 500 is ~55
   days, with a tail extending to 90+ days.

2. **Restatement bias.** Companies restate financials — sometimes
   materially. Free sources silently overwrite originals with the latest
   revision. A model trained on restated data sees a history no investor
   had.

### Defensive strategies (when data is PIT-contaminated)

Apply these in order of priority:

| Strategy | What to do | What it addresses |
|----------|-----------|-------------------|
| **+90 day lag** (always apply) | Shift all fundamental dates forward by 90 days before merging with price data. | Reporting-lag bias. Covers ~95% of 10-K filing delays. |
| **Lag sensitivity analysis** (when fundamentals dominate) | Run with no lag and +90 day lag, report both. | Quantifies maximum PIT exposure for the specific analysis. |
| **Annual fundamentals only** | Prefer annual (10-K) over quarterly (10-Q) data. | Reduces restatement frequency (annual data restated less often). |
| **Structured output disclosure** (always) | Print the PIT mitigation applied and residual bias in output. | Transparency for downstream agents and reviewers. |

**The one thing no lag fixes:** Restatement bias. Structured free
fundamental data reflects latest restatements, not as-originally-reported
values. This residual bias cannot be eliminated from structured sources
without a commercial PIT database. (As-originally-reported values may
exist in unstructured sources — see `data.md` for alternatives.)
Always acknowledge the residual bias in structured output.
