# Quality Audit Report: Weeks 1-4 Notebooks

**Date:** 2026-02-12
**Audited against:** `notebook_creation_guide.md`, `writing_guidelines.md`, per-week `README.md`

---

## Scores Overview

| | Lecture | Seminar | Homework |
|---|---|---|---|
| **Week 1** | 8.5 | 7.5 | 7.0 |
| **Week 2** | 8.0 | 7.5 | 7.5 |
| **Week 3** | 8.0 | 7.5 | 7.0 |
| **Week 4** | 8.5 | 7.5 | 7.0 |

---

## Systemic Issues (Recurring Across Multiple Weeks)

### 1. Insufficient Workspace Cells (ALL seminars & homeworks)
Every seminar and homework has only **1 workspace cell per exercise/deliverable**. The guide requires 2-4 (seminar) or 2-5 (homework). This is the most pervasive violation -- affects all 8 non-lecture notebooks.

### 2. Code Cell Line Limits Exceeded
- **W1 Lecture**: All within 15 lines
- **W2 Lecture**: 3 cells at 16 lines (1 over preferred limit)
- **W2 HW**: **5 cells exceed 25-line limit** (worst: 32 lines)
- **W3**: All within limits
- **W4 Lecture**: 1 cell at 16 lines
- **W4 HW**: 2 cells exceed 25 lines (27, 29 lines)

### 3. Missing README Stories/Content
| Week | Missing Stories |
|------|---------------|
| W1 | Flash Crash of 2010, XIV blowup (2018), dark pools |
| W2 | LTCM collapse, square-root-of-time rule |
| W3 | Markowitz retirement story overlap (appears in all 3 notebooks instead of just one) |
| W4 | "Death of the value premium" |

### 4. Voice Formality (W1 HW specifically)
W1 homework has ~15+ instances of "Let us", "you will", "it is" instead of contractions. Reads significantly more formal than the Taleb+Lewis voice requires.

### 5. Warning Suppression (W4 seminar + HW)
Both W4 seminar and homework have `warnings.filterwarnings('ignore', category=FutureWarning)` -- violates the "never suppress warnings globally" rule.

### 6. Non-Overlap Principle Violations
- **W3**: Markowitz retirement story appears in all three notebooks
- **W3**: HRP vs. Markowitz vs. 1/N comparison is essentially the same exercise in seminar and homework
- **W4**: Feature engineering pipeline code is highly similar across all three notebooks

---

# WEEK 1: Markets & Data

## Lecture (`lecture.ipynb`) -- 8.5/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 64.4% markdown (req: >=60%) |
| Cell Count | **PASS** | 59 cells (req: 45-70) |
| Code Cell Lines | **PASS** | All within 15-line limit |
| First Cell | **PASS** | Markdown title |
| Second Cell | **PASS** | Code imports |
| Monkey-patching | **PASS** | None present |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **MINOR VIOLATION** | Cell 36: parenthetical narrative in print ("NOT $100!", "says breakeven -- WRONG", "says small loss -- CORRECT"). Cell 19: "EMPTY -- erased from history" is a narrative flourish in print. |
| Transition Cells | **PASS** | All transitions follow Interpret-Connect-Tension pattern |
| "Did You Know?" | **PASS** | 4 moments, all in blockquote format |
| Workspace Cells | **N/A** | Not required for lectures |

### README Alignment

| Concept | Status |
|---------|--------|
| Order book structure | COVERED |
| Bid-ask spread & hidden costs | COVERED |
| Survivorship bias | COVERED |
| Returns math (simple vs. log) | COVERED |
| Alternative bars (volume, dollar) | COVERED |
| Transaction costs | PARTIAL -- formula and narrative present but no code cell computes cost for multiple stocks |
| Data pipeline (DataLoader class) | PARTIAL -- skeleton shown (acceptable per guide's "no reusable classes in lecture" rule) |

### Missing Stories
- **Flash Crash of 2010** -- not mentioned anywhere
- **XIV blowup (2018)** -- not mentioned
- **Dark pools** -- zero coverage despite being listed as a key concept

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| L1 | print() prohibition | Minor | Cells 19, 36 |
| L2 | README stories | Medium | Flash Crash and XIV missing |
| L3 | Concept coverage | Minor | Transaction cost computation is narrative-only |
| L4 | Dark pools | Minor | Zero coverage |

---

## Seminar (`seminar.ipynb`) -- 7.5/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 59.0% markdown (req: >=45%) |
| Cell Count | **PASS** | 39 cells (req: 25-45) |
| Code Cell Lines | **PASS** | All within 25-line limit |
| First Cell | **PASS** | Markdown title |
| Second Cell | **PASS** | Code imports |
| Monkey-patching | **PASS** | None present |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **PASS** | All print() used for data output only |
| Workspace Cells | **PASS** | Present at cells 4, 12, 22, 30 with `# YOUR WORKSPACE` comments |
| Solution Separators | **PASS** | Correct `### > Solution` format |
| Exercise Questions | **PASS** | All framed as genuine questions with stakes |

### README Alignment

| Exercise | Status |
|----------|--------|
| Ex1: Fat-Tail Safari (10 ETFs) | COVERED |
| Ex2: Dollar Bars at Scale (10 ETFs) | COVERED |
| Ex3: Survivorship Bias quantification | COVERED |
| Ex4: Transaction Cost impact on MA strategy | COVERED |

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| S1 | Voice/Forbidden pattern | Minor | Cell 0: "These are the questions we'll answer today" -- announcing structure |

---

## Homework (`hw.ipynb`) -- 7.0/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **VIOLATION** | Cells 5-6: two consecutive workspace code cells outside a solution block |
| Prose-to-Code Ratio | **PASS** | 58.8% markdown (req: >=40%) |
| Cell Count | **PASS** | 68 cells (req: 35-75) |
| Code Cell Lines | **PASS** | All within 25-line limit |
| First Cell | **PASS** | Markdown title |
| Monkey-patching | **PASS (allowed)** | Used correctly for DataLoader class build |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **PASS** | All print() for computed data |
| Workspace Cells | **PASS** | Present at cells 5/6, 20, 30, 42 |
| Solution Separators | **PASS** | Correct format |

### README Alignment

| Deliverable | Status |
|------------|--------|
| 1. Download 200 equities, QQ-plots, kurtosis | COVERED |
| 2. Implement volume + dollar bars | **PARTIAL -- volume bars missing, only dollar bars implemented** |
| 3. 3+ data quality issues | COVERED |
| 4. DataLoader class | COVERED (but no disk caching) |
| 5. 1-page data quality report | COVERED (but pre-written discoveries) |

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| H1 | No-Silent-Code | Minor | Cells 5-6: consecutive workspace cells |
| H2 | Voice/Contractions | **Medium** | ~15+ instances of "Let us", "you will", "it is" throughout |
| H3 | Volume bars missing | **Medium** | README requires both volume and dollar bars |
| H4 | Caching | Minor | DataLoader has no disk-caching mechanism |
| H5 | Pre-written discoveries | Minor | Sample report pre-writes generic findings |

---

# WEEK 2: Time Series Properties & Stationarity

## Lecture (`lecture.ipynb`) -- 8.0/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 63.9% markdown (req: >=60%) |
| Cell Count | **PASS** | 61 cells (req: 45-70) |
| Code Cell Lines | **VIOLATION** | Cells 26, 43, 56: 16 lines each (limit: 15) |
| First Cell | **PASS** | Markdown title with epigraph |
| Monkey-patching | **PASS** | None present |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **PASS** | No print() calls |
| Transition Cells | **PASS** | All substantive with 2-3+ sentences |
| "Did You Know?" | **PASS** | 4 moments: Fama-Shiller Nobel, M4 Competition, Engle Nobel/GARCH, Lopez de Prado career |
| ACF Demo | **Minor deviation** | Uses absolute returns instead of squared returns per README |

### README Alignment

| Concept | Status |
|---------|--------|
| Stationarity (ADF/KPSS) on SPY | COVERED |
| Autocorrelation & market efficiency | COVERED |
| Classical time series (ARIMA) | COVERED |
| GARCH(1,1) volatility modeling | COVERED |
| Fractional differentiation (FFD) | COVERED |
| Finding optimal d* | COVERED |
| Section 7: "Putting It Together" pipeline | **PARTIAL** -- skeleton shown, not full standalone section |

### Missing Stories
- **LTCM collapse (1998)** -- not mentioned
- **Square-root-of-time rule and its failures** -- not mentioned

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| L1 | Code cell line limit | Low | Cells 26, 43, 56: 16 lines each |
| L2 | README stories | Medium | LTCM collapse missing |
| L3 | README stories | Medium | Square-root-of-time rule missing |
| L4 | README alignment | Medium | No full "Putting It Together" section |
| L5 | ACF demo | Low | Uses absolute returns instead of squared returns |
| L6 | Data download | Minor | Two separate download cells instead of one |

---

## Seminar (`seminar.ipynb`) -- 7.5/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 60.5% markdown (req: >=45%) |
| Cell Count | **PASS** | 43 cells (req: 25-45) |
| Code Cell Lines | **VIOLATION** | Cell 8: 26 lines (limit: 25) |
| Monkey-patching | **PASS** | None present |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **PASS** | No print() calls |
| Workspace Cells | **PASS** | Present at cells 6, 16, 28, 36 |
| Solution Separators | **PASS** | Correct format |

### README Alignment

| Exercise | Status |
|----------|--------|
| Ex1: Stationarity Testing Marathon | COVERED (uses BTC-USD instead of JNJ -- minor) |
| Ex2: Cross-Sectional d* Map | COVERED |
| Ex3: GARCH(1,1) Fitting | **PARTIAL -- missing QLIKE loss evaluation, 1-day-ahead forecasting, held-out evaluation** |
| Ex4: GARCH Variants | **PARTIAL -- missing GARCH(2,1), missing out-of-sample MSE/QLIKE comparison** |

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| S1 | Code cell line limit | Low | Cell 8: 26 lines |
| S2 | README Exercise 3 | **High** | Missing QLIKE loss, forecasting, held-out evaluation |
| S3 | README Exercise 4 | **Medium** | Missing GARCH(2,1) model |
| S4 | README Exercise 4 | **High** | Missing out-of-sample evaluation metrics |

---

## Homework (`hw.ipynb`) -- 7.5/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 58.3% markdown (req: >=40%) |
| Cell Count | **PASS** | 72 cells (req: 35-75) |
| Code Cell Lines | **VIOLATION** | 5 cells exceed 25 lines: cells 28, 30, 37, 45 (28 lines each), cell 59 (32 lines) |
| Monkey-patching | **PASS (allowed)** | Used correctly for FractionalDifferentiator class |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **PASS** | No print() calls |
| Workspace Cells | **PASS** | Present at cells 10, 24, 35, 43, 53 |
| Solution Separators | **PASS** | Correct format |

### README Alignment

| Deliverable | Status |
|------------|--------|
| 1. Find d* for 50 stocks | COVERED (missing "correlation at d*" column) |
| 2. Ridge regression with 3 feature sets | COVERED |
| 3. Compare R-squared across feature sets | COVERED |
| 4. GARCH(1,1) for 50 stocks | COVERED |
| 5. FractionalDifferentiator class | **PARTIAL -- does not inherit BaseEstimator/TransformerMixin, missing get_params()** |

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| H1 | Code cell line limit | Medium | Cells 28, 30, 37, 45: 28 lines each |
| H2 | Code cell line limit | **High** | Cell 59: 32 lines |
| H3 | README D1 | Low | Missing "correlation at d*" column |
| H4 | README D5 | **Medium** | Class doesn't inherit BaseEstimator/TransformerMixin |
| H5 | README D5 | **Medium** | Missing get_params() method |

---

# WEEK 3: Portfolio Theory, Factor Models & Risk

## Lecture (`lecture.ipynb`) -- 8.0/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 64.4% markdown (req: >=60%) |
| Cell Count | **PASS** | 59 cells (req: 45-70) |
| Code Cell Lines | **PASS** | All within 15-line limit |
| Monkey-patching | **PASS** | None present |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **PASS** | No print() calls in any notebook |
| "Did You Know?" | **PASS** | 5 moments found in blockquote format |
| Forbidden Patterns | **PASS** | No instances of "In this section", "Let's explore", etc. |

### README Alignment

| Concept | Status |
|---------|--------|
| Mean-Variance Optimization | COVERED |
| CAPM/Fama-French | **Uses 3-factor instead of 5-factor model** |
| Risk Metrics (VaR, CVaR, Sortino) | COVERED |
| Fundamental Law of Active Management | COVERED |
| Covariance Estimation / RMT / Marchenko-Pastur | COVERED |
| HRP | COVERED |
| Transaction Costs | COVERED |
| QuantStats tear sheet demo | **MISSING** (README explicitly includes it) |

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| L1 | "Did You Know?" format | Minor | All use boxed blockquote format (guide prefers woven into narrative) |
| L2 | README alignment | Medium | Missing QuantStats tear sheet demo |
| L3 | README alignment | Medium | Uses 3-factor instead of 5-factor model |
| L4 | README alignment | Minor | Stock universe consistently undersized (10 vs. 20/50/100 in README) |

---

## Seminar (`seminar.ipynb`) -- 7.5/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | ~49% markdown (req: >=45%) |
| Cell Count | **PASS** | 43 cells (req: 25-45) |
| Code Cell Lines | **PASS** | All within 25-line limit |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| Workspace Cells | **Only 1 per exercise** (guide says 2-4) |
| Non-Overlap | **VIOLATION** | Markowitz retirement story recycled from lecture |

### README Alignment

| Exercise | Status |
|----------|--------|
| Ex1: FF 5-factor regressions for 20 stocks | COVERED |
| Ex2: Eigenvalue/MP analysis, denoise covariance | **PARTIAL -- missing denoised portfolio construction** |
| Ex3: HRP vs. Markowitz vs. 1/N rolling OOS | COVERED (uses 5-year not 10-year period) |

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| S1 | Workspace cells | Medium | Only 1 per exercise (guide says 2-4) |
| S2 | Non-Overlap | Medium | Markowitz retirement story recycled from lecture |
| S3 | README alignment | Medium | Exercise 2 missing denoised portfolio construction |
| S4 | Insight grounding | Minor | Exercise 1 insight doesn't reference specific output numbers |

---

## Homework (`hw.ipynb`) -- 7.0/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 54.0% markdown (req: >=40%) |
| Cell Count | **PASS** | 63 cells (req: 35-75) |
| Code Cell Lines | **PASS** | All within 25-line limit |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **PASS** | No print() calls |
| Workspace Cells | **VIOLATION** | Only 1 total workspace cell in entire notebook (cell 24, for Deliverable 2 only) |
| Solution Separators | **VIOLATION** | Only Deliverable 2 has separator. Deliverables 3-6 have NO separators |
| Pre-written Discoveries | **VIOLATION** | Cell 61 fully pre-writes student findings |

### README Alignment

| Deliverable | Status |
|------------|--------|
| 1. 100 stocks + FF5 + alignment | COVERED |
| 2. Fama-MacBeth regressions | COVERED (missing R-squared per factor) |
| 3. Three portfolios (1/N, Markowitz, HRP) | COVERED |
| 4. Full performance evaluation | COVERED |
| 5. Transaction cost analysis at 10 bps | COVERED |
| 6. QuantStats tear sheets | COVERED |

### Cross-Notebook Overlap Issues
- **Markowitz retirement story**: Appears in lecture (cell 14), seminar (cell 40), AND homework (cell 1) -- 3rd repetition
- **HRP vs. Markowitz vs. 1/N**: Seminar Exercise 3 and Homework Deliverables 3-5 are essentially the same comparison at different scale (40 vs. 100 stocks). Violates "homework should integrate, not scale up" principle
- **DeMiguel "1/N beats optimization"**: Referenced in all three notebooks with same insight

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| H1 | Solution separators | **High** | Missing for Deliverables 3, 4, 5, 6 |
| H2 | Workspace cells | **High** | Only 1 in entire notebook |
| H3 | Non-Overlap | **High** | Markowitz retirement story (3rd repetition) |
| H4 | Pre-written discoveries | Medium | Cell 61 pre-writes findings |
| H5 | Insight overlap | Medium | Aha moments repeat seminar/lecture insights |
| H6 | README alignment | Minor | Fama-MacBeth missing R-squared per factor |
| H7 | Exercise overlap | **High** | Deliverables 3-5 essentially repeat Seminar Ex3 at larger scale |

---

# WEEK 4: Cross-Sectional Return Prediction -- Linear Models

## Lecture (`lecture.ipynb`) -- 8.5/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 62.9% markdown (req: >=60%) |
| Cell Count | **PASS** | 70 cells (req: 45-70, at exact upper boundary) |
| Code Cell Lines | **VIOLATION** | Cell 55: 16 lines (limit: 15) |
| Monkey-patching | **PASS** | None present |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **PASS** | No print() calls anywhere |
| Transition Cells | **PASS** | Most follow Interpret-Connect-Tension well |
| "Did You Know?" | **PASS** | 4 moments: Factor zoo (Harvey), rank normalization, IC calibration + AQR, Kenneth French library |

### Writing Quality
Consistently strong voice:
- "An ML engineer starts thinking about LSTMs, attention mechanisms, and historical price sequences. A quant researcher says: *wrong question.*"
- "If you use standard 5-fold cross-validation on financial data, you will get a beautiful, impressive, and completely fake result."
- "Every impressive financial ML result that used shuffled CV is lying -- not maliciously, but structurally."

### README Alignment

| Concept | Status |
|---------|--------|
| Cross-sectional prediction framing | COVERED |
| Feature engineering | COVERED |
| OLS/Ridge/Lasso/Elastic Net | COVERED |
| Expanding-window CV | COVERED |
| GKX framework | COVERED |
| Quantile portfolio analysis | COVERED |
| Transaction cost integration | COVERED |

### Missing Stories
- **"Death of the value premium"** -- HML's post-2007 decline never mentioned

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| L1 | Code cell line limit | Minor | Cell 55: 16 lines |
| L2 | README stories | Medium | "Death of the value premium" missing |
| L3 | Transition cell | Minor | Cell 48 is borderline caption-only (2 sentences) |

---

## Seminar (`seminar.ipynb`) -- 7.5/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 56.1% markdown (req: >=45%) |
| Cell Count | **PASS** | 41 cells (req: 25-45) |
| Code Cell Lines | **PASS** | All within 25-line limit |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| Workspace Cells | **Only 1 per exercise** (guide says 2-4) |
| Warning Suppression | **VIOLATION** | `warnings.filterwarnings('ignore', category=FutureWarning)` in Cell 1 |

### README Alignment

| Exercise | Status |
|----------|--------|
| Ex1: Building the Feature Matrix | COVERED (**missing RSI and turnover features**) |
| Ex2: Regularization Path | COVERED |
| Ex3: Leakage Trap (5-fold vs. expanding) | COVERED |
| Ex4: IC Analysis by Feature | COVERED |

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| S1 | Workspace cells | Medium | Only 1 per exercise |
| S2 | README alignment | Medium | Missing RSI feature |
| S3 | README alignment | Medium | Missing turnover feature |
| S4 | Warning suppression | Minor | Global FutureWarning suppression |
| S5 | Feature overlap | Minor | Feature engineering very similar to lecture demo |

---

## Homework (`hw.ipynb`) -- 7.0/10

### Structural Rules

| Rule | Status | Details |
|------|--------|---------|
| No-Silent-Code | **PASS** | Zero consecutive code cells |
| Prose-to-Code Ratio | **PASS** | 51.2% markdown (req: >=40%) |
| Cell Count | **VIOLATION** | 80 cells (req: 35-75, exceeds by 5) |
| Code Cell Lines | **VIOLATION** | Cell 36: 27 lines, Cell 52: 29 lines (limit: 25) |

### Content Rules

| Rule | Status | Details |
|------|--------|---------|
| print() Prohibition | **PASS** | No print() calls |
| Workspace Cells | **Only 1 per deliverable** (guide says 2-5) |
| Warning Suppression | **VIOLATION** | `warnings.filterwarnings('ignore', category=FutureWarning)` |

### README Alignment

| Deliverable | Status |
|------------|--------|
| 1. Feature matrix (20+ features, 200+ stocks) | **PARTIAL -- only 16 features (req: 20+)** |
| 2. Expanding-window CV | COVERED |
| 3. Model comparison (OLS, Ridge, Lasso, ElasticNet) | COVERED (**but hyperparameter tuning not implemented -- VAL_MONTHS=12 is dead code**) |
| 4. Long-short portfolio (top/bottom decile) | COVERED |
| 5. Transaction costs (10 bps) | COVERED |
| 6. Alphalens tear sheet | **PARTIAL -- manual reimplementation instead of using alphalens library** |

### Violations Summary

| # | Rule | Severity | Location |
|---|------|----------|----------|
| H1 | Cell count | Medium | 80 cells (max 75) |
| H2 | Code cell lines | Minor | Cell 36: 27 lines |
| H3 | Code cell lines | Medium | Cell 52: 29 lines |
| H4 | Workspace cells | Medium | Only 1 per deliverable |
| H5 | Feature count | **High** | 16 features instead of required 20+ |
| H6 | Hyperparameter tuning | **High** | VAL_MONTHS=12 defined but never used (dead code) |
| H7 | Alphalens | Medium | Library not used; components built manually |
| H8 | Warning suppression | Minor | Global FutureWarning suppression |
| H9 | Feature overlap | Minor | Same features computed in lecture, seminar, AND homework |

---

# Consolidated Fix List (Priority Order)

## HIGH Priority (Must Fix)

| # | Week | Notebook | Issue |
|---|------|----------|-------|
| 1 | W3 | HW | Add solution separators for Deliverables 3-6 |
| 2 | W3 | ALL | Remove Markowitz retirement story from seminar + homework (keep in lecture only) |
| 3 | W3 | HW | Differentiate Deliverables 3-5 from Seminar Ex3 (should integrate, not repeat at scale) |
| 4 | W4 | HW | Add 4+ features to reach 20+ requirement |
| 5 | W4 | HW | Implement hyperparameter tuning (remove dead VAL_MONTHS code or use it) |
| 6 | W2 | Seminar | Add QLIKE loss evaluation + 1-day-ahead forecasting to Exercise 3 |
| 7 | W2 | Seminar | Add GARCH(2,1) + out-of-sample evaluation to Exercise 4 |
| 8 | W2 | HW | Make FractionalDifferentiator inherit BaseEstimator/TransformerMixin + add get_params() |
| 9 | W1 | HW | Add volume bars (README requires both volume + dollar bars) |
| 10 | W1 | HW | Fix voice: replace all "Let us"/"you will"/"it is" with contractions |

## MEDIUM Priority (Should Fix)

| # | Week | Notebook | Issue |
|---|------|----------|-------|
| 11 | ALL | Sem+HW | Add 2-4 workspace cells per exercise (currently all have only 1) |
| 12 | W2 | HW | Break up 5 over-length code cells (esp. cell 59 at 32 lines) |
| 13 | W4 | HW | Reduce cell count from 80 to under 75 |
| 14 | W4 | HW | Break up cells 36 (27 lines) and 52 (29 lines) |
| 15 | W1 | Lecture | Add Flash Crash and XIV blowup stories |
| 16 | W2 | Lecture | Add LTCM collapse + square-root-of-time stories |
| 17 | W4 | Lecture | Add "death of the value premium" story |
| 18 | W3 | Lecture | Upgrade to 5-factor model (add RMW/CMA) |
| 19 | W3 | Lecture | Add QuantStats tear sheet demo |
| 20 | W3 | Seminar | Add denoised portfolio construction to Exercise 2 |
| 21 | W4 | Seminar | Add RSI + turnover features to Exercise 1 |
| 22 | W4 | HW | Use actual alphalens library for Deliverable 6 |
| 23 | W3 | HW | Make discoveries a template, not pre-written conclusions |

## LOW Priority (Nice to Fix)

| # | Week | Notebook | Issue |
|---|------|----------|-------|
| 24 | W2 | Lecture | 3 code cells at 16 lines (split to stay under 15) |
| 25 | W4 | Lecture | Cell 55 at 16 lines (split) |
| 26 | W2 | Lecture | Use squared returns for ACF (not absolute) |
| 27 | W2 | Lecture | Consolidate data downloads into single cell |
| 28 | W4 | Sem+HW | Replace global warning suppression |
| 29 | W1 | HW | Add disk caching to DataLoader |
| 30 | W1 | HW | Separate consecutive workspace cells 5-6 with markdown |
| 31 | W1 | Lecture | Add code demo for transaction cost computation |
| 32 | W1 | Lecture | Move print() narrative annotations to markdown |
| 33 | W1 | Seminar | Tighten opening paragraph |
| 34 | W1 | Lecture | Add dark pools coverage |

---

# Patterns to Fix Before Creating More Weeks

These are lessons that should be internalized for all future notebook creation:

1. **Always create 2-4 workspace cells per exercise** with scaffolding comments
2. **Keep code cells strictly under limits** (15 lecture, 25 seminar/HW)
3. **Check README stories list** and ensure ALL are woven in
4. **Use contractions consistently** -- "you'll", "it's", "let's", never "you will", "it is", "let us"
5. **Never suppress warnings globally**
6. **Don't repeat stories across lecture/seminar/homework** -- each fact appears once
7. **Homework must integrate, not scale up** -- it shouldn't be "seminar but with more stocks"
8. **Always include solution separators** for every deliverable in homework
9. **Don't pre-write student discoveries** -- provide templates with headings, not pre-filled answers
10. **Check feature/deliverable counts** against README requirements before finishing
