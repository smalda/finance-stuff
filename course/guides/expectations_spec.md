# Expectations Specification — Step 3 Guide

> **Consumer:** Expectations agent (Step 3). This document bridges the ideal (blueprint) and the real (code). It assesses what our sandbox data can deliver, assigns data to every section and exercise, sets acceptance criteria, and identifies what only code can answer. Once approved, expectations.md is immutable.

---

## Inputs

In addition to `common.md`, `task_design.md`, and `rigor.md`:
- **`blueprint.md`** from Step 2 — the teaching plan (structure, exercises, narrative arc)
- **`research_notes.md`** from Step 1 — data constraints, library findings, domain context

**`rigor.md`** defines the ML engineering quality standard that code must meet. Read it so your methodology specifications and code probes reflect proper ML practice (temporal validation, hyperparameter search expectations, preprocessing discipline, tiered rigor).

## Outputs

**`expectations.md`** in the week folder (immutable once approved).

---

## Before Writing — Gap Scan → Search → Probe → Risk Analysis → Write

The research agent (Step 1) gathers broad domain knowledge. But it doesn't know the blueprint yet — it can't anticipate which specific analyses you'll need benchmarks for, or which data constraints matter for a particular exercise. You fill these gaps with targeted web searches and code probes before writing.

**The process has five phases:**

### Phase 1: Gap Scan

Read every blueprint section, exercise, and deliverable. For each, ask: "Can I set acceptance criteria and a production benchmark using only research_notes.md?" When the answer is no, record the gap and how to fill it:

```
GAP SCAN

Search gaps (fill via web):
- "S3 builds SMB from S&P 500 only — what does SMB-Ken French correlation
  look like with a large-cap-only universe? Need published result."
- "Ex2 runs Fama-MacBeth on 5 characteristics — what risk premia magnitudes
  are typical for a 10-year S&P 500 sample?"

Probe gaps (fill via code):
- "How many S&P 500 tickers have complete 10-year daily histories via yfinance?"
- "Does yfinance provide operating income reliably for 200+ stocks?"
- "D1 uses yfinance fundamentals going back 10 years — how far back does
  quarterly income_stmt data actually go?"
```

Don't search or run code during the scan — just collect gaps. This keeps the scan fast and ensures all subsequent work is focused.

### Phase 2: Targeted Search

Run web searches for each identified search gap. These searches are naturally focused because they target specific analyses, not broad topics.

**What to search for:**
- **Production benchmarks for specific setups:** "SMB factor correlation Ken French large-cap-only universe" — the research agent searched for factor models generally; you're searching for the exact comparison your exercise produces.
- **Data constraint validation:** "yfinance fundamental data coverage years quarterly" — confirming whether the data plan's assumptions hold for the specific fields and timeframes you need.
- **Acceptance criteria calibration:** "cross-sectional R-squared Fama-MacBeth S&P 500 monthly" — finding realistic ranges for your specific universe/timeframe/method combination.

**What NOT to search for:**
- Broad domain knowledge (that's Step 1's job — it's already in research_notes)
- Exercise design ideas (the blueprint is immutable)
- Teaching angles for potential divergences (that's Step 6's job)

### Phase 3: Code Probes

Run short Python scripts to verify data assumptions and calibrate acceptance criteria ranges. Search results may reveal what needs probing — that's why this phase follows web search.

**Three levels, with escalating justification required:**

**Data probes — always appropriate.** Quick API calls to verify data exists, has expected fields, and covers the needed range. These answer binary questions.
```python
# Does yfinance return operating income for AAPL?
import yfinance as yf
t = yf.Ticker("AAPL")
print(t.income_stmt.loc["Operating Income"])
```

**Coverage audits — always appropriate.** Download a small sample and check completeness across the universe. These answer "how many?" and "how far back?" questions that the data plan depends on.
```python
# How many S&P 500 tickers have complete 10-year fundamental data?
# (download for 20 representative tickers, check field coverage)
```

**Range calibration — when uncertainty is high.** Run a simplified version of an analysis on a small sample to set realistic acceptance criteria ranges. Justified when an unknown is Category B (knowable but uncertain) with a range so wide it's useless.
```python
# What's a realistic cross-sectional R² for Fama-MacBeth on 50 S&P 500 stocks?
# (run on 50 stocks × 3 years as a quick sanity check)
```

**Full feasibility prototypes — flag to the user first.** If you believe an exercise might not work at all with free data (e.g., a signal might not exist, a model might not converge), describe the concern and ask the user whether to run a prototype. Do not run one unprompted — this borders on Step 4 territory.

**Rules for all code probes:**
- **Throwaway.** Probe code is not saved, not committed, and is not input to Step 4. The code agent writes everything from scratch.
- **Minimal sample.** Use the smallest data sample that answers the question — 5-20 tickers, 1-3 years. Don't download the full universe.
- **Document findings, not code.** Record results in expectations.md: "Probe: 187 of 200 tickers have ≥5 years of income_stmt data via yfinance. 13 tickers missing operating income entirely." The code itself is ephemeral.

### Phase 4: Risk & Interaction Analysis

After probes complete, run an adversarial pass over the emerging data plan before writing. The gap scan and probes answer "does the data exist?" — this phase answers "what goes wrong even when the data exists?"

**Per-dimension risks.** For each recommended choice (universe, date range, frequency, data source), ask: "What's the most likely way this fails or produces misleading results?" A large-cap-only universe might kill small-cap effects. A short date range might miss a full business cycle. A daily frequency might introduce microstructure noise that obscures the signal being taught. Document each risk concretely.

**Interaction effects.** Choices interact in ways no single dimension reveals. A small universe + high frequency may lack cross-sectional power. Adding fundamentals to a large equity universe may hit API rate limits that make the download unreliable (a reliability problem, not a time problem — handle with retry logic in data_setup.py). A large universe + model training may push runtime beyond practical limits. Map the interactions between your recommended choices and flag compound risks.

**Contingencies.** For every identified risk, document a fallback: "If [failure mode], switch to [alternative] at the cost of [tradeoff]." The code agent should never encounter a data problem that wasn't anticipated here. If a risk has no viable contingency, escalate it — that's a signal the data plan may need a different recommendation.

This analysis becomes part of the data plan checkpoint — the user sees all of it before approving.

### Phase 5: Write

Write expectations.md with all search results, probe findings, and risk analysis in hand. Cite search findings with author/year/dataset. Cite probe findings as: "(verified via code probe: [result])." If neither search nor probe resolved a gap, note it: "(no benchmark found — range estimated from [rationale])."

---

## expectations.md — Format

### Header

`# Week N: [Title] — Expectations` followed by **Band**, **Implementability** (from blueprint), and **Rigor**.

```
**Rigor:** [Tier 1 / Tier 2 / Tier 3 — per the rigor schedule in rigor.md.
Defaults to the week's assigned tier. Only specify if overriding.]
- Teaches: [aspect name] (implement from scratch).
  [Only for first introductions within this tier.]
```

The Rigor field is stated once in the header and inherited by all sections. Individual sections override only when operating at a different tier than the week default (e.g., a non-ML section in a Tier 2 week uses Tier 1 rules).

**Signal viability stays as a separate field** — it's about IC statistical power, not about which rigor rules apply.

### Data Plan

The data plan defines the week's shared data parameters. Present **2-3 options per dimension** with tradeoffs and a **recommended choice**. Then analyze how the recommended choices interact and identify failure modes with contingencies.

**⏸ Checkpoint:** Write the full Data Plan section (including all options, tradeoffs, interaction analysis, failure modes, and contingencies) to `expectations.md` and **STOP**. **This checkpoint can be as long as needed; thoroughness matters more than brevity.** The purpose is to surface every potential problem *before* it propagates to Steps 4-7. The orchestrator will present the data plan for approval and resume you with the approved dimension choices. When resumed, append the per-section expectations below the Data Plan without modifying it.

```
## Data Plan

### Universe
- **Option A:** [size] — [tradeoff: statistical power, coverage, data availability]
- **Option B:** [size] — [tradeoff]
- **Recommendation:** [which and why]
- **Risk:** [most likely way this choice causes problems]

### Date Range
- **Option A:** [range] — [tradeoff: recency vs. depth]
- **Option B:** [range] — [tradeoff]
- **Recommendation:** [which and why]
- **Risk:** [most likely way this choice causes problems]

### Frequency
[daily / monthly / tick — determined by the week's content]
- **Risk:** [if applicable — e.g., daily data introduces microstructure
  noise, monthly data may lack granularity for intraday exercises]

### Interaction Analysis
[How do the recommended choices combine? Flag compound risks that no
single dimension reveals. Focus on data quality and reliability risks,
NOT download duration.
E.g., "200 stocks × 10 years daily = ~500K rows per field — manageable.
But adding fundamentals (quarterly, sparse) means 200 × 40 quarters with
~15% missing data, requiring imputation that affects Sections 3-4."
E.g., "Large universe + long date range + fundamentals requires batch
downloading with retry logic. If yfinance throttles mid-run, partial
data corrupts the cache — data_setup.py must handle partial failures."]

### Failure Modes & Contingencies
| Risk | Likelihood | Affected Sections | Contingency |
|------|-----------|-------------------|-------------|
| [specific failure mode] | high / med / low | [sections] | [fallback approach + what it costs pedagogically] |
| ... | ... | ... | ... |

If a risk has no viable contingency, flag it prominently — it may indicate
the recommended choice needs to change.

### Feasibility Concerns
[Only include this subsection if probes or research revealed that one or
more blueprint sections are infeasible with the available data. For each:]

- **Section [N]: [Title]** — INFEASIBLE
  Evidence: [what probes/research showed — concrete numbers, not vibes]
  Why it can't work: [1-2 sentences]
  Recommendation: [drop / restructure into X / merge with Section M]

[If no sections are infeasible, omit this subsection entirely.]

### Statistical Implications
[What the chosen size enables or limits. E.g., "200 stocks × 15 years
gives ~36,000 stock-months, sufficient for cross-sectional significance
at p < 0.05."]
```

**Rules:**
- **Bias toward MORE data — aggressively.** Weak results from undersized universes are pedagogical failures — the student blames the concept, not the sample. Download time is almost never a valid reason to reduce universe size, shorten date ranges, or lower frequency. A 20-minute download that produces statistically meaningful results is vastly preferable to a 30-second download that produces pedagogically useless noise. The only legitimate data-sizing constraints are: (1) API reliability — downloads that fail partway due to rate limiting or timeouts, (2) data availability — the data simply doesn't exist for the requested scope, (3) computational cost — model training (not downloading) becomes impractical. When presenting options, do not frame "faster download" as a pro or "slower download" as a con. Frame tradeoffs in terms of statistical power, coverage, and pedagogical quality.
- Name the API calls implied (yfinance tickers, Ken French dataset keys, FRED series IDs).
- Note estimated download time for planning purposes, but do NOT let it influence the recommendation.
- If the week uses multiple distinct datasets (equities + options, prices + fundamentals), give a separate plan per dataset.

### Known Constraints

Extract and restate every relevant limitation from `research_notes.md`. The code agent and consolidation agent should not need to re-read the research notes for constraint information.

```
## Known Constraints

- **[Constraint name]:** [What it is, why it matters, which sections it affects]
```

Examples of constraints:
- Survivorship bias: "yfinance returns only current listings. Our 'historical S&P 500' is actually today's S&P 500, inflating backtest returns by 1-3% annually."
- Universe bias: "S&P 500 'small-caps' have market caps > $10B. True small-cap effects (SMB) require the full NYSE/AMEX/NASDAQ universe."
- Point-in-time violation: "yfinance fundamental data is 'as-reported' but not point-in-time — we can't verify exact reporting dates, introducing potential look-ahead bias."
- API limitations: "yfinance fundamental data covers ~5 years reliably; older data may have gaps."

### Data Allocation Summary

A top-level table giving a quick overview of what data each section/exercise/deliverable uses.

```
## Data Allocation Summary

| Section / Exercise | Data Source | Universe Subset | Specifics |
|--------------------|------------|-----------------|-----------|
| S1: [title]        | yfinance   | 10 tickers      | Daily OHLCV, 2014-2024 |
| S2: [title]        | Ken French | FF3 factors     | Monthly, 1963-2024 |
| Ex1: [title]       | yfinance   | 50 tickers      | + fundamentals |
| D1: [title]        | yfinance   | Full universe   | Daily + monthly |
```

This table is a summary — the detailed data specifications and acceptance criteria live in the per-section blocks below.

---

## Per-Section Expectations

Mirror the blueprint's structure exactly — same section titles, same numbering. Every lecture section, seminar exercise, and homework deliverable gets an entry.

### Lecture Sections

```
### Section N: [Title — from blueprint]

**Data:** [Specific data inputs: tickers, date range, fields, frequency.
NOT acceptance criteria — these are inputs the code agent needs to
write data_setup.py and section files.]

**ML methodology:** [Only for sections that train a model or evaluate
predictive performance. Omit for descriptive/visualization sections.]
- Validation: [walk-forward / expanding window / purged k-fold / cross-sectional]
- Train window: [N months/observations, or "expanding from [date]"]
- Hyperparameter search: [yes — key params to tune] or [no — justify defaults]
- Prediction horizon: [1-month / 3-month / etc. — justify choice]

**Signal viability:** [Only for sections that compute rolling IC or OOS R².
Omit for descriptive sections, skill-building, or non-predictive tasks.
Assess whether statistical significance is achievable given the sample:]
- High — [N] observations, significance likely at 5% level
- Moderate — [N] observations, significance possible but not guaranteed
- Low — [N] observations, significance unlikely. Teaching angle: [how to
  frame insignificance constructively — e.g., methodology demonstration,
  sign consistency as secondary indicator]

**Acceptance criteria:**
- [Verifiable condition with approximate range]
- [Verifiable condition]
- ...

**Production benchmark:** [What this result looks like with institutional
data (CRSP, Bloomberg, etc.), with specific citation:
"Author (Year) found X using [dataset]"]

**Open questions:** [Only if genuinely unknowable — see taxonomy below.
Omit this field if none.]
```

### Seminar Exercises

Same format. The blueprint's exercises have no data field — that's your job. Assign data that makes the exercise work while staying within the week's data plan.

```
### Exercise N: [Title — from blueprint]

**Data:** [Specific data inputs for this exercise. May use a different
subset of the week's data, or require additional downloads.]

**ML methodology:** [Only if the exercise trains a model. Same 3-field
format as lecture sections above. Omit for non-ML exercises.]

**Signal viability:** [Only if exercise computes rolling IC or OOS R².
Same format as lecture sections above.]

**Acceptance criteria:**
- [Per task_design.md patterns for this task type]
- ...

**Production benchmark:** [If applicable — some exercises don't need
production comparison]
```

### Homework Deliverables

Same format. Homework deliverables are integrative — they may use the full week's data plus additional downloads.

```
### Deliverable N: [Title — from blueprint]

**Data:** [Specific data inputs. Note dependencies on prior deliverables'
data if any.]

**ML methodology:** [Only if the deliverable trains a model. Same 3-field
format as lecture sections above. Omit for non-ML deliverables.]

**Signal viability:** [Only if deliverable computes rolling IC or OOS R².
Same format as lecture sections above.]

**Acceptance criteria:**
- [Per task_design.md patterns for this task type]
- ...

**Production benchmark:** [Cited reference point]
```

---

## Acceptance Criteria Rules

All four base rules from `task_design.md` apply: approximate ranges, machine-checkable, two-sided, tied to learning. Read them there — they are not restated here. The following two rules are specific to expectations.md:

**1. Annotate sensitivity.**
When a criterion depends on the sample period or data source, note it: "HML return is typically positive long-run, but may be negative in post-2007 samples — the 'death of value.' Negative HML is expected and pedagogically valuable." This annotation helps the consolidation agent (Step 6) distinguish data limitations from code bugs.

**2. Match task type to verification pattern.**
Each criterion should align with its task type's verification pattern from `task_design.md` (demonstrations verify what the student sees, discoveries verify the insight emerged, constructions verify behavior at scale, etc.).

**3. Signal significance criterion for predictive sections.**
When a section trains a model to predict returns or prices and computes a rolling IC series (or equivalent), at least one acceptance criterion must address whether the signal is *statistically distinguishable from zero*, not just whether the metric falls within a range.

Pattern: `IC t-stat > 1.96 (5% significance)` or, when statistical significance is unlikely given the sample, an explicit `⚠ SIGNAL VIABILITY` annotation (see below).

**Why this matters:** An IC of 0.03 with t=2.4 and an IC of 0.03 with t=0.8 tell completely different stories. The first is a real (if weak) signal; the second is indistinguishable from noise. Without a significance criterion, the consolidation agent can't distinguish "weak but real" from "learned nothing." The code agent computes this automatically via `ic_summary()` in `course/shared/metrics.py`.

**When significance is unlikely — signal viability annotation.**
Some sections will produce IC series too short for statistical significance (e.g., 24 monthly observations on 50 stocks). If you predict this, annotate the section:

```
**Signal viability:** Low — [N] monthly observations. IC significance
unlikely at 5% level. Teaching angle: demonstrate proper methodology;
use IC sign consistency (pct_positive) as a secondary indicator of
directional learning even without statistical significance.
```

This annotation is critical. It tells the consolidation agent (Step 6) that insignificance was *anticipated and planned for* — so it can frame the result honestly rather than spinning it as a success or flagging it as a failure. If Step 3 does NOT provide this annotation but the code produces insignificant IC, Step 6 must treat the result with suspicion: either the code has a problem, or the expectations missed something.

**4. Provide pedagogical alternatives for demonstration-dependent sections.**
Some sections need results to land in a specific range to demonstrate the teaching point (e.g., "substitution ratio > 1.5x shows feature redundancy," or "NN IC close to GBM IC shows trees suffice"). When the teaching point depends on a particular outcome:

```
**Pedagogical alternatives:**
- **Primary:** [default approach from the section design]
- **Alternative A:** [different setup that may produce a clearer result]
  Trigger: [when to try this — e.g., "if substitution ratio < 1.0"]
  Expected effect: [why this alternative helps the teaching point]
- **Alternative B:** [another option, if applicable]
  Trigger: [condition]
  Expected effect: [rationale]
```

**When to provide alternatives:** For sections where assertions pass but results could be pedagogically flat — the numbers are technically within range but don't clearly demonstrate the intended insight. Common cases:
- Comparative sections ("model A vs model B" where the difference might be negligible)
- Feature importance / knockout experiments (where effects might be too small to see)
- Substitution / redundancy demonstrations (where the expected pattern might not emerge)

**When to skip:** Sections where the acceptance criteria themselves guarantee the teaching point (e.g., "CAPM R² < 0.20" — if the assertion passes, the point is made). Also skip for pure data-loading or preprocessing sections.

**5. Criteria flexibility — non-standard metrics.**
Criteria may reference any statistical test or metric — you are not limited to what exists in `course/shared/`. However, if you specify a non-standard metric (beyond correlations, t-tests, R², IC, Sharpe, or functions in shared/metrics.py), verify during Phase 3 code probes that the computation is feasible. Annotate the library and function call (e.g., "Johansen trace statistic via `statsmodels.tsa.vector_ar.vecm.coint_johansen()`"). The code agent reads this annotation.

**6. Discriminating structural tests for build-from-scratch deliverables.**
When a section or deliverable asks students to implement an algorithm from scratch (marked "Teaches: [aspect] (implement from scratch)" in the rigor header, or explicitly described as a build-from-scratch exercise in the blueprint), at least one acceptance criterion must be a **discriminating structural test** — a test that can *only* pass with the correct algorithm, not with a simpler variant that happens to satisfy the other criteria.

**Why this matters:** File agents are incentivized to make assertions pass. A simpler algorithm variant often satisfies all outcome-based criteria (metric ranges, error conditions, output shapes) while being fundamentally different from the specified algorithm. Outcome-based criteria test *what the code produces*; discriminating structural tests verify *how it produces it*. Without at least one discriminating test, 4C has no way to catch a correct-but-wrong implementation.

**How to write discriminating tests:** Identify the defining structural property that distinguishes the target algorithm from its nearest simpler variant. The test should *fail* if the implementation is simplified.

```
Pattern: "For [algorithm], at least one [structural property] must hold
that would NOT hold for [simpler variant]."
```

Examples:
- **PurgedKFold vs. PurgedWalkForward:** "For at least one fold that is not the last, the training set must include observations with positional index > max(test_idx) + embargo — confirming post-test data is used for training (the defining property of k-fold vs. walk-forward)."
- **Expanding window vs. fixed window:** "The training set size must monotonically increase across folds — window_size[fold_k] > window_size[fold_1] for k > 1."
- **CPCV vs. standard purged k-fold:** "The number of unique train/test paths must equal C(k, p) where k=n_splits and p=n_test_splits — combinatorial path generation is the defining property."
- **Custom loss function vs. MSE:** "Gradient values on a synthetic input where custom loss and MSE disagree must differ by > ε — confirming the custom loss is actually used."

**When to include:** Any section where:
1. The deliverable is "build X from scratch" AND
2. A simpler variant (walk-forward instead of k-fold, fixed instead of expanding, default loss instead of custom) would pass all other criteria

**When to skip:** Sections that import an algorithm from `shared/` or a library (the implementation is already correct), or where the outcome criteria inherently distinguish the algorithm (e.g., if the only way to achieve the expected R² is with the correct approach).

---

## ML Methodology Specifications

For any section, exercise, or deliverable that trains a model or evaluates predictive performance, the `**ML methodology:**` field specifies the four methodology choices that affect what results to expect. This field directly informs the acceptance criteria ranges — walk-forward validation produces lower metrics than random splits, hyperparameter tuning produces higher metrics than defaults, etc.

**The four fields (methodology level):**
- **Validation strategy:** Walk-forward, expanding window, purged k-fold, or cross-sectional. This is the single biggest driver of expected metric ranges.
- **Train window:** How much data the model trains on. Affects model quality and how many OOS observations are available.
- **Hyperparameter search:** Whether the code agent should search (and roughly what — "key tree parameters" or "regularization strength"), or whether defaults are acceptable with justification.
- **Prediction horizon:** The target return horizon (1-month, 3-month, etc.) and its justification. Shorter horizons have higher IC but more microstructure noise; longer horizons are less predictable but yield cleaner signals.

Everything else — preprocessing, architecture, exact HP values, code structure, random seeds — is the code agent's domain (specified in `code_plan.md` during Step 4, governed by `rigor.md`).

**Why this matters for acceptance criteria:** If you specify "expanding-window validation with 60-month initial window," your IC prediction should reflect OOS performance under that regime — not in-sample or random-split performance. Methodology and criteria must be consistent.

**When to skip:** Sections that only compute descriptive statistics, visualize data, or demonstrate a concept without predictive evaluation don't need this field.

---

## Production Benchmarks

Production benchmarks let the consolidation agent (Step 6) contextualize sandbox results. Without them, the agent can't distinguish "our data is limited" from "our code is wrong."

**Rules:**
- **Cite specifically:** Author, year, dataset, and result. "Gu-Kelly-Xiu (2020) report OOS R² of 3.2% for gradient boosting on CRSP monthly data, 1957-2016."
- **Name the dataset:** CRSP, Compustat, Bloomberg, TAQ, OptionMetrics. This tells the consolidation agent what "production" means for this result.
- **Two sources:** research_notes.md (broad domain findings from Step 1) and your targeted searches (specific benchmarks for the exact analysis being done). If neither yields a usable benchmark, use your own knowledge but flag it: "(no published benchmark found — range estimated from [rationale])."
- **Not every section needs a benchmark.** Skill-building exercises (e.g., "download and clean data") have correct/incorrect, not production-vs-sandbox. Only sections where sandbox data limitations materially affect results need benchmarks.

---

## Open Questions — Taxonomy

Not everything uncertain belongs in "open questions." Classify unknowns into three categories:

| Category | Test | Goes in | Example |
|----------|------|---------|---------|
| **A: Genuinely unknowable** | No reliable prior exists; result depends on specific data/model interaction | Open Questions section | "Will R² be high enough, or will we need to aggregate across months?" |
| **B: Knowable but uncertain** | Direction is predictable; magnitude depends on sample | Acceptance criteria with wider ranges | "HML return: sign uncertain post-2007. Range [-0.02, +0.04] monthly." |
| **C: Knowable** | Reliably predictable from domain knowledge | Acceptance criteria with normal ranges | "CAPM cross-sectional R² < 0.20" |

**Only Category A items go in Open Questions.** Categories B and C go in acceptance criteria — the range width communicates the uncertainty. If you're tempted to put something in Open Questions because you're unsure of the range, widen the range instead.

The code agent reads open questions as "investigate this and report in execution_log.md." The consolidation agent uses the answers to frame the narrative.

**Open question format:**
```
## Open Questions

1. [Question in plain language]
   **Why unknowable:** [1 sentence]
   **Affects:** [Which sections/exercises]
   **If X:** [Implication]  **If Y:** [Implication]
```

---

## Boundaries

The expectations document does NOT contain:

| Excluded | Why |
|----------|-----|
| Code | Step 4 implements |
| Narrative or prose strategy | Step 6 decides teaching angles |
| Blueprint changes (unless infeasible) | Blueprint is authoritative on structure |
| Teaching angles for divergences | Step 6's job |

**The expectations agent predicts; it doesn't teach.** If a predicted result is pedagogically awkward, note it as a constraint or open question — don't redesign the exercise.

**Exception — feasibility vetoes.** If code probes or research demonstrate that a blueprint section *cannot work* with the available data (not "might produce weak results" — that's what pedagogical alternatives handle — but genuinely cannot be implemented), flag it in the data plan's Feasibility Concerns subsection with concrete evidence. The user decides at the data plan checkpoint whether to drop, restructure, or override. This is the cheapest place in the pipeline to kill an infeasible idea — far cheaper than discovering it during Step 4 coding or Step 6 consolidation.

---

## Quality Bar

- [ ] Gap scan completed — search gaps and probe gaps identified before any writing
- [ ] Code probes run for data availability and coverage questions; findings cited in doc
- [ ] Data plan presents options with tradeoffs and a recommendation per dimension
- [ ] Risk identified for each recommended choice (most likely failure mode)
- [ ] Interaction effects analyzed across combined dimension choices
- [ ] Failure modes & contingencies table with fallback for every identified risk
- [ ] Data plan checkpoint is comprehensive — no length constraint, all analysis included
- [ ] Known constraints extracted from research_notes — document is self-contained
- [ ] Data allocation summary table covers every section, exercise, and deliverable
- [ ] Every section/exercise/deliverable has a **Data** field (inputs) separate from criteria
- [ ] Week header includes **Rigor** field with tier assignment per rigor.md schedule
- [ ] "Teaches" annotations identify first-introduction aspects
- [ ] Any section using yfinance fundamentals has a PIT constraint in Known Constraints
- [ ] Fundamental-heavy sections specify reporting lag strategy in ML methodology
- [ ] Predictive sections include prediction horizon justification
- [ ] Every model-training section has an **ML methodology** field (validation, train window, hyperparam search, prediction horizon — 4 fields)
- [ ] ML methodology choices are consistent with acceptance criteria ranges (e.g., walk-forward criteria ≠ random-split calibration)
- [ ] Predictive sections with rolling IC include a **Signal viability** assessment (High / Moderate / Low)
- [ ] Low-viability sections include an explicit teaching angle for insignificance
- [ ] Every section/exercise/deliverable has **Acceptance criteria** using approximate ranges
- [ ] Criteria are machine-checkable (convertible to assertions)
- [ ] Criteria are two-sided where applicable (not just "X > threshold")
- [ ] Production benchmarks cite author, year, dataset, and result
- [ ] Open questions use the ABC taxonomy — no Category B/C items in Open Questions
- [ ] Pedagogical alternatives provided for demonstration-dependent sections (where teaching point needs results in a specific range)
- [ ] Feasibility concerns (if any) backed by concrete probe/research evidence, not speculation
- [ ] Build-from-scratch deliverables include at least one discriminating structural test that distinguishes the target algorithm from simpler variants
- [ ] Structure mirrors blueprint exactly (same titles, numbering) — minus any sections vetoed at checkpoint
- [ ] Non-standard metric criteria include library/function annotation
- [ ] No code, no narrative strategy, no unilateral blueprint modifications
