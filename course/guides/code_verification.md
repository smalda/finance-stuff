# Code Verification — Step 4 Guide

> **Consumer:** Three agent types execute Step 4 in sequence. **4A (Plan):** a single Task agent that reads upstream artifacts and produces the implementation plan + data layer. **4B (Implement):** parallel Task agents (one per code file) launched by the main session, each writing and testing one file — they read `file_agent_guide.md` for the code format and implementation rules. **4C (Verify):** a single Task agent that audits all outputs for criteria coverage and pedagogical quality, then produces the final log artifacts.

---

## Inputs

In addition to `common.md`, `task_design.md`, and `rigor.md`:
- **`blueprint.md`** from Step 2 — what to implement (sections, exercises, deliverables, narrative arc)
- **`expectations.md`** from Step 3 — acceptance criteria, approved data plan, known constraints, production benchmarks, ML methodology specifications, pedagogical alternatives, open questions

**Blueprint** tells you WHAT to build (structure, ordering, scope). **Expectations** tells you HOW to verify it (criteria, data parameters, constraints) and specifies ML methodology decisions (validation strategy, hyperparameter search expectations, preprocessing). **`rigor.md`** defines the quality standard your ML code must meet. Together they define the full specification.

---

## Outputs

Five artifacts, each serving a different purpose:

1. **`code_plan.md`** inside the week folder — the implementation contract. Written by 4A, consumed by all 4B file agents and 4C. Contains criteria map, execution waves, per-file implementation strategy (including pedagogical alternatives), device + runtime info, and cross-file data flow.
   - **Used by:** 4B file agents (as their sole specification), 4C (for coverage audit), downstream steps for context.

2. **`code/` directory** inside the week folder — all implementation files plus cached data and plots (see Directory Structure).

3. **`code/logs/`** inside the code directory — per-file stdout logs (`.log`), implementation notes (`logs/notes/`), and plot PNGs (`logs/plots/`) from each 4B file agent.
   - **Used by:** 4C (to build execution_log.md and run criteria coverage), main session (for progress monitoring), Step 5 (reads plots from `logs/plots/`).

4. **`run_log.txt`** inside the week folder — consolidated stdout from all scripts, compiled by 4C from per-file logs in execution order. This is the ground truth of what the code produced.
   - **Used by:** downstream steps as raw execution evidence.

5. **`execution_log.md`** inside the week folder — developer report. Written by 4C, stitching per-file notes with cross-file analysis.
   - **Consumed by:** Step 6 (Consolidation) for implementation context and open question answers.

Code is mutable until the user approves. Log files are regenerated on each run cycle.

---

## Step 4A: Plan + Data

**Executed by:** A single Task agent.

**Reads:** `common.md`, `code_verification.md`, `task_design.md`, `rigor.md`, `blueprint.md`, `expectations.md`. If the blueprint references prior-week code, also read those files (read-only).

**Process:**

```
1. Read blueprint.md + expectations.md in full
2. If blueprint references prior-week code: read those files
3. Shared infrastructure check: read course/shared/API.md (the auto-generated
   API reference for all shared modules — metrics, temporal, evaluation,
   backtesting, dl_training, portfolio, derivatives, microstructure, regime).
   For each acceptance criterion and implementation need, check whether a
   shared function already exists. Prefer shared imports over reimplementation.
   For metrics specifically: verify the metric is (a) in shared/metrics.py,
   (b) in another shared module, (c) a standard numpy/scipy/sklearn/statsmodels
   function, or (d) annotated with a library reference in expectations.md.
   If no implementation path exists, flag with a suggested alternative.
   Do not silently drop criteria.
4. Stack audit: identify every third-party package the week's code will
   need (from research_notes.md recommendations, expectations.md library
   annotations, and your own implementation plan). For each:
   a. Check whether it is already installed (`poetry show <pkg>`)
   b. If missing: install it (`poetry add <pkg>`)
   c. If installation fails (version conflict, platform incompatible):
      flag the package — propose an alternative library or mark the
      affected files as blocked
   Record all packages in the Stack section of code_plan.md (see below).
5. Write code_plan.md (see Code Plan below)
6. Write data_setup.py (implementing the approved data plan from expectations.md)
7. Run data_setup.py — verify data downloads and cache
8. Present code_plan.md + data_setup.py status to the main session
```

**Contributing new shared datasets:** If, while building the data plan, you discover a dataset that (a) is used by 2+ weeks and (b) is raw/untransformed (prices, factor returns, macro series), add it to `course/shared/data.py` rather than downloading it per-week. Follow the existing patterns in that module: single-file cache under `.data_cache/`, idempotent download, clean return type. Update the "Available shared datasets" list in the Cross-Week Dependencies section below, and update `shared/__init__.py`'s module catalog.

**Done when:** `code_plan.md` is complete and `data_setup.py` runs successfully with data cached.

### Code Plan

`code_plan.md` is the **single interface contract** between 4A and all downstream agents. It compiles blueprint + expectations into a self-contained specification that file agents can work from without reading upstream artifacts.

**code_plan.md format:**

```markdown
# Code Plan — Week N: [Title]

## Stack

| Package | Min version | Used by | Status |
|---------|-------------|---------|--------|
| lightgbm | 4.6 | s3_gradient_boosting.py, ex2_knockout.py | already installed |
| ruptures | 1.1 | s4_changepoint.py | newly installed |

[Every third-party package beyond the standard scientific stack
(numpy, pandas, matplotlib, scipy, scikit-learn, statsmodels) that
this week's code imports. "Status" is either "already installed"
or "newly installed" (added via poetry add during this step).]

## Criteria Map

| # | Criterion (from expectations.md) | Assertion range | Target file | Notes |
|---|---|---|---|---|
| S3-1 | "GBM OOS IC" | [-0.01, 0.06] | lecture/s3_gradient_boosting.py | — |
| S4-1 | "|NN IC - GBM IC| ≤ 0.05" | [0, 0.05] | lecture/s4_neural_vs_trees.py | comparative: needs s3 output |

[Every criterion from expectations.md, with full assertion range copied
here so file agents don't need to read expectations.md.]

## Execution Waves

- **Wave 0:** data_setup.py
- **Wave 1:** s1_descriptive_stats.py, s2_factor_model.py, s3_gradient_boosting.py
  [independent — no cross-file dependencies]
- **Wave 2:** s4_neural_vs_trees.py (needs s3 cache), ex1_rolling_window.py
  [s4 depends on s3; ex1 is independent but grouped here for balance]
- **Wave 3:** ex2_feature_knockout.py (needs s3 cache), ex3_shap_analysis.py (needs s3 cache)
- **Wave 4:** d1_full_pipeline.py, d2_report.py (needs d1 output)

[Group independent files into the same wave. Sequential dependencies
force files into later waves. Within a wave, all files run in parallel.]

## Per-File Implementation Strategy

### lecture/s3_gradient_boosting.py
- **Runtime:** medium (1-5 min)
- **Device:** mps (downshift: cpu)
- **Shared infra:** temporal.WalkForwardSplitter, metrics.ic, metrics.rank_ic
- **HP search:** GridSearchCV — learning_rate ∈ [0.01, 0.05, 0.1], max_depth ∈ [3, 5, 8]
- **Early stopping:** 50 rounds, temporal validation MAE
- **Visualization:** bar chart of monthly IC (one bar per month, color by sign); cumulative return line chart for long-short portfolio
- **Caches for downstream:** gbm_predictions.parquet, gbm_ic_series.parquet
- **Criteria:**
  - S3-1: GBM OOS IC ∈ [-0.01, 0.06]
  - S3-2: Monthly IC series has ≥ 60 observations
- **Pedagogical alternatives:** none

### seminar/ex2_feature_knockout.py
- **Runtime:** slow (5-15 min)
- **Device:** cpu
- **Criteria:**
  - EX2-1: Max single-feature IC drop ∈ [0.001, 0.020]
  - EX2-2: Substitution ratio ∈ [1.5, 5.0]
- **Pedagogical alternatives:**
  - Alternative A: Use only momentum + reversal + volatility features (3 highly correlated)
    Trigger: substitution ratio < 1.0 with full feature set
    Expected effect: correlated features → higher substitution ratio, clearer redundancy demo

[One entry per file. Include ALL fields for files with ML, devices, or
pedagogical concerns. For simple files, only runtime + criteria are needed.]

## Cross-File Data Flow

- s3 → gbm_predictions.parquet, gbm_ic_series.parquet → consumed by s4, ex2, ex3
- ex1 → rolling_ic.parquet → consumed by d1
```

**Building the criteria map:**

1. List every acceptance criterion from every section, exercise, and deliverable in `expectations.md`
2. **Copy the full assertion range** into the map — file agents read only `code_plan.md`
3. Assign each criterion to the code file that will assert it (based on blueprint section → file mapping)
4. Identify **comparative criteria** — those that reference results from multiple analyses (e.g., "boosting IC > Ridge IC"). Assign these to the **downstream file** — the one written later that has access to both results
5. Identify **decomposable criteria** — single criteria that span multiple computations across files. Split into sub-assertions and assign each to its respective file, with the composite assertion in the downstream file

**Determining execution waves:**

1. Start with the default positional order: `s1_ → s2_ → ... → ex1_ → ex2_ → ... → d1_ → d2_ → ...`
2. Identify files with no upstream dependencies — these can be grouped into the same wave
3. Files that depend on another file's cached output go into a later wave
4. Within a wave, all files run in parallel via separate agents
5. Document the wave structure and dependency rationale

**Per-file implementation strategy:**

For each file, document the implementation choices. **Always include:**
- **Runtime:** `fast` (<1 min), `medium` (1-5 min), `slow` (5-15 min), `very_slow` (>15 min)
- **Criteria:** Full text + assertion ranges (copied from expectations)

**Include when applicable:**
- **Device:** `cpu`, `mps (downshift: cpu)` for Mac PyTorch, or `gpu` for files requiring GPU training (runs remotely via `course/shared/kaggle_gpu_runner.py`). Omit for non-PyTorch files
- **Shared infra:** imports from `course/shared/` or the external stack. List every shared function the file should use (e.g., `from shared.backtesting import sharpe_ratio, performance_summary`). The Step 4A agent discovers these from `course/shared/API.md`.
- **HP search:** method, grid values, search tool
- **Early stopping:** rounds, validation metric
- **Visualization:** plot type with justification (see below)
- **Caches for downstream:** files written to `.cache/` that other files consume
- **Pedagogical alternatives:** mapped from expectations, with trigger conditions

**Visualization choices:** For every file that produces a plot, specify the plot type and justify it briefly. The choice must match the data structure: bar charts for categorical comparisons or small discrete sets (e.g., IC by model), line charts for time series (e.g., cumulative returns), heatmaps for matrices (e.g., correlation), scatter for continuous bivariate data with many points (e.g., beta vs. return). Common mistakes to avoid: scatter plots with < 10 points (use bar), line charts for unordered categories (use bar), bar charts for hundreds of time-series observations (use line).

**Shared infra vs from-scratch rule:** If this week first introduces a concept (e.g., Week 5 introduces walk-forward validation), implement it from scratch — the implementation IS the educational content. If the concept was introduced in a prior week, import from `course/shared/` or the external stack.

---

## Step 4B: Implement

**Executed by:** The main session launches parallel Task agents (one per file), organized by execution waves from `code_plan.md`.

**File agents read `file_agent_guide.md`** for code format, verification block structure, assertion rules, special cases, and quality standard. They do NOT read this file (`code_verification.md`).

### Main Session Orchestration

```
1. Read code_plan.md — extract execution waves
2. For each wave (0, 1, 2, ...):
   a. Launch one background Task agent per file in the wave
      - Instruct each agent to prefix ALL runs with `PYTHONUNBUFFERED=1`
        and pipe to code/logs/{filename}.log via `2>&1 | tee`
   b. Monitor progress via code/logs/*.log files (tail -f or periodic read)
   c. Wait for ALL agents in the wave to complete
   d. If a file agent fails after 3 attempts → flag for user, continue wave
   e. Upstream context scan: read code/logs/notes/*_notes.md from this
      wave (all sections — Deviations, Challenges, ML Choices, etc.).
      Extract anything useful for downstream agents: environment
      workarounds, cache interface changes, assertion adjustments,
      runtime gotchas. Inject as {UPSTREAM_CONTEXT} in the next wave's
      launch prompts.
3. When all waves complete:
   a. Check that all expected code/logs/*.log files exist
   b. Grep logs for ✗ — any failures → stop and report
   c. Grep logs for ⚠ PEDAGOGICALLY FLAT — note for 4C
   d. Proceed to Step 4C
```

---

## Step 4C: Verify

**Executed by:** A single Task agent with a fresh context.

**Reads:** `code_plan.md`, `expectations.md`, all `.log` files in `code/logs/`, and all `_notes.md` files in `code/logs/notes/`.

**Process:**

```
1. Read code_plan.md criteria map in full
2. Read expectations.md for pedagogical alternatives, teaching intent,
   and ML methodology specifications
3. Read all code/logs/*.log files — verify every file has a ✓ status line
4. Read all code/logs/notes/*_notes.md files — collect implementation context
5. Criteria coverage check:
   a. For each criterion in the map: verify it appears in the corresponding log
   b. Flag any missing, migrated, or dropped criteria
6. Pedagogical quality check:
   a. For each file with pedagogical alternatives in code_plan.md:
      compare actual results (from logs) against the teaching point
   b. Flag files where assertions pass but results are pedagogically flat
   c. Flag any assertion ranges that were relaxed from expectations' original values
   d. Grep logs for ⚠ PEDAGOGICALLY FLAT markers — these are honest surrenders
      from file agents who tried all alternatives
7. Methodology fidelity check (see below):
   a. For each _notes.md with a "Deviations from Plan" entry that describes
      an algorithmic change: cross-reference against expectations' ML
      methodology specification and code_plan.md per-file strategy
   b. For build-from-scratch files: read the code file and verify the
      implementation matches the algorithm specified, not a simpler variant
   c. Flag ⚠ METHODOLOGY DEVIATION for any file where the implemented
      algorithm is structurally different from what was specified
8. Compile run_log.txt from per-file logs using bash (see below)
9. Write execution_log.md (see below)
10. Present findings to the main session
```

### Pedagogical Quality Check

This is the independent verification that the code agent's incentive problem (make assertions pass) didn't compromise the teaching point. The verify agent checks:

1. **Threshold fidelity:** Compare each assertion range in the code against the range in `expectations.md` (via `code_plan.md` criteria). Flag any assertion where the lower bound was decreased or upper bound was increased beyond what's justified.

2. **Teaching point clarity:** For files with pedagogical alternatives, check whether the actual results clearly demonstrate the intended insight. A substitution ratio of 0.9 technically passes `[0.3, 5.0]` but doesn't demonstrate redundancy. The verify agent flags this.

3. **Alternative adoption:** Check `_notes.md` for whether alternatives were tried when results were flat. If alternatives exist in the plan but weren't tried despite flat results, flag this.

4. **Pedagogical surrender escalation:** For any file with `⚠ PEDAGOGICALLY FLAT` in its log, the verify agent surfaces this prominently in `execution_log.md`. These are sections where the file agent tried all available alternatives and the teaching point still didn't emerge. **Do not narrate around these** — report them as potential section failures so Step 6 can evaluate whether to flag as "Pedagogically Unusable" and recommend reworking or dropping the section.

5. **Independent viability detection:** Do not rely solely on file agents to flag problems. File agents are incentivized to make their file pass — they may not recognize when their results are pedagogically dead. After completing checks 1–4, review each file's actual output against the blueprint's teaching intent for that section and ask: **does this result teach what the section is supposed to teach?** Flag a `⚠ VIABILITY CONCERN` for any file where:
   - Predictions are near-constant or degenerate (even if the file agent didn't flag PEDAGOGICALLY FLAT)
   - Signal metrics (IC, R², hit rate) are indistinguishable from zero despite the section's purpose being signal demonstration
   - Assertions passed only because ranges were relaxed to the point of meaninglessness (e.g., IC ∈ [-0.10, 0.10] when expectations said [0.01, 0.06])
   - The file "succeeded" but its output contradicts the teaching point (e.g., a feature importance exercise where all features have equal importance)
   - Multiple related files in the same section all show flat or contradictory results, suggesting the section's premise doesn't hold

   Treat `⚠ VIABILITY CONCERN` with the same force as `⚠ PEDAGOGICALLY FLAT` — report it as a potential section failure, not as a soft note. The difference: PEDAGOGICALLY FLAT is the file agent's honest surrender; VIABILITY CONCERN is your independent assessment that the file agent may not have recognized the problem.

### Methodology Fidelity Check

This check addresses a specific failure mode that outcome-based checks cannot catch: an implementation that satisfies all criteria while being a *structurally different algorithm* from what was specified.

**Why this exists:** File agents are incentivized to make assertions pass. A simpler algorithm variant (e.g., walk-forward-with-purging instead of true purged k-fold) often satisfies all outcome-based criteria — metric ranges, error conditions, structural checks — while being fundamentally different from the specified algorithm. The file agent may even document this deviation honestly in `_notes.md`. But outcome checks alone won't catch it because the simpler variant produces results in the correct direction and range. This check catches it.

**The check has two parts:**

1. **Deviation cross-reference.** Scan every `_notes.md` file's "Deviations from Plan" section. For each deviation that describes a change in algorithmic approach (not just a parameter adjustment or threshold change), cross-reference against:
   - The `**ML methodology:**` field in `expectations.md` for that section
   - The per-file implementation strategy in `code_plan.md`

   Ask: **does the deviation change WHAT algorithm is implemented, or just HOW it's tuned?** A parameter change ("used learning_rate=0.05 instead of 0.01") is a tuning deviation — note it but don't flag. An algorithmic change ("training restricted to pre-test data only" when k-fold was specified, "used fixed window instead of expanding") is a methodology deviation — flag it.

2. **Build-from-scratch code audit.** For every file that implements an algorithm from scratch (identified by "Teaches: [aspect] (implement from scratch)" in the expectations header, or by `code_plan.md` noting "Build from scratch"), **read the actual code file** — not just the logs and notes. Verify:
   - The defining structural property of the specified algorithm is present in the implementation
   - The implementation would fail any discriminating structural test from `expectations.md`'s acceptance criteria for that section
   - If no discriminating structural test exists in expectations (an upstream gap), apply your own: identify the nearest simpler variant and check whether the implementation is that variant

   This is the one case where 4C reads code, not just logs. The reason: notes and logs reflect what the agent *claims* it built; code is what it *actually* built. For build-from-scratch files, the implementation *is* the deliverable — checking it is not scope creep, it's the core verification.

**Flag format:** `⚠ METHODOLOGY DEVIATION: [file] — [what was specified] vs. [what was implemented]`. Report in the Methodology Fidelity section of `execution_log.md`. Treat methodology deviations as blocking issues — the implementation may need to be corrected before proceeding. They are more serious than pedagogical flatness (which is about teaching quality) because they mean the student would learn the *wrong algorithm*.

### execution_log.md — The Developer Report

Written by the 4C verify agent, stitching per-file notes into a coherent cross-file report.

```markdown
# Execution Log — Week N: [Title]

## Implementation Overview

[2-3 sentences: overall assessment, any notable patterns across files.]

## Per-File Notes

[For each file with something worth documenting, stitch from the
file agent's _notes.md. Skip files that were straightforward.]

### lecture/s3_gradient_boosting.py
- **Approach:** [from _notes.md]
- **ML choices:** [from _notes.md]
- **Challenges:** [from _notes.md]

### seminar/ex2_feature_knockout.py
- **Approach:** [from _notes.md]
- **Pedagogical alternatives:** [from _notes.md — what was tried, outcome]
...

## Deviations from Plan

[Aggregated from all _notes.md files. If no deviations: "No deviations
from plan." Each entry is self-contained:]

- **[file]:** Planned: [what]. Actual: [what]. Reason: [why].

## Open Questions Resolved

[For each open question from expectations.md:]

1. **[Question text]**
   **Finding:** [What the code revealed — from logs]
   **Implication:** [What this means — handed to Step 6]

## Criteria Coverage

[Status of every acceptance criterion from the criteria map.]

- **✓ Asserted:** [criterion] → [file]
- **↗ Migrated:** [criterion] → was [original file] → now [actual file] ([reason])
- **÷ Decomposed:** [criterion] → [sub-assertion 1] in [file], [sub-assertion 2] in [file]
- **✗ Dropped:** [criterion] → [why infeasible + actual result]

## Pedagogical Quality Assessment

[Results of the pedagogical quality check:]

- **Threshold fidelity:** [any relaxed ranges flagged, with original vs actual]
- **Teaching point clarity:** [any files where results are flat despite passing]
- **Alternatives tried:** [summary of which alternatives were attempted and outcomes]
- **Pedagogical surrenders:** [list any ⚠ PEDAGOGICALLY FLAT files — what was tried,
  why the teaching point failed. These are candidates for Step 6 to flag as
  "Pedagogically Unusable" for potential rework or removal.]

## Methodology Fidelity

[Results of the methodology fidelity check. If no build-from-scratch files
and no algorithmic deviations: "No methodology deviations identified."

For each flagged file:]

### ⚠ METHODOLOGY DEVIATION: [file path]

- **Specified algorithm:** [What expectations.md / code_plan.md specified — name + defining property]
- **Implemented algorithm:** [What the code actually implements — name + how it differs]
- **Evidence:** [Concrete: the code line(s) or notes excerpt that reveals the deviation]
- **Impact:** [What students would learn wrong — e.g., "Students would implement walk-forward-with-purging believing it is purged k-fold, missing that true k-fold uses post-test data for training"]
- **Source:** [Deviation cross-reference (from _notes.md) | Build-from-scratch code audit (from reading the code)]

[Methodology deviations are blocking — the implementation teaches the wrong
algorithm. Unlike pedagogical flatness (teaching point is weak) or viability
concerns (results are degenerate), methodology deviations mean the code is
structurally incorrect relative to the specification. Flag for correction
before proceeding to Step 5.]

## Viability Concerns

[List every file flagged with ⚠ VIABILITY CONCERN or ⚠ PEDAGOGICALLY FLAT.
If none: "No viability concerns identified."

For each flagged file:]

### ⚠ [file path] — [VIABILITY CONCERN or PEDAGOGICALLY FLAT]

- **Blueprint teaching intent:** [What this section is supposed to teach — from code_plan.md / expectations.md]
- **What the code actually produced:** [Concrete: the numbers, the output shape, what students would see]
- **Why this is a concern:** [Why the output fails to serve the teaching intent]
- **Source:** [File agent surrender (PEDAGOGICALLY FLAT) | Independent 4C assessment (VIABILITY CONCERN)]

[These are potential section failures. Downstream steps — especially Step 6
(consolidation) and Step 6¾ (brief audit) — should treat these as sections
whose pedagogical value is in question, not as sections that need better framing.]
```

### run_log.txt

**Use bash to compile this file — do NOT write it agentically.** This is a mechanical concatenation task: reading log files into your context and using the Write tool risks truncation, paraphrasing, or context overflow. Instead, use a bash command to concatenate all `code/logs/*.log` files in execution order (from `code_plan.md` waves) into a single file with headers.

**Bash pattern:**
```bash
# Build run_log.txt by concatenating per-file logs in execution order.
# List files in wave order (from code_plan.md), not alphabetical.
cd course/week{NN}_{TOPIC}
> run_log.txt  # start empty
for f in \
  code/logs/data_setup.log \
  code/logs/s1_topic.log \
  code/logs/s2_topic.log \
  code/logs/ex1_topic.log \
  code/logs/d1_topic.log; do
  name=$(basename "$f" .log)
  printf -- '--- %s ---\n' "$name" >> run_log.txt
  cat "$f" >> run_log.txt
  printf '\n' >> run_log.txt
done
```

Adapt the file list to match the actual execution waves from `code_plan.md`. The resulting file should look like:

```
--- data_setup ---
[contents of data_setup.log]

--- s1_topic ---
[contents of s1_topic.log]

...
```

The per-file `code/logs/*.log` files are the primary debugging evidence; `run_log.txt` is the consolidated view for downstream steps.

---

## Directory Structure

```
weekNN_topic/
├── blueprint.md              # From Step 2 (immutable)
├── expectations.md           # From Step 3 (immutable)
├── code_plan.md              # From 4A (implementation contract)
├── run_log.txt               # From 4C (compiled from per-file logs)
├── execution_log.md          # From 4C (developer report)
├── code/
│   ├── data_setup.py         # From 4A (shared data layer)
│   ├── .cache/               # Downloaded data + intermediate caches (gitignored)
│   ├── logs/                 # Per-file output
│   │   ├── plots/                # Plot PNGs from all files
│   │   │   ├── s1_capm_scatter.png
│   │   │   ├── s3_factor_cumulative.png
│   │   │   └── ...
│   │   ├── notes/                # Implementation notes from file agents
│   │   │   ├── s1_topic_notes.md
│   │   │   ├── s2_topic_notes.md
│   │   │   └── ...
│   │   ├── s1_topic.log          # Captured stdout from running s1
│   │   ├── s2_topic.log
│   │   └── ...
│   ├── lecture/
│   │   ├── s1_topic.py
│   │   ├── s2_topic.py
│   │   └── ...
│   ├── seminar/
│   │   ├── ex1_topic.py
│   │   └── ...
│   └── hw/
│       ├── d1_topic.py
│       └── ...
├── lecture.ipynb              # Written in Step 7 (not your concern)
├── seminar.ipynb
└── hw.ipynb
```

**Naming:** Files are prefixed with position (`s1_`, `s2_`, `ex1_`, `d1_`) and use `snake_case` names derived from blueprint section titles. Logs go in `logs/`: `s3_gradient_boosting.log`. Notes go in `logs/notes/`: `s3_gradient_boosting_notes.md`.

---

## The Shared Data Layer

`data_setup.py` handles all data downloads for the week. Written by the 4A plan agent, it runs once, caches data locally, and provides a clean interface for section files.

**Rules:**

1. **Implement the approved data plan.** `expectations.md` specifies universe, date range, frequency, API calls, and dataset-specific parameters. Implement these exactly.

2. **All downloads happen here.** Section files never call `yf.download()`, `web.DataReader()`, or equivalent. They import from `data_setup.py`.

3. **Cache to `.cache/` as Parquet.** Check for cached file first; download only if missing. Avoids re-downloading on every run.

4. **Module-level constants for shared parameters:** `TICKERS`, `START`, `END`, `UNIVERSE_SIZE`, etc. Section files import these alongside data-loading functions.

5. **Named load functions** — one per distinct dataset: `load_equity_data()`, `load_factor_data()`, `load_treasury_data()`. Each returns a well-defined DataFrame.

6. **Export `CACHE_DIR` and `PLOT_DIR`** — section files import `CACHE_DIR` for reading intermediate caches and `PLOT_DIR` for saving plot PNGs. Define both:
   ```python
   CACHE_DIR = Path(__file__).parent / ".cache"
   PLOT_DIR = Path(__file__).parent / "logs" / "plots"
   CACHE_DIR.mkdir(exist_ok=True)
   PLOT_DIR.mkdir(parents=True, exist_ok=True)
   ```

7. **Handle large downloads gracefully:**
   - Progress indicators (`tqdm` or periodic `print()`) for downloads over ~2 minutes
   - Retry logic with brief pauses between batches for rate-limited APIs (e.g., yfinance)
   - Partial failure handling: download what succeeds, report what failed, let section code work with the available universe
   - Print final data shape on success: ticker count, date range, row count, any failures

**Notebooks never import from `data_setup.py`.** That file exists for code verification only. Step 7 (notebook agents) reads it to understand what to download, then writes equivalent inline code. This is not your concern.

**Section file import pattern:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_equity_data, TICKERS, START, END, CACHE_DIR, PLOT_DIR
```

---

## Cross-Week Dependencies

The blueprint may reference prior-week concepts or code. Every week must be **self-contained** at runtime for notebooks — no imports or file-path coupling across weeks. But `data_setup.py` may use the shared data layer to avoid redundant downloads.

### Rule 1: Use the shared data layer for common datasets

If this week uses datasets that appear across multiple weeks (S&P 500 prices, Fama-French factors, fundamentals), import from `course/shared/data.py` in this week's `data_setup.py`. The shared module caches downloads at `course/shared/.data_cache/` — the first week to run downloads the data, subsequent weeks read from cache instantly.

```python
# In data_setup.py — importing shared data
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import SP500_TICKERS, load_sp500_prices, load_ff_factors

# Shared cache has the full superset (2000–2025, all tickers).
# Slice to this week's date range; derive week-specific data locally.
prices = load_sp500_prices(start=START, end=END)
monthly = prices.resample("ME").last().pct_change().dropna(how="all")
monthly.to_parquet(CACHE_DIR / "monthly_returns.parquet")
```

**What lives in shared vs. per-week:**

| Shared (`course/shared/data.py`) | Per-week (`.cache/`) |
|---|---|
| Raw daily prices (full history, all tickers + ETFs) | Monthly returns, subsets, derived features |
| Fama-French factor returns (full history) | Merged factor+return panels |
| Raw fundamentals (balance sheet, income) | Computed ratios (B/M, E/P), feature matrices |
| FRED economic series (per-series cache) | Week-specific transformations |

**Notebooks are still self-contained.** `data_setup.py` uses shared data for pipeline efficiency, but Step 7 notebook agents write inline download code — notebooks never import from `shared.data`.

**Available shared datasets:**
- `SP500_TICKERS` — full S&P 500 constituents (~503 tickers)
- `DEMO_TICKERS` — 20-ticker demo subset for lightweight analyses
- `ETF_TICKERS` — SPY, QQQ, IWM, TLT, GLD, XLE, XLF, XLK
- `ALL_TICKERS` — superset of SP500 + ETF tickers (what gets downloaded)
- `CRYPTO_TICKERS` — BTC-USD, ETH-USD, SOL-USD, BNB-USD, UNI-USD, AAVE-USD, LINK-USD, CRV-USD
- `load_sp500_prices(start, end, tickers)` — daily adjusted close; full superset cached once, sliced at read time
- `load_sp500_ohlcv(start, end, tickers, fields)` — full OHLCV with MultiIndex columns (field, ticker); use for spread estimation (High/Low) or volume analysis
- `load_ff_factors(model, frequency)` — FF3/5/6 factor returns (full history)
- `load_carhart_factors(frequency)` — Carhart 4-factor model (full history)
- `load_ff_portfolios(name, frequency)` — Ken French sorted portfolio returns (value-weighted). Names: `'25_size_bm'`, `'17_industry'`, `'49_industry'`, `'10_momentum'`. Monthly + daily.
- `KF_PORTFOLIOS` — dict mapping short names to Ken French dataset names
- `load_sp500_fundamentals(tickers)` — balance sheet, income, sector, market cap, shares
- `load_fred_series(series_ids, start, end)` — FRED economic data (per-series cache)
- `load_crypto_prices(start, end, tickers)` — crypto/DeFi token daily close prices (BTC, ETH, SOL, etc.)
- `FRED_TREASURY_YIELDS` — DGS1MO through DGS30 (pre-cached)
- `FRED_RATES` — FEDFUNDS, DFF (pre-cached)
- `FRED_VOLATILITY` — VIXCLS (pre-cached)
- `FRED_MACRO` — GDP, CPI, unemployment, sentiment, industrial production, 10Y-2Y spread, USREC recession indicator (pre-cached)
- `FRED_CREDIT` — high yield OAS, BBB OAS, TED spread (pre-cached)
- `FRED_ALL` — all of the above combined

### Rule 2: Read prior code as reference, not dependency

Before writing code_plan.md, the 4A agent reads the prior week(s)' relevant code files listed in the blueprint's prerequisites. This is read-only — it informs the plan but creates no runtime dependency.

### Rule 3: Build what you need, not what the prior week built

A prior week's class often serves a different purpose than what this week requires. Identify what THIS week needs, use the prior code to understand the computation logic, then build the needed form from scratch.

---

## Boundaries

| Excluded | Why |
|----------|-----|
| Prose, markdown narrative, or teaching strategy | Step 7 writes notebooks; Step 6 decides teaching angles |
| Interpretation of what results mean | Step 6 (Consolidation) reconciles ideal vs. actual |
| `.ipynb` files | Step 7 creates these |
| Blueprint or expectations modifications | Upstream artifacts are immutable |
| Teaching angles for divergences | Step 6 handles all divergence framing |
| Visual verification of plots | Step 5 handles this in a clean, fresh context |

**If results diverge from expectations:** The file agent implements the analysis code so the actual result is visible. If the criterion is demonstrably infeasible, it omits the assertion (see `file_agent_guide.md` Special Cases). The 4C verify agent tracks all divergences. `expectations.md` is immutable — Step 6 reconciles the ideal (blueprint) with the actual (code output).

---

## Quality Bar

### Step 4A (Plan + Data)
- [ ] `code_plan.md` written with: stack, criteria map (full assertion ranges), execution waves, per-file strategy, cross-file data flow
- [ ] Stack section lists all non-standard packages; any newly installed packages noted
- [ ] Every acceptance criterion from `expectations.md` assigned to a code file in the criteria map with its assertion range
- [ ] Execution waves account for cross-file dependencies
- [ ] Runtime estimates and device specifications included for all files
- [ ] Pedagogical alternatives from expectations mapped to specific files with trigger conditions
- [ ] `data_setup.py` implements the approved data plan from `expectations.md`
- [ ] `data_setup.py` exports both `CACHE_DIR` and `PLOT_DIR`
- [ ] `data_setup.py` runs successfully with data cached

### Step 4C (Verify)
- [ ] All `code/logs/*.log` files have ✓ status lines
- [ ] Criteria coverage verified: every criterion asserted or justified as dropped
- [ ] Pedagogical quality check completed: threshold fidelity, teaching point clarity, alternatives tried
- [ ] Pedagogical surrenders (⚠ PEDAGOGICALLY FLAT) escalated — not narrated around
- [ ] Independent viability detection applied: every file's output checked against blueprint teaching intent
- [ ] Methodology fidelity check completed: deviations cross-referenced against expectations' ML methodology, build-from-scratch files code-audited for correct algorithm
- [ ] Methodology Fidelity section in execution_log.md populated (or explicitly "No methodology deviations identified")
- [ ] Viability Concerns section in execution_log.md populated (or explicitly "No viability concerns identified")
- [ ] `run_log.txt` compiled from per-file logs in execution order
- [ ] `execution_log.md` written: per-file notes stitched, deviations aggregated, criteria coverage, pedagogical assessment, viability concerns
- [ ] Open questions from `expectations.md` investigated and answered
- [ ] No prose, no `.ipynb`, no upstream modifications, no interpretation
