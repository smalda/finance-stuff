# Orchestrator Guide — Week Build

> **Consumer:** The main Claude Code session acting as orchestrator. You launch Task agents for each pipeline step, review their output, make gate decisions, and log everything to `orchestration.md`. You do NOT do the heavy work yourself — agents do.

---

## Context Discipline — What You May and May NOT Read

Your context window must survive across all 7+ steps. Every file you read eats into that budget. Follow these rules strictly.

### Files you SHOULD read

| File | When |
|------|------|
| `course/guides/orchestrator.md` | Start of session (this file — your only guide) |
| `course/COURSE_OUTLINE.md` | Once, at start, to look up the week title |
| `course/curriculum_state.md` | Once, at start, to check if it exists |
| `course/weekNN_TOPIC/orchestration.md` | At start (resume check) and after each gate (write) |
| Produced artifacts (e.g., `research_notes.md`) | After each Task returns, for gate review |
| `course/weekNN_TOPIC/code_plan.md` | During Step 4B — to read execution waves and launch file agents |
| `course/weekNN_TOPIC/code/logs/*.log` | During Step 4B — to monitor progress and check for failures |
| `course/guides/flag_resolution.md` | Only if Step 6½ is triggered |
| Step 8 export summary (returned by agent) | After Step 8 — review what was exported to shared/ |

### Files you must NEVER read

| File | Why |
|------|-----|
| `course/guides/common.md` | For agents only — you don't need pipeline principles |
| `course/guides/research.md` | For the Step 1 agent |
| `course/guides/blueprint_spec.md` | For the Step 2 agent |
| `course/guides/expectations_spec.md` | For the Step 3 agent |
| `course/guides/task_design.md` | For Steps 2/3/4 agents |
| `course/guides/code_verification.md` | For Step 4A/4C agents — you orchestrate 4B directly |
| `course/guides/file_agent_guide.md` | For Step 4B file agents |
| `course/guides/rigor.md` | For the Steps 3/4/4C/6¾/8 agents |
| `course/guides/data.md` | For the Steps 3/4/5/7 agents |
| `course/guides/observation_review.md` | For the Step 5 agent |
| `course/guides/consolidation_spec.md` | For the Step 6 agent |
| `course/guides/brief_audit_spec.md` | For the Step 6¾ agent |
| `course/guides/notebook.md` | For the Step 7 agents |
| Any `.py` file in `code/` | For agents only — you review via code/logs/*.log and execution_log.md |
| `course/shared/API.md` | For Step 4A agent — auto-generated API reference |
| Plot PNGs in `code/logs/plots/` | For the Step 5 agent — you review via observations.md |

### How to read artifacts for gate review

**Don't read entire files when a targeted check suffices.** Examples:

- **Step 3A gate (supervised):** Do NOT read `expectations.md` — tell the user the data plan is ready and ask them to review the file directly
- **Step 3A gate (autonomous):** Read the Data Plan section of `expectations.md` — evaluate against checkpoint criteria
- **Step 3B gate:** Read `expectations.md` — verify data plan is unmodified, check per-section criteria
- **Step 4A gate:** Read `code_plan.md` headers — verify criteria map, waves, per-file strategy exist
- **Step 4B gate:** Grep `code/logs/*.log` for `✗` — don't read full logs
- **Step 4C gate:** Grep per-file logs for `✗`; skim execution_log.md pedagogical quality section
- **Step 6 gate:** Read the first 20 lines of `narrative_brief.md` to find the `Status:` field, then skim section headers
- **Step 6¾ gate:** Read `brief_audit.md` fully — all findings (critical, moderate, minor) should be applied; check viability verdicts for any Reframe/Redesign/Remove
- **Step 8 gate:** Read the export summary returned by the agent — verify exports are general-purpose, no duplicates, no week-code modifications
- **Step 7 gate:** Read the `nb_builder.py` terminal output (printed during the build command) — don't read the .ipynb files

When in **supervised** mode, you may read more of the artifact to produce a useful summary for the user. But still avoid reading guide files or code.

---

## Modes

| Mode | Gate behavior | When to use |
|------|--------------|-------------|
| **supervised** | Present summary + approval question to the USER. Wait for their response before proceeding. | Default. Human reviews every gate. |
| **autonomous** | Evaluate gate criteria yourself. Log your assessment. Proceed unless HARD STOP. | User explicitly requests it. Faster, but deferred review. |

In **supervised** mode, after each Task returns:
1. Read the produced artifact
2. Summarize what the agent did (3-5 bullet points)
3. List any concerns or questions
4. Ask the user: "Approve, revise, or skip?"

In **autonomous** mode, after each Task returns:
1. Read the produced artifact
2. Evaluate against the gate criteria (see per-step sections below)
3. Assign a verdict: **APPROVED**, **SOFT FLAG** (concern noted, proceeding), or **HARD STOP** (cannot proceed)
4. Log your reasoning to `orchestration.md`
5. On APPROVED or SOFT FLAG: proceed to next step
6. On HARD STOP: stop the pipeline. Explain what went wrong. The user will intervene.

---

## State File: `orchestration.md`

Location: `course/weekNN_TOPIC/orchestration.md`

This file is your external memory. Write to it after every gate decision. If your context gets compressed or the session restarts, re-read this file to know where you left off.

### Format

```markdown
# Week NN: [Title] — Orchestration Log
## Mode: [supervised | autonomous]
## Started: [timestamp]

---

### Step 1: RESEARCH — [APPROVED | SOFT FLAG | HARD STOP | PENDING]
- **Artifact:** research_notes.md
- **Summary:** [3-5 bullets of what the agent produced]
- **Gate assessment:** [what you checked and found]
- **Decision:** [APPROVED / SOFT FLAG: reason / HARD STOP: reason]
- **Notes:** [any orchestrator decisions, e.g., "told agent to exclude crypto"]

### Step 2: BLUEPRINT — PENDING
...
```

### Resuming

When the user says "continue building week NN" or you detect an existing `orchestration.md`:
1. Read it fully
2. Find the last completed step
3. Announce: "Resuming Week NN from Step X. Steps 1–(X-1) were already approved."
4. Launch the next pending step

---

## Week Registry

To launch a step, you need the week number (NN) and folder name (TOPIC). Parse these from the user's input. The folder name follows the pattern `weekNN_TOPIC` where TOPIC is a snake_case slug. Look up the week title from `course/COURSE_OUTLINE.md`.

If the week folder doesn't exist yet, create it before launching Step 1:
```bash
mkdir -p course/weekNN_TOPIC
```

---

## Step Prompts & Gate Criteria

For every step below: the **Launch prompt** is what you send as the `prompt` parameter to `Task(general-purpose)`. The **Gate criteria** are what you evaluate (autonomous mode) or present to the user (supervised mode).

### Important Rules

1. **You are the orchestrator — you do NOT run the step yourself.** Always launch a Task agent.
2. **Each Task gets a fresh context.** The agent has no memory of prior steps. Include all file paths it needs in the prompt.
3. **Never edit artifacts produced by agents** except in these two cases:
   - **Step 6½** (flag resolution): you execute it directly and patch multiple artifacts.
   - **Step 6¾ gate**: you apply audit findings as prose edits to `narrative_brief.md`.
4. **After every gate decision, update `orchestration.md`** before launching the next step.
5. **Step 3 has two phases.** The agent writes the data plan to `expectations.md` and stops (Step 3A). After approval, resume the same agent to write the rest (Step 3B).
6. **Step 5 has two phases.** Phase 1 and Phase 2 are separate Task launches (or resume the Phase 1 agent). There is a gate between them.
7. **Step 6½ is NOT a Task.** You execute it directly, interactively. Read `course/guides/flag_resolution.md` for the full process.
8. **Step 7 launches 3 parallel Tasks**, then you run `nb_builder.py` to convert the builder scripts to notebooks.
9. **Step 8 is a Task agent.** It scans the week's code for reusable utilities and exports them to `course/shared/`. Launch it after Step 7.

---

### Step 1: RESEARCH

**Launch prompt:**
```
You are the Step 1 research agent for Week {NN}: {TITLE}.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/research.md

Then read the FULL course outline (course/COURSE_OUTLINE.md) — Sections 1.1–1.7
for audience/context, then the Week {NN} entry for your specific scope.

Produce research_notes.md in course/week{NN}_{TOPIC}/ following the format
in research.md. Do NOT present a research plan for approval — proceed
directly with your research and write the complete artifact.
```

**Gate criteria (autonomous mode):**
- [ ] `research_notes.md` exists and is non-empty
- [ ] Contains identified data sources with access methods
- [ ] Contains tool/library recommendations
- [ ] No unresolved blockers that would prevent blueprint design
- [ ] HARD STOP if: agent reports data sources are inaccessible or topic is fundamentally unresearchable

---

### Step 2: BLUEPRINT

**Launch prompt:**
```
You are the Step 2 blueprint agent for Week {NN}: {TITLE}.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/blueprint_spec.md
3. course/guides/task_design.md

Then read these artifacts:
- course/COURSE_OUTLINE.md (full preamble + Week {NN} entry)
- course/week{NN}_{TOPIC}/research_notes.md
- course/curriculum_state.md

Produce blueprint.md in course/week{NN}_{TOPIC}/ following the format
in blueprint_spec.md.
```

Note: if `curriculum_state.md` doesn't exist, add to the prompt: "curriculum_state.md does not exist yet (this is the first week being built). Skip reading it."

**Gate criteria (autonomous mode):**
- [ ] `blueprint.md` exists with narrative arc, section breakdown, concept matrix
- [ ] No numerical predictions (Vision level — no specific numbers)
- [ ] Concept matrix has no overlapping exercises across lecture/seminar/hw
- [ ] Lecture/seminar/homework structure follows UNDERSTAND → APPLY → BUILD
- [ ] HARD STOP if: blueprint fundamentally misunderstands the week's scope per course outline

---

### Step 3: EXPECTATIONS (Two Phases)

Step 3 runs in two phases with a data plan checkpoint between them. The agent writes the data plan first, stops, and is **resumed** after approval to write the rest. This preserves the agent's full context (gap scan, web searches, code probes) across both phases.

#### Step 3A: Data Plan

**Launch prompt:**
```
You are the Step 3 expectations agent for Week {NN}: {TITLE}.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/expectations_spec.md
3. course/guides/task_design.md
4. course/guides/rigor.md
5. course/guides/data.md

Then read these artifacts:
- course/week{NN}_{TOPIC}/blueprint.md
- course/week{NN}_{TOPIC}/research_notes.md

Run your full preparation (gap scan, web searches, code probes, risk
analysis — Phases 1-4 from the spec). Then write ONLY the Data Plan
section to course/week{NN}_{TOPIC}/expectations.md — including Known
Constraints and Data Allocation Summary.

STOP after writing the data plan. Do NOT write per-section expectations.
You will be resumed with approval and dimension choices.
```

**Data plan checkpoint:**

The data plan is the single most consequential decision point in the pipeline — it constrains every downstream step. The user must see every option, tradeoff, risk, interaction effect, and contingency exactly as written.

- **Supervised:** Tell the user the data plan is ready at `course/week{NN}_{TOPIC}/expectations.md`. Do NOT read the file yourself — the user reviews it directly. Ask: *"The data plan is written to `expectations.md`. Please review the full data plan and let me know your dimension choices (or approve the recommendations). Approve / revise?"*
- **Autonomous:** Read the Data Plan section of `expectations.md` and evaluate:
  - [ ] Universe, date range, frequency, source specified with tradeoffs
  - [ ] Failure modes & contingencies table present
  - [ ] No reliance on sources identified as inaccessible in Step 1
  - [ ] Feasibility concerns (if any) have concrete evidence
  - [ ] HARD STOP if: any feasibility concern is INFEASIBLE — escalate to user even in autonomous mode

#### Step 3B: Full Expectations

**Resume the Step 3A agent** with the approved choices:

```
The data plan has been approved. {USER_CHOICES}

Continue writing expectations.md — append the per-section expectations
(lecture sections, seminar exercises, homework deliverables) using the
approved data plan parameters. Follow the format in expectations_spec.md.
Do NOT modify the Data Plan section already written.
```

Set `{USER_CHOICES}` to the user's specific dimension selections, e.g.:
- **Supervised:** `"Approved choices: Universe = Option B (200 S&P 500 stocks), Date Range = Option A (2014-2024), Frequency = daily."` — or `"All recommendations approved."` if the user approved without changes.
- **Autonomous:** `"Using the recommended options for all dimensions."`

**Gate criteria (after Step 3B):**
- [ ] `expectations.md` exists with data plan + per-section acceptance criteria + benchmark citations
- [ ] Data plan section is unmodified from Step 3A
- [ ] Acceptance criteria are approximate ranges (not exact numbers)
- [ ] Model-training sections have **ML methodology** field (validation, train window, hyperparam search — 3 fields only)
- [ ] Production benchmarks are cited with sources
- [ ] Open questions use ABC taxonomy (if applicable)
- [ ] HARD STOP if: feasibility concerns flag any section as INFEASIBLE — escalate to user even in autonomous mode (orchestrator cannot unilaterally remove blueprint sections)

---

### Step 4A: CODE — Plan + Data

**Launch prompt:**
```
You are the Step 4A planning agent for Week {NN}: {TITLE}.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/code_verification.md (Step 4A section)
3. course/guides/task_design.md
4. course/guides/rigor.md
5. course/guides/data.md

Then read these artifacts:
- course/shared/API.md (auto-generated shared library reference — check
  what functions/classes already exist before planning implementations)
- course/week{NN}_{TOPIC}/blueprint.md
- course/week{NN}_{TOPIC}/expectations.md

If the blueprint references prior-week code, also read the relevant
files from those weeks (read-only — no runtime dependencies).

Your job: write code_plan.md and data_setup.py per the spec in
code_verification.md. This includes a stack audit — identify all
third-party packages the week needs, install any missing ones via
`poetry add`, and record them in the Stack section of code_plan.md.
For each file's per-file strategy, explicitly list shared imports
the file should use (e.g., `from shared.backtesting import sharpe_ratio`).
Run data_setup.py to verify data downloads.
Do NOT write any section/exercise/homework code files — that's Step 4B.
```

**Gate criteria (autonomous mode):**
- [ ] `code_plan.md` exists with: stack, criteria map (full assertion ranges), execution waves, per-file strategy, cross-file data flow
- [ ] Stack section lists all non-standard packages; any newly installed packages noted
- [ ] Every acceptance criterion from `expectations.md` is in the criteria map with its assertion range
- [ ] Execution waves have clear dependency rationale
- [ ] Runtime estimates and device specifications present for all files
- [ ] Pedagogical alternatives from expectations mapped to files with trigger conditions
- [ ] `data_setup.py` runs successfully — data cached in `.cache/`
- [ ] HARD STOP if: data download fails entirely, or code_plan.md is missing the criteria map

---

### Step 4B: CODE — Implement

**This is NOT a single Task. The main session orchestrates parallel file agents.**

Read `code_plan.md` to extract execution waves. Then launch file agents wave by wave:

```
1. Create the logs directories: mkdir -p course/week{NN}_{TOPIC}/code/logs/notes
2. For each wave in code_plan.md:
   a. Launch one background Task agent per file in the wave (see prompt below)
   b. Periodically check code/logs/*.log for progress
   c. Wait for ALL agents in the wave to complete
   d. If a file agent reports MPS failure in its log:
      relaunch with --device cpu (counts as one of 3 attempts)
   e. If a file agent fails after 3 attempts:
      flag for user, continue with remaining files in wave
   f. Upstream context scan (before launching next wave):
      Read code/logs/notes/*_notes.md for files completed in this wave.
      Scan ALL sections — Deviations, Challenges, ML Choices, Approach,
      Threshold Reasoning. Extract anything a downstream agent would
      benefit from knowing: environment workarounds (thread limits,
      deadlock fixes), cache interface changes (filenames, column names,
      schemas), assertion adjustments, runtime gotchas, library quirks.
      Build an upstream context summary and inject it into the next
      wave's prompts via {UPSTREAM_CONTEXT}.
      Skip this step for the final wave (no downstream consumers).
3. After all waves complete:
   a. Verify all expected code/logs/*.log files exist
   b. Grep all logs for ✗ — any failures → stop and report
```

**Per-file agent launch prompt:**
```
You are a Step 4B file agent for Week {NN}: {TITLE}.
Your task: implement and test ONE file: {FILENAME}

Read these files:
1. course/guides/file_agent_guide.md — your code format and implementation rules.
2. course/week{NN}_{TOPIC}/code_plan.md — find the entry for {FILENAME}
   under "Per-File Implementation Strategy". This is your specification.
   Also read the criteria assigned to your file in the "Criteria Map".
3. course/week{NN}_{TOPIC}/code/data_setup.py — understand the data layer.
{UPSTREAM_CACHE_NOTE}

Specifically:
- Write the .py file at course/week{NN}_{TOPIC}/code/{SUBDIR}/{FILENAME}
- Run it{DEVICE_FLAG}, capturing stdout
- If assertions fail: diagnose, fix, re-run (up to 3 attempts)
- If results are pedagogically flat: check your code_plan.md entry for
  alternatives and try them before accepting
- Save stdout to course/week{NN}_{TOPIC}/code/logs/{FILENAME_STEM}.log
- Write implementation notes to
  course/week{NN}_{TOPIC}/code/logs/notes/{FILENAME_STEM}_notes.md
{UPSTREAM_CONTEXT}
Do NOT read blueprint.md, expectations.md, code_verification.md,
or any other guide files. file_agent_guide.md + code_plan.md are
your complete specification.
```

Set `{UPSTREAM_CACHE_NOTE}` to list specific cache files if the file depends on upstream outputs, e.g.:
```
3. Verify these upstream caches exist in course/week{NN}_{TOPIC}/code/.cache/:
   - gbm_predictions.parquet (from s3_gradient_boosting.py)
```

Set `{DEVICE_FLAG}` to ` with --device mps` or ` with --device cpu` based on code_plan.md, or leave blank for non-PyTorch files. For `device: gpu` files, set `{DEVICE_FLAG}` to ` with --device cuda` — the file agent guide explains the GPU workflow (local CPU debug first, then remote execution via `course/shared/kaggle_gpu_runner.py`).

Set `{UPSTREAM_CONTEXT}` based on the upstream context scan from the previous wave (step 2f above). For Wave 0 and Wave 1 agents (no prior waves), leave blank. For later waves, compile findings from upstream `logs/notes/*_notes.md` files into a summary block:
```
IMPORTANT — lessons from upstream agents that affect your work:

Interface changes (code_plan.md is stale on these):
- {file}: cached to {actual_name} (plan said {planned_name})
- {file}: column "{actual}" (plan said "{planned}")

Environment / runtime:
- {file}: LightGBM + PyTorch deadlock — add os.environ["OMP_NUM_THREADS"]="1"
  and torch.set_num_threads(1) before imports
- {file}: MPS failed on {op} — fell back to CPU

Assertion adjustments:
- {file}: {criterion} upper bound relaxed from {original} to {new} ({reason})
```
Include anything from the upstream notes that a downstream agent would benefit from — environment workarounds, cache interface changes, library gotchas, assertion adjustments, runtime discoveries. Omit sections that have no entries. If nothing relevant was found, leave blank.

**Gate criteria (autonomous mode):**
- [ ] All expected `code/logs/*.log` files exist (one per code file)
- [ ] All expected `code/logs/notes/*_notes.md` files exist
- [ ] Grep all `.log` files for `✗` — any failures → HARD STOP
- [ ] All expected `.py` files created in `lecture/`, `seminar/`, `hw/`
- [ ] Plot PNGs exist in `code/logs/plots/`
- [ ] HARD STOP if: any ✗ in logs, or missing code files

---

### Step 4C: CODE — Verify

**Launch prompt:**
```
You are the Step 4C verification agent for Week {NN}: {TITLE}.
You run in a FRESH context. Your role is independent quality audit —
you did not write the code and have no incentive to defend it.

Read these guide files:
1. course/guides/common.md
2. course/guides/code_verification.md (Step 4C section)
3. course/guides/rigor.md

Then read these artifacts:
- course/week{NN}_{TOPIC}/code_plan.md (criteria map + per-file strategy)
- course/week{NN}_{TOPIC}/expectations.md (for pedagogical intent)
- All .log files in course/week{NN}_{TOPIC}/code/logs/
- All _notes.md files in course/week{NN}_{TOPIC}/code/logs/notes/

Your job:
1. Verify criteria coverage — every criterion from the map is asserted
2. Pedagogical quality check — flag relaxed thresholds and flat results
3. Compile run_log.txt using bash (concatenate per-file logs in execution
   order — do NOT read logs into context and Write agentically)
4. Write execution_log.md per the spec

Present your findings to the main session.
```

**Gate criteria (autonomous mode):**
- [ ] `run_log.txt` exists (compiled from per-file logs) — grep for `✗`. If ANY `✗` → HARD STOP
- [ ] `execution_log.md` exists with: per-file notes, deviations, criteria coverage, pedagogical quality assessment
- [ ] Criteria coverage section accounts for every criterion (asserted, migrated, decomposed, or dropped with justification)
- [ ] Pedagogical quality check completed — threshold fidelity assessed, flat results flagged
- [ ] Open questions from `expectations.md` investigated and answered in execution_log.md
- [ ] HARD STOP if: any ✗ in run_log.txt, or pedagogical quality check reveals silently relaxed thresholds (threshold in code differs from code_plan.md range without justification)

---

### Step 5: OBSERVATION REVIEW (Phase 1)

**Launch prompt:**
```
You are the Step 5 observation agent for Week {NN}: {TITLE}.
You run in a FRESH context. Minimal anchoring.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/observation_review.md
3. course/guides/data.md

PHASE 1 — Read these and write Part 1 of observations.md:
- course/week{NN}_{TOPIC}/blueprint.md (structural context only)
- All plot PNGs in course/week{NN}_{TOPIC}/code/logs/plots/
- course/week{NN}_{TOPIC}/run_log.txt

Do NOT read any files beyond those listed above — in particular,
do NOT read execution_log.md, expectations.md, or any .py files in code/.

Write Part 1 (Visual + Numerical Observations) per the guide.
Save to course/week{NN}_{TOPIC}/observations.md.

STOP after Part 1. Do not proceed to Phase 2.
```

**Gate criteria (autonomous mode) — Phase 1:**
- [ ] `observations.md` exists with Part 1 content
- [ ] Each plot from `logs/plots/` is referenced and described
- [ ] Numerical observations reference `run_log.txt` values
- [ ] No references to expectations.md, execution_log.md, or .py files (information leak)
- [ ] SOFT FLAG if: descriptions seem superficial (fewer observations than plots)

**After Phase 1 gate approval:** Snapshot Part 1 for Step 7 notebook agents:
```bash
cp course/week{NN}_{TOPIC}/observations.md course/week{NN}_{TOPIC}/observations_p1.md
```
This copy is taken before Phase 2 appends criteria and benchmarks. Notebook agents use it for visual and numerical context. The narrative brief remains authoritative on teaching angles and framing.

---

### Step 5: OBSERVATION REVIEW (Phase 2)

**Launch prompt (new Task or resume Phase 1 agent):**
```
You are continuing the Step 5 observation for Week {NN}: {TITLE}.

Phase 1 of observations.md is approved and frozen.

Read these files:
1. course/guides/common.md
2. course/guides/observation_review.md
3. course/guides/data.md
4. course/week{NN}_{TOPIC}/observations.md (your Phase 1 output — do NOT modify Part 1)
5. course/week{NN}_{TOPIC}/expectations.md

Append Part 2 (Acceptance Criteria Audit, Benchmark Comparison,
Divergence Summary) to observations.md WITHOUT modifying Part 1.
```

**Gate criteria (autonomous mode) — Phase 2:**
- [ ] Part 1 is unmodified (compare first 5 lines to what was approved)
- [ ] Every acceptance criterion from expectations.md has a verdict
- [ ] Divergence summary captures meaningful gaps
- [ ] SOFT FLAG if: divergence summary is empty (suspiciously perfect results)

---

### Step 6: CONSOLIDATION

**Launch prompt:**
```
You are the Step 6 consolidation agent for Week {NN}: {TITLE}.
You run in a FRESH context.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/consolidation_spec.md

Then read ALL of these artifacts:
- course/week{NN}_{TOPIC}/blueprint.md
- course/week{NN}_{TOPIC}/expectations.md
- course/week{NN}_{TOPIC}/execution_log.md
- course/week{NN}_{TOPIC}/observations.md

You may run targeted web searches when you need a production benchmark
or domain verification that expectations.md doesn't provide — see the
"Targeted Web Search" section in consolidation_spec.md for scope rules.

Produce narrative_brief.md in course/week{NN}_{TOPIC}/ following the format
in consolidation_spec.md. Start from the Divergence Summary in
observations.md Phase 2 for your categorization work.
{FIRST_WEEK_NOTE}
```

Set `{FIRST_WEEK_NOTE}` to: "This is the first week being built. Note: do NOT update curriculum_state.md — that happens later, after the brief audit." — OR leave blank if curriculum_state.md already exists.

**Important:** The Step 6 agent must NOT update `curriculum_state.md`. The curriculum state is only written after Step 6¾ audit findings have been applied to the brief — so it reflects the final, audited teaching plan.

**Gate criteria (autonomous mode):**
- [ ] `narrative_brief.md` exists with Status field
- [ ] If Status is COMPLETE: every section categorized, teaching angles have frame/numbers/takeaway
- [ ] If Status is INCOMPLETE: flags have clear diagnosis, evidence, options, recommendation
- [ ] Searched benchmarks marked with source
- [ ] Note any `Confidence: low` sections — log them in orchestration.md for 6¾ awareness
- [ ] If COMPLETE: proceed to Step 6¾. Do NOT update `curriculum_state.md` — that happens after the audit.
- [ ] If INCOMPLETE → proceed to Step 6½. Do NOT update `curriculum_state.md` — that happens after the audit.
- [ ] HARD STOP if: >40% of code files flagged (need full rerun loop, not 6½)

---

### Step 6½: FLAG RESOLUTION

**This is NOT a Task. You (the orchestrator) execute it directly.**

Read `course/guides/flag_resolution.md` for the full process. Summary:

1. **Diagnose & Propose** — Read flags from narrative_brief.md, diagnose root causes, present fix options
2. **Targeted Code Edits + Gate** — Fix affected `.py` files, rerun only those scripts, present before/after metrics
3. **Artifact Cascade** — Update: run_log.txt → execution_log.md → observations.md → narrative_brief.md (do NOT update curriculum_state.md here — that happens after Step 6¾)

In **supervised mode**: present options at Phase 1, present before/after table at Phase 2 gate.
In **autonomous mode**: choose the recommended option at Phase 1, proceed through Phase 2 if metrics improve, cascade through Phase 3.

**HARD STOP (even in autonomous):** if before/after metrics are WORSE, or if the fix introduces new ✗ lines in run_log.txt.

**After resolution:** The brief is now ratified (Status: COMPLETE). Proceed to Step 6¾ for the audit.

---

### Step 6¾: BRIEF AUDIT

**Launch prompt:**
```
You are the Step 6¾ brief audit agent for Week {NN}: {TITLE}.
You run in a FRESH context. Your role is adversarial — you are
looking for problems in the narrative brief, not confirming it.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/brief_audit_spec.md
3. course/guides/rigor.md

Then read ALL of these artifacts IN THIS ORDER (order matters —
see brief_audit_spec.md for why):
1. course/COURSE_OUTLINE.md
2. course/week{NN}_{TOPIC}/expectations.md
3. course/week{NN}_{TOPIC}/execution_log.md
4. course/week{NN}_{TOPIC}/observations.md
5. course/week{NN}_{TOPIC}/narrative_brief.md
6. course/week{NN}_{TOPIC}/blueprint.md

Produce brief_audit.md in course/week{NN}_{TOPIC}/ following the format
in brief_audit_spec.md. Apply all 7 checks to every section, then
produce the viability assessment for every section.
```

**Gate criteria (autonomous mode):**
- [ ] `brief_audit.md` exists with findings categorized as critical/moderate/minor and viability verdicts for every section
- [ ] Apply ALL findings — critical, moderate, AND minor. The audit exists to improve the brief; don't ignore suggestions just because they're low severity.
- [ ] For each finding: if the fix is a prose edit to `narrative_brief.md`, make the edit yourself and log it in orchestration.md
- [ ] If the audit says a low-confidence section should be flagged → HARD STOP — escalate to user (flagging triggers 6½, which needs user authorization)
- [ ] If a finding requires code changes (not just prose) → HARD STOP — escalate to user
- [ ] **Viability verdicts:** Check the Viability Assessment section of `brief_audit.md`:
  - **Viable** sections: no structural action needed (finding fixes are sufficient)
  - **Reframe** sections: rewrite the section's teaching angle in `narrative_brief.md` following the audit's recommended direction. Log the reframe in orchestration.md.
  - **Redesign** sections → HARD STOP — escalate to user. Redesign requires rework at the code/expectations level, which needs user authorization and scope decision.
  - **Remove** sections → HARD STOP — escalate to user. Removal is a structural decision the user must approve. Present the audit's downstream impact analysis.
- [ ] If no findings and all sections Viable: APPROVED with no edits needed
- [ ] **After all findings and viable reframes are applied:** update `curriculum_state.md`. This is the ONE place in the pipeline where curriculum state gets written — after the brief is final and audited. If the file doesn't exist yet, create it with the header `# Curriculum State`. Append a Week {NN} entry summarizing what was actually taught (based on the now-final `narrative_brief.md`).

**In supervised mode:** present the full list of findings AND the viability assessment to the user. Apply all that the user approves (default: apply all). For Reframe/Redesign/Remove verdicts, present the audit's justification and downstream impact analysis. Then update `curriculum_state.md` and show the user the entry.

**After the gate:** The brief is now final and curriculum_state.md is up to date. Proceed to Step 7.

---

### Step 7: NOTEBOOKS

**Launch 3 parallel Tasks:**

LECTURE prompt:
```
You are the Step 7 LECTURE notebook agent for Week {NN}: {TITLE}.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/notebook.md
3. course/guides/cell_design.md

Then read these artifacts:
- course/week{NN}_{TOPIC}/blueprint.md
- course/week{NN}_{TOPIC}/narrative_brief.md
- course/week{NN}_{TOPIC}/observations_p1.md (Phase 1 observations —
  visual descriptions and numerical results. Use for writing accurate
  prose around plots. The narrative brief remains authoritative on
  teaching angles, framing, and key number selection.)
- course/shared/data.py (the shared data layer — understand what each
  load function does: filtering, ticker lists, parameters. Your notebook
  must replicate this behavior with direct API calls.)
- course/week{NN}_{TOPIC}/code/data_setup.py
- All files in course/week{NN}_{TOPIC}/code/lecture/

Produce course/week{NN}_{TOPIC}/_build_lecture.md following the builder
format in notebook.md. The file is markdown with ~~~python fenced code
blocks — no Python wrapper, no build logic.
```

SEMINAR prompt: (same structure, replace `lecture` with `seminar`)
```
You are the Step 7 SEMINAR notebook agent for Week {NN}: {TITLE}.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/notebook.md
3. course/guides/cell_design.md

Then read these artifacts:
- course/week{NN}_{TOPIC}/blueprint.md
- course/week{NN}_{TOPIC}/narrative_brief.md
- course/week{NN}_{TOPIC}/observations_p1.md (Phase 1 observations —
  visual descriptions and numerical results. Use for writing accurate
  prose around plots. The narrative brief remains authoritative on
  teaching angles, framing, and key number selection.)
- course/shared/data.py (the shared data layer — understand what each
  load function does: filtering, ticker lists, parameters. Your notebook
  must replicate this behavior with direct API calls.)
- course/week{NN}_{TOPIC}/code/data_setup.py
- All files in course/week{NN}_{TOPIC}/code/seminar/

Produce course/week{NN}_{TOPIC}/_build_seminar.md following the builder
format in notebook.md. The file is markdown with ~~~python fenced code
blocks — no Python wrapper, no build logic.
```

HOMEWORK prompt: (same structure, replace `seminar` with `hw`)
```
You are the Step 7 HOMEWORK notebook agent for Week {NN}: {TITLE}.

Read these guide files in full before doing anything:
1. course/guides/common.md
2. course/guides/notebook.md
3. course/guides/cell_design.md

Then read these artifacts:
- course/week{NN}_{TOPIC}/blueprint.md
- course/week{NN}_{TOPIC}/narrative_brief.md
- course/week{NN}_{TOPIC}/observations_p1.md (Phase 1 observations —
  visual descriptions and numerical results. Use for writing accurate
  prose around plots. The narrative brief remains authoritative on
  teaching angles, framing, and key number selection.)
- course/shared/data.py (the shared data layer — understand what each
  load function does: filtering, ticker lists, parameters. Your notebook
  must replicate this behavior with direct API calls.)
- course/week{NN}_{TOPIC}/code/data_setup.py
- All files in course/week{NN}_{TOPIC}/code/hw/

Produce course/week{NN}_{TOPIC}/_build_hw.md following the builder
format in notebook.md. The file is markdown with ~~~python fenced code
blocks — no Python wrapper, no build logic.
```

**After all 3 Tasks complete, run the builder:**
```bash
for f in course/week{NN}_{TOPIC}/_build_*.md; do python3 nb_builder.py "$f" && rm "$f"; done
```

`nb_builder.py` now **rejects** builds that violate structural constraints (consecutive code cells in lectures, prose ratio below minimum, code runs >2 in seminar/hw). If a build is rejected, fix the `_build_*.md` file and rebuild.

**Gate criteria (autonomous mode):**
- [ ] All three `.ipynb` files exist (nb_builder.py exited 0 for each)
- [ ] No REJECT lines in nb_builder.py output
- [ ] SOFT FLAG if: quality warnings exist (oversized cells, thin prose — log them)
- [ ] HARD STOP if: any builder script was rejected (exit code 1)

---

### Step 8: SHARED EXPORT

**Launch prompt:**
```
You are the Step 8 shared export agent for Week {NN}: {TITLE}.

Your job: identify reusable functions or classes in this week's code
that would benefit future weeks, and extract them to course/shared/.

Read this guide file first:
1. course/guides/rigor.md — the quality standard for all course code.
   Exported shared utilities must embody these standards since every
   future week will import them.

Then read these artifacts:
2. course/week{NN}_{TOPIC}/narrative_brief.md — understand what the
   week actually teaches and which techniques are novel
3. course/week{NN}_{TOPIC}/code_plan.md — understand the file structure
   and what each file implements
4. All .py files in course/week{NN}_{TOPIC}/code/lecture/,
   code/seminar/, and code/hw/
5. course/shared/__init__.py — understand the current shared module
   inventory and public API
6. Skim the existing shared modules that are closest to the week's
   domain (e.g., metrics.py, temporal.py, evaluation.py, etc.)

Extraction criteria — export a function/class ONLY if it meets ALL of:
- General-purpose: useful beyond this specific week's dataset or model
- Non-trivial: more than a thin wrapper around a single library call
- Not already covered: no equivalent exists in shared/ (check carefully)
- Clean interface: takes standard inputs (DataFrames, arrays, configs),
  not week-specific column names or hardcoded parameters

For each candidate:
1. Decide which shared module it belongs to (existing or new)
2. Generalize it: remove week-specific defaults, add parameters for
   anything that was hardcoded, add a clear docstring
3. Add it to the appropriate shared module
4. Update course/shared/__init__.py if the module is new or the
   function should be in the public API
5. Do NOT modify the week's code files — shared/ exports are for
   future weeks, not retroactive refactoring

If you find NOTHING worth exporting (the week only used standard
library patterns with no novel utilities), that's a valid outcome.
Write a short note explaining why.

Produce a summary of what you exported (or why nothing was exported)
and present it to the main session. Format:

## Shared Utility Export — Week {NN}
### Exported
- `shared/{module}.py`: `function_name()` — one-line description
### Skipped (considered but rejected)
- `function_name` in {file} — reason not exported
### No candidates
- [if nothing was worth exporting, explain why]
```

**Gate criteria (autonomous mode):**
- [ ] Agent completed and returned a summary
- [ ] Any exported functions are general-purpose (not week-specific)
- [ ] No existing shared functions were duplicated
- [ ] `__init__.py` updated if new public functions were added
- [ ] Week code files were NOT modified (read-only extraction)
- [ ] SOFT FLAG if: nothing was exported (valid but worth noting in orchestration.md)
- [ ] HARD STOP if: agent modified week code files, or introduced import errors in shared/

**After gate approval (both modes):** Regenerate the shared API reference:
```bash
python3 course/shared/generate_api.py
```
This updates `course/shared/API.md` — the auto-generated reference that Step 4A agents read to discover available shared infrastructure. Must be run after every Step 8 that exports new functions.

**In supervised mode:** Present the export summary to the user. Show each exported function with its signature and one-line description. Ask: "Approve these exports to shared/? (approve / revise / skip)"

**In autonomous mode:** Review the summary. If all criteria pass, APPROVED. Log the exported functions in orchestration.md.

---

## Pipeline Complete

After Step 8 gate is approved, write to `orchestration.md`:

```markdown
## PIPELINE COMPLETE
- All artifacts produced
- [List any SOFT FLAGs that were noted during the run]
- Total steps: N (including any reruns)
```

In **supervised mode**: tell the user the week is complete and list any soft flags for their review.
In **autonomous mode**: the user should review `orchestration.md` and spot-check artifacts.
