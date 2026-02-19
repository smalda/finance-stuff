# Pipeline Redesign — Guide System v2

> **Status:** Approved architecture. Implementation pending.
>
> This document captures the complete redesigned pipeline for building weekly course content. It replaces the previous 4-phase system (week_build_guide.md and its satellite guides) with a 7-step pipeline where each step produces an immutable artifact and no step ever edits a prior step's output.

---

## Why the Redesign

The original guide system (6 files, ~3,000 lines) suffered from:

1. **Massive duplication.** The same content (e.g., "notebooks never import from data_setup.py") appeared in 3 different guides. The orchestrator guide restated ~80% of each specialist guide despite promising not to.

2. **Broken encapsulation.** Phase 4 (notebook) instructions lived inside the Phase 3 (code verification) guide. The orchestrator guide contained Phase 2 agent instructions. Content leaked across every boundary.

3. **No consolidation step.** The README made educated guesses about expected results. The code produced actual results. These often diverged, but there was no dedicated step to reconcile them — it was handled ad-hoc inside Phase 3's "contradiction resolution," which could trigger README edits (breaking artifact immutability).

4. **README was overloaded.** It simultaneously served as creative vision, task spec, outcome predictor, and acceptance criteria source. The outcome predictions were frequently wrong (the README can't know what the code will produce), causing cascading revisions.

5. **Plot verification was unreliable.** Vision models misread plots after long contexts. The plot review step was buried inside Phase 3 rather than isolated in a clean context.

6. **Prior-week coupling.** Each new README had to read N prior READMEs to check prerequisites — wasteful and fragile.

---

## Design Principles

1. **One step, one artifact, one responsibility.** Each pipeline step produces exactly one artifact. That artifact is immutable once approved.

2. **Nothing flows backward.** If Step 6 discovers a problem, it either documents it (teaching angle) or flags a code bug (loop to Step 4). It never edits the blueprint, expectations, or research notes.

3. **Each agent reads at most 2 guide files.** Clean context, focused instructions, no noise.

4. **Separate what you know from what you predict from what you get.** Research = what's true in finance. Blueprint = what we'd ideally teach. Expectations = what we predict given our data. Code = what we actually get. Consolidation = what we teach given all of the above.

5. **Observation review always gets a clean context.** Plot reading degrades catastrophically in long contexts, and numerical output deserves fresh-eyes analysis. The observation step reviews both visual and numerical output in a clean context with minimal input.

6. **The course creator doesn't need to know finance.** The pipeline must be robust to the orchestrator not being able to validate financial correctness. External benchmarks, structured cross-checks, and explicit categorization of divergences serve as the safety net.

---

## The 7-Step Pipeline

```
  COURSE_OUTLINE.md
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: RESEARCH                                       │
│  "What's current in this domain?"                       │
│  → research_notes.md                                    │
└────────────────────────┬────────────────────────────────┘
                         │  ⏸ approval gate
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: BLUEPRINT                                      │
│  "What's the ideal teaching week?"                      │
│  → blueprint.md + curriculum_state.md update             │
└────────────────────────┬────────────────────────────────┘
                         │  ⏸ approval gate
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: EXPECTATIONS                                   │
│  "Given our data reality, what should we expect?"       │
│  → expectations.md                                      │
└────────────────────────┬────────────────────────────────┘
                         │  ⏸ approval gate
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: CODE                                           │
│  "Make it work."                                        │
│  → code/ + run_log.txt + execution_log.md               │
└────────────────────────┬────────────────────────────────┘
                         │  ⏸ approval gate
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5: OBSERVATION REVIEW  (clean context)            │
│  "What did the code actually produce?"                  │
│  → observations.md                                      │
└────────────────────────┬────────────────────────────────┘
                         │  ⏸ user verifies observations
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 6: CONSOLIDATION  (fresh context)                 │
│  "What story are we actually telling?"                  │
│  → narrative_brief.md                                   │
│                                                         │
│  May loop back to Step 4 if code issues found.          │
└────────────────────────┬────────────────────────────────┘
                         │  ⏸ approval gate
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 7: NOTEBOOKS  (3 parallel agents)                 │
│  "Teach it."                                            │
│  → lecture.ipynb, seminar.ipynb, hw.ipynb               │
└─────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Specification

### Step 1: RESEARCH

**Purpose:** Get up to date on the current state of the domain. Pure knowledge gathering — no task design, no data assessment, no outcome prediction.

**Agent reads:**
- `research.md` (guide)
- Course outline entry for this week

**Produces:** `research_notes.md` inside the week folder. [Immutable once approved.]

**Responsibilities:**
- Domain knowledge update (papers, tools, libraries, what's deprecated, what's new)
- Accessibility check (are libraries free? maintained?)
- Current best practices and practitioner reality
- University course coverage for this topic

**Does NOT:**
- Design tasks or exercises
- Predict what our code will produce
- Assess data universe implications for specific analyses
- Set acceptance criteria

**Approval gate:** User reviews research_notes.md. Confirms no blockers for the blueprint step.

---

### Step 2: BLUEPRINT

**Purpose:** Design the ideal teaching week. The creative vision — what to teach, in what order, with what narrative arc, exercises, and career connections.

**Agent reads:**
- `blueprint_spec.md` (guide)
- `task_design.md` (shared guide — task types, difficulty calibration)
- Course outline entry
- `research_notes.md` (from Step 1)
- `curriculum_state.md` (what prior weeks have already taught)

**Produces:**
- `blueprint.md` inside the week folder. [Immutable once approved.]
- An appended entry in `course/curriculum_state.md` summarizing what this week teaches.

**Responsibilities:**
- Learning objectives (concrete, assessable)
- Prerequisites (referencing curriculum_state, not prior blueprints)
- Narrative arc (the week's central tension and resolution)
- Lecture outline (sections with hooks, concepts, demonstrations, bridges)
- Seminar outline (exercises with questions, task types, expected insights)
- Homework outline (deliverables with mission framing, task types)
- Career connections (concrete role mappings)
- Data sources to use (but NOT what to expect from them)
- Key papers & references

**Does NOT:**
- Set acceptance criteria (that's Step 3)
- Predict specific numerical outcomes
- Branch on possible data-dependent results
- Assess sandbox limitations vs. production reality

**The blueprint is a pure creative/structural document.** It describes the ideal week as if data constraints didn't exist. It knows which data SOURCES to use (yfinance, Ken French, FRED) but not what specific numbers will emerge.

**Approval gate:** User reviews blueprint.md and curriculum_state.md update.

---

### Step 3: EXPECTATIONS

**Purpose:** Assess what our sandbox data can and can't do, given the blueprint's ideal plan. Set acceptance criteria. Find production benchmarks. Identify open questions that only code can answer.

**Agent reads:**
- `expectations_spec.md` (guide)
- `task_design.md` (shared guide)
- `blueprint.md` (from Step 2)
- `research_notes.md` (from Step 1, for data constraint information)

**Produces:** `expectations.md` inside the week folder. [Immutable once approved.]

**Responsibilities:**
- **Data plan** — universe size, date range, frequency, with options and tradeoffs. User approves the data plan before proceeding.
- **Known constraints** — structural limitations of our data (e.g., "S&P 500 has no real small-caps, which means SMB factor construction will lack true size dispersion")
- **Acceptance criteria** — per section, exercise, and deliverable. Verifiable conditions the code must satisfy. Uses approximate ranges, not exact values.
- **Production benchmarks** — what results look like with institutional data (CRSP, Bloomberg), cited from authoritative literature. These give the consolidation agent a reference point.
- **Open questions** — things we genuinely can't predict until code runs (e.g., "Will cross-sectional R² be high enough to demonstrate explanatory power, or will we need to aggregate across months?")

**Does NOT:**
- Write code
- Change the blueprint
- Decide teaching strategy for divergences (that's Step 6)

**Approval gate:** User reviews expectations.md, approves the data plan.

---

### Step 4: CODE

**Purpose:** Implement and verify all code that will appear in the week's notebooks. Pure engineering — no interpretation, no teaching strategy, no narrative.

**Agent reads:**
- `code_verification.md` (guide)
- `task_design.md` (shared guide — solution strategies per task type)
- `blueprint.md` (from Step 2 — what to implement)
- `expectations.md` (from Step 3 — acceptance criteria, data plan)

**Produces:**
- `code/` directory inside the week folder:
  - `data_setup.py` — shared data downloads and caching
  - `lecture/s1_*.py`, `s2_*.py`, ... — one file per lecture section
  - `seminar/ex1_*.py`, `ex2_*.py`, ... — one file per seminar exercise
  - `hw/d1_*.py`, `d2_*.py`, ... — one file per homework deliverable
  - `.cache/` — plot PNGs and cached data (gitignored)
- `run_log.txt` — captured stdout from running all scripts in order. Raw machine-captured output (not agent-written). The ground truth of what the code produced.
- `execution_log.md` — the agent's developer report: implementation analysis, design choices, threshold rationale, open question resolutions.
  [Code is mutable until user approves. Both logs are regenerated on each run cycle.]

**Code file format (2 parts):**
1. **Cell-marked implementation** — `# ── CELL: name` structural markers dividing code into notebook cells. Markers are structural only — no interpretive annotations. Interpretation of what results mean is Step 6's job.
2. **Verification block** — `if __name__ == "__main__":` block with assertions, structured results output (printed to stdout → captured in run_log.txt), and plot saving. Never included in notebooks.

**The code agent iterates internally** until all assertions pass. If stuck after 3 attempts on a file, it flags the issue for the user.

**Responsibilities:**
- Write all code files with CELL markers and verification blocks
- Run all files, iterate until assertions pass
- Generate and save all plots to `.cache/`
- Capture structured stdout to `run_log.txt` (clean final run)
- Write `execution_log.md` (engineering analysis, not output transcription)
- Investigate open questions from expectations.md and report findings

**Does NOT:**
- Write prose, markdown, or narrative
- Interpret what results mean (that's Step 6)
- Create .ipynb files
- Modify the blueprint or expectations
- Decide teaching angles for divergences (that's Step 6)

**Approval gate:** User reviews run_log.txt and execution_log.md. All assertions pass. Plots saved. Code approved (becomes immutable).

---

### Step 5: OBSERVATION REVIEW

**Purpose:** Review both visual output (plot PNGs) and numerical output (run_log.txt) in a clean context. Produce observations — what the code actually produced — for the user to verify. This step observes; it does not interpret. Interpretation is Step 6's job.

**This step always runs in a fresh context with minimal input.**

**Fresh agent reads:**
- `observation_review.md` (guide — prompt template + format spec)
- Plot PNGs from `code/.cache/`
- `run_log.txt` (from Step 4 — structured execution output with metrics and plot metadata)

**Produces:** `observations.md` inside the week folder. [Immutable once user verifies.]

**The agent reviews two things:**

1. **Visual observations** — What do the plots show? Describe patterns, trends, axis values, line counts, and anything visually notable. No teaching interpretation — just accurate descriptions of what's visible.

2. **Numerical observations** — What do the run_log.txt metrics say? Flag values that are at range extremes, metrics that are consistent (or inconsistent) across files, and any warnings or unexpected output. Cross-reference numerical metrics against what the plots show.

**Format:**
```markdown
## Visual Observations

### s3_ff5_cumulative.png
![](code/.cache/s3_ff5_cumulative.png)

**Observes:** Five lines showing cumulative factor returns. MKT dominates,
reaching ~$4.2 by 2024. HML declines visibly from 2017 onward, ending negative.
SMB is nearly flat. RMW and CMA show modest positive trends.

**Cross-check with run_log:** MKT final = $4.21 ✓ | HML final = -$0.32 ✓ |
SMB final = $1.04 ✓ (matches visual)

## Numerical Observations

### Cross-File Consistency
- SMB correlation = 0.04 in s3, 0.05 in s5 — consistent ✓
- All files report 192 months of data — consistent ✓

### Notable Values
- s1: R² = 0.31 — low end of [0.15, 0.45] range. Passes but barely.
- s3: SMB-KF correlation = 0.04 — passes the < 0.3 criterion easily,
  but extremely low. The factor is essentially uncorrelated with Ken French.
- s6: No characteristic reaches |t| > 1.5 in-sample.

### Warnings / Unexpected Output
[Any stderr, deprecation warnings, or unexpected print output from run_log.txt]
```

**User verification process:**
1. Open observations.md
2. For each plot: compare the inline image against the agent's visual observation
3. For numerical observations: spot-check a few values against run_log.txt
4. Flag any mismatches or mischaracterizations
5. Amend observations as needed
6. Approve the document

The user does NOT need to verify financial interpretations — only that the descriptions match what the plots show and the numbers say.

**Gate:** User verifies observations. Document approved.

---

### Step 6: CONSOLIDATION

**Purpose:** Reconcile the ideal (blueprint) with the actual (code results). Produce the definitive teaching plan for notebooks.

**This step always runs in a fresh context.**

**Fresh agent reads:**
- `consolidation_spec.md` (guide)
- `blueprint.md` (from Step 2 — narrative arc, structure, creative vision)
- `expectations.md` (from Step 3 — what we predicted, acceptance criteria, benchmarks)
- `run_log.txt` (from Step 4 — raw execution output with actual values)
- `execution_log.md` (from Step 4 — implementation analysis, design choices, open question answers)
- `observations.md` (from Step 5 — what the plots show and what the numbers say)

**Produces:** `narrative_brief.md` inside the week folder. [Immutable once approved.]

**Structure — per section/exercise/deliverable:**
```markdown
### Section 3: Factor Construction

**Blueprint intended:** Build SMB and HML from our data, compare to Ken French
**Expected (from expectations.md):** SMB correlation < 0.2 due to S&P 500 universe
**Actual (from run_log + observations):** SMB correlation = 0.04; cumulative SMB nearly flat; HML declining from 2017
**Category:** Data limitation
**Teaching angle:** Universe bias — S&P 500 "small" stocks are the market's "large."
  Frame as: "If you ran this at a fund with CRSP, correlation would be 0.75+.
  We're measuring large-vs-very-large, not small-vs-large."
**Key numbers for prose:** corr = 0.04, production benchmark = 0.75,
  smallest S&P 500 stock ≈ $10B market cap
```

**Divergence categories:**

| Category | Meaning | How notebook frames it |
|----------|---------|----------------------|
| **Matches** | Code confirms expectations | Teach straightforwardly |
| **Data limitation** | Sandbox can't replicate production | "Here's what we got, here's what a fund sees, here's why" |
| **Real phenomenon** | The result IS what production looks like | "Your code is correct. This IS what happens." |
| **Expectation miscalibration** | Step 3 predicted wrong, but code is fine | Adjusted framing in the brief; upstream docs unchanged |

**The consolidation → code loop:**

If the consolidation agent identifies an actual code bug (not a divergence, but broken code):
1. Flag the specific issue to the user
2. User approves looping back
3. Re-run Step 4 (affected files only) → Step 5 (affected observations) → Step 6 (fresh consolidation)

Only code bugs trigger the loop. "Results differ from expectations" is NEVER a code bug — it's a teaching opportunity documented in the brief.

**Approval gate:** User reviews narrative_brief.md. Once approved, this becomes the authoritative teaching plan.

---

### Step 7: NOTEBOOKS

**Purpose:** Turn the consolidated plan into teaching notebooks. Prose, layout, and voice — using verified code verbatim.

**3 parallel agents, one per notebook:**

Each agent reads:
- `notebook.md` (guide — voice, structure, quality standards)
- `blueprint.md` (from Step 2 — narrative arc, hooks, bridges, career connections)
- `expectations.md` (from Step 3 — data plan rationale, production benchmarks, known constraints)
- `narrative_brief.md` (from Step 6 — results, teaching angles, key numbers)
- `code/data_setup.py` (to understand data downloads)
- The relevant `code/` subfolder (e.g., `code/lecture/` for the lecture agent)

**Produces:** `lecture.ipynb`, `seminar.ipynb`, `hw.ipynb` inside the week folder.

**Authority rules:**
- **Blueprint** is authoritative on: structure, narrative arc, hooks, bridges, career connections, exercise framing
- **Expectations** provides background context: data plan rationale, production benchmarks with citations, known constraints in full detail. Used to enrich prose, not to override the brief.
- **Narrative brief** is authoritative on: results, teaching angles, key numbers, how to frame divergences
- **Code** is authoritative on: all computation. Code is used verbatim — never modified.
- When blueprint and brief conflict on data-dependent content, the **brief wins**.

**What notebook agents do:**
1. Read `data_setup.py` to understand what data to download, then write equivalent direct API calls in the notebook (notebooks are standalone — no imports from code files, no caching logic)
2. Read each code file in order. For each `# ── CELL:` block, create: a markdown cell before, a code cell (verbatim), and a markdown cell after. Use `narrative_brief.md` for what to write in prose cells — teaching angles, key numbers, how to frame results. Use the code itself to understand what each cell does.
3. Write prose following the notebook guide's voice and structure rules
4. Add section headers, transitions, opening, and closing

**What notebook agents do NOT do:**
- Include the `if __name__` block or `# ── CELL:` markers
- Invent new code not present in verified files
- Modify computation logic
- Reference the narrative brief, execution log, or observations as documents (their content is woven into prose naturally)

**Gate:** User reviews all three notebooks.

---

## Artifact Summary

| Step | Artifact | Lives in | Immutable after |
|------|----------|----------|-----------------|
| 1 | `research_notes.md` | `weekNN_topic/` | Step 1 approval |
| 2 | `blueprint.md` | `weekNN_topic/` | Step 2 approval |
| 2 | `curriculum_state.md` (updated) | `course/` | Step 2 approval (amended if needed after Step 6) |
| 3 | `expectations.md` | `weekNN_topic/` | Step 3 approval |
| 4 | `code/` directory | `weekNN_topic/code/` | Step 4 approval (may loop from Step 6) |
| 4 | `run_log.txt` | `weekNN_topic/` | Step 4 approval (regenerated on loop) |
| 4 | `execution_log.md` | `weekNN_topic/` | Step 4 approval (regenerated on loop) |
| 5 | `observations.md` | `weekNN_topic/` | Step 5 user verification |
| 6 | `narrative_brief.md` | `weekNN_topic/` | Step 6 approval |
| 7 | `*.ipynb` | `weekNN_topic/` | Step 7 review |

---

## Guide File Architecture

Each guide file maps to one pipeline step. No guide is consumed by more than 2 agent types. No agent reads more than 2 guide files.

```
course/guides/
├── common.md               ~55 lines    System-level: pipeline map, info flow, notebook
│                                         progression, audience. Read by ALL agents.
│                                         Does NOT count toward the 2-guide limit.
├── pipeline.md              ~80 lines    Human cheat sheet: 7 steps, gates, agent inputs
├── research.md              ~350 lines   Step 1 agent guide
├── blueprint_spec.md        ~150 lines   Step 2 agent guide
├── expectations_spec.md     ~200 lines   Step 3 agent guide (NEW)
├── task_design.md           ~160 lines   Shared by Steps 2, 3, 4
├── code_verification.md     ~430 lines   Step 4 agent guide
├── observation_review.md    ~100 lines   Step 5 agent guide (plots + numerical review)
├── consolidation_spec.md    ~200 lines   Step 6 agent guide (NEW)
└── notebook.md              ~775 lines   Step 7 agent guide (merged voice + structure)
```

**What each agent reads:**

All agents read `common.md` (system-level — pipeline map, information flow, notebook progression, audience). This does not count toward the 2-guide limit below.

| Step | Guide file(s) | Plus these artifacts |
|------|--------------|---------------------|
| 1: Research | `research.md` | Course outline entry |
| 2: Blueprint | `blueprint_spec.md` + `task_design.md` | Course outline + research_notes + curriculum_state |
| 3: Expectations | `expectations_spec.md` + `task_design.md` | Blueprint + research_notes |
| 4: Code | `code_verification.md` + `task_design.md` | Blueprint + expectations |
| 5: Observation Review | `observation_review.md` | Plot PNGs + run_log.txt |
| 6: Consolidation | `consolidation_spec.md` | Blueprint + expectations + run_log + execution_log + observations |
| 7: Notebooks | `notebook.md` | Blueprint + expectations + narrative_brief + code/ + data_setup.py |

---

## The curriculum_state.md Document

A cumulative document tracking what the course has established. Updated after each blueprint approval. Lives at `course/curriculum_state.md`.

**Structure (per completed week):**

```markdown
## After Week N: [Title]

### Concepts Taught
- [Concept]: [1-line description of depth covered]

### Skills Acquired
- [Skill]: [what students can now do]

### Finance Vocabulary Established
[comma-separated list of terms introduced and defined]

### Artifacts Built
- [Artifact name]: [what it does, which notebook built it]

### Key Caveats Established
- [Caveat]: [what students explicitly know about limitations]

### Reusable Downstream
- Can assume: [skills/concepts that don't need re-teaching]
- Can reference by name: [artifacts, patterns]
- Don't re-teach: [concepts fully covered]
```

Blueprint agents for subsequent weeks read this instead of prior blueprints. This gives them a focused summary (~15-20 lines per week) of what's been established, without the noise of full blueprint documents.

---

## Migration Notes

The old guide system (6 files in `course/guides/`) has been moved to `legacy/old_guides/` for reference. The new guides will be written from scratch following this specification — not by editing the old files, which are structurally incompatible with the new architecture.

Content from the old guides will be selectively incorporated into the new guides where it remains relevant, particularly:
- The CELL marker pattern and verification block structure (from old `code_verification_guide.md`, simplified from 3 layers to 2 parts)
- The voice specification and 10 rules (from old `writing_guidelines.md`)
- The notebook type specifications and quality checklist (from old `notebook_creation_guide.md`)
- Task type definitions (from old `task_design_guide.md`)
- Research tier assignments (from old `phase1_research_guide.md`)
