# Flag Resolution — Step 6½ Guide

> **Consumer:** The orchestrator (you), working interactively. Unlike other guides in this directory, this guide addresses the orchestrator directly — not a launched agent — because Step 6½ is executed interactively. It's a scoped patch workflow you execute yourself when Step 6 produces a brief with FLAGS. It avoids rerunning full pipeline stages (which are expensive) by doing targeted code fixes + artifact cascade over only the affected sections.

---

## When to Use

Step 6 produced a `narrative_brief.md` with `Status: INCOMPLETE (N flags pending resolution)`. The normal pipeline says: resolve flags → rerun Steps 4→5→6. This guide replaces that full loop with a minimal-blast-radius alternative: fix only what's broken, rerun only what changed, update only the affected artifact sections.

**Do NOT use this guide if:**
- The flag requires a fundamentally different implementation approach (new data source, different exercise structure) — that's a real Step 4 rerun
- More than ~40% of the code files are affected — at that point the full loop is cheaper than patching
- The flag is an observation error only (no code change needed) — just correct `observations.md` and rerun Step 6

---

## Inputs

- **`narrative_brief.md`** — the flagged version. Each `### FLAG N` section contains: Diagnosis, Evidence, Affected teaching, Options, Recommendation.
- **`blueprint.md`** — for original section intents and narrative arc (needed when filling in previously-stubbed sections and updating the arc reconciliation)
- **Flagged code files** — paths referenced in FLAG sections
- **`run_log.txt`** — for before-metrics
- **`execution_log.md`** — for context on original implementation choices
- **`observations.md`** — to identify which observations and tables need updating
- **`expectations.md`** — reference for acceptance criteria ranges when updating observations.md audit tables

---

## The Process: 3 Phases, 1 Gate

### Phase 1 — Diagnose & Propose

1. Read `narrative_brief.md`, extract all `### FLAG N` sections
2. Read each flagged code file and its corresponding `run_log.txt` output
3. Diagnose the root cause per flag using domain + statistical reasoning
4. Present fix recommendations to the user, with options where applicable:
   - **Option A: Fix code** — when the issue is a code bug, wrong approach, or missing industry-standard practice
   - **Option B: Accept result, reframe narrative** — when the result is correct but the teaching angle needs rethinking (e.g., a reversal of expected direction that is itself pedagogically valuable)
   - Options are not exclusive — a flag may need both a code fix AND a narrative reframe

**User approval required before any edits.**

### Phase 2 — Targeted Code Edits + Gate

1. Edit **only** the affected `.py` files — minimal diff, no refactoring beyond the fix
2. If a fix mechanically changes a secondary metric (e.g., z-score std compression from clipping), adjust the assertion threshold with a comment explaining why — don't suppress the signal
3. Rerun **only** the edited scripts. Capture stdout.
4. **GATE:** Present a before/after metric comparison table to the user:

```
| Metric              | Before | After  | Verdict         |
|---------------------|--------|--------|-----------------|
| max_abs_z           | 7.32   | 3.00   | Fixed (target)  |
| z_std_min           | 1.00   | 0.84   | Expected (clip) |
| n_features          | 9      | 8      | Correct (drop)  |
```

The table must include:
- Every metric that changed in the rerun output
- A verdict column explaining whether each change is intentional/expected/concerning
- Any new assertion threshold adjustments and why

**User must approve before Phase 3.** If metrics look wrong, iterate on the code fix.

### Phase 3 — Artifact Cascade

Update each artifact in dependency order. Only touch sections affected by the flags.

#### 1. `run_log.txt`
- Splice in new stdout for corrected scripts, replacing the old entries
- Add `[CORRECTED]` marker to updated section headers
- Do not touch entries for unaffected scripts

#### 2. `execution_log.md`
- Rewrite the per-file section for each corrected script
- Include a `**Correction (post-consolidation):**` note explaining what changed and why
- Add a per-file section if one was missing (common for homework deliverables)

#### 3. `observations.md`
- **Part 1:** Replace plot observations for corrected scripts (e.g., new heatmap dimensions, histogram bounds)
- **Part 2:** Update the expectations audit table rows for affected scripts — new metric values, updated verdicts
- **Divergence Summary:** Mark resolved items with strikethrough and `[RESOLVED via flag resolution]`
- **Warnings:** Mark resolved warnings similarly
- Do not touch Part 1 observations or Part 2 rows for unaffected scripts

#### 4. `narrative_brief.md`
Full rewrite as a ratified document:
- Status → `APPROVED`
- Delete the `## Flags for User` section entirely
- Fill in previously-stubbed sections (flags had no teaching angle entry) with full entries using corrected data
- Each corrected section **must include a "What was wrong and why it matters" subsection** — see [Corrections as Teaching Material](#corrections-as-teaching-material) below
- Update the Narrative Arc reconciliation if the corrections change the story
- Verify cross-section consistency with the now-corrected data

#### 5. `curriculum_state.md` entry
- If the Step 6 agent already appended a preliminary entry (COMPLETE brief that later got flagged during review): update it — remove caveats, add new concepts/vocabulary
- If the brief was INCOMPLETE at the Step 6 gate (the standard case): create the entry now, based on the ratified brief. Follow the format in `consolidation_spec.md`: Concepts Taught, Skills Acquired, Finance Vocabulary, Artifacts Built, Key Caveats, Reusable Downstream
- Include new concepts surfaced by the corrections (e.g., "P/E pathology", "two-stage outlier control")
- Update artifact descriptions if outputs changed (e.g., feature count)
- Update Reusable Downstream if output files changed

---

## Corrections as Teaching Material

Flags exist because something unexpected happened. Unexpected results — when diagnosed and corrected — are often more educational than results that work on the first try. The narrative brief should frame each correction as a teaching moment, not just a fix.

**Pattern for corrected sections:**

```markdown
### What was wrong and why it matters

**The problem:** [1-2 sentences: what the original implementation did wrong,
in domain terms — not "we had a bug" but "P/E ratio is undefined when
earnings are zero and sign-flips for loss-making firms"]

**Why it matters:** [1-2 sentences: what goes wrong in practice if you don't
catch this — "a quant model trained on P/E would learn that losing $1B and
earning $1B are equidistant from zero earnings, which is economically absurd"]

**The fix:** [1 sentence: what the corrected implementation does —
"earnings yield (E/P) is bounded, continuous, and handles losses gracefully"]

**Industry context:** [Optional. 1 sentence citing industry practice —
"Barra USE4 and Axioma use E/P, not P/E, for exactly this reason"]
```

This subsection goes inside the teaching angle, after the standard fields (Frame as, Key numbers, Student takeaway). It provides the "before vs. after" contrast that makes the correction pedagogically valuable.

**Do NOT include the problematic code itself** — the brief is a teaching plan, not a code review. The domain-level explanation ("why P/E is pathological") is what notebook agents need.

---

## Key Constraints

| Do | Don't |
|---|---|
| Fix only what's flagged | Refactor unflagged code |
| Rerun only affected scripts | Rerun the full `run_log.txt` |
| Update only affected artifact sections | Rewrite entire artifacts |
| Adjust assertions when fixes have mechanical side effects | Suppress assertions or widen them without justification |
| Frame corrections as teaching moments | Pretend the correction didn't happen |
| Present before/after metrics at the gate | Skip the gate and cascade directly |
| Cite domain reasons for fixes | Say "we had a bug" without explaining why |

---

## Outputs

1. Edited `.py` files (minimal targeted fixes)
2. Updated `run_log.txt` (corrected entries with `[CORRECTED]` markers)
3. Updated `execution_log.md` (corrected per-file sections with correction notes)
4. Updated `observations.md` (corrected observations, tables, resolved divergences)
5. Ratified `narrative_brief.md` (Status: APPROVED, no flags, corrections as teaching material)
6. Updated `curriculum_state.md` entry (caveats removed, new concepts added)

---

## Quality Bar

- [ ] Only flagged scripts were edited — no drive-by refactoring
- [ ] All edited scripts pass their assertions
- [ ] Before/after metric table presented and approved at the gate
- [ ] Assertion adjustments have explanatory comments
- [ ] `run_log.txt` entries for corrected scripts have `[CORRECTED]` markers
- [ ] `execution_log.md` has `**Correction (post-consolidation):**` notes for each fix
- [ ] `observations.md` Part 1 updated only for affected plots; Part 2 tables updated only for affected rows
- [ ] Divergence/warning items resolved with strikethrough notation
- [ ] `narrative_brief.md` status is APPROVED with no remaining flags
- [ ] Every corrected section has a "What was wrong and why it matters" subsection
- [ ] `curriculum_state.md` entry reflects actual post-correction state
- [ ] No unaffected artifacts were modified
