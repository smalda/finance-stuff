# Blueprint Specification — Step 2 Guide

> **Consumer:** Blueprint agent (Step 2). The blueprint is a pure creative and structural document — the ideal teaching week without numerical predictions or acceptance criteria. It knows which data SOURCES to use but not what numbers will emerge. Once approved, it is immutable.

---

## Inputs

In addition to `common.md` and `task_design.md`:
- **Course outline entry** for this week
- **`research_notes.md`** from Step 1
- **`curriculum_state.md`** — what prior weeks have taught (`course/curriculum_state.md`)

## Outputs

1. **`blueprint.md`** in the week folder (immutable once approved)

---

## blueprint.md — Format

Target: ~150 lines. Dense but scannable.

### Header

`# Week N: [Title]` followed by **Band** and **Implementability** (from course outline).

### Learning Objectives

5-8 outcomes, each starting with an action verb. These define what a student can DO after all three notebooks.

### Prerequisites

- **Required prior weeks:** reference by number
- **Assumed from curriculum_state:** specific skills/concepts from prior entries
- **Assumed ML knowledge:** specific ML skills the audience already has
- **Prerequisite explanations in this week:** finance/math concepts needing brief intro (from course outline). Note WHICH lecture section introduces each.

### Opening Hook

1-3 sentences in the course's **full narrative voice**. A story, provocation, or disaster — the first thing the student reads. Not the narrative arc; the attention-grabbing moment.

### Narrative Arc

2-3 paragraphs: (1) **The setup** — what the ML expert thinks they know. (2) **The complication** — what makes this harder or stranger than expected. (3) **The resolution** — how understanding transforms by week's end. Every section should advance this arc; every exercise should test it.

### Lecture Outline

One entry per section. **Each section maps 1:1 to a code file** (`sN_topic_name.py`). Aim for 5-8 sections; multi-topic weeks may exceed this with justification.

**Per section:**

```
### Section N: [Evocative Title] → `sN_snake_case_name.py`

- **Hook:** [1-2 sentence provocation — full narrative voice]
- **Core concepts:** [terms, frameworks, mechanics this section teaches]
- **Key demonstration:** [concept + visualization type — directional, not prescriptive.
  Describe WHAT to demonstrate, not exact data or expected numbers]
- **Key formulas:** [name 1-2 formulas, e.g., "CAPM: E[Rᵢ] = Rf + βᵢ(E[Rm] - Rf)"]
- **"So what?":** [why this matters — to models, money, or careers. 1-2 sentences]
- **Bridge:** [connection to next section. 1 sentence]
```

**Conceptual sections** (minimal code): note "Conceptual — diagram/discussion" in Key demonstration.

**Sidebar topics** (MENTIONED / MENTIONED ABSTRACTLY): add inline within the relevant section as `- **Sidebar:** [Topic] ([depth]) — [1-sentence scope]`.

### Narrative Hooks

3-4 standout facts from the research, with suggested section placement. Full narrative voice — the kind of thing a student retells at the bar. The notebook agent may add more from actual results.

### Seminar Outline

3-5 exercises, each going BEYOND the lecture. No data/setup field — data allocation is Step 3's job.

**Per exercise:**

```
### Exercise N: [Title as a question or challenge]

- **Task type:** [from task_design.md]
- **The question:** [specific question the student answers]
- **Expected insight:** [what the student discovers — for the agent, NEVER shown to students]
```

### Homework Outline

3-4 deliverables integrating multiple concepts at scale — combining lecture concepts, not enlarging seminar exercises.

**Per deliverable:**

```
### Deliverable N: [Title as a mission]

- **Task type:** [from task_design.md]
- **Mission framing:** [2-3 sentences: why it matters, what they build, what they'll discover]
- **Scope:** [classes, analyses, reports the deliverable includes]
- **Requires:** [prior deliverables if dependent. Omit if independent]
```

### Concept Matrix

Every blueprint includes a table mapping each major concept to exactly ONE notebook's exercises (per the Non-Overlap Principle in `common.md`). If a concept appears in two notebooks' exercises, cut one.

```
| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| [X]     | Demo: … | Ex: …   | —        |
```

### Career Connections

2-4 career paths: `**[Job title] at [firm type]:** [concrete daily workflow]`. Specific enough to picture the job — real firm types, real workflows, real tools.

### Data Sources

Week-level list of APIs/libraries. **Name only** — universe, date range, frequency are Step 3. Format: `- [API] — [data type]`.

### Key Papers & References

**Core** (3-5): author, year, title, one-line "why you'd read this." **Advanced** (2-3): for deeper dives.

### Bridge to Next Week

1-2 sentences connecting this week to the next in the course arc.

---

## curriculum_state.md

The blueprint agent READS `curriculum_state.md` to know what prior weeks taught. It does NOT write to it. The entry for this week is written after Step 6 (Consolidation) approval, when we know what was actually taught — not what we planned to teach. See `consolidation_spec.md`.

---

## Boundaries

The blueprint does NOT contain:

| Excluded | Why |
|----------|-----|
| Acceptance criteria | Step 3 sets these |
| Numerical predictions | Step 3 assesses data reality |
| Data plan (universe, dates, frequency) | Step 3 designs the data plan |
| Per-exercise data allocation | Step 3 maps data to exercises |
| Data-dependent branching | Step 6 reconciles ideal vs. actual |
| Teaching angles for divergences | Step 6 decides how to frame gaps |

**The blueprint is optimistic by design.** Steps 3-6 handle reality.

---

## Deviations from the Course Outline

If `research_notes.md` reveals a topic needs different treatment (deprecated library, paradigm shift, barrier): **flag** the deviation with rationale, **propose** an alternative preserving the learning objective, and let the **user decide** at the approval gate. The outline is authoritative on scope and depth; the blueprint adjusts HOW, not WHETHER.

---

## Quality Bar

- [ ] Learning objectives are concrete and assessable (action verb + outcome)
- [ ] Prerequisites reference `curriculum_state.md`, not prior blueprints
- [ ] Narrative arc has tension and resolution — not a topic list
- [ ] Opening hook would make an ML expert stop scrolling
- [ ] Every lecture section hook is in full narrative voice
- [ ] Sections map 1:1 to code files with snake_case names
- [ ] Key formulas named per section where applicable
- [ ] Concept matrix filled in — no concept exercised in two notebooks
- [ ] Seminar exercises go BEYOND lecture demos
- [ ] Homework deliverables integrate multiple concepts; dependencies noted
- [ ] Career connections name specific roles, firm types, workflows
- [ ] Data sources are APIs/libraries only — no parameters or expectations
- [ ] No acceptance criteria or numerical predictions anywhere
- [ ] 3-4 narrative hooks with suggested placement
- [ ] Bridge to next week included
- [ ] ~150 lines total (dense weeks may run slightly longer)
