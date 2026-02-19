# Week Build Guide — The Paramount Agent Task Document

> **This is the single entry point for building a week's content.** It orchestrates the full workflow: research → README → notebooks. Structural and voice rules live in `notebook_creation_guide.md` and `writing_guidelines.md` respectively — this doc references them but never restates them.

---

## Overview

Building a week has **four phases**, each producing a concrete artifact:

| Phase | Output | Depends on |
|-------|--------|-----------|
| 1. Research | `research_notes.md` in the week folder | COURSE_OUTLINE.md + web sources |
| 2. README | `README.md` in the week folder | Research notes + COURSE_OUTLINE.md |
| 3. Code Verification | `code/` directory with verified `.py` files + `observation_report.md` | README + `code_verification_guide.md` |
| 4. Notebooks | `lecture.ipynb`, `seminar.ipynb`, `hw.ipynb` | README + writing guides + verified code + observation report |

**Each phase must complete before the next begins.** The README is the single source of truth for notebook content — notebooks never reference the outline or research notes directly. Verified code files are the single source of truth for notebook code — notebook agents never invent or modify code logic.

**Approval gates:** After completing each phase, present the output to the user and **wait for explicit approval** before starting the next phase. Do not proceed between phases without user sign-off.

**Delegation rule:** Writing the README (Phase 2), implementing code (Phase 3), and writing notebooks (Phase 4) must ALWAYS be delegated to agent calls (subagents). Never generate README, code, or notebook content in the main orchestrator context. If generation begins in the main context, abort immediately. The orchestrator's job is to manage phases, present results, and collect approvals — not to write content. Each agent receives only the documents specified in its phase and works with a clean, focused context.

---

## Phase 1: Research

### Purpose

Ensure the week's content reflects the state of the art as of the current date, not just the agent's training data. More critically, **verify implementation feasibility** before Phase 2 README writing begins. Phase 1 research answers the fundamental question: "Can this week's content be built without institutional barriers?"

### Complete Methodology

**See `phase1_research_guide.md` for the complete rigorous research protocol.** That guide defines:
- The verification principle (verify before you trust)
- Critical gap analysis checklist (data, tools, prerequisites, acceptance criteria)
- Follow-up verification protocol
- Research tier assignments with query budgets
- readiness assessment criteria
- Examples and anti-patterns from real week builds

### Quick Reference: Research Tiers

| Tier | Weeks | Query Budget |
|------|-------|--------------|
| **HEAVY** | W7, W8, W17, W18 | 6-10 initial + unlimited follow-ups |
| **MEDIUM** | W4, W10, W14, W15, W16 | 3-5 initial + follow-ups as needed |
| **LIGHT** | W1-3, W5-6, W9, W11-13 | 1-3 initial + follow-ups for verification |

**Initial queries** = discovery (paradigm shifts, new tools, recent papers)
**Follow-up queries** = verification (test libraries, fetch docs, resolve ambiguities)

### The Core Research Questions

Every Phase 1 must answer:

1. **Data availability:** Can students access required data WITHOUT institutional subscriptions (WRDS, Bloomberg)?
2. **Tool accessibility:** Are Python libraries FREE, maintained, and functional?
3. **Prerequisites verified:** What do students already know from prior weeks?
4. **Conceptual clarity:** Are there ambiguous distinctions that must be clarified (e.g., Barra vs Fama-French)?
5. **Acceptance criteria:** What quantitative targets should Phase 3 code hit?

**If ANY answer is uncertain:** Phase 1 is incomplete. Continue research.

### Critical Verification Rule

**Every finding must be either verified or dropped.** Search result snippets are often vague or misleading.

**Verified =** At least ONE of:
- Primary source fetched (actual page read: docs, syllabus, paper)
- 3+ independent sources corroborate
- Executable verification (tested: installed library, fetched data, ran code)

**Dropped =** Cannot verify specifics (paywall, dead link, vague snippet with no source)

**Query budget:** Initial queries (LIGHT 1-3, MEDIUM 3-5, HEAVY 6-10) are for discovery. Follow-up verification queries are UNLIMITED — do as many as needed to confirm findings.

### Research Output

Save findings as `research_notes.md` inside the week's folder. See `phase1_research_guide.md` for the complete required structure. Essential sections:

- Latest papers & results (with URLs)
- Tools & libraries (with **access requirements explicitly labeled**: ⚠️ REQUIRES [X] or ✅ FREE)
- **Data sources for implementation** (CRITICAL — verify FREE access)
- Technical implementation (verified library availability)
- Conceptual clarifications (resolved ambiguities)
- **Prerequisites verification** (read prior week READMEs)
- **Acceptance criteria** (quantitative targets for Phase 3)
- Phase 2 readiness assessment (confidence level, checklist)

### Phase 1 Completion Gate

✅ **Phase 1 is COMPLETE when:**
- All critical gaps verified (data, tools, prerequisites, acceptance criteria)
- research_notes.md written with all required sections
- Confidence level ≥ 8/10 that Phase 2 README can be written
- No known blockers for Phase 3 code implementation

❌ **Phase 1 is INCOMPLETE when:**
- Data access uncertain ("probably free but didn't verify")
- Library requirements unknown ("seems to work but didn't check access")
- Conceptual ambiguities unresolved
- Acceptance criteria vague ("code should work reasonably well")

**The Gate Question:** "If I delegated Phase 2 README writing RIGHT NOW with only COURSE_OUTLINE.md + research_notes.md + prior READMEs, would the agent have EVERYTHING needed to specify what data to use, what libraries to install, what prerequisites to assume, and what quantitative targets to hit?"

**If Yes:** Proceed to Phase 2.
**If No:** Continue research until all uncertainties resolved.

---

## Phase 2: README

### Purpose

The README is the **spec** for all three notebooks. It defines what to teach, in what order, with what emphasis. Notebook-writing agents read ONLY the README (plus the two writing guides) — they never read the outline or research notes.

### Source Documents

The README is synthesized from:
1. **COURSE_OUTLINE.md** — the week's entry (topics, depths, career paths, prerequisites, implementability grade, multi-topic notes)
2. **research_notes.md** — latest findings from Phase 1
3. **Prior week READMEs** — only to check prerequisite handoffs (what was already taught that this week can assume)

### README Structure

Every README must follow this structure:

```markdown
# Week N: [Full Title]

**Band:** [1/2/3/4]
**Implementability:** [HIGH / MEDIUM-HIGH / MEDIUM]

## Learning Objectives

After this week, students will be able to:
1. [Concrete, assessable objective — "build X", "evaluate Y", "explain why Z"]
2. ...
3. [5-8 objectives total]

## Prerequisites

- **Required:** [Week numbers and what specifically is needed from them]
- **Assumed ML knowledge:** [Any ML concepts used without explanation]
- **Prerequisite explanations in this week:** [Any finance/math concepts that get a brief intro within this week]

## Narrative Arc

[2-3 paragraphs describing the week's story. What's the central question or tension? How does the week resolve it? What's the "aha moment"? This is the creative vision that guides all three notebooks.]

## Lecture Outline

### Section 1: [Title]
- **Hook:** [The opening story, provocation, or disaster]
- **Core concepts:** [What gets taught, at what depth]
- **Key demonstration:** [What code shows, what the student should notice]
- **Bridge:** [How this connects to Section 2]
- **Acceptance criteria:** [Verifiable conditions the code must satisfy — see `code_verification_guide.md`]

### Section 2: [Title]
...

[Continue for all sections. Typically 4-7 sections per lecture.]

### Closing
- Summary table of key formulas/concepts
- Career connections: how this week's skills map to specific roles and daily work (see Career Connections below)
- Bridge to next week
- Suggested reading with annotations

## Seminar Outline

### Exercise 1: [Title — framed as a question]
- **Task type:** [One of: Guided Discovery, Skill Building, Construction, Investigation — from `task_design_guide.md`]
- **The question:** [What are we investigating?]
- **Data needed:** [What data, from where]
- **Tasks:** [Numbered, specific]
- **Expected insight:** [What the student should discover — NOT what the lecture already said]
- **Acceptance criteria:** [Verifiable conditions — see `code_verification_guide.md`]

### Exercise 2: [Title]
...

[3-5 exercises total. Each must be NEW — never repeat a lecture demonstration.]

## Homework Outline

### Mission Framing
[2-3 paragraphs: what are you building, why it matters, what makes it different from the seminar]

### Deliverables
1. **[Deliverable title]**
   - **Task type:** [One of: Skill Building, Construction, Investigation — from `task_design_guide.md`. For Investigation tasks, note which layers (Baseline / Smarter / Frontier) the solution should cover.]
   - [Clear success criteria]
   - **Acceptance criteria:** [Verifiable conditions — see `code_verification_guide.md`]
2. ...
3. [3-5 deliverables total]

### Expected Discoveries
[Bullet list of insights students should encounter — these become the "aha moments" in the solution]

## Key Papers & References
- [Author (Year)] — [Title]. [1-line "why you'd read this" annotation]
- ...

## Career Connections
[What roles use this week's skills daily? Map 2-4 specific roles (from the COURSE_OUTLINE.md career paths) to specific skills taught this week. Be concrete: "A risk analyst at a multi-strategy fund runs exactly this analysis every morning" — not "this is useful in finance." The lecture closing will turn these into a 2-3 paragraph narrative; seminar/homework prose should weave career relevance into "so what?" moments throughout.]

## Data Sources
- [What datasets are needed, where to get them, any API keys required]
- [Approximate data sizes and download times]
```

### README Quality Bar

- **The narrative arc is the most important section.** If you can't articulate the week's story in 2-3 paragraphs, the notebooks will be disjointed.
- **Every exercise and deliverable must be concrete, unambiguous, and implementable.** "Explore the data" is not an exercise. "Compare GARCH(1,1) vs. LSTM volatility forecasts on 5 stocks using QLIKE loss and determine when the complexity is worth it" is. Consult `task_design_guide.md` for task types, the Implementability Principle, and the Layered Task Pattern for open-ended tasks.
- **No overlap across notebook outlines.** If the lecture outline demonstrates X, the seminar must extend or challenge X, never redo it. If the seminar covers Y, the homework must integrate Y at scale, not repeat it. (See the Non-Overlap Principle in `notebook_creation_guide.md`.)
- **Every exercise and deliverable must specify its task type** from `task_design_guide.md` (Guided Discovery, Skill Building, Construction, or Investigation). This forces the README author to think about *what kind of learning* each task produces and ensures the solution strategy matches the type.
- **Career connections must be concrete and role-specific.** "This is useful in finance" is not a career connection. "A quant researcher at a stat arb fund runs exactly this cointegration test every Monday morning on ~200 pairs" is.
- **Cross-week references: keep artifact names, use conceptual verbs.** When the README references prior-week outputs (e.g., Week 3's `FactorBuilder`, Week 2's `compute_monthly_returns`), keep the artifact name — it tells Phase 3 agents exactly which file to read as reference. But phrase the reference as a conceptual connection, not a code instruction. Write "Build the same feature matrix that Week 3's `FactorBuilder` produced" rather than "Load/import/reuse Week 3's `FactorBuilder` output." Each week's code is self-contained (see `code_verification_guide.md`, Cross-Week Dependencies); the README must not promise otherwise. The artifact name is a pointer; the verb tells the notebook agent what the student actually does (rebuild, not import).
- **Incorporate Phase 1 research findings — but only verified ones.** If the research found a newer approach, a better library, or a recent paper that changes the teaching, it should be reflected in the README. However, only incorporate findings that have specific, verified details (concrete numbers, dates, dataset names, confirmed tool behavior). Vague or unverified findings from search snippets should be omitted — it's better to teach something well-established correctly than to reference something recent but poorly understood.

---

## Phase 3: Code Verification

### Purpose

Implement and verify all code that will appear in the week's notebooks BEFORE any notebook is written. This separates engineering (making code work) from pedagogy (making prose teach). See `code_verification_guide.md` for the complete specification.

### What the Agent Reads

The code verification agent receives exactly three documents:
1. `course/guides/code_verification_guide.md` — file format, annotation syntax, verification loop
2. `course/guides/task_design_guide.md` — task types and solution strategies
3. The week's `README.md` — content spec with acceptance criteria

### What It Produces

The complete `code/` directory inside the week folder:
- `data_setup.py` — shared data downloads and caching
- `lecture/s1_*.py`, `s2_*.py`, ... — one file per lecture section
- `seminar/ex1_*.py`, `ex2_*.py`, ... — one file per seminar exercise
- `hw/d1_*.py`, `d2_*.py`, ... — one file per homework deliverable
- `observation_report.md` — the visual plot review report (from Stage 2)
- `.cache/` — generated plot PNGs and cached data (gitignored)

Every code file has three layers: docstring (acceptance criteria), annotated code (cell boundaries with `Purpose`/`Takeaway` metadata), and assertions + plot summaries (`if __name__ == "__main__":` block). Every file runs without error and passes all assertions.

### Agent Strategy

**One agent** handles all code for all three notebooks. This is NOT parallelized — the agent needs to:
- Manage the shared data layer (`data_setup.py`)
- Avoid redundant computation across notebooks
- Verify non-overlap in what each notebook's code computes

**Before writing `data_setup.py`**, the agent proposes a data plan to the user (universe size, date range, frequency) with options and trade-offs. The user approves the data plan before code writing begins. See `code_verification_guide.md` for the data plan format.

### The 3-Stage Verification Process

Phase 3 runs in **three stages** that may repeat. See `code_verification_guide.md` for the full specification.

**Stage 1 — Code + Assertions + Summaries (automated):**
The agent writes all code files, runs them, iterates until all assertions pass. Enriches annotations with actual numbers. Collects plot summary output. Produces a Visual Annotations Summary listing every plot with its filename and `Visual` annotation (for Stage 2). Presents results to user.

**Stage 2 — Visual Plot Review (user, fresh context):**
The user opens a **new Claude conversation** and reviews the generated plot PNGs alongside the Visual Annotations Summary. Short context = accurate vision. Produces an `observation_report.md` with verification status, detailed observations, discoveries, and any issues requiring code fixes. The user reviews and amends the report.

**Stage 3 — README Compliance Audit (automated):**
The agent compares the printed summary output (numeric criteria) and the observation report (visual criteria) against every README acceptance criterion. Produces contradiction entries for unmet criteria with resolution options. The user resolves contradictions. If code fixes are needed, loop back to Stage 1. After contradictions are resolved, produces **Implementation Reality notes** documenting significant gaps between ideal (README/literature) and actual (code output) — with teaching angles for each. These notes are appended to the observation report and become key narration material for Phase 4.

The loop repeats until Stage 3 produces no unresolved contradictions.

### Verification Gate

Phase 3 is complete when all three stages pass cleanly:
- All assertions pass (Stage 1)
- All plots verified and observation report approved (Stage 2)
- All README criteria met or contradictions resolved (Stage 3)

Do NOT proceed to Phase 4 without user approval of the final state.

---

## Phase 4: Notebooks

### What the Agent Reads

Notebook-writing agents receive exactly seven documents:
1. `course/guides/notebook_creation_guide.md` — structural rules, prose ratios, anti-patterns
2. `course/guides/writing_guidelines.md` — voice, tone, 10 rules, examples
3. `course/guides/task_design_guide.md` — task types, difficulty calibration, solution strategies
4. The week's `README.md` — the content spec
5. `code/data_setup.py` — the shared data layer
6. **The relevant `code/` subfolder** (e.g., `code/lecture/` for the lecture agent)
7. **`code/observation_report.md`** — visual observations (Stage 2) + Implementation Reality notes (Stage 3)

**Nothing else.** Not the outline, not other weeks' notebooks, not the research notes.

### How Notebook Agents Use Verified Code

The notebook agent's job is **prose and layout**, not code invention:

1. **Read `data_setup.py`** to understand what data to download, then write equivalent direct download calls (e.g., `yf.download()`) in the notebook. **Notebooks must NOT import from `data_setup.py`** — that file is for `.py` code files only. Notebooks are standalone documents. **No caching logic** — no cache directories, no parquet caches, no file-existence checks. Just call the API and keep data in memory.
2. **Read each code file** in order (s1, s2, ... or ex1, ex2, ... or d1, d2, ...).
3. **For each `# ── CELL:` block**, create:
   - A markdown cell BEFORE the code (using `Purpose` annotation to guide the prose)
   - A code cell containing the code VERBATIM
   - A markdown cell AFTER the code (using `Takeaway` or `Visual` to guide interpretation)
4. **Read the observation report** for plot interpretation. Use its detailed observations, specific numbers, and discoveries to write richer prose — especially for plot interpretation cells and "so what?" bridges. Code annotations provide cell-level structure; the report provides narrative depth.
5. **Write prose** following `writing_guidelines.md` voice and `notebook_creation_guide.md` structure.
6. **Add section headers, transitions, opening, closing** per notebook structure rules.
7. **Never modify code logic.** The agent may consolidate imports into the setup cell and adjust minor formatting, but must NOT change what the code computes.
8. **Never include** the `if __name__` block (assertions and plot summaries), `# ── CELL:` markers, or file docstrings in the notebook. Never reference the observation report as a document — its content is woven into prose naturally.

### Production Order & Agent Strategy

Each notebook is written by a **separate agent call** — one agent per notebook, launched **in parallel**:

- **Lecture agent** → produces `lecture.ipynb` (reads `code/lecture/`)
- **Seminar agent** → produces `seminar.ipynb` (reads `code/seminar/`)
- **Homework agent** → produces `hw.ipynb` (reads `code/hw/`)

All three agents launch simultaneously. The orchestrator reviews all three outputs together.

**Why separate agents?** Each agent gets a clean, focused context window. A single agent writing all three notebooks risks context exhaustion — quality degrades by the third notebook.

**Why parallel?** Each agent reads only the README + guide docs + its own code subfolder — never the other notebooks. The Non-Overlap Principle is enforced at the README level (each notebook's outline specifies distinct content), so agents don't need to cross-reference each other's output.

### Quality Verification

Before marking a notebook complete, verify against the **Quality Checklist** at the end of `notebook_creation_guide.md`. That checklist is the single source of truth for structural, narrative density, voice, and code quality standards — do not duplicate it here.

---

## Quick Reference: The Full Pipeline

```
┌─────────────────────────────────────────────────────┐
│  COURSE_OUTLINE.md  (what this week covers)         │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  Phase 1: RESEARCH                                  │
│  Web search (tiered: HEAVY / MEDIUM / LIGHT)        │
│  → research_notes.md                                │
└──────────────┬──────────────────────────────────────┘
               │  ⏸ approval gate
               ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2: README  (delegated to agent)              │
│  Outline + research → README.md                     │
│  (the spec for all notebooks)                       │
└──────────────┬──────────────────────────────────────┘
               │  ⏸ approval gate
               ▼
┌─────────────────────────────────────────────────────┐
│  Phase 3: CODE VERIFICATION  (3-stage loop)         │
│  See code_verification_guide.md                     │
│                                                     │
│  Stage 1: Code + Assertions + Summaries (agent)     │
│    → all .py files pass, annotations enriched       │
│                ⏸ present to user                    │
│  Stage 2: Visual Plot Review (user, fresh context)  │
│    → observation_report.md produced & approved       │
│                ⏸ user approves report               │
│  Stage 3: README Compliance Audit (agent)           │
│    → contradictions resolved                        │
│    → if code fixes needed: back to Stage 1          │
└──────────────┬──────────────────────────────────────┘
               │  ⏸ all 3 stages pass
               ▼
┌─────────────────────────────────────────────────────┐
│  Phase 4: NOTEBOOKS  (one agent per notebook)       │
│  Each reads: guides + README + code + obs. report   │
│                                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │ Agent 1      │ │ Agent 2      │ │ Agent 3     │ │
│  │ lecture.ipynb │ │ seminar.ipynb│ │ hw.ipynb    │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬──────┘ │
│         └────────────────┼────────────────┘         │
│                          ▼                          │
│                    ⏸ review all                     │
│                                                     │
│  Each agent reads: guides + README + verified code  │
│  Quality: verify against notebook_creation_guide.md │
└─────────────────────────────────────────────────────┘
```

---

## Appendix: Research Tier Assignment Table

| Week | Topic | Tier | Why |
|------|-------|------|-----|
| W1 | Markets & Financial Data | LIGHT | Foundational, stable |
| W2 | Financial Time Series & Volatility | LIGHT | Well-established (GARCH, stationarity) |
| W3 | Factor Models & Cross-Sectional Analysis | LIGHT | Decades of established literature |
| W4 | ML for Alpha — Features to Signals | MEDIUM | Core stable but new techniques emerge |
| W5 | Backtesting & Transaction Costs | LIGHT | Methodology well-established |
| W6 | Portfolio Construction & Risk Management | LIGHT | Markowitz, risk parity, Black-Litterman are stable |
| W7 | NLP & LLMs for Financial Alpha | HEAVY | LLM landscape evolves monthly; FinBERT alternatives, open models, LLM-as-alpha |
| W8 | DL for Financial Time Series | HEAVY | Foundation models (Chronos, TimesFM, Moirai) are brand new and evolving |
| W9 | Derivatives Pricing & the Greeks | LIGHT | Black-Scholes, Greeks are decades-old fundamentals |
| W10 | Causal Inference & Modern Factor Investing | MEDIUM | Methods stable but finance applications are cutting-edge |
| W11 | Bayesian Methods & Uncertainty | LIGHT | Methods well-established; applications evolving slowly |
| W12 | Fixed Income & Interest Rates | LIGHT | Yield curves, duration, convexity are stable |
| W13 | Market Microstructure & Optimal Execution | LIGHT | Fundamentals stable (LOB, Almgren-Chriss) |
| W14 | Statistical Arbitrage & Regime Detection | MEDIUM | Transformer-based regime detection, online methods are new |
| W15 | Reinforcement Learning for Finance | MEDIUM | Rapidly evolving; latest reward shaping, sim-to-real |
| W16 | Deep Hedging & Neural Derivatives Pricing | MEDIUM | Active research area (Buehler et al. line of work) |
| W17 | Generative Models & GNNs for Finance | HEAVY | Diffusion models for finance, latest GNN architectures for graphs |
| W18 | DeFi, Emerging Markets & Frontier | HEAVY | Evolves weekly — new protocols, exploits, ML applications |
