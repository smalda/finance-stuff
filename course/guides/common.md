# Common Reference — Pipeline & Principles

> **System-level document.** Every agent reads this alongside their step-specific guide(s). It does not count toward the 2-guide limit. It provides shared context about where you sit in the pipeline and what principles span all steps.

---

## The Pipeline at a Glance

```
Step 1:  RESEARCH         → research_notes.md       (domain knowledge)
Step 2:  BLUEPRINT        → blueprint.md            (ideal teaching plan)
Step 3:  EXPECTATIONS     → expectations.md         (data reality + acceptance criteria)
Step 4:  CODE             → code/ + run_log.txt + execution_log.md (working implementation)
Step 5:  OBSERVATION REVIEW → observations.md       (visual + numerical review)
Step 6:  CONSOLIDATION    → narrative_brief.md      (reconciled teaching plan)
Step 6½: FLAG RESOLUTION  → patched code + artifacts (only if Step 6 flags exist)
Step 6¾: BRIEF AUDIT      → brief_audit.md          (adversarial honesty review)
Step 7:  NOTEBOOKS        → *.ipynb                 (the actual course material)
```

Each of the 7 main steps produces one immutable artifact. Nothing flows backward — if a later step discovers a problem, it either documents it as a teaching angle or flags it. Flags are resolved via Step 6½ (targeted patches to existing artifacts) or a full rerun loop (Steps 4→5→6), depending on scope. Step 6¾ audits the brief's intellectual honesty before notebooks consume it. Agents never edit upstream artifacts; Step 6½ is an orchestrator-executed exception scoped only to flagged sections.

**Scope boundary: you must NEVER create, modify, or delete ANY file outside the current week's folder (`course/weekNN_TOPIC/`).** Other weeks' folders, guide files, `COURSE_OUTLINE.md`, `curriculum_state.md`, and everything else in the repo are strictly off-limits. The only exception is the orchestrator updating `curriculum_state.md` after Step 6¾ — individual step agents never touch it. If you believe something outside the week folder needs changing, document it as a note in your output artifact; do not make the change yourself.

---

## Information Flow: Who Knows What

Three levels of information exist in the pipeline. Understanding your level prevents you from overstepping:

| Level | Artifacts | What they contain |
|-------|-----------|-------------------|
| **Vision** | blueprint.md | The ideal — what to teach, in what order, with what narrative arc. Knows data SOURCES but not what numbers will emerge. |
| **Reality** | expectations.md → run_log.txt + execution_log.md → observations.md | What we predicted, what actually happened, what the plots and numbers show. The empirical ground truth. |
| **Teaching** | narrative_brief.md → brief_audit.md → notebooks | The reconciled story — how to teach what actually happened, given what we ideally wanted. The brief audit stress-tests the brief's honesty before notebooks consume it. |

**The cardinal rule:** Vision artifacts never predict specific numbers. Reality artifacts never redesign the curriculum. Teaching artifacts never modify code. Each level feeds forward, never backward.

---

## The Notebook Progression

The three notebooks serve distinct roles in a strict progression:

```
LECTURE:  UNDERSTAND — "Here's WHY this matters"        (demonstrations only)
SEMINAR:  APPLY     — "HOW does this work in practice?" (guided exercises)
HOMEWORK: BUILD     — "Can you BUILD a system?"         (integrative projects)
```

### The Non-Overlap Principle

A concept gets exercised in **ONE** notebook. The lecture SHOWS it; the seminar has students MEASURE it on new data; the homework INTEGRATES it at scale. No notebook re-does what another already covered.

When designing tasks (Step 2) or writing notebooks (Step 7), draw up a concept matrix:

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| Fat tails | Demo: SPY histogram | Compare: 10 stocks by sector | Integrate into pipeline at 200-stock scale |
| Dollar bars | Demo: 5-line sketch | Build from scratch: SPY + TSLA | Method inside DataLoader class |

If a concept appears in two notebooks' exercises, one must be cut.

---

## The Sandbox-Reality Principle

This course runs on free data (yfinance, Ken French, FRED) — not institutional data (CRSP, Bloomberg, OptionMetrics). Many analyses will produce different results than a fund would see. **Students must always know what production looks like.** When our sandbox result differs from institutional reality, we don't hide the gap — we teach it: "Here's what we got, here's what a fund sees, here's why." These moments often carry more teaching value than the sections that work as expected.

This principle flows through the pipeline mechanically: Step 3 sets production benchmarks, Step 5 compares against them, Step 6 categorizes divergences, Step 6¾ audits whether the brief is honest about limitations, and Step 7 weaves the framing into prose. But the principle matters beyond the mechanics — even in Steps 1 and 2, keep in mind that the course's credibility depends on honest contextualization of our data limitations.

---

## Data Sufficiency Over Speed

Download time is not a cost that matters in this pipeline. Statistical power and pedagogical quality are. A 20-minute download that produces meaningful results is always preferable to a 30-second download that produces noisy, undersized output where the student can't tell whether the concept failed or the data was just too thin. When choosing universe sizes, date ranges, and frequencies, optimize for the strongest possible results — not the fastest possible download. The only legitimate constraints on data scope are API reliability (downloads that break), data availability (the data doesn't exist), and computational cost of model training (not downloading).

---

## Shared Data Layer

Common datasets (S&P 500 prices, Fama-French factors, fundamentals, FRED series) are downloaded once and cached at `course/shared/.data_cache/`. Each week's `data_setup.py` imports from `course/shared/data.py` to read from this cache instead of re-downloading.

**What's shared:** Raw, untransformed data that appears across multiple weeks. The canonical S&P 500 universe (200 tickers), factor returns, fundamental data, and FRED economic series.

**What stays per-week:** Derived data — monthly returns, feature matrices, merged panels, computed ratios. These go in each week's `.cache/` as before.

**Notebooks are independent.** The shared layer is a pipeline optimization. Step 7 notebook agents write inline download code — notebooks never depend on the shared cache.

See `code_verification.md` Rule 1 and `shared/data.py` for the full API.

---

## The Audience

ML/DL experts learning finance. They don't need ML fundamentals taught. The finance is new — and the intersections where standard ML intuition breaks in financial contexts are where the deepest learning happens.
