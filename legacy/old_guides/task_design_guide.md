# Task Design Guide

> **The core principle:** Every task must have a complete, working solution that the writing agent can confidently produce. If you can't solve it, don't assign it. The solution's scope defines the task's scope — never the reverse.

This guide governs the design of exercises (seminar), deliverables (homework), and demonstrations (lecture). Voice, prose structure, and formatting rules live in `writing_guidelines.md` and `notebook_creation_guide.md` respectively — this doc covers **what to assign and why**, not how to write it up.

---

## The Implementability Principle

**Never assign a task you cannot fully implement in the solution.**

This is the single most important rule in this document. It sounds obvious. It isn't. The failure mode is: the README describes an ambitious, interesting-sounding task; the agent writing the notebook can't actually produce a clean, complete solution; the result is hand-waving, placeholder code, or a "solution" that doesn't run.

What this means in practice:

- Before putting a task in the README, mentally write the solution. If you can see the code, the expected output, and the interpretation — the task is implementable. If you're thinking "the agent will figure it out" — it won't.
- If the most interesting version of a task is too hard to solve completely, **shrink the scope** until the solution is airtight, then add the ambitious version as a clearly marked extension (see "The Layered Task Pattern" below).
- The agent writing the notebook inherits this rule: if the README specifies a task the agent can't confidently solve, the agent should flag it rather than produce a shaky solution.

---

## Task Types

Not every task serves the same purpose. Using the wrong type for the moment is a common failure — a "build this system" task when the student needs to "discover this insight" produces engineering practice when it should produce understanding, and vice versa.

### 1. Demonstration (lecture only)

**Purpose:** Show a concept working. Build intuition before formalism.

The instructor runs the code; the student watches and reads. The "task" is the code + the interpretation around it. No student work.

**When to use:** Introducing new concepts. Showing what goes wrong (the catastrophe before the fix). Motivating why something matters.

**Solution strategy:** The solution IS the demonstration. Full code + prose interpretation. Every demonstration must answer "so what?" — what does this result mean in dollars, in model quality, in career relevance?

**Example:** *Plot AAPL's unadjusted vs. adjusted price across the 2014 split. Show the 85% "crash" in unadjusted data. Explain why a model trained on this would learn catastrophically wrong patterns.*

### 2. Guided Discovery (seminar, occasionally homework)

**Purpose:** The student follows a constrained path and discovers a specific insight. The insight is the payoff, not the code.

The question is specified. The data is specified. The steps are specified. But the answer is discovered through execution, not told in advance.

**When to use:** When the "aha moment" matters more than the implementation. When the domain reveals something that ML intuition wouldn't predict. When you want students to SEE a result before being told what it means.

**Solution strategy:** The solution shows the exact path AND the exact insight. Numbers are concrete ("the bias is 3.2% per year, compounding to 54% over 14 years"). The insight must pass the Aha Moment Quality Bar from `notebook_creation_guide.md`.

**Difficulty source:** The domain, not the code. The code should be straightforward; the surprise comes from what the data reveals.

**Example:** *Compute equal-weight portfolio returns for current S&P 500 constituents back to 2010. Compare to returns using only tickers with complete histories. The student discovers that survivorship bias inflates returns by 1-4% per year — a result they can compute in 10 lines of code but wouldn't have guessed.*

### 3. Skill Building (seminar and homework)

**Purpose:** Build fluency with tools, procedures, or domain-specific workflows. The answer is well-defined; the challenge is doing it correctly and efficiently.

**When to use:** Early weeks. New tools or libraries. Procedures the student will repeat throughout the course (computing returns, handling corporate actions, storing data). When "get comfortable" is genuinely the goal.

**Solution strategy:** The solution shows the idiomatic, production-quality approach. Brief interpretation — the educational value is in the doing, not in a surprise result.

**Difficulty source:** Getting the details right. Financial data has edge cases (missing days, timezone alignment, ticker format variations) that reward careful implementation. The difficulty should come from the domain's messiness, not from algorithmic complexity.

**Example:** *Download 10 years of daily data for 20 stocks across market caps. Compute daily log returns. Detect and handle missing trading days. Store in Parquet with Polars. The student builds muscle memory for the data pipeline they'll use in every subsequent week.*

### 4. Construction (homework, occasionally seminar)

**Purpose:** Build a system or tool. The spec is clear; the engineering and design decisions are the student's.

**When to use:** When the deliverable is a reusable artifact (a DataLoader class, a backtesting pipeline, a risk model). When the educational value comes from the design process — choosing data structures, handling edge cases, making tradeoff decisions.

**Solution strategy:** The solution shows ONE complete implementation. Design choices are explained in prose ("we chose long format over wide format because..."). The solution acknowledges meaningful alternatives but doesn't implement them all. Code is production-quality: docstrings, clean structure, thoughtful variable names.

**Difficulty source:** Integration and scale. Building for 5 stocks is easy; building for 50 reveals edge cases. The challenge is robustness, not cleverness.

**Example:** *Build a `FinancialDataLoader` class that downloads, cleans, validates, and stores multi-asset data. It should handle at least 50 tickers, detect quality issues, and produce a per-ticker quality scorecard.*

### 5. Investigation (homework, advanced seminar exercises)

**Purpose:** Explore an open-ended question where multiple approaches are valid. This is the closest to real research — the student must formulate an approach, not just execute one.

**When to use:** Later weeks. When the domain question is genuinely interesting and the "right" answer depends on context or judgment. When you want students to experience the ambiguity of real quantitative research.

**Solution strategy:** Use **The Layered Task Pattern** (see below). Never leave the solution open-ended — always show concrete, working code at every layer.

**Difficulty source:** Judgment and interpretation. The code may be simple; knowing WHAT to compute and WHY it matters is the hard part.

**Example:** *Can you improve a pairs trading strategy beyond simple cointegration? The baseline uses Engle-Granger. The solution shows why Johansen catches baskets that Engle-Granger misses, then explores OU parameter estimation for entry/exit timing. The frontier: "adaptive thresholds based on regime — here's one approach, but this is where your judgment matters."*

---

## The Layered Task Pattern

For Investigation tasks (and ambitious Construction tasks), structure the solution in explicit layers:

```
Layer 1: THE BASELINE
Complete, working, no-excuses implementation.
This is the minimum — every student should get here.
The solution explains every design choice.

Layer 2: THE SMARTER APPROACH
A clearly better method. The solution shows WHY it's better
(with numbers: "Sharpe improves from 0.8 to 1.2"),
and explains the thinking that gets you from Layer 1 to Layer 2.

Layer 3: THE FRONTIER (optional)
"This is where your ingenuity matters." The solution shows
ONE tentative direction, clearly marked as illustrative,
not authoritative. The prose says: "this is one path —
the right approach depends on your data, your universe,
and your risk appetite."
```

**Critical rule:** Layers 1 and 2 must have complete, running code with concrete results. Layer 3 can be a sketch or a thoughtful discussion with partial code, but it must be honest about its limitations. Never present a Layer 3 sketch as if it's a complete answer.

**Why this works:** The student gets a complete, satisfying solution at Layer 2. Layer 3 shows them that the frontier is open without leaving them with broken code or hand-waving. The progression teaches HOW a practitioner thinks, not just WHAT they build.

---

## Difficulty Calibration

The audience is ML/DL experts learning finance. This has a specific implication for task difficulty:

| Dimension | Difficulty level | Why |
|-----------|-----------------|-----|
| **ML/code** | Low to moderate | They already know this. Don't make them implement backprop from scratch. |
| **Finance domain** | High | This is what they're learning. New concepts, new vocabulary, new intuitions. |
| **ML-meets-finance** | The sweet spot | Standard ML approaches fail in domain-specific ways. This is where the real learning happens. |

The best tasks exploit the **ML-meets-finance intersection**: the student applies a technique they know well (random forest, LSTM, cross-validation) and discovers that the financial domain breaks it in a specific, interesting way (look-ahead bias, non-stationarity, survivorship bias, small effective sample sizes).

**Bad difficulty:** "Implement a transformer with rotary positional embeddings from scratch." (Hard ML, no finance insight.)

**Bad difficulty:** "Memorize the tick-size regime for NMS stocks." (Finance trivia, no ML connection.)

**Good difficulty:** "Train an LSTM on raw daily prices. Then train it on stationary returns. Compare. Discover that the raw-price model learns a spurious trend and the returns model doesn't — but the returns model can't capture mean-reversion because differencing killed the memory." (Standard ML technique, domain-specific failure mode, requires domain understanding to diagnose.)

---

## How Tasks Evolve Across Notebooks

The `notebook_creation_guide.md` defines the progression as UNDERSTAND → APPLY → BUILD. Task types map to this:

| Notebook | Primary task types | What's being tested |
|----------|-------------------|---------------------|
| **Lecture** | Demonstration only | Nothing — the student watches and learns |
| **Seminar** | Guided Discovery, Skill Building | Can the student apply concepts to new data and discover patterns? |
| **Homework** | Construction, Investigation, Skill Building | Can the student build systems and reason about open questions? |

**Key constraint from the Non-Overlap Principle:** A concept gets exercised in ONE notebook. The lecture SHOWS survivorship bias; the seminar has students MEASURE it on real data; the homework incorporates survivorship-awareness into a pipeline. The seminar doesn't re-show what the lecture showed. The homework doesn't re-measure what the seminar measured.

This means: when designing tasks for the seminar, check what the lecture demonstrated and go BEYOND it. When designing homework, check what both the lecture and seminar covered and INTEGRATE it at scale.

---

## Solution Completeness Rules

These rules resolve tensions between existing guides:

### The README knows everything. The prompt hints. The solution reveals.

Three audiences see three different levels of information:

1. **The README** (read by the notebook-writing agent): States the expected insight explicitly. "Students should discover that survivorship bias inflates returns by ~2-4%/year." The agent needs this to engineer the path to discovery.

2. **The notebook prompt** (read by the student): Gives direction without spoiling. "Compute returns for both portfolios. What's the annual difference? Is it trivially small or disturbingly large?" The student discovers, not reads.

3. **The solution** (read by the student after attempting): Confirms the insight with specific numbers and explains why. "That 3.2% per year compounds to 54% over 14 years. If your model shows 15% annualized, one-fifth of that might be ghosts."

### "Scaffold" means the prompt. "Production-quality" means the solution.

The homework prompt is a scaffold: clear deliverables, enough direction to start, no pseudo-code. The solution is production-quality: complete, clean, well-structured, explained. These address different readers at different times — they don't contradict.

### Open-ended tasks still need closed solutions.

An Investigation task can ask an open question. The solution cannot be open. The solution must show a concrete baseline, a concrete improvement, and (optionally) a sketched frontier. "It depends" is never a solution — "here's what it depends on, here's how we chose for THIS data, and here's the result" is.

---

## Anti-Patterns

1. **The Kaggle Trap.** Assigning a task where the "real" solution requires weeks of feature engineering, hyperparameter search, and domain-specific iteration — then showing a toy solution that wouldn't compete. If the task is Kaggle-hard, the solution must either be Kaggle-good or the task must be scoped down to what can be solved completely and educationally. Education beats leaderboard performance.

2. **The Hand-Wave Solution.** "Results will vary depending on your parameter choices." This is not a solution. Show specific parameter choices, specific results, and explain why those choices are reasonable. Acknowledge that alternatives exist, but commit to one path.

3. **The Reverse Difficulty.** Making tasks hard because the code is complex rather than because the domain is interesting. An ML expert who struggles with your task should be struggling with the finance, not the Python.

4. **The Trivial Extension.** "Now do the same thing for 100 stocks instead of 10." Scale is only interesting if it reveals something new (edge cases, performance issues, distributional differences). If the result at 100 is the same as at 10, the extension is busywork.

5. **The Impossible Verification.** Assigning a task where the student can't tell if their answer is correct. Every task should have at least one sanity check: a known result to reproduce, a range the answer should fall in, or a qualitative property the output should have.

6. **The Solution That Doesn't Run.** The most important anti-pattern. Every solution code cell must produce actual output. No `# TODO: implement this`. No `pass` placeholders. No code that requires API keys the student doesn't have (unless a clearly documented fallback exists). If the solution doesn't run, the task isn't ready.

7. **Discovery Without Stakes.** "Compute the mean return for each sector." So what? Every discovery task must connect to consequences — money, model quality, or career relevance. If the result doesn't change how you'd build something, it's not worth the student's time.
