# Task Design Guide — Shared Reference

> **Consumers:** Blueprint agent (Step 2), Expectations agent (Step 3), Code agent (Step 4). This guide defines the shared task vocabulary — what kinds of tasks exist, how to calibrate their difficulty, what acceptance criteria look like for each type, and what patterns to avoid.

Voice, prose structure, and formatting rules live in `notebook.md`. This doc covers **what to assign and why**, not how to write it up.

---

## The Implementability Check

Before assigning any task in the blueprint, mentally prototype the solution. If you can see the code, the expected output shape, and the interpretation — the task is assignable. If you're thinking "the downstream agents will figure it out" — they won't. Step 3 (Expectations) formally assesses feasibility, but catching infeasible tasks early saves the pipeline a round trip.

---

## Task Types

Not every task serves the same purpose. Using the wrong type for the moment is a common failure — a "build this system" task when the student needs to "discover this insight" produces engineering practice when it should produce understanding, and vice versa.

### 1. Demonstration (lecture only)

**Purpose:** Show a concept working. Build intuition before formalism. The instructor runs the code; the student watches and reads. No student work.

**When to use:** Introducing new concepts. Showing what goes wrong (the catastrophe before the fix). Motivating why something matters.

**Solution strategy:** The solution IS the demonstration. Full code + prose interpretation. Every demonstration must answer "so what?" — what does this result mean in dollars, in model quality, in career relevance?

**Acceptance criteria pattern:** Output matches a known result or shows a visually dramatic effect. Criteria verify **what the student sees**, not what they produce.

**Example:** *Plot AAPL's unadjusted vs. adjusted price across the 2014 split. Show the 85% "crash" in unadjusted data. Explain why a model trained on this would learn catastrophically wrong patterns.*

### 2. Guided Discovery (seminar, occasionally homework)

**Purpose:** The student follows a constrained path and discovers a specific insight. The insight is the payoff, not the code.

**When to use:** When the "aha moment" matters more than the implementation. When the domain reveals something that ML intuition wouldn't predict.

**Solution strategy:** The solution shows the exact path AND the exact insight. Numbers are concrete ("the bias is 3.2% per year, compounding to 54% over 14 years"). The insight must be new, grounded in data, and connected to consequences.

**Difficulty source:** The domain, not the code. Code should be straightforward; the surprise comes from what the data reveals.

**Acceptance criteria pattern:** The insight itself is the criterion — a specific numerical relationship, a pattern, or a comparison that the student discovers. Criteria verify the insight emerged, not just that code ran.

**Example:** *Compute equal-weight portfolio returns for current S&P 500 constituents back to 2010. Compare to returns using only tickers with complete histories. The student discovers survivorship bias inflates returns by 1-4% per year.*

### 3. Skill Building (seminar and homework)

**Purpose:** Build fluency with tools, procedures, or domain-specific workflows. The answer is well-defined; the challenge is doing it correctly and efficiently.

**When to use:** Early weeks. New tools or libraries. Procedures the student will repeat throughout the course (computing returns, handling corporate actions, storing data). When "get comfortable" is genuinely the goal.

**Solution strategy:** The solution shows the idiomatic, production-quality approach. Brief interpretation — the educational value is in the doing, not in a surprise result.

**Difficulty source:** Getting the details right. Financial data has edge cases (missing days, timezone alignment, ticker format variations) that reward careful implementation. Difficulty comes from the domain's messiness, not from algorithmic complexity.

**Acceptance criteria pattern:** Correct implementation of a procedure — data shape, dtype, completeness, round-trip integrity. Criteria verify **the artifact's properties**.

**Example:** *Download 10 years of daily data for 20 stocks across market caps. Compute daily log returns. Detect and handle missing trading days. Store in Parquet with Polars.*

### 4. Construction (homework, occasionally seminar)

**Purpose:** Build a system or tool. The spec is clear; the engineering and design decisions are the student's.

**When to use:** When the deliverable is a reusable artifact (a DataLoader class, a backtesting pipeline, a risk model). When the educational value comes from the design process — choosing data structures, handling edge cases, making tradeoff decisions.

**Solution strategy:** The solution shows ONE complete implementation. Design choices are explained in prose ("we chose long format over wide format because..."). Meaningful alternatives are acknowledged but not all implemented. Code is production-quality: docstrings, clean structure, thoughtful variable names.

**Difficulty source:** Integration and scale. Building for 5 stocks is easy; building for 50 reveals edge cases. The challenge is robustness, not cleverness.

**Acceptance criteria pattern:** The artifact works under stress — handles edge cases, meets robustness requirements, passes integration tests. Criteria verify **behavior at scale**.

**Example:** *Build a `FinancialDataLoader` class that downloads, cleans, validates, and stores multi-asset data. It should handle at least 50 tickers, detect quality issues, and produce a per-ticker quality scorecard.*

### 5. Investigation (homework, advanced seminar)

**Purpose:** Explore an open-ended question where multiple approaches are valid. This is the closest to real research — the student must formulate an approach, not just execute one.

**When to use:** Later weeks. When the domain question is genuinely interesting and the "right" answer depends on context or judgment. When you want students to experience the ambiguity of real quantitative research.

**Solution strategy:** Use **The Layered Task Pattern** (below). Never leave the solution open-ended — always show concrete, working code at every layer. "It depends" is never a solution — "here's what it depends on, here's how we chose for THIS data, and here's the result" is.

**Difficulty source:** Judgment and interpretation. The code may be simple; knowing WHAT to compute and WHY it matters is the hard part.

**Acceptance criteria pattern:** Criteria verify each layer independently. Layer 1 baseline has concrete thresholds. Layer 2 demonstrates measurable improvement over Layer 1. Layer 3 (if present) has qualitative criteria only.

**Example:** *Can you improve a pairs trading strategy beyond simple cointegration? Baseline: Engle-Granger. Layer 2: Johansen for basket detection. Frontier: adaptive thresholds based on regime.*

---

## The Layered Task Pattern

For Investigation tasks (and ambitious Construction tasks), structure the solution in explicit layers:

**Layer 1: THE BASELINE** — Complete, working, no-excuses implementation. Every student should get here. The solution explains every design choice.

**Layer 2: THE SMARTER APPROACH** — A clearly better method. The solution shows WHY it's better (with numbers: "Sharpe improves from 0.8 to 1.2") and explains the thinking that gets you from Layer 1 to Layer 2.

**Layer 3: THE FRONTIER** (optional) — "This is where your ingenuity matters." ONE tentative direction, clearly marked as illustrative, not authoritative. Honest about its limitations.

**Critical rule:** Layers 1 and 2 must have complete, running code with concrete results. Layer 3 can be a sketch or a thoughtful discussion with partial code, but it must be honest about being a sketch. Never present a Layer 3 sketch as a complete answer.

---

## Difficulty Calibration

The audience is ML/DL experts learning finance. This shapes every difficulty decision:

| Dimension | Difficulty | Why |
|-----------|-----------|-----|
| **ML/code** | Low to moderate | They already know this. Don't make them implement backprop. |
| **Finance domain** | High | This is what they're learning. New concepts, vocabulary, intuitions. |
| **ML-meets-finance** | The sweet spot | Standard ML fails in domain-specific ways. This is the real learning. |

The best tasks exploit the **ML-meets-finance intersection**: the student applies a technique they know well (random forest, LSTM, cross-validation) and discovers that the financial domain breaks it in a specific, interesting way (look-ahead bias, non-stationarity, survivorship bias, small effective sample sizes).

**Good difficulty:** "Train an LSTM on raw daily prices. Then train it on stationary returns. Compare. Discover that the raw-price model learns a spurious trend and the returns model can't capture mean-reversion because differencing killed the memory."

**Bad difficulty:** "Implement a transformer with rotary positional embeddings from scratch." (Hard ML, no finance insight.)

**Bad difficulty:** "Memorize the tick-size regime for NMS stocks." (Finance trivia, no ML connection.)

---

## Acceptance Criteria by Task Type

The Expectations agent (Step 3) sets criteria for each section, exercise, and deliverable. The Code agent (Step 4) verifies them as assertions. This table maps task types to what "correct" means:

| Task Type | What criteria verify | Example criterion |
|-----------|---------------------|-------------------|
| **Demonstration** | The output shows the intended effect | "Nominal return on split date < -80%" |
| **Guided Discovery** | The insight emerged from the data | "Survivorship bias premium > 1% annualized" |
| **Skill Building** | The procedure was done correctly | "DataFrame has ≥2500 rows per ticker, no NaNs in Close" |
| **Construction** | The artifact works under stress | "Handles ≥50 tickers without crash; scorecard flags known-bad tickers" |
| **Investigation** | Each layer meets its own bar | "Layer 1 Sharpe > 0; Layer 2 Sharpe > Layer 1 Sharpe" |

**Rules for all criteria:**
- **Approximate:** Use ranges and inequalities, not exact values. Financial data shifts over time.
- **Machine-checkable:** Every criterion must be verifiable via assertion or structured inspection.
- **Two-sided:** Cover both the happy path (expected results) and the failure mode (what goes wrong without correct implementation).
- **Tied to learning:** If a criterion doesn't support the section's "so what?", it's noise. Cut it.
- **ML-aware:** For tasks that train models, criteria should reflect proper ML methodology (OOS evaluation, appropriate validation). See `rigor.md` for the engineering standard that affects what results are realistic and how they must be produced.

---

## Anti-Patterns

1. **The Kaggle Trap.** Assigning a task where the "real" solution requires weeks of feature engineering — then showing a toy solution that wouldn't compete. If the task is Kaggle-hard, the solution must be Kaggle-good or the scope must shrink. Education beats leaderboard performance.

2. **The Hand-Wave Solution.** "Results will vary depending on your parameter choices." This is not a solution. Show specific choices, specific results, and explain why those choices are reasonable. Acknowledge alternatives, but commit to one path.

3. **The Reverse Difficulty.** Making tasks hard because the code is complex rather than because the domain is interesting. An ML expert who struggles with your task should be struggling with the finance, not the Python.

4. **The Trivial Extension.** "Now do the same thing for 100 stocks instead of 10." Scale is only interesting if it reveals something new (edge cases, distributional differences, performance issues). If the result at 100 is the same as at 10, the extension is busywork.

5. **The Impossible Verification.** Assigning a task where the student can't tell if their answer is correct. Every task should have at least one sanity check: a known result to reproduce, a range the answer should fall in, or a qualitative property the output should have.

6. **The Solution That Doesn't Run.** Every solution code cell must produce actual output. No `# TODO: implement this`. No `pass` placeholders. No code that requires API keys the student doesn't have (unless a clearly documented fallback exists). If the solution doesn't run, the task isn't ready.

7. **Discovery Without Stakes.** "Compute the mean return for each sector." So what? Every discovery task must connect to consequences — money, model quality, or career relevance. If the result doesn't change how you'd build something, it's not worth the student's time.
