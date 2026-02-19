# Notebook Creation Guide

> **The cardinal rule:** Each notebook type has a distinct role. If a student could skip one notebook and lose nothing, that notebook has failed.

> **The storytelling rule:** Code teaches *what*. Prose teaches *why*, *so what*, and *what it costs you*. A notebook that is mostly code is a script, not a lesson. Every code cell must be surrounded by prose that gives it meaning.

---

## The Narrative Standard

Before anything else: **the single most important quality of a good notebook is narrative density.** The writing guidelines describe a voice — "a sharp, opinionated practitioner who respects your intelligence." That voice must be present *throughout* the notebook, not just in section headers and "So what?" bridges.

### The No-Silent-Code Rule

**Never place two code cells back-to-back without a prose cell between them.**

This is the most important rule in this document. Violations are the #1 failure mode.

A "prose cell" means a markdown cell with **at least 2-3 substantive sentences** that do one or more of:
- Interpret what the previous code showed ("Notice that kurtosis varies by a factor of 10x across sectors...")
- Create tension about what comes next ("But here's the problem nobody tells you about...")
- Bridge two concepts ("Now that we've seen the disease, let's look at the cure...")
- Provide a "Did you know?" moment or real-world grounding
- Challenge the student's assumptions ("You might expect dollar bars to always win. They don't.")

**Not** a prose cell:
- "Let's see this visually." (6 words — this is a caption, not commentary)
- "### Step 3: Return Computation" (a heading is not prose)
- "Now let's run the Jarque-Bera test." (a plot setup is not commentary)
- Any cell under 15 words

### Prose-to-Code Ratios (Hard Minimums)

| Notebook | Min markdown cells (% of total) | Min cells with ≥3 sentences | Max consecutive code cells |
|----------|--------------------------------|-----------------------------|----|
| Lecture | 60% | 40% of total | 1 (NEVER two code cells in a row) |
| Seminar | 45% | 25% of total | 2 (between exercise sections only) |
| Homework | 40% | 20% of total | 2 (within solution blocks only) |

### What Goes in Prose vs. Code

**Prose (markdown cells):**
- All stories, analogies, "Did you know?" moments
- All interpretations of results ("This means...")
- All bridges between concepts
- All "So what?" conclusions
- All formula introductions and derivations
- All plot setups ("Here's what to watch for...") AND plot interpretations ("See how the tails...")

**Code cells:**
- Computation and visualization ONLY
- Short (≤15 lines preferred, ≤25 max)
- No `print()` statements that deliver narrative content
- Minimal inline comments (the surrounding prose does the explaining)

**The print() Prohibition:** Never use `print()` to deliver narrative, teaching, or storytelling content. This pattern:
```python
print("Companies you CANNOT download from yfinance:")
print("  Enron (bankrupt 2001) -- ticker ENRNQ, delisted")
print("\n--> This is survivorship bias: the dead are invisible.")
```
is **banned**. This belongs in a markdown cell with proper formatting, emphasis, and voice. `print()` is for data output (statistics, table rows, confirmation messages), never for teaching.

### The Transition Cell Template

Between every code cell and the next, use this mental template for the prose cell:

> **[Interpret]** — What did we just see? What number/pattern should the student notice?
> **[Connect]** — How does this connect to a real-world consequence, a story, or the next concept?
> **[Tension]** — What question does this raise? What's surprising, wrong, or incomplete?

Not every transition needs all three, but it needs at least one done well. A 2-sentence transition that interprets the output and creates tension for the next cell is the minimum.

**Bad transition (caption):**
> "Now let's visualize the order book."

**Good transition (interpretation + tension):**
> "Look at those volumes: 500-800 shares stacked on the bid side, 300-600 on the ask. That asymmetry isn't random — it means more people want to buy than sell at these prices. In practice, HFT firms monitor exactly this ratio, updating their models every 50 microseconds. We'll work with a much slower version of this signal in Week 13.
>
> But for now, let's see what this book *looks like*. The visualization below plots bids on the left and asks on the right — two armies facing each other across the spread."

---

## The Three Notebook Types

### 1. `lecture.ipynb` — The Conceptual Journey

**Purpose:** Teach the *why* behind every concept. Build intuition, tell stories, show carefully chosen demonstrations. The student reads/watches this — they don't "do" it.

**What it IS:**
- A narrative-driven document that reads like a great technical blog post or a recorded lecture with code demonstrations
- A curated sequence of ideas, each building on the last, with the full personality of the writing guidelines
- Demonstration code: short, focused cells that prove a single point, always wrapped in narrative
- The place where all formulas, definitions, and theory live

**What it is NOT:**
- A workshop or lab (that's the seminar)
- A code repository with narrative bookends
- A place where students write code (they run yours)
- An exhaustive reference (link to docs/papers for depth)
- A playground for building classes or pipelines (that's the homework)

**Structure (45-70 cells, ≥60% markdown):**

```
Cell 0:   Title + epigraph (1 markdown cell)
Cell 1:   Opening Hook — a story, disaster, or provocation (1 LONG markdown cell, 3-5 paragraphs)
Cell 2:   Setup — imports + pip install + utility functions (1 code cell, hide the plumbing)

Sections 1-N (following the README arc):
  Each section follows the RHYTHM: Hook → Intuition → Formal → Code → Interpret → "So what?"
    - Section header + narrative hook (markdown, 2-4 paragraphs with personality)
    - Intuition / story / analogy (markdown, can be same cell or separate)
    - Formal concept + LaTeX (markdown)
    - Prose setup: "here's what to look for" (markdown, 2-3 sentences min)
    - DEMONSTRATION code (1 cell, ≤15 lines)
    - Prose interpretation: "here's what it means and what it costs you" (markdown, 3-5 sentences)
    - (Optional: second code cell for visualization, again with prose before and after)
    - "So what?" bridge (markdown, 2-3 sentences connecting to next concept or real money)

Closing:
  - Summary table of key formulas/concepts (markdown)
  - Career Connections (markdown, 2-3 paragraphs)
    → Maps this week's skills to 2-4 specific roles (from README's Career Connections section)
    → Concrete: "A risk analyst at a multi-strategy fund runs exactly this analysis every morning"
    → Names real firms, real workflows, real job titles — not generic "useful in finance"
    → Tone: conversational, like a senior colleague telling you what the job actually looks like
  - Bridge to next week — create anticipation (markdown, 1-2 paragraphs)
  - Suggested reading with "why you'd read this" annotations (markdown)
```

**Key rules for lectures:**

1. **NEVER two code cells in a row.** Every code cell gets a prose cell before AND after it. The prose before says what to look for. The prose after says what it means.

2. **Demonstration code, not exercise code.** Show the output of `make_dollar_bars()` and its effect on kurtosis. Don't have the student build `make_dollar_bars()` — that's the seminar's job.

3. **One concept per code cell.** Never combine "download data" + "compute returns" + "make plot" in one cell. Each cell proves exactly one thing, in ≤15 lines.

4. **Hide the plumbing.** Put a thin utility cell early on for common boilerplate (MultiIndex handling, plot defaults), then never repeat it. Students shouldn't see `if isinstance(data.columns, pd.MultiIndex)` more than once.

5. **Show the catastrophe.** Before showing the fix, show what goes wrong. Plot raw prices into an LSTM's input range. Show the 95% "crash" from an unadjusted split. Let them *see* the problem before you solve it.

6. **The lecture should NOT build reusable classes.** It can sketch out pseudo-code or show a 10-line prototype to motivate the homework, but the full DataLoader build belongs in the homework.

7. **Pace the formulas.** Maximum 2 LaTeX formulas per section. Follow the formula pattern from `writing_guidelines.md`.

8. **Every plot follows Rule 7 from `writing_guidelines.md`** — a prose cell before (what to look for) and a prose cell after (what it means). This is a specific case of the No-Silent-Code Rule.

9. **"Did you know?" moments (Rule 5 from `writing_guidelines.md`)** — at least 3-4 per lecture, woven into the narrative flow, never boxed as sidebars.

10. **The voice never disappears.** The `writing_guidelines.md` voice must be present in every prose cell, not just section headers. If a section reads like a textbook, rewrite it.

11. **Career relevance is woven in, not bolted on.** The closing has a dedicated Career Connections section (see structure above), but career implications should also appear naturally in "so what?" moments and interpretation prose throughout the lecture. When a concept maps directly to a job function, say so: "This is literally what a junior quant at Citadel does on day one." Don't save all career content for the closing — the best moments are mid-explanation, when the student realizes this abstract concept is someone's daily workflow.

**Cell budget (example for a 60-cell lecture):**
- Markdown cells: 38 (63%)
- Code cells: 22 (37%)
- Of the 38 markdown cells, at least 24 should have ≥3 substantive sentences
- Zero consecutive code cells anywhere


### 2. `seminar.ipynb` — The Hands-On Lab

**Purpose:** The student practices *doing* things the lecture only demonstrated. Exercises push beyond the lecture — they apply concepts to new data, new situations, or new scales. They make mistakes and fix them.

**What it IS:**
- Guided exercises where students write real code to answer specific questions
- Applications of lecture concepts to situations NOT covered in the lecture
- A place where students encounter surprising results and learn from the surprise
- Where code gets built, tested, and interpreted — with prose guiding the interpretation

**What it is NOT:**
- A repeat of the lecture with "now you try" prompts
- A code dump with thin exercise headers
- A place for extensive theory (refer back to lecture)
- A set of trivial warmup exercises (students already heard the lecture)

**How it differs from the lecture:**
| Aspect | Lecture | Seminar |
|--------|---------|---------|
| SPY QQ-plot shown? | Yes, as demonstration | No — done in lecture |
| Dollar bars built? | Shown: 5-line napkin version | Exercise: build from scratch, optimize, compare multiple assets |
| Survivorship bias? | Story + printed examples | Exercise: quantify the actual return gap |
| DataLoader class? | Not built (just motivated) | Not built (that's homework) |
| Transaction costs? | Mentioned with numbers | Exercise: build a cost model and watch a strategy die |
| Voice? | Full narrative | Leaner, but still present — every exercise has personality |

**Structure (25-45 cells, ≥45% markdown):**

```
Cell 0:   Brief intro — what we're doing and why, with personality (markdown, 4-6 sentences)
Cell 1:   Shared imports + data download (ALL data for ALL exercises, downloaded ONCE)

Exercises (3-5 per seminar):
  Each exercise:
    - Exercise header + "The question we're answering:" (markdown, 2-3 paragraphs with motivation)
    - Task list: specific, numbered, unambiguous (markdown, can be in same cell)
    - Student workspace: empty code cells with # guiding comments (2-4 cells)
    - ─── Solution ─── separator (markdown)
    - Solution code cell 1 (code, ≤15 lines)
    - Interpretation prose (markdown, 2-4 sentences: "Notice that...")
    - Solution code cell 2 (code, ≤15 lines)
    - Interpretation prose (markdown, 2-4 sentences)
    - (Optional: cell 3 + interpretation)
    - Insight paragraph (markdown, 4-8 sentences)
      → This insight must be DIFFERENT from the lecture — something new the data reveals
      → It should reference specific numbers from the output
      → It should connect to a real-world consequence

Summary:
  - 3-5 bullet points recapping what was discovered (each bullet is a genuine insight, not a concept restatement)
  - Brief bridge to homework (1-2 sentences)
```

**Key rules for seminars:**

1. **Never repeat a lecture exercise.** If the lecture showed a QQ-plot of SPY, the seminar should NOT ask the student to make a QQ-plot of SPY. It could ask them to compare QQ-plots across 10 stocks and discover that kurtosis varies by sector — that's a new insight.

2. **Exercises should have a question, not just a task.** Bad: "Build volume bars." Good: "Does the dollar-bar advantage hold for volatile stocks like TSLA the same way it holds for SPY? Build bars for both and compare. Is Lopez de Prado's claim universal, or does it depend on the stock?"

3. **The solution insight must surprise and reference the data.** Bad: "Fat tails exist." Good: "TSLA's kurtosis dropped from 18.4 to 9.2 with dollar bars — a 50% reduction. SPY only dropped from 11.3 to 10.1. The dollar-bar advantage is proportional to how much volume variation the stock has. This makes sense: TSLA's earnings days see 5x normal volume, so resampling by dollar volume makes a bigger difference."

4. **Prose between every solution code cell.** Even in solutions, no two code cells back-to-back without a prose cell. The prose should interpret what the code just showed and set up what comes next. At minimum: "Look at the numbers above. [Interpretation]. Now let's visualize this:"

5. **Keep theoretical prose lean, but never eliminate voice.** The seminar is not a second lecture, but it's not a silent lab either. Exercise setups should have 2-3 sentences of motivation. Solution interpretations should have 2-4 sentences of commentary. The personality of the writing guidelines should be present, just compressed.

6. **One data download for the whole seminar.** Don't re-download SPY in every exercise. Download everything needed at the top.

7. **Build toward the homework.** The last exercise should naturally lead into what the homework asks. End with something like: "You've now seen this for 2 stocks. In the homework, you'll scale this to 200 — and discover that the patterns get more interesting at scale."


### 3. `hw.ipynb` — The Challenge

**Purpose:** An extended, integrative project that pushes students to apply, combine, and extend what they learned. The homework is where students build real things — classes, pipelines, reports. It's substantially harder than the seminar.

**What it IS:**
- A mission with clear deliverables and a narrative framing
- Where students build production-quality code (DataLoader, pipeline, model)
- Where students discover things that surprise them
- Where the solution teaches things the lecture and seminar couldn't cover

**What it is NOT:**
- A larger version of the seminar exercises
- A repeat of lecture content at larger scale
- A hand-holding walkthrough (the outline gives direction, the student figures out how)
- A code repository with a title page (solutions need narrative and interpretation too)

**How it differs from the seminar:**
| Aspect | Seminar | Homework |
|--------|---------|---------|
| Scale | 1-2 stocks, focused exercises | 50-200 stocks, full pipeline |
| Complexity | Single concepts applied | Multiple concepts combined |
| Code output | Throwaway analysis cells | Reusable class/module |
| Discovery | Guided ("build X, observe Y") | Open-ended ("find and document 3 issues") |
| Time investment | 30-45 minutes | 2-4 hours |
| Voice | Lean but present | Mission framing has full voice; solutions have genuine "aha" commentary |

**Structure (35-75 cells, ≥40% markdown):**

```
Cell 0:   Title (markdown)
Cell 1:   Mission framing — why this matters, what you'll build, why it's not busywork
          (markdown, 3-4 paragraphs with the full narrative voice)
Cell 2:   Deliverables list — numbered, specific, unambiguous (markdown)
Cell 3:   Imports (code)

For each deliverable:
  - Deliverable header + instructions (markdown, 1-2 paragraphs)
    → Instructions are clear but NOT pseudo-code. Direction, not dictation.
    → Include a teaser: "Step 3 is where it gets interesting — you'll find that..."
  - Student workspace: empty cells or scaffolded code with # TODOs (2-5 cells)
  - ━━━ SOLUTION ━━━ separator (markdown, visually distinct from seminar)
  - Solution code cell 1 (code, ≤25 lines)
  - Interpretation prose (markdown, 3-5 sentences: what the output shows, why it matters)
  - Solution code cell 2 (code, ≤25 lines)
  - Interpretation prose (markdown, 3-5 sentences)
  - (Continue pattern for remaining solution cells)
  - "Aha moment" paragraph (markdown, 4-8 sentences)
    → Must provide insight BEYOND what lecture + seminar covered
    → Must reference specific numbers or patterns from the output
    → Should connect to what professionals care about

Closing:
  - Summary of discoveries (markdown, 5-8 bullet points)
  - Each bullet is a genuine finding, not a concept restatement
```

**Key rules for homework:**

1. **The outline is a scaffold, not a tutorial.** Give clear deliverables with enough direction to start, but don't write pseudo-code for the solution. Bad: "Use `.cumsum()` and `// threshold` to group bars." Good: "Build dollar bars using any approach. Test with at least 2 threshold values."

2. **Solution code follows the same prose rules as lectures.** No two code cells back-to-back. Every code cell gets interpretation prose after it. The prose should say "here's what those numbers mean" and "here's what surprised me."

3. **Break large code into digestible pieces.** A 200-line DataLoader class is NEVER one cell. Break it into logical sections (downloading, cleaning, returns, bars, quality) with prose between each section explaining the design choice.

4. **The "aha moments" in solutions must pass the Aha Moment Quality Bar** (see Storytelling Standards below). They should reveal something the student couldn't know until they ran the full homework.

5. **The mission framing matters.** Follow the homework voice from `writing_guidelines.md` — briefing, not bureaucracy.

6. **Don't pre-write the "discovery" text.** The data quality report, for example, should be structured as a template with clear headings that students fill in based on what they find — not a pre-written essay that happens to match the data exactly.

7. **Solution code should be production-quality.** Docstrings, clean structure, thoughtful variable names. This is the code students will reference for the rest of the course.

---

## How the Three Notebooks Connect

```
LECTURE: "Here's WHY fat tails matter" → [shows SPY histogram, tells 1987 story]
    ↓
SEMINAR: "HOW do fat tails vary across assets?" → [student compares 10 stocks, finds sector patterns]
    ↓
HOMEWORK: "BUILD a system that detects and handles fat tails at scale" → [200-stock pipeline with anomaly flags]
```

The progression is: **UNDERSTAND → APPLY → BUILD**

### The Non-Overlap Principle

Draw up a concept matrix before writing. For each major concept, decide WHERE it lives:

| Concept | Lecture | Seminar | Homework |
|---------|---------|---------|----------|
| QQ-plots | Demo: SPY | Compare: 10 stocks by sector | At scale: 200 stocks, cross-sectional |
| Dollar bars | Demo: 5-line napkin version | Build from scratch: SPY + TSLA | Integrate into DataLoader class |
| Survivorship bias | Story: Enron/WorldCom | Quantify: survivor vs removed returns | Document in quality report |
| Transaction costs | Anchored number: "25% per year" | Build: cost model + strategy erosion | Integrate into pipeline |
| DataLoader class | Motivate: "why you need one" | Not covered | Build: full class, 200 tickers |
| Returns math | Derive: simple vs log | Not covered (done in lecture) | Compute at scale, find divergences |

A concept should appear in ONE notebook's exercises. The other notebooks can reference it, but shouldn't re-do it.

---

## Storytelling Standards

### Voice Calibration Per Notebook Type

The voice for all notebooks is defined in `writing_guidelines.md` (the 10 rules + practical notes). The calibration per notebook type:

- **Lecture:** Full voice — all 10 rules at maximum intensity
- **Seminar:** Compressed — exercise setups still hook the student, solution interpretations still have personality, but prose is leaner between exercises
- **Homework:** Mission framing gets full voice; solution commentary delivers genuine insight with personality

See the practical notes in `writing_guidelines.md` for examples of each.

### The "Aha Moment" Quality Bar

Every insight/aha moment in a seminar or homework must pass this test:

1. **Is it NEW?** Does it say something the lecture didn't already say? If the lecture proved fat tails exist, the seminar insight can't just say "fat tails exist." It must say something like "fat tail severity varies by sector, and the variation is predictable."

2. **Is it GROUNDED?** Does it reference specific numbers from the output? Bad: "Kurtosis varies across stocks." Good: "DUK's kurtosis is 4.2. TSLA's is 31.7. That's a 7.5x difference — and they're both in the S&P 500."

3. **Does it have CONSEQUENCES?** Does it connect to something that costs money, breaks a model, or changes how you'd build a system? Bad: "The survivorship bias premium is positive." Good: "That 3.8% per year compounds to a 68% overstatement over our 14-year backtest. If your model shows 15% annualized, 3.8 of those points might be ghosts."

4. **Would a student REMEMBER it at the bar?** The best insights are the ones students retell to friends. "Did you know that TSLA's kurtosis drops by 50% when you switch to dollar bars, but SPY's only drops 10%?" is memorable. "Dollar bars produce more Gaussian returns" is not.

### The "Ideal vs. Actual" Rule

**When code results diverge from textbook expectations, the notebook must teach the gap — explicitly, with voice, and with the full weight of the writing guidelines.** These are not footnotes or apologies. They are often the most valuable teaching moments in the entire week.

The observation report's `Implementation Reality` section documents these gaps with teaching angles. The notebook agent must weave each one into the narrative at the point where the student would first notice it. Two framings, depending on the type of gap:

**Data limitations** (our universe/data is constrained):
> "With a CRSP universe of 4000+ stocks, this correlation would be north of 0.75. We're getting 0.04 — not because our code is wrong, but because S&P 500 'small' stocks are the rest of the market's 'large' stocks. This is universe bias: the single most common silent error in quant research. The first thing you'd do at a fund is fix the universe."

**Real phenomena** (the result IS what happens in production):
> "Your HML is correct. Value really did lose money over this period. You're looking at what the industry calls 'the death of value' — one of the most contested debates in modern quant finance. Whether this is temporary mean-reversion or a structural break from intangible-asset dominance is still unresolved."

**Rules:**
- Never hide or minimize a gap. If the student would expect X from the lecture but the code shows Y, the prose must address it head-on.
- Always make the gap actionable: what would you do differently in production? What does this teach about the method's limitations?
- Data limitations and real phenomena demand different tones. A data limitation is a lesson in data engineering ("here's how you'd fix it"). A real phenomenon is a lesson in markets ("here's what's actually happening and why the textbooks might be wrong").
- These moments deserve the most narrative weight. They're where the student transitions from textbook learner to practitioner.

### The Prose-Around-Code Discipline

Each lecture section follows the rhythm from `writing_guidelines.md` Rule 10 (Hook → Intuition → Formal → Code → "So what?"). In notebook execution, the Code step expands into three cells:

1. **Prose setup cell** (2-3 sentences) — "Here's what we're about to see, and what the boring outcome would be."
2. **Code cell** (1 cell, ≤15 lines) — Show it working. One concept only.
3. **Prose interpretation cell** (3-5 sentences) — "See that number? Here's what it means. Here's what it costs you."

**This is where most notebooks fail.** They skip the setup and skip the interpretation, leaving bare code between narrative islands. The No-Silent-Code Rule exists because of this failure mode.

---

## Formatting Standards

### Imports and Setup
- ONE imports cell per notebook, placed immediately after the intro
- Include `pip install` in the same cell (or the cell before)
- Set matplotlib defaults in the imports cell, never repeat
- Include utility functions as needed for data loading quirks (e.g., yfinance MultiIndex handling, API auth wrappers). Keep these minimal and invisible — students shouldn't linger on plumbing
- This utility cell should be as short as possible; if more than ~10 lines of boilerplate are needed, put them in a collapsed helper or a brief "Setup" section with a one-line explanation
- **Notebooks are standalone documents.** They must NEVER import from `code/data_setup.py` or any other file in the `code/` directory. The `code/` directory exists only for Phase 3 verification — `.py` files import from `data_setup.py` so they can share a data layer during development, but notebooks download data directly via API calls (e.g., `yf.download()`). The notebook agent reads `data_setup.py` to understand *what* data to download and *how*, then writes equivalent direct download calls in the notebook. Students should be able to open any notebook and run it without any external files.
- **No caching logic in notebooks.** Do NOT create cache directories, write/read parquet caches, or add file-existence checks. `data_setup.py` uses caching because `.py` code files are run repeatedly during Phase 3 development — that's appropriate there. Notebooks are run once top-to-bottom by students; caching adds complexity for zero benefit. Just call the API directly and keep the data in memory.

### Separators

**Seminar answer separator:**
```markdown
---
### ▶ Solution
```

**Homework solution separator:**
```markdown
---
## ━━━ SOLUTION: Deliverable N ━━━
```

### Code Cell Guidelines
- Max 25 lines per code cell (prefer 10-15)
- Each cell does ONE thing
- `print()` prohibition applies (see above)
- Never suppress warnings globally — fix them or explain them
- Class definitions: break into logical sections with prose between them. In homework notebooks, the `ClassName.method = method` monkey-patching pattern is acceptable for building classes incrementally across cells. In lectures and seminars, present classes complete or as short sketches.

### Narrative Voice
- All 10 rules from `writing_guidelines.md` apply to all three notebook types
- Per-notebook voice calibration: see **Voice Requirements Per Notebook Type** above

### LaTeX
- Primarily in lectures and homework solutions
- Always follows: intuition → formula → code → insight
- Use `\underbrace` for dense formulas
- Maximum 2 formulas per section (lectures), 1 per deliverable (homework solutions)

---

## Anti-Patterns (Do NOT Do These)

### Structural Anti-Patterns

1. **Don't repeat exercises across notebooks.** If the lecture shows SPY's QQ-plot, the seminar and homework must NOT ask students to produce SPY's QQ-plot.

2. **Don't copy the README verbatim.** The README is a spec. The notebook is the realization. Use the README's ideas and hooks, but write the actual narrative fresh — the README describes what to teach, the notebook teaches it.

3. **Monkey-patching in homework only.** The `ClassName.method = method` pattern is acceptable in homework notebooks for building classes incrementally across cells with prose between each piece. Avoid in lectures and seminars.

4. **Don't have three identical import blocks.** Each notebook gets ONE imports cell.

5. **Don't write 200-line code cells.** Break them up. Each cell ≤ 25 lines, ideally ≤ 15.

6. **Don't download data multiple times.** Each notebook downloads ALL needed data in one cell near the top.

7. **Don't build the same thing in lecture and homework.** If the homework builds a DataLoader, the lecture should motivate it and maybe show a 5-line sketch.

### Storytelling Anti-Patterns

8. **Don't place two code cells back-to-back.** (See the No-Silent-Code Rule above — this is the same rule stated as a "don't.")

9. **Don't use `print()` for storytelling.** (See the print() Prohibition above.)

10. **Don't write caption-only transitions.** "Let's see this visually" is not a prose cell. It's a caption. Replace with 2-3 sentences that interpret, bridge, or create tension.

11. **Don't let the voice disappear in code sections.** If a section reads like a textbook between the hook and the "So what?", the middle needs rewriting.

12. **Don't recycle lecture insights as seminar/homework insights.** Every insight must be new, grounded in specific numbers, and connected to consequences. (See the Aha Moment Quality Bar above.)

13. **Don't pre-write the student's discoveries.** The homework data quality report should be a template to fill in, not a pre-written document.

14. **Don't let every exercise succeed on the first try.** Good pedagogy shows failure. Show what goes wrong with raw prices, unadjusted splits, or bad data — then fix it.

15. **Don't treat section headers as prose.** "### Step 3: Return Computation" is a heading. It needs at least a paragraph after it.

---

## Quality Checklist (Before Shipping)

### Structure
- [ ] **No exercise repetition** across the three notebooks
- [ ] **Lecture has NO exercise prompts** (only demonstrations)
- [ ] **Seminar exercises are all NEW** (not lecture repeats)
- [ ] **Homework deliverables are integrative** (combine multiple concepts)
- [ ] **Each notebook has exactly ONE imports/setup cell**
- [ ] **No MultiIndex handling code appears more than once per notebook**

### Narrative Density
- [ ] **ZERO consecutive code cells** in the lecture
- [ ] **Max 2 consecutive code cells** in seminar/homework (and only in solution blocks)
- [ ] **Every markdown cell between code cells has ≥2-3 sentences** of substantive prose
- [ ] **No `print()` statements delivering narrative content** — only data output
- [ ] **No caption-only transitions** ("Let's visualize this" without interpretation is not enough)
- [ ] **Prose-to-code ratio meets minimums** (60%/45%/40% for lecture/seminar/hw)

### Storytelling & Voice
- [ ] **The writing guidelines voice is present throughout** — not just in section headers
- [ ] **At least 3-4 "Did you know?" moments** in the lecture, woven naturally
- [ ] **Every seminar exercise setup has personality** — a question with stakes, not a task list
- [ ] **Every seminar insight says something the lecture didn't** — with specific numbers
- [ ] **Every homework "aha moment" says something the seminar didn't** — with specific numbers
- [ ] **The mission framing reads like a mission**, not a government form
- [ ] **At least one "failure mode" is demonstrated** (in lecture or seminar)

### Code Quality
- [ ] **No code cell exceeds 25 lines** (prefer ≤ 15)
- [ ] **No monkey-patching in lectures/seminars** (acceptable in homework for incremental class builds)
- [ ] **Data is downloaded once per notebook**, not per exercise
- [ ] **Solution code is production quality** (docstrings, clean structure)
- [ ] **The closing bridge connects to next week** (lecture only)

### Career Relevance
- [ ] **The lecture closing has a Career Connections section** mapping skills to 2-4 specific roles
- [ ] **Career relevance appears at least 2-3 times in the lecture body** (woven into "so what?" moments, not just the closing)
- [ ] **Seminar/homework "so what?" prose occasionally connects to careers** (not required in every exercise, but at least 1-2 per notebook)
