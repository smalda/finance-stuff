# Notebook Guide — Step 7

> **Consumer:** Notebook agents (Step 7). Three of you run in parallel — one per notebook type (lecture, seminar, homework). Your job is prose, layout, and voice: turn verified code and the consolidated teaching plan into polished course notebooks. You use code verbatim — never modified, never reordered. You never invent new computation. You never decide how to frame divergences — `narrative_brief.md` has already made those decisions for you.

---

## Inputs

In addition to `common.md`:
- **`blueprint.md`** from Step 2 — narrative arc, opening hook, section hooks, bridges, career connections, exercise framing. The creative vision for the week.
- **`narrative_brief.md`** from Step 6 — per-section teaching angles, key numbers, divergence categories, how to frame results. The authoritative teaching plan.
- **`observations_p1.md`** from Step 5 Phase 1 — visual descriptions of every plot (with inline images) and raw numerical observations from run_log.txt. Use for writing accurate "before" and "after" prose around visualizations — what patterns to point students toward, what axis values to reference. Does NOT override the narrative brief on teaching angles or framing.
- **`code/data_setup.py`** — shared data layer. Read to understand what data the week uses. The notebook imports from this file directly (see The Process § Data loading).
- **The relevant `code/` subfolder:**
  - Lecture agent → `code/lecture/`
  - Seminar agent → `code/seminar/`
  - Homework agent → `code/hw/`

## Output

Your deliverable is a **builder script**: `_build_lecture.md`, `_build_seminar.md`, or `_build_hw.md` in the week folder. The orchestrator runs it through `nb_builder.py` (project root) after you finish to produce the `.ipynb`. You never produce `.ipynb` directly.

---

## Builder Format

Your builder script is a **markdown file** with fenced code blocks. The orchestrator runs it through `nb_builder.py`, which splits on fence markers and produces the `.ipynb`.

**Your entire file looks like this:**

```
# Week NN: Title

> *"Opening epigraph."*

~~~python
import pandas as pd
import numpy as np
import yfinance as yf
~~~

Prose cell between code cells. At least 2-3 substantive sentences
that interpret, bridge, or create tension.

~~~python
# verbatim from CELL block
smb = small_port - big_port
hml = value_port - growth_port
~~~
```

**Rules:**
- Everything outside `~~~python` / `~~~` fences is a **markdown cell**.
- Everything inside `~~~python` / `~~~` fences is a **code cell**.
- No Python wrapper, no imports, no build logic. Just markdown and fenced code blocks, top to bottom.
- LaTeX works natively: `$$\sigma^2$$`, `$\beta$` — no escaping needed.
- Code is plain Python — docstrings, triple quotes, backslashes all work normally. No delimiter conflicts.
- Everything else in this guide (voice, prose quality, code extraction, quality bar) applies identically — the builder format changes only the delivery mechanism, not the content standards.

**Cell structure rules (sizing, splitting, ratios) are in `cell_design.md`.** That guide is the single source of truth for cell structure — this guide covers narrative quality, voice, and teaching standards.

---

## Authority Rules

Four sources govern different aspects of your notebook:

| Source | Authoritative on |
|--------|-----------------|
| **Blueprint** | Structure, narrative arc, hooks, bridges, career connections, exercise framing |
| **Narrative brief** | Results, teaching angles, key numbers, how to frame divergences |
| **Observations (Phase 1)** | What plots look like (visual patterns, axis values), raw numerical results |
| **Code** | All computation — used verbatim, never modified |

**When blueprint and brief conflict on data-dependent content, the brief wins.** The blueprint was written before code ran; the brief incorporates what actually happened. The blueprint's arc, section ordering, and creative framing remain authoritative — but if the blueprint says "students will see a strong correlation" and the brief says "correlation is near zero due to universe bias," follow the brief's teaching angle.

---

## The Process

### All notebook types

1. **Data loading.** The notebook's setup cell imports from `data_setup.py` using the same imports as the code files: `sys.path.insert(0, "code")` followed by `from data_setup import ...`. This reuses the tested data layer rather than reimplementing fragile inline downloads. If a data loading function does something non-obvious (filtering, date alignment, caching), explain what it returns in a brief prose cell after the setup — pack the pedagogy into prose, not into reimplemented code.

2. **Code extraction.** Read each code file in your subfolder, in order. For each `# ── CELL:` block:
   - Write a **markdown cell before** — use the narrative brief's teaching angle + blueprint's hooks
   - Write a **code cell** — the code between two consecutive CELL markers, **verbatim**
   - Write a **markdown cell after** — interpretation using the brief's key numbers and framing

3. **Skip entirely:** the `# ── CELL:` marker lines themselves, `matplotlib.use("Agg")`, and the entire `if __name__ == "__main__":` block. Keep `sys.path` and `from data_setup import` lines — they go in the notebook's setup cell. **Keep threading env vars** (`os.environ["OMP_NUM_THREADS"]`, `os.environ["MKL_NUM_THREADS"]`, `torch.set_num_threads(1)`) — these prevent a macOS OpenMP/LibTorch deadlock when LightGBM and PyTorch coexist, and notebooks need them just as much as scripts do.

4. **Add structure.** Opening hook (from blueprint), section headers, transitions between sections, closing, career connections, bridge to next week.

### Additional steps for seminar and homework agents

Before each exercise/deliverable's solution code, create:
- Exercise/deliverable header with question or mission framing (from blueprint)
- Student workspace: empty code cells with `#` guiding comments (2-4 cells for seminar, 2-5 for homework)
- Solution separator (see Formatting Standards)

Then extract the corresponding code file's CELL blocks as the solution, with prose cells as usual.

---

## Code → Notebook: A Worked Example

**Source:** `code/lecture/s3_factor_construction.py` (abbreviated)

```python
"""Section 3: Factor Construction"""
import matplotlib; matplotlib.use("Agg")
import sys; sys.path.insert(0, ...)
from data_setup import load_factor_data, CACHE_DIR

factors, kf_factors = load_factor_data()

# ── CELL: build_factors ──────────────────────────────────

smb = small_port - big_port
hml = value_port - growth_port
factors = pd.DataFrame({"SMB": smb, "HML": hml})

# ── CELL: compare_to_kf ─────────────────────────────────

smb_corr = factors["SMB"].corr(kf_factors["SMB"])
print(f"SMB correlation with Ken French: {smb_corr:.4f}")

if __name__ == "__main__":
    assert smb_corr < 0.3, f"SMB-KF corr {smb_corr:.4f}"
    print(f"✓ s3_factor_construction: ALL PASSED")
```

**Becomes in `_build_lecture.md`:**

```
## Section 3: Factor Construction

The CAPM says one factor explains everything. Fischer Black didn't buy it —
and neither should you. Let's build the factors that Fama and French argued
were missing...

~~~python
smb = small_port - big_port
hml = value_port - growth_port
factors = pd.DataFrame({"SMB": smb, "HML": hml})
~~~

We now have our own hand-built SMB and HML. But how close are they to the
"official" Ken French factors that every academic paper references?
The answer might surprise you — and it reveals something important about
why sandbox exercises are never the full story.

~~~python
smb_corr = factors["SMB"].corr(kf_factors["SMB"])
print(f"SMB correlation with Ken French: {smb_corr:.4f}")
~~~

A correlation of 0.04. Our hand-built SMB barely resembles the published
version. This isn't a bug — it's a lesson in data infrastructure...
```

**Skipped:** `matplotlib.use("Agg")`, all CELL marker lines, entire `if __name__` block. The `sys.path` and `from data_setup import` lines go in the notebook's setup cell.

**Added:** Prose cells drawing on the brief's teaching angle (category: Data limitation, key numbers: ours = 0.04, production = 0.75+) and the blueprint's section hook.

---

## The Voice

> *A sharp, opinionated practitioner who respects your intelligence but not your existing knowledge of finance — someone who tells you stories, shows you code, and never lets you forget that real money is on the line.*

### The Blend We're Emulating

Our narrative voice is a deliberate mix of four influences:

**Nassim Taleb — the attitude.** Opinionated, memorable, anchors everything to real catastrophes. Makes you *feel* why fat tails matter. Intellectually honest about what we don't know. Delivers one-liners that stick:

> *"LTCM had two Nobel laureates, $125 billion in assets, and models that said a loss this big wouldn't happen in the lifetime of the universe. It happened in 4 months."*

We borrow: the conviction, the real-world grounding, the refusal to hand-wave.
We leave behind: the tangents, the combativeness, the 3-page digressions about Levantine merchants.

**Michael Lewis — the storytelling.** The best in the business at making finance a page-turner. Characters, tension, "holy shit" moments. *The Big Short*, *Flash Boys*, *Liar's Poker* — he turns abstract mechanisms into human drama:

> *"When Knight Capital deployed buggy code in 2012, they lost $440 million in 45 minutes. That's $10 million per minute. The bug? It reactivated old test code that bought high and sold low, over and over."*

We borrow: the storytelling instinct, the human stakes, the concrete examples.
We leave behind: the pure-journalist approach (we need math and code too).

**Jeremy Howard (fast.ai) — the pedagogy.** The gold standard for technical education. Top-down, code-first, "let me show you before I explain." Never more than 2 minutes between idea and code. Builds complexity gradually. Respects the student's time.

We borrow: show-then-explain structure, gradual complexity, no unnecessary abstraction.
We leave behind: nothing — his pedagogical instincts are nearly flawless.

**Our own addition — the ML engineer's bridge.** The audience is ML people learning finance, not the other way around. We constantly bridge to what they already know:

> *"Think of survivorship bias like training on a dataset where all the failed examples have been removed. You'd get great training accuracy and terrible real-world performance. That's exactly what happens when you backtest on Yahoo Finance data."*

### Who We Are NOT Emulating

**Lopez de Prado.** Brilliant methodologist, terrible narrator. Dense, academic, assumes PhD-level math. Writes to impress peers, not to teach students. His *ideas* are central to this course; his *prose style* is not.

---

## The 10 Rules

### 1. Lead with a Story, Not a Definition

Every major concept gets introduced through an analogy, a historical event, or a "what would happen if" scenario *before* the formal version appears.

**Bad:**
> "An exchange maintains an order book — a list of buy (bid) and sell (ask) orders: Limit order: sits in the book. Market order: executes immediately."

**Good:**
> "Here's something that might surprise you: when you click 'buy' on Robinhood, your order doesn't go to 'the stock market.' It goes to Citadel Securities — a single company that handles about 25% of all US equity trades. They look at your order, decide whether to fill it themselves, and pocket a fraction of a penny for their trouble.
>
> But let's back up. To understand *why* that system exists, we need to understand the thing it replaced: the order book. Think of it as two queues standing face to face..."

### 2. Explain the "Why" Before the "What"

Don't define something and then explain why it matters. Flip it — create the *need* first, then introduce the concept as the *solution*.

**Bad:**
> "Log returns are additive over time: R = R₁ + R₂ + ... This property is useful for multi-period analysis."

**Good:**
> "Quick puzzle. You invest $100. It goes up 10% in January — great, you have $110. Then it drops 10% in February. Are you back to $100?
>
> Nope. You're at $99. You *lost* money on two moves that should cancel out. This is the compounding trap, and it's bitten more junior quants than any bug in production code.
>
> Log returns fix this. Take the log of 1.10, add the log of 0.90, and you get a small negative number — honest about the loss. They're the only return type that adds up cleanly across time. Let's prove it with code..."

### 3. Anchor Numbers to Real Life

Every time a number appears, ground it in something tangible. A naked number teaches nothing; a number with context changes how you think.

- *"The bid-ask spread for Apple is about 1 cent on a $190 share — you'd barely notice. For a micro-cap stock, it can be 50 cents on a $5 share — that's 10%. Your model needs to be right by more than 10% just to break even on a round trip. Most models aren't right by 1%."*

- *"The S&P 500 has excess kurtosis around 20. A Gaussian distribution has 0. What does that mean in practice? It means the October 19, 1987 crash — a 22% single-day drop — had a probability of roughly 10⁻¹⁶⁰ under Gaussian assumptions. That's not 'unlikely.' That's 'the universe isn't old enough for this to happen once.' And yet it did."*

- *"A strategy that turns over daily at 10 bps round-trip burns 25% per year in transaction costs alone. For reference, the average hedge fund's gross return is about 10-15%. You'd be spending twice your expected revenue on shipping costs. That's not a strategy — that's a donation to market makers."*

### 4. Use a Consistent Narrative Voice

The tone is a knowledgeable friend explaining things over coffee — not a textbook.

- **Use "we" and "you":** *"When we feed raw prices into an LSTM, we're asking it to learn that 150 and 300 are the same stock at different times. That's a lot to ask."* — not "When raw prices are fed into an LSTM, the model must learn price-level invariance."
- **Admit what's hard:** *"If fractional differentiation feels weird right now, good — it is weird. The idea that you can take the 0.4th derivative of a time series sounds like something a mathematician made up to win an argument. But it works, and we'll see why."*
- **Be opinionated:** *"GARCH(1,1) is the only GARCH variant you need. GARCH(2,1), EGARCH, GJR-GARCH — they exist, people publish papers about them, and in practice they barely beat (1,1). We'll test this ourselves in the seminar."*
- **Okay to be informal:** Contractions, short sentences, occasional dry humor. Never sloppy. *"This function returns NaN for weekends. Because, of course, markets are closed on weekends. Unless you're trading crypto, in which case nothing is ever closed, sleep is optional, and the concept of 'business days' doesn't apply."*

### 5. Drop "Did You Know?" Moments

Sprinkle real-world facts that make concepts sticky. These are the bits students remember at the bar three weeks later. Weave them into the narrative naturally — not as boxed sidebars, but as the kind of thing you'd say mid-explanation with a raised eyebrow:

- When introducing market makers: *"Citadel Securities — one company — executes about 25% of all US equity volume. They're not a hedge fund (that's Citadel LLC, different entity). They're the plumbing. Every time your Robinhood order fills, there's a good chance it went through them. They made $7.5 billion in revenue in 2022. On pennies per share."*

- When discussing survivorship bias: *"Let's play a game. Name the 10 biggest US companies from the year 2000. You'll probably remember Microsoft, GE, Walmart. You probably won't remember WorldCom (#20 by market cap, filed the largest bankruptcy in US history two years later), or Enron (#7 by revenue, ceased to exist by 2002). If your training data starts in 2005, these companies simply don't exist. Your model will learn that large-cap stocks always survive. They don't."*

- When introducing volatility clustering: *"On February 5, 2018, the VIX — Wall Street's 'fear gauge' — doubled in a single day. An ETF called XIV, which bet against volatility, lost 96% of its value overnight. It had $1.9 billion in assets that morning. It was liquidated within the week. The people who bought it thought volatility was low and would stay low. Volatility clustering says: the market remembers its shocks. High vol begets high vol."*

### 6. Build Code Gradually (No "Download Walls")

Never dump a 40-line function and then explain it. The reader's eyes will glaze over at line 15 and they'll scroll past the whole thing. Instead, build it in front of them.

**Bad:** A single cell with a complete `make_dollar_bars()` function (40 lines of accumulators, edge-case handling, and index management) followed by "Let's see the output."

**Good — a three-cell sequence:**

> *"Dollar bars are supposed to sample one bar every time a fixed dollar amount trades. Let's see if we can build this in 5 lines:"*
>
> ```python
> dollar_vol = spy['Close'] * spy['Volume']
> cum_dollars = dollar_vol.cumsum()
> bar_ids = (cum_dollars // threshold).astype(int)
> dollar_bars = spy.groupby(bar_ids).agg({'Open': 'first', 'High': 'max', ...})
> ```
>
> *"That actually works — 5 lines. But look at the output. We're losing the proper OHLC semantics because `groupby` doesn't know which row is first vs. last. And we get one giant bar whenever volume dries up over a holiday. Let's fix both..."*
>
> [Next cell: improved version addressing those two issues]

**Note:** In this course, code cells come from verified `.py` files — you don't write new code. But the CELL markers in the code files are already designed with gradual building in mind: early cells show the simple version, later cells refine it. Your prose should narrate this progression.

### 7. Every Plot Tells a Story

**Before** every visualization — tell the reader what to look for, and what the "boring" outcome would be. This way the interesting thing *pops*:

> *"We're about to plot the distribution of daily S&P 500 returns against a Gaussian with the same mean and standard deviation. If markets were the well-behaved system that most textbooks assume, these two curves would overlap perfectly. They won't. Watch the tails."*

**After** every visualization — don't just describe what we see. Explain what it *costs* you:

> *"See those heavy tails? On March 16, 2020, the S&P dropped 12% in a single day. Under Gaussian assumptions, that's a 1-in-10²⁵ event — it shouldn't happen once in trillions of universe lifetimes. Under the actual distribution, it's maybe a 1-in-500-year event. If your risk model uses Gaussian VaR, it told you that day was impossible. Your portfolio disagreed."*

### 8. End Sections with "So What?"

After each major concept, a 1-2 sentence bridge that connects the abstract idea to something concrete — preferably something that will break your model or lose you money if you ignore it:

> *"So here's what survivorship bias does to your model in practice: it learns that 'buy the dip' always works — because every stock in its training set eventually recovered. The ones that went to zero? They were quietly removed from the dataset years ago. Your model has never seen a company die. It will be very confused when one does."*

> *"Why does non-stationarity matter for you specifically? Because if you train an LSTM on raw Apple prices from 2015 ($130) and test on 2024 ($190), the model has never seen numbers in the test range. It's extrapolating every single prediction. This is the equivalent of training an image classifier on cats and testing on dogs — except it's harder to notice."*

### 9. Progressive Disclosure of Jargon

Don't use 5 financial terms in one paragraph. Your reader is a capable ML engineer — they know what a loss function is, what cross-validation does, what a transformer is. They do NOT know what "alpha" means, what a "basis point" is, or why anyone cares about "the bid-ask spread." Introduce one term, make sure it's anchored, then build.

**Bad:**
> "We compute the IC of our alpha signal using cross-sectional Fama-MacBeth regressions on decile-sorted portfolios, controlling for the Fama-French 5-factor exposures."

**Good:**
> "We need a way to measure whether our predictions actually rank stocks correctly. In ML you'd use Spearman correlation between predicted and actual values. In finance, they call the exact same thing the **information coefficient** (IC). An IC of 0.05 sounds pathetic — but in a universe of 500 stocks, rebalanced monthly, it's enough to build a career on. We'll see why when we get to the Fundamental Law of Active Management in Week 4."

Never assume the reader knows a financial term unless it was explicitly taught in a prior week of this course. ML terms can be assumed freely.

### 10. The Rhythm of a Good Lecture Section

Each major section follows this pattern:

1. **Hook** — Why should you care? A provocation, a dollar amount, a disaster. (1-2 sentences)
   > *"Every time your model says 'buy,' someone is quietly taking money from you."*
2. **Intuition** — Analogy, story, or "imagine..." scenario. Make it visceral. (1 paragraph)
3. **Formal concept** — The actual definition, formula, or framework. Keep it tight. The reader is ready for it because they understand *why* it exists.
4. **Code + visualization** — See it working. Build the code gradually (Rule 6), make the plot tell a story (Rule 7).
5. **"So what?"** — Bridge to ML, to real money, to the next concept. (1-2 sentences)

### Formulas and LaTeX

Don't shy away from math — this is a quant course, and the math IS the content. LaTeX should appear generously wherever a formula makes the concept click. But math must never arrive cold. The pattern is: **intuition → formula → code → insight.**

**Bad — formula dropped without context:**
> $\delta_a + \delta_b = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)$
>
> This is the optimal spread in the Avellaneda-Stoikov model.

**Good — the formula is the punchline of a story:**
> *"So the market maker has two enemies: volatility (the price moves against you while you hold inventory) and illiquidity (nobody takes the other side of your trade). The optimal spread has to account for both. Avellaneda and Stoikov showed it's exactly:"*
>
> $$\delta_a + \delta_b = \underbrace{\gamma\sigma^2(T-t)}_{\text{volatility penalty}} + \underbrace{\frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)}_{\text{liquidity premium}}$$
>
> *"Read that left to right: the first term says 'widen your spread when vol is high or you have a long time left.' The second says 'widen it more when the book is thin (low κ).' That's the entire intuition — two fears, one formula. Let's implement it."*

**When formulas are dense**, break them into named pieces with `\underbrace` or build term by term:

> *"Start with mid-price S(t). Now adjust for inventory:"*
>
> $$r(t) = S(t) - q \cdot \gamma \sigma^2 (T - t)$$
>
> *"When q > 0 (long), the reservation price drops below mid — you're eager to sell. When q < 0 (short), it rises above mid. The γσ²(T-t) term controls how aggressively."*

---

## The Narrative Standard

### The No-Silent-Code Rule

**Never place two code cells back-to-back without a prose cell between them.**

This is the most important formatting rule in this document. Violations are the #1 failure mode.

A "prose cell" means a markdown cell with **at least 2-3 substantive sentences** that do one or more of:
- Interpret what the previous code showed ("Notice that kurtosis varies by a factor of 10x across sectors...")
- Create tension about what comes next ("But here's the problem nobody tells you about...")
- Bridge two concepts ("Now that we've seen the disease, let's look at the cure...")
- Provide a real-world grounding or "Did you know?" moment
- Challenge the student's assumptions ("You might expect dollar bars to always win. They don't.")

**Not** a prose cell:
- "Let's see this visually." (6 words — a caption, not commentary)
- "### Step 3: Return Computation" (a heading is not prose)
- "Now let's run the Jarque-Bera test." (a plot setup without substance)
- Any cell under 15 words

### The `print()` Prohibition

Never use `print()` to deliver narrative, teaching, or storytelling content. This pattern:

```python
print("Companies you CANNOT download from yfinance:")
print("  Enron (bankrupt 2001) -- ticker ENRNQ, delisted")
print("\n--> This is survivorship bias: the dead are invisible.")
```

is **banned**. This belongs in a markdown cell with proper formatting, emphasis, and voice. `print()` is for data output (statistics, table rows, confirmation messages), never for teaching.

### Prose Quality Minimums

Prose-to-code ratios and max consecutive code cells are defined in `cell_design.md`. The following narrative quality minimums apply on top of those structural constraints:

| Notebook | Min markdown cells with ≥3 substantive sentences |
|----------|--------------------------------------------------|
| Lecture | 40% of total cells |
| Seminar | 25% of total cells |
| Homework | 20% of total cells |

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

### The Prose-Around-Code Discipline

Each lecture section's code step (Rule 10, step 4) expands into three cells:

1. **Prose setup cell** (2-3 sentences) — "Here's what we're about to see, and what the boring outcome would be."
2. **Code cell** (1 cell, ≤15 lines) — Show it working. One concept only.
3. **Prose interpretation cell** (3-5 sentences) — "See that number? Here's what it means. Here's what it costs you."

This is where most notebooks fail. They skip the setup and skip the interpretation, leaving bare code between narrative islands. The No-Silent-Code Rule exists because of this failure mode.

---

## Using the Narrative Brief

The narrative brief provides a per-section entry with: **Blueprint intended → Expected → Actual → Category → Teaching angle → Key numbers**. Each section falls into one of four divergence categories:

| Category | What it means | How you write the prose |
|----------|--------------|------------------------|
| **Matches** | Code confirms expectations | Teach straightforwardly — brief, direct, no drama |
| **Data limitation** | Sandbox can't replicate production | "Here's what we got → here's what a fund sees → here's why the gap exists" |
| **Real phenomenon** | The result IS production reality | "Your code is correct. This IS what happens." Lean into the surprise. |
| **Expectation miscalibration** | Prediction was off, code is fine | Teach the actual result; reference the original prediction only if the gap is itself instructive |

For **non-Matches** entries, the brief includes a **"Sandbox vs. reality"** field. Weave this into your prose at the point where the student first encounters the surprising result. These moments often carry more teaching value than the "works as expected" sections — give them the most narrative weight. Use the brief's "Frame as," "Key numbers," and "Student takeaway" fields directly.

**Never reference the narrative brief, observations, or execution log by name.** Their content is woven into your prose naturally. Students should never know these documents exist.

---

## The Three Notebook Types

**Voice across types — the voice shifts register, it doesn't lower volume:**
- **Lecture — the storytelling teacher:** Full 10 Rules intensity. Builds intuition through narrative, stories, analogies. Explains the "why" before the "what."
- **Seminar — the curious lab partner:** Reacts to data with genuine surprise. Challenges assumptions with questions. Grounds every result in practice. *"Look at that — TSLA's kurtosis dropped 50% but SPY barely moved. What's going on?"*
- **Homework — the senior colleague:** Frames missions with conviction. Interprets results with professional insight. Tells you what this looks like on the job. *"The pipeline you just built handles the happy path. In production, the unhappy path is 80% of your week."*

All three have personality throughout — no dead zones. The voice never disappears; it adapts.

### 1. `lecture.ipynb` — The Conceptual Journey

**Purpose:** Teach the *why* behind every concept. Build intuition, tell stories, show carefully chosen demonstrations. The student reads/watches this — they don't "do" it.

**What it IS:**
- A narrative-driven document that reads like a great technical blog post
- A curated sequence of ideas, each building on the last, with the full personality of the 10 Rules
- Demonstration code: short, focused cells that prove a single point, always wrapped in narrative
- The place where all formulas, definitions, and theory live

**What it is NOT:**
- A workshop or lab (that's the seminar)
- A code repository with narrative bookends
- A place where students write code (they run yours)
- An exhaustive reference (link to docs/papers for depth)

**Structure** (cell counts and ratios per `cell_design.md`)**:**

```
Cell 0:   Title + epigraph (1 markdown cell)
Cell 1:   Opening Hook (from blueprint) — a story, disaster, or provocation
          (1 LONG markdown cell, 3-5 paragraphs)
Cell 2:   Setup — imports + pip install + utility functions (1 code cell, hide the plumbing)

Sections 1-N (following the blueprint's lecture outline):
  Each section follows the RHYTHM: Hook → Intuition → Formal → Code → Interpret → "So what?"
    - Section header + narrative hook (markdown, from blueprint's section hook)
    - Intuition / story / analogy (markdown, can be same cell or separate)
    - Formal concept + LaTeX (markdown)
    - Prose setup: "here's what to look for" (markdown, 2-3 sentences min)
    - DEMONSTRATION code (1 cell — verbatim from code file)
    - Prose interpretation: "here's what it means" (markdown, using brief's teaching angle + key numbers)
    - (Optional: second code cell for visualization, again with prose before and after)
    - "So what?" bridge (markdown, 2-3 sentences connecting to next concept or real money)

Closing:
  - Summary table of key formulas/concepts (markdown)
  - Career Connections (markdown, 2-3 paragraphs, from blueprint)
    → Maps this week's skills to 2-4 specific roles
    → Concrete: "A risk analyst at a multi-strategy fund runs exactly this analysis every morning"
    → Names real firms, real workflows, real job titles — not generic "useful in finance"
    → Tone: conversational, like a senior colleague describing the job
  - Bridge to next week (from blueprint, 1-2 paragraphs — create anticipation)
  - Suggested reading with "why you'd read this" annotations (markdown)
```

**Key rules for lectures:**

1. **NEVER two code cells in a row.** Every code cell gets a prose cell before AND after it. The prose before says what to look for. The prose after says what it means.

2. **Demonstration code, not exercise code.** Show the output of `make_dollar_bars()` and its effect on kurtosis. Don't have the student build `make_dollar_bars()` — that's the seminar's job.

3. **One concept per code cell.** Each cell proves exactly one thing. Cell sizing rules are in `cell_design.md`.

4. **Hide the plumbing.** Put a thin utility cell early on for common boilerplate (MultiIndex handling, plot defaults), then never repeat it.

5. **Show the catastrophe.** Before showing the fix, show what goes wrong. Plot raw prices into an LSTM's input range. Show the 95% "crash" from an unadjusted split. Let them *see* the problem before you solve it.

6. **The lecture should NOT build reusable classes.** It can sketch pseudo-code or show a 10-line prototype to motivate the homework, but the full DataLoader build belongs in the homework.

7. **Pace the formulas.** Maximum 2 LaTeX formulas per section. Follow the intuition → formula → code → insight pattern from the Formulas and LaTeX section.

8. **Every plot follows Rule 7** — a prose cell before (what to look for) and a prose cell after (what it means). This is a specific case of the No-Silent-Code Rule.

9. **"Did you know?" moments** — at least 3-4 per lecture, woven into the narrative flow, never boxed as sidebars.

10. **The voice never disappears.** The 10 Rules voice must be present in every prose cell, not just section headers. If a section reads like a textbook, rewrite it.

11. **Career relevance is woven in, not bolted on.** The closing has a dedicated Career Connections section (see structure above), but career implications should also appear naturally in "so what?" moments throughout. When a concept maps directly to a job function, say so: "This is literally what a junior quant at Citadel does on day one." Don't save all career content for the closing.

**Humor:** Dry, earned, and never punching down. The humor comes from the absurdity of finance itself — you don't need to add jokes when the VIX just doubled and erased $2 billion. *"The efficient market hypothesis says this can't happen. The market says hold my beer."* Never forced, never a pun, never an emoji.

**Length:** Lecture notebooks should be substantial — students are paying (in time) for depth. But every paragraph must earn its seat. If it doesn't teach, motivate, or connect, cut it. A 60-cell lecture that flows is better than a 40-cell lecture that plods.

**References:** Never a bare citation. Always "why you'd read this": *"Gu, Kelly & Xiu (2020) — the paper that ended the debate about whether ML works for stock prediction. Tested 8 models on 30,000 stocks over 60 years. Yes, neural nets win. No, not by as much as you'd think. Still the benchmark everyone cites."*

**Cell budget:** See `cell_design.md` for cell count targets, prose ratios, and consecutive code limits. As an example, a 60-cell lecture should have ~38 markdown cells (63%), of which at least 24 have ≥3 substantive sentences.


### 2. `seminar.ipynb` — The Hands-On Lab

**Purpose:** Students practice *doing* things the lecture only demonstrated. Exercises push beyond the lecture — new data, new situations, new scales. Students make mistakes and learn from the surprise.

**What it IS:**
- Guided exercises where students write real code to answer specific questions
- Applications of lecture concepts to situations NOT covered in the lecture
- A place where students encounter surprising results and learn from them
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
| Dollar bars built? | Shown: 5-line napkin version | Exercise: build from scratch, compare multiple assets |
| Code? | Students run yours | Students write their own |
| Voice? | Full narrative | Leaner, but still present — every exercise has personality |

**Structure** (cell counts and ratios per `cell_design.md`)**:**

```
Cell 0:   Opening — a testable claim or provocative question the exercises will answer
          (markdown, 1-2 paragraphs with personality, not just a table of contents)
Cell 1:   Shared imports + data loading from data_setup.py (ALL data for ALL exercises, loaded ONCE)

Exercises (3-5 per seminar):
  Each exercise:
    - Exercise header + "The question we're answering:" (markdown, 2-3 paragraphs from blueprint)
      → Must frame a testable claim or challenge an assumption, not just assign a task
      → Include stakes: what's wrong if the answer is X? What changes if it's Y?
    - Task list: specific, numbered, unambiguous (markdown, can be in same cell)
    - Student workspace: empty code cells with # guiding comments (2-4 cells)
    - ─── Solution ─── separator (markdown)
    - Solution code cell 1 (verbatim from code file, ≤15 lines)
    - Interpretation prose (markdown, 3-5 sentences, using brief's teaching angle)
      → At least one sentence grounding the result in practice
      → React to the data: surprise, confirmation, or contradiction
    - Solution code cell 2 (verbatim, ≤15 lines)
    - Interpretation prose (markdown, 3-5 sentences)
    - (Optional: cell 3 + interpretation)
    - Insight paragraph (markdown, 4-8 sentences — see Aha Moment Quality Bar)
      → Must be DIFFERENT from the lecture — something new the data reveals
      → Must reference specific numbers from the output
      → Must connect to a real-world consequence ("so what does this cost you?")
    - Transition to next exercise (markdown, 2-3 sentences — connect what you just
      discovered to the question the next exercise asks, or deepen the mystery)

Closing:
  - "What you just proved" reflection (markdown, 2-3 paragraphs)
    → Synthesize across exercises — what story do the results tell together?
    → Contrast with the lecture's claims: "The lecture said X. You just showed it's more nuanced."
    → Reference specific numbers from the exercises
  - Career connection (markdown, 1 paragraph)
    → Brief but specific: "The analysis you just ran is what a [role] at [firm] does every [timeframe]."
  - Bridge to homework (markdown, 2-3 sentences — create anticipation)
    → Not just "the homework does more of this" — tease what the scale change reveals
```

**Key rules for seminars:**

1. **Never repeat a lecture exercise.** If the lecture showed a QQ-plot of SPY, the seminar should NOT ask the student to make a QQ-plot of SPY. It could ask them to compare QQ-plots across 10 stocks and discover that kurtosis varies by sector — that's a new insight.

2. **Exercises should have a question, not just a task.** Bad: "Build volume bars." Good: "Does the dollar-bar advantage hold for volatile stocks like TSLA the same way it holds for SPY? Build bars for both and compare. Is Lopez de Prado's claim universal, or does it depend on the stock?"

3. **The solution insight must surprise and reference the data.** Bad: "Fat tails exist." Good: "TSLA's kurtosis dropped from 18.4 to 9.2 with dollar bars — a 50% reduction. SPY only dropped from 11.3 to 10.1. The dollar-bar advantage is proportional to how much volume variation the stock has."

4. **Prose between every solution code cell.** Even in solutions, no two code cells back-to-back without a prose cell. Interpretation prose should be 3-5 sentences — react to the data, ground it in practice, then set up the next cell. The lab-partner voice lives here: curiosity about results, not mechanical description.

5. **The voice never thins out.** The lab-partner voice is present in every prose cell — reacting to data, challenging assumptions, connecting to practice. If an interpretation cell reads like "The plot shows X. Now let's compute Y." — that's dead prose. Add a reaction: why is X surprising? What would it cost you? How does it compare to what the lecture predicted?

6. **"Did you know?" moments — at least 1-2 per seminar.** Woven into exercise interpretations where they deepen the insight. Not boxed sidebars — the kind of fact that makes the student look up from the notebook.

7. **One data load for the whole seminar.** Don't re-load data in every exercise. Load everything needed via data_setup.py at the top.

8. **Transition between exercises.** Don't let exercises start cold. A 2-3 sentence transition connects what you just discovered to the next question — builds narrative momentum across the seminar.

9. **Build toward the homework.** The last exercise should naturally lead into what the homework asks. End with something like: "You've now seen this for 2 stocks. In the homework, you'll scale this to 200 — and discover that the patterns get more interesting at scale."


### 3. `hw.ipynb` — The Challenge

**Purpose:** An extended, integrative project that pushes students to apply, combine, and extend what they learned. Students build real things — classes, pipelines, reports. Substantially harder than the seminar.

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

**Structure** (cell counts and ratios per `cell_design.md`)**:**

```
Cell 0:   Title (markdown)
Cell 1:   Mission framing — why this matters, what you'll build, why it's not busywork
          (markdown, 3-4 paragraphs with the full narrative voice)
Cell 2:   Deliverables list — numbered, specific, unambiguous (markdown)
Cell 3:   Imports (code)

For each deliverable:
  - Deliverable header + instructions (markdown, 2-3 paragraphs from blueprint)
    → Instructions are clear but NOT pseudo-code. Direction, not dictation.
    → Include a "why this matters on the job" paragraph — ground the deliverable in practice
    → Include a teaser: "Step 3 is where it gets interesting — you'll find that..."
  - Student workspace: empty cells or scaffolded code with # TODOs (2-5 cells)
  - ━━━ SOLUTION ━━━ separator (markdown, visually distinct from seminar)
  - Solution code cell 1 (verbatim from code file, ≤25 lines)
  - Interpretation prose (markdown, 3-5 sentences, using brief's teaching angle)
    → At least one sentence connecting to professional practice
    → The senior-colleague voice: what does this design choice look like at scale?
  - Solution code cell 2 (verbatim, ≤25 lines)
  - Interpretation prose (markdown, 3-5 sentences)
  - (Continue pattern for remaining solution cells)
  - "Aha moment" paragraph (markdown, 4-8 sentences — see Aha Moment Quality Bar)
    → Must provide insight BEYOND what lecture + seminar covered
    → Must reference specific numbers or patterns from the output
    → Should connect to what professionals care about
  - Transition to next deliverable (markdown, 2-3 sentences — connect what you just
    built to what the next deliverable needs, or deepen the challenge)

Closing:
  - "What you built" reflection (markdown, 2-3 paragraphs)
    → What does your pipeline/class actually do? Why would someone pay for this?
    → The surprising findings: reference specific numbers from the deliverables
    → What's still missing? (honest about limitations — this is where intellectual
      honesty about our sandbox appears most naturally)
  - Career connection (markdown, 1-2 paragraphs)
    → Map the deliverable to a real job: "The FactorBuilder you wrote is a simplified
      version of what Barra sells for $X/year. A risk analyst at a multi-strategy
      fund runs a production version of this every morning."
    → Name real firms, real roles, real workflows
  - Bridge to next week (markdown, 1-2 sentences — create anticipation)
    → The homework is the last thing before the next lecture; this creates continuity
```

**Key rules for homework:**

1. **The outline is a scaffold, not a tutorial.** Give clear deliverables with enough direction to start, but don't write pseudo-code for the solution. Bad: "Use `.cumsum()` and `// threshold` to group bars." Good: "Build dollar bars using any approach. Test with at least 2 threshold values."

2. **Solution code follows the same prose rules as lectures.** No two code cells back-to-back. Every code cell gets interpretation prose after it. The senior-colleague voice lives in interpretation prose: "here's what this design choice means at scale" and "here's what surprised me."

3. **Break large code into digestible pieces.** A 200-line DataLoader class is NEVER one cell. Break it into logical sections (downloading, cleaning, returns, bars, quality) with prose between each section explaining the design choice.

4. **The "aha moments" in solutions must pass the Aha Moment Quality Bar.** They should reveal something the student couldn't know until they ran the full homework.

5. **The mission framing matters.** Frame as a mission, not a government form.

6. **Don't pre-write the "discovery" text.** Data quality reports should be templates with clear headings that students fill in based on what they find — not pre-written essays.

7. **Solution code should be production-quality.** Docstrings, clean structure, thoughtful variable names. This is the code students will reference for the rest of the course.

8. **Class definitions: break into logical sections.** The `ClassName.method = method` monkey-patching pattern is the standard way to build classes incrementally across cells with prose between each piece. Per `cell_design.md`, this applies to all notebook types — lectures, seminars, and homework.

9. **At least one "in production, this is where it breaks" moment per deliverable.** The homework is where students first build at production-like scale. Point out where the happy-path code would fail in a real fund — missing data, corporate actions, universe changes, look-ahead bias. This is the senior-colleague perspective that separates coursework from real work.

10. **Transition between deliverables.** Don't let deliverables start cold. A 2-3 sentence transition connects what you just built to what the next deliverable needs — "You now have the factors. But factors alone don't predict returns — you need to combine them with price-based signals. That's Deliverable 2."

**Mission framing example:**

**Bad:**
> "Homework: Fractional Differentiation Study. 1. For a universe of 50 US stocks, find the optimal d* per stock."

**Good:**
> "Your mission: prove that fractional differentiation actually works. The lecture made a bold claim — that there's a sweet spot between raw prices (too much memory, non-stationary) and returns (stationary but amnesia). You're going to test that claim on 50 real stocks and see if it holds up or if Lopez de Prado was having us on."

Each step should still be clear and unambiguous — the playfulness lives in the framing, not in vague instructions. And toss in a teaser: *"Step 5 is where it gets interesting — you'll see that the 'optimal' d varies wildly across stocks. Defensive utilities need barely any differencing. Meme stocks are another story entirely."*

**Solution insight example:**

> *"Look at the Sharpe ratio for the uncertainty-filtered strategy (B) vs. trading everything (A). The improvement isn't huge — maybe 0.2 Sharpe points. But look at the max drawdown. The filtered strategy dodged the COVID crash almost entirely, because model uncertainty spiked in late February 2020 and the filter pulled you out before March hit. That's not a backtest artifact — that's the model saying 'I have no idea what's happening' and your system listening."*


### How the Three Notebooks Connect

```
LECTURE: "Here's WHY fat tails matter" → [shows SPY histogram, tells 1987 story]
    ↓
SEMINAR: "HOW do fat tails vary across assets?" → [student compares 10 stocks, finds sector patterns]
    ↓
HOMEWORK: "BUILD a system that detects and handles fat tails at scale" → [200-stock pipeline]
```

The progression is: **UNDERSTAND → APPLY → BUILD**

A concept gets exercised in **ONE** notebook. The lecture SHOWS it; the seminar has students MEASURE it on new data; the homework INTEGRATES it at scale. No notebook re-does what another already covered. When writing prose, check the concept matrix in `blueprint.md` — if a concept appears in two notebooks' exercises, one must be cut.

---

## The "Aha Moment" Quality Bar

Every insight in a seminar exercise or homework deliverable must pass these four tests:

1. **Is it NEW?** Does it say something the lecture didn't already say? If the lecture proved fat tails exist, the seminar insight can't just say "fat tails exist." It must say something like "fat tail severity varies by sector, and the variation is predictable."

2. **Is it GROUNDED?** Does it reference specific numbers from the output? Bad: "Kurtosis varies across stocks." Good: "DUK's kurtosis is 4.2. TSLA's is 31.7. That's a 7.5x difference — and they're both in the S&P 500."

3. **Does it have CONSEQUENCES?** Does it connect to something that costs money, breaks a model, or changes how you'd build a system? Bad: "The survivorship bias premium is positive." Good: "That 3.8% per year compounds to a 68% overstatement over our 14-year backtest. If your model shows 15% annualized, 3.8 of those points might be ghosts."

4. **Would a student REMEMBER it at the bar?** The best insights are the ones students retell to friends. "Did you know that TSLA's kurtosis drops by 50% when you switch to dollar bars, but SPY's only drops 10%?" is memorable. "Dollar bars produce more Gaussian returns" is not.

---

## Formatting Standards

### Imports and Setup

- ONE imports cell per notebook, placed immediately after the intro
- Include `pip install` in the same cell (or the cell just before)
- Set matplotlib defaults in the imports cell, never repeat
- Include utility functions as needed for data quirks (e.g., yfinance MultiIndex handling, API auth wrappers) — keep these minimal and invisible
- **Notebooks import from `data_setup.py`** using the same pattern as code files. The setup cell includes `sys.path.insert(0, "code")` and `from data_setup import ...`. This reuses the tested data layer. If a function does something non-obvious, explain it in prose — don't reimplement it inline.

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

Code cell sizing, splitting strategies, and content rules are defined in `cell_design.md`. That guide is the single source of truth for cell structure.

---

## Anti-Patterns

1. **Don't place two code cells back-to-back.** The most common failure mode. Every code cell gets a prose cell before AND after. (See the No-Silent-Code Rule.)

2. **Don't use `print()` for storytelling.** `print("This is survivorship bias: the dead are invisible.")` is banned. Put it in a markdown cell with formatting, emphasis, and voice. (See the print() Prohibition.)

3. **Don't write caption-only transitions.** "Let's see this visually" is not a prose cell. Replace with 2-3 sentences that interpret, bridge, or create tension.

4. **Don't copy the blueprint verbatim.** The blueprint is a spec. The notebook is the realization. Use the blueprint's ideas and hooks, but write the actual narrative fresh.

5. **Don't reference upstream documents by name.** Students should never see "as the narrative brief notes..." or "per the execution log..." Weave upstream content into your prose naturally.

6. **Don't recycle insights across notebooks.** If the lecture proved fat tails exist, the seminar insight can't just say "fat tails exist." Every insight must be new, grounded, and connected to consequences.

7. **Don't repeat exercises across notebooks.** If the lecture demonstrates SPY's QQ-plot, the seminar and homework must NOT reproduce it.

8. **Don't let the voice disappear.** If a section reads like a textbook between the hook and the "So what?", the middle needs rewriting.

9. **Don't pre-write the student's discoveries.** Homework data quality reports should be templates students fill in, not pre-written essays that happen to match the data exactly.

10. **Don't invent new code.** All computation comes from verified code files. You write prose and structure, not Python.

11. **Don't treat section headers as prose.** "### Step 3: Return Computation" is a heading. It needs at least a paragraph of substantive text after it.

---

## Boundaries

| This step does | This step does NOT do |
|---|---|
| Write prose, layout, and narrative | Invent new computation or modify code |
| Extract code verbatim from CELL blocks | Include verification blocks or CELL markers |
| Follow the narrative brief for teaching angles | Decide how to frame divergences (Step 6 decided) |
| Follow the blueprint for structure and arc | Modify the blueprint or brief |
| Import from `data_setup.py` for data loading | Reimplement data downloads inline |
| Produce `_build_<type>.md` builder scripts | Produce `.ipynb` directly (orchestrator builds it) |
| Finish when the builder script is written | Run `nb_builder.py` or verify the builder script |

---

## Quality Bar

### Structure
- [ ] Code extracted verbatim from CELL blocks — no modifications
- [ ] No `if __name__` blocks, CELL markers, `matplotlib.use("Agg")`, or `sys.path` imports in notebooks
- [ ] No exercise repetition across the three notebooks (check concept matrix)
- [ ] Lecture has NO exercise prompts (demonstrations only)
- [ ] Each notebook has exactly ONE imports/setup cell
- [ ] Data loaded via `data_setup.py` imports — not reimplemented inline
- [ ] No MultiIndex handling code appears more than once per notebook

### Narrative Density
- [ ] Structural cell constraints met (see `cell_design.md` quality bar: prose ratios, consecutive code limits)
- [ ] Every markdown cell between code cells has ≥2-3 sentences of substantive prose
- [ ] No `print()` statements delivering narrative content — only data output
- [ ] No caption-only transitions ("Let's visualize this" without interpretation is not enough)

### Storytelling & Voice
- [ ] The voice never disappears in any notebook — lecture (storytelling teacher), seminar (curious lab partner), homework (senior colleague)
- [ ] At least 3-4 "Did you know?" moments in the lecture, woven naturally
- [ ] At least 1-2 "Did you know?" moments in the seminar, woven into exercise interpretations
- [ ] Every seminar exercise setup has a testable claim or provocative question — not just a task list
- [ ] Every seminar insight says something the lecture didn't — with specific numbers
- [ ] Seminar interpretation prose reacts to data (surprise, curiosity, challenge) — not mechanical description
- [ ] Every homework "aha moment" says something the seminar didn't — with specific numbers
- [ ] At least one "in production, this is where it breaks" moment per homework deliverable
- [ ] The mission framing reads like a mission, not a government form
- [ ] At least one "failure mode" is demonstrated (in lecture or seminar)
- [ ] Transitions exist between exercises (seminar) and between deliverables (homework) — no cold starts

### Narrative Brief Integration
- [ ] Teaching angles from the brief are woven into prose at the right moments
- [ ] Key numbers from the brief are cited — not approximate guesses
- [ ] Non-Matches entries get "here's what we got → here's what a fund sees → here's why" treatment
- [ ] No upstream documents referenced by name (brief, observations, execution log)

### Code Quality
- [ ] Cell sizing per `cell_design.md` (target ≤15, limit 25, visualizations uncapped)
- [ ] Data loaded once per notebook via `data_setup.py`, not per exercise
- [ ] Solution code is production quality (docstrings, clean structure)
- [ ] Lecture closing bridges to next week; homework closing bridges to next week; seminar closing bridges to homework

### Closings
- [ ] Seminar closing has "What you just proved" reflection (2-3 paragraphs with specific numbers), career connection, and bridge to homework
- [ ] Homework closing has "What you built" reflection (2-3 paragraphs with specific numbers), career connection, and bridge to next week
- [ ] Closings reference specific numbers from the exercises/deliverables — not generic summaries

### Career Relevance
- [ ] The lecture closing has a Career Connections section mapping skills to 2-4 specific roles
- [ ] Career relevance appears at least 2-3 times in the lecture body (woven into "so what?" moments, not just the closing)
- [ ] Seminar closing has a career connection paragraph (brief but specific — names a role and firm)
- [ ] Homework closing has a career connection paragraph (maps deliverable to a real job function)
- [ ] Career connections appear naturally in exercise/deliverable interpretations (at least 1-2 per notebook)
