# Code Verification Guide

> **The core principle:** Every line of code in a notebook must run correctly, produce expected output, and satisfy explicit acceptance criteria BEFORE the notebook is written. Notebook agents handle pedagogy — prose, layout, voice. The code is already settled.

This guide governs Phase 3 of the week build pipeline: implementing and verifying all notebook code. Voice, prose, and formatting rules live in `writing_guidelines.md` and `notebook_creation_guide.md` — this doc covers **what the code does, how it's verified, and how notebook agents consume it**.

---

## Why This Phase Exists

Writing code inside notebook cells is the wrong abstraction for engineering. An agent creating a notebook simultaneously handles two unrelated concerns: making code work (engineering) and making prose teach (pedagogy). When the code breaks — wrong data, silent API behavior, edge cases — diagnosing and fixing it inside `.ipynb` cells is slow, fragile, and error-prone.

Code verification separates these concerns. The code is written, run, and validated as standalone Python scripts. Only after every script passes its acceptance criteria does the notebook agent see it. The notebook agent's job narrows to: split verified code into cells, write prose around it, apply voice and formatting. It never invents or modifies code logic.

---

## Where It Fits

```
Phase 1: RESEARCH → ⏸ → Phase 2: README → ⏸ → Phase 3: CODE → ⏸ → Phase 4: NOTEBOOKS
         (agent)                 (agent)         (one agent)           (3 parallel agents)
```

Phase 3 runs after the README is approved. It produces the `code/` directory. Phase 4 (notebook writing) cannot start until all code files pass verification and the user approves.

---

## Directory Structure

```
weekNN_topic/
├── README.md
├── code/
│   ├── data_setup.py              # Shared data downloads + caching
│   ├── .cache/                    # Downloaded data (gitignored)
│   ├── lecture/
│   │   ├── s1_topic_name.py       # One file per lecture section
│   │   ├── s2_topic_name.py
│   │   └── ...
│   ├── seminar/
│   │   ├── ex1_exercise_name.py   # One file per seminar exercise
│   │   └── ...
│   └── hw/
│       ├── d1_deliverable_name.py # One file per homework deliverable
│       └── ...
├── lecture.ipynb                   # Written in Phase 4, using verified code
├── seminar.ipynb
└── hw.ipynb
```

**Naming convention:** Files are prefixed with their position (`s1_`, `s2_`, `ex1_`, `d1_`) and use snake_case topic names matching the README section titles.

**The `code/` directory persists.** It is committed alongside the notebooks as the verified reference implementation. If notebooks need regeneration, the code is already there. The `.cache/` subdirectory is gitignored — data is large and reproducible.

---

## The Shared Data Layer

`data_setup.py` handles all data downloads for the week. It runs once, caches data locally, and provides a clean interface for section files (the `.py` code files in `lecture/`, `seminar/`, and `hw/`).

**Important:** `data_setup.py` is for Phase 3 code files only — notebooks never import from it. Notebooks are standalone documents that inline their own download logic. The notebook agent reads `data_setup.py` to understand what data to download, then writes equivalent code directly in the notebook's setup cell.

```python
"""
Week N Data Setup
Downloads and caches all datasets needed by lecture, seminar, and homework code.
Run this file first: python data_setup.py
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Week-wide parameters ──────────────────────────────────
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
START = "2012-01-01"
END = "2025-01-01"

def load_or_download():
    """Download OHLCV data if not cached, otherwise load from Parquet."""
    cache_file = CACHE_DIR / "ohlcv_raw.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=False)
    raw.to_parquet(cache_file)
    return raw

if __name__ == "__main__":
    df = load_or_download()
    print(f"✓ Data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
```

**Rules for `data_setup.py`:**
1. All data downloads happen here — section files never call `yf.download()` or equivalent.
2. Cache to `.cache/` as Parquet. Avoid re-downloading on every run.
3. Provide `load_or_download()` (or similar functions) that section files call.
4. Define all shared parameters (tickers, date ranges, universes) as module-level constants.
5. May provide multiple download functions if the week uses distinct datasets (e.g., `load_equity_data()` and `load_treasury_data()`).

### Data Size: User-Gated, Biased Toward More

**The default instinct should be: more data is better, even if downloads are slow.** Weak statistical results often stem from too-small universes or too-short date ranges, not from methodological errors. A plot showing "no significance" because of 50 stocks is a pedagogical failure — the student concludes the concept is broken when the real problem is sample size.

But the right data size varies dramatically across weeks. Factor models need 500 stocks; derivatives pricing needs 5 underlyings with rich option chains. The agent cannot hardcode defaults.

**Before writing `data_setup.py`, the agent must propose a data plan to the user:**

```
DATA PLAN — Week N

Universe:
  Option A: [size] — [trade-off: speed, coverage, statistical power]
  Option B: [size] — [trade-off]
  Option C: [size] — [trade-off]
  Recommendation: [which option and why]

Date range:
  Option A: [range] — [trade-off: recency vs. statistical depth]
  Option B: [range] — [trade-off]
  Recommendation: [which option and why]

Frequency: [daily / monthly / tick — determined by the week's content]

Estimated download time: [rough estimate based on API constraints]
Statistical implications: [what the chosen size enables or limits —
  e.g., "200 stocks × 15 years gives ~36,000 stock-months,
  sufficient for cross-sectional significance at p < 0.05"]
```

**The user approves the data plan before the agent writes `data_setup.py`.** This prevents both under-scoping (too few stocks, weak results) and over-scoping (3000 stocks, 30-minute downloads that fail halfway).

**Implementation rules for large downloads:**
- Add progress indicators (`tqdm` or periodic `print()`) for downloads exceeding 2 minutes.
- Implement retry logic for API-limited sources (yfinance, FRED). Batch tickers with brief pauses between batches.
- Handle partial failures gracefully — download what succeeds, report what failed, let section code work with the available universe.
- Document the final data shape in the success message: ticker count, date range, row count, any failed downloads.

Section files import from it:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_or_download, TICKERS, START, END
```

---

## Cross-Week Dependencies

Weeks often build on prior weeks' outputs. The README may say "use Week 3's feature matrix" or "reuse your FactorBuilder from Week 3." For code verification, every week must be **fully self-contained** — no runtime imports, cache sharing, or file-path coupling across weeks.

### Rule 1: Re-download, don't reuse caches

If Week N needs the same data that a prior week downloaded (e.g., S&P 500 prices, Ken French factors), Week N's `data_setup.py` downloads it independently. Do not read from another week's `.cache/` directory. Each week's notebooks will run standalone — students cannot import from other weeks' notebooks, so the verified code must not depend on other weeks' code either.

### Rule 2: Read prior code as reference, not as dependency

Before writing `data_setup.py`, the agent should **read** the prior week(s)' `data_setup.py` and any relevant code files (e.g., a `FactorBuilder` class) listed in the README's prerequisites. This is read-only — it informs the implementation (tickers, date ranges, computation logic, data structures) but creates no runtime dependency. The agent then writes equivalent or adapted code in the new week's `data_setup.py`.

### Rule 3: Build what you need, not what the prior week built

A prior week's class or pipeline often serves a different purpose than what the new week requires. Example: Week 3's `FactorBuilder` constructs factor returns via portfolio sorts; Week 4 needs the same firm characteristics as a flat stock-month panel for ML features. The right approach: identify what the new week actually needs, read the prior implementation to understand the computation logic, then build the needed form from scratch. Don't import, copy, or subclass the prior week's class.

### The agent's prep step

The README's prerequisites section lists which prior weeks matter. Before writing any code, the agent reads:
1. The prior week(s)' `data_setup.py` — to understand tickers, date ranges, download patterns
2. Any specific prior-week code files referenced in the README — to understand data structures and computation logic

This reading step is mandatory. It prevents the agent from re-inventing data pipelines that a prior week already solved (and possibly getting different results).

---

## Code File Format

Each code file has three layers: **docstring** (acceptance criteria), **annotated code** (cell-by-cell implementation), and **assertions** (machine-checkable verification).

### Layer 1: Docstring — Acceptance Criteria

The file docstring states the acceptance criteria for this section, exercise, or deliverable. These come directly from the README and define what "correct" means.

```python
"""
Section 6: Corporate Actions — The Hidden Data Landmine

Acceptance criteria (from README):
- Reconstructed nominal AAPL price on 2014-06-06: ~$645 (pre-7:1 split)
- Reconstructed nominal AAPL price on 2014-06-09: ~$92 (post-split)
- Daily return on split date: ~-85% nominal vs ~+1.6% adjusted
- Plot shows visually dramatic discontinuity in nominal price line
- Adjustment factor plot shows clear step function at split dates
"""
```

### Layer 2: Annotated Code — Cell Boundaries and Context

The implementation is divided into cells using `# ── CELL:` markers. Each marker includes metadata that guides the notebook agent's prose.

```python
# ── CELL: cell_name ──────────────────────────────────────
# Purpose: What this code does — guides the "prose before" markdown cell.
# Takeaway: What the output reveals — guides the "prose after" markdown cell.
```

**Annotation fields:**
- **`Purpose`** (required): One or two lines describing what the code block does. The notebook agent uses this to write setup prose ("here's what to look for").
- **`Takeaway`** (use for non-plot output): What the output means. The notebook agent uses this for interpretation prose ("here's what it means"). Omit for pure setup cells that don't produce interpretable output.
- **`Visual`** (use for plot cells): What the plot shows. Describes the visual pattern the student should see. Use instead of `Takeaway` when the cell produces a plot.

**What goes between markers IS the cell.** The notebook agent takes the code between two consecutive `# ── CELL:` markers and places it into one notebook code cell verbatim. The agent does not modify, reorder, or refactor the code logic.

### Layer 3: Assertions and Plot Summaries — Machine-Checkable Verification

The `if __name__ == "__main__":` block at the bottom contains two things: **assertions** (pass/fail checks) and **plot summaries** (structured data about generated plots). Both are NEVER included in notebooks — they exist solely for verification. Notebook agents skip the entire `if __name__` block.

```python
if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    pre_split = nominal_close.loc["2014-06-06", "AAPL"]
    post_split = nominal_close.loc["2014-06-09", "AAPL"]
    assert pre_split > 600, f"Pre-split AAPL should be ~$645, got {pre_split:.2f}"
    assert post_split < 100, f"Post-split AAPL should be ~$92, got {post_split:.2f}"

    # ── PLOT SUMMARY: s6_corporate_actions ──────────────
    print(f"[PLOT] s6_corporate_actions.png")
    print(f"  type: dual-panel time series")
    print(f"  left_panel: nominal vs adjusted close for AAPL")
    print(f"  pre_split_price: {pre_split:.2f}")
    print(f"  post_split_price: {post_split:.2f}")
    print(f"  discontinuity_magnitude: {(pre_split - post_split) / pre_split:.1%}")
    print(f"  n_split_events: {n_splits}")
    print(f"  title: {ax.get_title()}")
    print(f"  xlabel: {ax.get_xlabel()}, ylabel: {ax.get_ylabel()}")

    print("✓ Section 6: All acceptance criteria passed")
```

**Assertion design rules:**
- Use **approximate ranges**, not exact values. Financial data shifts slightly over time (corporate action recalculations, data vendor corrections). Write `assert price > 600`, not `assert price == 645.57`.
- Every assertion includes a **descriptive failure message** with the actual value.
- Print a **success message** at the end so the agent confirms passage.
- Assert on **data properties that ensure plots will look correct**. You can't assert on a plot, but you can assert that "the two lines diverge by at least 50% at this date" — which guarantees the visual will be dramatic.

**Plot summary design rules** (required for every file that generates plots):
- Print one `[PLOT]` block per plot generated by the file, after assertions pass.
- Include the **saved filename** so the plot can be located for visual review.
- Include **key metrics** that the `Visual` annotation references — R², slopes, correlations, ranges, point counts, magnitudes. Only metrics relevant to the acceptance criteria or `Visual` annotation.
- Include **axis metadata** from the matplotlib axes object: `ax.get_title()`, `ax.get_xlabel()`, `ax.get_ylabel()`.
- Use matplotlib's introspection API when useful: `ax.get_lines()` for line count, `ax.get_xlim()` / `ax.get_ylim()` for axis ranges, `len(ax.patches)` for bar counts.
- Keep summaries concise — 5-12 printed lines per plot. This is verification data, not a full data dump.
- Plot summaries serve Stage 1 of verification (automated metric checks). Stage 2 (visual review of the actual PNG) happens separately — see the Verification Loop below.

---

## Code Quality Standard

All code in verified files must be production-quality:
- Clear, descriptive variable names (`nominal_close`, not `df2`)
- Docstrings on non-trivial functions
- Consistent style (Black-compatible formatting)
- No dead code, no commented-out alternatives, no TODOs
- Each cell ≤ 25 lines (prefer ≤ 15) — matching the notebook cell budget from `notebook_creation_guide.md`
- Imports organized: stdlib → third-party → local

This code will appear in notebooks essentially unchanged. Students read it. Make it worth reading.

---

## Complete Example

A full code file for a lecture section that downloads OHLCV data and computes daily returns:

```python
"""
Section 4: From Tick Data to OHLCV Bars — The Aggregation Pipeline

Acceptance criteria (from README):
- DataFrame has ≥ 2500 rows per ticker (10+ years of daily data)
- All 5 expected columns present (Open, High, Low, Close, Volume)
- Mean daily return within ±0.002 for each ticker
- Price series and return series plotted side by side
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_or_download, TICKERS, START, END

raw = load_or_download()


# ── CELL: inspect_ohlcv ─────────────────────────────────
# Purpose: Examine the raw DataFrame structure — columns, dtypes, shape.
# Takeaway: Daily OHLCV is the atomic unit of most quant research.
#   Five columns, one row per trading day. Simple — but every number
#   hides an aggregation (thousands of trades compressed into 4 prices).

close = raw["Close"]
print(f"Shape: {raw.shape}")
print(f"Date range: {raw.index[0].date()} to {raw.index[-1].date()}")
print(f"Columns: {list(raw.columns.get_level_values(0).unique())}")
close.head()


# ── CELL: compute_returns ────────────────────────────────
# Purpose: Compute simple daily returns from Close prices.
# Takeaway: Returns are roughly symmetric and mean-zero. Prices are not.
#   This is the single most important transformation in quantitative
#   finance — we almost never model prices directly.

returns = close.pct_change().dropna()
returns.describe().round(4)


# ── CELL: plot_price_vs_returns ──────────────────────────
# Purpose: Side-by-side plot of price series and return series.
# Visual: Left panel shows trending, non-stationary prices (each ticker
#   at a different scale). Right panel shows mean-zero, noisy returns.
#   The visual contrast motivates why we work with returns, not prices.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

close.plot(ax=axes[0], linewidth=0.8)
axes[0].set_title("Price Series (Close)")
axes[0].set_ylabel("Price ($)")

returns.plot(ax=axes[1], linewidth=0.3, alpha=0.7, legend=False)
axes[1].set_title("Daily Returns")
axes[1].set_ylabel("Return")

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ──────────────��──────────────────────
    for ticker in TICKERS:
        n = close[ticker].dropna().shape[0]
        assert n >= 2500, f"{ticker}: expected ≥2500 rows, got {n}"

    expected_cols = {"Open", "High", "Low", "Close", "Volume"}
    actual_cols = set(raw.columns.get_level_values(0).unique())
    assert expected_cols <= actual_cols, f"Missing columns: {expected_cols - actual_cols}"

    for ticker in TICKERS:
        mu = returns[ticker].mean()
        assert abs(mu) < 0.002, f"{ticker}: mean return {mu:.5f} outside ±0.002"

    # ── PLOT SUMMARY: s4_price_vs_returns ───────────────
    print(f"[PLOT] s4_price_vs_returns.png")
    print(f"  type: dual-panel (prices left, returns right)")
    print(f"  n_tickers: {len(TICKERS)}")
    print(f"  price_range: [{close.min().min():.2f}, {close.max().max():.2f}]")
    print(f"  return_range: [{returns.min().min():.4f}, {returns.max().max():.4f}]")
    print(f"  left_title: {axes[0].get_title()}")
    print(f"  right_title: {axes[1].get_title()}")
    print(f"  left_n_lines: {len(axes[0].get_lines())}")
    print(f"  right_n_lines: {len(axes[1].get_lines())}")

    print("✓ Section 4: All acceptance criteria passed")
```

---

## The Code Verification Agent

### What It Reads

The code verification agent receives exactly three documents:
1. `course/guides/code_verification_guide.md` — this document
2. `course/guides/task_design_guide.md` — task types and solution strategies
3. The week's `README.md` — content spec with acceptance criteria

It does NOT read writing guidelines or notebook creation guide — those govern prose, not code.

### What It Produces

The complete `code/` directory:
- `data_setup.py` with all downloads and caching
- One `.py` file per lecture section, seminar exercise, and homework deliverable
- Every file follows the three-layer format (docstring, annotated cells, assertions + plot summaries)
- Every file runs without error and passes all assertions
- `observation_report.md` — the visual plot review report (produced during Stage 2 by the user, stored here for notebook agents)
- `.cache/` — generated plot PNGs and cached data (gitignored)

### What the Agent Does NOT Do

- Write prose, markdown, or narrative
- Create `.ipynb` files
- Read other weeks' content or other guide docs
- Modify the README (unless resolving contradictions in Stage 3 with user approval)
- Add code not specified in the README outline

### The Verification Loop — 3-Stage Process

Phase 3 runs in three stages. If Stage 3 finds issues requiring code changes, the loop repeats from Stage 1.

```
┌────────────────────────────────────────────────────────────┐
│  STAGE 1: Code + Assertions + Summaries  (automated)       │
│  Agent writes all code, runs it, checks assertions and     │
│  summary output. Enriches annotations with actual numbers. │
│  → All files pass. Summary output collected.               │
└──────────────────────────┬─────────────────────────────────┘
                           │  ⏸ present summary results to user
                           ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 2: Visual Plot Review  (user, fresh context)        │
│  User opens a NEW Claude conversation with only the PNGs   │
│  and Visual annotations. Short context = accurate vision.  │
│  → Observation report (.md) produced and reviewed by user. │
└──────────────────────────┬─────────────────────────────────┘
                           │  ⏸ user approves observation report
                           ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 3: README Compliance Audit  (automated)             │
│  Compare summary output + observation report against       │
│  every README acceptance criterion. Produce contradiction  │
│  entries for unmet criteria.                               │
│  → If contradictions require code fixes → back to Stage 1. │
│  → If all resolved → Phase 3 complete.                     │
└────────────────────────────────────────────────────────────┘
```

The loop repeats until Stage 3 produces no unresolved contradictions and the user approves the final state.

---

#### Stage 1: Code + Assertions + Summaries (Automated)

The agent writes, runs, and verifies all code files. This is the core engineering stage.

```
1. Write data_setup.py and run it (establishes data cache)
2. For each notebook (lecture, seminar, hw):
   a. For each section/exercise/deliverable:
      i.   Write the code file (three-layer format)
      ii.  Run it: python code/lecture/s6_corporate_actions.py
      iii. If assertions fail → diagnose, fix, re-run (up to 3 attempts)
      iv.  If stuck after 3 attempts → flag explicitly for user
      v.   If all pass → move to next file
3. When all files pass:
   a. Collect all printed output (assertions + plot summaries)
   b. Enrich annotations with actual numbers (see below)
   c. Verify plots are saved to .cache/ (for Stage 2)
   d. Produce a Visual Annotations Summary (for Stage 2)
```

The agent does NOT present partial results. It iterates until all files pass, then presents the complete `code/` directory.

**Summary check:** After all files pass, the agent reviews the printed plot summaries to catch obvious metric-level issues — values outside expected ranges, missing axis labels, wrong line counts. This is a quick automated check using structured data, not a visual review.

**Comment enrichment:** After seeing actual output, the agent updates `Purpose`, `Takeaway`, and `Visual` annotations in code files with specific numbers from the run:
- A `Visual` that says "shows the divergence" → enrich to "AAPL starts at ~33x and drops in two clean steps at 2014-06-09 and 2020-08-31"
- A `Takeaway` that says "the error is large" → enrich to "RMSE of 0.0847, max error of 85%"
- Do NOT change what the code computes. Only change comments and annotations.
- Target 2-4 lines per annotation — specific enough for a notebook agent to write a compelling paragraph, not so long that they become the paragraph.

**Visual Annotations Summary:** At the end of Stage 1, the agent produces a text block listing every plot with its filename and `Visual` annotation. This is handed to the user for Stage 2 so they don't have to extract annotations from code files manually:

```
VISUAL ANNOTATIONS — Week N

s1_capm_scatter.png
  Visual: Cloud of ~100 points with slight positive slope but enormous
  dispersion. OLS line has R² ≈ 0.31. Demonstrates CAPM's empirical failure.

s2_ff3_cumulative.png
  Visual: Cumulative factor returns for SMB and HML. SMB shows modest
  positive trend. HML declines visibly from ~2014 onward.

[... one entry per plot ...]
```

**Gate:** Present to the user: all files pass, summary output collected, annotations enriched, plots saved. The user reviews and approves before proceeding to Stage 2.

**When the agent gets stuck:** If a file cannot pass after 3 attempts, the agent flags it explicitly: "Section 6 fails because [specific reason]. The acceptance criterion may need revision, or the approach needs rethinking." The user decides.

---

#### Stage 2: Visual Plot Review (Fresh Context, User-Gated)

**Why a separate stage:** Language models read plots accurately in short conversations but degrade significantly in long ones. By the time Stage 1 completes, the agent's context is packed with code, assertion output, and debugging history — the worst conditions for precise visual analysis. Stage 2 solves this by moving plot review to a **fresh conversation with minimal context**, where vision accuracy is highest.

**Who runs this:** The user. This is a manual step, not automated. The user opens a new Claude conversation (or a fresh Claude Code session), uploads the plot PNGs alongside the Visual Annotations Summary from Stage 1, and uses the prompt template below.

**What it produces:** An **observation report** (`observation_report.md`) — a structured `.md` file saved in the week's `code/` directory. This report is a first-class artifact that serves two purposes:
1. **Verification:** Does each plot match its `Visual` annotation and acceptance criteria?
2. **Discovery:** What does the plot reveal that wasn't anticipated? Plots can always surface something instructive that the README, code, and summaries didn't catch — unusual patterns, pedagogically rich anomalies, or visual stories that deserve narration.

**Process:**
1. Gather all `.png` files from the week's `.cache/` directory.
2. Take the Visual Annotations Summary produced at the end of Stage 1.
3. Open a **fresh** Claude conversation. Use the prompt template below.
4. Upload the PNGs alongside their annotations (batch if needed — quality degrades past ~15-20 images per conversation).
5. Review the generated report. Amend observations, add your own insights, correct any misreadings.
6. Save the final report as `code/observation_report.md`.

**Prompt template for the review session:**

```
I'm reviewing plots for Week N: [Topic] of an ML in Quantitative Finance course.

For each plot below, I'll provide: the filename, the Visual annotation from the
code (what the plot SHOULD show), and the image itself.

For each plot, please produce:

1. VERIFY — Does the plot match the Visual annotation? Note specific matches
   and mismatches. Read actual numbers from axes, titles, and legends.

2. OBSERVE — What specific patterns, values, and relationships are visible?
   Be precise: "MKT reaches $4.2 by 2024" not "MKT goes up."

3. DISCOVER — Note anything surprising or instructive that ISN'T in the
   annotation. These often become the best teaching moments in notebooks.

4. FLAG — Any readability issues: crushed axes, indistinguishable colors/lines,
   clipped labels, misleading scales, or visual emphasis that doesn't match
   the intended story.

After reviewing ALL plots, produce the observation report in this format:

## Verification Table

| Plot | Annotation Match? | Key Observations | Issues |
|------|-------------------|------------------|--------|
| [filename] | ✅ Yes / ⚠️ Partial / ❌ No | [1-2 sentence summary] | [or "None"] |

## Detailed Observations

### [filename]
[2-4 paragraphs: what you see, specific numbers, what it reveals,
 what's surprising. These paragraphs will inform notebook narration.]

## Discoveries
[Observations not anticipated by any annotation — patterns, anomalies,
 or insights that could become "aha moments" in the notebooks.]

## Issues Requiring Code Fixes
[Plots that need re-rendering. Specify exactly what to fix:
 "s3_ff5_rmw_cma.png: RMW and CMA lines are both blue,
 indistinguishable. Use distinct colors."]
```

**Gate:** The user reviews, amends, and approves the observation report before proceeding to Stage 3. If the report identifies issues requiring code fixes (under "Issues Requiring Code Fixes"), those fixes happen in the next Stage 1 iteration — not here.

---

#### Stage 3: README Compliance Audit (Automated)

Systematically compare every acceptance criterion in the README against two sources: the **printed summary output** from Stage 1 (numeric criteria) and the **observation report** from Stage 2 (visual criteria). This catches semantic contradictions that assertions alone miss — assertions verify that the code *runs correctly*, but the README defines what *correct means pedagogically*.

**Process:**

1. Re-read the week's `README.md` in full.
2. Read the observation report (`code/observation_report.md`).
3. For each section, exercise, and deliverable, walk through every acceptance criterion line by line.
4. For each criterion, determine whether it's **numeric** (checkable against summary output) or **visual** (checkable against the observation report), and verify against the appropriate source:
   - **Was it implemented?** Does the code actually test or demonstrate this?
   - **Does the output match?** Compare the README's stated expectation against the printed summary (for numeric criteria like "R² < 0.35") or the observation report (for visual criteria like "plot shows dramatic discontinuity").
   - **Is the narrative consistent?** If the README says "SMB should be positive" but the code produces negative SMB and the annotation explains why, that's a documented deviation. If the code produces negative SMB and the annotation says "SMB is positive," that's a contradiction.
5. Also check cross-cutting concerns that span multiple files:
   - **Narrative logic:** If s2 says "HML has negative mean" and s5 says "HML is significantly priced with positive lambda," is that consistent? (Yes — factor returns and risk premia are different things. No — if the lambda sign contradicts the factor return sign without explanation.)
   - **Cross-file consistency:** Do all files that compute SMB get the same sign? Do stock counts match across files that use the same universe? Does the date range used in seminar exercises align with what the lecture demonstrated?
   - **Economic plausibility:** If a factor that the README describes as "robust" shows |t| < 0.5, something is wrong — either the code has a bug, or the README's claim needs qualification.
   - **Observation-annotation alignment:** Does each observation in the report support the enriched `Visual` annotation? Discovery observations (not in the README) should be flagged as potential narrative additions.
6. For criteria that are met as written, record them as **Met** — no action needed.
7. For criteria that are NOT met, produce a **contradiction entry** with resolution options (see format below).

**The audit report:**

The agent produces a structured report. Criteria that are met are listed briefly (file, criterion, status). The bulk of the report is the contradiction entries — each one is a decision point for the user.

Each contradiction entry has this structure:

```
#### [file] — [short description]

**README says:** [exact criterion text from the README]
**Actual output:** [what the code actually produces — from summary output
  or observation report, with specific numbers]
**Root cause:** [why the discrepancy exists — universe size, market regime,
  data quality, code bug, etc.]

**Options:**

1. **[Option name]** — [what changes]
   - *Reasoning:* [why you'd pick this]
   - *Trade-off:* [what you lose or risk]

2. **[Option name]** — [what changes]
   - *Reasoning:* [why you'd pick this]
   - *Trade-off:* [what you lose or risk]

3. (optional third option if applicable)
```

**Option types** — most contradictions resolve to one of these patterns:

| Option pattern | When it applies | Example |
|----------------|-----------------|---------|
| **Relax the README criterion** | The criterion assumed conditions the code can't meet (larger universe, different market regime). The pedagogical point survives with a weaker threshold. | README says "SMB corr >= 0.75" → relax to ">= 0.10" with a note explaining why S&P 500 universe lacks small stocks |
| **Fix the code** | The code has a bug or suboptimal implementation that, once fixed, would meet the criterion. | Off-by-one in date alignment causing 50 fewer months than expected |
| **Add a README caveat** | The criterion is reasonable in general but fails in the specific data window or universe used. Add qualifying language rather than changing the number. | README says "SMB positive historically" → add "in full-market universes; S&P 500-only samples may show negative SMB due to large-cap dominance" |
| **Restructure the demonstration** | The criterion is sound, but the code approaches it in a way that can't satisfy it. A different approach would work. | README requires "|t| > 2 IS but |t| < 1 OOS for at least one characteristic" → switch from price-based characteristics to fundamental-based ones that have stronger IS signal |
| **Drop the criterion** | The criterion tests something that doesn't actually serve the learning objective, or is redundant with other criteria. | A criterion that checks an intermediate computation irrelevant to the section's "so what" |
| **Add to the narrative** | A discovery from the observation report suggests a new teaching moment not in the README. Add it as a new insight or acceptance criterion. | Observation report notes "HML collapse accelerates sharply in 2020 — COVID regime shift visible" → add as a narrative hook in Section 3 |

**Example contradiction entries:**

```
#### s4_cross_sectional_regression.py — Cross-sectional R² too low

README says: "Regression R² > 0.05 (some cross-sectional explanatory power)"
Actual output: R² = 0.007 (0.7%) [from summary output]
Root cause: Single-month cross-sectional R² is inherently noisy with 100 stocks.
  Characteristics explain very little return variation in any single month —
  this is actually a correct result that motivates Fama-MacBeth (aggregate
  over many months to get statistical power).

Options:

1. **Relax README to R² > 0.0** — accept any positive explanatory power
   - *Reasoning:* The actual R² is economically correct. Cross-sectional
     R² in a single month is known to be very low (Fama-MacBeth exists
     precisely because single-month regressions are noisy). The code
     already has a comment explaining this.
   - *Trade-off:* The criterion becomes almost trivially satisfied.

2. **Use multi-month average R²** — average R² across all months
   - *Reasoning:* The average cross-sectional R² across many months is
     typically 3-8%, which would clear the 5% bar. This better represents
     the methodology's explanatory power.
   - *Trade-off:* Requires restructuring the code to run cross-sectional
     regressions for every month and average, which changes the section's
     scope (it's meant to be a single-month demonstration).

3. **Add README caveat** — keep the criterion but qualify it
   - *Reasoning:* Change to "Average cross-sectional R² > 0.05 when
     aggregated over multiple months (single-month R² will be much lower)"
   - *Trade-off:* More accurate but more verbose criterion.
```

```
#### s6_factor_zoo.py — No characteristic crosses IS/OOS threshold

README says: "At least one characteristic has |t-stat| > 2 in-sample
  but |t-stat| < 1 out-of-sample (demonstrates overfitting)"
Actual output: No characteristic reaches |t| > 1.96 in either period.
  [from summary output; confirmed by observation report: "IS vs OOS
  bar chart shows all bars below the red significance threshold"]
Root cause: With 100 S&P 500 stocks, the cross-section is too small
  and homogeneous for any characteristic to reach significance.

Options:

1. **Relax to |t| > 1.5 IS and |t| < 1.0 OOS** — lower the bar
   - *Reasoning:* Still demonstrates the IS→OOS degradation pattern,
     just at a lower significance level. The pedagogical point (factors
     weaken OOS) is preserved.
   - *Trade-off:* Less dramatic illustration of the factor zoo problem.

2. **Use Ken French official factor returns instead of stock characteristics**
   — test IS/OOS on factor return series (MKT, SMB, HML, RMW, CMA, MOM)
   - *Reasoning:* Factor returns aggregate across thousands of stocks,
     so t-stats are much higher. MOM and HML are likely to show the
     IS-significant-but-OOS-weak pattern.
   - *Trade-off:* Changes the demonstration from "characteristics as
     factors" to "pre-built factors," which is a different exercise.

3. **Expand to 500 stocks** — increase cross-sectional power
   - *Reasoning:* More stocks = more statistical power = characteristics
     more likely to reach significance. This is the README's original intent.
   - *Trade-off:* yfinance rate limits make 500-stock downloads slow
     and fragile. May require restructuring data_setup.py.
```

**The agent's role in the audit:**

The agent produces the report but does NOT make resolution decisions. It may indicate which option it considers strongest (by listing it first or noting "recommended"), but the user chooses. Once the user decides, the agent applies the chosen resolution — editing README criteria, fixing code, adding caveats, or restructuring demonstrations as directed. If any resolution requires code changes, the loop returns to Stage 1.

### Implementation Reality Notes

After all contradictions are resolved, the agent produces one more artifact: **Implementation Reality notes**. These document every significant gap between the README's ideal expectations and what the code actually produces — and, crucially, *why* the gap exists and *how the notebook should teach it*.

**Why this matters:** The README is written from theory and literature — it describes what a quant fund with CRSP access and 30 years of data would see. The code hits the constraints of accessible data, free APIs, and limited universes. These gaps aren't failures — they're some of the most valuable teaching moments in the course. A student who understands *why* their SMB factor doesn't match Ken French has learned more about real quant research than one who gets a perfect replication.

**The agent produces one entry per significant gap:**

```
### [Short title]

**Ideal (README / literature):** [What the theory or a production setup would show]
**Actual (our implementation):** [What the code produces, with specific numbers]
**Why the gap:** [Root cause — data limitations, universe constraints, market
  regime, API restrictions, methodological simplification, etc.]
**Teaching angle:** [How the notebook should frame this — what lesson does
  the gap teach? What would a student need to do differently in production?]
```

**Example entries:**

```
### SMB Factor — Universe Bias

Ideal (README / literature): SMB constructed from full CRSP universe
  (~4000 stocks) correlates ≥0.75 with Ken French factors.
Actual (our implementation): SMB correlation = 0.04 with our S&P 500
  universe of ~450 stocks.
Why the gap: S&P 500 "small" stocks have market caps of $10-30B —
  they're Ken French's "large" stocks. True small-cap effects require
  stocks below ~$2B, which aren't in any major index.
Teaching angle: This is the most common silent error in quant research:
  universe bias. The notebook should show the student that "small" is
  relative to your universe, not absolute. Frame it as: "If you ran
  this at a fund with CRSP access, SMB correlation would be 0.75+.
  With S&P 500, you're measuring large-vs-very-large, not small-vs-large.
  The effect disappears because the effect was never in your data."
```

```
### HML Factor — The Death of Value

Ideal (README / literature): Value premium (HML) was positive and
  significant in Fama-French (1993), ~5% annualized over 1963-1992.
Actual (our implementation): HML cumulative return is negative in
  our sample (2010-2024). The "value premium" lost money.
Why the gap: This is NOT a data limitation — this is the real result.
  Value has underperformed growth since ~2007, dramatically so since
  2017 (tech dominance). The literature calls this the "death of value."
Teaching angle: Factors are not permanent. The notebook should use this
  as the week's most powerful "so what?": students are seeing, with
  their own code, one of the most debated phenomena in modern quant
  finance. Frame it as: "Your HML is correct. Value really did lose
  money. The question the industry is still debating: is this temporary
  (mean-reversion coming) or structural (intangible assets broke the
  book-to-market signal)?"
```

**Rules:**
- Only document **significant** gaps — ones where the student would notice a difference from textbook expectations or where the gap teaches something important. Minor numerical deviations (R² = 0.31 vs. README's "≈0.35") don't need entries.
- Every entry MUST have a **Teaching angle**. A gap without a teaching angle is just a bug report. The teaching angle is the reason this artifact exists.
- Distinguish between **data limitations** (our universe is too small → "in production you'd fix this") and **real phenomena** (value is dead → "this IS the production result"). The notebook should frame these very differently.
- The agent appends these notes to `code/observation_report.md` as a new `## Implementation Reality` section after Stage 3 resolves all contradictions. This way the notebook agent receives one unified artifact.

---

## How Notebook Agents Use Verified Code

Phase 4 (notebook writing) changes in one critical way: **notebook agents receive verified code files, the observation report, and must use them as the sole source of truth for all code cells and narrative content.**

### Updated Agent Inputs (Phase 4)

Notebook-writing agents receive:
1. `course/guides/notebook_creation_guide.md`
2. `course/guides/writing_guidelines.md`
3. `course/guides/task_design_guide.md`
4. The week's `README.md`
5. **`code/data_setup.py`** — the shared data layer
6. **The relevant `code/` subfolder** (e.g., `code/lecture/` for the lecture agent)
7. **`code/observation_report.md`** — the observation report (visual observations from Stage 2 + Implementation Reality notes from Stage 3)

### What the Notebook Agent Does

1. **Reads `data_setup.py`** to understand what data to download and how, then writes equivalent direct download calls (e.g., `yf.download()`) in the notebook. **The notebook must NOT import from `data_setup.py`** — that file exists only for the `.py` code files to share during Phase 3 verification. Notebooks are standalone; students run them without any external files. **No caching logic in notebooks** — no cache directories, no parquet caches, no file-existence checks. `data_setup.py` uses caching because code files are run repeatedly during development; notebooks are run once top-to-bottom, so caching adds complexity for zero benefit.
2. **Reads each code file** in order (s1, s2, ... or ex1, ex2, ... or d1, d2, ...).
3. **For each `# ── CELL:` block**, creates:
   - A markdown cell BEFORE the code, using `Purpose` to guide the prose
   - A code cell containing the code between this marker and the next, **verbatim**
   - A markdown cell AFTER the code, using `Takeaway` or `Visual` to guide interpretation
4. **Reads the observation report** for the plots in its notebook. Uses it to write richer interpretation prose — the report's detailed observations and discoveries provide specific numbers, patterns, and narrative hooks that go beyond what the code annotations alone can supply.
5. **Writes prose** following `writing_guidelines.md` voice and `notebook_creation_guide.md` structure. The annotations provide cell-level structure (what each cell does, what to look for). The observation report provides narrative depth (what the plots actually reveal, what's surprising, what connects to real-world consequences).
6. **Adds section headers, transitions, opening, and closing** per notebook structure rules.

### How Annotations and the Observation Report Interact

These two sources serve different roles — the notebook agent uses both:

- **Code annotations** (`Purpose`, `Visual`, `Takeaway`) are **structural markers**. They tell the agent what each cell does and what output type to expect. They guide the "before" and "after" prose cells at a per-cell level.
- **The observation report** is **narrative enrichment**. It has three layers:
  - *Detailed Observations* and *Discoveries* (from Stage 2) — specific visual observations, surprising patterns, and numbers read from plots. The richest source for interpretation prose and "aha moments."
  - *Implementation Reality* (from Stage 3) — gaps between textbook expectations and what the code actually produces, with teaching angles. **These are among the most valuable teaching moments in the course.** The notebook must explicitly surface them — framing limitations as lessons, not footnotes.

**When they conflict:** The observation report wins — it's based on actual observed output and has been reviewed by the user. If an annotation says "MKT dominates at $4.5" but the observation report says "MKT reaches $3.8," use $3.8.

**How to use Implementation Reality notes:** Each note distinguishes between *data limitations* ("in production with CRSP, you'd see X — we can't because Y") and *real phenomena* ("this IS the production result — value really did underperform"). The notebook should frame these differently:
- **Data limitations** → "Here's what we got, here's what a fund with better data would see, and here's why the gap exists. When you work in industry, this is the first thing you'd fix."
- **Real phenomena** → "Your code is correct. This IS what happens. The industry is debating whether this is temporary or structural." These moments deserve the most voice and weight in the narrative — they're where the student transitions from textbook learner to practitioner.

### What the Notebook Agent Does NOT Do

- Include the `if __name__` block (assertions and plot summaries are verification infrastructure, not student-facing code)
- Include the `# ── CELL:` markers or annotation comments
- Include the file docstring
- Invent new code cells not present in the verified files
- Modify computation logic, data sources, or variable names
- Reference the observation report as a document (students never see it — its content is woven into the prose naturally)

The agent MAY consolidate imports from individual files into the notebook's setup cell, and MAY adjust minor formatting (line breaks within a cell). It must NOT change what the code computes.

---

## Changes to the README Format

To support code verification, the README template gains **acceptance criteria** for each section, exercise, and deliverable. These define what "correct" means in verifiable terms.

### Lecture sections add acceptance criteria:

```markdown
### Section 6: Corporate Actions — The Hidden Data Landmine
**Concepts:** stock splits, dividends, adjusted vs. unadjusted prices
**Demo:** Download AAPL data spanning the 2014 7:1 split. Reconstruct
  nominal prices. Plot nominal vs. adjusted. Show return comparison table.
**"So what?":** A model trained on nominal prices learns phantom crashes.

**Acceptance criteria:**
- Nominal AAPL price on 2014-06-06 > $600 (pre-split)
- Nominal AAPL price on 2014-06-09 < $100 (post-split)
- Nominal daily return on split date < -80%
- Adjusted daily return on split date within ±5%
- Plot shows visible discontinuity in nominal line
```

### Seminar exercises add acceptance criteria:

```markdown
### Exercise 2: Corporate Action Forensics
- **Task type:** Guided Discovery
- **The question:** How do different corporate actions distort raw price data?
- **Acceptance criteria:**
  - At least 2 split events with visible price discontinuities
  - Dividend-only adjustment drift > 1% annually for dividend payers
  - Student can distinguish split-adjusted from fully-adjusted returns
```

### Homework deliverables add acceptance criteria:

```markdown
1. **FinancialDataLoader class**
   - **Task type:** Construction
   - **Acceptance criteria:**
     - Downloads ≥ 50 tickers without crashing
     - Handles missing data for at least 3 tickers with gaps
     - Produces valid Parquet that round-trips without data loss
     - Quality scorecard flags known-bad tickers correctly
```

### Acceptance Criteria Quality Bar

- **Verifiable:** Each criterion can be checked with an assertion or a concrete inspection. "The plot looks good" is not a criterion. "The two lines diverge by > 50% at the split date" is.
- **Approximate:** Use ranges and inequalities, not exact values. Financial data shifts.
- **Complete:** Cover both the "happy path" (expected results) and the "failure mode" (what would go wrong without the correct implementation).
- **Tied to the teaching goal:** Every criterion should connect to why this section exists. If a criterion doesn't support the learning objective, it's noise.
