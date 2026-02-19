# File Agent Guide — Step 4B

> **Consumer:** Step 4B file agents. Each agent implements and tests ONE code file. You read this guide + `code_plan.md` (your file's entry) + `data_setup.py`. Nothing else.

---

## Your Process

```
1. Read your entry in code_plan.md (criteria, strategy, alternatives)
2. Read data_setup.py for data layer context
3. If dependent on upstream cache: verify cache files exist in .cache/
4. Write the .py code file
5. Debug run: run with slashed computation to verify code correctness
   (see "Debug Run First, Full Run Second" below)
   - Fix bugs, up to 3 attempts
6. Full run: once code is correct, run with full computation
   - For PyTorch: use the --device flag specified in the launch prompt
   - For GPU files: use course/shared/kaggle_gpu_runner.py (see below)
7. Check results:
   - Assertions fail → diagnose, fix, re-run
   - Assertions pass but results are pedagogically flat → try alternatives
     from code_plan.md before accepting (see Special Case 5)
   - All alternatives exhausted, still flat → pedagogical surrender
     (see Special Case 5)
8. Write code/logs/notes/{filename}_notes.md (see Per-File Notes below)
```

**Every run writes to the log file.** Debug runs, full runs, re-runs after fixes — all of them. Pipe stdout+stderr to `code/logs/{filename}.log` at launch time so output streams live:

```bash
PYTHONUNBUFFERED=1 poetry run python code/lecture/s3_gradient_boosting.py 2>&1 | tee code/logs/s3_gradient_boosting.log
```

**`PYTHONUNBUFFERED=1` is required.** Python block-buffers stdout when piped, delaying output by minutes. This env var forces immediate flushing so every `print()` appears in the log in real time.

Each new run overwrites the previous log. The main session monitors these files to track progress in real time — if you run without writing to the log, there is no visibility into whether your script is alive, stuck, or failing. The final log file (from the last successful run) is what 4C reads for criteria coverage.

**Never re-launch a running script.** Long-running scripts (NN training, walk-forward backtests) may take 10+ minutes. Wait for the previous run to finish before retrying. Check the log file for progress output.

**Progress logging:** Print progress for any loop that might take more than a few seconds. When in doubt, add it — the cost is one `print` line, the cost of omitting it is staring at a silent log wondering if the script is alive. Target: a new progress line at least every 30 seconds of wall time.

```python
for i, date in enumerate(dates):
    if i % 10 == 0:
        print(f"  [{i+1}/{len(dates)}] processing {date}")
    # ... computation
```

This applies to walk-forward loops, cross-validation folds, hyperparameter searches, training epochs, bootstrap iterations — anything iterative. For libraries with built-in progress (e.g., `tqdm`, LightGBM verbose), use those instead of manual prints.

---

## Code File Format

Each code file has two parts: **cell-marked implementation** (the code that will appear in notebooks) and a **verification block** (assertions + structured output + plot saving, never included in notebooks).

### Cell-Marked Implementation

The implementation is divided into cells using `# ── CELL:` markers. **Read `cell_design.md` for the full CELL specification** — marker format, sizing rules, and splitting strategies (classes, loops, visualizations). That guide is the single source of truth for cell structure; this section covers only execution-specific notes.

**Backend note:** Set a non-interactive matplotlib backend (e.g., `matplotlib.use("Agg")`) in each file's import section, before any CELL markers. This prevents `plt.show()` from blocking during script execution. Notebook agents skip this line when consolidating imports.

**`plt.show()` in cell code is intentional.** It appears inside CELL-marked sections (which get extracted into notebooks) so that notebooks display plots inline. With the Agg backend, `plt.show()` is a no-op during script execution — harmless. The actual plot saving happens in the verification block via `fig.savefig()`.

### Debug Run First, Full Run Second

**Every file with non-trivial computation** (ML training, walk-forward loops, HP searches, bootstrap iterations) must be debugged on a cheap run before the full run. Slash everything expensive: 1-2 epochs, 2 CV folds, tiny param grid, first N dates. The goal is catching code errors (shape mismatches, wrong column names, broken data flow) — not hitting acceptance criteria. Assertions may fail on reduced data; that's fine.

Once code is correct, do the full run. Your 3 attempts apply to debug runs. The full run should ideally succeed on the first try.

### Device Handling for PyTorch Files

Files that use PyTorch should accept a `--device` argument:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
args = parser.parse_args()
device = args.device
```

Use `device` for all tensor operations and model training. The main session passes the appropriate device when launching the script. This argparse block goes in the import section, before CELL markers — notebook agents skip it.

**Device fallback:** If your script fails with an error containing "MPS", "mps", or "Metal", rerun it with `--device cpu`. This counts as one of your 3 allowed attempts. The `code_plan.md` notation `mps (downshift: cpu)` means: try MPS first, fall back to CPU on failure.

**GPU files (`device: gpu`):** The full run goes through `course/shared/kaggle_gpu_runner.py` instead of `poetry run python`. The script is unaware of this — it sees `--device cuda` and runs normally. The runner packages the week's `code/` directory, runs on a remote GPU, and writes output back to the expected local paths. Debug runs still happen locally on CPU via `poetry run python <script> --device cpu`.

### Device Choice: When to Use What

Not every PyTorch file benefits from a GPU. Choosing the wrong device wastes time — either through MPS overhead on small workloads or through unnecessarily slow CPU training on large ones. Use this decision framework:

**Default to CPU** for:
- Tabular models with < 1M parameters and < 100K training samples (most of this course)
- Scripts where training takes < 30 seconds on CPU
- Any workload dominated by data loading, pandas, or sklearn (GPU doesn't help)
- Walk-forward loops that retrain small models many times (MPS startup overhead per retrain dominates)

**Use MPS** (`mps (downshift: cpu)`) for:
- Neural network training that takes > 60 seconds on CPU
- Batch sizes > 512 with matrix-heavy operations (large matmuls benefit from GPU parallelism)
- LSTM/Transformer training on sequence data (naturally parallelizable)
- **NOT for**: LightGBM, XGBoost, sklearn models — these don't use PyTorch and ignore `--device`

**Use GPU** (`device: gpu`, via Kaggle) for:
- Training that takes > 10 minutes on CPU (even with MPS)
- Models with > 10M parameters (transformers, large NNs)
- NLP fine-tuning (FinBERT, sentence-transformers) — requires significant VRAM
- Any workload that crashes on MPS with no CPU fallback completing in time

#### MPS Overhead and Failure Modes

MPS carries two hidden costs that make it slower than CPU for small workloads:

1. **Startup overhead (1-3s per session):** Metal framework initialization and shader compilation happen on first GPU operation. For a script that runs 5 seconds on CPU, MPS adds 1-3s of overhead before any compute begins — a net loss.

2. **CPU↔GPU data transfer:** Every `.to("mps")` call copies data across the memory bus. For small tensors this takes longer than computing on CPU directly. Walk-forward loops that repeatedly create and move tensors are especially penalized.

MPS also has **unsupported operations** that cause runtime crashes:

| Failure type | Example error | Common trigger |
|---|---|---|
| Op not implemented | `NotImplementedError: ... on MPS` | Certain einsum variants, sparse operations, complex number ops |
| Numerical precision | Silent wrong results, NaN gradients | Some reduction operations at float16, cumulative ops |
| Memory errors | `RuntimeError: MPS backend out of memory` | Large batch sizes on machines with shared GPU memory |

When an MPS failure occurs, the error message typically contains "MPS", "mps", or "Metal" — this is what triggers the fallback protocol.

#### `dl_training.py` Device Support

The shared `course/shared/dl_training.py` utility accepts a `device` parameter on both `fit_nn()` and `predict_nn()`. Pass `None` (default) for auto-detection (CUDA > MPS > CPU), or pass the `--device` argument from your script's argparse:

```python
from shared.dl_training import fit_nn, predict_nn

info = fit_nn(model, X_train, y_train, device=args.device)
preds = predict_nn(model, X_test, device=args.device)
```

`SequenceDataset` keeps data on CPU — move batches to device in your training loop when building custom sequence training. For standard feedforward networks, `fit_nn()` handles all device transfers internally.

### Verification Block

The `if __name__ == "__main__":` block contains assertions, structured results output, and plot saving. **Never included in notebooks.**

#### Assertions

Fail-fast `assert` statements. If any fails, the script stops with a clear error message.

```python
if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    assert len(factors) >= 180, f"Expected ≥180 months, got {len(factors)}"
    mkt_mean = betas["MKT"].mean()
    assert 0.8 <= mkt_mean <= 1.2, f"MKT beta mean {mkt_mean:.3f} outside [0.8, 1.2]"
    assert smb_corr < 0.3, f"SMB-KF correlation {smb_corr:.4f} expected < 0.3"
```

**Assertion rules:**

The four base criteria rules from `task_design.md` apply to assertions — approximate ranges, machine-checkable, two-sided, tied to learning. Read them there; they are not restated here. The following rules are specific to writing assertions in code:

- **Descriptive failure message** with actual value: `f"Expected ..., got {actual}"`. The message is your debugging output when a run fails.
- **Assert on data properties that ensure plots look correct.** You can't assert on a plot directly, but asserting "the two lines diverge by > 50%" guarantees the visual will be dramatic.
- **Use the assertion ranges from code_plan.md.** Do NOT widen ranges to make assertions pass. If the result falls outside the planned range, try alternatives (from the plan) or flag as infeasible. Silently relaxing thresholds defeats the pedagogical purpose.

#### Structured Results Output

After all assertions pass, the script prints structured results in a standard format. This output is captured to `code/logs/{filename}.log` and becomes the raw evidence of what the code produced.

```python
    # ── RESULTS ────────────────────────────────────
    print(f"══ lecture/s3_factor_construction ══════════════════")
    print(f"  n_months: {len(factors)}")
    print(f"  mkt_beta_mean: {mkt_mean:.4f}")
    print(f"  smb_kf_corr: {smb_corr:.4f}")
    print(f"  hml_mean_monthly: {hml_mean:.6f}")
    print(f"  hml_cumulative_final: {hml_cum.iloc[-1]:.4f}")
```

**Format rules:**
- **File header first:** `══ subfolder/filename_without_ext ══` — makes logs parseable per file.
- **Key metrics as `key: value` pairs**, indented. Only values that matter for acceptance criteria or downstream narrative. Not a full data dump.
- **Consistent numeric formatting:** `.4f` for most values, `.6f` for very small values, `.2f` for prices.

#### Plot Saving and Output

For files that produce plots, save to `logs/plots/` and print plot metadata after the results:

```python
    # ── PLOT ───────────────────────────────────────
    fig.savefig(PLOT_DIR / "s3_factor_cumulative.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s3_factor_cumulative.png ──")
    print(f"     type: multi-line cumulative returns")
    print(f"     n_lines: {len(ax.get_lines())}")
    print(f"     y_range: [{ax.get_ylim()[0]:.2f}, {ax.get_ylim()[1]:.2f}]")
    print(f"     title: {ax.get_title()}")
```

**Plot rules:**
- **Save to `PLOT_DIR`** (imported from `data_setup`) with filename matching the file's prefix: `s3_factor_cumulative.png`.
- **`dpi=150, bbox_inches="tight"`** — readable PNGs without clipped labels.
- **Print enough metadata to characterize the plot** — type, key metrics, axis ranges, line/bar counts, title. Use matplotlib's introspection API: `ax.get_lines()`, `ax.get_xlim()`, `ax.get_ylim()`, `len(ax.patches)`, `ax.get_title()`, `ax.get_xlabel()`, `ax.get_ylabel()`. Simple plots may need 3-4 lines; complex multi-panel plots may need more.

#### Libraries That Produce HTML Reports (quantstats, plotly, etc.)

Some libraries generate rich HTML reports instead of (or in addition to) matplotlib figures. These are valuable artifacts but the pipeline expects PNGs — Step 5 reads images, not HTML. Handle both:

**In the Python code:**

1. **Generate the HTML** and save to `PLOT_DIR`:
   ```python
   html_path = PLOT_DIR / "{prefix}_{report_name}.html"
   library.generate_report(..., output=str(html_path))
   ```

2. **Extract key metrics programmatically** and print to stdout — don't rely on the HTML as the only place metrics live. Use the library's API, not HTML parsing:
   ```python
   print(f"  ── report: {html_path.name} ──")
   print(f"     metric_1: {value_1:.4f}")
   print(f"     metric_2: {value_2:.4f}")
   ```

3. **Convert the HTML to PNG** using Playwright so the image enters the pipeline for Step 5:
   ```python
   from playwright.sync_api import sync_playwright

   png_path = html_path.with_suffix(".png")
   with sync_playwright() as p:
       browser = p.chromium.launch()
       page = browser.new_page(viewport={"width": 1200, "height": 800})
       page.goto(f"file://{html_path.resolve()}")
       page.wait_for_load_state("networkidle")
       page.screenshot(path=str(png_path), full_page=True)
       browser.close()
   print(f"  ── plot: {png_path.name} (from HTML report) ──")
   ```

   The HTML stays in `PLOT_DIR` as a rich interactive artifact for the user; the PNG enters the pipeline for Step 5. Both live in `logs/plots/`.

#### Status Line

Every file ends with a status line:

```python
    print(f"✓ s3_factor_construction: ALL PASSED")
```

Scan log files for `✓` lines to confirm all files passed.

---

## Special Cases

During coding, the plan may need updates. Five cases are recognized — document deviations in your `_notes.md`.

**1. Infeasible criterion.** The data or model demonstrably cannot produce the expected result. This is the **last resort** — exhaust all other options first. Before marking a criterion infeasible:
   - Verify the implementation is correct (not a bug producing the wrong result)
   - Try at least 2 reasonable alternative approaches (different hyperparameters, different preprocessing, different algorithm variant)
   - Check code_plan.md for pedagogical alternatives and try those
   - Confirm the result is stable, not a single unlucky run

   Only after all checks fail is the criterion genuinely infeasible. **Action:** implement the analysis code so the actual result is visible in `code/logs/{filename}.log`. Omit the assertion. In `_notes.md`, document: which criterion, what alternatives were attempted, what each produced, and why the criterion is infeasible.

**2. Criterion migrates.** A criterion fits better in a different file than initially mapped. **Action:** note the migration in `_notes.md`.

**3. Comparative criterion across files.** Already handled by the wave structure. The assertion lives in the downstream file. Ensure the upstream file caches the needed result.

**4. Criterion decomposes.** A single criterion spans multiple files. **Action:** split into sub-assertions. Assert your part. Note the decomposition in `_notes.md`.

**5. Pedagogically flat result.** Assertions pass but results don't demonstrate the teaching point. This is NOT a failure — it means the numbers are technically fine but don't clearly show what the section intends to teach.

   **Step-by-step response:**
   1. Check code_plan.md for pedagogical alternatives
   2. Try each alternative (different feature set, different parameters, etc.)
   3. If an alternative produces clearer results, adopt it
   4. If no alternative helps → **pedagogical surrender:**
      - Keep the best-passing version of the script (assertions pass)
      - Print `⚠ PEDAGOGICALLY FLAT: {criterion_id}` in stdout before the status line
      - In `_notes.md`, document: what was tried, what each produced, why the teaching point didn't emerge
      - This is a valid, honest outcome — do NOT force results by relaxing thresholds, cherry-picking seeds, or overfitting to make the demo look good

   **Why surrender is acceptable:** Downstream pipeline stages handle this. The 4C verify agent flags it in execution_log.md. Step 6 (Consolidation) can flag it as "Pedagogically Unusable" and recommend reworking or dropping the section. Honest reporting now is vastly better than silently producing misleading results.

In all five cases, `expectations.md` remains immutable. The criteria map in `code_plan.md` tracks the original assignment; `_notes.md` files track deviations.

---

## Per-File Notes

Write `code/logs/notes/{filename}_notes.md` after completing your file. This captures implementation context that only you know.

```markdown
# Notes: {filename}

## Approach
[Why this implementation over alternatives — 1-2 sentences]

## Threshold Reasoning
[Why specific assertion values — e.g., "R² upper bound set to 0.45 because
adding more stocks pushes it higher"]

## ML Choices
[For model-training files only. What was searched, what was chosen,
IS/OOS gap. E.g., "LightGBM learning_rate searched [0.01, 0.05, 0.1]
via 3-fold temporal CV; 0.05 chosen (best OOS IC). Train IC=0.08,
OOS IC=0.03 — typical 2-3× gap for this universe."]

## Pedagogical Alternatives Tried
[If any alternatives from code_plan.md were attempted:
what was tried, what it produced, whether it was adopted.
"Tried Alternative A (3-feature subset). Substitution ratio improved
from 0.8 to 2.1. Adopted."]

## Challenges
[What was tricky, how it was resolved — only if non-trivial]

## Deviations from Plan
[Any changes from what code_plan.md specified for this file.
"None" if faithful to plan.]
```

---

## Example

A condensed file showing both parts — cell-marked implementation and verification block:

```python
"""Section 1: The CAPM — One Factor to Rule Them All"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_equity_data, CACHE_DIR, PLOT_DIR

returns = load_equity_data()["Close"].pct_change().dropna()
mkt = returns.mean(axis=1)


# ── CELL: capm_scatter ───────────────────────────────────

betas = returns.apply(lambda s: np.polyfit(mkt, s, 1)[0])
avg_ret = returns.mean()
m, b = np.polyfit(betas, avg_ret, 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(betas, avg_ret, alpha=0.5, s=20)
ax.plot(sorted(betas), [m * x + b for x in sorted(betas)], "r-")
ax.set(title="CAPM Cross-Sectional Test", xlabel="Beta", ylabel="Avg Return")
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    from sklearn.metrics import r2_score
    r2 = r2_score(avg_ret, m * betas + b)
    assert 0.15 <= r2 <= 0.45, f"R² = {r2:.4f}, outside [0.15, 0.45]"
    assert m > 0, f"Slope = {m:.4f}, expected > 0"
    assert len(betas) >= 100, f"n_stocks = {len(betas)}, expected ≥100"

    fig.savefig(PLOT_DIR / "s1_capm_scatter.png", dpi=150, bbox_inches="tight")
    print(f"══ lecture/s1_capm_scatter ══════════════════════════")
    print(f"  r_squared: {r2:.4f}")
    print(f"  slope: {m:.4f}")
    print(f"  n_stocks: {len(betas)}")
    print(f"  ── plot: s1_capm_scatter.png ──")
    print(f"     n_points: {len(betas)}")
    print(f"     title: {ax.get_title()}")
    print(f"✓ s1_capm_scatter: ALL PASSED")
```

---

## Code Quality Standard

- Clear, descriptive variable names (`nominal_close`, not `df2`)
- Docstrings on non-trivial functions
- Consistent style (Black-compatible formatting)
- No dead code, commented-out alternatives, or TODOs
- Cell sizing per `cell_design.md` (target ≤15, limit 25, visualizations uncapped)
- Imports organized: stdlib → third-party → local

This code appears in notebooks essentially unchanged. Students read it. Make it worth reading.

---

## Quality Bar

- [ ] Code file written with CELL markers and verification block
- [ ] All assertions use ranges from code_plan.md (no silent relaxation)
- [ ] Assertions pass (or alternatives tried before flagging infeasible)
- [ ] Plot saved to `PLOT_DIR` with correct naming and `dpi=150`
- [ ] HTML reports (if any) screenshotted to PNG via Playwright MCP, key metrics extracted to stdout
- [ ] Structured output follows the format spec (header → metrics → plot metadata → status)
- [ ] Progress logging for any loop >30 seconds
- [ ] `code/logs/{filename}.log` captured from successful run
- [ ] `code/logs/notes/{filename}_notes.md` written with approach, challenges, deviations
- [ ] Code quality standard met: clear names, consistent style
- [ ] Cell design quality bar met (see `cell_design.md`)
