# Cell Design Guide

> **Consumers:** Step 4B file agents (writing code files) and Step 7 notebook agents (building notebooks). This guide is the single source of truth for cell structure, sizing, splitting, and structural constraints. Both `file_agent_guide.md` and `notebook.md` reference this guide — neither repeats its rules inline.

---

## How Code Files Become Notebooks

Step 4 code files contain `# ── CELL:` markers. Step 7 extracts the code between consecutive markers as **verbatim notebook code cells** — never reordered, never refactored, never split. Step 7 then writes prose (markdown) cells between them.

Cell granularity is **Step 4's responsibility**. By the time Step 7 runs, code cell structure is locked.

### CELL Marker Format

```python
# ── CELL: compute_betas ──────────────────────────────────

betas = returns.apply(lambda s: np.polyfit(mkt, s, 1)[0])
avg_ret = returns.mean()


# ── CELL: capm_scatter ───────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(betas, avg_ret, alpha=0.5, s=20)
ax.plot(sorted(betas), [m * x + b for x in sorted(betas)], "r-")
ax.set(title="CAPM Cross-Sectional Test", xlabel="Beta", ylabel="Avg Return")
plt.tight_layout()
plt.show()
```

- **Markers are structural, not interpretive.** They define cell boundaries. Interpretation of what results mean is Step 6's job (Consolidation), not Step 4's.
- **What goes between two consecutive markers IS one notebook cell.** The notebook agent extracts this code verbatim.
- Each marker has a unique name identifying the cell for cross-referencing.

---

## Cell Sizing Rules

| Cell type | Target | Limit |
|-----------|--------|-------|
| **Standard computation** | ≤15 lines | 25 lines |
| **Helper function** (factored from loop) | ≤20 lines | 25 lines |
| **Class `__init__` + docstring** | ≤20 lines | 25 lines |
| **Individual method** (monkey-patched) | ≤20 lines | 25 lines |
| **Visualization** | — | No cap |

**Hard cap for non-visualization cells: 40 lines.** Anything exceeding 40 lines requires structural redesign — factor into helpers, split into sub-steps, or use monkey-patching for classes.

---

## Splitting Strategies

### Classes

Any class >25 lines **must** use monkey-patching at the code file level. Define the class with `__init__` in one CELL, then define each method as a standalone function in its own CELL and attach it to the class.

```python
# ── CELL: pipeline_init ──────────────────────────────────

class AlphaModelPipeline:
    """Walk-forward alpha evaluation pipeline."""

    def __init__(self, model, features, target):
        self.model = model
        self.features = features
        self.target = target


# ── CELL: pipeline_impute ────────────────────────────────

def _impute_window(self, X_train):
    """Cross-sectional median imputation within training window."""
    return X_train.fillna(X_train.median())

AlphaModelPipeline._impute_window = _impute_window


# ── CELL: pipeline_fit_predict ───────────────────────────

def fit_predict(self):
    """Run walk-forward prediction loop."""
    results = {}
    for test_date in self.dates:
        results[test_date] = self._train_one(test_date)
    self.predictions = pd.concat(results)

AlphaModelPipeline.fit_predict = fit_predict
```

Each CELL is ≤25 lines. The script runs correctly top-to-bottom. Step 7 extracts each CELL verbatim and writes prose between method definitions.

**This applies to all notebook types** — lectures, seminars, and homework alike.

### Computation Loops

Loop bodies >15 lines → factor the inner logic into a helper function defined in a preceding CELL.

```python
# ── CELL: wf_one_step ────────────────────────────────────

def _train_predict_one(model, features, target, train_dates, test_date):
    """Train on train_dates, predict on test_date."""
    train_idx = features.index.get_level_values(0).isin(train_dates)
    X_train = features.loc[train_idx].fillna(features.loc[train_idx].median())
    model_clone = clone(model)
    model_clone.fit(X_train, target.loc[train_idx])
    test_idx = features.index.get_level_values(0) == test_date
    return pd.Series(model_clone.predict(features.loc[test_idx]), ...)


# ── CELL: walk_forward_loop ──────────────────────────────

results = {}
for i, test_date in enumerate(dates[24:]):
    results[test_date] = _train_predict_one(
        model, features, target, dates[:24+i], test_date
    )
    if i % 10 == 0:
        print(f"  [{i+1}/{len(dates)-24}] {test_date}")
```

**Exception: do not refactor calls to `shared/` utilities.** If the loop body calls functions from `course/shared/` (e.g., `fit_nn()`, `walk_forward_split()`), use them as-is. The shared layer is pre-built; don't wrap it in another helper just to meet the line limit.

### Visualizations

**One plot = one cell. No size cap.** Everything that produces a single visual artifact (`plt.show()`) belongs in one CELL — figure creation, subplot population, annotations, formatting. Never split a plot mid-construction.

**Separate plots = separate CELLs.** Two independent figures producing two `plt.show()` calls go in two CELLs.

---

## Notebook Structure Targets

Step 7 agents must meet these structural constraints. Step 4 agents should be aware of them — properly sized CELLs make them achievable.

### Cell Counts and Ratios

| Notebook | Cell count target | Min markdown (% of total) | Max consecutive code cells |
|----------|------------------|--------------------------|---------------------------|
| **Lecture** | 45–70 cells | 60% | 0 (NEVER two in a row) |
| **Seminar** | 25–45 cells | 45% | 2 (solution blocks only) |
| **Homework** | 35–75 cells | 40% | 2 (solution blocks only) |

### Code Cell Content Rules

- **Each cell does ONE thing.** Never combine "download data" + "compute returns" + "make plot."
- **`print()` is for data output only** — statistics, table rows, confirmation. Never for teaching or narrative (see the print() Prohibition in `notebook.md`).
- Minimal inline comments — surrounding prose cells handle explanation.
- Never suppress warnings globally — fix them or explain them.

---

## Quality Bar

### For Step 4B (code files)
- [ ] Every non-visualization CELL block ≤25 lines (target ≤15)
- [ ] No non-visualization CELL block >40 lines
- [ ] Classes >25 lines use monkey-patching with method-per-CELL
- [ ] Loop bodies >15 lines factored into helper function CELLs
- [ ] `shared/` utility calls left intact (not wrapped in helpers)
- [ ] Each CELL does one thing
- [ ] One plot per visualization CELL; separate plots in separate CELLs

### For Step 7 (notebooks)
- [ ] Prose-to-code ratio meets minimum (60% / 45% / 40%)
- [ ] Max consecutive code cells respected (0 / 2 / 2)
- [ ] Cell count within target range
- [ ] `print()` never used for narrative content
