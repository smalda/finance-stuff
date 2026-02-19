# Observation Review — Step 5 Guide

> **Consumer:** Observation agent (Step 5). You run in a **fresh context** with minimal input. Your job is to describe what the code produced — what the plots show, what the numbers say — first without any expectations anchoring, then compared against acceptance criteria. You observe; you never interpret. Teaching angles, narrative framing, and divergence strategy are Step 6's job.

---

## Inputs

Phase 1:
- **This guide** (`observation_review.md`)
- **`blueprint.md`** from Step 2 — structural context only (section names, what each section demonstrates). Contains no numerical predictions.
- **Plot PNGs** from `code/logs/plots/`
- **`run_log.txt`** from Step 4 — structured execution output with metrics and plot metadata

Phase 2 (injected into same context after Part 1 is written):
- **`expectations.md`** from Step 3 — acceptance criteria, production benchmarks, known constraints, data plan

## Output

**`observations.md`** in the week folder. Immutable once the user verifies it.

---

## The Two Phases

This step has two sequential phases in a single agent session. Phase 1 writes unbiased observations. Phase 2 adds criteria comparison. **Phase 1 content is frozen once written — Phase 2 appends, never modifies.**

```
Phase 1: OBSERVE (no expectations)
  Read:  observation_review.md + blueprint.md + plot PNGs + run_log.txt
  Write: Part 1 of observations.md

Phase 2: COMPARE (expectations added)
  Read:  expectations.md (injected into same context)
  Write: Part 2 of observations.md (appended — Part 1 untouched)
```

Why two phases: expectations anchor perception. An agent told "R² should be 0.15–0.45" reads a plot differently than one describing what it sees. Phase 1 captures unbiased ground truth. Phase 2 adds the analytical layer.

---

## Phase 1: "What Did We Get?"

**Inputs:** This guide + `blueprint.md` + plot PNGs from `code/logs/plots/` + `run_log.txt`

The blueprint gives you structural context — section names, what each section demonstrates — but contains no numerical predictions. You know WHAT each section is about, not what values to expect.

**Write Part 1 of `observations.md` with these sections:**

### Visual Observations

One entry per plot. For each:

```markdown
### s3_factor_cumulative.png
![](code/logs/plots/s3_factor_cumulative.png)

**Observes:** Five lines showing cumulative factor returns. MKT dominates,
reaching ~$4.2 by 2024. HML declines visibly from 2017 onward, ending
negative. SMB is nearly flat. RMW and CMA show modest positive trends.

**Cross-check with run_log:** MKT final = $4.21 ✓ | HML final = -$0.32 ✓ |
SMB final = $1.04 ✓ (matches visual)
```

- **Describe what you see**, not what you think it means. Patterns, trends, axis values, line/bar counts, colors, notable features.
- **Cross-check** every plot against its run_log.txt plot metadata (line counts, y-range, title) and related metrics. Mark matches with ✓, mismatches with ✗ and a note.
- **Flag plot quality issues:** clipped labels, overlapping legends, unreadable elements, misleading scales.

### Numerical Observations

Work through `run_log.txt` systematically:

**Cross-file consistency** — When the same metric key appears in multiple file sections, compare values. Note whether they're consistent or divergent:
```markdown
- SMB correlation = 0.04 in s3, 0.05 in s5 — consistent ✓
- All files report 192 months of data — consistent ✓
```

**Notable values** — Flag metrics that are extreme or surprising based on general financial domain knowledge. You don't have acceptance criteria yet — use your understanding of finance:
```markdown
- s6: No characteristic reaches |t| > 1.5 in-sample — very weak signals
- s1: R² = 0.95 in a cross-sectional regression — suspiciously high
```

**Signal significance** — For every section that reports IC statistics (from `ic_summary()` or manual computation), record:
- IC t-statistic and p-value
- Whether the `⚠ WEAK SIGNAL` warning appeared in run_log.txt
- Whether the `⚠ DEGENERATE PREDICTIONS` warning appeared (spread ratio)
- IC sign consistency (pct_positive)

```markdown
- s4: mean rank IC = 0.031, t = 2.41, p = 0.019 (n=69) — significant ✓
- s5: mean rank IC = 0.018, t = 0.94, p = 0.35 (n=24) — ⚠ WEAK SIGNAL
- s5: spread_ratio = 0.07 — ⚠ DEGENERATE PREDICTIONS
```

These observations feed directly into Step 6's classification of whether a model learned a real signal vs. noise. Record them even when you don't have acceptance criteria yet — the t-stat and spread ratio are meaningful on their own.

**Warnings / unexpected output** — Any stderr, deprecation warnings, convergence warnings, or unexpected prints from run_log.txt (including `⚠ WEAK SIGNAL`, `⚠ DEGENERATE PREDICTIONS`, `⚠ OVERFIT`, `⚠ SIGNAL DECAY`, and `⚠ HIGH TURNOVER` markers).

- `⚠ SIGNAL DECAY` — record in signal significance section: note first-half vs second-half IC and t-stats.
- `⚠ HIGH TURNOVER` — record in strategy evaluation section: note turnover level and estimated cost drag.

---

## Phase 2: "How Does It Compare?"

**New input added to context:** `expectations.md` (acceptance criteria, production benchmarks, known constraints, data plan)

**Append Part 2 to `observations.md`. Do not modify Part 1.**

Write these sections:

### Acceptance Criteria Audit

Per section/exercise/deliverable from expectations.md. Every criterion gets a verdict:

```markdown
### Section 3: Factor Construction

| Criterion | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| n_months ≥ 180 | ≥ 180 | 192 | ✓ pass |
| SMB-KF correlation < 0.3 | < 0.3 | 0.04 | ✓ pass (far below) |
| MKT beta mean ∈ [0.8, 1.2] | [0.8, 1.2] | 1.03 | ✓ pass |
```

Note margin — "pass (far below)" or "pass (near upper bound)" — when a value is near a range boundary or far from it. This matters for Step 6.

### Production Benchmark Comparison

For each section where expectations.md cites a production benchmark:

```markdown
- **S3 SMB-KF correlation:** ours = 0.04, production = 0.75+
  (Fama-French 1993, CRSP full universe). Gap reflects S&P 500 universe bias.
- **S1 cross-sectional R²:** ours = 0.31, production = 0.42
  (Gu-Kelly-Xiu 2020, CRSP monthly). Moderate gap.
```

Report the gap. Do not explain how to teach it — that's Step 6.

### Known Constraints Manifested

Which constraints from expectations.md are visibly reflected in the results? Connect each manifested constraint to specific Phase 1 observations:

```markdown
- **Survivorship bias:** Consistent with the uniformly positive cumulative
  returns observed in s2 (Phase 1) — no tickers show delisting losses.
- **S&P 500 universe bias:** Directly explains the near-zero SMB correlation
  in s3 and the flat SMB cumulative line observed in s3_factor_cumulative.png.
```

Repeat an inline plot here if the constraint's manifestation is best shown visually and Phase 1's textual description doesn't fully capture it.

### Data Plan Verification

Confirm that actual data matches the approved plan from expectations.md:

```markdown
- Universe: 200 tickers requested → 187 downloaded (13 failed) ✓ (within tolerance)
- Date range: 2010-01-01 to 2024-12-31 requested → confirmed in run_log ✓
- Frequency: daily → confirmed ✓
```

### Divergence Summary

A concise handoff list for Step 6 — the biggest gaps between expected and actual, without suggesting teaching strategy:

```markdown
### Divergence Summary

- S3 SMB-KF correlation: criterion < 0.3, actual = 0.04 — far below range floor
- S5 cross-sectional R²: criterion [0.15, 0.45], actual = 0.31 — passes,
  but production benchmark = 0.42
- D1 in-sample t-stats: no characteristic reaches |t| > 1.5
```

Only include items where there's a meaningful gap — criteria that pass comfortably with no benchmark gap don't need listing.

---

## Boundaries

| This step does | This step does NOT do |
|---|---|
| Describe what plots show | Interpret what results mean for teaching |
| Report metrics from run_log.txt | Suggest narrative angles or framing |
| Check criteria pass/fail (Phase 2) | Decide how to teach around divergences |
| Compare to production benchmarks | Resolve open questions (Step 6's responsibility) |
| Flag plot quality issues | Modify code, blueprint, or expectations |

---

## Quality Bar

**Phase 1:**
- [ ] Every plot in `logs/plots/` has a visual observation entry with inline image
- [ ] Every plot cross-checked against run_log.txt metadata (✓/✗ marked)
- [ ] Cross-file consistency checked for repeated metric keys
- [ ] Notable values flagged using domain knowledge (not acceptance criteria)
- [ ] Signal significance recorded for every section with IC stats (t-stat, p-value, warnings)
- [ ] Warnings and unexpected output reported (including ⚠ WEAK SIGNAL, ⚠ DEGENERATE PREDICTIONS, ⚠ OVERFIT, ⚠ SIGNAL DECAY, ⚠ HIGH TURNOVER)
- [ ] Plot quality issues noted

**Phase 2:**
- [ ] Every acceptance criterion from expectations.md has a verdict with margin note
- [ ] Production benchmarks compared with quantified gaps
- [ ] Known constraints connected to specific Phase 1 observations
- [ ] Data plan parameters verified against run_log.txt
- [ ] Divergence summary lists meaningful gaps only
- [ ] Part 1 is unmodified
