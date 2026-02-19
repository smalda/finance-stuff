# Week 04: ML for Alpha — From Features to Signals — Orchestration Log
## Mode: autonomous
## Started: 2026-02-17

---

### Step 1: RESEARCH — APPROVED
- **Artifact:** research_notes.md
- **Summary:**
  - Identified Gu-Kelly-Xiu (2020) as canonical reference; surveyed Chen-Pelger-Zhu (2024), Kelly-Malamud-Zhou (2023) extensions
  - Recommended LightGBM as primary gradient boosting library, PyTorch for neural nets, shap for interpretability
  - Data plan: Week 3 feature_matrix_ml.parquet as primary input, yfinance for additional features, Ken French for benchmarks
  - Documented gradient boosting vs. neural network practitioner gap (GBM dominates production, DL dominates research)
  - 29 verification queries, 5 findings dropped for failed verification
- **Gate assessment:** All criteria met — data sources accessible, tool recommendations present, no blockers
- **Decision:** APPROVED

### Step 2: BLUEPRINT — APPROVED
- **Artifact:** blueprint.md
- **Summary:**
  - 7 lecture sections (cross-sectional setup, signal evaluation, gradient boosting, NN comparison, feature engineering, signal-to-portfolio, alternative data)
  - 4 seminar exercises (IC autopsy, feature knockout, complexity ladder, turnover tax)
  - 3 homework deliverables (AlphaModelPipeline class, feature engineering lab, model comparison report)
  - Narrative arc: ML expert's intuitions about model complexity are systematically challenged by cross-sectional finance reality
  - Concept matrix: 14 concepts, no overlapping exercises across notebooks
- **Gate assessment:** All criteria met — no numerical predictions, proper UNDERSTAND→APPLY→BUILD flow, scope matches outline
- **Decision:** APPROVED

### Step 3A: DATA PLAN — APPROVED
- **Artifact:** expectations.md (Data Plan section only)
- **Summary:**
  - Ran 9 search gaps, 8 web searches, 7 code probes
  - Universe: 179 tickers (Week 3 feature matrix), Date Range: 2014-03 to 2024-12 (130 months), Frequency: monthly cross-sections
  - 69 OOS months with 60-month train window → t-stat ~3.1 at IC=0.03 (sufficient for significance)
  - 8 failure modes documented with contingencies; 8 known constraints including PIT contamination, survivorship bias, small feature set
  - Rigor: Tier 2 (teaches signal significance, prediction quality, baseline significance, feature preprocessing)
- **Gate assessment:** All criteria met — dimensions specified with tradeoffs, failure modes & contingencies present, no INFEASIBLE sections, no inaccessible sources
- **Decision:** APPROVED — using recommended options for all dimensions

### Step 3B: FULL EXPECTATIONS — APPROVED
- **Artifact:** expectations.md (complete — data plan + per-section expectations)
- **Summary:**
  - 7 lecture sections: S1 (cross-sectional setup), S2 (signal evaluation), S3 (GBM alpha), S4 (NN vs trees), S5 (feature engineering), S6 (signal to portfolio), S7 (alternative data — conceptual only)
  - 4 seminar exercises: Ex1 (IC autopsy by VIX regime), Ex2 (feature knockout/substitution), Ex3 (complexity ladder — 5 models), Ex4 (turnover tax at 5/20/50 bps)
  - 3 homework deliverables: D1 (AlphaModelPipeline class), D2 (feature expansion to 15-25 features), D3 (model comparison report with CIO recommendation)
  - ML methodology specified for 8 sections (S3, S4, S5, Ex2, Ex3, D1, D2, D3): walk-forward, 60-month train window, HP search grids, 1-month prediction horizon
  - Signal viability: Moderate for all predictive sections (69 OOS months, t~3.1 at IC=0.03)
  - Pedagogical alternatives provided for S4 (degenerate NN fallback) and Ex2 (permutation importance if retraining too slow)
  - 3 open questions (Category A): GBM significance, NN degeneracy, SHAP ranking PIT contamination
  - Production benchmarks: GKX (2020), Kelly-Malamud-Zhou (2023), Frazzini-Israel-Moskowitz (2018), Daniel & Moskowitz (2016), Lopez de Prado (2018)
- **Gate assessment:** All criteria met — ranges are approximate and two-sided, ML methodology fields present for all training sections, benchmarks cited with author/year/dataset, open questions use ABC taxonomy, no INFEASIBLE sections
- **Decision:** APPROVED

### Step 4: CODE — APPROVED
- **Note:** Step 4 artifacts (code_plan.md, code/, run_log.txt, execution_log.md) were completed in a prior session. 21 plots generated, 0 ✗ in run_log.txt (1558 lines). Orchestration log not updated at the time — marking retroactively.

### Step 5a: OBSERVATION (Phase 1) — APPROVED
- **Artifact:** observations.md (Part 1)
- **Summary:**
  - 21 plot visual observations with inline images, descriptions, and run_log cross-checks (all ✓, no mismatches)
  - Cross-file consistency verified across 7 dimensions (feature matrix shape, forward return stats, OOS months, stocks/month, GBM IC, missing rates, survivorship)
  - 14 notable values flagged: GBM overfitting ratio 6.73, GBM does not significantly outperform naive baseline (p=0.57), GBM and NN indistinguishable (p=0.99), feature expansion degraded IC by 0.020, monthly turnover 0.79, LightGBM depth-8 significantly underperforms depth-3
  - Signal significance recorded for 8 sections (S2, S3, S4, S5, Ex1, Ex3, D1, D2, D3)
  - Warnings documented: OVERFIT (S3), HIGH TURNOVER (Ex4), PIT WARNING (D2), matplotlib cosmetic warnings
- **Gate assessment:** All criteria met — every plot referenced, numerical observations from run_log, no information leak (no references to expectations/execution_log/.py files), thorough quality
- **Decision:** APPROVED

### Step 5b: OBSERVATION (Phase 2) — APPROVED
- **Artifact:** observations.md (Part 2 appended, Part 1 unmodified)
- **Summary:**
  - Acceptance criteria audit across all 13 sections (S1-S7, Ex1-Ex4, D1-D3)
  - 3 criteria failures: S3 rank IC gap (0.020 > 0.01, structurally infeasible threshold), Ex3 OLS IC (0.060 > 0.04 ceiling), Ex3 Ridge IC (0.061 > 0.05 ceiling)
  - 2 boundary cases: D2 IC change at -0.0203 (lower boundary), Ex4 breakeven 60.6 bps (above 50 bps ceiling, positive divergence)
  - 13 production benchmark comparisons (GKX 2020, Kelly-Malamud-Zhou 2023, Frazzini-Israel-Moskowitz 2018, practitioner consensus)
  - 7 known constraints connected to Phase 1 observations (survivorship bias, PIT contamination, static fundamentals, small feature set, large-cap efficiency, limited fundamental history, VIX circularity)
  - Data plan verified: all parameters confirmed (179 tickers, 2014-03 to 2024-12, monthly, 68 OOS months)
  - 10-item divergence summary for Step 6 handoff
- **Gate assessment:** Part 1 unmodified, every criterion has verdict with margin notes, production benchmarks quantified, constraints connected to observations, divergence summary meaningful (not empty)
- **Decision:** APPROVED

### Step 6: CONSOLIDATION — APPROVED
- **Artifact:** narrative_brief.md
- **Summary:**
  - Status: COMPLETE — no flags raised
  - 17 sections categorized: 8 Matches, 4 Data limitation, 3 Expectation miscalibration, 2 Data limitation (homework)
  - Narrative arc confirmed and strengthened — "more features/depth can actively hurt on small universes" added as sharpened complication
  - 3 open questions all resolved: GBM IC significant (t=2.15), NN non-degenerate, technicals dominate SHAP
  - No low-confidence markers on any section
  - 3 additional notebook agent observations documented (D1 vs S6 Sharpe discrepancy, Ex2 baseline IC discrepancy, Ex4 quintile vs decile)
- **Gate assessment:** Status field present (COMPLETE), every section has full entry (Blueprint intended/Expected/Actual/Category/Teaching angle), categories use only the 4 defined types, no Confidence: low markers, non-Matches include Sandbox vs. reality, key numbers cite specific run_log values, production benchmarks cited with author/year, open questions resolved and incorporated, no flags → Step 6½ not needed
- **Decision:** APPROVED — proceeding to Step 6¾

### Step 6½: FLAG RESOLUTION — SKIPPED
- Brief Status was COMPLETE with no flags — Step 6½ not needed.

### Step 6¾: BRIEF AUDIT — APPROVED
- **Artifact:** brief_audit.md
- **Summary:**
  - 12 findings total: 2 critical, 6 moderate, 4 minor
  - Overall assessment: Adequate
  - Critical: F1 (D3 omits individual model significance — none reach 5% on expanded features), F2 (Ex3 omits per-model t-stats — LightGBM_d8 not significant)
  - Moderate: F3 (S4 NN borderline significance language), F4 (D1 vs S6 Sharpe cross-ref), F5 (Ex2 baseline IC caveat), F6 (S6 excess kurtosis), F7 (S3 HP grid divergence), F8 (Ex1 VIX circularity)
  - Minor: F9 (S5 IC change lacks p-value), F10 (D2 static ratios vs PIT lag), F11 (Ex4 quintile vs decile), F12 (D2 category → Real phenomenon)
  - All viability verdicts: Viable (no Reframe/Redesign/Remove)
- **Gate assessment:** All 12 findings are prose edits to narrative_brief.md — no code changes required. No Reframe/Redesign/Remove verdicts. All findings applied to narrative_brief.md. curriculum_state.md updated with final audited Week 4 entry.
- **Decision:** APPROVED

### Step 7: NOTEBOOKS — PENDING
