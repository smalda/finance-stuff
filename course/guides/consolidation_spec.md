# Consolidation Specification — Step 6 Guide

> **Consumer:** Consolidation agent (Step 6). You run in a **fresh context**. Your job is to reconcile the ideal teaching plan (blueprint) with what the code actually produced. You triangulate three layers — what we wanted to teach, what we expected to see, and what actually happened — and produce the definitive teaching plan for notebook agents. You never modify upstream artifacts.

---

## Inputs

In addition to `common.md`:
- **`blueprint.md`** from Step 2 — narrative arc, structure, creative vision. The ideal teaching week.
- **`expectations.md`** from Step 3 — acceptance criteria, production benchmarks, known constraints, open questions. What we predicted given our data reality.
- **`execution_log.md`** from Step 4 — implementation analysis, design choices, open question resolutions. The code agent's engineering judgment.
- **`observations.md`** from Step 5 — visual and numerical observations (Phase 1), criteria audit, benchmark comparison, and divergence summary (Phase 2). The verified analytical record.

**`observations.md` is your primary analytical reference.** Phase 2 already cross-checked run_log.txt against expectations.md — criteria verdicts, benchmark gaps, constraint manifestations, divergence summary. Start categorization from the **Divergence Summary** in observations.md Phase 2 — it was designed as a handoff to this step. All specific metrics you need are in observations.md — Phase 1 transcribes values from run_log.txt with explicit ✓/✗ cross-checks.

## Outputs

1. **`narrative_brief.md`** in the week folder. Immutable once approved.

   This is the **authoritative teaching plan**. When blueprint and brief conflict on data-dependent content, the **brief wins**. Notebook agents follow the brief for: how to frame results, which numbers to cite, how to present divergences, and what story to tell.

**Note:** `curriculum_state.md` is updated by the orchestrator after the Step 6¾ audit — not by this agent. Do not write to it.

---

## The Process

```
1. Read all inputs in full
2. Check execution_log.md's Viability Concerns section for any
   ⚠ VIABILITY CONCERN or ⚠ PEDAGOGICALLY FLAT markers — these are
   sections whose pedagogical value is already in question from Step 4C
3. Write the narrative arc reconciliation
4. For each section/exercise/deliverable:
   a. Triangulate: blueprint intended → expectations predicted → observations found
   b. If section has a viability concern from Step 4C: apply extra scrutiny —
      does the concern hold up after seeing observations.md?
   c. Categorize (Matches | Data limitation | Real phenomenon | Expectation miscalibration)
   d. If a resolved open question affects this section: incorporate the finding
   e. If issue found: flag it (no teaching angle for flagged sections)
   f. Otherwise: write the full section entry
5. If any flags exist: write the flags appendix, mark brief as INCOMPLETE
6. If no flags: mark brief as COMPLETE
```

**Flagged sections get no teaching angle entry.** Writing a teaching angle for buggy or unusable data produces unreliable guidance. Flags are resolved downstream — via Step 6½ (targeted in-place patches; see `flag_resolution.md`) or a full rerun loop (Steps 4→5→6 producing a fresh consolidation), depending on scope.

---

## Targeted Web Search

The expectations agent (Step 3) sets production benchmarks *before* code runs. But some results diverge in ways Step 3 didn't anticipate — and accurate categorization often depends on knowing what production data would show for *this specific result*. You may run targeted web searches to fill these gaps.

**When to search:**

- A divergence needs categorization (Data limitation vs. Real phenomenon) and `expectations.md` lacks a production benchmark for that specific metric. E.g., HML turns negative — is that a sandbox artifact or the documented "death of value"? A search for "HML negative post-2020 Fama French" resolves it.
- An open question was resolved by the code in a surprising way, and you need external context to write a credible teaching angle. E.g., cross-sectional R² came in far higher than expected — is this a known property of the specific time period?
- You need a citable production benchmark to populate the **Key numbers** field in a teaching angle. E.g., the brief needs "production = X (Author, Year)" but expectations.md only provides a range.

**When NOT to search:**

- Broad domain research — that's Step 1's job. Don't survey the field.
- Data source investigation or library evaluation — that's Step 3's job.
- Searching for information that's already in your inputs. Check `expectations.md` and `research_notes.md` (via expectations.md's citations) before searching.
- Confirming things you can categorize confidently from existing inputs. If the constraint is already documented in expectations.md and the gap is cleanly explained, don't search — just categorize.

**Rules:**

- **Cite what you find.** Any benchmark or finding from a search goes into the teaching angle's **Key numbers** field with author/year/dataset, just like expectations.md benchmarks.
- **Note the source.** In the narrative brief entry, mark searched benchmarks: "(searched: [query summary] → Author, Year)". This distinguishes them from expectations.md benchmarks for audit purposes.
- **Don't overdo it.** Most sections will categorize cleanly from existing inputs. Expect 0–3 searches per week, not one per section.

---

## narrative_brief.md — Format

### Header

```markdown
# Narrative Brief — Week N: [Title]

**Status:** COMPLETE | INCOMPLETE (N flags pending resolution)
```

### Narrative Arc Reconciliation

The blueprint defines a narrative arc (setup → complication → resolution). Actual results may reshape it. This section gives notebook agents the big picture before per-section details.

```markdown
## Narrative Arc

**Blueprint arc:**
- Setup: [from blueprint]
- Complication: [from blueprint]
- Resolution: [from blueprint]

**How results reshape it:**
[1-2 paragraphs. Which parts confirmed? Which shifted? If the complication
landed differently than expected, describe the adjusted emphasis.]

**Adjusted arc** (only if changed):
- Complication: [revised]
- Resolution: [revised]
```

If results fully confirm the blueprint's arc: "Results confirm the intended arc. No adjustment needed."

### Per-Section Entries

One entry per lecture section, seminar exercise, and homework deliverable. Same structure and ordering as the blueprint.

```markdown
### Section N: [Title — from blueprint]

**Blueprint intended:** [1-2 sentences: what the blueprint says this section
teaches or demonstrates]
**Expected:** [Key acceptance criteria + production benchmark from expectations.md]
**Actual:** [Key metrics from observations. Cite specific values.]
**Category:** Matches | Data limitation | Real phenomenon | Expectation miscalibration
**Confidence:** [omit if confident] low — [1 sentence: why you're unsure]

**Teaching angle:**
- **Frame as:** [1 sentence: how to introduce this result to students]
- **Key numbers:** [Values for notebook prose — our result, production benchmark,
  gap magnitude. Format: "ours = X, production = Y (Author, Year)"]
- **Student takeaway:** [1 sentence: the one thing students should remember]

**Sandbox vs. reality:** [Non-Matches only. 1-2 sentences: why our sandbox
result differs from production, citing the specific constraint from
expectations.md. Notebook agents use this for the "here's what a fund sees"
framing.]
```

**For Matches:** omit the "Sandbox vs. reality" field.

### The Confidence Marker

Most sections should be categorized with full confidence — omit the Confidence field entirely. Use `Confidence: low` only when you wrote a teaching angle but aren't sure the section earns its place. This is the gray zone between "clearly teachable" and "clearly broken" (flag).

**When to mark low confidence:**
- The categorization relies on a plausible but not clearly evidenced structural explanation
- The result technically passes acceptance criteria but feels too weak to anchor a teaching moment on
- You considered flagging but talked yourself out of it — the doubt itself is the signal
- The teaching angle works only if students don't ask probing follow-up questions

**When NOT to mark low confidence:**
- A clear Data limitation with documented constraint, known benchmark, and straightforward explanation — that's full confidence
- A result comfortably within acceptance criteria — full confidence
- A section you would flag if the flag format weren't more work — be honest and flag it instead

**What happens downstream:** Low-confidence sections are an explicit audit target for Step 6¾. The auditor will apply extra scrutiny to these sections — checking whether the categorization is justified or whether the section should have been flagged. This is cheaper for the pipeline than a false flag (which triggers Step 6½) but still ensures scrutiny on borderline cases.

**The quality test:** An INCOMPLETE brief with honest flags and a COMPLETE brief with honest low-confidence markers are both higher-quality outputs than a COMPLETE brief where borderline sections are categorized with false confidence. Do not treat low confidence as a failure — treat false confidence as a failure.

### Example Entry (Data Limitation)

```markdown
### Section 3: Factor Construction

**Blueprint intended:** Build SMB and HML from our data, compare against
Ken French published factors to demonstrate factor replication.
**Expected:** SMB-KF correlation < 0.3 (expectations.md); production
benchmark: 0.75+ (Fama-French 1993, CRSP full universe).
**Actual:** SMB-KF correlation = 0.04. Cumulative SMB nearly flat.
HML declining from 2017 onward, ending negative.
**Category:** Data limitation

**Teaching angle:**
- **Frame as:** Our S&P 500 "small" stocks are the market's large stocks —
  we're measuring large-vs-very-large, not small-vs-large.
- **Key numbers:** ours = 0.04, production = 0.75+ (Fama-French 1993, CRSP),
  smallest S&P 500 stock ≈ $10B market cap.
- **Student takeaway:** Factor replication depends critically on the universe.
  S&P 500 can't produce a real size factor.

**Sandbox vs. reality:** S&P 500 excludes stocks below ~$10B market cap.
The SMB factor requires true small-caps (NYSE/AMEX/NASDAQ full universe via
CRSP) to produce meaningful size dispersion. The near-zero correlation is a
direct consequence of universe bias, not a code error.
```

### Open Questions Resolved

Resolved open questions from `execution_log.md` are incorporated into the relevant section entries. Additionally, list them for cross-reference:

```markdown
## Open Questions Resolved

1. **[Question text from expectations.md]**
   **Finding:** [From execution_log.md]
   **Affects:** [Which section(s)]
   **Teaching implication:** [How this shapes the section's teaching angle]
```

---

## Divergence Categories

Every section falls into exactly one category. The category determines how notebook agents frame the result.

| Category | Meaning | How to identify | Notebook framing |
|----------|---------|----------------|-----------------|
| **Matches** | Code confirms expectations | Criteria pass within expected range; benchmark close or N/A | Teach straightforwardly |
| **Data limitation** | Sandbox can't replicate production | Criteria pass but far from benchmark; known constraint explains gap | "Here's what we got → here's what a fund sees → here's why" |
| **Real phenomenon** | The result IS production reality | Matches benchmark; result is counterintuitive or dramatic | "Your code is correct. This IS what happens." |
| **Expectation miscalibration** | Prediction was wrong, code is fine | Outside expected range but makes domain sense | Teach the actual result; note gap only if instructive |

**Distinguishing similar categories:**

- **Matches vs. Data limitation:** Check the production benchmark gap in observations.md Phase 2. Small gap with no manifested constraint → Matches. Large gap explained by a known constraint → Data limitation.
- **Data limitation vs. Real phenomenon:** Would institutional data (CRSP, Bloomberg) produce a different result? If yes → Data limitation. If institutional data shows similar numbers → Real phenomenon.
- **Expectation miscalibration vs. Code bug:** Can you explain WHY the result differs from the prediction using domain logic? If yes → miscalibration (write a teaching angle). If the result defies domain logic → possible code bug (flag it).

**Distinguishing "weak but real" from "learned nothing":**

For sections with predictive models, observations.md Phase 1 records signal significance (IC t-stat, p-value) and prediction spread (⚠ markers). Use these to classify:

| IC t-stat | Signal viability (from expectations.md) | Classification |
|-----------|----------------------------------------|----------------|
| ≥ 1.96 | Any | Signal is statistically significant — categorize normally using the four categories above |
| < 1.96 | Low (anticipated) | **Data limitation** — frame as: proper methodology demonstrated, significance limited by sample size. Use IC sign consistency (pct_positive) as secondary evidence of directional learning |
| < 1.96 | Moderate / High (not anticipated) | **Investigate.** Either the code has a problem, or expectations misjudged the viability. Flag as possible code bug or expectation miscalibration — do NOT narrate around unexpected insignificance |
| < 1.96 | Not assessed (no signal viability field) | **Flag as expectation gap.** The expectations agent should have assessed signal viability for this section. Treat with suspicion — do not write a teaching angle claiming the model learned a real signal |

**Special case — ⚠ DEGENERATE PREDICTIONS:** If observations.md records a `⚠ DEGENERATE PREDICTIONS` warning (spread ratio < 0.10), the model is outputting near-constant values. **Flag as pedagogically unusable** regardless of other metrics. A model that doesn't differentiate stocks has no teaching value — even if IC happens to be non-zero through random correlation of near-constant predictions with varying actuals.

**Special case — ⚠ SIGNAL DECAY:** If observations.md records a `⚠ SIGNAL DECAY` warning, the signal may be regime-dependent. If expectations.md didn't anticipate non-stationarity, flag for investigation — the signal might only work in certain market conditions, which changes the teaching angle substantially.

**Special case — ⚠ OVERFIT with high parameter-to-observation ratio:** If `⚠ OVERFIT` appeared alongside a model with many parameters relative to training observations, this is strong evidence of overfitting. Do not narrate around it — flag it or categorize as Expectation miscalibration and teach the overfitting itself.

**The critical rule:** You can only teach "weak signal is the domain reality" if expectations.md's signal viability annotation predicted it would be weak. Post-hoc rationalization of unexpected weakness is prohibited. If a model was expected to learn and didn't, that's a problem to flag — not a narrative to spin.

---

## Teaching Angle by Category

### Matches
Brief and direct. "Frame as" states the concept plainly. Key numbers are straightforward. No "Sandbox vs. reality" field needed. Don't over-explain working results.

### Data limitation
The most common non-match in this course. The teaching angle must: (1) present our result honestly, (2) name the production benchmark with citation, (3) explain the structural reason for the gap, (4) make the limitation itself a teaching moment.

**"Frame as" pattern:** "[What we observe] because [structural reason]. With [production data], you'd see [benchmark]. The gap shows [insight about data quality]."

### Real phenomenon
Counterintuitive results that ARE correct. Lean into the surprise.

**"Frame as" pattern:** "You might expect [intuitive wrong answer]. Here's what actually happens: [result]. This is because [domain explanation]."

### Expectation miscalibration
Use the actual result as the starting point, not the prediction. Only reference the original prediction if the gap is itself pedagogically useful.

---

## The Flag Protocol

You are the last *authoring* gate before the brief audit (Step 6¾) and notebooks. Every section must produce material that is genuinely useful, factually consistent, and pedagogically sound. If a section can't justify its place in the week — if a nitpicky professor of finance would raise an eyebrow and ask "why are we teaching this?" — flag it. A weak exercise that slips through here becomes a weak notebook that wastes students' time.

Flags identify issues that require user attention before the brief can be approved. **Only the user authorizes reruns.** You diagnose, present options, and recommend.

### When to Flag

1. **Code bug:** Result defies domain logic in a way only explainable by implementation error. E.g., correlation = 0.97 where the mathematical maximum given the setup is ~0.4.

2. **Wrong implementation approach:** Code implements something different from the blueprint's intent, eliminating the teaching insight. E.g., blueprint says "compare Ridge vs. OLS" but code only implements OLS.

3. **Pedagogically unusable:** Result is so extreme or meaningless that no teaching angle can salvage it. E.g., all-zero coefficients, zero-variance factor, regression that doesn't converge. **Check execution_log.md's Viability Concerns section** — any section flagged there with `⚠ VIABILITY CONCERN` or `⚠ PEDAGOGICALLY FLAT` by Step 4C is a strong candidate for this flag. The viability concern may or may not hold up after seeing observations.md, but it warrants extra scrutiny.

4. **Internal contradiction in observations:** `observations.md` contains internally inconsistent values — e.g., Phase 1 reports "R² = 0.31" but the Phase 2 criteria audit uses 0.13 for the same metric, or a visual observation contradicts the numerical observation for the same section. A teaching angle built on contradictory facts is worse than no teaching angle.

5. **Cross-section contradiction:** Results across two or more sections send conflicting messages that would confuse students. E.g., Section 2 shows "value premium is positive" but Section 5 shows "HML is negative" with no time-period or methodological explanation connecting them. Flag when the contradiction lacks an obvious reconciliation — if you can't explain it in a sentence, students can't either.

6. **Fragile pass:** A result passes its acceptance criterion but sits so close to the boundary that a slightly different data sample (different time period, ±20 tickers, different random seed) would likely flip it. An exercise that barely works teaches nothing reliably. Flag it so the user can decide whether to adjust the approach, widen the data, or accept the fragility as the explicit teaching point.

7. **Unanticipated signal failure:** A predictive model's IC is not statistically significant (t < 1.96) AND expectations.md's signal viability was Moderate or High (or absent). The model may not have learned anything real. This is distinct from pedagogically unusable (#3) — the model may produce non-trivial predictions, but we can't distinguish its IC from zero. Flag it so the user can decide whether to widen the data, simplify the model, or reframe the section's teaching goal around the insignificance itself.

8. **Other concern:** Something feels wrong but doesn't fit the categories above. Trust your judgment — if you can articulate why a section is problematic, flag it even if the problem is novel. Describe the concern clearly in the diagnosis; the user will decide what to do. This is not a lazy bucket — you must still explain what's wrong and why it matters.

### When NOT to Flag

- A clear Data limitation with a known structural explanation and a production benchmark to contextualize it — categorize and teach it
- A result comfortably within acceptance criteria that is merely weaker than the ideal — the criteria were designed with this range in mind

When in doubt, flag. A flag the user dismisses costs one minute of review. A bad section that reaches a notebook costs hours of student confusion.

### Flag Format

```markdown
## Flags for User

### FLAG N: [Section(s)] — [Code Bug | Wrong Approach | Pedagogically Unusable | Internal Contradiction | Cross-Section Contradiction | Fragile Pass | Unanticipated Signal Failure | Other Concern]

**Diagnosis:** [What's wrong, factually]
**Evidence:** [Specific metrics from observations showing the issue]
**Affected teaching:** [What blueprint insight is lost or broken]

**Options:**
A. [Option with tradeoff]
B. [Option with tradeoff]
C. [Option — include "teach as extreme case" or "accept with explicit fragility framing" if viable]

**Recommendation:** [Which option and why]
```

**If ANY flags exist:**
- Brief status = INCOMPLETE
- Flagged sections have NO entry in the per-section list
- Resolution: the orchestrator resolves flags via Step 6½ (targeted patches — see `flag_resolution.md`) or, for large-scope changes (>40% of code files, fundamentally different approach), via the full loop (Steps 4→5→6). Observation-only errors correct `observations.md` and rerun Step 6. Contradictions and fragile passes may resolve via narrative reframing without code changes — the user decides

**After the user approves the brief (COMPLETE only):** Append the `curriculum_state.md` entry for this week. Base it on what the brief says was actually taught — categories, teaching angles, and resolved open questions — not on the blueprint's original plan. If the brief is INCOMPLETE (has flags), do NOT append the entry — Step 6½ handles it after flag resolution.

---

## Boundaries

| This step does | This step does NOT do |
|---|---|
| Categorize divergences between expected and actual | Modify code, blueprint, or expectations |
| Decide teaching angles per section | Write notebook prose (Step 7) |
| Produce the authoritative teaching plan | Fix code or rerun scripts |
| Flag bugs and pedagogically unusable results | Authorize reruns (user decides) |
| Incorporate resolved open questions | Invent new analyses or exercises |
| Write narrative arc reconciliation | Redesign curriculum structure |
| Cite specific numbers for notebook agents | Create .ipynb files |
| Targeted web search for missing benchmarks or categorization | Broad domain research (Step 1) or data investigation (Step 3) |

---

## Quality Bar

- [ ] Status marked COMPLETE or INCOMPLETE with flag count
- [ ] Narrative arc reconciliation written — adjustments noted or "no adjustment needed"
- [ ] Every section/exercise/deliverable from blueprint has either a full entry or a flag (not both)
- [ ] Every entry has all template fields: Blueprint intended, Expected, Actual, Category, Teaching angle
- [ ] Categories use only the 4 defined types
- [ ] Low-confidence markers used honestly — borderline sections marked rather than categorized with false confidence
- [ ] Non-Matches entries include "Sandbox vs. reality" field
- [ ] Teaching angles include: Frame as, Key numbers, Student takeaway
- [ ] Key numbers cite specific values from observations.md, not approximate guesses
- [ ] Production benchmarks cited with author/year where expectations.md provides them
- [ ] Resolved open questions incorporated into relevant section entries and listed in summary
- [ ] Observations.md checked for internal consistency — no contradictions between Phase 1 and Phase 2 values left unflagged
- [ ] Cross-section consistency verified — no unreconciled contradictions between sections
- [ ] No fragile passes left unflagged — if a different data sample could flip the result, it's flagged
- [ ] Signal significance checked for every predictive section — IC t-stat from observations.md compared against expectations.md signal viability
- [ ] No unanticipated signal failures narrated around — if IC is insignificant and expectations didn't predict it, section is flagged
- [ ] Degenerate predictions (⚠ markers in observations.md) flagged as pedagogically unusable
- [ ] Viability concerns from execution_log.md (⚠ VIABILITY CONCERN / ⚠ PEDAGOGICALLY FLAT) reviewed — each either flagged or justified as resolved by observations
- [ ] Flags (if any) include: Diagnosis, Evidence, Affected teaching, Options, Recommendation
- [ ] Flagged sections have no teaching angle entry
- [ ] No code, no notebook prose, no upstream modifications
