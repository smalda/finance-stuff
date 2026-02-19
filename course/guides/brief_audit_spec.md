# Brief Audit Specification — Step 6¾ Guide

> **Consumer:** Brief audit agent (Step 6¾). You run in a **fresh context**. Your job is to stress-test the narrative brief — the document that notebook agents will follow as their authoritative teaching plan. You read what was predicted (expectations), what the code chose to do (execution_log), what actually happened (observations), and how the brief framed it all. You look for gaps in honesty, transparency, and conservatism — and you assess whether each section still earns its place in the week. You never modify the brief — you produce an audit report. The orchestrator decides what to fix.

---

## Why This Step Exists

The consolidation agent (Step 6) is incentivized to produce a complete, teachable brief. This creates a structural bias: when results are better than expected, the brief tends to celebrate rather than interrogate. When the code sidesteps a constraint that expectations.md carefully identified, the brief may not notice or may frame the sidestepping as a positive outcome. When a methodology introduces bias, the brief may absorb the biased result into its teaching angles without disclosure.

The consolidation agent can mark sections as `Confidence: low` when it's unsure whether the categorization is honest. **These are your priority audit targets** — the agent is telling you "I wrote a teaching angle but I'm not confident this section earns its place." Apply extra scrutiny to every low-confidence section: is the categorization justified, or should this have been flagged? A low-confidence section that survives your audit is fine. One that doesn't should become a finding.

This step exists because **the writer of a document is the worst auditor of its intellectual honesty.** A separate agent, reading expectations.md first and the brief last, can catch what the consolidation agent's completionist bias obscures.

The completionist bias has a second effect: the consolidation agent will almost never recommend that a section be removed or fundamentally reimagined. It would rather write a teaching angle with heavy caveats than admit the section has nothing meaningful to teach. This audit step is the place where that judgment gets made — not just "is the brief honest?" but "given honest disclosure, does the section still earn its place?"

---

## Inputs

In addition to `common.md`:

- **`COURSE_OUTLINE.md`** — the full course structure. Read this early, alongside rigor.md. Use it to understand what adjacent weeks cover, what students bring into this week, and what later weeks depend on from this week. This informs your viability verdicts — it does NOT give you license to question the week's topic assignment or suggest moving content between weeks.
- **`rigor.md`** (shared guide) — the tiered ML engineering quality standard. Read this before expectations.md. It defines what production-grade methodology looks like at each tier. Without it, you can only check whether the brief is honest about what expectations.md said — with it, you can also check whether the brief is honest about the *quality tier the code claims to satisfy*.
- **`expectations.md`** from Step 3 — acceptance criteria, production benchmarks, known constraints, data plan, risks, interactions, open questions. Internalize every risk, concern, interaction, and open question before you read anything else.
- **`execution_log.md`** from Step 4 — implementation choices, design decisions, how the code agent resolved ambiguities. This is where methodology divergences from expectations.md's assumptions live.
- **`observations.md`** from Step 5 — verified factual record: visual observations (Phase 1), acceptance criteria audit, benchmark comparison, divergence summary (Phase 2). This is ground truth about what the code produced.
- **`narrative_brief.md`** from Step 6/6½ — the document being audited. **Read this last**, after you have built your own understanding of what happened and what concerns exist.
- **`blueprint.md`** from Step 2 — the original teaching vision. Used to check whether intent shifts between blueprint and brief are acknowledged.

**Reading order matters.** Read COURSE_OUTLINE.md and rigor.md first (structural context), then the rest in the order listed above. The goal is to form your own assessment of "what concerns should the brief address?" before seeing how the brief actually addresses them. If you read the brief first, you anchor to its framing and lose the ability to notice what's missing.

### The Role of Each Input

| Input | What you use it for |
|-------|-------------------|
| `COURSE_OUTLINE.md` | Course-level context for viability assessment. What do adjacent weeks teach? What do students already know entering this week? What later weeks depend on topics covered here? Use this to judge whether removing or redesigning a section creates a course-level gap — not just a within-week gap. |
| `rigor.md` | The quality standard. Extract the tier assigned in expectations.md, then use the tier's rules as a checklist: did the code satisfy them, and does the brief honestly represent whether it did? |
| `expectations.md` | Extract every risk, concern, interaction, open question, and data plan assumption. These form your audit checklist. |
| `execution_log.md` | Identify implementation choices that diverge from expectations.md's assumptions. These are the methodology gaps the brief must disclose. |
| `observations.md` | Ground truth. The divergence summary (Phase 2) is a checklist: did the brief address every item? Phase 1 observations are a factual anchor: does the brief accurately represent what happened? |
| `narrative_brief.md` | The thing you're auditing. Check it against everything above. |
| `blueprint.md` | The original intent. If the brief's teaching angles have drifted from the blueprint's vision, is the drift acknowledged? |

---

## Output

**`brief_audit.md`** in the week folder.

This is a structured audit report. It does not modify the brief. The orchestrator reads it, decides which findings to act on, and either edits the brief directly or requests the consolidation agent to revise.

---

## The Seven Checks

Apply these checks systematically across every lecture section, seminar exercise, and homework deliverable. Not every check will produce a finding for every section — but every check must be applied.

### Check 1: Methodology Transparency

**Question:** Did the code agent make implementation choices that differ from what expectations.md assumed? If so, does the brief disclose these choices and their implications?

**How to apply:**
1. Read execution_log.md for each section. Note design choices, shortcuts, simplifications.
2. Compare against expectations.md's data plan and assumptions for that section.
3. Check whether the brief mentions the divergence and explains its impact.

**Example finding:** expectations.md assumed annual rebalancing limited to the fundamental window (~36-48 months). execution_log.md shows the code used a static sort with the most recent fundamentals applied over the full 11-year window (131 months). The brief reports the 131-month result without noting the look-ahead bias this introduces.

**What makes it a finding:** The brief presents a result whose quality is partly attributable to a methodological shortcut, without disclosing the shortcut or its implications.

### Check 2: Divergence Honesty

**Question:** For each section where actual ≠ expected, is the brief's explanation for the divergence supported by evidence in the inputs, or is it post-hoc rationalization?

**How to apply:**
1. Use observations.md Phase 2 divergence summary as your list of divergences.
2. For each: read the brief's explanation.
3. Ask: is the causal claim ("this happened because X") documented in expectations.md, observations.md, or execution_log.md? Or did the brief invent the explanation?
4. Pay special attention when results are *better* than expected. Does the brief ask "why is this better?" or does it just celebrate?

**What makes it a finding:** The brief asserts a causal explanation that isn't grounded in the input artifacts — especially when that explanation conveniently makes the result look good.

### Check 3: Expectations Regression

**Question:** For each risk, concern, interaction, and constraint explicitly raised in expectations.md — was it addressed in the brief? Did any silently disappear?

**How to apply:**
1. Extract from expectations.md:
   - Every item in the "Risk" fields of the data plan
   - Every entry in "Interaction Analysis"
   - Every entry in "Failure Modes & Contingencies"
   - Every item in "Known Constraints"
   - Every "Statistical Implications" warning
2. For each: search the brief for any mention or acknowledgment.
3. Mark items as: addressed / partially addressed / silently dropped.

**What makes it a finding:** A risk or concern that expectations.md raised — especially one labeled "High" likelihood — that the brief never mentions. This doesn't mean the brief must have a paragraph for every risk; but risks that manifested (per observations.md) or were sidestepped (per execution_log.md) must be acknowledged.

### Check 4: Open Question Integrity

**Question:** For each open question in expectations.md, does the brief's resolution answer the question *as originally posed*? Or does it answer a different question because the code took a different path?

**How to apply:**
1. Read each open question in expectations.md carefully. Note its exact scope and the conditions it describes.
2. Read the "Open Questions Resolved" section of the brief.
3. Check: does the brief's "Finding" actually address the scenario the question described? Or did the code sidestep the scenario, making the question moot — in which case the brief should say "not tested as framed" rather than claiming resolution.

**Example finding:** Open question asks "will factors show signal given short window (~36-48 months) and large-cap universe?" The code used a static sort producing 131 months, eliminating the short-window constraint. The brief resolves the question as "Yes, with notable asymmetry" — but the question's premise (short window) was never tested.

**What makes it a finding:** The brief claims to resolve a question that was never actually tested under the conditions the question specified.

### Check 5: Conservative Framing

**Question:** When results are better than expected or when the brief presents a positive teaching angle, does it distinguish genuine signal from methodological artifact? Does it partition the result into "what's real" vs. "what's flattered by our approach"?

**How to apply:**
1. Identify every section where the actual result exceeds the upper bound of the expected range.
2. For each: does the brief explore *why* the result is better than predicted?
3. Check whether the "Sandbox vs. reality" field (required for non-Matches) honestly states what production would show.
4. Look for teaching angles that frame a result as purely positive when the inputs suggest it's partly an artifact.

**What makes it a finding:** A result that exceeds expectations and is presented without questioning whether the methodology contributed to the overshoot. The brief should be conservative — if in doubt, disclose the possible artifact and let the notebook add nuance, rather than omit the concern and let the notebook teach a misleading number.

### Check 6: Cross-Document Consistency

**Question:** Do the brief's claims align with what observations.md and execution_log.md actually say? Are there numbers, framings, or claims in the brief that contradict or go beyond what the upstream artifacts document?

**How to apply:**
1. Spot-check key numbers in the brief against observations.md Phase 2 criteria audit.
2. Check whether the brief's category assignments (Matches / Data limitation / Real phenomenon / Expectation miscalibration) are consistent with the evidence in observations.md.
3. Verify that the brief's "Actual" fields match observations.md's documented values.
4. Check the divergence summary in observations.md — is every item addressed in the brief?

**What makes it a finding:** A factual discrepancy between the brief and its source artifacts, or a divergence item from observations.md that the brief ignores.

### Check 7: Signal Integrity

**Question:** For every section where a predictive model was trained, does the brief honestly report whether the signal is statistically real?

**How to apply:**
1. From observations.md Phase 1, extract every section's IC t-statistic, p-value, and any ⚠ WEAK SIGNAL or ⚠ DEGENERATE PREDICTIONS warnings.
2. From expectations.md, extract the signal viability annotation (High / Moderate / Low) for each predictive section.
3. For each predictive section in the brief, check:

   a. **Does the brief report the t-statistic and p-value?** Statistical significance must be stated alongside IC in the "Actual" and "Key numbers" fields. Omitting the t-stat while citing the IC point estimate is a finding.

   b. **When t-stat < 1.96, does the brief use appropriate language?** The brief should NOT use language implying the model "learned," "captured," or "detected" a pattern when the signal is not statistically significant. Acceptable: "the model produces a positive but statistically insignificant IC of 0.03 (t=0.9, p=0.37)." Unacceptable: "the model captures a weak momentum signal (IC=0.03)."

   c. **When t-stat < 1.96, does the brief cite the signal viability annotation from expectations.md?** If expectations.md annotated signal viability as Low and explained why, the brief should reference this: "consistent with the low signal viability anticipated in expectations.md given 24 monthly observations." If there was no annotation and the brief doesn't flag the gap, that's a finding.

   d. **When ⚠ DEGENERATE PREDICTIONS appeared, does the brief acknowledge it?** A model outputting near-constant predictions should never be described as producing meaningful results. If the brief writes a positive teaching angle for a degenerate model, that's a critical finding.

   e. **When the model "beats" a baseline, is the improvement statistically significant?** The brief should not claim "model X outperforms baseline Y" if the paired comparison lacks statistical significance. Check whether the brief distinguishes significant from insignificant improvements.

   f. **When ⚠ SIGNAL DECAY appeared, does the brief acknowledge non-persistence?** A signal that degrades substantially in the second half of the OOS period is regime-dependent. If the brief presents the full-period IC without mentioning the decay, that's a finding — students need to know the signal may not persist.

   g. **When ⚠ HIGH TURNOVER appeared, does the brief address implementability?** A strategy with turnover exceeding 50% per period has substantial cost drag. If the brief presents Sharpe ratios without discussing turnover and net-of-cost performance, that's a finding.

**What makes it a finding:** A brief that presents statistically insignificant predictions as real signal, omits t-statistics that observations.md recorded, frames degenerate predictions as meaningful results, or claims model superiority without significance testing. Severity is Critical when the framing would lead students to believe a non-existent signal is real; Moderate when significance context is merely incomplete.

---

## Viability Assessment

After applying all seven checks, step back and assess each section's **pedagogical viability** — whether it can still teach what the blueprint intended, given everything the findings revealed.

The seven checks ask "is the brief honest?" The viability assessment asks a different question: **"Given honest disclosure of all issues, does this section still earn its place in the week?"**

### When findings compound

A single moderate finding rarely threatens viability. But findings compound. Three moderate findings on the same section — methodology shortcut, post-hoc rationalization, insignificant signal — may collectively mean the section now demonstrates something fundamentally different from the blueprint's intent. The viability assessment is where you surface these compound effects that no individual finding captures.

### The four verdicts

Assess every section, exercise, and deliverable — not just those with findings. Sections with no issues get **Viable**, which proves the assessment was comprehensive.

| Verdict | Meaning | Signal to orchestrator |
|---------|---------|----------------------|
| **Viable** | Section teaches what the blueprint intended. Honest framing (applying finding fixes) is sufficient. | Apply finding fixes; no structural changes needed. |
| **Reframe** | Section produces something *valuable but different* from the blueprint's intent. The results are real, but the teaching angle needs to be reconceived — not just caveated. | Consolidation agent rewrites the section's teaching angle around what the results actually demonstrate. |
| **Redesign** | Results are too compromised for the current approach, but the *topic* deserves coverage. The section needs a fundamentally different exercise, methodology, or data approach. | Significant rework — may require returning to code/expectations level. Orchestrator decides scope. |
| **Remove** | Results are degenerate, empty, or so compromised that no feasible redesign within this week's scope would produce meaningful teaching content. | Orchestrator drops the section and considers whether its time budget should be redistributed. |

### Decision heuristic

For each section, ask these questions in order:

1. **After applying all suggested finding fixes, would the section still teach what the blueprint intended?** → If yes: **Viable**.
2. **Do the actual results teach something different but genuinely valuable?** → If yes: **Reframe**. Specify what the section now actually demonstrates and what the new teaching angle should target.
3. **Is the topic worth teaching but the current approach fundamentally unable to deliver it?** → If yes: **Redesign**. Specify what went wrong with the approach and what a redesign would need to change (different data, different methodology, different scope).
4. **Is there no feasible path to teaching this topic within the week's constraints?** → If yes: **Remove**.

### Week-level narrative consideration

Do not assess sections in isolation. When recommending Reframe, Redesign, or Remove, check the blueprint and COURSE_OUTLINE.md for:

- **Within-week dependencies:** Does a later section in this week build on this one's output or concepts? If so, that section is also affected — state this explicitly.
- **Cross-week dependencies:** Does a later week (per COURSE_OUTLINE.md) depend on concepts introduced here? Removing a topic from week 5 that week 9 assumes students know is a course-level problem, not just a week-level problem.
- **Concept coverage:** Is this the only place — in the week *and* in the course — where students encounter a key concept? Check adjacent weeks in the outline. If the concept appears elsewhere, removal is lower-stakes. If this is the sole coverage point, flag it.
- **Time redistribution:** If a section is removed, its time budget is freed. Note whether another section could benefit from expansion, or whether a simpler replacement exercise could fill the slot.

State these considerations explicitly in your verdict justification. The orchestrator needs this context to make structural decisions.

---

## Output Format

```markdown
# Brief Audit — Week N: [Title]

## Summary

- **Findings:** N total (X critical, Y moderate, Z minor)
- **Viability:** N Viable, N Reframe, N Redesign, N Remove
- **Overall assessment:** Conservative | Adequate | Optimistic | Misleading

## Findings

### Finding 1: [Short descriptive title]

**Check:** [Which of the 7 checks: Methodology Transparency | Divergence Honesty |
  Expectations Regression | Open Question Integrity | Conservative Framing |
  Cross-Document Consistency | Signal Integrity]
**Severity:** Critical | Moderate | Minor
**Section(s):** [Which brief section(s) are affected]

**The expectations document said:**
[Exact quote or precise paraphrase with section reference]

**What happened:**
[What the code did (from execution_log.md) and/or what results showed
(from observations.md)]

**How the brief handled it:**
[What the brief says — quote or paraphrase]

**The problem:**
[Why the brief's handling is insufficient — be specific about what's
misleading, missing, or not conservative enough]

**Suggested fix:**
[Concrete: what the brief should add, change, or reframe. Include draft
language where possible so the orchestrator can apply it directly.]

---

[...additional findings...]

## Viability Assessment

| Section | Verdict |
|---------|---------|
| L1: [title] | Viable |
| L2: [title] | Reframe |
| S1: [title] | Viable |
| S2: [title] | Redesign |
| H1: [title] | Viable |

[For each non-Viable verdict, add an expanded block:]

### [Section identifier]: [Verdict]

**Blueprint intent:** [What the section was supposed to teach — cite blueprint.md]
**What the results actually demonstrate:** [What honest disclosure leaves you with]
**Why framing alone is insufficient:** [The gap between intent and reality that caveats can't bridge]
**Recommended direction:** [For Reframe: what the new teaching angle should target.
For Redesign: what the approach needs to change and why.
For Remove: why no redesign is feasible within scope.]
**Downstream impact:** [Sections that depend on this one, concept coverage gaps,
time redistribution considerations. Or "None — section is self-contained."]

---

## Observations Divergence Checklist

[List every item from observations.md Phase 2 Divergence Summary.
For each: "Addressed in brief ✓" or "Not addressed ✗ — see Finding N"]

## Expectations Regression Checklist

[List every High-likelihood risk, every interaction, and every open question
from expectations.md. For each: "Addressed ✓" / "Partially addressed ~"
/ "Silently dropped ✗ — see Finding N"]

## Low-Confidence Section Review

[For each section marked `Confidence: low` in the brief:
"Section N: [title] — Categorization justified ✓ / Should be flagged ✗ — see Finding N"
If no low-confidence sections exist: "None marked."]

## Sections with No Issues

[List sections that passed all 7 checks. This proves the audit was thorough,
not just negative.]
```

### Overall Assessment Scale

| Rating | Meaning |
|--------|---------|
| **Conservative** | Brief consistently acknowledges limitations and distinguishes signal from artifact. No critical findings. |
| **Adequate** | Brief handles most divergences honestly. Minor gaps exist but no misleading framings. |
| **Optimistic** | Brief tends to celebrate results without questioning methodology. Multiple moderate findings. Students would get an incomplete picture. |
| **Misleading** | Brief presents methodologically flawed results as genuine without disclosure. Critical findings present. Must fix before Step 7. |

### Severity Levels

| Severity | Meaning | Action |
|----------|---------|--------|
| **Critical** | Brief would mislead students or notebook agents about the reliability or meaning of a result. Teaching angles built on this framing produce incorrect understanding. | Must fix before Step 7 |
| **Moderate** | Brief omits important context that students should have. The teaching angle isn't wrong, but it's incomplete in a way that matters. | Should fix; orchestrator decides |
| **Minor** | Brief could be more precise, transparent, or conservative, but isn't materially wrong. A careful reader wouldn't be misled. | Fix if convenient |

---

## Boundaries

| This step does | This step does NOT do |
|---|---|
| Audit the brief's handling of divergences and methodology | Modify the narrative brief |
| Check whether expectations.md concerns were addressed | Write teaching angles or narrative prose |
| Identify missing disclosures and insufficient caveats | Modify code, observations, or expectations |
| Verify factual consistency across artifacts | Authorize brief revisions (orchestrator decides) |
| Produce a structured audit report with specific findings | Run web searches or verify domain claims |
| Flag open questions that were answered under different conditions than posed | Design replacement exercises (orchestrator + consolidation agent's job) |
| Assess each section's pedagogical viability given the findings | |
| Recommend Reframe, Redesign, or Remove when findings compromise a section's teaching purpose | |
| Flag downstream dependencies and narrative gaps when recommending structural changes | |

---

## Quality Bar

- [ ] All 7 checks applied to every section/exercise/deliverable
- [ ] Every finding includes: check type, severity, affected sections, quotes from expectations and brief, the problem, and a suggested fix
- [ ] Viability assessment covers every section/exercise/deliverable — no section skipped
- [ ] Every non-Viable verdict includes: blueprint intent, what results actually demonstrate, why framing is insufficient, recommended direction, and downstream impact
- [ ] Observations divergence checklist is complete — every item from Phase 2 divergence summary accounted for
- [ ] Expectations regression checklist covers all High-likelihood risks, all interactions, and all open questions
- [ ] Low-confidence section review complete — every `Confidence: low` section assessed for categorization validity
- [ ] Sections with no issues are listed (proves thoroughness)
- [ ] Overall assessment rating assigned with justification
- [ ] Suggested fixes are concrete enough for the orchestrator to apply directly
- [ ] No modifications to any upstream artifacts
- [ ] Reading order followed: COURSE_OUTLINE + rigor → expectations → execution_log → observations → brief → blueprint
