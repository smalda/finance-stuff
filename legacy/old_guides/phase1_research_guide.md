# Phase 1 Research Guide — Rigorous Investigation Protocol

> **Purpose:** This guide defines the complete methodology for Phase 1 research. It ensures every week's content reflects current best practices, uses accessible tools, and can be implemented without institutional barriers. Phase 1 research is NOT about "finding cool papers" — it's about **verifying implementation feasibility** and **identifying blockers** before Phase 2 README writing begins.

---

## Table of Contents

1. [Overview & Philosophy](#overview--philosophy)
2. [Research Tier Assignment](#research-tier-assignment)
3. [The Verification Principle](#the-verification-principle)
4. [Phase 1 Deliverable: research_notes.md](#phase-1-deliverable-research_notesmd)
5. [Critical Gap Analysis Checklist](#critical-gap-analysis-checklist)
6. [Search Strategy by Tier](#search-strategy-by-tier)
7. [Follow-Up Verification Protocol](#follow-up-verification-protocol)
8. [Readiness Assessment](#readiness-assessment)
9. [Examples & Anti-Patterns](#examples--anti-patterns)

---

## Overview & Philosophy

### What Phase 1 Is

Phase 1 research **verifies implementation feasibility** for the week's content. It answers:
- Can students access the required data **without institutional subscriptions**?
- Are the necessary Python libraries **available and maintained**?
- Do the prerequisites from prior weeks **actually cover what this week needs**?
- Are there **conceptual distinctions** (e.g., Barra vs Fama-French) that must be clarified?
- What are the **quantitative acceptance criteria** for code verification (Phase 3)?

### What Phase 1 Is NOT

- ❌ A literature review (that's for academic papers, not courses)
- ❌ A search for "the latest cool technique" (this creates scope creep)
- ❌ Padding research notes with unverified findings to look thorough
- ❌ Trusting search result snippets at face value

### The Core Principle: "No Blockers"

**The Phase 1 gate question:** "Can Phase 2 README be written with confidence, knowing that Phase 3 code verification will succeed?"

If the answer is "I don't know because I haven't verified [data sources / library availability / prerequisite coverage]," then **Phase 1 is incomplete**.

---

## Research Tier Assignment

Not every week needs the same research depth. Use this tier system to calibrate effort:

| Tier | Weeks | Rationale | Search Budget |
|------|-------|-----------|---------------|
| **HEAVY** | W7 (NLP/LLM), W8 (DL Time Series), W17 (Generative/GNN), W18 (DeFi/Frontier) | Fields evolving monthly; training data likely stale | 6-10 initial searches + unlimited follow-ups |
| **MEDIUM** | W4 (ML Alpha), W10 (Causal Inference), W14 (Stat Arb/Regime), W15 (RL), W16 (Deep Hedging) | Core stable but recent papers/practices matter | 3-5 initial searches + follow-ups as needed |
| **LIGHT** | W1-3, W5-6, W9, W11-13 | Well-established foundations; training data sufficient | 1-3 initial searches + follow-ups for verification |

**Initial search budget** = discovery queries to check for paradigm shifts, new tools, etc.

**Follow-up searches** = verification queries (unlimited) to confirm findings with specifics.

### What to Search For (All Tiers)

Every research phase should answer these questions:

1. **Latest developments (2024-present):** Any paradigm shifts? New model architectures? Regulatory changes?
2. **Current best practices:** What tools/libraries are standard now? What's deprecated?
3. **University course coverage:** What do top programs (Stanford, CMU, Princeton, NYU) teach for this topic? Recent syllabi?
4. **Data availability:** Can students access required data without institutional subscriptions?
5. **Implementation barriers:** Are Python libraries maintained? Any breaking API changes?
6. **Practitioner reality:** What do r/quant, QuantNet, Wilmott discussions reveal? Gap between theory and practice?

---

## The Verification Principle

### Core Rule: Verify Before You Trust

**Every finding must be either verified or dropped. No search result gets recorded at face value.**

Search result summaries are often:
- Vague ("the SEC changed round-lot sizes" — which stocks? what tiers? when?)
- Incomplete ("university course covers factor models" — what depth? which factors?)
- Misleading (paywall papers, dead links, outdated information)

### Verification Standards

A finding is **verified** if ANY of these conditions hold:

1. **Primary source fetched:** You read the actual page (course syllabus, SEC press release, library changelog, paper abstract)
2. **Multiple independent corroborations:** 3+ sources independently confirm the same fact (e.g., "Polars adoption is growing" confirmed across tutorials, blog posts, and GitHub stars)
3. **Executable verification:** You tested it (e.g., `pip install linearmodels` succeeded, library imports work, documentation loads)

A finding is **dropped** if:
- Behind a paywall with no alternative source
- Dead link or 404 error
- Vague snippet with no way to verify specifics
- Single source with no corroboration and unable to fetch

### Example: Week 3 Library Verification

**Initial finding (search snippet):** "famafrench library constructs Fama-French factors"

**Verification questions:**
- Does it require institutional data access?
- Is it free or paid?
- What data sources does it use?
- Is it maintained (last update)?

**Verification steps:**
1. Fetch PyPI page → confirms v0.1.4, May 2020, beta status
2. Fetch GitHub documentation → **discovers WRDS subscription requirement**
3. Search "WRDS cost" → confirms ~$1000s/year institutional subscription
4. Test `pip install famafrench` → installs but unusable without WRDS credentials

**Verified conclusion:** Library requires expensive institutional subscription → creates accessibility barrier → **should not be required in course**.

**If verification had stopped at step 1:** Would have incorrectly recommended inaccessible tool.

---

## Phase 1 Deliverable: research_notes.md

Save all findings in `course/weekNN_topic/research_notes.md` following this structure:

### Required Sections

```markdown
# Week N Research Notes: [Topic]

**Research tier:** HEAVY / MEDIUM / LIGHT
**Date:** [YYYY-MM-DD]
**Queries run (initial research):**
1. [query 1]
2. [query 2]
...

**Follow-up queries (deep verification):**
[N]. [follow-up query 1]
[N+1]. [follow-up query 2]
...

## Key Findings

### Latest Papers & Results
- [Paper/result]: [1-2 sentence summary + URL]
- **Why it matters for this week:** [Actionable implication]

### Current State of the Art
- [What's standard now, what's changed recently]
- [Verified with sources]

### University Course Coverage
- [What top programs cover, with syllabus URLs if found]

### Tools & Libraries
**CRITICAL: Include accessibility analysis for EVERY recommended tool**

[Tool name] ([version, last update])
- **What it does:** [Construction vs. download vs. analysis]
- **Data source:** [Where it gets data]
- **Access requirement:** ⚠️ **REQUIRES [X]** or ✅ **FREE**
- **Verified access pattern:** [URL to documentation confirming requirements]
- **DataFrame support:** [Pandas, Polars, PyArrow]
- **Installation:** [pip command, conda, etc.]
- **Status:** [Maintained, deprecated, beta]
- **Pedagogical value:** [HIGH/MEDIUM/LOW — does it teach mechanics or hide them?]
- **Student accessibility:** [HIGH/MEDIUM/LOW — can self-learners use it?]

### Data Sources for Implementation (CRITICAL SECTION)
**Research question:** Can students build [X] from scratch without institutional subscriptions?

**Answer:** ✅ YES / ❌ NO — [explanation with verified sources]

[For each data type needed:]
- **What it provides:** [Fields, coverage]
- **Access method:** [API, library, manual download]
- **Cost:** FREE / $$ / requires institutional access
- **Limitations:** [Survivorship bias, point-in-time issues, etc.]
- **Pedagogical assessment:** [How limitations become teaching moments]

**Data sufficiency verification:**

| Requirement | Data Needed | Available via [free source]? | Verified How |
|-------------|-------------|------------------------------|--------------|
| [Feature 1] | [Fields] | ✅ Yes ([source]) | [Verification method] |
| [Feature 2] | [Fields] | ✅ Yes ([source]) | [Verification method] |

**Bottom line:** ✅ All requirements can be met with free data / ❌ Requires institutional access

### Technical Implementation (VERIFIED)
[For each technical component:]

**[Component name]** ([library, method])
- **Installation:** [Command]
- **Documentation:** [URL]
- **Key features:** [What it does, API surface]
- **Status:** [Maintained? Last update?]
- **Pedagogical value:** [Manual implementation vs. library usage]

### Conceptual Clarifications (VERIFIED)
[For any ambiguous or commonly confused concepts:]

**Research question:** [What's the distinction between X and Y?]

**Answer:** [Verified explanation with sources]

**Teaching strategy:** [How to present this clearly]

### Prerequisites Verification (Week X & Week Y Content)
**Purpose:** Ensure Phase 2 README accurately reflects what students already know.

**What students learned in Week X (relevant to this week):**
[Bulleted list with specifics]

**What they CAN assume for this week:**
[Foundations that are in place]

**What they CANNOT assume:**
[New material that must be taught from scratch]

**Prerequisites gap analysis:**
✅ [Confirmed prerequisites in place]
🆕 [New material in this week]

### Paradigm Shifts
[Any verified changes in best practices, methodology, or standard approaches]

### Practitioner Reality
[What practitioners actually use vs. academic theory; verified from forums/discussions]

## Implications for This Week's Content

### What Should Be Included (Verified)
[Concrete recommendations with rationale]

### What Should Be Mentioned (But Not Deep-Dived)
[Brief treatment, sidebars, references]

### What Should Be Avoided
[Anti-patterns, outdated approaches, common mistakes]

### Specific Implementation Recommendations
1. **Data sources:** [PRIMARY, VALIDATION, MENTION strategies]
2. **Exercises should include:** [Concrete examples]
3. **Career connections:** [Specific roles and daily tasks]
4. **Acceptance criteria for code verification:** [Quantitative targets]

## References to Include in README
### Core Papers (All Students Should Know)
- [Author (Year)] — [Title]. [Journal]. [Why read this]

### Modern Integration (Optional/Advanced)
- [Recent papers]

### Classic Methodology
- [Foundational references]

## Phase 2 Readiness: Complete Summary

### Research Completeness Assessment
✅ **ALL CRITICAL GAPS VERIFIED:**
1. ✅ [Gap 1 verified]
2. ✅ [Gap 2 verified]
...

### Key Actionable Findings for README Author
[CRITICAL DECISIONS section with numbered recommendations]

### Summary: [Changes Needed / No Major Changes]
**Bottom line:** [1-2 sentences on overall assessment]

**Week N should:**
[Numbered list of teaching recommendations]

**Research confirms:** [Outline calibration assessment]

### Confidence Level: READY FOR PHASE 2
**High confidence** (X/10) that Phase 2 README can be written without further research.

[Bulleted checklist of what's confirmed]

**Phase 2 agent will receive:**
1. COURSE_OUTLINE.md (Week N entry)
2. This research_notes.md (complete findings)
3. Prior week READMEs (prerequisite context)

**No additional research needed.** / **Need to investigate [X] before proceeding.**

---

## Research Audit Trail
**Initial hypothesis:** [What you expected to find]
**Hypothesis confirmed/refuted:** [What you actually found]
**Unexpected finding:** [Surprises that changed recommendations]
**Research quality:** [Self-assessment of rigor]
**Time invested:** [Query count + approximate hours]
**Sources consulted:** [Count + types]
```

### Quality Bar for research_notes.md

- **Specific, not vague:** "Correlation > 0.95 required" not "factors should match"
- **Quantitative where possible:** "SMB mean return ≈ 0.2-0.4% monthly" not "small stocks outperform"
- **URLs for every claim:** Every tool, paper, API, tutorial gets a link
- **Access requirements explicit:** Every tool labeled ⚠️ REQUIRES [X] or ✅ FREE
- **Verification trail visible:** Reader can see how you confirmed each finding
- **Actionable for Phase 2:** README author knows exactly what to specify

---

## Critical Gap Analysis Checklist

Before marking Phase 1 complete, verify ALL of these:

### 1. Data Availability (BLOCKER if not resolved)

- [ ] **Identified all data types needed** (prices, fundamentals, alternative data, etc.)
- [ ] **Verified FREE access** (no institutional subscriptions required)
- [ ] **Tested API/library access** (can actually download data)
- [ ] **Documented limitations** (survivorship bias, point-in-time issues)
- [ ] **Confirmed data sufficiency** (all required fields available)
- [ ] **Estimated download time/size** (students know what to expect)

**If any checkbox is unchecked:** Phase 1 is NOT complete. Data access is a hard blocker.

### 2. Technical Implementation (BLOCKER if not resolved)

- [ ] **Verified library availability** (on PyPI, conda, maintained)
- [ ] **Tested installation** (`pip install [library]` works)
- [ ] **Confirmed API is functional** (not deprecated, breaking changes documented)
- [ ] **Checked for institutional requirements** (WRDS, Bloomberg, etc.)
- [ ] **Documented alternatives** (if primary tool has barriers)
- [ ] **Identified manual implementation path** (if libraries are problematic)

**If any checkbox is unchecked:** Phase 1 is NOT complete. Technical blockers will halt Phase 3.

### 3. Prerequisites (IMPORTANT but not blocker)

- [ ] **Read prior week READMEs** (Week N-1, Week N-2 if relevant)
- [ ] **Listed what students already know** (specific skills, not "they know data science")
- [ ] **Identified new material** (what must be taught from scratch)
- [ ] **Confirmed no circular dependencies** (Week N doesn't require Week N+1 content)

**If unchecked:** Phase 2 README may specify wrong prerequisites or redundant content.

### 4. Conceptual Clarity (IMPORTANT for teaching quality)

- [ ] **Identified ambiguous terminology** (e.g., Barra vs Fama-French)
- [ ] **Verified distinctions with sources** (not just personal understanding)
- [ ] **Determined teaching strategy** (which to emphasize, how to contrast)
- [ ] **Documented common confusions** (from forums, student questions)

**If unchecked:** Phase 2 README may perpetuate confusion or omit critical distinctions.

### 5. Acceptance Criteria (IMPORTANT for Phase 3)

- [ ] **Specified quantitative targets** (correlation > 0.95, t-stat > 2.0, R² = 10-30%)
- [ ] **Grounded in historical data** (not arbitrary thresholds)
- [ ] **Included validation strategy** (how to compare student implementation to ground truth)
- [ ] **Documented realistic performance** (out-of-sample degradation expected)

**If unchecked:** Phase 3 code verification lacks clear success criteria.

### 6. Teaching Recommendations (IMPORTANT for content quality)

- [ ] **Identified paradigm shifts** (cross-sectional vs time-series, etc.)
- [ ] **Documented best practices** (what industry actually uses)
- [ ] **Specified what to avoid** (deprecated tools, outdated approaches)
- [ ] **Connected to career paths** (which roles use these skills daily)

**If unchecked:** Phase 2 README may teach outdated or irrelevant content.

---

## Search Strategy by Tier

### LIGHT Tier (1-3 initial queries)

**Goal:** Confirm stability, identify any recent changes, verify tool status.

**Query strategy:**
1. `"[topic]" new developments 2025 2026 finance`
2. `"[primary library]" python update 2025 2026`
3. `"[topic]" course syllabus [top university] 2025 2026`

**Expected outcome:** "No major changes since training data" is a VALID finding for LIGHT tier.

**Follow-up verification:**
- If library found: Fetch PyPI page, check last update, verify installation works
- If data source found: Verify free access, test API, confirm fields available
- If conceptual ambiguity found: Fetch authoritative source, clarify distinction

**Week 3 example (Factor Models):**
- Initial: 5 queries on paradigm shifts, tools, university courses
- Finding: Cross-sectional paradigm shift (needs verification)
- Follow-up: 5 verification queries on data sources, library access, Fama-MacBeth implementation
- **Total: 10 queries, ~90 minutes** — appropriate for LIGHT tier with critical implementation dependencies

### MEDIUM Tier (3-5 initial queries)

**Goal:** Identify recent papers, check for methodological shifts, verify practitioner adoption.

**Query strategy:**
1. `"[topic]" latest papers finance 2025 2026`
2. `"[topic]" best practices quant finance`
3. `"[topic]" python library 2025 update`
4. `"[topic]" course [top university]`
5. `site:reddit.com/r/quant "[topic]"`

**Follow-up verification:**
- Fetch at least 2-3 recent papers (abstracts + key findings)
- Verify library versions, API stability
- Test data access for new sources mentioned

### HEAVY Tier (6-10 initial queries)

**Goal:** Comprehensive landscape scan, identify SOTA, verify cutting-edge tools.

**Query strategy:**
1. `"[topic]" survey paper 2025 2026`
2. `"[topic]" state of the art 2026`
3. `"[specific model/tool]" vs [alternative] finance 2025`
4. `"[topic]" course syllabus [university] 2025 2026` (multiple universities)
5. `"[topic]" python library 2025`
6. `site:arxiv.org "[topic] financial" 2025`
7. `site:reddit.com/r/quant "[topic]"`
8. `"[topic]" practitioner implementation 2025`
9. `"[topic]" production deployment challenges`
10. `"[topic]" benchmark datasets 2025`

**Follow-up verification:**
- Fetch 5+ papers (at least abstracts)
- Test 3+ libraries for installation and basic functionality
- Verify claims with multiple independent sources

---

## Follow-Up Verification Protocol

### When to Do Follow-Up Searches

Trigger a follow-up verification search when:

1. **Access requirements unclear:** "Library X provides data" → need to verify free vs. paid
2. **Version/maintenance status unknown:** "Library Y is available" → need to check last update
3. **Implementation details vague:** "Method Z is standard" → need to find actual code examples
4. **Conflicting information:** Source A says X, Source B says Y → need tie-breaker
5. **Quantitative claims unsupported:** "Factors have 0.5% monthly returns" → need historical data
6. **Conceptual ambiguity:** "Barra vs Fama-French" → need authoritative distinction

### Follow-Up Search Patterns

**Pattern 1: Library Deep Dive**
```
Initial: "factor models python library 2025"
Finding: "famafrench library available"
Follow-ups:
  1. WebFetch PyPI page for famafrench
  2. WebFetch GitHub documentation for access requirements
  3. Search "WRDS subscription cost academic"
  4. Search "free alternative to CRSP Compustat"
Outcome: Library requires $$ institutional access → find free alternative
```

**Pattern 2: Data Source Verification**
```
Initial: "free fundamental data API python"
Finding: "yfinance provides balance sheet data"
Follow-ups:
  1. WebFetch yfinance tutorial for API details
  2. Test yfinance installation: pip install yfinance
  3. Test data fetch: ticker.balance_sheet
  4. Verify fields: check if book value, assets, equity available
Outcome: Confirms yfinance sufficient for factor construction
```

**Pattern 3: Conceptual Clarification**
```
Initial: "Barra risk model vs Fama French"
Finding: "Both are factor models" (vague)
Follow-ups:
  1. Search "Barra methodology cross-sectional regression"
  2. Search "Fama French portfolio sorting approach"
  3. WebFetch AnalystForum discussion on difference
  4. WebFetch academic paper comparing both
Outcome: Clear distinction — portfolio-sorting vs. cross-sectional regression
```

**Pattern 4: Quantitative Target Setting**
```
Initial: "Fama French factor returns historical"
Finding: "HML has positive returns historically" (vague)
Follow-ups:
  1. Search "HML factor mean return 1926-2023"
  2. WebFetch Ken French data library for actual numbers
  3. Calculate mean/std from downloaded data
  4. Verify with academic paper citing same statistics
Outcome: HML mean ≈ 0.4% monthly, vol ≈ 12% annualized
```

### Verification Depth Guidelines

**Minimum verification (acceptable):**
- 2 independent sources agree
- At least 1 source is authoritative (official docs, academic paper, top university)
- No contradictory evidence found

**Strong verification (preferred):**
- 3+ independent sources agree
- At least 1 primary source fetched and read
- Executable verification performed (tested library, downloaded data, ran code)

**Insufficient verification (not acceptable):**
- Single search snippet, no corroboration
- Vague claims without specifics
- Contradictory sources, no resolution attempt

---

## Readiness Assessment

### Phase 1 is COMPLETE when:

✅ All checklist items verified (data, tools, prerequisites, acceptance criteria)
✅ research_notes.md written with all required sections
✅ Confidence level ≥ 8/10 that Phase 2 README can be written
✅ No known blockers for Phase 3 code implementation
✅ All claims traceable to specific sources (URLs provided)

### Phase 1 is INCOMPLETE when:

❌ Data access uncertain ("probably free but didn't verify")
❌ Library requirements unknown ("seems to work but didn't check WRDS requirement")
❌ Conceptual ambiguities unresolved ("not sure about Barra vs FF distinction")
❌ Acceptance criteria vague ("code should work reasonably well")
❌ Prerequisites not verified ("assume students know regression")

### The Gate Question

**"If I delegated Phase 2 README writing to an agent RIGHT NOW, with only COURSE_OUTLINE.md + research_notes.md + prior READMEs, would they have everything needed to specify:**
1. **What data to use and where to get it?** (Yes → proceed / No → investigate data sources)
2. **What libraries to use and how to install them?** (Yes → proceed / No → verify tools)
3. **What prerequisites students have from prior weeks?** (Yes → proceed / No → read prior READMEs)
4. **What quantitative targets code verification should hit?** (Yes → proceed / No → set acceptance criteria)
5. **What conceptual distinctions to clarify?** (Yes → proceed / No → resolve ambiguities)

**If all 5 are "Yes":** Phase 1 is ready.
**If any are "No":** Continue research.

---

## Examples & Anti-Patterns

### Example 1: Week 3 Factor Models (GOOD)

**Initial finding:** "Use famafrench library to access Ken French data"

**Rigorous verification:**
1. Fetched PyPI page → saw v0.1.4, May 2020, beta status
2. Fetched GitHub docs → **discovered WRDS subscription requirement** ($1000s/year)
3. Searched for free alternative → found `getfactormodels` (uses public data)
4. Verified yfinance provides fundamentals → confirmed book value, market cap available
5. Tested all 3 options → documented access requirements explicitly

**Outcome:** Changed recommendation from "use famafrench" to "build from scratch with yfinance, validate with getfactormodels, mention famafrench as institutional tool"

**Why this is GOOD:**
- Discovered accessibility barrier that would have blocked self-learners
- Found free alternative that works
- Documented all 3 options with clear access requirements
- Teaching strategy respects accessibility principle

### Example 2: Hypothetical Bad Research (ANTI-PATTERN)

**Initial finding:** "Transformers are revolutionizing factor models (2025)"

**Insufficient verification:**
- Cited one ArXiv paper, didn't fetch it
- Assumed it means "teach transformers in Week 3"
- Didn't verify if transformers are in practitioner use (they're not for basic factor models)
- Didn't check if transformer implementation is accessible to students

**What this would cause:**
- Phase 2 README specifies transformer implementation
- Phase 3 code verification tries to build transformer factor model
- Complexity explosion, no clear acceptance criteria
- Misalignment with course goal (teach fundamentals, not cutting edge)

**How rigorous research would fix this:**
1. Fetch the paper → see it's academic research, not practitioner adoption
2. Search "transformer factor models production use" → find minimal adoption
3. Verify course outline → Week 3 is LIGHT tier, fundamentals focus
4. **Correct conclusion:** Mention transformers as Week 4 preview, don't implement in Week 3

### Example 3: Data Source Verification (GOOD)

**Research question:** "Can students build Fama-French factors without WRDS?"

**Rigorous verification:**
1. Listed required data: market cap, book value, earnings, price history
2. Verified yfinance API: `ticker.info['marketCap']`, `ticker.balance_sheet`, price history
3. Tested on 5 tickers → confirmed all fields present
4. Built sufficiency table mapping each factor to data fields
5. Documented limitations (survivorship bias, no point-in-time fundamentals)

**Outcome:** ✅ Confirmed FREE data is sufficient, documented pedagogical benefits of limitations

**Why this is GOOD:**
- Specific, testable verification
- Table format makes sufficiency clear
- Limitations acknowledged (not hidden)
- Accessibility ensured

### Example 4: Vague Acceptance Criteria (ANTI-PATTERN)

**Bad specification:** "Factors should match Ken French factors reasonably well"

**Problems:**
- What's "reasonably well"? Correlation 0.7? 0.9? 0.99?
- How to verify this in code?
- What if they don't match — is that a bug or a data difference?

**Good specification (after rigorous research):**
```markdown
**Factor construction validation:**
- Correlation with Ken French official factors > 0.95
- Mean returns have economically sensible signs:
  - SMB (size): historical mean ≈ 0.2-0.4% monthly
  - HML (value): historical mean ≈ 0.3-0.5% monthly
- Volatilities are realistic: annualized vol ≈ 10-15% for each factor
```

**Why this is GOOD:**
- Quantitative thresholds (0.95, 0.2-0.4%, 10-15%)
- Grounded in historical data (not arbitrary)
- Verifiable in code (can compute correlation, mean, vol)
- Allows for data differences (0.95, not 0.99)

---

## Meta: Improving This Guide

This guide should evolve as we build more weeks. After each Phase 1:

**What went well?** → Document as example in this guide
**What was unclear?** → Add clarification section
**What did we miss?** → Add to checklist
**What verification pattern worked?** → Add to Follow-Up Protocol

The research methodology should get MORE rigorous over time, not less.

---

## Summary: The Phase 1 Contract

**Input:** COURSE_OUTLINE.md entry for Week N
**Process:** Rigorous research with verification
**Output:** research_notes.md with ALL critical gaps resolved
**Deliverable:** Phase 2 agent can write README with confidence

**Time estimate by tier:**
- LIGHT: 1-2 hours (3-5 queries + verifications)
- MEDIUM: 2-3 hours (5-8 queries + verifications)
- HEAVY: 3-5 hours (10-15 queries + deep verifications)

**Quality over speed:** Better to spend an extra 30 minutes verifying than to create a blocker for Phase 3.

---

**Last updated:** 2026-02-17 (after Week 3 Factor Models research)
**Next review:** After completing 3 more weeks, assess if methodology needs refinement
