---
name: build-week
description: "Build a course week through the full 7-step pipeline. Orchestrates Task agents for each step with approval gates. Supports supervised (human approves) and autonomous (AI approves) modes."
argument-hint: "NN TOPIC [mode] — e.g., 05 backtesting supervised"
user-invocable: true
---

# Build Week — Orchestrator Skill

You are the **orchestrator** for the week-build pipeline. You launch Task agents for each step, review their output, make gate decisions, and log everything. You do NOT do the heavy lifting yourself.

## Parse Arguments

Arguments: `$ARGUMENTS`

Parse:
- **NN** = week number (zero-padded, e.g., `05`)
- **TOPIC** = folder slug (snake_case, e.g., `backtesting`)
- **MODE** = `supervised` (default if omitted) or `autonomous`

The week folder is: `course/week{NN}_{TOPIC}/`
Look up the full week title from `course/COURSE_OUTLINE.md` (search for `Week {N}:` where N is the number without leading zero).

## Before Starting

1. Read the orchestrator guide: `course/guides/orchestrator.md` — this is your complete reference for launch prompts, gate criteria, and mode behavior. Follow it exactly.

2. Check if `course/week{NN}_{TOPIC}/orchestration.md` already exists:
   - **If yes**: you are resuming. Read it, find the last completed step, announce where you're resuming from, and continue.
   - **If no**: create the week folder (if needed) and initialize `orchestration.md` with the header.

3. Check if `course/curriculum_state.md` exists (affects Step 2 and Step 6 prompts).

**CRITICAL: Read the "Context Discipline" section in orchestrator.md and obey it. You must NEVER read guide files meant for agents (common.md, research.md, blueprint_spec.md, etc.), code files, or plot PNGs. Those are for the Task agents. Your only guide is orchestrator.md.**

## Execution

Follow the orchestrator guide step by step. For each step:

1. Construct the launch prompt from the guide, substituting `{NN}`, `{TOPIC}`, and `{TITLE}`
2. Launch a `Task(general-purpose)` with that prompt
3. When the Task returns, read the produced artifact
4. Apply the gate (supervised → ask user; autonomous → evaluate criteria yourself)
5. Update `orchestration.md` with the result
6. Proceed to next step (or stop on HARD STOP)

### Special Cases

- **Step 5** has two phases — two separate Task launches with a gate between them
- **Step 6½** is NOT a Task — you execute it directly if Step 6 status is INCOMPLETE. Read `course/guides/flag_resolution.md`.
- **Step 7** launches 3 parallel Tasks, then you run `nb_builder.py`:
  ```
  for f in course/week{NN}_{TOPIC}/_build_*.py; do python3 nb_builder.py "$f" && rm "$f"; done
  ```

## Context Management

Your context will fill up over a full pipeline run. To manage this:
- After each gate, write your decisions to `orchestration.md` BEFORE proceeding
- When reading artifacts for gate review, focus on the gate criteria — don't read entire files line by line unless needed
- If you feel context pressure, note in `orchestration.md` where you are, then tell the user: "Context is getting long. You can start a new session and say 'continue building week {NN}' — I'll resume from orchestration.md."