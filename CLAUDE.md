# Project: ML for Quantitative Finance — 18-Week Course

## Repo Mind Map

```
finance_stuff/
│
├── CLAUDE.md                      # This file — repo instructions
│
├── .claude/
│   └── skills/
│       └── build-week/            # /build-week skill — orchestrates the full pipeline
│           └── SKILL.md           #   Entry point: parses args, reads orchestrator.md
│
├── pyproject.toml / poetry.lock / poetry.toml / .python-version
│                                  # Python environment (3.12, managed by Poetry)
│
├── nb_builder.py                  # Converts _build_*.md scripts into .ipynb notebooks.
│                                  # Step 7 agents produce markdown with ~~~python fences;
│                                  # orchestrator runs: python3 nb_builder.py <script>
│                                  # Rejects on: consecutive code (lecture), prose ratio, runs >2
│
├── .venv/                         # Virtual environment (not checked in)
│
├── course/
│   │
│   ├── COURSE_OUTLINE.md          # THE active course outline (20 weeks, 4 bands).
│   │                                Approved structure that all content follows.
│   │
│   ├── curriculum_state.md        # Cumulative record of what prior weeks taught.
│   │                                Also contains "Shared Data Infrastructure" section:
│   │                                full inventory of all 48 cached datasets in shared/.
│   │                                Updated after Step 6¾ audit (the ONE place it's written).
│   │
│   ├── guides/                    # How to create course content (v2 pipeline)
│   │   ├── common.md                    # System-level context read by ALL agents
│   │   │                                  (pipeline map, info flow, audience)
│   │   ├── research.md                  # Step 1: domain knowledge gathering
│   │   ├── blueprint_spec.md            # Step 2: ideal teaching plan design
│   │   ├── expectations_spec.md         # Step 3: data reality + acceptance criteria
│   │   ├── task_design.md               # Shared: task types, difficulty, criteria
│   │   │                                  (consumed by Steps 2, 3, 4)
│   │   ├── rigor.md                     # Shared: tiered ML engineering quality standard
│   │   │                                  (consumed by Steps 3, 4, 6¾)
│   │   ├── code_verification.md         # Step 4: plan (4A) + implement (4B) + verify (4C)
│   │   ├── file_agent_guide.md          # Step 4B: code format + implementation rules for file agents
│   │   ├── observation_review.md        # Step 5: fresh-eyes plot + numerical review
│   │   ├── consolidation_spec.md        # Step 6: reconcile ideal vs. actual
│   │   ├── flag_resolution.md           # Step 6½: resolve flags without full rerun
│   │   ├── brief_audit_spec.md          # Step 6¾: adversarial honesty review of brief
│   │   ├── cell_design.md                # Shared: cell sizing, splitting, structure targets
│   │   │                                  (consumed by Steps 4B, 7)
│   │   ├── notebook.md                  # Step 7: prose, layout, voice for .ipynb
│   │   └── orchestrator.md              # Orchestrator guide: step prompts, gate criteria,
│   │                                      context discipline, modes (supervised/autonomous)
│   │
│   ├── research/                  # Why the outline looks the way it does
│   │   ├── COURSE_RESTRUCTURE_REQUIREMENTS.md   # Research methodology & topic tiers
│   │   ├── OUTLINE_SPEC.md                      # Spec that informed the outline
│   │   ├── RESEARCH_SYNTHESIS.md                # Compiled synthesis of all research
│   │   └── raw/                                 # Raw output from 5 research agents
│   │       ├── 1_quant_fund_jobs.md
│   │       ├── 2_bank_am_roles.md
│   │       ├── 3_mfe_syllabi_certifications.md
│   │       ├── 4_community_practitioner_views.md
│   │       └── 5_emerging_trends_wildcards.md
│   │
│   ├── shared/                    # Reusable infrastructure across weeks
│   │   ├── data.py                    # Cross-week data cache layer (48 files, ~143 MB).
│   │   │                                OHLCV prices (456 tickers), FF3/5/6 + Carhart factors,
│   │   │                                Ken French sorted portfolios (25 size/BM, 17/49 industry,
│   │   │                                10 momentum), FRED (24 series incl. full yield curve,
│   │   │                                VIX, macro, credit spreads, USREC), fundamentals
│   │   │                                (455 tickers), crypto (8 tokens). Full inventory in
│   │   │                                curriculum_state.md § "Shared Data Infrastructure".
│   │   ├── .data_cache/               # Shared download cache (gitignored)
│   │   ├── metrics.py                 # IC, rank IC, ICIR, hit rate, R²_OOS, etc.
│   │   ├── temporal.py                # Walk-forward, expanding, purged CV splitters
│   │   ├── evaluation.py              # Rolling cross-sectional prediction harness
│   │   ├── backtesting.py             # Quantile portfolios, long-short, Sharpe, drawdowns
│   │   ├── dl_training.py             # NN fit/predict, SequenceDataset, device support
│   │   ├── kaggle_gpu_runner.py       # Remote GPU execution via Kaggle API
│   │   └── [domain modules]           # portfolio, derivatives, microstructure, regime,
│   │                                    nlp, causal, rl_env
│   │
│   └── weekNN_<topic>/            # 18 week folders (populated via the pipeline)
│       ├── research_notes.md      #   Step 1 output
│       ├── blueprint.md           #   Step 2 output
│       ├── expectations.md        #   Step 3 output
│       ├── code/                  #   Step 4 output
│       │   ├── data_setup.py      #     Shared data downloads + caching (4A)
│       │   ├── .cache/            #     Downloaded data + intermediate caches (gitignored)
│       │   ├── logs/              #     Per-file stdout logs (4B)
│       │   │   ├── notes/        #     Implementation notes from file agents
│       │   │   └── plots/        #     Plot PNGs saved by file agents
│       │   ├── lecture/           #     One .py per lecture section
│       │   ├── seminar/           #     One .py per seminar exercise
│       │   └── hw/                #     One .py per homework deliverable
│       ├── run_log.txt            #   Step 4C output (clean sequential run)
│       ├── execution_log.md       #   Step 4 output (developer report)
│       ├── observations.md        #   Step 5 output
│       ├── narrative_brief.md     #   Step 6 output
│       ├── brief_audit.md        #   Step 6¾ output
│       ├── orchestration.md      #   Orchestrator state log (decisions, gate verdicts)
│       ├── lecture.ipynb          #   Step 7 output
│       ├── seminar.ipynb          #   Step 7 output
│       └── hw.ipynb               #   Step 7 output
│
└── legacy/                        # Previous course structure (pre-restructure)
    ├── COURSE_PLAN.md             # Old 18-week syllabus
    ├── COURSE_DEPENDENCIES.md     # Old dependency graph
    ├── PIPELINE_REDESIGN.md       # Redesign spec that produced the v2 guides
    ├── ML_Quant_Finance_Course_Research.md
    ├── ML_in_Quantitative_Finance_Research.md
    ├── quant_finance_skill_map_2025_2026__from_gpt_research.md
    ├── PHASE1_OPEN_DISCOVERY_RAW_FINDINGS.md
    ├── QUALITY_AUDIT_W1_W4.md
    ├── old_guides/                # v1 guide system (replaced by course/guides/)
    ├── weeks/                     # Old week folders with notebooks & READMEs
    │   └── week01_markets_data/ ... week18_backtesting_capstone/
    └── old_pipeline_weeks/        # Weeks 3-4 built under v1 pipeline (partial)
        ├── week03_factor_models/
        └── week04_ml_alpha/
```

## Key Context

- **Target audience**: ML/DL experts learning finance — do NOT teach ML fundamentals
- **Active outline**: `course/COURSE_OUTLINE.md` is the source of truth for week structure
- **Week build pipeline**: 7-step pipeline (research → blueprint → expectations → code → observation review → consolidation → notebooks) with approval gates. Run via **`/build-week NN TOPIC [supervised|autonomous]`** — skill that orchestrates the full pipeline via Task agents. `course/guides/orchestrator.md` is the authoritative reference for launch prompts, gate criteria, and context discipline. Step-specific guides live in `course/guides/`.
