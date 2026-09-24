# Autonomous Factor Research Loop

A constrained, memory-guided research agent that proposes economically motivated factor
recipes, executes them through a deterministic point-in-time engine, evaluates them under
fixed temporal validation, and uses structured evidence to guide later generations.

The implementation adapts Huang and Fan (2026),
[_Beyond Prompting: An Autonomous Framework for Systematic Factor Investing via Agentic AI_](references/2603.14288v1.pdf),
and the memory-driven Macro/Micro/Cross design in
[_XALPHA: A Memory-Driven AI Quant Researcher for Hypothesis-to-Code Alpha Discovery_](references/2607.08332v2.pdf),
plus the resource-aware search and strict feedback/test split in
[_AutoScientist-Quant: Self-Evolving Coding Agents for Automatic Research in Quantitative Investment_](references/2608.28632v2.pdf),
to the shorter 2010-2016 Anthelion price, fundamental, and news dataset. The project-specific
rules are documented in the [instruction](Anthelion_Autonomous_Factor_Research_Instruction.pdf)
and the [XALPHA V1 implementation brief](references/Anthelion_Autonomous_Factor_Researcher_V1_Codex_Instructions.md).

## Architecture

```text
src/
├── core/                  # DataContract, ResearchPlan, memory, outcome, artifact schemas
├── brains/
│   ├── macro.py           # Budget-driven IMPROVE/COMBINE/PIVOT/STOP planning
│   ├── micro.py           # Constrained mutation, crossover, and refinement
│   └── cross.py           # GOOD/BAD lessons and cross-cycle memory updates
├── data/
│   ├── loader.py          # Extracted CSV schemas, chunked news loading, source metadata
│   ├── point_in_time.py   # Availability-date joins and conservative news timing
│   ├── quality.py         # Key, overlap, year, and availability invariants
│   └── preprocess.py      # Universe filters and physical research/holdout split
├── factors/
│   ├── expression.py      # Immutable V2 AST, canonicalization, hashes, and static validation
│   ├── operator_registry.py # Authoritative DSL operator catalog and implementations
│   ├── schema.py          # FactorSpec grammar plus legacy-to-AST migration
│   ├── primitives.py      # Historical price, volume, volatility, and news primitives
│   ├── transforms.py      # Date-local and group-local cross-sectional transforms
│   └── builder.py         # Deterministic AST/FactorSpec execution; no generated Python
├── evaluation/
│   ├── integrity.py       # Leakage, availability, complexity, and coverage gates
│   ├── ic.py              # Daily rank IC, ICIR, hit rate, and t-statistic
│   ├── portfolio.py       # Quantile monotonicity, long-short returns, costs, turnover
│   ├── validation.py      # Walk-forward folds and holdout firewall
│   ├── redundancy.py      # Signal correlation and residual/incremental IC
│   └── evaluator.py       # One common evaluator for every candidate
├── quality/
│   ├── static_checks.py   # Forbidden syntax, feature contract, and complexity checks
│   ├── dynamic_leakage.py # Truncation and future-noise causality tests
│   └── alignment.py       # Hypothesis/mechanism/formula tri-alignment
├── selection/
│   ├── gates.py           # Separate RETIRED/PARENT/ELITE thresholds
│   ├── archive.py         # Bounded parent pool and persistent elite archive
│   └── library.py         # Dedupe, correlation, residual-IC, diversity pruning
├── memory/
│   └── store.py           # Append-only checkpoints and persistent freeze seal
├── research/
│   ├── generator.py       # Generation-0 seeds and cross-family exploration
│   ├── llm_generator.py   # Structured LLM proposals with local whitelist validation
│   ├── failures.py        # Stable failure taxonomy for gates, memory, and prompts
│   ├── failure_policy.py  # Distinct next-round responses to integrity failures
│   ├── comparison.py      # Search-efficiency and stability comparison metrics
│   ├── mutator.py         # Bounded exploitation around promoted parents
│   ├── selector.py        # Pre-committed Promote/Hold/Retire rules
│   ├── memory.py          # Explicit ResearchState built from prior outcomes
│   ├── loop.py            # Original fixed-generation baseline
│   └── controller.py      # Adaptive XALPHA-inspired research controller
└── utils/
    ├── config.py          # Configuration validation and hashing
    └── logging.py         # Append-only records, trajectory table, and lineage tree

scripts/
├── prepare_data.py        # Build research_2010_2015 and holdout_2016 separately
├── run_loop.py            # Original fixed-generation baseline on 2010-2015 only
├── run_research.py        # Budget-driven Macro/Micro/Cross V1 entrypoint
├── compare_search.py       # Isolated deterministic/LLM research arms and reports
├── compare_adaptive.py     # Equal-budget adaptive deterministic/Luna comparison
└── final_holdout.py       # One-shot evaluation of a frozen library on 2016
```

See [docs/architecture.md](docs/architecture.md) for the mapping from the paper to this
repository and the decisions that intentionally differ from the paper.
See [docs/dsl_v2.md](docs/dsl_v2.md) for the typed expression DSL, operator catalog, validation
limits, and LLM proposal schema.

## Data integration

All six supplied archives are extracted next to their zip files under `data/`. Extracted
CSVs and archives remain local and are ignored by Git.

The active research panel uses:

- `prices-split-adjusted.csv` for OHLCV;
- `fundamentals.csv`, shifted by a documented 90-day reporting lag;
- `analyst_ratings_processed.csv` plus `raw_partner_headlines.csv` for news;
- `securities.csv` for sector and security metadata.

`raw_analyst_ratings.csv` is retained for audit only because 286,261 rows have unparseable
dates, whereas the processed version has 2,578. News is loaded in chunks, restricted to the
price universe and requested period, deduplicated by symbol/timestamp/headline, and delayed
one calendar day before it becomes signal-eligible. The unadjusted price file is retained for
corporate-action diagnostics but is not the factor-construction source.

`prepare_data.py` writes two different files:

- `artifacts/processed/research_2010_2015.parquet`
- `artifacts/processed/holdout_2016.parquet`

The research entrypoint imports only the first path and aborts if it contains any 2016 rows.
The final holdout script may prepend the research history solely to warm up rolling features;
all reported targets and portfolio returns are filtered to 2016.

## Validation policy

| Construction window | Adaptive validation |
| --- | --- |
| 2010-2012 | 2013 |
| 2010-2013 | 2014 |
| 2010-2014 | 2015 |

The loop may learn from 2013-2015 because these are research validation folds. It cannot read
2016. After research, `--freeze` persists the factor library, configuration, thresholds,
random seed, configuration hash, and Git commit. Only then can `final_holdout.py` load 2016;
it refuses to overwrite an existing holdout result.

## Run sequence

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"       # optional; omit for deterministic fallback
export OPENAI_FACTOR_MODEL="gpt-6-luna"  # optional model override

python scripts/prepare_data.py
python scripts/run_loop.py                  # safe dry-run
python scripts/run_loop.py --smoke          # one real factor, research panel only
python scripts/run_loop.py --execute        # research only; never reads 2016
python scripts/run_loop.py --execute --freeze
python scripts/run_research.py --dry-run        # validate adaptive-controller setup
python scripts/run_research.py                  # adaptive research, 2010-2015 only
python scripts/run_research.py --execute --freeze
python scripts/final_holdout.py             # exactly once after freeze

python scripts/test_llm_connection.py       # one minimal structured Luna request
python scripts/compare_search.py --arm deterministic
python scripts/compare_search.py --arm llm
python scripts/compare_adaptive.py --arm deterministic --budget 60
python scripts/compare_adaptive.py --arm both --budget 60

pytest -q
```

## Research discipline

- Every candidate is a typed `FactorSpec` with an ex-ante economic hypothesis.
- The LLM can emit only a schema-validated recipe from a fixed feature/operator catalog; it
  cannot emit executable pandas/Python code.
- Only compact research metrics and promoted recipes are sent to the LLM. Raw observations
  and the 2016 holdout are never included in its context.
- If the LLM is disabled, unavailable, missing an API key, refuses, or returns no valid
  proposals, the loop records the reason and falls back to deterministic exploration.
- Time-series operations run within ticker history; cross-sectional operations run within date.
- All candidates use the same IC and portfolio evaluator.
- Promotion uses hard gates before tie-breaking; no opaque mega-score is used.
- Search history is explicit state, not hidden chat context.
- Correlation and residual IC prevent a library of near-duplicate parameter variants.
- Parent candidates above 0.90 signal correlation within a mechanism share one pool
  representative; historical Parent/cluster counts guide Macro Brain pivots.
- The seed set includes a deliberately leaky future-return factor that must be retired before
  performance evaluation.

## How generation feedback works

Generation 0 is always the fixed, auditable seed library. After each generation, every
candidate's 1/5/10-day fold IC, Newey-West IC t-statistic, positive-fold count, net and stressed-cost Sharpe,
turnover, drawdown, redundancy, residual IC, decision, and rejection reasons are written to
the experiment log and summarized in `ResearchState`.

For the next generation, promoted factors receive deterministic bounded mutations. When
`llm.enabled` is true and `OPENAI_API_KEY` is available, the restricted LLM generator also
receives the cumulative state, a bounded set of prior experiment records, eligible Promote/Hold parent
specifications, the allowed DSL catalog, and remaining budget. Its structured proposals are
validated again locally for windows, complexity, uniqueness, allowed primitives, and leakage
before evaluation. In the adaptive path, Macro-owned mechanism, lineage, and proposal budget
are omitted from Micro output and bound locally. Operator-specific params and separate
windowed/scalar feature schemas prevent the dominant malformed DSL cases before semantic
validation. `artifacts/experiments/generation_events.jsonl` records whether the LLM was used,
one allowed structured-output retry, stage-level rejection counters, and deterministic fill.

`ResearchState` also maintains a stable failure taxonomy (`unstable_ic`, `weak_signal`,
`cost_sensitivity`, `excessive_turnover`, `redundancy`, and integrity failures) and a
family-level summary of decision counts, valid evidence, unique formulas, mean/best IC,
Newey-West significance, turnover, and failure counts. The comparison runner reports first
promotion index, effective valid-and-novel evidence per ten proposals, duplicate and invalid
rates, promoted-family entropy, and walk-forward sign stability. It never reads the holdout.

The deterministic-versus-Luna experiments and their search-efficiency results are summarized
in [`docs/experiment_results.md`](docs/experiment_results.md). The generated, fully auditable
trajectories remain local under `artifacts/experiments/adaptive_comparisons/`.

## XALPHA-inspired adaptive V1

The new controller operates at two timescales. The Macro Brain chooses one research action
and a candidate budget; the Micro Brain creates only valid `FactorSpec` recipes; the Cross
Brain converts outcomes into bounded GOOD/BAD lessons and mechanism-level statistics. A
lenient parent pool supports exploration, while the stricter elite archive is persistent.
Neither archive automatically becomes the final library: freezing performs another pass for
canonical-formula duplicates, cross-sectional correlation, residual IC, simplicity, and
mechanism diversity.

Before either arm may exploit a parent or stop, it must cover all eight mechanism families
with at least two evaluated candidates per family. This coverage phase is deterministic and
identical in the deterministic and Luna arms, so the LLM comparison starts from the same
minimum evidence base. After coverage, Luna may propose the cycle-level action and bounded
factor recipes. Every plan and recipe is revalidated locally, invalid output is logged, and
unused budget is filled deterministically. The 60-candidate comparison reports time to first
Parent/Elite, effective information yield, duplicate and invalid rates, archive diversity,
and fold-sign stability rather than selecting on final Sharpe alone.

The LLM receives factor formulas, hypotheses, 2010-2015 aggregate research metrics, eligible
parent summaries, failure taxonomy, and structured research memory. It does not receive raw
market rows or any 2016 data. Set `store: false` in `config.yaml` to keep API-side response
storage disabled.

This is deliberately not a full reproduction of XALPHA. V1 does not ingest papers at run
time, implement the complete archetype taxonomy, generate arbitrary executable code, or use
a multi-agent swarm. Those features would weaken auditability relative to this take-home's
small dataset and hard final-holdout policy.
