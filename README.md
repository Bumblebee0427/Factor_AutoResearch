# Autonomous Factor Research Loop

A constrained, memory-guided research agent that proposes economically motivated factor
recipes, executes them through a deterministic point-in-time engine, evaluates them under
fixed temporal validation, and uses structured evidence to guide later generations.

The implementation adapts Huang and Fan (2026),
[_Beyond Prompting: An Autonomous Framework for Systematic Factor Investing via Agentic AI_](references/2603.14288v1.pdf),
to the shorter 2010-2016 Anthelion price, fundamental, and news dataset. The project-specific
rules are documented in the [instruction](Anthelion_Autonomous_Factor_Research_Instruction.pdf)
and [Codex reference guide](references/Auto_Factor_Research_Codex_Reference_Guide.pdf).

## Architecture

```text
src/
├── data/
│   ├── loader.py          # Extracted CSV schemas, chunked news loading, source metadata
│   ├── point_in_time.py   # Availability-date joins and conservative news timing
│   ├── quality.py         # Key, overlap, year, and availability invariants
│   └── preprocess.py      # Universe filters and physical research/holdout split
├── factors/
│   ├── schema.py          # Typed FactorSpec grammar and canonical formulas
│   ├── primitives.py      # Historical price, volume, volatility, and news primitives
│   ├── transforms.py      # Date-local rank, z-score, and winsorization
│   └── builder.py         # Deterministic FactorSpec execution; no generated Python
├── evaluation/
│   ├── integrity.py       # Leakage, availability, complexity, and coverage gates
│   ├── ic.py              # Daily rank IC, ICIR, hit rate, and t-statistic
│   ├── portfolio.py       # Quantile monotonicity, long-short returns, costs, turnover
│   ├── validation.py      # Walk-forward folds and holdout firewall
│   ├── redundancy.py      # Signal correlation and residual/incremental IC
│   └── evaluator.py       # One common evaluator for every candidate
├── research/
│   ├── generator.py       # Generation-0 seeds and cross-family exploration
│   ├── llm_generator.py   # Structured LLM proposals with local whitelist validation
│   ├── failures.py        # Stable failure taxonomy for gates, memory, and prompts
│   ├── comparison.py      # Search-efficiency and stability comparison metrics
│   ├── mutator.py         # Bounded exploitation around promoted parents
│   ├── selector.py        # Pre-committed Promote/Hold/Retire rules
│   ├── memory.py          # Explicit ResearchState built from prior outcomes
│   └── loop.py            # Closed-loop orchestration and frozen library creation
└── utils/
    ├── config.py          # Configuration validation and hashing
    └── logging.py         # Append-only records, trajectory table, and lineage tree

scripts/
├── prepare_data.py        # Build research_2010_2015 and holdout_2016 separately
├── run_loop.py            # Dry-run or execute adaptive research on 2010-2015 only
├── compare_search.py       # Isolated deterministic/LLM research arms and reports
└── final_holdout.py       # One-shot evaluation of a frozen library on 2016
```

See [docs/architecture.md](docs/architecture.md) for the mapping from the paper to this
repository and the decisions that intentionally differ from the paper.

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
export OPENAI_FACTOR_MODEL="gpt-5.6-luna"  # optional model override

python scripts/prepare_data.py
python scripts/run_loop.py                  # safe dry-run
python scripts/run_loop.py --smoke          # one real factor, research panel only
python scripts/run_loop.py --execute        # research only; never reads 2016
python scripts/run_loop.py --execute --freeze
python scripts/final_holdout.py             # exactly once after freeze

python scripts/test_llm_connection.py       # one minimal structured Luna request
python scripts/compare_search.py --arm deterministic
python scripts/compare_search.py --arm llm

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
validated again locally for parent lineage, windows, complexity, uniqueness, and allowed
primitives before evaluation. `artifacts/experiments/generation_events.jsonl` records whether
the LLM was used and why any proposals were rejected.

`ResearchState` also maintains a stable failure taxonomy (`unstable_ic`, `weak_signal`,
`cost_sensitivity`, `excessive_turnover`, `redundancy`, and integrity failures) and a
family-level summary of decision counts, valid evidence, unique formulas, mean/best IC,
Newey-West significance, turnover, and failure counts. The comparison runner reports first
promotion index, effective valid-and-novel evidence per ten proposals, duplicate and invalid
rates, promoted-family entropy, and walk-forward sign stability. It never reads the holdout.

The first deterministic-versus-Luna Low experiment and its negative result are summarized in
[`docs/experiment_results.md`](docs/experiment_results.md). The generated, fully auditable
trajectory remains local under `artifacts/experiments/comparisons/`.
