# Architecture and source mapping

## What is borrowed from the paper

The paper's central abstraction is the closed loop `H_k -> F_k -> M_k -> D_k`: hypotheses
become factor recipes, recipes become deterministic factor series, a common evaluator produces
metrics, and a transparent gate produces decisions. The next generation is conditioned on
structured outcomes. This repository maps those roles as follows:

| Paper role | Repository implementation |
| --- | --- |
| Constrained hypothesis generator | `src/research/generator.py`, `src/research/llm_generator.py`, `src/research/mutator.py` |
| Symbolic factor grammar | `src/factors/schema.py` |
| Deterministic execution | `src/factors/builder.py` |
| Unified evaluator | `src/evaluation/evaluator.py` |
| Transparent gatekeeper | `src/research/selector.py` |
| Memory and policy update | `src/research/memory.py`, `src/research/failures.py` |
| Structured experiment trace | `src/utils/logging.py` |
| Frozen out-of-sample policy | physical parquet split plus `scripts/final_holdout.py` |

The design also follows the paper's separation between date-local cross-sectional transforms
and ticker-local historical transforms, its requirement for an economic rationale before a
backtest, and its emphasis on turnover, transaction costs, monotonic portfolio sorts, and
multiple-testing discipline.

## XALPHA-inspired V1 layer

The second reference adds a useful separation of responsibilities without changing the
trusted evaluator:

| XALPHA concept | V1 implementation | Safety boundary |
| --- | --- | --- |
| Macro Brain | `src/brains/macro.py` | Selects action, mechanism, parents, and budget only |
| Micro Brain | `src/brains/micro.py` | Emits typed `FactorSpec`; never emits Python |
| Cross Brain | `src/brains/cross.py` | Writes bounded GOOD/BAD lessons and mechanism statistics |
| Normal parent population | `src/selection/archive.py::ParentPool` | Lenient, bounded, evictable |
| Elite archive | `src/selection/archive.py::EliteArchive` | Strict, deduplicated, persistent |
| Adaptive cycle | `src/research/controller.py` | Budget/patience stopping instead of fixed generations |
| Dynamic leak detector | `src/quality/dynamic_leakage.py` | Truncation and future-noise invariance |
| Final library builder | `src/selection/library.py` | Correlation/residual-IC pruning after elite selection |

Each round is `plan -> propose -> static/alignment/dynamic checks -> common evaluator ->
tier -> reflect -> checkpoint`. `IMPROVE` repairs a parent, `COMBINE` crosses distinct
mechanisms, `PIVOT` explores an under-tested mechanism, and `STOP` freezes the evidence.
The Macro Brain does not author formulas, and the Cross Brain does not alter metrics.

The implementation follows the project brief rather than reproducing all XALPHA machinery:
there is no runtime PDF retrieval, 48-archetype ontology, arbitrary code generation, swarm,
or automatic LightGBM ensemble. The constrained grammar is a deliberate adaptation to make
lineage, leakage review, and exact replay tractable.

AutoScientist-Quant contributes the single global candidate budget, dynamic
`IMPROVE/COMBINE/PIVOT/STOP` routing, and the insistence that search feedback be disjoint
from the final test. V1 applies the same idea to factor discovery and library pruning. It does
not implement the paper's downstream model/hyperparameter tree search, which the project brief
explicitly leaves for a later phase.

## Restricted LLM boundary

The LLM is a proposal mechanism, not an execution engine. It receives compact cumulative
research state, selected experiment metrics, and eligible Promote/Hold `FactorSpec` recipes, but no raw
rows and no holdout results. Structured output constrains it to a closed feature/operator
vocabulary. A nested union makes single-factor and interaction recipes mutually exclusive,
so single-factor output cannot carry interaction-only fields. Local code then independently
rejects unknown parents, invalid windows, excessive complexity, duplicate IDs/formulas, and
malformed interactions. Only accepted `FactorSpec` objects reach the same deterministic
builder and evaluator used by every non-LLM candidate.
API failures are append-only audit events and trigger the deterministic exploration fallback.

The OpenAI adapter uses the Responses API with GPT-5.6 Luna, standard mode, low reasoning,
low verbosity, structured Pydantic output, no tools, and remote response storage disabled.
The local validator remains authoritative even when API schema validation succeeds.

## Evaluation and search comparison

The primary five-day rank IC is reported with a Newey-West HAC t-statistic; the automatic
bandwidth is at least the target overlap (`horizon - 1`) and also respects a sample-size rule
of thumb. One-, five-, and ten-day IC diagnostics are retained for every validation fold.
The naive t-statistic remains in the trace for audit but is not the promotion statistic.

Deterministic and LLM arms use the same panel, folds, gates, candidate budget, and evaluator.
Search comparison uses only 2013-2015 walk-forward evidence. “Effective new information” is
defined operationally as a unique candidate that passes integrity checks and produces a
complete walk-forward record; invalid structured proposals and duplicate formulas remain in
the denominator. The untouched 2016 parquet is not loaded by the comparison runner.

## Anthelion-specific adaptations

The paper uses a much larger CRSP price/volume panel and a 2021-2024 out-of-sample period.
This take-home has 501 price-series tickers, point-in-time-limited fundamentals, general news,
and only 2010-2016. Therefore:

1. Research uses expanding walk-forward validation in 2013, 2014, and 2015 rather than one
   large in-sample block.
2. 2016 is a physically separate blind holdout, not an adaptive validation year.
3. Fundamental values become available 90 days after fiscal period end because announcement
   timestamps are absent.
4. News becomes available one day after its supplied timestamp/date to handle mixed timezone
   quality conservatively.
5. The baseline final combination should be equal-weighted standardized survivors or ridge.
   LightGBM is optional and must not precede a credible simple baseline.
6. The library explicitly prunes correlated variants and measures residual IC because the
   paper's reported factor set contains several closely related turnover constructions.
7. Gate thresholds are pre-committed in `config.yaml`; they must never be changed after viewing
   2016 results.
8. The price history warm-up is 60 trading days rather than the paper's 252-observation
   screen. The supplied panel begins in 2010 and the longest baseline lookback is 60 days;
   a 252-day row-level warm-up would discard almost the entire first research year.

## Known data limitations

- The security master resembles a contemporary S&P 500 membership list rather than a complete
  point-in-time NYSE membership history, creating unavoidable survivorship/universe bias.
- Fundamental release dates are imputed rather than observed.
- News timestamps mix timezone-aware values and date-only values.
- The dataset has no bid-ask spread or market-cap field, limiting realistic cost, capacity, and
  neutralization models.

These limitations must be reported rather than silently repaired.
