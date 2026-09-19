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
| Memory and policy update | `src/research/memory.py` |
| Structured experiment trace | `src/utils/logging.py` |
| Frozen out-of-sample policy | physical parquet split plus `scripts/final_holdout.py` |

The design also follows the paper's separation between date-local cross-sectional transforms
and ticker-local historical transforms, its requirement for an economic rationale before a
backtest, and its emphasis on turnover, transaction costs, monotonic portfolio sorts, and
multiple-testing discipline.

## Restricted LLM boundary

The LLM is a proposal mechanism, not an execution engine. It receives compact cumulative
research state, selected experiment metrics, and promoted `FactorSpec` recipes, but no raw
rows and no holdout results. Structured output constrains it to a closed feature/operator
vocabulary. Local code then independently rejects unknown parents, invalid windows, excessive
complexity, duplicate IDs/formulas, and malformed interactions. Only accepted `FactorSpec`
objects reach the same deterministic builder and evaluator used by every non-LLM candidate.
API failures are append-only audit events and trigger the deterministic exploration fallback.

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
