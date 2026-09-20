# Implementation Instruction
## Anthelion Autonomous Factor Researcher — V1

You are implementing a small but rigorous autonomous quantitative factor research system.

The system is inspired by two research frameworks:

1. XALPHA:
   - Macro Brain for research planning
   - Micro Brain for hypothesis-to-factor construction and evolution
   - Cross Brain for empirical feedback consolidation
   - GOOD/BAD structured memory
   - parent pool vs elite archive
   - static and dynamic leakage checks
   - redundancy-aware final factor library

2. AutoScientist-Quant:
   - research actions: IMPROVE / COMBINE / PIVOT / STOP
   - adaptive rather than fixed-generation search
   - strict separation between research feedback data and final held-out test data
   - library quality is different from individual factor quality

The goal is NOT to reproduce either paper completely.

The goal is to build a minimal, transparent, auditable version suitable for the Anthelion take-home assignment.

The key principle is:

> The system should behave like a disciplined quant researcher, not like a brute-force formula generator.

The complete research trajectory must be reproducible and explainable.

---

## 1. HIGH-LEVEL ARCHITECTURE

Implement the following loop:

```text
                    Research Memory
                          |
                          v
                 +----------------+
                 |   Macro Brain  |
                 | research plan  |
                 +-------+--------+
                         |
                         v
                    Hypothesis
                         |
                         v
                 +----------------+
                 |   Micro Brain  |
                 | factor design  |
                 | build/evolve   |
                 +-------+--------+
                         |
                         v
                 Quality / Leakage
                       Gates
                         |
                         v
                     Evaluator
                         |
                  +------+------+
                  |             |
                Parent        Elite
                  |             |
                  +------+------+
                         |
                         v
                 +----------------+
                 |   Cross Brain  |
                 | GOOD / BAD     |
                 | reflection     |
                 +-------+--------+
                         |
                         v
                    Research Memory
                         |
                         +------ next round
```

After autonomous research ends:

```text
                    Elite Archive
                         |
                         v
                Redundancy Filtering
                         |
                         v
              Incremental Alpha Test
                         |
                         v
                Final Factor Library
                         |
                         v
                 Simple Aggregation
                         |
                         v
                  2016 FINAL TEST
```

**CRITICAL RULE**

The final test period must NEVER influence:

- hypothesis generation
- Macro Brain decisions
- factor mutation
- factor selection
- GOOD/BAD memory
- threshold tuning
- library construction
- stopping decisions

The final test is evaluated exactly once after the research process is frozen.

---

## 2. SCOPE REDUCTION

Do **not** implement the full XALPHA architecture.

### Implement in V1

- Macro Brain
- Micro Brain
- Cross Brain
- structured research memory
- FactorSpec DSL
- deterministic factor builder
- static leakage checks
- dynamic leakage checks
- normal parent pool
- elite archive
- IMPROVE / COMBINE / PIVOT / STOP
- final library filtering

### Do not implement yet

- 48 research archetypes
- report-to-memory PDF ingestion
- multi-agent swarm
- vector database
- arbitrary autonomous Python execution
- downstream LightGBM search
- portfolio model hyperparameter search
- complicated RL controller

Keep the architecture modular so those features could be added later.

---

## 3. DATA CONTRACT

Unlike XALPHA, this project has three data families:

```text
PRICE
FUNDAMENTALS
NEWS
```

Create an explicit `DataContract`.

Example:

```python
@dataclass
class DataContract:
    price_fields: set[str]
    fundamental_fields: set[str]
    news_fields: set[str]
    allowed_transforms: set[str]
    unavailable_fields: set[str]
```

Candidate hypotheses must be feasible under this contract.

### Valid example

> Medium-term momentum may be stronger during persistent negative-news regimes.

Uses:

- price
- news

### Valid example

> High earnings yield may be more predictive when investor attention is low.

Uses:

- fundamentals
- news

### Invalid example

> Analyst revision momentum predicts returns.

If analyst estimates are unavailable, reject this hypothesis **before backtesting**.

---

## 4. TIME SPLIT AND INFORMATION ISOLATION

Make all dates configurable.

Recommended initial split if data coverage supports it:

```text
TRAIN:
2010-01-01 to 2013-12-31

VALIDATION:
2014-01-01 to 2014-12-31

FEEDBACK:
2015-01-01 to 2015-12-31

FINAL_TEST:
2016-01-01 to 2016-12-31
```

Interpretation:

### TRAIN

Used for:

- basic factor estimation
- direction determination

### VALIDATION

Used for:

- robustness
- elite selection

### FEEDBACK

May be observed by the autonomous research loop and may influence future Macro Brain decisions.

### FINAL_TEST

Completely invisible to the autonomous research process.

Important:

> Anything that affects the next research action is part of the research environment and is NOT a true final OOS test.

Physically separate:

```text
data/processed/research.parquet
```

and

```text
data/processed/final_holdout_2016.parquet
```

`scripts/run_research.py` must not import or load `final_holdout_2016.parquet`.

Only:

```text
scripts/final_holdout.py
```

may load it.

---

## 5. CORE SCHEMAS

Create:

```text
src/core/schemas.py
```

### 5.1 FactorSpec

Use a constrained `FactorSpec` rather than arbitrary LLM-generated Python.

Example:

```python
@dataclass
class FactorSpec:
    factor_id: str

    hypothesis: str
    mechanism: str

    base_feature: str
    secondary_feature: str | None

    time_series_transform: str | None
    cross_sectional_transform: str | None

    window: int | None
    secondary_window: int | None

    interaction: str | None
    direction_hint: int | None

    parent_ids: list[str]
    generation: int

    complexity: int
```

Example:

```python
FactorSpec(
    factor_id="F042",
    hypothesis=(
        "Medium-term momentum may strengthen when "
        "negative news persists."
    ),
    mechanism="news_conditioned_momentum",
    base_feature="return",
    secondary_feature="negative_news_score",
    time_series_transform="rolling_mean",
    cross_sectional_transform="rank",
    window=20,
    secondary_window=5,
    interaction="multiply",
    parent_ids=["F019"],
    generation=3,
    complexity=3,
)
```

The `FactorSpec` must be serializable to JSON.

### 5.2 ExperimentRecord

```python
@dataclass
class ExperimentRecord:
    factor_id: str
    round_id: int
    generation: int

    action: str
    mechanism: str
    hypothesis: str

    factor_spec: dict

    integrity_passed: bool
    integrity_failures: list[str]

    metrics: dict

    decision: str
    decision_reasons: list[str]

    parent_ids: list[str]

    reusable_principle: str | None
    failure_pattern: str | None
    possible_repair: str | None
```

Decision values:

```text
RETIRED
PARENT
ELITE
```

### 5.3 ResearchMemory

```python
@dataclass
class ResearchMemory:
    good_lessons: list[dict]
    bad_lessons: list[dict]

    recent_experiments: list[ExperimentRecord]

    parent_pool_ids: list[str]
    elite_archive_ids: list[str]

    mechanism_stats: dict

    recent_round_summaries: list[dict]

    current_theme: str | None
    current_budget: int

    rounds_without_improvement: int
```

Persist this memory after every research round.

Use JSON or JSONL.

Do not require a vector database.

---

## 6. MACRO BRAIN

Create:

```text
src/brains/macro.py
```

The Macro Brain does **not** build factors.

Its job is to decide:

1. what mechanism to research
2. what hypothesis direction to explore
3. which research action to take
4. which parents should be used
5. whether research should stop

Use four actions:

```text
IMPROVE
COMBINE
PIVOT
STOP
```

Create:

```python
@dataclass
class ResearchPlan:
    action: str
    theme: str
    mechanism: str

    hypothesis_goal: str

    parent_ids: list[str]

    reason: str

    candidate_budget: int
```

### 6.1 IMPROVE

Use when:

- a parent has promising predictive evidence
- but has a known weakness

Examples of weaknesses:

- high turnover
- unstable periods
- excessive complexity
- weak news interaction
- short alpha decay

Macro Brain should explicitly state what needs improvement.

Example:

```text
action:
IMPROVE

parent:
F042

reason:
"Strong RankIC but unstable during high-volatility periods."

goal:
"Test whether volatility normalization improves stability."
```

### 6.2 COMBINE

Use when two promising factors have complementary mechanisms and low redundancy.

Example:

```text
F012:
medium-term momentum

F031:
negative-news regime

COMBINE:

"Test whether persistent negative news changes the
strength of medium-term momentum."
```

Do not combine highly correlated parents simply because both perform well.

### 6.3 PIVOT

Use when:

- a mechanism family repeatedly fails
- local mutations stop improving
- the current search direction becomes saturated

Example:

```text
5-day momentum
10-day momentum
15-day momentum
20-day momentum
```

all weak.

Do **not** generate:

```text
17-day momentum
18-day momentum
```

Instead:

```text
PIVOT
```

to another mechanism, e.g.:

```text
price-volume confirmation
news attention
value-quality
reversal
```

### 6.4 STOP

STOP should be evidence-driven.

Trigger STOP when one or more conditions hold:

- total candidate budget exhausted
- no meaningful improvement for N rounds
- current research space is saturated
- elite archive is stable
- recent actions produce mostly redundant candidates

Default:

```text
rounds_without_improvement >= 2
```

can trigger STOP if the total research evidence is sufficient.

### 6.5 Initial Macro Brain implementation

Implement two modes:

1. `deterministic_controller`
2. `llm_controller` interface

The deterministic controller must work without any API.

The LLM controller can be a stub/interface initially.

Do not make the whole system dependent on an external LLM API.

---

## 7. RESEARCH MECHANISM TAXONOMY

Do not implement XALPHA's full 48-archetype taxonomy.

Use approximately 8 mechanism families:

```text
PRICE_TREND
PRICE_REVERSAL
VOLATILITY
PRICE_VOLUME
FUNDAMENTAL_VALUE
FUNDAMENTAL_QUALITY
NEWS_ATTENTION
CROSS_DOMAIN_REGIME
```

Each factor has exactly one primary mechanism.

Optional supporting tags may be stored.

This taxonomy exists to organize memory and search behavior, not to mechanically determine the factor formula.

---

## 8. MICRO BRAIN

Create:

```text
src/brains/micro.py
```

Responsibilities:

```text
ResearchPlan
    ->
candidate hypotheses
    ->
FactorSpecs
    ->
factor construction
    ->
quality pipeline
    ->
evaluation
    ->
mutation / crossover / refinement
```

The Micro Brain should not directly manipulate the final test.

### 8.1 Seed generation

Given `ResearchPlan`, generate a small candidate set.

Example:

```text
candidate_budget = 5
```

The generator should produce 5 `FactorSpec`s corresponding to the research goal.

All candidates must have:

- explicit hypothesis
- mechanism
- implementation recipe
- expected predictive direction
- complexity estimate

### 8.2 Evolution operators

Implement:

```text
MUTATION
CROSSOVER
REFINEMENT
```

#### MUTATION

Change one meaningful component while preserving the parent mechanism.

Allowed examples:

- change horizon
- smooth signal
- volatility-adjust signal
- rank vs z-score
- condition on news regime

#### CROSSOVER

Combine two mechanisms.

Example:

```text
parent A:
momentum

parent B:
negative-news regime

child:
news-conditioned momentum
```

#### REFINEMENT

Simplify a factor while preserving its core hypothesis.

Example:

Before:

```text
rank(
    ewma(momentum20)
    * volatility_filter
    * news_filter
)
```

After:

```text
rank(
    momentum20
    * news_regime
)
```

Prefer the simpler version when performance is comparable.

---

## 9. DETERMINISTIC FACTOR BUILDER

Create:

```text
src/factors/builder.py
```

The Macro/Micro brains propose `FactorSpec`s.

The `FactorBuilder` implements them deterministically.

Example API:

```python
series = builder.build(
    factor_spec,
    research_panel
)
```

LLMs should **not** invent arbitrary dataframe operations in V1.

Allowed primitives should live in:

```text
src/factors/primitives.py
```

Examples:

```text
return_k
volatility_k
volume_change
relative_volume
book_to_price
earnings_yield
roe
asset_growth
news_sentiment
negative_news_score
news_volume
```

Allowed transforms:

```text
rank
zscore
winsorize
rolling_mean
rolling_std
ewma
delta
lag
divide
multiply
subtract
```

Cross-sectional transforms must operate within date.

Time-series transforms must operate within ticker history.

---

## 10. QUALITY PIPELINE

Create:

```text
src/quality/
```

Every candidate must pass:

```text
FactorSpec validation
        |
        v
Static leakage checks
        |
        v
Hypothesis / implementation alignment
        |
        v
Factor materialization
        |
        v
Numerical validation
        |
        v
Dynamic leakage tests
        |
        v
Empirical evaluation
```

---

## 11. STATIC LEAKAGE CHECKS

Create:

```text
src/quality/static_checks.py
```

Reject:

- negative shift
- centered rolling window
- use of future return columns
- full-sample normalization
- backfill from future observations
- use of FINAL_TEST
- unsupported columns
- unavailable fields
- invalid fundamental availability
- excessive factor complexity

Examples that must fail:

```python
shift(-1)
```

```python
rolling(..., center=True)
```

```text
future_return_5d used as feature
```

```python
df.mean()
```

over the entire panel when constructing a historical time-t signal.

Fundamental data must use point-in-time availability.

If true filing/publication dates are unavailable, apply a configurable conservative publication lag.

---

## 12. DYNAMIC LEAKAGE CHECKS

Create:

```text
src/quality/dynamic_leakage.py
```

Implement at least two tests.

### 12.1 Truncation invariance test

For selected ticker/date cutoff `t`:

1. compute factor using full historical panel
2. truncate data so observations after `t` are removed
3. recompute factor
4. compare factor values at or before `t`

Requirement:

```text
factor_full[t]
≈
factor_truncated[t]
```

within numerical tolerance.

If the value changes materially:

```text
FAIL:
dynamic future dependency detected
```

Run this test on several sampled:

- tickers
- cutoff dates

### 12.2 Future-noise perturbation test

For cutoff `t`:

1. copy input data
2. replace observations **after** `t` with random noise
3. recompute factor
4. verify factor at `t` and earlier does not change

Requirement:

```text
f_t(original data)
≈
f_t(data with future corrupted)
```

If future perturbation changes historical factor values:

```text
FAIL LOOKAHEAD
```

These tests should be implemented efficiently using a small sample of tickers and dates.

---

## 13. TRI-ALIGNMENT CHECK

Create:

```text
src/quality/alignment.py
```

Check consistency between:

```text
HYPOTHESIS
FACTOR SPEC
IMPLEMENTED SIGNAL
```

Example failure:

```text
Hypothesis:
"short-term reversal"

Implementation:
positive 20-day momentum
```

Result:

```text
FAIL alignment
```

Another failure:

```text
Hypothesis:
"news-conditioned momentum"

Implementation:
does not use any news variable
```

Result:

```text
FAIL alignment
```

For V1 this can be rule-based.

Later it can be LLM-assisted.

---

## 14. NUMERICAL VALIDATION

Reject signals with:

- excessive NaNs
- infinite values
- almost constant cross-sections
- insufficient coverage
- extreme outlier concentration

Make thresholds configurable.

Example:

```text
max_nan_ratio = 0.30
```

---

## 15. EVALUATION

Create:

```text
src/evaluation/
```

Compute:

```text
IC
RankIC
ICIR
RankICIR
positive IC fraction
```

and portfolio diagnostics:

```text
long-short return
annualized Sharpe
max drawdown
turnover
transaction-cost-adjusted Sharpe
bps per turnover
```

Use future returns strictly **after** signal formation.

Example:

```text
signal_t
```

predicts:

```text
return t+1 to t+h
```

Do not use contemporaneous return accidentally.

---

## 16. TEMPORAL ROBUSTNESS

Do not evaluate only one full-sample statistic.

Within the research-visible period, implement rolling or expanding validation.

At minimum report:

```text
mean RankIC
RankIC stability
positive-window fraction
worst-window RankIC
portfolio Sharpe by subperiod
```

A factor with:

```text
excellent average RankIC
```

but:

```text
negative predictive performance in most subperiods
```

should not become elite.

---

## 17. PARENT VS ELITE SELECTION

Create:

```text
src/selection/gates.py
```

Use three outcomes:

```text
RETIRED
PARENT
ELITE
```

### 17.1 PARENT

PARENT means:

> promising enough to continue research, but not strong enough for final archive.

Example requirements:

- positive aligned RankIC
- adequate coverage
- at least moderate temporal consistency
- passed all integrity checks
- no catastrophic turnover / cost problem

PARENT thresholds should be lenient enough to maintain research diversity.

### 17.2 ELITE

ELITE means:

> strong empirical evidence and worthy of persistent archive.

Require stricter evidence:

- stronger RankIC
- stronger temporal consistency
- stable validation / feedback performance
- acceptable implementation cost
- acceptable complexity
- nontrivial incremental information

Do not require exact XALPHA thresholds.

Put all thresholds in `config.yaml`.

### 17.3 RETIRED

Examples:

- leakage failure
- weak predictive signal
- unstable direction
- excessive missing data
- excessive turnover
- high redundancy with better factor
- complexity without incremental benefit

Every retirement must include a reason.

---

## 18. ELITE ARCHIVE

Create:

```text
EliteArchive
```

Elite factors persist across research rounds.

Fields should include:

```text
factor_id
FactorSpec
hypothesis
mechanism
metrics
lineage
GOOD memory summary
```

The elite archive should not automatically equal the final factor library.

Elite means:

> strong individual candidate

not:

> must appear in final portfolio

---

## 19. CROSS BRAIN

Create:

```text
src/brains/cross.py
```

The Cross Brain converts experiment outcomes into research knowledge.

It should **not** merely save metrics.

For each useful success produce GOOD memory.

For each informative failure produce BAD memory.

---

## 20. GOOD MEMORY FORMAT

Example:

```json
{
  "type": "GOOD",

  "factor_id": "F042",

  "mechanism":
      "news_conditioned_momentum",

  "validated_hypothesis":
      "Medium-term momentum becomes more stable when negative-news persistence is elevated.",

  "evidence": {
      "rank_ic": 0.021,
      "positive_windows": 0.75,
      "net_sharpe": 0.91
  },

  "reusable_principle":
      "Persistent rather than one-day news signals appear more useful as regime conditioners.",

  "do_not_copy":
      "Do not simply reproduce the same 20-day window.",

  "next_directions": [
      "test longer momentum horizon",
      "test sentiment persistence vs news volume"
  ]
}
```

GOOD memory should preserve:

> WHY it may work

not only:

> WHAT its score was.

---

## 21. BAD MEMORY FORMAT

Example:

```json
{
  "type": "BAD",

  "factor_id": "F051",

  "mechanism":
      "raw_news_momentum_interaction",

  "failure_type":
      "temporal_instability",

  "failed_assumption":
      "Daily news sentiment is sufficiently persistent to condition momentum directly.",

  "evidence": {
      "rank_ic": 0.003,
      "positive_windows": 0.33
  },

  "avoidance_rule":
      "Avoid using one-day raw sentiment as a direct multiplier.",

  "possible_repairs": [
      "smooth news sentiment",
      "use news-volume regime",
      "interact only in extreme news states"
  ]
}
```

BAD memory must distinguish:

```text
terminal failure
```

from:

```text
repairable failure
```

---

## 22. MEMORY LEVELS

Use only two memory levels in V1.

### SHORT-TERM MEMORY

Contains:

- recent GOOD/BAD records
- recent rounds
- recent failures
- recent improvements

Used for:

- IMPROVE
- COMBINE
- immediate PIVOT decisions

### LONG-TERM MEMORY

Contains:

- elite factors
- validated mechanisms
- persistent failure patterns
- mechanism-level statistics

Used for:

- research-theme selection
- avoiding repeated search

Do not implement more memory infrastructure than this.

---

## 23. MECHANISM ATTRIBUTION

After evolution, a factor's final mechanism may differ from its parent mechanism.

The Cross Brain should assign the final factor back to the closest mechanism family.

Example:

```text
started from:
PRICE_VOLUME

after refinement:
becomes primarily short-term reversal

Final mechanism:
PRICE_REVERSAL
```

This prevents memory contamination.

---

## 24. RESEARCH CONTROLLER

Create:

```text
src/research/controller.py
```

Main loop:

```python
while budget_remaining > 0:

    plan = macro_brain.plan(state)

    if plan.action == STOP:
        break

    candidates = micro_brain.generate(
        plan,
        state
    )

    results = []

    for candidate in candidates:

        spec_validation

        static_checks

        build factor

        numerical_validation

        dynamic_leakage_tests

        evaluation

        parent_or_elite_selection

        results.append(...)

    feedback = cross_brain.reflect(results)

    memory.update(feedback)

    state.update(results)

    save_checkpoint()
```

The search should therefore be:

```text
ADAPTIVE
```

not:

```python
for generation in range(4)
```

---

## 25. DETERMINISTIC INITIAL ACTION POLICY

Implement an initial heuristic controller.

Suggested rules:

### IMPROVE if

- a recent parent has positive RankIC
- but one identifiable weakness exists
- and the mechanism is not saturated

### COMBINE if

- at least two promising parents exist
- absolute correlation < 0.50
- their mechanisms are complementary

### PIVOT if

- same mechanism produces >= 3 recent failures

or

- no meaningful improvement for 2 consecutive rounds

### STOP if

- budget exhausted

or

- no meaningful improvement for 2+ rounds and elite archive is stable

or

- candidate redundancy is extremely high

These thresholds belong in `config.yaml`.

---

## 26. RESEARCH BUDGET

Use a candidate-evaluation budget.

Example:

```yaml
max_candidate_evaluations: 150
```

Instead of:

```text
fixed 5 generations × fixed 30 candidates
```

Macro Brain receives:

```text
remaining_budget
```

and chooses:

```text
candidate_budget
```

for the current action.

Typical:

```text
IMPROVE:
3-5 candidates

COMBINE:
2-4 candidates

PIVOT:
5-8 candidates

STOP:
0
```

---

## 27. REDUNDANCY CONTROL

Create:

```text
src/selection/redundancy.py
```

Use multiple levels.

### SPEC DUPLICATION

Identical `FactorSpec`s:

```text
=> remove
```

### SIGNAL CORRELATION

If:

```text
abs(corr(candidate, existing)) > threshold
```

Default example:

```text
0.85
```

Prefer:

- simpler factor
- more stable factor
- lower turnover factor

For the final library use a stricter threshold.

Example:

```text
0.70
```

---

## 28. INCREMENTAL INFORMATION TEST

Create:

```text
src/selection/incremental.py
```

A factor should not enter the final library merely because its standalone RankIC is high.

Test whether it adds information beyond already selected factors.

Simple implementation:

Regress / project candidate signal onto selected factor signals cross-sectionally.

Construct residual:

```text
candidate_residual
=
candidate
-
projection(candidate | selected factors)
```

Then compute:

```text
Residual RankIC
```

against future return.

If:

```text
Residual RankIC ≈ 0
```

the candidate is redundant.

Alternative acceptable V1:

Compare predictive model / ensemble performance with and without the candidate using the research-visible period.

---

## 29. FINAL LIBRARY BUILDER

Create:

```text
src/selection/library.py
```

Input:

```text
elite_archive
```

Procedure:

1. remove integrity failures
2. sort by research-visible quality
3. remove near-duplicate `FactorSpec`s
4. correlation pruning
5. incremental-information test
6. prefer lower complexity when evidence is similar
7. enforce mechanism diversity if reasonable
8. output frozen `FinalFactorLibrary`

Example:

```text
max_final_factors: 8
```

Each final factor should have:

```text
factor_id
mechanism
hypothesis
formula/spec
lineage
research metrics
why selected
closest rejected redundant factor
```

---

## 30. FINAL AGGREGATION

Keep this simple.

Implement first:

```text
equal-weight normalized factor score
```

\[
\alpha_{i,t}
=
\text{mean}\left(
z(f_{1,i,t}), \dots, z(f_{K,i,t})
\right)
\]

Optional second method:

```text
Ridge regression
```

Do **not** implement complex model search in V1.

The assignment is about factor research, not maximizing predictive-model complexity.

---

## 31. FINAL HOLDOUT

After research is complete, freeze:

- Macro Brain
- Micro Brain
- Cross Brain
- ResearchMemory
- thresholds
- factor library
- aggregation method

Then run:

```bash
python scripts/final_holdout.py
```

This may load 2016 data.

It must **not**:

- mutate factors
- change thresholds
- write GOOD/BAD memory
- change factor directions
- replace factors
- retrigger research

Final test results are reporting only.

---

## 32. EXPERIMENT LOGGING

Persist:

```text
experiments/experiments.jsonl
```

Each record should include:

```text
round
action
factor_id
parents
mechanism
hypothesis
FactorSpec
integrity results
metrics
decision
decision reason
GOOD/BAD summary
```

Also create:

```text
experiments/rounds.jsonl
```

Example:

```json
{
  "round": 4,
  "action": "IMPROVE",
  "theme": "news_conditioned_momentum",
  "parents": ["F019"],
  "reason":
      "Parent has strong RankIC but unstable news sensitivity.",
  "candidates_generated": 4,
  "parents_retained": 2,
  "elites_added": 1,
  "best_improvement": 0.004,
  "budget_remaining": 73
}
```

This log is necessary for the take-home requirement:

> What was proposed, what survived, what was killed and why?

---

## 33. RECOMMENDED REPOSITORY STRUCTURE

```text
src/
    core/
        schemas.py
        state.py
        config.py

    brains/
        macro.py
        micro.py
        cross.py

    data/
        loader.py
        preprocess.py
        point_in_time.py

    factors/
        primitives.py
        transforms.py
        builder.py

    quality/
        static_checks.py
        dynamic_leakage.py
        alignment.py
        numerical.py

    evaluation/
        metrics.py
        portfolio.py
        temporal.py

    selection/
        gates.py
        redundancy.py
        incremental.py
        library.py

    memory/
        store.py
        summaries.py

    research/
        controller.py

scripts/
    prepare_data.py
    run_research.py
    final_holdout.py

experiments/
    experiments.jsonl
    rounds.jsonl
    checkpoints/

tests/
    test_factor_builder.py
    test_static_leakage.py
    test_truncation_leakage.py
    test_future_noise_leakage.py
    test_holdout_isolation.py
    test_redundancy.py
    test_incremental_information.py
    test_macro_actions.py

config.yaml

README.md
```

---

## 34. TESTS THAT MUST EXIST

### TEST 1

Negative shift is rejected.

### TEST 2

Centered rolling is rejected.

### TEST 3

Future-return column cannot be used as a factor feature.

### TEST 4

Truncation invariance catches an intentionally leaking factor.

### TEST 5

Future-noise perturbation catches an intentionally leaking factor.

### TEST 6

`run_research` cannot load final holdout.

### TEST 7

`final_holdout` cannot update `ResearchMemory`.

### TEST 8

Near-identical factors are removed by correlation pruning.

### TEST 9

A redundant candidate with zero residual IC is rejected.

### TEST 10

Macro Brain can produce all four actions:

```text
IMPROVE
COMBINE
PIVOT
STOP
```

---

## 35. IMPLEMENTATION ORDER

Implement incrementally.

### PHASE 1

- schemas
- `FactorSpec`
- `FactorBuilder`
- data contract
- basic evaluator

### PHASE 2

- static leakage
- truncation test
- future-noise test
- point-in-time protection

### PHASE 3

- parent / elite gates
- `EliteArchive`
- experiment logs

### PHASE 4

- GOOD/BAD memory
- Cross Brain

### PHASE 5

- Macro Brain
- IMPROVE / COMBINE / PIVOT / STOP

### PHASE 6

Micro Brain evolution:

- mutation
- crossover
- refinement

### PHASE 7

Final library:

- correlation pruning
- incremental IC

### PHASE 8

- 2016 final holdout
- README
- diagnostic outputs

Do not implement the LLM integration before the deterministic pipeline works end-to-end.

---

## 36. MINIMUM END-TO-END DEMO

Before expanding factor families, demonstrate the entire loop using a small factor universe.

Seed mechanisms may include:

```text
PRICE_TREND
    return_20
    return_60

PRICE_REVERSAL
    -return_5

VOLATILITY
    volatility_20

FUNDAMENTAL_VALUE
    book_to_price
    earnings_yield

NEWS_ATTENTION
    negative_news_5d
    news_volume_5d
```

Run the autonomous loop on these first.

Example trajectory:

```text
Round 1
PIVOT / initialization
generate seed factors

Round 2
IMPROVE
momentum has positive RankIC but high volatility sensitivity

Round 3
COMBINE
momentum + negative-news regime

Round 4
PIVOT
raw news factors repeatedly fail

Round 5
IMPROVE
smooth fundamental signal

Round 6
STOP
no material incremental improvement
```

The specific trajectory must emerge from evaluation, not be hard-coded.

---

## 37. DEFINITION OF DONE

V1 is complete when a single command:

```bash
python scripts/run_research.py
```

can autonomously:

- read research-visible data
- initialize research state
- choose research actions
- generate `FactorSpec`s
- construct factor values
- reject invalid/leaking factors
- evaluate valid factors
- classify candidates as `RETIRED` / `PARENT` / `ELITE`
- create GOOD/BAD memory
- update future research decisions
- stop based on evidence or budget
- construct a final non-redundant factor library
- save the complete trajectory

And:

```bash
python scripts/final_holdout.py
```

can:

- load the frozen library
- evaluate it on 2016
- produce final metrics
- never modify the research state

---

## 38. DESIGN PHILOSOPHY

Prioritize:

```text
correctness
auditability
point-in-time validity
simple interpretable hypotheses
structured learning from failure
research trajectory
reproducibility
```

over:

```text
maximum factor count
maximum Sharpe
complex formula search
large multi-agent orchestration
fancy model architecture
```

The project should demonstrate:

> The intelligence is in the research loop and the judgment embedded in it, not in one lucky factor.
