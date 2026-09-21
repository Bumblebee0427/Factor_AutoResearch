# Anthelion Factor DSL V2 — P0/P1 Implementation Instruction

## Objective

Upgrade the current Factor_AutoResearch DSL from a flat typed recipe into a small, safe, composable expression language.

This is **not** a request to add a huge indicator zoo.

Implement only:

- **P0 — DSL architecture**
  - typed expression tree / AST
  - typed operator registry
  - canonical serialization
  - deterministic compilation/execution
  - backward compatibility with the current `FactorSpec`
  - separate primitive windows from transform windows
  - complexity accounting and structural validation
- **P1 — core operator expansion**
  - arithmetic composition
  - core time-series operators
  - core cross-sectional operators
  - group-relative cross-sectional operators

Do **not** add TA-Lib, entropy operators, regression operators, candlestick patterns, or arbitrary conditional execution in this phase.

Preserve these principles:

1. no generated Python
2. no access to the 2016 final holdout during research
3. deterministic local execution
4. explicit point-in-time semantics
5. static and dynamic leakage checks
6. auditable canonical formulas
7. bounded expression complexity
8. exact reproducibility

---

## 1. Why this change is needed

The current DSL is effectively a flat recipe:

```python
FactorSpec(
    base_feature="return",
    window=20,
    ts_operator="rolling_mean",
    cs_operator="rank",
    interaction_feature="news_volume",
    interaction_window=5,
)
```

This is sufficient for simple recipes but does not scale cleanly.

Problems:

- one `window` field can ambiguously serve both primitive and transform semantics;
- interactions are essentially limited to multiplication;
- nested transformations are awkward;
- operators are hard-coded through branching logic;
- the LLM cannot cleanly express momentum acceleration, price-volume correlation, sector-relative value, time-series rank followed by cross-sectional rank, or safe ratios;
- adding more operators to the flat structure will make validation increasingly fragile.

Target flow:

```text
Factor hypothesis
      ↓
Expression AST
      ↓
typed validation
      ↓
operator registry
      ↓
deterministic evaluator
      ↓
factor series
```

The LLM must never produce executable Python.

---

## 2. Target architecture

```text
FactorSpec
│
├── metadata
│   ├── factor_id
│   ├── hypothesis
│   ├── mechanism
│   ├── parent_ids
│   ├── generation
│   └── research metadata
│
└── expression
    └── ExprNode AST
            │
            ├── Feature
            ├── Constant
            └── Op
                 │
                 ▼
          Operator Registry
                 │
                 ▼
          Deterministic Builder
```

Recommended repository additions:

```text
src/
├── factors/
│   ├── schema.py
│   ├── expression.py
│   ├── operator_registry.py
│   ├── builder.py
│   ├── primitives.py
│   └── transforms.py
│
├── quality/
│   ├── static_checks.py
│   ├── dynamic_leakage.py
│   └── alignment.py
│
└── research/
    └── llm_generator.py
```

Recommended tests:

```text
tests/
├── test_expression_ast.py
├── test_operator_registry.py
├── test_expression_builder.py
├── test_dsl_backward_compatibility.py
├── test_dsl_leakage.py
├── test_dsl_canonicalization.py
├── test_group_operators.py
└── test_llm_expression_schema.py
```

---

# 3. P0 — Expression AST

Create:

```text
src/factors/expression.py
```

Use typed dataclasses or Pydantic models. All nodes must be JSON serializable and immutable after construction.

Every expression must support:

```python
to_dict()
from_dict()
canonical()
complexity()
required_history()
required_features()
```

### Feature node

```python
@dataclass(frozen=True)
class Feature:
    name: str
    window: int | None = None
```

Examples:

```python
Feature("return", window=20)
Feature("volatility", window=20)
Feature("earnings_yield")
Feature("news_volume", window=5)
```

The `window` on a feature belongs only to that primitive. It must never implicitly become the window of a parent transform.

### Constant node

```python
@dataclass(frozen=True)
class Constant:
    value: float
```

Use only for simple arithmetic. Recommended global range:

```text
-100 <= value <= 100
```

unless an operator has a tighter rule.

### Generic operator node

Prefer a generic call node rather than dozens of Python classes:

```python
@dataclass(frozen=True)
class Op:
    name: str
    args: tuple["Expr", ...]
    params: dict[str, object]
```

Example:

```python
Op(
    name="rolling_mean",
    args=(Feature("return", window=20),),
    params={"window": 5},
)
```

This explicitly means a 5-day rolling mean of a 20-day return.

---

# 4. FactorSpec V2

Keep research metadata separate from mathematics.

Recommended:

```python
@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    generation: int
    parent_ids: tuple[str, ...]

    family: str
    mechanism: str
    hypothesis: str

    expression: Expr

    mutation_reason: str | None = None
    proposal_type: str = "deterministic"
    evidence_factor_ids: tuple[str, ...] = ()
    targeted_failure: str | None = None
    expected_metric_effect: str | None = None
    falsification_condition: str | None = None
```

Derived fields must come from the expression tree:

```text
complexity
canonical_formula
expression_hash
required_features
required_history
```

Do not duplicate these manually.

---

# 5. Backward compatibility

Do not immediately delete the current flat `FactorSpec`.

Add an explicit migration layer:

```python
def legacy_spec_to_expr(spec: LegacyFactorSpec) -> Expr:
    ...
```

The transition should be:

```text
Legacy FactorSpec
      ↓
legacy compiler
      ↓
Expr AST
      ↓
new builder
```

Do not maintain two independent execution engines.

After compilation, there must be one authoritative AST execution path.

Existing current-generation factors must remain numerically equivalent after migration.

---

# 6. Example AST expressions

### 20-day momentum rank

```python
Op(
    "cs_rank",
    args=(Feature("return", window=20),),
    params={},
)
```

Canonical:

```text
cs_rank(return[20])
```

### 5-day reversal

```python
Op(
    "neg",
    args=(
        Op(
            "cs_rank",
            args=(Feature("return", window=5),),
            params={},
        ),
    ),
    params={},
)
```

Canonical:

```text
neg(cs_rank(return[5]))
```

### Momentum acceleration

```python
Op(
    "sub",
    args=(
        Feature("return", window=20),
        Feature("return", window=60),
    ),
    params={},
)
```

Canonical:

```text
sub(return[20],return[60])
```

### Volatility-adjusted momentum

```python
Op(
    "safe_div",
    args=(
        Feature("return", window=20),
        Feature("volatility", window=20),
    ),
    params={},
)
```

Canonical:

```text
safe_div(return[20],volatility[20])
```

### Price-volume correlation

```python
Op(
    "rolling_corr",
    args=(
        Feature("return", window=1),
        Op(
            "delta",
            args=(Feature("raw_volume"),),
            params={"periods": 1},
        ),
    ),
    params={"window": 20},
)
```

### Time-series rank then cross-sectional rank

```python
Op(
    "cs_rank",
    args=(
        Op(
            "ts_rank",
            args=(Feature("volume_shock", window=20),),
            params={"window": 60},
        ),
    ),
    params={},
)
```

### Sector-relative profitability

```python
Op(
    "group_zscore",
    args=(Feature("operating_margin"),),
    params={"group": "sector"},
)
```

---

# 7. P0 — Operator Registry

Create:

```text
src/factors/operator_registry.py
```

Do not continue growing `if/elif` blocks in `builder.py`.

Recommended metadata:

```python
@dataclass(frozen=True)
class OperatorSpec:
    name: str
    category: str
    arity: int
    implementation: Callable
    parameter_schema: dict
    scope: str
    complexity_cost: int
    minimum_history_fn: Callable
    allows_nan: bool
    may_create_inf: bool
    commutative: bool = False
    description: str = ""
```

Possible scopes:

```text
scalar
time_series
cross_section
group_cross_section
```

The registry must be the single source of truth for:

```text
Factor Builder
Static Validator
Complexity Calculator
Canonicalizer
LLM allowed operator schema
Documentation
Tests
```

Reduce duplicated hard-coded operator lists currently spread across schema, builder, and LLM adapter code.

---

# 8. Operator scope semantics

### Time-series scope

Runs independently inside each ticker.

Examples:

```text
lag
delta
rolling_mean
rolling_std
ts_rank
rolling_corr
```

Never mix tickers. Never read future rows.

### Cross-sectional scope

Runs independently inside each date.

Examples:

```text
cs_rank
cs_zscore
winsorize
```

Never use statistics from future dates.

### Group cross-sectional scope

Runs independently inside:

```text
date × group
```

Examples:

```text
group_rank
group_zscore
group_neutralize
```

Never compute group statistics across time.

---

# 9. P1 — Arithmetic operators

Implement exactly:

```text
add(x, y)
sub(x, y)
mul(x, y)
safe_div(x, y)

neg(x)
abs(x)
clip(x, lower, upper)
```

Do not expose raw Python arithmetic to generated code.

### safe_div

Never implement division as unprotected `/`.

Required semantics:

```python
safe_div(x, y, eps=1e-12)
```

If:

```text
abs(y) <= eps
```

return `NaN`, never infinity.

Do not silently replace zero with an arbitrary economic value.

### add / sub / mul

Binary operators must align on the original panel index.

Do not allow hidden broadcasting across dates or symbols.

### clip

Require finite scalar bounds and:

```text
lower < upper
```

---

# 10. P1 — Core time-series operators

Implement:

```text
lag(x, periods)
delta(x, periods)

rolling_sum(x, window)
rolling_mean(x, window)
rolling_std(x, window)
rolling_min(x, window)
rolling_max(x, window)

ts_rank(x, window)
ts_zscore(x, window)

ewma(x, halflife)
decay_linear(x, window)

rolling_corr(x, y, window)
rolling_cov(x, y, window)
```

All must be ticker-local.

### lag

```text
lag(x,n)
```

must equal ticker-local:

```python
groupby(symbol).shift(n)
```

Require `n >= 1`.

Negative lag is forbidden.

### delta

```text
delta(x,n) = x_t - x_{t-n}
```

Require `n >= 1`.

### rolling operators

Use trailing windows only.

Always:

```text
center = False
```

Recommended:

```text
min_periods = window
```

unless there is a documented exception.

### ts_rank

For each ticker at time `t`, rank the current observation against its own trailing history.

Choose one output convention and document it:

```text
[0,1]
```

or centered:

```text
[-0.5,0.5]
```

Do not confuse `ts_rank` with `cs_rank`.

### ts_zscore

For each ticker:

```text
(x_t - rolling_mean_t) / rolling_std_t
```

If rolling standard deviation is zero, return `NaN`.

### ewma

Use historical exponentially weighted observations.

Parameter:

```text
halflife > 0
```

Use `adjust=False` unless another convention is explicitly documented.

### decay_linear

Use trailing weights:

```text
1,2,...,n
```

with the latest observation getting weight `n`.

Normalize weights to sum to 1.

### rolling_corr / rolling_cov

Both must be ticker-local.

For insufficient history or zero variance in correlation, return `NaN`.

---

# 11. P1 — Cross-sectional operators

Keep current functionality but standardize names:

```text
cs_rank(x)
cs_zscore(x)
winsorize(x)
winsorize_zscore(x)
```

Temporary migration aliases may support:

```text
rank -> cs_rank
zscore -> cs_zscore
```

but all canonical formulas should use explicit names.

This distinction is important once `ts_rank` and `ts_zscore` exist.

---

# 12. P1 — Group cross-sectional operators

Implement:

```text
group_rank(x, group)
group_zscore(x, group)
group_neutralize(x, group)
```

Before hard-coding groups:

1. inspect actual columns produced from merged `securities.csv`;
2. create a canonical mapping;
3. whitelist only available group fields.

Desired canonical group names if supported:

```text
sector
industry
subindustry
```

Do not invent unavailable groups.

### group_rank

For every `date × group`, rank `x` among stocks in the same group.

### group_zscore

For every `date × group`:

```text
(x - group_mean) / group_std
```

If a group has too few valid stocks or zero variance, return `NaN`.

Add configurable minimum group size, e.g.:

```yaml
dsl:
  min_group_size: 5
```

### group_neutralize

For P1 implement simple group demeaning:

```text
x - group_mean(x)
```

inside each `date × group`.

Document clearly that this is group demeaning, not a full regression or risk-model neutralization.

---

# 13. DataContract changes

Extend the current `DataContract` with group metadata.

Recommended:

```python
@dataclass(frozen=True)
class DataContract:
    ...
    group_fields: dict[str, str]
```

Example only if supported by actual columns:

```python
group_fields = {
    "sector": "gics_sector",
    "industry": "gics_industry",
    "subindustry": "gics_sub_industry",
}
```

Build this mapping from real panel columns.

The LLM may only emit canonical group names in this whitelist.

---

# 14. Raw primitives needed for composition

Preserve the current primitive feature set.

Do not make primitive expansion the main focus.

Current examples:

```text
return
volatility
volume_shock
distance_to_high

after_tax_roe
operating_margin
profit_margin
earnings_yield
asset_growth

news_volume
```

If P1 composition needs raw OHLCV leaves, add them explicitly:

```text
raw_close
raw_open
raw_high
raw_low
raw_volume
```

Keep raw primitives clearly distinguished from derived features.

---

# 15. Recursive AST validation

Validate every expression before execution.

Checks must include:

```text
known operator
correct arity
valid parameter names
valid parameter ranges
known feature
valid feature window
valid group
depth limit
node-count limit
rolling-op limit
binary-op limit
group-op limit
no future-return feature
no negative lag
no unsupported input combination
```

---

# 16. Complexity limits

Expanding the DSL increases multiple-testing risk.

Add structural limits immediately.

Recommended initial defaults:

```yaml
dsl:
  max_ast_depth: 5
  max_operator_nodes: 6
  max_rolling_nodes: 3
  max_binary_nodes: 2
  max_group_nodes: 2
```

Do not tune these after seeing 2016.

---

# 17. Complexity score

Compute complexity from AST nodes.

Recommended costs:

```text
Feature                 0
Constant                0

neg                     1
abs                     1
clip                    1

add/sub/mul             1
safe_div                2

lag/delta               1

rolling_sum             1
rolling_mean            1
rolling_std             1
rolling_min/max         1

ts_rank                 2
ts_zscore               2
ewma                    1
decay_linear            2

rolling_corr            2
rolling_cov             2

cs_rank                 1
cs_zscore               1
winsorize               1

group_rank              2
group_zscore            2
group_neutralize        2
```

Use the score for:

```text
proposal validation
final-library simplicity preference
audit
optional tie-breaking
```

Do not let complexity alone promote a factor.

---

# 18. Required-history calculation

Every expression must recursively compute the minimum history required before a signal is valid.

Examples:

```text
return[20]
=> 20

rolling_mean(return[20],5)
=> base history plus rolling history

lag(return[20],5)
=> base history plus lag

ts_rank(volume_shock[20],60)
=> underlying history plus trailing rank history
```

The exact off-by-one convention must match actual implementation and be unit tested.

Use this for:

```text
coverage validation
dynamic leakage sampling
warm-up handling
```

---

# 19. Canonical formula

Canonical serialization is required for deduplication.

Semantically identical expressions must canonicalize identically.

For commutative operators:

```text
add(a,b) == add(b,a)
mul(a,b) == mul(b,a)
```

Canonicalize child order.

For non-commutative operators:

```text
sub(a,b) != sub(b,a)
safe_div(a,b) != safe_div(b,a)
```

Canonical formulas must include all operator parameters, feature windows, and group names.

Example:

```text
cs_rank(safe_div(return[20],volatility[20]))
```

---

# 20. Expression hash

Add:

```python
expr_hash = sha256(canonical_formula.encode()).hexdigest()
```

Use it for:

```text
duplicate detection
experiment logging
archive audit
LLM proposal rejection
reproducibility
```

Do not rely only on `factor_id`.

---

# 21. Deterministic builder

Refactor `src/factors/builder.py` into a recursive AST evaluator.

Pseudo-code:

```python
def evaluate_expr(expr, panel, registry):
    if isinstance(expr, Feature):
        return evaluate_feature(expr, panel)

    if isinstance(expr, Constant):
        return scalar_series(expr.value, panel.index)

    if isinstance(expr, Op):
        operator = registry[expr.name]

        args = [
            evaluate_expr(child, panel, registry)
            for child in expr.args
        ]

        return operator.implementation(
            panel=panel,
            args=args,
            params=expr.params,
        )
```

Never use:

```text
eval
exec
generated Python
```

---

# 22. Static leakage checks

Preserve all current leakage protections.

Reject:

```text
future_return
negative lag
centered rolling
unknown operator
unknown feature
unknown group
illegal parameter
full-sample normalization
arbitrary function
expression depth beyond limit
```

The production registry must contain no operator that can read future rows.

---

# 23. Dynamic leakage checks

Keep both existing tests:

```text
truncation invariance
future-noise invariance
```

Run them against AST-built signals.

Do not disable dynamic leakage tests just because the AST is typed.

Maintain an intentionally leaking test-only implementation to prove detection still works, but never expose it to production DSL or LLM generation.

---

# 24. Cross-sectional leakage rule

All cross-sectional operators must operate on one date at a time:

```text
cs_rank
cs_zscore
winsorize
group_rank
group_zscore
group_neutralize
```

Never compute mean, standard deviation, or rank over the full date history during factor construction.

---

# 25. LLM schema upgrade

Update:

```text
src/research/llm_generator.py
```

The LLM should propose typed AST JSON.

Do not add a free-form expression parser in P1.

Use discriminated Pydantic unions if useful.

Example concept:

```json
{
  "factor_id": "price_volume_corr_20",
  "hypothesis": "Persistent price-volume co-movement may indicate stronger information confirmation.",
  "mechanism": "PRICE_VOLUME",
  "expression": {
    "op": "cs_rank",
    "args": [
      {
        "op": "rolling_corr",
        "args": [
          {
            "feature": "return",
            "window": 1
          },
          {
            "op": "delta",
            "args": [
              {
                "feature": "raw_volume"
              }
            ],
            "params": {
              "periods": 1
            }
          }
        ],
        "params": {
          "window": 20
        }
      }
    ],
    "params": {}
  }
}
```

Local validation remains authoritative.

---

# 26. LLM operator catalog

Generate the LLM operator catalog from the registry.

Example:

```text
Arithmetic:
add(x,y)
sub(x,y)
mul(x,y)
safe_div(x,y)
neg(x)
abs(x)
clip(x,lower,upper)

Time Series:
lag(x,periods)
delta(x,periods)
rolling_mean(x,window)
rolling_std(x,window)
ts_rank(x,window)
rolling_corr(x,y,window)

Cross Section:
cs_rank(x)
cs_zscore(x)
group_rank(x,group)
group_neutralize(x,group)
```

Do not manually maintain a second operator list inside prompts.

---

# 27. Hypothesis-expression alignment

Extend existing tri-alignment checks.

Examples:

Hypothesis:

```text
"volume confirms momentum"
```

Expression:

```text
cs_rank(return[20])
```

should fail or warn because no volume input exists.

Hypothesis:

```text
"sector-relative value"
```

Expression:

```text
cs_rank(earnings_yield)
```

should warn because no group operator exists.

Hypothesis:

```text
"short-term reversal"
```

Expression:

```text
cs_rank(return[20])
```

with positive direction should fail alignment.

Keep P1 alignment rule-based.

---

# 28. Micro Brain mutation rules

P1 mutations should be structural and bounded.

Allowed mutation types:

```text
replace feature window
insert/remove one smoothing operator
replace cs_rank ↔ cs_zscore
insert ts_rank
insert ts_zscore
replace raw signal with delta(signal,n)
combine two expressions with add/sub/mul/safe_div
wrap a fundamental signal in group_rank/group_zscore/group_neutralize
replace multiplication interaction with rolling_corr when economically meaningful
```

Do not perform unconstrained random tree generation.

Every mutation requires a `mutation_reason`.

---

# 29. Refinement

Micro Brain must be able to simplify expressions.

Example:

```text
cs_rank(
    rolling_mean(
        ts_rank(x,20),
        5
    )
)
```

may refine to:

```text
cs_rank(
    ts_rank(x,20)
)
```

if smoothing adds no incremental evidence.

Prefer simpler expressions when evidence is comparable.

---

# 30. Duplicate control

Reject duplicates before evaluation using:

```text
canonical_formula
expression_hash
```

A larger DSL will generate many syntactic variants of the same idea, so semantic canonicalization is important.

---

# 31. Do not add these yet

Explicitly exclude from P0/P1:

```text
TA-Lib indicators
RSI
MACD
ATR
MFI
OBV

rolling_beta
rolling_alpha
rolling_r2
rolling_resid

skew
kurtosis
rolling_quantile

partial_corr

where
if_else
trade_when

entropy
sample_entropy
approximate_entropy
permutation_entropy

candlestick patterns

arbitrary expression strings
arbitrary Python code
```

These belong to later phases.

---

# 32. Final required P1 operator set

## Arithmetic

```text
add
sub
mul
safe_div
neg
abs
clip
```

## Time Series

```text
lag
delta

rolling_sum
rolling_mean
rolling_std
rolling_min
rolling_max

ts_rank
ts_zscore

ewma
decay_linear

rolling_corr
rolling_cov
```

## Cross Section

```text
cs_rank
cs_zscore
winsorize
winsorize_zscore
```

## Group Cross Section

```text
group_rank
group_zscore
group_neutralize
```

Do not expand beyond this set before rerunning the same research-budget protocol.

---

# 33. Required example hypotheses

The new DSL must express all of these without generated Python.

### Momentum acceleration

```text
cs_rank(
    sub(
        return[20],
        return[60]
    )
)
```

### Volatility-adjusted momentum

```text
cs_rank(
    safe_div(
        return[20],
        volatility[20]
    )
)
```

### Abnormal recent volume

```text
cs_rank(
    ts_zscore(
        raw_volume,
        20
    )
)
```

### Price-volume confirmation

```text
cs_rank(
    rolling_corr(
        return[1],
        delta(raw_volume,1),
        20
    )
)
```

### Sector-relative value

```text
group_rank(
    earnings_yield,
    sector
)
```

### Sector-neutral quality

```text
group_neutralize(
    operating_margin,
    sector
)
```

### News-attention persistence

```text
cs_rank(
    ts_rank(
        news_volume[5],
        20
    )
)
```

---

# 34. Required tests

## AST tests

```text
Feature serialization
Constant serialization
nested Op serialization
from_dict(to_dict(expr)) round-trip
immutability
canonical string stability
expression hash stability
```

## Registry tests

Every operator must have:

```text
unique name
valid category
valid scope
valid arity
parameter schema
implementation
complexity cost
minimum-history rule
```

## Semantic tests

Verify:

```text
lag(x,1) == ticker-local shift(1)
delta(x,5) == x - lag(x,5)
rolling_mean has no cross-ticker contamination
ts_rank ranks within ticker history
cs_rank ranks within one date
rolling_corr is ticker-local
group_rank is date×group local
group_neutralize produces approximately zero group mean
safe_div produces NaN rather than inf on zero denominator
```

## Canonicalization tests

Same canonical form:

```text
add(a,b)
add(b,a)
```

Same canonical form:

```text
mul(a,b)
mul(b,a)
```

Different:

```text
sub(a,b)
sub(b,a)
```

Different:

```text
safe_div(a,b)
safe_div(b,a)
```

## Complexity tests

Test exact boundary and boundary+1 for:

```text
max_ast_depth
max_operator_nodes
max_rolling_nodes
max_binary_nodes
max_group_nodes
```

## Leakage tests

Run existing dynamic leakage checks on:

```text
lag
delta
rolling_mean
ts_rank
rolling_corr
group_rank
group_neutralize
```

All must pass.

## Backward compatibility tests

Legacy factors must match AST execution numerically for at least:

```text
return rank
volatility zscore
rolling-mean momentum
volume shock
earnings yield
asset growth
news volume
current multiplicative interaction
```

Use:

```python
np.allclose(..., equal_nan=True)
```

where appropriate.

---

# 35. Configuration

Add a dedicated section:

```yaml
dsl:
  enabled: true

  allowed_feature_windows: [1, 5, 10, 20, 60]

  max_ast_depth: 5
  max_operator_nodes: 6
  max_rolling_nodes: 3
  max_binary_nodes: 2
  max_group_nodes: 2

  min_group_size: 5

  safe_div_epsilon: 1.0e-12

  allowed_groups:
    - sector
    - industry
    - subindustry
```

Only include group names actually supported by current data.

If only `sector` is available, configure only `sector`.

---

# 36. Documentation

Update:

```text
README.md
docs/architecture.md
```

Add:

```text
docs/dsl_v2.md
```

`docs/dsl_v2.md` should explain:

1. primitive features;
2. operator categories;
3. time-series vs cross-sectional semantics;
4. group semantics;
5. canonical expressions;
6. complexity limits;
7. examples;
8. leakage guarantees;
9. unsupported operators;
10. migration from legacy FactorSpec.

---

# 37. Research-loop integration

Macro/Micro/Cross architecture remains conceptually unchanged.

Macro Brain still decides:

```text
IMPROVE
COMBINE
PIVOT
STOP
```

Micro Brain changes from:

```text
choose flat FactorSpec fields
```

to:

```text
construct / mutate bounded AST
```

Cross Brain still records:

```text
GOOD lessons
BAD lessons
repair ideas
```

and should additionally log:

```text
canonical expression
expression hash
AST depth
operator count
rolling operator count
group operator count
```

---

# 38. Search-space discipline

Do not increase candidate budget just because the DSL is larger.

After P0/P1, rerun the same 60-candidate deterministic adaptive protocol on research-visible data only.

Compare against V1:

```text
valid information per 10 proposals
duplicate rate
invalid proposal rate
time to first Parent
time to first Elite
Parent count
Elite count
mechanism coverage
operator diversity
AST complexity distribution
fold-sign stability
turnover
cost sensitivity
```

Do not use 2016.

Only after deterministic V2 works should the equal-budget Luna comparison be rerun.

Do not simultaneously increase:

```text
candidate budget
token budget
round count
```

The experiment should isolate the effect of the expanded DSL.

---

# 39. 2016 firewall

This change must not alter final holdout policy.

Research:

```text
2010–2015 only
```

Final test:

```text
2016 only after freeze
```

The following must never use 2016:

```text
operator selection
DSL design tuning
Macro decisions
Micro mutations
GOOD/BAD memory
complexity thresholds
group thresholds
candidate gates
library filtering
```

---

# 40. Implementation order

## Phase A — AST foundation

Implement:

```text
Expr
Feature
Constant
Op
serialization
canonicalization
hashing
complexity
required_history
```

Compile legacy factors into AST.

Run existing tests.

## Phase B — operator registry

Implement:

```text
OperatorSpec
registry
lookup
parameter validation
scope metadata
complexity metadata
```

Refactor builder to evaluate through the registry.

Run backward-compatibility tests.

## Phase C — arithmetic P1

Implement:

```text
add
sub
mul
safe_div
neg
abs
clip
```

## Phase D — time-series P1

Implement:

```text
lag
delta
rolling_sum
rolling_mean
rolling_std
rolling_min
rolling_max
ts_rank
ts_zscore
ewma
decay_linear
rolling_corr
rolling_cov
```

Add semantic and leakage tests.

## Phase E — cross-sectional P1

Standardize:

```text
cs_rank
cs_zscore
winsorize
winsorize_zscore
```

Use aliases only for migration.

## Phase F — group operators

Inspect real security-master columns.

Implement supported canonical groups.

Then add:

```text
group_rank
group_zscore
group_neutralize
```

## Phase G — LLM schema

Move structured proposals from flat recipes to AST JSON.

Generate allowed-operator documentation from the registry.

Keep local validation authoritative.

## Phase H — end-to-end smoke test

Verify:

```text
Macro
→ Micro AST
→ static checks
→ dynamic leakage
→ builder
→ evaluator
→ Parent/Elite
→ Cross memory
```

## Phase I — equal-budget research experiment

Run the same 60-candidate research-only protocol.

Do not touch 2016.

---

# 41. Definition of Done

P0/P1 is complete when:

- the DSL is represented by a typed AST;
- legacy FactorSpecs compile to the AST;
- one deterministic builder executes all factors;
- operators are defined in one authoritative registry;
- primitive windows and transform windows are separate;
- arithmetic composition works;
- all required P1 time-series operators work;
- `ts_rank` and `cs_rank` have explicit distinct semantics;
- group-relative cross-sectional operators work on actual available metadata;
- AST complexity is bounded;
- canonicalization and hashes prevent duplicates;
- static leakage checks operate on AST structure;
- truncation and future-noise tests still pass;
- the LLM can only emit schema-valid AST JSON;
- arbitrary Python cannot be emitted or executed;
- V1 factor outputs remain backward compatible;
- the adaptive research loop works end-to-end;
- the 2016 holdout remains untouched.

---

# 42. Design philosophy

Prefer:

```text
small composable operators
clear semantics
point-in-time correctness
typed validation
explicit scope
auditability
canonical formulas
structural simplicity
```

over:

```text
large indicator count
arbitrary formulas
maximum search breadth
TA indicator proliferation
clever but opaque expressions
```

The goal is not to create the largest possible DSL.

The goal is to create the smallest language that lets the autonomous researcher express materially different economic hypotheses while keeping every candidate safe, interpretable, reproducible, and testable.

Final engineering rule:

> Add an abstraction only when it increases factor expressiveness without weakening point-in-time safety or auditability.

P0/P1 should leave a clean foundation for later TA-Lib macros, rolling regression, distribution-shape operators, and controlled conditional expressions, but those are explicitly outside this phase.
