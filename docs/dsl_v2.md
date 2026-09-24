# Factor DSL V2

Factor DSL V2 is a small typed expression language layered beneath the existing research
metadata. The LLM may propose a JSON AST, but it never emits executable Python. Legacy flat
`FactorSpec` recipes are compiled into the same AST before execution, so there is one
authoritative builder path.

## Expression nodes

```text
Feature(name, window=None)
Constant(value)
Op(name, args, params)
```

Examples:

```text
cs_rank(return[20])
cs_rank(safe_div(return[20],volatility[20]))
cs_rank(mul(return[20],after_tax_roe))
group_neutralize(operating_margin,group=sector)
```

Every node supports JSON serialization, canonical serialization, complexity, required
history, required feature discovery, AST depth, operator counts, and a SHA-256 expression
hash. `add` and `mul` canonicalize their children in sorted order; `sub` and `safe_div`
preserve argument order.

## Operator registry

`src/factors/operator_registry.py` is the single source of truth for operator name, scope,
arity, parameters, complexity cost, minimum-history rule, and deterministic implementation.
The recursive builder looks up every `Op` in this registry.

The current P1 set is:

- Arithmetic: `add`, `sub`, `mul`, `safe_div`, `neg`, `abs`, `clip`.
- Time series: `lag`, `delta`, `rolling_sum`, `rolling_mean`, `rolling_std`,
  `rolling_min`, `rolling_max`, `ts_rank`, `ts_zscore`, `ewma`, `decay_linear`,
  `rolling_corr`, `rolling_cov`.
- Cross section: `cs_rank`, `cs_zscore`, `winsorize`, `winsorize_zscore`.
- Group cross section: `group_rank`, `group_zscore`, `group_neutralize`.

Ticker-local operators never mix symbols. Cross-sectional operators run within one date.
Group operators run within one date and one available security-master group. This dataset
supports `sector` and `subindustry`; `industry` is intentionally not invented because it is
not an actual merged panel field.

`safe_div` returns NaN when the denominator is within the configured epsilon of zero.
Trailing rolling operators use `min_periods=window` and do not use centered windows.

## Validation and complexity

AST validation rejects unknown features/operators/groups, negative lags, invalid parameters,
future-return leaves, unsupported windows, and structural limits. The default limits are in
`config.yaml`:

```yaml
dsl:
  max_ast_depth: 5
  max_operator_nodes: 6
  max_rolling_nodes: 3
  max_binary_nodes: 2
  max_group_nodes: 2
```

The AST complexity score is the sum of operator costs. Complexity is an integrity and
selection input; it is never sufficient to promote a factor.

## Legacy migration

The old fields (`base_feature`, `window`, `ts_operator`, `cs_operator`, and
`interaction_feature`) remain accepted. `legacy_spec_to_expr()` maps them to `Feature` and
`Op` nodes, including the old `vol_adjust` semantics as `safe_div(return, volatility)`.
`FactorBuilder.build()` always executes `FactorSpec.effective_expression`, so legacy and V2
recipes share the same recursive evaluator.

## LLM boundary

The legacy LLM path supports the existing flat recipe union and
`recipe_kind="expression"`. The adaptive Micro path uses a narrower expression-only schema.
Windowed and scalar feature nodes are separate types, and every operator family has its own
parameter model, so `after_tax_roe(window=20)`, `return(window=0)`, and
`cs_rank(window=20)` fail schema parsing. The operator catalog in the prompt is generated
from the registry, and a consistency test prevents the registry and proposal schema from
drifting. Local AST validation, plan binding, formula deduplication, complexity limits, and
dynamic leakage checks remain authoritative.

## Leakage and reproducibility

The AST contains no arbitrary execution primitive. Static checks reject future-return
features and negative lags. Existing truncation-invariance and future-noise tests run against
AST-built signals. Canonical formulas and expression hashes are written into every
`ExperimentRecord` together with AST depth and operator counts.

The DSL change does not change the holdout policy: research and DSL decisions use 2010–2015;
2016 remains locked until a library is frozen.
