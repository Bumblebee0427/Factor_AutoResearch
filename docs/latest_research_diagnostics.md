# Latest elite-bottleneck diagnostics

## Status

The deterministic 60-candidate V3 run completed on the 2010–2015 research folds. The
2016 holdout was not loaded or evaluated; Elite/Parent thresholds, costs, folds, and the
60-candidate budget were unchanged. The Luna High arm is **incomplete**: its first
sandboxed attempt recorded 16 coverage-phase candidates, then could not resolve the API
host when it reached the first model request. No complete LLM comparison is available.

## Deterministic V3

| Measure | Result |
| --- | ---: |
| Candidates evaluated | 60 |
| First Parent | Candidate 3 |
| First Elite | Not reached |
| Parent candidates observed | 32 |
| Unique Parent clusters | 17 |
| Cluster / Parent ratio | 0.53 |
| Parent pool representatives | 17 |
| Mechanisms covered | 8 / 8 |
| Actions | 12 PIVOT, 6 COMBINE, 1 IMPROVE |
| Targeted repair attempts | 3 |
| Successful targeted repairs | 0 |
| Integrity failures | 0 |
| Duplicate formula ratio | 0.00 |
| Holdout evaluated | No |

### Independent Elite-gate failures among 32 Parents

| Gate | Parents failing |
| --- | ---: |
| High-cost Sharpe | 31 / 32 |
| Positive folds | 25 / 32 |
| Newey–West t-stat | 18 / 32 |
| Mean IC | 14 / 32 |
| Turnover | 0 / 32 |

Failures overlap: 19 Parents failed at least three gates, 7 failed exactly two, and 6
failed exactly one. This points to economics/statistical robustness rather than turnover
as the main bottleneck. In particular, all 11 CROSS_DOMAIN_REGIME Parents failed the
high-cost Sharpe gate and 10/11 failed positive-fold consistency.

The three cost-sensitivity repairs improved high-cost Sharpe versus their parent, but
all materially reduced mean IC, so none counted as successful. This is the intended
collateral-damage guard, not a reason to relax the Elite gates. The previous generic
stagnation-to-PIVOT override was removed after V2 diagnostics showed it prevented
repairable Parents from reaching IMPROVE; V3 now exercises targeted repair.

## Luna High attempt

The preserved partial run is at
`artifacts/experiments/adaptive_comparisons/elite_bottleneck_luna_high_v2/`. It contains
16 candidates from deterministic mechanism-coverage rounds (8 rounds); the Macro LLM
request was reached next, but DNS resolution failed inside the restricted execution
environment. There are no completed Luna responses or token usage in this artifact, so
these 16 candidates are **not** an LLM-vs-deterministic result. A full 60-candidate Luna
High run requires explicit approval to send the research state and generated prompts to
the OpenAI API endpoint; the network escalation was rejected by the execution reviewer.

## Reproducible artifacts

- Deterministic run: `artifacts/experiments/adaptive_comparisons/elite_bottleneck_deterministic_v3/`
- Gate counts: `artifacts/experiments/adaptive_comparisons/elite_bottleneck_deterministic_v3/deterministic/diagnostics/outputs/elite_bottleneck_summary.json`
- Per-candidate failures: `artifacts/experiments/adaptive_comparisons/elite_bottleneck_deterministic_v3/deterministic/diagnostics/outputs/tables/elite_gate_failures.csv`
- Mechanism breakdown: `artifacts/experiments/adaptive_comparisons/elite_bottleneck_deterministic_v3/deterministic/diagnostics/outputs/tables/elite_gate_failures_by_mechanism.csv`
- Distance to thresholds: `artifacts/experiments/adaptive_comparisons/elite_bottleneck_deterministic_v3/deterministic/diagnostics/outputs/tables/elite_gate_distance.csv`
