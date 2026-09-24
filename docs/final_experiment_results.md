# Final factor research experiments

Run: `final_20260924_v1` · Git commit: `999914b1f4fdaa5e349d791a12d7e5813adf20f7` · Config SHA-256: `f4489390b74e381e8b9b5ddf7090a9223023f522a0f54ac3cf389eae4c14a136`

Protocol SHA-256: `4de3fb0b223ba2973d597e8d84d2e03dc37c2165422b79de14cb32697b6aaa98` · Seed: `42` · Budget: `60` per arm

Research: 2010–2015; validation folds: 2013, 2014, 2015. Holdout evaluated: **false**.

Model comparisons are descriptive single-budget research runs, not statistically powered estimates of expected performance.

## Run status

| Arm | Status | Macro / Micro |
| --- | --- | --- |
| fixed_deterministic | complete | —/— |
| adaptive_deterministic | complete | —/— |
| luna_luna | not run | Luna High/Luna High |
| sol_sol | not run | Sol High/Sol High |
| sol_luna | not run | Sol High/Luna High |
| luna_sol | not run | Luna High/Sol High |

The model arms were not executed: this environment did not grant permission to send project-derived research-state summaries and factor proposals to the OpenAI API. Their absent results are not zero-valued outcomes. No LLM advantage or role allocation can be inferred from this partial run.

## A · Initial vs post-feedback

| Arm | Cohort | N | Median IC | P75 IC | Median NW t | All-positive folds | Median high-cost Sharpe | Parent rate | Elite rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_deterministic | INITIAL_COHORT | 16 | -0.000 | 0.013 | -0.127 | 0.267 | -3.984 | 0.500 | 0.000 |
| adaptive_deterministic | POST_FEEDBACK_COHORT | 44 | 0.003 | 0.007 | 0.081 | 0.073 | -11.880 | 0.545 | 0.000 |

## B · Architecture

| Arm | N | Median IC | Median high-cost Sharpe | Common Parents | Common Elites | Unique clusters | Clusters / 10 | Valid / 10 | Duplicate rate | Invalid rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_deterministic | 33 | -0.001 | -2.863 | 11 | 0 | 6 | 1.818 | 8.378 | 0.108 | 0.054 |
| adaptive_deterministic | 60 | 0.001 | -9.489 | 32 | 0 | 17 | 2.833 | 9.333 | 0.000 | 0.000 |

Both arms had a 60-candidate cap, but the fixed generator exhausted novel proposals after 33 evaluated candidates. Counts therefore have unequal denominators; per-10 rates are shown, but this single run does not isolate architecture from proposal coverage. The fixed-generation arm is classified post hoc using the current Parent/Elite gates; its original search decisions are preserved.

## C · Deterministic vs Luna High

| Arm | N | First Parent | First Elite | Unique clusters | Median IC | Median high-cost Sharpe | Raw proposal rejection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_deterministic | 60 | 3 | — | 17 | 0.001 | -9.489 | 0.000 |

## D · Model roles, tokens and cost

No completed arms are available for this comparison.

Pricing as of 2026-09-24 from [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) and [Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) model pages. Costs are estimates from recorded token usage, not invoice amounts.

## Repair evidence

| Arm | Parent | Child | Target | Δ IC | Δ high-cost Sharpe | Target improved | Collateral damage | Repair succeeded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_deterministic | r1_pivot_price_reversal_30 | r9_repair_r1_pivot_price_reversal_30_0 | cost_sensitivity | -0.012 | 11.898 | True | True | False |
| adaptive_deterministic | r1_pivot_price_reversal_30 | r9_repair_r1_pivot_price_reversal_30_1 | cost_sensitivity | -0.031 | 11.798 | True | True | False |
| adaptive_deterministic | r1_pivot_price_reversal_30 | r9_repair_r1_pivot_price_reversal_30_2 | cost_sensitivity | -0.012 | 10.233 | True | True | False |

The main Elite bottleneck was high-cost Sharpe: 31/32 common Parents failed that gate. Targeted repairs improved their intended metric in 3/3 paired attempts, but 3/3 caused collateral damage. No candidate passed every Elite gate. The post-feedback cohort's median IC increased slightly, while its median high-cost Sharpe and all-positive-fold rate worsened, so iteration did not establish a broad quality improvement.


## Representative lineages

### Successful Parent

Factor: `r1_pivot_price_reversal_30` (adaptive_deterministic); tier: PARENT; action: PIVOT.

Hypothesis: Very recent relative winners may reverse as temporary price pressure and liquidity demand dissipate.

Formula: `-1 * rank(return(1))`; parents: none.

IC 0.015; NW t 1.918; positive folds 3; high-cost Sharpe -12.474; turnover 0.771; failed gates high_cost_sharpe.

### Near Elite

Factor: `r1_pivot_price_reversal_30` (adaptive_deterministic); tier: PARENT; action: PIVOT.

Hypothesis: Very recent relative winners may reverse as temporary price pressure and liquidity demand dissipate.

Formula: `-1 * rank(return(1))`; parents: none.

IC 0.015; NW t 1.918; positive folds 3; high-cost Sharpe -12.474; turnover 0.771; failed gates high_cost_sharpe.

### Targeted Repair

Factor: `r9_repair_r1_pivot_price_reversal_30_0` (adaptive_deterministic); tier: PARENT; action: IMPROVE.

Hypothesis: Very recent relative winners may reverse as temporary price pressure and liquidity demand dissipate.

Formula: `-1 * rank(rolling_mean(return(20)))`; parents: r1_pivot_price_reversal_30.

IC 0.004; NW t 0.293; positive folds 2; high-cost Sharpe -0.576; turnover 0.051; failed gates mean_ic, tstat, positive_folds, high_cost_sharpe.

Repair target: cost_sensitivity; target improved: True; collateral damage: True; success: False.

## Figures

- `figures/figure1_feedback_cohorts.png`
- `figures/figure2_architecture.png`
- `figures/figure5_elite_gates.png`

## Audit trail

Manifest: `/Users/bumblebee/Documents/Project/Factor_AutoResearch/artifacts/experiments/final_experiments/final_20260924_v1/manifest.json`
Tables: `/Users/bumblebee/Documents/Project/Factor_AutoResearch/artifacts/experiments/final_experiments/final_20260924_v1/tables`
Machine summary: `/Users/bumblebee/Documents/Project/Factor_AutoResearch/artifacts/experiments/final_experiments/final_20260924_v1/final_experiment_summary.json`

The 2016 holdout remained unopened. An empty Elite archive does not create a final library.
