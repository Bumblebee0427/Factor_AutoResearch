# Final factor research experiments

Run: `final_gpt6_20260924_v1` · Git commit: `13bc79740a5922183035670e3c652fe1fe9d6b38` · Config SHA-256: `6022fa7f9a4aa821f526fc4ff9562b31a758794364afda9ca7d9f76ec5985137`

Protocol SHA-256: `4de3fb0b223ba2973d597e8d84d2e03dc37c2165422b79de14cb32697b6aaa98` · Seed: `42` · Budget: `60` per arm

Research: 2010–2015; validation folds: 2013, 2014, 2015. Holdout evaluated: **false**.

Model comparisons are descriptive single-budget research runs, not statistically powered estimates of expected performance.

## Run status

| Arm | Status | Macro / Micro |
| --- | --- | --- |
| fixed_deterministic | complete | —/— |
| adaptive_deterministic | complete | —/— |
| luna_luna | complete | Luna High/Luna High |
| sol_sol | complete | Sol High/Sol High |
| sol_luna | complete | Sol High/Luna High |
| luna_sol | complete | Luna High/Sol High |

## A · Initial vs post-feedback

| Arm | Cohort | N | Median IC | P75 IC | Median NW t | All-positive folds | Median high-cost Sharpe | Parent rate | Elite rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_deterministic | INITIAL_COHORT | 16 | -0.000 | 0.013 | -0.127 | 0.267 | -3.984 | 0.500 | 0.000 |
| adaptive_deterministic | POST_FEEDBACK_COHORT | 44 | 0.003 | 0.007 | 0.081 | 0.073 | -11.880 | 0.545 | 0.000 |
| luna_luna | INITIAL_COHORT | 16 | -0.000 | 0.013 | -0.127 | 0.267 | -3.984 | 0.500 | 0.000 |
| luna_luna | POST_FEEDBACK_COHORT | 44 | -0.003 | 0.002 | -0.351 | 0.053 | -2.883 | 0.295 | 0.000 |
| luna_sol | INITIAL_COHORT | 16 | -0.000 | 0.013 | -0.127 | 0.267 | -3.984 | 0.500 | 0.000 |
| luna_sol | POST_FEEDBACK_COHORT | 44 | -0.003 | 0.007 | -0.410 | 0.075 | -4.199 | 0.409 | 0.023 |
| sol_luna | INITIAL_COHORT | 16 | -0.000 | 0.013 | -0.127 | 0.267 | -3.984 | 0.500 | 0.000 |
| sol_luna | POST_FEEDBACK_COHORT | 44 | -0.004 | 0.008 | -0.411 | 0.128 | -2.561 | 0.318 | 0.000 |
| sol_sol | INITIAL_COHORT | 16 | -0.000 | 0.013 | -0.127 | 0.267 | -3.984 | 0.500 | 0.000 |
| sol_sol | POST_FEEDBACK_COHORT | 44 | -0.000 | 0.005 | -0.107 | 0.025 | -2.717 | 0.432 | 0.000 |

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
| luna_luna | 60 | 3 | — | 14 | -0.002 | -3.420 | 0.000 |

At equal 60-candidate budgets, Luna/Luna produced 14 unique Parent clusters versus 17 for the adaptive deterministic policy. Its median IC was -0.002 versus 0.001; this run does not show a Luna/Luna search-efficiency gain by those criteria.

## D · Model roles, tokens and cost

| Arm | Macro model | Micro model | Total tokens | Est. USD | Unique clusters | Clusters / 100k tokens | Median IC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| luna_luna | gpt-6-luna | gpt-6-luna | 390797 | 0.066 | 14 | 3.582 | -0.002 |
| sol_sol | gpt-6-sol | gpt-6-sol | 565518 | 1.663 | 21 | 3.713 | -0.000 |
| sol_luna | gpt-6-sol | gpt-6-luna | 514106 | 0.686 | 15 | 2.918 | -0.004 |
| luna_sol | gpt-6-luna | gpt-6-sol | 584347 | 0.995 | 20 | 3.423 | -0.002 |

Pricing as of 2026-09-24 from [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) and [Sol](https://developers.openai.com/api/docs/models/gpt-6-sol) model pages. Costs are estimates from recorded input, cache-read, cache-write and output tokens, not invoice amounts.

### Search yield and robustness by model role

| Arm | First Elite | Parents | Clusters | Elites | Valid / 10 | Invalid evaluated | Raw rejection | Median high-cost Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| luna_luna | — | 21 | 14 | 0 | 8.833 | 0.100 | 0.000 | -3.420 |
| sol_sol | — | 27 | 21 | 0 | 9.167 | 0.033 | 0.000 | -3.172 |
| sol_luna | — | 22 | 15 | 0 | 9.000 | 0.083 | 0.000 | -2.951 |
| luna_sol | 41 | 26 | 20 | 1 | 9.167 | 0.050 | 0.024 | -4.150 |

With Sol in Micro, the two assignments found 21 and 20 independent Parent clusters; with Luna in Micro, they found 15 and 14. In this single run, spending on Micro appears more associated with cluster yield than spending on Macro. The only Elite appeared at candidate 41 in Luna Macro/Sol Micro. This is an observed allocation pattern, not a causal or statistically powered estimate.

### API usage outside completed arm trajectories

| Source | Model | Recorded calls | Tokens | Est. USD | Unmeasured attempts |
| --- | --- | --- | --- | --- | --- |
| model_smoke_check | gpt-6-luna | 1 | 7327 | 0.001 | 0 |
| model_smoke_check | gpt-6-sol | 1 | 7120 | 0.022 | 0 |
| aborted_luna_sol_macro | gpt-6-luna | 0 | 0 | 0.000 | 1 |
| aborted_luna_sol_micro | gpt-6-sol | 1 | 24714 | 0.083 | 0 |

Known API cost including the recorded smoke checks and aborted-attempt usage: $3.516. A model attempt without a response/usage record may have incurred additional charges; this is a lower-bound estimate.

## Repair evidence

| Arm | Parent | Child | Target | Δ IC | Δ high-cost Sharpe | Target improved | Collateral damage | Repair succeeded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_deterministic | r1_pivot_price_reversal_30 | r9_repair_r1_pivot_price_reversal_30_0 | cost_sensitivity | -0.012 | 11.898 | True | True | False |
| adaptive_deterministic | r1_pivot_price_reversal_30 | r9_repair_r1_pivot_price_reversal_30_1 | cost_sensitivity | -0.031 | 11.798 | True | True | False |
| adaptive_deterministic | r1_pivot_price_reversal_30 | r9_repair_r1_pivot_price_reversal_30_2 | cost_sensitivity | -0.012 | 10.233 | True | True | False |
| luna_luna | r1_pivot_price_reversal_30 | ewma_smoothed_short_reversal | cost_sensitivity | -0.028 | 8.023 | True | True | False |
| luna_luna | r1_pivot_price_reversal_30 | rolling_mean_short_reversal | cost_sensitivity | -0.031 | 5.220 | True | True | False |
| luna_luna | r1_pivot_price_reversal_30 | five_day_reversal_horizon | cost_sensitivity | -0.031 | 5.178 | True | True | False |
| luna_luna | r1_pivot_price_reversal_30 | linear_decay_short_reversal | cost_sensitivity | -0.033 | 5.617 | True | True | False |
| luna_luna | r5_pivot_fundamental_quality_23 | sector_relative_roe_quality | low_statistical_significance | -0.024 | -1.657 | — | — | — |
| luna_luna | r5_pivot_fundamental_quality_23 | ranked_profitability_composite | low_statistical_significance | -0.024 | -0.704 | — | — | — |
| luna_luna | r5_pivot_fundamental_quality_23 | sector_relative_profitability_composite | low_statistical_significance | -0.028 | -1.882 | — | — | — |
| luna_luna | r1_pivot_price_reversal_30 | r1_reversal_mean20_cost_repair | cost_sensitivity | — | — | False | False | False |
| luna_luna | r1_pivot_price_reversal_30 | r1_reversal_ewma20_cost_repair | cost_sensitivity | — | — | False | False | False |
| luna_luna | r1_pivot_price_reversal_30 | r1_reversal_rank_smooth20_cost_repair | cost_sensitivity | — | — | False | False | False |
| luna_luna | r1_pivot_price_reversal_30 | r13_repair_r1_pivot_price_reversal_30_0 | cost_sensitivity | -0.012 | 11.898 | True | True | False |
| luna_luna | r5_pivot_fundamental_quality_23 | roe_smoothed_persistence | low_statistical_significance | 0.001 | 0.182 | — | — | — |
| luna_luna | r5_pivot_fundamental_quality_23 | roe_historical_strength | low_statistical_significance | -0.018 | -0.353 | — | — | — |
| luna_luna | r5_pivot_fundamental_quality_23 | improving_roe_trend | low_statistical_significance | -0.019 | -0.469 | — | — | — |
| luna_luna | r5_pivot_fundamental_quality_23 | standalone_profit_margin_quality | — | -0.046 | -0.971 | — | — | — |
| luna_luna | roe_smoothed_persistence | g16_roe_rolling_mean_quality | low_statistical_significance | -0.015 | -0.171 | — | — | — |
| luna_luna | roe_smoothed_persistence | g16_roe_trailing_floor_quality | low_statistical_significance | -0.011 | -0.338 | — | — | — |
| luna_luna | roe_smoothed_persistence | g16_roe_consistency_quality | low_statistical_significance | -0.037 | -0.911 | — | — | — |
| luna_luna | volatility_contraction_17_c | vol_contraction_ewma10_repair | excessive_turnover | -0.009 | 4.021 | — | — | — |
| luna_luna | volatility_contraction_17_c | vol_contraction_mean10_repair | cost_sensitivity | -0.014 | 3.913 | — | — | — |
| luna_luna | volatility_contraction_17_c | vol_contraction_ewma20_slow_repair | excessive_turnover | -0.020 | 4.868 | — | — | — |
| luna_sol | r1_pivot_price_reversal_30 | reversal_ewma_three_day | cost_sensitivity | -0.031 | 6.600 | — | — | — |
| luna_sol | r1_pivot_price_reversal_30 | reversal_persistent_relative_losers | cost_sensitivity | -0.033 | 3.891 | — | — | — |
| luna_sol | r1_pivot_price_reversal_30 | reversal_ten_day_pressure | weak_monetization | -0.028 | 8.225 | — | — | — |
| luna_sol | r1_pivot_price_reversal_30 | reversal_smoothed_within_sector | cost_sensitivity | -0.031 | 1.935 | — | — | — |
| luna_sol | r1_pivot_price_reversal_30 | recent_selloff_extreme_reversal | cost_sensitivity | -0.013 | 8.442 | True | True | False |
| luna_sol | r1_pivot_price_reversal_30 | recent_upside_spike_reversal | cost_sensitivity | -0.031 | 5.982 | True | True | False |
| luna_sol | r1_pivot_price_reversal_30 | profitability_anchored_reversal | cost_sensitivity | — | — | False | False | False |
| luna_sol | r1_pivot_price_reversal_30 | r10_repair_r1_pivot_price_reversal_30_0 | cost_sensitivity | -0.012 | 11.898 | True | True | False |
| luna_sol | r1_pivot_price_reversal_30 | downside_pressure_ewma_reversal | cost_sensitivity | -0.018 | 8.225 | True | True | False |
| luna_sol | r1_pivot_price_reversal_30 | five_day_cumulative_reversal | cost_sensitivity | -0.031 | 5.178 | True | True | False |
| luna_sol | r1_pivot_price_reversal_30 | r12_repair_r1_pivot_price_reversal_30_1 | cost_sensitivity | -0.031 | 11.798 | True | True | False |
| luna_sol | r1_pivot_price_reversal_30 | r12_repair_r1_pivot_price_reversal_30_2 | cost_sensitivity | -0.012 | 10.233 | True | True | False |
| luna_sol | underfollowed_profitable_firms | slow_attention_profitability_spread | low_statistical_significance | 0.001 | 0.420 | — | — | — |
| luna_sol | slow_attention_profitability_spread | sector_relative_underfollowed_profitability | weak_predictive_signal | 0.001 | -0.048 | False | True | False |
| luna_sol | slow_attention_profitability_spread | slow_attention_earnings_yield_spread | weak_predictive_signal | -0.010 | -0.745 | — | — | — |
| luna_sol | sector_relative_underfollowed_profitability | operating_margin_sector_underattention | low_statistical_significance | -0.013 | -0.223 | False | True | False |
| luna_sol | sector_relative_underfollowed_profitability | roe_weighted_sector_underattention | low_statistical_significance | 0.000 | 0.000 | False | False | False |
| luna_sol | sector_relative_underfollowed_profitability | net_margin_sector_underattention | low_statistical_significance | -0.012 | -0.330 | False | True | False |
| luna_sol | sector_relative_underfollowed_profitability | sector_quality_global_underattention | low_statistical_significance | -0.022 | -1.280 | — | — | — |
| luna_sol | r5_pivot_fundamental_quality_23 | persistent_roe_quality_floor | low_statistical_significance | 0.010 | 0.019 | — | — | — |
| luna_sol | r5_pivot_fundamental_quality_23 | sector_relative_roe_quality | low_statistical_significance | -0.024 | -1.657 | — | — | — |
| luna_sol | r5_pivot_fundamental_quality_23 | roe_less_asset_expansion | — | -0.013 | -0.884 | — | — | — |
| sol_luna | r1_pivot_price_reversal_30 | reversal_sma5_smoothing | cost_sensitivity | -0.031 | 5.220 | True | True | False |
| sol_luna | r1_pivot_price_reversal_30 | reversal_ewma5_smoothing | cost_sensitivity | -0.028 | 8.023 | True | True | False |
| sol_luna | r1_pivot_price_reversal_30 | reversal_decay5_smoothing | cost_sensitivity | -0.033 | 5.617 | True | True | False |
| sol_luna | r1_pivot_price_reversal_30 | reversal_five_day_horizon_exploration | cost_sensitivity | -0.031 | 5.178 | True | True | False |
| sol_luna | r5_pivot_fundamental_quality_23 | roe_sector_relative_rank_repair | low_statistical_significance | -0.024 | -1.657 | — | — | — |
| sol_luna | r5_pivot_fundamental_quality_23 | roe_profit_margin_rank_blend_repair | low_statistical_significance | -0.024 | -0.704 | — | — | — |
| sol_luna | r3_pivot_price_volume_17 | volume_shock_smoothed_persistence_repair | cost_sensitivity | -0.013 | 7.745 | — | — | — |
| sol_luna | r3_pivot_price_volume_17 | volume_shock_temporal_persistence_rank | cost_sensitivity | 0.002 | -0.377 | — | — | — |
| sol_luna | r1_pivot_price_reversal_30 | price_reversal_ewma10_repair | cost_sensitivity | -0.008 | 10.303 | True | True | False |
| sol_luna | r1_pivot_price_reversal_30 | price_reversal_mean10_repair | cost_sensitivity | -0.002 | 9.302 | True | True | False |
| sol_luna | r1_pivot_price_reversal_30 | price_reversal_rank_ewma10_repair | cost_sensitivity | -0.002 | 9.913 | True | True | False |
| sol_luna | r1_pivot_price_reversal_30 | price_reversal_return5_mean10_repair | cost_sensitivity | -0.007 | 11.249 | True | True | False |
| sol_luna | volatility_relative_rank20_60 | volatility_relative_rank_ewma10_repair | cost_sensitivity | -0.004 | 2.924 | — | — | — |
| sol_luna | volatility_relative_rank20_60 | volatility_smoothed_relative_rank_repair | cost_sensitivity | -0.003 | 3.629 | — | — | — |
| sol_luna | volatility_smoothed_relative_rank_repair | sector_rank_smoothed_volatility_repair | low_statistical_significance | -0.013 | -1.649 | — | — | — |
| sol_luna | volatility_smoothed_relative_rank_repair | sector_zscore_smoothed_volatility_repair | low_statistical_significance | -0.012 | -1.470 | — | — | — |
| sol_luna | r1_pivot_price_reversal_30 | reversal_mean20_smoother | cost_sensitivity | -0.012 | 10.440 | True | True | False |
| sol_luna | r1_pivot_price_reversal_30 | reversal_decay20_smoother | cost_sensitivity | -0.005 | 10.230 | True | True | False |
| sol_luna | r1_pivot_price_reversal_30 | reversal_winsorized_mean10 | cost_sensitivity | -0.002 | 9.285 | True | True | False |
| sol_luna | r1_pivot_price_reversal_30 | reversal_smoothed_return10_exploration | cost_sensitivity | -0.007 | 11.248 | True | True | False |
| sol_sol | r1_pivot_price_reversal_30 | smoothed_daily_reversal_ewma5 | cost_sensitivity | -0.003 | 9.199 | True | True | False |
| sol_sol | r1_pivot_price_reversal_30 | smoothed_daily_reversal_mean10 | cost_sensitivity | -0.002 | 9.302 | True | True | False |
| sol_sol | r1_pivot_price_reversal_30 | weekly_return_reversal | cost_sensitivity | 0.001 | 8.189 | True | False | True |
| sol_sol | r1_pivot_price_reversal_30 | smoothed_daily_reversal_ewma10 | cost_sensitivity | -0.008 | 10.303 | True | True | False |
| sol_sol | r5_pivot_fundamental_quality_23 | roe_rolling_mean_20_quality | low_statistical_significance | -0.010 | -0.585 | — | — | — |
| sol_sol | r5_pivot_fundamental_quality_23 | roe_rolling_floor_60_quality | low_statistical_significance | 0.010 | -0.589 | — | — | — |
| sol_sol | r1_pivot_price_reversal_30 | upside_spike_reversal_max20 | cost_sensitivity | -0.017 | 11.207 | True | True | False |
| sol_sol | r1_pivot_price_reversal_30 | persistent_down_day_breadth_reversal | cost_sensitivity | -0.022 | 7.529 | True | True | False |
| sol_sol | r1_pivot_price_reversal_30 | volatility_scaled_weekly_reversal | cost_sensitivity | -0.028 | 5.314 | True | True | False |
| sol_sol | r1_pivot_price_reversal_30 | sector_relative_weekly_price_reversal | cost_sensitivity | -0.031 | 2.481 | True | True | False |
| sol_sol | r1_pivot_price_reversal_30 | selloff_only_daily_reversal | cost_sensitivity | -0.006 | 7.423 | True | True | False |
| sol_sol | r1_pivot_price_reversal_30 | recent_selloff_event_reversal | cost_sensitivity | -0.017 | 8.606 | True | True | False |
| sol_sol | r1_pivot_price_reversal_30 | upside_only_daily_reversal | cost_sensitivity | -0.001 | 4.983 | True | False | True |
| sol_sol | r1_pivot_price_reversal_30 | r21_repair_r1_pivot_price_reversal_30_0 | cost_sensitivity | -0.012 | 11.898 | True | True | False |

For the adaptive deterministic arm, the main Elite bottleneck was high-cost Sharpe: 31/32 common Parents failed that gate. Targeted repairs improved their intended metric in 3/3 paired attempts, but 3/3 caused collateral damage. That arm had no Elite. Its post-feedback cohort's median IC increased slightly, while its median high-cost Sharpe and all-positive-fold rate worsened, so iteration did not establish a broad quality improvement.


## Representative lineages

### Elite

Factor: `slow_attention_profitability_spread` (luna_sol); tier: ELITE; action: IMPROVE.

Hypothesis: Profitable firms with persistently low news coverage may be underappreciated; comparing profitability and slow-moving coverage directly may produce a steadier signal than multiplying their ranks.

Formula: `sub(cs_rank(after_tax_roe),cs_rank(news_volume[60]))`; parents: underfollowed_profitable_firms.

IC 0.020; NW t 1.079; positive folds 3; high-cost Sharpe 0.733; turnover 0.023; failed gates none.

Validation-fold ICs: validate_2013=0.047, validate_2014=0.003, validate_2015=0.010. Multi-horizon ICs: 10d=0.032, 1d=0.017, 5d=0.020. The NW t-stat is only slightly above the pre-committed threshold; this remains a research candidate, not established out-of-sample alpha.

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
- `figures/figure3_agent_comparison.png`
- `figures/figure4_model_roles.png`
- `figures/figure5_elite_gates.png`

## Audit trail

Manifest: `/Users/bumblebee/Documents/Project/Factor_AutoResearch/artifacts/experiments/final_experiments/final_gpt6_20260924_v1/manifest.json`
Tables: `/Users/bumblebee/Documents/Project/Factor_AutoResearch/artifacts/experiments/final_experiments/final_gpt6_20260924_v1/tables`
Machine summary: `/Users/bumblebee/Documents/Project/Factor_AutoResearch/artifacts/experiments/final_experiments/final_gpt6_20260924_v1/final_experiment_summary.json`

`luna_sol` recovered from 50 committed candidates at round 18; aborted tail archived at `artifacts/experiments/final_experiments/final_gpt6_20260924_v1/luna_sol_aborted_partial`. Original run commit `13bc79740a5922183035670e3c652fe1fe9d6b38`; recovery guard commit `d35b5784cc5b773cf1a5278912d1c11712ae62fa`. The guard changed only malformed deterministic crossover fallback behavior; evaluator, gates and research panel remained fixed.

1 candidate(s) passed all pre-committed Elite gates, and the corresponding research-only candidate library is saved under its arm's `adaptive/` directory. This experiment did not freeze a final library or open the 2016 holdout; the one-shot holdout is a separate post-freeze step.

## Post-research final holdout

The selected research library and its `luna_sol` run configuration were subsequently frozen, and the 2016 holdout was opened exactly once. The detailed result is in [final_holdout_results.md](final_holdout_results.md); the research-period tables above are unchanged.
