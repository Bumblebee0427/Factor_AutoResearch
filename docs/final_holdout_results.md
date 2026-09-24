# Final 2016 holdout: one-shot evaluation

The final research library was frozen before opening the physically separate 2016 panel. The source was the completed final_gpt6_20260924_v1 run, luna_sol arm (Luna High Macro, Sol High Micro). Its existing library-selection audit selected one research-period Elite, slow_attention_profitability_spread, with no failed Elite gates. The frozen formula is sub(cs_rank(after_tax_roe),cs_rank(news_volume[60])). No factor was selected using 2016.

The command python scripts/final_holdout.py completed successfully once on 2026-09-24. Research history supplied rolling-feature warm-up; the evaluator scored the frozen factor on 2016. The result is preserved in [holdout_results.json](../outputs/holdout/holdout_results.json), with the pre-holdout [freeze manifest](../artifacts/frozen/freeze_manifest.json) and post-run [audit manifest](../outputs/holdout/holdout_run_manifest.json).

| Metric | Research validation, 2013–2015 | Final holdout, 2016 |
| --- | ---: | ---: |
| Mean five-day RankIC | 0.019945 | -0.004874 |
| Newey–West IC t-statistic | 1.079 | -0.686 |
| Net Sharpe, 10 bp cost | 1.034 | -1.317 |
| High-cost Sharpe, 25 bp cost | 0.733 | -1.637 |
| Mean daily turnover | 0.02335 | 0.02155 |
| Maximum drawdown | -0.06881 | -0.05000 |

Research figures are from the frozen factor's three validation folds: mean IC, t-statistic, Sharpes, and turnover are means of fold values; research drawdown is the worst fold. The 2016 column is one holdout fold. Its net annualized return was -0.03331, quantile monotonicity -0.70, five-day IC hit rate 0.478, and five-day scoring used 109,999 stock-day observations.

| 2016 IC horizon | Mean RankIC | Newey–West t-statistic | IC observations |
| --- | ---: | ---: | ---: |
| 1 day | -0.001046 | -0.264 | 251 |
| 5 days | -0.004874 | -0.686 | 247 |
| 10 days | -0.011027 | -1.007 | 242 |

The factor cleared the pre-committed research gates but did not generalize strongly to the untouched 2016 holdout: IC was negative at all reported horizons and both after-cost Sharpe measures were negative, while turnover remained similar. This is one final out-of-sample observation, not a new selection round; the three-fold Elite gate is not reapplied to a single year. The frozen library, formula, evaluator, gates, and cost assumptions were not changed after seeing 2016.
