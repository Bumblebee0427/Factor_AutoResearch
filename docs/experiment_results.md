# Initial search-efficiency experiment

## Data split

- Research panel: 2010-03-30 through 2015-12-31, 694,799 stock-days and 497 symbols.
- Walk-forward validation years: 2013, 2014, and 2015, with expanding prior training history.
- Locked holdout: 2016-01-04 through 2016-12-30, 125,676 stock-days and 501 symbols.

The search and comparison scripts loaded only the research panel. The 2016 file was not
evaluated, exposed to the LLM, or used to change thresholds. It remains reserved for a
one-shot OOS evaluation after a non-empty factor library is frozen.

## Pre-committed comparison metrics

| Metric | Deterministic | GPT-5.6 Luna Low |
| --- | ---: | ---: |
| Candidate proposals | 37 | 55 |
| Candidates evaluated | 33 | 33 |
| Candidates to first Promote | Not reached | Not reached |
| Effective new information per 10 proposals | 8.6486 | 5.8182 |
| Duplicate-formula rate | 10.81% | 7.27% |
| Invalid/noncompliant rate | 2.70% | 34.55% |
| Promoted factors | 0 | 0 |
| Mean walk-forward IC-sign consistency | 0.7396 | 0.7396 |
| Positive IC in every fold | 6.25% | 6.25% |

“Effective new information” means a unique candidate that passed integrity checks and
produced complete walk-forward evidence. Rejected and duplicate proposals remain in the
denominator.

## Interpretation

The initial Luna Low arm did **not** improve search efficiency. It completed three API calls
and used 34,430 tokens, but all 18 raw LLM proposals failed deterministic recipe validation.
Most failures came from populating interaction-only fields in single-feature proposals; two
interaction recipes also supplied a window for a fundamental interaction feature. The local
validator prevented those malformed recipes from reaching the evaluator, and deterministic
fallback produced the same 33 evaluated candidates as the control arm.

This is a useful negative result rather than evidence that LLM-generated factors improve the
search. A credible next experiment should replace the flat nullable recipe schema with a
discriminated single-feature/interaction schema or add one bounded validation-feedback retry,
then repeat the same fixed comparison without changing gates or viewing the 2016 holdout.

No factors passed every promotion gate, so promoted-library diversity and final 2016 OOS
performance are undefined at this stage. The empty library must not be made non-empty by
relaxing gates after observing these results.
