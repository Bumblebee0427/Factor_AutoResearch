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
| Candidate proposals | 37 | 49 |
| Candidates evaluated | 33 | 40 |
| Candidates to first Promote | Not reached | Not reached |
| Candidates with complete walk-forward evidence | 31 | 36 |
| Effective new information per 10 proposals | 8.3784 | 7.3469 |
| Duplicate-formula rate | 10.81% | 2.04% |
| Invalid/noncompliant rate | 5.41% | 26.53% |
| Promoted factors | 0 | 0 |
| Mean walk-forward IC-sign consistency | 0.7527 | 0.7500 |
| Positive IC in every fold | 6.45% | 5.56% |

“Effective new information” means a unique candidate that passed integrity checks and
produced complete walk-forward evidence. Rejected and duplicate proposals remain in the
denominator.

## Interpretation

The first flat nullable schema allowed Luna to populate interaction-only fields in
single-feature proposals, so none of its proposals reached evaluation. Replacing it with a
nested `anyOf` union made single-factor and interaction recipes structurally exclusive. In
the final same-code comparison, Luna completed three API calls and used 35,514 tokens. Nine
LLM proposals passed recipe validation; seven produced complete walk-forward evidence and two
volatility-adjusted recipes were retired as `degenerate_signal` before backtesting.

The schema change materially improved the integration, but Luna Low still did **not** improve
search efficiency. It generated more evaluated information and fewer duplicate formulas, but
its effective-information yield remained below deterministic search and no candidate was
promoted. Most remaining schema rejections assigned windows to fundamental features. A next
schema iteration can model windowed and fundamental feature legs as separate nested unions;
a bounded validation-feedback retry is another pre-registrable experiment. Neither change
should alter the gates or expose the 2016 holdout.

No factors passed every promotion gate, so promoted-library diversity and final 2016 OOS
performance are undefined at this stage. The empty library must not be made non-empty by
relaxing gates after observing these results.
