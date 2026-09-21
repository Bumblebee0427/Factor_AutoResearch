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

## Adaptive Macro/Micro/Cross deterministic baseline

The XALPHA-inspired adaptive controller was then run with a fixed budget of 60 evaluated
candidates. The protocol required all eight mechanisms to be sampled before exploitation or
STOP, used the same walk-forward evaluator for every candidate, and did not load 2016.

| Metric | Adaptive deterministic v6 |
| --- | ---: |
| Candidates evaluated | 60 |
| Candidates to first Parent | 3 |
| Candidates to first Elite | Not reached |
| Effective valid information per 10 | 9.6667 |
| Duplicate-formula rate | 0.0% |
| Invalid evaluated-proposal rate | 0.0% |
| Mechanisms covered | 8 / 8 |
| Parent pool size after its cap | 20 |
| Elite archive size | 0 |

The mandatory coverage phase evaluated exactly two candidates in each of the eight
mechanisms, consuming 16 of the 60-candidate budget. The adaptive phase then allocated the
remaining 44 candidates, producing final counts of 8 price-trend, 8 price-reversal,
8 volatility, 8 price-volume, 6 fundamental-value, 4 fundamental-quality, 2 news-attention,
and 16 cross-domain-regime tests. Thirty-one candidates reached the lenient Parent tier over
the trajectory; the bounded pool retained 20. No candidate reached Elite because none
simultaneously satisfied the pre-committed Newey-West significance, three-positive-fold,
stressed-cost, and turnover gates. The gates were not relaxed after seeing this result.

This run also verified three implementation corrections: reversal hypotheses using words
such as “reverse” are aligned with negative return direction; fundamental changes use the
first point-in-time appearance of a report rather than a future repeated row; and interaction
direction is the product of both parent directions. All 60 candidates passed the integrity
gate after these fixes. The complete local trajectory is under
`artifacts/experiments/adaptive_comparisons/mechanism_coverage_v6/`.

The equal-budget adaptive Luna arm is intentionally pending explicit authorization to send
factor formulas, hypotheses, 2010-2015 aggregate research metrics, and structured research
memory to the OpenAI API. Raw observations and the 2016 holdout will not be sent.
