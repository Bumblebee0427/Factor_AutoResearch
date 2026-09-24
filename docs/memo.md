# Evaluation Methodology — Autonomous Factor Research Loop

## A. What Counts as Evidence

- **Fixed validation.** I used 2010–2015 for research, with expanding folds 2010–2012 → 2013, 2010–2013 → 2014, and 2010–2014 → 2015. Every candidate used the same deterministic construction and evaluator. I froze the selected library before opening the separate 2016 holdout once.
- **Several tests, one decision.** Evidence includes daily cross-sectional five-day RankIC and its Newey–West t-statistic, one- and ten-day IC, positive-fold consistency, top/bottom 20% long-short returns, turnover, drawdown, and Sharpe at 10 and 25 bp transaction costs. A Parent needs positive mean IC, at least one positive fold, and turnover ≤ 2.0. Elite requires mean IC ≥ 0.005, Newey–West t ≥ 1.0, positive IC in all three folds, 25 bp Sharpe ≥ 0, and turnover ≤ 1.5. No single statistic is sufficient.

## B. Why I Trust the Process More Than the Winner

- **Controls against lucky discoveries.** Folds, gates, and candidate budgets were fixed. Canonical-formula deduplication, signal-level Parent clustering, and library redundancy checks limit repeated variants. I did not relax thresholds after seeing results or use 2016 to select a factor.
- **Rejections are evidence.** Six research arms finished; only one candidate cleared every Elite gate. In the 60-candidate adaptive deterministic arm, 31/32 Parents failed high-cost Sharpe and 25/32 failed positive-fold consistency. Three targeted repairs improved the target cost metric but damaged IC, so none counted as a successful repair.
- **Final check.** The frozen factor, `slow_attention_profitability_spread` (rank of after-tax ROE minus rank of 60-day news volume), had research RankIC 0.0199, Newey–West t 1.079, and 25 bp Sharpe 0.733. In the one-shot 2016 holdout, RankIC was -0.0049 and 25 bp Sharpe was -1.637. It passed the research gates but did not generalize strongly; one holdout year is not a new selection round.

## C. Look-Ahead and Data Integrity

- **Availability.** Fundamentals become usable 90 days after fiscal period end; news is delayed one day. Time-series operators use each stock's history, while cross-sectional transforms stay within the current date.
- **Construction firewall.** The LLM proposes only a typed expression, not executable Python. Local schema, feature, operator, window, complexity, and coverage checks precede evaluation. Sampled truncation and future-noise tests require past signals to remain unchanged when future data changes. A deliberately leaky `future_return` factor is rejected before its apparent predictiveness can affect selection.

## D. What I Would Improve With More Time

- Obtain true historical universe membership and actual filing timestamps; calibrate spreads, capacity, and execution costs; develop richer deterministic news features.
- Repeat model-role comparisons to measure run-to-run variance and add formal multiple-testing controls. I trust the fixed record of hypotheses, rejections, and repairs more than the highest-Sharpe candidate; the weak 2016 result reinforces that this is a research filter, not proof of alpha.
