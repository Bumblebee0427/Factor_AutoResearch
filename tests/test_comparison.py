from src.research.comparison import summarize_search
from src.utils.logging import ExperimentRecord


def incomplete_record() -> ExperimentRecord:
    return ExperimentRecord(
        factor_id="degenerate",
        generation=1,
        parent_ids=(),
        family="price",
        hypothesis="A degenerate test record should not count as new information.",
        canonical_formula="+1 * rank(vol_adjust(volatility(20)))",
        integrity_passed=True,
        integrity_issues=(),
        fold_metrics=(
            {
                "mean_ic": float("nan"),
                "ic_tstat": float("nan"),
                "net_sharpe": float("nan"),
                "high_cost_sharpe": float("nan"),
                "turnover": float("nan"),
                "max_drawdown": float("nan"),
                "stock_day_observations": 0,
            },
        ),
        mean_rank_ic=float("nan"),
        ic_tstat=float("nan"),
        positive_fold_count=0,
        long_short_sharpe=float("nan"),
        high_cost_sharpe=float("nan"),
        turnover=float("nan"),
        max_drawdown=float("nan"),
        redundancy_corr=0.0,
        closest_factor_id=None,
        residual_ic=None,
        decision="RETIRE",
        reasons=("degenerate evidence",),
    )


def test_llm_call_counts_completed_response_even_when_all_proposals_rejected() -> None:
    summary = summarize_search(
        [],
        [
            {
                "event": "proposal",
                "used_llm": False,
                "response_id": "resp_fake",
                "rejected": ["candidate: invalid"],
                "usage": {"total_tokens": 25},
            }
        ],
        arm="llm",
    )

    assert summary["llm_calls"] == 1
    assert summary["llm_total_tokens"] == 25
    assert summary["invalid_or_noncompliant_count"] == 1


def test_incomplete_walk_forward_record_is_not_effective_new_information() -> None:
    summary = summarize_search(
        [incomplete_record()],
        [{"event": "evaluation", "proposed_count": 1}],
        arm="llm",
    )

    assert summary["informative_candidates"] == 0
    assert summary["effective_new_information_per_10_candidates"] == 0.0
    assert summary["invalid_or_noncompliant_count"] == 1
