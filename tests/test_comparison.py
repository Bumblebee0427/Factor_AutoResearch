from src.research.comparison import summarize_search


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
