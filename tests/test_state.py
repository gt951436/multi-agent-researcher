import pytest
from graph.state import ResearchState, SubQuestion, EvidenceItem, SummaryItem, FollowUpQuery


def test_research_state_structure():
    """Verify that ResearchState initializes with all required fields."""
    state: ResearchState = {
        "query": "Test query",
        "sub_questions": [{"id": "Q1", "question": "What is quantum computing?"}],
        "research_data": [],
        "sources": [],
        "summaries": [],
        "critique": "",
        "critic_review": None,
        "follow_up_queries": [],
        "summary_scores": [],
        "needs_more_research": False,
        "research_loops": 0,
        "final_report": "",
        "status_log": []
    }

    assert state["query"] == "Test query"
    assert len(state["sub_questions"]) == 1
    assert state["sub_questions"][0]["id"] == "Q1"
    assert state["research_loops"] == 0
    assert state["needs_more_research"] is False


def test_follow_up_query_schema():
    """Verify that FollowUpQuery schema holds required targeted search fields."""
    fq: FollowUpQuery = {
        "question_id": "Q2",
        "query": "quantum error correction threshold 2025",
        "gaps": ["Missing recent benchmark figures"],
        "rationale": "Summary lacked numerical error thresholds"
    }

    assert fq["question_id"] == "Q2"
    assert "2025" in fq["query"]
    assert len(fq["gaps"]) == 1
