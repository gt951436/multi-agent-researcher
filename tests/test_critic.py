import pytest
from unittest.mock import MagicMock, patch
from agents.critic import critic_agent, CriticEvaluation, QuestionReview


@patch("agents.critic.get_llm")
def test_critic_triggers_re_research_with_targeted_queries(mock_get_llm):
    """Verify that Critic identifies failing summaries and generates targeted follow_up_queries."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    mock_eval = CriticEvaluation(
        verdict="NEEDS_MORE_RESEARCH",
        reviews=[
            QuestionReview(
                question_id="Q1",
                status="PASS",
                reason="Comprehensive coverage of quantum superposition.",
                gaps=[]
            ),
            QuestionReview(
                question_id="Q2",
                status="FAIL",
                reason="Lacks recent 2025-2026 experimental coherence benchmarks.",
                gaps=["No recent quantitative coherence measurements"],
                follow_up_query="quantum processor coherence time benchmarks 2025 2026"
            )
        ],
        overall_assessment="Q1 is solid but Q2 misses recent experimental benchmarks."
    )

    mock_structured.invoke.return_value = mock_eval
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    state = {
        "query": "Current state of quantum computing",
        "summaries": [
            {"question_id": "Q1", "question": "What is quantum?", "summary": "Superposition..."},
            {"question_id": "Q2", "question": "What are limitations?", "summary": "Errors exist..."}
        ],
        "research_loops": 0
    }

    result = critic_agent(state)

    assert result["needs_more_research"] is True
    assert result["summary_scores"] == ["PASS", "FAIL"]
    assert len(result["follow_up_queries"]) == 1

    fq = result["follow_up_queries"][0]
    assert fq["question_id"] == "Q2"
    assert fq["query"] == "quantum processor coherence time benchmarks 2025 2026"
    assert "No recent quantitative coherence measurements" in fq["gaps"]


@patch("agents.critic.get_llm")
def test_critic_approval_when_all_pass(mock_get_llm):
    """Verify that Critic approves when all summaries pass."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    mock_eval = CriticEvaluation(
        verdict="APPROVED",
        reviews=[
            QuestionReview(question_id="Q1", status="PASS", reason="Detailed and accurate."),
            QuestionReview(question_id="Q2", status="PASS", reason="Strong quantitative figures.")
        ],
        overall_assessment="All research questions are thoroughly answered."
    )

    mock_structured.invoke.return_value = mock_eval
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    state = {
        "query": "Quantum computing",
        "summaries": [
            {"question_id": "Q1", "summary": "Summary 1"},
            {"question_id": "Q2", "summary": "Summary 2"}
        ],
        "research_loops": 1
    }

    result = critic_agent(state)

    assert result["needs_more_research"] is False
    assert result["summary_scores"] == ["PASS", "PASS"]
    assert len(result["follow_up_queries"]) == 0


def test_critic_max_loops_guard():
    """Verify that Critic stops looping when research_loops reaches MAX_RESEARCH_LOOPS (2)."""
    state = {
        "query": "Quantum computing",
        "summaries": [{"question_id": "Q1", "summary": "Summary 1"}],
        "research_loops": 2
    }

    result = critic_agent(state)

    assert result["needs_more_research"] is False
    assert "Maximum research loops (2) reached" in result["critique"]
