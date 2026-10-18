import pytest
from unittest.mock import MagicMock, patch
from agents.summarizer import summarizer_agent


@patch("agents.summarizer.get_llm")
def test_summarizer_initial_synthesis(mock_get_llm):
    """Verify Summarizer produces one summary per sub-question, tagged with question_id."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Quantum computing leverages superposition and entanglement.")
    mock_get_llm.return_value = mock_llm

    state = {
        "sub_questions": [
            {"id": "Q1", "question": "What is quantum computing?"},
            {"id": "Q2", "question": "What are hardware limitations?"}
        ],
        "research_data": [
            {
                "question_id": "Q1",
                "key_facts": "• Superposition allows multiple states.\n• Entanglement connects qubits.",
                "result_count": 2,
                "iteration": 0
            },
            {
                "question_id": "Q2",
                "key_facts": "• Decoherence causes errors.\n• Cryogenic cooling required.",
                "result_count": 2,
                "iteration": 0
            }
        ],
        "summaries": [],
        "research_loops": 0
    }

    result = summarizer_agent(state)
    summaries = result["summaries"]

    assert len(summaries) == 2
    assert summaries[0]["question_id"] == "Q1"
    assert summaries[0]["has_gap"] is False
    assert summaries[1]["question_id"] == "Q2"
    assert summaries[1]["has_gap"] is False
    assert mock_llm.invoke.call_count == 2


@patch("agents.summarizer.get_llm")
def test_summarizer_incremental_update_avoids_duplicates(mock_get_llm):
    """
    CRITICAL REGRESSION TEST:
    Verify that when Q2 receives follow-up evidence on loop 1:
    1. Q2 summary is refreshed with new evidence.
    2. Q1 summary is reused without redundant LLM call.
    3. The total summary list has exactly 2 items (NO duplicate entries for Q2!).
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="UPDATED Q2 SUMMARY: Recent 2025 benchmarks demonstrate 127-qubit coherence.")
    mock_get_llm.return_value = mock_llm

    previous_summaries = [
        {
            "question_id": "Q1",
            "question": "What is quantum computing?",
            "summary": "Original Q1 summary: Superposition and entanglement.",
            "has_gap": False,
            "iteration": 0
        },
        {
            "question_id": "Q2",
            "question": "What are hardware limitations?",
            "summary": "Original Q2 summary: Cryogenic cooling and decoherence.",
            "has_gap": True,
            "iteration": 0
        }
    ]

    # Accumulated research_data across pass 0 and pass 1
    research_data = [
        {"question_id": "Q1", "key_facts": "Facts for Q1", "result_count": 2, "iteration": 0},
        {"question_id": "Q2", "key_facts": "Old facts for Q2", "result_count": 2, "iteration": 0},
        # Pass 1 targeted follow-up for Q2
        {"question_id": "Q2", "key_facts": "New 2025 coherence benchmarks", "result_count": 2, "iteration": 1, "is_targeted_gap": True}
    ]

    state = {
        "sub_questions": [
            {"id": "Q1", "question": "What is quantum computing?"},
            {"id": "Q2", "question": "What are hardware limitations?"}
        ],
        "research_data": research_data,
        "summaries": previous_summaries,
        "research_loops": 1
    }

    result = summarizer_agent(state)
    updated_summaries = result["summaries"]

    # Must contain exactly 2 summaries (one per sub-question, NO duplicates!)
    assert len(updated_summaries) == 2

    # Q1 must be reused intact
    assert updated_summaries[0]["question_id"] == "Q1"
    assert updated_summaries[0]["summary"] == "Original Q1 summary: Superposition and entanglement."

    # Q2 must be updated
    assert updated_summaries[1]["question_id"] == "Q2"
    assert "UPDATED Q2 SUMMARY" in updated_summaries[1]["summary"]
    assert updated_summaries[1]["iteration"] == 1

    # LLM was invoked only ONCE for Q2, NOT for Q1
    assert mock_llm.invoke.call_count == 1
