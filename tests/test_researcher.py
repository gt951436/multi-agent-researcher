import pytest
from unittest.mock import MagicMock, patch
from agents.researcher import researcher_agent


@patch("agents.researcher.time.sleep", return_value=None)
@patch("agents.researcher.web_search")
@patch("agents.researcher.get_llm")
def test_researcher_initial_pass(mock_get_llm, mock_web_search, mock_sleep):
    """Verify Researcher on pass 0 searches all planned sub-questions."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="• Fact 1: Breakthrough reported.")
    mock_get_llm.return_value = mock_llm

    mock_web_search.return_value = [
        {"title": "Quantum Paper", "snippet": "Breakthrough reported in 2025.", "url": "https://example.com/paper"}
    ]

    state = {
        "sub_questions": [
            {"id": "Q1", "question": "What is quantum computing?"},
            {"id": "Q2", "question": "What are hardware limitations?"}
        ],
        "follow_up_queries": [],
        "research_loops": 0
    }

    result = researcher_agent(state)

    assert result["research_loops"] == 0
    assert len(result["research_data"]) == 2
    assert result["research_data"][0]["question_id"] == "Q1"
    assert result["research_data"][1]["question_id"] == "Q2"
    assert "https://example.com/paper" in result["sources"]
    assert mock_web_search.call_count == 2


@patch("agents.researcher.time.sleep", return_value=None)
@patch("agents.researcher.web_search")
@patch("agents.researcher.get_llm")
def test_researcher_targeted_follow_up_pass(mock_get_llm, mock_web_search, mock_sleep):
    """
    CRITICAL REGRESSION TEST:
    Verify that on re-research, the Researcher targets the Critic's specific follow-up query,
    does NOT repeat all original sub-questions, and accurately increments research_loops.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="• Specific gap filled: 127 qubit coherence measured.")
    mock_get_llm.return_value = mock_llm

    mock_web_search.return_value = [
        {"title": "Coherence Benchmark", "snippet": "Measured 127 qubits.", "url": "https://nature.com/coherence"}
    ]

    state = {
        "sub_questions": [
            {"id": "Q1", "question": "What is quantum computing?"},
            {"id": "Q2", "question": "What are hardware limitations?"},
            {"id": "Q3", "question": "Who are key leaders?"}
        ],
        "follow_up_queries": [
            {
                "question_id": "Q2",
                "query": "quantum hardware coherence benchmarks 2025",
                "gaps": ["Missing recent coherence metrics"],
                "rationale": "Prior summary lacked exact qubit coherence metrics"
            }
        ],
        "research_loops": 0
    }

    result = researcher_agent(state)

    # Must increment research_loops to 1
    assert result["research_loops"] == 1

    # Must search ONLY the targeted follow-up query (1 search call, NOT 3!)
    assert mock_web_search.call_count == 1
    mock_web_search.assert_called_once_with("quantum hardware coherence benchmarks 2025", max_results=5)

    # Must produce targeted evidence tagged with Q2 and is_targeted_gap=True
    assert len(result["research_data"]) == 1
    entry = result["research_data"][0]
    assert entry["question_id"] == "Q2"
    assert entry["is_targeted_gap"] is True
    assert entry["iteration"] == 1
    assert "https://nature.com/coherence" in result["sources"]

    # follow_up_queries should be cleared after execution
    assert result["follow_up_queries"] == []
