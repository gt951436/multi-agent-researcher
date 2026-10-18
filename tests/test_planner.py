import pytest
from unittest.mock import MagicMock, patch
from agents.planner import planner_agent, _parse_sub_questions


def test_parse_sub_questions_numbered():
    """Verify parser extracts questions with standard numbered formats."""
    raw = """1. What is the architecture of LLaMA 3?
2. How does instruction tuning work?
3. What are the key benchmark results?
"""
    parsed = _parse_sub_questions(raw)
    assert len(parsed) == 3
    assert parsed[0] == "What is the architecture of LLaMA 3?"
    assert parsed[1] == "How does instruction tuning work?"
    assert parsed[2] == "What are the key benchmark results?"


def test_parse_sub_questions_parentheses():
    """Verify parser handles '1) ' style prefixes."""
    raw = """1) What are the latest breakthrough materials?
2) Which companies produce them commercially?
3) What are the main cost barriers?
4) How do they compare to legacy alloys?
"""
    parsed = _parse_sub_questions(raw)
    assert len(parsed) == 4
    assert parsed[0] == "What are the latest breakthrough materials?"


@patch("agents.planner.get_llm")
def test_planner_agent_output_structure(mock_get_llm):
    """Verify planner_agent returns structured questions with stable IDs (Q1, Q2, etc.)."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """1. What is quantum computing?
2. Who leads quantum research?
3. What are current technical limitations?
"""
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    state = {"query": "Explain quantum computing state of the art"}
    result = planner_agent(state)

    assert "sub_questions" in result
    questions = result["sub_questions"]
    assert len(questions) == 3

    assert questions[0]["id"] == "Q1"
    assert questions[0]["question"] == "What is quantum computing?"
    assert questions[1]["id"] == "Q2"
    assert questions[2]["id"] == "Q3"

    assert len(result["status_log"]) == 1
    assert "Planner: Generated 3 sub-questions" in result["status_log"][0]
