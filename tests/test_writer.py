import pytest
from unittest.mock import MagicMock, patch
from agents.writer import (
    writer_agent,
    _format_content_for_writer,
    _format_sources_block,
    _determine_confidence,
    _validate_citations,
    _convert_source_tags_to_numbers
)


def test_writer_content_pairing_by_question_id():
    """
    CRITICAL REGRESSION TEST:
    Verify that Writer pairs questions and summaries by question_id,
    guaranteeing that refreshed summaries for Q2 are used rather than stale ones.
    """
    sub_questions = [
        {"id": "Q1", "question": "What is quantum computing?"},
        {"id": "Q2", "question": "What are hardware limitations?"}
    ]

    # Summaries where Q2 has the refreshed text from loop 1
    summaries = [
        {"question_id": "Q1", "summary": "Q1 Baseline summary.", "has_gap": False},
        {"question_id": "Q2", "summary": "Q2 REFRESHED with 2025 coherence benchmarks.", "has_gap": False}
    ]

    block = _format_content_for_writer(sub_questions, summaries)

    assert "SUB-QUESTION 1 [Q1]: What is quantum computing?" in block
    assert "SUMMARY: Q1 Baseline summary." in block
    assert "SUB-QUESTION 2 [Q2]: What are hardware limitations?" in block
    assert "SUMMARY: Q2 REFRESHED with 2025 coherence benchmarks." in block


def test_writer_source_counting_deduplication():
    """
    REGRESSION TEST:
    Verify that source_count in the report matches the number of unique sources in the list.
    """
    # 4 URLs with 2 duplicates
    sources = [
        "https://nature.com/paper1",
        "https://arxiv.org/abs/1234",
        "https://nature.com/paper1",
        "https://ibm.com/quantum"
    ]

    block = _format_sources_block(sources)

    # Numbered reference list has 3 items
    assert "1. https://nature.com/paper1" in block
    assert "2. https://arxiv.org/abs/1234" in block
    assert "3. https://ibm.com/quantum" in block
    assert "4." not in block


def test_convert_source_tags_to_numbers():
    """Verify internal [S_k] tags translate to numbered [k] citations."""
    text = "Key quantum logic is described in [S1] and reinforced by [S2, S3]."
    sid_to_num = {"S1": 1, "S2": 2, "S3": 3}
    converted = _convert_source_tags_to_numbers(text, sid_to_num)
    assert converted == "Key quantum logic is described in [1] and reinforced by [2, 3]."


def test_validate_citations_metrics():
    """Verify regex citation extraction and precision calculation."""
    # Perfectly cited text with total_sources = 3
    body_valid = "Quantum coherence times were measured at 127 qubits [1]. Later studies confirmed [2, 3]."
    metrics = _validate_citations(body_valid, total_sources=3)
    assert metrics["total_citations"] == 3
    assert metrics["valid_citations"] == 3
    assert metrics["invalid_citations"] == []
    assert metrics["precision"] == 1.0

    # Text containing hallucinated citation [5] when only 3 sources exist
    body_invalid = "Unverified claim cited here [5], but verified here [1]."
    metrics_invalid = _validate_citations(body_invalid, total_sources=3)
    assert metrics_invalid["total_citations"] == 2
    assert metrics_invalid["valid_citations"] == 1
    assert metrics_invalid["invalid_citations"] == [5]
    assert metrics_invalid["precision"] == 0.5


def test_format_sources_block_with_registry():
    """Verify rich bibliography output with source types and domains."""
    registry = [
        {
            "source_id": "S1",
            "url": "https://nature.com/articles/123",
            "title": "Quantum Supremacy",
            "domain": "nature.com",
            "content_type": "full_text"
        },
        {
            "source_id": "S2",
            "url": "https://arxiv.org/abs/456",
            "title": "Error Mitigation",
            "domain": "arxiv.org",
            "content_type": "snippet"
        }
    ]
    block = _format_sources_block(registry)
    assert "1. [Quantum Supremacy](https://nature.com/articles/123) (nature.com) — _Full text analyzed_" in block
    assert "2. [Error Mitigation](https://arxiv.org/abs/456) (arxiv.org) — _Search snippet_" in block


@patch("agents.writer.get_llm")
def test_writer_agent_unique_source_count_in_metadata(mock_get_llm):
    """Verify that writer_agent passes unique source count and citation metrics to report metadata."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="## 1. Quantum Overview\nProse text citing [1].")
    mock_get_llm.return_value = mock_llm

    sources = [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/1",  # duplicate
        "https://example.com/3"
    ]
    source_registry = [
        {
            "source_id": "S1",
            "url": "https://example.com/1",
            "title": "Example One",
            "domain": "example.com",
            "content_type": "full_text"
        }
    ]

    state = {
        "query": "Quantum Computing",
        "sub_questions": [{"id": "Q1", "question": "What is it?"}],
        "summaries": [{"question_id": "Q1", "summary": "Summary text citing [S1]", "has_gap": False}],
        "sources": sources,
        "source_registry": source_registry,
        "critique": "VERDICT: APPROVED\nSUMMARY SCORES:\n- Q1: PASS",
        "summary_scores": ["PASS"],
        "research_loops": 0
    }

    result = writer_agent(state)
    report = result["final_report"]

    # Must state 3 sources consulted, NOT 4
    assert "| Sources consulted | 3 |" in report
    assert "3 unique web sources were consulted" in report
    assert result["confidence"] == "High"
    assert result["citation_metrics"] is not None
    assert result["citation_metrics"]["valid_citations"] == 1
    assert result["citation_metrics"]["precision"] == 1.0
