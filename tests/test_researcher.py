import pytest
from unittest.mock import MagicMock, patch
from agents.researcher import researcher_agent


@patch("agents.researcher.time.sleep", return_value=None)
@patch("agents.researcher.fetch_page_content")
@patch("agents.researcher.search_with_status")
@patch("agents.researcher.get_llm")
def test_researcher_initial_pass(mock_get_llm, mock_search, mock_fetch, mock_sleep):
    """Verify Researcher on pass 0 searches all planned sub-questions and registers sources."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="• Breakthrough reported [S1].")
    mock_get_llm.return_value = mock_llm

    mock_search.return_value = {
        "status": "SUCCESS",
        "results": [
            {"title": "Quantum Paper", "snippet": "Breakthrough reported in 2025.", "url": "https://example.com/paper"}
        ],
        "error": None
    }
    mock_fetch.return_value = {
        "url": "https://example.com/paper",
        "text": "Full article text discussing quantum breakthrough.",
        "content_type": "full_text",
        "status": "ok"
    }

    state = {
        "sub_questions": [
            {"id": "Q1", "question": "What is quantum computing?"},
            {"id": "Q2", "question": "What are hardware limitations?"}
        ],
        "follow_up_queries": [],
        "research_loops": 0,
        "source_registry": [],
        "sources": []
    }

    result = researcher_agent(state)

    assert result["research_loops"] == 0
    assert len(result["research_data"]) == 2
    assert result["research_data"][0]["question_id"] == "Q1"
    assert result["research_data"][1]["question_id"] == "Q2"
    assert "https://example.com/paper" in result["sources"]
    assert mock_search.call_count == 2

    # Verify source registry population
    assert len(result["source_registry"]) == 1
    src = result["source_registry"][0]
    assert src["source_id"] == "S1"
    assert src["url"] == "https://example.com/paper"
    assert src["content_type"] == "full_text"


@patch("agents.researcher.time.sleep", return_value=None)
@patch("agents.researcher.fetch_page_content")
@patch("agents.researcher.search_with_status")
@patch("agents.researcher.get_llm")
def test_researcher_targeted_follow_up_pass(mock_get_llm, mock_search, mock_fetch, mock_sleep):
    """
    CRITICAL REGRESSION TEST:
    Verify that on re-research, the Researcher targets the Critic's specific follow-up query,
    does NOT repeat all original sub-questions, and accurately increments research_loops.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="• Specific gap filled: 127 qubit coherence measured [S1].")
    mock_get_llm.return_value = mock_llm

    mock_search.return_value = {
        "status": "SUCCESS",
        "results": [
            {"title": "Coherence Benchmark", "snippet": "Measured 127 qubits.", "url": "https://nature.com/coherence"}
        ],
        "error": None
    }
    mock_fetch.return_value = {
        "url": "https://nature.com/coherence",
        "text": "Full paper on 127 qubit coherence benchmarks.",
        "content_type": "full_text",
        "status": "ok"
    }

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
        "research_loops": 0,
        "source_registry": [],
        "sources": []
    }

    result = researcher_agent(state)

    # Must increment research_loops to 1
    assert result["research_loops"] == 1

    # Must search ONLY the targeted follow-up query (1 search call, NOT 3!)
    assert mock_search.call_count == 1
    mock_search.assert_called_once_with("quantum hardware coherence benchmarks 2025", max_results=5)

    # Must produce targeted evidence tagged with Q2 and is_targeted_gap=True
    assert len(result["research_data"]) == 1
    entry = result["research_data"][0]
    assert entry["question_id"] == "Q2"
    assert entry["is_targeted_gap"] is True
    assert entry["iteration"] == 1
    assert "https://nature.com/coherence" in result["sources"]

    # follow_up_queries should be cleared after execution
    assert result["follow_up_queries"] == []


@patch("agents.researcher.time.sleep", return_value=None)
@patch("agents.researcher.fetch_page_content")
@patch("agents.researcher.search_with_status")
@patch("agents.researcher.get_llm")
def test_researcher_fetch_fallback_to_snippet(mock_get_llm, mock_search, mock_fetch, mock_sleep):
    """Verify that when full webpage fetch fails, Researcher falls back gracefully to snippet."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="• Extracted from snippet [S1].")
    mock_get_llm.return_value = mock_llm

    mock_search.return_value = {
        "status": "SUCCESS",
        "results": [
            {"title": "Protected Site", "snippet": "Protected snippet text.", "url": "https://cloudflare-walled.com/doc"}
        ],
        "error": None
    }
    # Simulate scraper block returning snippet fallback
    mock_fetch.return_value = {
        "url": "https://cloudflare-walled.com/doc",
        "text": "",
        "content_type": "snippet",
        "status": "fetch_error"
    }

    state = {
        "sub_questions": [{"id": "Q1", "question": "What is the policy?"}],
        "follow_up_queries": [],
        "research_loops": 0,
        "source_registry": [],
        "sources": []
    }

    result = researcher_agent(state)

    assert len(result["source_registry"]) == 1
    src = result["source_registry"][0]
    assert src["content_type"] == "snippet"
    assert src["source_id"] == "S1"
    assert "Protected Site" in src["title"]


@patch("agents.researcher.time.sleep", return_value=None)
@patch("agents.researcher.fetch_page_content")
@patch("agents.researcher.search_with_status")
@patch("agents.researcher.get_llm")
def test_researcher_deduplicates_sources_across_loops(mock_get_llm, mock_search, mock_fetch, mock_sleep):
    """Verify that previously registered sources are reused by ID and not refetched."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="• New insight from existing source [S1].")
    mock_get_llm.return_value = mock_llm

    # Return same URL that was registered in loop 0
    mock_search.return_value = {
        "status": "SUCCESS",
        "results": [
            {"title": "Existing Doc", "snippet": "Known snippet.", "url": "https://example.com/existing"}
        ],
        "error": None
    }

    existing_registry = [
        {
            "source_id": "S1",
            "url": "https://example.com/existing",
            "title": "Existing Doc",
            "domain": "example.com",
            "content_type": "full_text"
        }
    ]

    state = {
        "sub_questions": [{"id": "Q1", "question": "What is the update?"}],
        "follow_up_queries": [],
        "research_loops": 0,
        "source_registry": existing_registry,
        "sources": ["https://example.com/existing"]
    }

    result = researcher_agent(state)

    # fetch_page_content should NOT be called since URL already exists
    assert mock_fetch.call_count == 0
    # No new duplicate records added to source_registry
    assert len(result["source_registry"]) == 0
