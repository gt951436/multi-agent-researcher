import pytest
from graph.workflow import build_graph, route_after_critic


def test_build_graph_compilation():
    """Verify that the LangGraph StateGraph builds and compiles without error."""
    compiled_graph = build_graph()
    assert compiled_graph is not None


def test_route_after_critic_routing_rules():
    """
    CRITICAL ROUTING TEST:
    Verify the conditional edge logic:
    - Initial pass rejection (loops=0) -> loops back to "researcher"
    - First re-research rejection (loops=1) -> loops back to "researcher" (MAX_LOOPS=2 allows this!)
    - Second re-research (loops=2) -> forces forward to "writer"
    - Approval -> proceeds to "writer"
    """
    # Case 1: Initial pass, gaps found -> must re-research
    state_loop0 = {"needs_more_research": True, "research_loops": 0}
    assert route_after_critic(state_loop0) == "researcher"

    # Case 2: First re-research completed, gaps still found -> must allow 2nd re-research pass
    state_loop1 = {"needs_more_research": True, "research_loops": 1}
    assert route_after_critic(state_loop1) == "researcher"

    # Case 3: Second re-research completed, max loops reached -> must proceed to writer
    state_loop2 = {"needs_more_research": True, "research_loops": 2}
    assert route_after_critic(state_loop2) == "writer"

    # Case 4: Approved at any point -> proceeds to writer
    state_approved = {"needs_more_research": False, "research_loops": 0}
    assert route_after_critic(state_approved) == "writer"
