# LangGraph graph wiring
"""
Nodes  → the agents (planner, researcher, summarizer, critic, writer)
Edges  → fixed connections between nodes (planner → researcher)
Conditional edges → routing logic (critic → researcher OR writer)
"""
from langgraph.graph import StateGraph, END

from graph.state import ResearchState
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.summarizer import summarizer_agent
from agents.critic import critic_agent
from agents.writer import writer_agent
from config.settings import MAX_RESEARCH_LOOPS


# Routing function — the conditional edge logic
def route_after_critic(state: ResearchState) -> str:
    """
    Called by LangGraph after every Critic node execution.
    Returns a string key that LangGraph uses to select the next node.

    Decision logic:
        needs_more_research=True  AND loops < MAX → "researcher"
        needs_more_research=False OR loops >= MAX → "writer"

    Why this lives here and not inside critic_agent():
        Agents should not know about routing — that's the graph's job.
        The Critic writes facts into state (needs_more_research, loops).
        The graph reads those facts and decides routing.
        Clean separation of concerns.

    Args:
        state: Current ResearchState after Critic has run

    Returns:
        "researcher" → loop back for more research
        "writer"     → proceed to final report
    """
    needs_more   = state.get("needs_more_research", False)
    loops_so_far = state.get("research_loops", 0)

    if needs_more and loops_so_far < MAX_RESEARCH_LOOPS:
        return "researcher"
    return "writer"


# Graph builder
def build_graph():
    """
    Constructs and compiles the full multi-agent LangGraph StateGraph.

    Graph topology:
        planner → researcher → summarizer → critic ─┬→ writer → END
                      ↑                              │
                      └──────────────────────────────┘
                              (if needs more research)

    The conditional edge after 'critic' is the loop-back mechanism.
    All other edges are fixed (always go to the next node).

    Returns:
        A compiled LangGraph CompiledGraph ready to invoke or stream.
    """
    # Initialize the graph with our shared state schema
    # ResearchState is the TypedDict we defined in graph/state.py
    graph = StateGraph(ResearchState)

    # --- Register all agent nodes ---
    # Each node is a Python function that:
    #   1. Receives the full ResearchState
    #   2. Returns a dict of only the fields it updates
    # LangGraph handles the state merging automatically.
    graph.add_node("planner",    planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("critic",     critic_agent)
    graph.add_node("writer",     writer_agent)

    # --- Set the entry point ---
    # This is the first node LangGraph calls when the graph runs.
    graph.set_entry_point("planner")

    # --- Fixed edges (always follow this path) ---
    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "summarizer")
    graph.add_edge("summarizer", "critic")

    # --- Conditional edge after Critic ---
    # This is the loop-back mechanism.
    # route_after_critic() returns either "researcher" or "writer".
    # The dict maps those return values to actual node names.
    graph.add_conditional_edges(
        "critic",             # source node
        route_after_critic,   # routing function
        {
            "researcher": "researcher",   # loop back
            "writer":     "writer"        # proceed forward
        }
    )

    # --- Terminal edge ---
    # After Writer completes, the graph ends.
    graph.add_edge("writer", END)

    # --- Compile ---
    # compile() validates the graph (checks for unreachable nodes,
    # missing edges, etc.) and returns a runnable object.
    return graph.compile()


# Convenience runner — used by app.py
def run_research(query: str) -> dict:
    """
    High-level function that runs the full research pipeline
    for a given query and returns the completed state.

    Used by app.py for the non-streaming (simple) mode.

    Args:
        query: The user's research question

    Returns:
        The final ResearchState dict after all agents have run
    """
    graph = build_graph()

    # Build the initial state with just the query.
    # All other fields start empty — agents populate them.
    initial_state: ResearchState = {
        "query":               query,
        "sub_questions":       [],
        "research_data":       [],
        "sources":             [],
        "summaries":           [],
        "critique":            "",
        "summary_scores":      [],
        "needs_more_research": False,
        "research_loops":      0,
        "final_report":        "",
        "status_log":          []
    }

    # invoke() runs the full graph synchronously and returns
    # the final state after all nodes have completed.
    final_state = graph.invoke(initial_state)
    return final_state