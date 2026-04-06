# shared state schema
# Shared State : think of it as a whiteboard (state) in a meeting room. Every agent walks up, reads what's on the board, adds their contribution, and walks away. The next agent reads the updated board.
# TypedDict : python type that gives dict a strict schema - makes code self-documenting
# Annotated with operator.add : For fields that multiple agents append to (like research_data and sources), we use Annotated[list, operator.add]. This tells LangGraph: "when multiple nodes write to this field, don't overwrite — merge the lists." Without this, each agent would wipe out what the previous agent wrote.

from typing import TypedDict, Annotated, List, Optional
import operator


class ResearchState(TypedDict):
    """
    The shared whiteboard that all 5 agents read from and write to.
    This single object travels through the entire LangGraph pipeline.
    
    Field ownership:
    - query          --> set by user input, read by all agents
    - sub_questions  --> written by Planner, read by Researcher
    - research_data  --> written by Researcher, read by Summarizer
    - sources        --> written by Researcher, read by Writer
    - summaries      --> written by Summarizer, read by Critic + Writer
    - critique       --> written by Critic, read by routing logic + Researcher
    - research_loops --> incremented each time Critic sends back to Researcher
    - final_report   --> written by Writer, read by UI
    - status_log     --> each agent appends its status here (for live UI display)
    """

    # input
    query: str                          # The original user research query

    # Planner output
    sub_questions: List[str]            # 3–5 focused sub-questions

    # Researcher output
    # Annotated with operator.add = LangGraph MERGES new items instead of overwriting
    research_data: Annotated[List[dict], operator.add]  # [{question, results}]
    sources: Annotated[List[str], operator.add]          # source URLs collected

    # Summarizer output
    summaries: List[dict]               # [{question, summary}]

    # Critic output
    critique: str                       # Critic's feedback text
    summary_scores: List[str]   # ["PASS", "FAIL", "PASS", ...] per summary
    needs_more_research: bool           # True = loop back to Researcher
    research_loops: int                 # Counter: prevents infinite loops

    # Writer output
    final_report: str                   # The finished markdown report

    # System
    status_log: Annotated[List[str], operator.add]  # Agent activity log for UI