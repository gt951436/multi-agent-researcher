from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm, MAX_RESEARCH_LOOPS
from graph.state import ResearchState


def critic_agent(state: ResearchState) -> dict:
    """
    The Critic Agent — fourth node in the LangGraph pipeline.

    Responsibility:
        Evaluates all summaries for completeness, specificity, and
        internal consistency. Decides whether research quality is
        sufficient to proceed to writing, or needs another loop.

    Why this agent exists:
        Without quality control, the Writer blindly compiles whatever
        the Summarizer produced — even if some summaries are vague,
        incomplete, or contradictory. The Critic catches these issues
        and triggers targeted re-research before the final report.

    The loop-back mechanism:
        If needs_more_research=True AND research_loops < MAX_RESEARCH_LOOPS:
            → LangGraph routes back to Researcher
        Else:
            → LangGraph routes forward to Writer
        This logic lives in graph/workflow.py (Step 8), not here.
        The Critic only evaluates — it doesn't control routing.

    Reads from state:  summaries, research_loops
    Writes to state:   critique, needs_more_research, research_loops,
                       status_log

    Args:
        state: The shared ResearchState

    Returns:
        dict with keys: critique, needs_more_research,
                        research_loops, status_log
    """
    print("\n[CRITIC] Starting quality review of summaries...")

    summaries       = state["summaries"]
    research_loops  = state.get("research_loops", 0)
    original_query  = state["query"]

    # --- Safety check ---
    if not summaries:
        print("[CRITIC] No summaries to evaluate — forcing forward to Writer.")
        return {
            "critique":            "No summaries available to critique.",
            "needs_more_research": False,
            "research_loops":      research_loops,
            "status_log": ["Critic: No summaries found — skipping review."]
        }

    # --- Check if we've hit the loop limit ---
    # Even if the Critic finds gaps, we can't loop forever.
    # After MAX_RESEARCH_LOOPS attempts, we proceed to Writer with
    # whatever we have, noting the limitation in the critique.
    if research_loops >= MAX_RESEARCH_LOOPS:
        print(f"[CRITIC] Max research loops ({MAX_RESEARCH_LOOPS}) reached.")
        print("[CRITIC] Proceeding to Writer despite any remaining gaps.")
        return {
            "critique":            f"Maximum research loops ({MAX_RESEARCH_LOOPS}) reached. "
                                   f"Proceeding to final report with available research. "
                                   f"Some gaps may remain.",
            "needs_more_research": False,   # force forward regardless
            "research_loops":      research_loops,
            "status_log": [
                f"Critic: Max loops reached ({MAX_RESEARCH_LOOPS}). "
                f"Proceeding to Writer."
            ]
        }

    # --- Format summaries for the LLM ---
    # We render all summaries as a numbered list so the LLM can
    # reference specific ones by number in its critique.
    formatted_summaries = _format_summaries_for_critique(summaries)

    # --- System Prompt ---
    # The Critic's role is evaluator, not improver.
    # It identifies problems and articulates them precisely —
    # it does NOT rewrite or fix the summaries itself.
    system_prompt = SystemMessage(content="""You are a rigorous research quality reviewer.

Your job is to evaluate a set of research summaries and determine if they 
provide sufficient, accurate, and complete coverage of the original research query.

Evaluate each summary for:
1. COMPLETENESS — Does it fully answer its sub-question? 
   Flag if it says things like "details not available" or is vague.
2. SPECIFICITY — Does it contain concrete facts, names, dates, numbers?
   Flag if it is generic and could apply to any topic.
3. CONSISTENCY — Do any summaries contradict each other?
   Flag specific contradictions if found.

Your response MUST follow this exact format:

VERDICT: APPROVED or NEEDS_MORE_RESEARCH

ISSUES FOUND:
- (list each issue with the summary number it refers to, or "None" if approved)

SPECIFIC GAPS TO ADDRESS:
- (list specific topics or questions that need better research, or "None" if approved)

OVERALL ASSESSMENT:
(2-3 sentences explaining your overall judgment)

Be strict but fair. Only approve if the summaries genuinely cover the topic well.
If even one summary has a significant gap, output NEEDS_MORE_RESEARCH.
""")

    human_prompt = HumanMessage(content=f"""Original research query: {original_query}

Research summaries to evaluate:

{formatted_summaries}

Evaluate these summaries and provide your verdict.""")

    # temperature=0.1 — Critic must be consistent and deterministic
    # We don't want random variation in quality assessments
    llm      = get_llm(temperature=0.1)
    response = llm.invoke([system_prompt, human_prompt])
    critique = response.content.strip()

    print(f"[CRITIC] Evaluation complete ({len(critique)} chars)")

    # --- Parse the verdict ---
    # We look for "VERDICT: APPROVED" or "VERDICT: NEEDS_MORE_RESEARCH"
    # in the LLM's response to determine the routing decision.
    needs_more_research = _parse_verdict(critique)

    if needs_more_research:
        print(f"[CRITIC] VERDICT: NEEDS MORE RESEARCH")
        print(f"[CRITIC] Will trigger research loop #{research_loops + 1}")
    else:
        print(f"[CRITIC] VERDICT: APPROVED — proceeding to Writer")

    # --- Increment loop counter ---
    # We increment here so the Researcher knows which loop it's on
    # if it gets called again.
    new_loop_count = research_loops + 1 if needs_more_research else research_loops

    return {
        "critique":            critique,
        "needs_more_research": needs_more_research,
        "research_loops":      new_loop_count,
        "status_log": [
            f"Critic: {'NEEDS_MORE_RESEARCH' if needs_more_research else 'APPROVED'} "
            f"(loop {research_loops}/{MAX_RESEARCH_LOOPS})"
        ]
    }


def _format_summaries_for_critique(summaries: list[dict]) -> str:
    """
    Renders all summaries into a numbered, clearly separated string
    so the LLM Critic can reference specific ones in its evaluation.

    Example output:
        SUMMARY 1:
        Question: What are the latest architectures...
        Summary: As of 2025, transformer-based models...

        SUMMARY 2:
        Question: Which organizations lead...
        Summary: OpenAI, Google DeepMind, and Meta...

    Args:
        summaries: List of summary dicts from Summarizer

    Returns:
        Formatted string ready to inject into the Critic's prompt
    """
    formatted = []
    for i, entry in enumerate(summaries, 1):
        question = entry.get("question", "Unknown question")
        summary  = entry.get("summary",  "No summary available")
        has_gap  = entry.get("has_gap",  False)

        gap_note = " [FLAGGED AS GAP BY SUMMARIZER]" if has_gap else ""

        formatted.append(
            f"SUMMARY {i}{gap_note}:\n"
            f"Question: {question}\n"
            f"Summary: {summary}"
        )

    return "\n\n".join(formatted)


def _parse_verdict(critique_text: str) -> bool:
    """
    Extracts the binary verdict from the Critic's response text.

    Why a separate parsing function:
        The LLM returns a full multi-section text. We need to extract
        one boolean from it. Keeping this logic isolated makes it easy
        to improve the parser without touching the agent logic.

    Parsing strategy:
        Look for "VERDICT: NEEDS_MORE_RESEARCH" → True (loop back)
        Look for "VERDICT: APPROVED"             → False (move forward)
        Default to False if verdict is ambiguous (don't block pipeline)

    Args:
        critique_text: The raw LLM critique response string

    Returns:
        True  = needs another research loop
        False = approved, proceed to Writer
    """
    # Normalize to uppercase for case-insensitive matching
    upper = critique_text.upper()

    if "NEEDS_MORE_RESEARCH" in upper or "NEEDS MORE RESEARCH" in upper:
        return True
    elif "APPROVED" in upper:
        return False
    else:
        # Ambiguous verdict — default to proceeding forward
        # Better to write an imperfect report than to loop infinitely
        print("[CRITIC] Warning: Ambiguous verdict — defaulting to APPROVED")
        return False