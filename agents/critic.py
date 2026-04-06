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
        print(f"[CRITIC] Max research loops ({MAX_RESEARCH_LOOPS}) reached")
        print("[CRITIC] Proceeding to Writer despite any remaining gaps.")
        return {
            "critique":  (         f"Maximum research loops ({MAX_RESEARCH_LOOPS}) reached. "
                                   f"Proceeding to final report with available research. "
                                   f"Some gaps may remain."),
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
    system_prompt = SystemMessage(content="""You are a demamding, rigorous, skeptical research quality reviewer and editor at a high-standard publication. Your job is to reject mediocre research.

Your job is to evaluate a set of research summaries and determine if they 
provide sufficient, accurate, and complete coverage of the original research query.

Evaluate each summary against this strict rubric:

CRITERION 1 — SPECIFICITY (most important)
FAIL if the summary:
  • Uses vague phrases like "various methods", "several approaches", "many researchers"
    without naming them specifically
  • Makes claims without concrete supporting evidence (numbers, names, dates, studies)
  • Could apply to almost any topic in the field (generic filler)

CRITERION 2 — COMPLETENESS  
FAIL if the summary:
  • Does not actually answer its specific sub-question
  • Says things like "details not available", "not fully specified", "may vary"
  • Covers only part of what the sub-question asks

CRITERION 3 — RECENCY
FAIL if the summary:
  • Cites only general background knowledge with no recent developments
  • Refers primarily to research older than 2 years without noting this
  • Misses obviously recent and important developments in a fast-moving topic

CRITERION 4 — DEPTH
FAIL if the summary:
  • Only provides surface-level definitions (what something IS, not how it WORKS)
  • Lacks any analytical insight or meaningful comparison
  • Is shorter than 3 substantial sentences

SCORING:
- If 2 or more summaries FAIL any criterion → NEEDS_MORE_RESEARCH
- If 1 summary has a MAJOR failure (Criterion 1 or 2) → NEEDS_MORE_RESEARCH  
- If all summaries pass all criteria → APPROVED
- If 1 summary has only a minor issue → APPROVED with note

You MUST respond in exactly this format:

VERDICT: APPROVED or NEEDS_MORE_RESEARCH

SUMMARY SCORES:
- Summary 1: PASS/FAIL — [one sentence reason]
- Summary 2: PASS/FAIL — [one sentence reason]
- Summary 3: PASS/FAIL — [one sentence reason]
- Summary 4: PASS/FAIL — [one sentence reason]
- Summary 5: PASS/FAIL — [one sentence reason]

ISSUES FOUND:
- [list specific issues, or "None" if all pass]

SPECIFIC GAPS TO ADDRESS:
- [list what the Researcher should search for next, or "None" if approved]

OVERALL ASSESSMENT:
[2-3 sentences. Be blunt. Name specific weaknesses.]
""")

    human_prompt = HumanMessage(content=f"""Original research query: {original_query}

Research summaries to evaluate:

{formatted_summaries}

Apply the strict rubric. Be a tough editor. Find the weaknesses.""")

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
    summary_scores = _extract_summary_scores(critique)
    
    if needs_more_research:
        fail_count = sum(1 for s in summary_scores if s == "FAIL")
        print(f"[CRITIC] VERDICT: NEEDS MORE RESEARCH ({fail_count} summaries failed)")
        print(f"[CRITIC] Will trigger research loop #{research_loops + 1}")
    else:
        fail_count = sum(1 for s in summary_scores if s == "FAIL")
        print(f"[CRITIC] VERDICT: APPROVED "
              f"({'all passed' if fail_count == 0 else f'{fail_count} minor issue(s)'})")

    new_loop_count = research_loops + 1 if needs_more_research else research_loops

    return {
        "critique":            critique,
        "needs_more_research": needs_more_research,
        "research_loops":      new_loop_count,
        "summary_scores":      summary_scores,
        "status_log": [
            f"Critic: {'NEEDS_MORE_RESEARCH' if needs_more_research else 'APPROVED'} "
            f"— {sum(1 for s in summary_scores if s == 'FAIL')}/5 summaries failed "
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
            f"Sub-Question: {question}\n"
            f"Content Summary: {summary}"
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
    
def _extract_summary_scores(critique_text: str) -> list[str]:
    """
    Parses the SUMMARY SCORES section to extract PASS/FAIL per summary.
    Returns list of "PASS" or "FAIL" strings, one per summary.
    """
    scores = []
    lines  = critique_text.split("\n")

    for line in lines:
        line_upper = line.upper()
        # Look for lines matching "- Summary N: PASS/FAIL"
        if "SUMMARY" in line_upper and ("PASS" in line_upper or "FAIL" in line_upper):
            if "FAIL" in line_upper:
                scores.append("FAIL")
            else:
                scores.append("PASS")

    # If parsing failed, return empty list
    return scores if scores else []