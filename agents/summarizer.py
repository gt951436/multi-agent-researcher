from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm
from graph.state import ResearchState


def summarizer_agent(state: ResearchState) -> dict:
    """
    The Summarizer Agent — third node in the LangGraph pipeline.

    Responsibility:
        For each entry in research_data, synthesize the raw key_facts
        (bullet points extracted by the Researcher) into a coherent,
        well-structured paragraph summary.

    Why it exists as a separate agent:
        - Researcher outputs fragmented bullet points
        - Writer needs polished, narrative-ready prose
        - Critic needs structured summaries to evaluate for gaps
        - Separation of concerns: extraction ≠ synthesis

    Reads from state: research_data
    Writes to state: summaries, status_log

    Args:
        state: The shared ResearchState

    Returns:
        dict with keys: summaries, status_log
    """
    print("\n[SUMMARIZER] Starting synthesis of research data...")

    research_data = state["research_data"]

    # Safety check: if researcher found nothing, handle gracefully
    if not research_data:
        print("[SUMMARIZER] No research data found in state.")
        return {
            "summaries": [],
            "status_log": ["Summarizer: No research data available to summarize."]
        }

    # temperature=0.4 -> slightly higher than factual agents
    # Allows fluent, natural prose while staying grounded in facts
    llm = get_llm(temperature=0.4)

    all_summaries = []

    for i, entry in enumerate(research_data, 1):
        question   = entry.get("question", "Unknown question")
        key_facts  = entry.get("key_facts", "No facts available")
        result_count = entry.get("result_count", 0)

        print(f"\n[SUMMARIZER] Synthesizing entry {i}/{len(research_data)}:")
        print(f"  Question: {question[:80]}...")

        # Handle case where Researcher found no results for this question
        if result_count == 0 or key_facts == "No search results were found for this question.":
            print(f"  [SUMMARIZER] No data available — flagging as gap")
            all_summaries.append({
                "question": question,
                "summary":  "RESEARCH GAP: No web results were found for this sub-question. "
                            "This area requires additional investigation.",
                "has_gap":  True
            })
            continue

        # --- System Prompt ---
        # We tell the LLM its role: synthesizer, not analyst.
        # It should convert bullet facts into readable prose —
        # not add new information or opinions.
        system_prompt = SystemMessage(content="""You are an expert research synthesizer.

Your job is to transform a list of extracted research facts into a clear, 
coherent, and well-structured summary paragraph.

Rules:
- Write in clear, professional prose (NOT bullet points)
- Only use information from the provided facts — do NOT add outside knowledge
- Preserve all specific details: names, dates, numbers, statistics
- If facts contain contradictions, acknowledge them in your summary
- If facts are sparse or incomplete, note this briefly at the end
- Length: 3-5 sentences for simple topics, up to 8 for complex ones
- Start directly with the summary — no preamble like "Here is a summary..."
- Write in third person, past or present tense (be consistent)
""")

        # --- Human Prompt ---
        # Injects the specific question and its extracted facts
        human_prompt = HumanMessage(content=f"""Research question: {question}

Extracted facts from web research:
{key_facts}

Write a coherent paragraph summary that synthesizes these facts 
into a clear answer to the research question.""")

        # LLM call — synthesizes bullet facts into prose summary
        response   = llm.invoke([system_prompt, human_prompt])
        summary    = response.content.strip()

        print(f"  [SUMMARIZER] Summary generated ({len(summary)} chars)")

        all_summaries.append({
            "question": question,
            "summary":  summary,
            "has_gap":  False   # no gap — we have a real summary
        })

    # Count how many summaries have gaps (for status log)
    gap_count  = sum(1 for s in all_summaries if s.get("has_gap", False))
    good_count = len(all_summaries) - gap_count

    print(f"\n[SUMMARIZER] Synthesis complete:")
    print(f"  Summaries generated: {good_count}")
    print(f"  Gaps flagged:        {gap_count}")

   # Confirm what we're returning
    status_entry = (
        f"Summarizer: Synthesized {good_count} summaries, "
        f"flagged {gap_count} research gaps."
    )
    print(f"  Status log: {status_entry}")

    return {
        "summaries":   all_summaries,
        "status_log":  [status_entry]   # must be a list — operator.add merges it
    }