# agents/researcher.py
import time
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm
from graph.state import ResearchState
from tools.search_tool import web_search, format_results_for_llm


def researcher_agent(state: ResearchState) -> dict:
    """
    The Researcher Agent — second node in the LangGraph pipeline.

    Responsibility:
        For each sub-question from the Planner, perform a live web
        search and use the LLM to extract relevant key facts from
        the raw search results.

    Why it loops over sub-questions:
        Each sub-question covers a distinct angle of the topic.
        Searching for all of them in one query would produce generic,
        unfocused results. Individual searches produce targeted, deep
        information per angle.

    Reads from state: sub_questions, research_loops (for context)
    Writes to state: research_data (appended), sources (appended),
                       status_log (appended)

    Note on operator.add:
        research_data and sources are Annotated with operator.add in
        the state schema. This means LangGraph MERGES the lists we
        return with whatever was already in the state. This is critical
        on loop-back runs - we don't want to wipe the previous research.

    Args:
        state: The shared ResearchState

    Returns:
        dict with keys: research_data, sources, status_log
    """
    print("\n[RESEARCHER] Starting web research...")

    sub_questions  = state["sub_questions"]
    research_loops = state.get("research_loops", 0)

    # On loop-back runs (Critic sent us back), the Critic's feedback
    # is in state["critique"]. We log it so the developer can see
    # what gaps the Critic identified.
    if research_loops > 0:
        print(f"[RESEARCHER] Loop #{research_loops} - addressing critic feedback:")
        print(f"  {state.get('critique', 'No critique available')}")

    # --- Accumulators ---
    # We collect all research here, then return them all at once.
    # Because research_data and sources use operator.add in the state
    # schema, LangGraph will APPEND these to whatever already exists
    # in state — not replace it.
    all_research_data = []
    all_sources       = []

    llm = get_llm(temperature=0.1)

    # --- Main research loop ---
    # Process each sub-question independently
    for i, question in enumerate(sub_questions, 1):
        print(f"\n[RESEARCHER] Researching sub-question {i}/{len(sub_questions)}:")
        print(f"  {question}")

        # --- Step A: Web Search ---
        # Search DuckDuckGo for this specific sub-question
        raw_results = web_search(question, max_results=5)

        if not raw_results:
            # Search returned nothing — record this gracefully
            # The Summarizer and Critic will flag the gap
            print(f"  [RESEARCHER] No results found for this question")
            all_research_data.append({
                "question":    question,
                "raw_results": [],
                "key_facts":   "No search results were found for this question.",
                "result_count": 0
            })
            continue  # move to next sub-question

        print(f"  [RESEARCHER] Found {len(raw_results)} web results")

        # --- Step B: Collect source URLs ---
        # We extract URLs now, before the LLM processes anything,
        # because we want ALL sources even if the LLM later decides
        # some snippets aren't relevant.
        for result in raw_results:
            url = result.get("url", "")
            if url and url != "No URL" and url not in all_sources:
                all_sources.append(url)

        # --- Step C: LLM processes the raw search results ---
        # The LLM reads all 5 search snippets and extracts only
        # the facts that are directly relevant to the sub-question.
        # This filters noise and makes downstream processing cleaner.
        formatted_results = format_results_for_llm(raw_results)

        system_prompt = SystemMessage(content="""You are a meticulous research analyst.

Your job is to read web search results and extract the most relevant, factual information 
related to a specific research question.

Rules:
- Extract ONLY information directly relevant to the question
- Present facts clearly and concisely as bullet points
- Preserve specific details: names, dates, numbers, statistics
- If a source contains contradictory information, note the contradiction
- Do NOT add your own opinions or knowledge — only use what the sources say
- Start directly with the bullet points, no preamble
- Format: use "• " for each bullet point
""")

        human_prompt = HumanMessage(content=f"""Research question: {question}

Web search results:
{formatted_results}

Extract the key facts from these search results that are relevant to the research question.
Present them as clear bullet points.""")

        # LLM call — extracts structured facts from noisy web snippets
        response = llm.invoke([system_prompt, human_prompt])
        key_facts = response.content.strip()

        print(f"  [RESEARCHER] Extracted key facts ({len(key_facts)} chars)")

        # --- Step D: Store structured result ---
        all_research_data.append({
            "question":     question,
            "raw_results":  raw_results,        # original search results
            "key_facts":    key_facts,           # LLM-extracted, filtered facts
            "result_count": len(raw_results)
        })

        # --- Rate limiting ---
        # DuckDuckGo will temporarily block us if we fire too many
        # requests in rapid succession. A 2-second pause between
        # searches keeps us well within their tolerance.
        # This is standard practice for any scraping/search pipeline.
        
        if i < len(sub_questions):  # no need to wait after the last one
            print(f"  [RESEARCHER] Waiting 2s before next search...")
            time.sleep(2)

    # --- Summary log ---
    total_sources  = len(all_sources)
    total_questions = len(all_research_data)
    print(f"\n[RESEARCHER] Research complete:")
    print(f"  Sub-questions researched: {total_questions}")
    print(f"  Sources collected: {total_sources}")

    return {
        "research_data": all_research_data,
        "sources":       all_sources,
        "status_log": [
            f"Researcher: Completed research on {total_questions} sub-questions, "
            f"collected {total_sources} sources "
            f"(loop #{research_loops})"
        ]
    }