# agents/researcher.py
import time
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm
from graph.state import ResearchState
from tools.search_tool import web_search, format_results_for_llm


def researcher_agent(state: ResearchState) -> dict:
    """
    The Researcher Agent — second node in the LangGraph pipeline.

    Dual-Mode Execution:
        1. Initial Pass (research_loops == 0):
           Loops over all planned sub-questions from Planner and gathers baseline evidence.
        2. Targeted Follow-up Pass (research_loops > 0 or follow_up_queries present):
           Does NOT repeat initial searches. Instead, executes targeted queries generated
           by the Critic to resolve specific evidence gaps for failed questions.

    Reads from state: sub_questions, follow_up_queries, research_loops
    Writes to state:  research_data (appended), sources (appended),
                      research_loops (updated), status_log (appended)
    """
    print("\n[RESEARCHER] Starting web research...")

    sub_questions     = state.get("sub_questions", [])
    follow_up_queries = state.get("follow_up_queries", [])
    research_loops    = state.get("research_loops", 0)

    # Determine execution mode: Initial baseline vs Targeted re-research
    is_follow_up = len(follow_up_queries) > 0

    if is_follow_up:
        current_pass = research_loops + 1
        print(f"\n[RESEARCHER] Loop #{current_pass} — Targeting {len(follow_up_queries)} specific gaps identified by Critic:")
        for fq in follow_up_queries:
            print(f"  [{fq.get('question_id')}] Query: {fq.get('query')}")
    else:
        current_pass = 0
        print(f"\n[RESEARCHER] Initial Pass (Loop #0) — Researching {len(sub_questions)} planned sub-questions")

    all_research_data = []
    all_sources       = []
    llm               = get_llm(temperature=0.1)

    # Build question lookup map: question_id -> question_text
    question_map = {}
    for i, item in enumerate(sub_questions, 1):
        if isinstance(item, dict):
            qid = item.get("id", f"Q{i}")
            qtext = item.get("question", str(item))
        else:
            qid = f"Q{i}"
            qtext = str(item)
        question_map[qid] = qtext

    # --- Mode 1: Targeted Follow-up Research ---
    if is_follow_up:
        for i, fq in enumerate(follow_up_queries, 1):
            qid = fq.get("question_id", f"Q{i}")
            search_query = fq.get("query", "")
            gaps = fq.get("gaps", [])
            original_q = question_map.get(qid, search_query)

            print(f"\n[RESEARCHER] Targeted Search {i}/{len(follow_up_queries)} for [{qid}]:")
            print(f"  Target Query: {search_query}")
            print(f"  Gaps to fill: {', '.join(gaps) if gaps else 'General enhancement'}")

            raw_results = web_search(search_query, max_results=5)

            if not raw_results:
                print(f"  [RESEARCHER] No results found for follow-up query: '{search_query}'")
                all_research_data.append({
                    "id": f"E_{qid}_{current_pass}_{int(time.time())}",
                    "question_id": qid,
                    "question": original_q,
                    "search_query": search_query,
                    "raw_results": [],
                    "key_facts": "Follow-up targeted search yielded no new web results.",
                    "result_count": 0,
                    "iteration": current_pass,
                    "is_targeted_gap": True,
                    "gaps_addressed": gaps
                })
                continue

            # Collect source URLs
            for result in raw_results:
                url = result.get("url", "")
                if url and url != "No URL" and url not in all_sources:
                    all_sources.append(url)

            formatted_results = format_results_for_llm(raw_results)

            system_prompt = SystemMessage(content="""You are a precision research analyst closing factual gaps.
Extract concrete facts from these search results that directly address the identified missing evidence.
Rules:
- Focus specifically on the identified gaps: names, figures, dates, verified metrics
- Present facts concisely as bullet points ("• ")
- Preserve exact technical specifics
- Do not invent outside facts
""")

            human_prompt = HumanMessage(content=f"""Sub-Question: {original_q}
Search Query: {search_query}
Gaps to resolve: {gaps}

Search Results:
{formatted_results}

Extract the facts that resolve the gaps above:""")

            response = llm.invoke([system_prompt, human_prompt])
            key_facts = response.content.strip()

            all_research_data.append({
                "id": f"E_{qid}_{current_pass}_{int(time.time())}",
                "question_id": qid,
                "question": original_q,
                "search_query": search_query,
                "raw_results": raw_results,
                "key_facts": key_facts,
                "result_count": len(raw_results),
                "iteration": current_pass,
                "is_targeted_gap": True,
                "gaps_addressed": gaps
            })

            if i < len(follow_up_queries):
                time.sleep(2)

    # --- Mode 2: Baseline Initial Research ---
    else:
        for i, item in enumerate(sub_questions, 1):
            if isinstance(item, dict):
                qid = item.get("id", f"Q{i}")
                question = item.get("question", str(item))
            else:
                qid = f"Q{i}"
                question = str(item)

            print(f"\n[RESEARCHER] Researching sub-question {i}/{len(sub_questions)} [{qid}]:")
            print(f"  {question}")

            raw_results = web_search(question, max_results=5)

            if not raw_results:
                print(f"  [RESEARCHER] No results found for [{qid}]")
                all_research_data.append({
                    "id": f"E_{qid}_0_{int(time.time())}",
                    "question_id": qid,
                    "question": question,
                    "search_query": question,
                    "raw_results": [],
                    "key_facts": "No search results were found for this question.",
                    "result_count": 0,
                    "iteration": 0,
                    "is_targeted_gap": False,
                    "gaps_addressed": []
                })
                continue

            for result in raw_results:
                url = result.get("url", "")
                if url and url != "No URL" and url not in all_sources:
                    all_sources.append(url)

            formatted_results = format_results_for_llm(raw_results)

            system_prompt = SystemMessage(content="""You are a meticulous research analyst.
Extract the most relevant, factual information from web search results for the given research question.
Rules:
- Extract ONLY information directly relevant to the question
- Present facts clearly and concisely as bullet points ("• ")
- Preserve specific details: names, dates, numbers, statistics
- Note any contradictions across sources
- Start directly with the bullet points, no preamble
""")

            human_prompt = HumanMessage(content=f"""Research question: {question}

Web search results:
{formatted_results}

Extract the key facts relevant to this question:""")

            response = llm.invoke([system_prompt, human_prompt])
            key_facts = response.content.strip()

            all_research_data.append({
                "id": f"E_{qid}_0_{int(time.time())}",
                "question_id": qid,
                "question": question,
                "search_query": question,
                "raw_results": raw_results,
                "key_facts": key_facts,
                "result_count": len(raw_results),
                "iteration": 0,
                "is_targeted_gap": False,
                "gaps_addressed": []
            })

            if i < len(sub_questions):
                time.sleep(2)

    total_sources = len(all_sources)
    total_entries = len(all_research_data)
    print(f"\n[RESEARCHER] Research completed: {total_entries} queries searched, {total_sources} new sources collected (pass #{current_pass}).")

    return {
        "research_data": all_research_data,
        "sources":       all_sources,
        "research_loops": current_pass,
        "follow_up_queries": [],  # Clear follow_up_queries now that they are executed
        "status_log": [
            f"Researcher: Completed {'targeted re-research' if is_follow_up else 'initial research'} "
            f"({total_entries} items, {total_sources} sources, pass #{current_pass})"
        ]
    }
