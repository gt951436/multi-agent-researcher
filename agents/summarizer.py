from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm
from graph.state import ResearchState


def summarizer_agent(state: ResearchState) -> dict:
    """
    The Summarizer Agent — third node in the LangGraph pipeline.

    Responsibility:
        Synthesizes research evidence into clear, authoritative prose summaries,
        exactly one summary per sub-question.

    Key Architectural Principles:
        1. Question ID Association: Every summary is strictly tagged with its question_id (e.g. Q1).
        2. Incremental Refresh: On re-research passes, questions with newly acquired evidence
           are updated with combined synthesis. Questions without new evidence retain their
           existing high-quality summaries, avoiding redundant LLM calls and token waste.
        3. Deterministic Cardinality: len(summaries) strictly equals len(sub_questions),
           permanently preventing duplicate accumulation or position-mismatch bugs.

    Reads from state: sub_questions, research_data, summaries, research_loops
    Writes to state:  summaries, status_log
    """
    print("\n[SUMMARIZER] Starting synthesis of research data...")

    sub_questions  = state.get("sub_questions", [])
    research_data  = state.get("research_data", [])
    prev_summaries = state.get("summaries", [])
    research_loops = state.get("research_loops", 0)

    if not research_data and not prev_summaries:
        print("[SUMMARIZER] No research data found in state.")
        return {
            "summaries": [],
            "status_log": ["Summarizer: No research data available to summarize."]
        }

    llm = get_llm(temperature=0.4)

    # Index existing summaries by question_id: qid -> summary dict
    existing_summary_map = {}
    for s in prev_summaries:
        qid = s.get("question_id")
        if qid:
            existing_summary_map[qid] = s

    # Group all collected evidence by question_id
    evidence_by_qid = {}
    for entry in research_data:
        qid = entry.get("question_id")
        if not qid:
            continue
        if qid not in evidence_by_qid:
            evidence_by_qid[qid] = []
        evidence_by_qid[qid].append(entry)

    all_summaries = []
    synthesized_count = 0
    reused_count = 0

    for i, item in enumerate(sub_questions, 1):
        if isinstance(item, dict):
            qid = item.get("id", f"Q{i}")
            question = item.get("question", str(item))
        else:
            qid = f"Q{i}"
            question = str(item)

        entries = evidence_by_qid.get(qid, [])

        # Check if we have new evidence for this question from the current pass
        has_new_evidence = any(e.get("iteration", 0) == research_loops for e in entries) if research_loops > 0 else True

        # If question already has an approved summary and NO new evidence was gathered for it, reuse it
        if qid in existing_summary_map and not has_new_evidence:
            print(f"  [SUMMARIZER] [{qid}] Reusing existing summary (no new evidence needed)")
            all_summaries.append(existing_summary_map[qid])
            reused_count += 1
            continue

        print(f"\n[SUMMARIZER] Synthesizing summary for [{qid}]: {question[:80]}...")

        # If no results were found across all passes for this question
        valid_entries = [e for e in entries if e.get("result_count", 0) > 0 and e.get("key_facts") != "No search results were found for this question."]
        if not valid_entries:
            print(f"  [SUMMARIZER] [{qid}] No valid evidence found — flagging as research gap")
            all_summaries.append({
                "question_id": qid,
                "question": question,
                "summary":  f"RESEARCH GAP: Insufficient web evidence retrieved for '{question}'. Additional investigation recommended.",
                "has_gap":  True,
                "iteration": research_loops
            })
            continue

        # Combine facts from all evidence entries for this question
        facts_sections = []
        for e in entries:
            tag = "Targeted Gap Search Facts" if e.get("is_targeted_gap") else "Initial Research Facts"
            iter_num = e.get("iteration", 0)
            facts_sections.append(f"--- {tag} (Pass #{iter_num}) ---\n{e.get('key_facts', '')}")

        combined_facts = "\n\n".join(facts_sections)

        system_prompt = SystemMessage(content="""You are an expert research synthesizer.
Your job is to transform extracted research facts into a clear, authoritative, coherent summary paragraph.

CRITICAL INSTRUCTIONS:
- Write in clear, professional prose (NOT bullet points)
- You MUST preserve all source citation markers (e.g. [S1], [S2], [S1, S3]) directly attached to their respective factual statements in the prose.
- Example: "Recent experimental evaluations demonstrated 127-qubit coherence [S1], though cryogenic heat dissipation remains an unresolved obstacle to scaling [S2, S3]."
- Only use information from the provided facts — do NOT hallucinate outside claims.
- Preserve specific details: names, numbers, statistics, benchmarks, dates.
- Length: 3 to 6 comprehensive sentences.
- Start directly with the synthesized prose without introductory filler.
""")

        human_prompt = HumanMessage(content=f"""Research question: {question}

Extracted research facts with source citations:
{combined_facts}

Synthesize these facts into an authoritative, coherent paragraph summary answering the research question. Preserve source citation tags [S_n] on every material statement:""")


        response = llm.invoke([system_prompt, human_prompt])
        summary = response.content.strip()

        print(f"  [SUMMARIZER] [{qid}] Summary generated ({len(summary)} chars)")
        all_summaries.append({
            "question_id": qid,
            "question": question,
            "summary": summary,
            "has_gap": False,
            "iteration": research_loops
        })
        synthesized_count += 1

    gap_count = sum(1 for s in all_summaries if s.get("has_gap", False))

    print(f"\n[SUMMARIZER] Synthesis complete: {synthesized_count} synthesized, {reused_count} reused, {gap_count} gaps.")
    status_entry = (
        f"Summarizer: Generated {len(all_summaries)} summaries "
        f"({synthesized_count} refreshed, {gap_count} gaps, pass #{research_loops})"
    )

    return {
        "summaries": all_summaries,
        "status_log": [status_entry]
    }
