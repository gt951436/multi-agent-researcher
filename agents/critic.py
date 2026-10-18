import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm, MAX_RESEARCH_LOOPS
from graph.state import ResearchState


class QuestionReview(BaseModel):
    """Evaluation of an individual sub-question summary."""
    question_id: str = Field(description="ID of the sub-question reviewed, e.g. Q1, Q2")
    status: Literal["PASS", "FAIL"] = Field(description="PASS if specific, complete, recent, and deep; FAIL otherwise")
    reason: str = Field(description="Specific, one-sentence rationale for the PASS or FAIL score")
    gaps: List[str] = Field(default_factory=list, description="Specific missing evidence points or shortcomings")
    follow_up_query: Optional[str] = Field(
        default=None,
        description="Targeted web search query designed specifically to find the missing evidence if status is FAIL"
    )


class CriticEvaluation(BaseModel):
    """Strict editorial quality evaluation across all research summaries."""
    verdict: Literal["APPROVED", "NEEDS_MORE_RESEARCH"] = Field(
        description="NEEDS_MORE_RESEARCH if >=2 summaries fail or 1 has a major failure (specificity/completeness); APPROVED otherwise"
    )
    reviews: List[QuestionReview] = Field(description="Review for each individual sub-question")
    overall_assessment: str = Field(description="2-3 sentence blunt editorial assessment naming strengths and weaknesses")


def critic_agent(state: ResearchState) -> dict:
    """
    The Critic Agent — fourth node in the LangGraph pipeline.

    Responsibility:
        Evaluates all summaries for completeness, specificity, recency,
        and depth against a strict rubric using structured Pydantic output.
        Identifies exact gaps and formulates targeted follow-up queries
        for the Researcher if re-research is needed.

    Loop-back mechanism:
        If needs_more_research=True AND research_loops < MAX_RESEARCH_LOOPS:
            → LangGraph routes to Researcher (targeted gap search)
        Else:
            → LangGraph routes forward to Writer
    """
    print("\n[CRITIC] Starting quality review of summaries...")

    summaries      = state.get("summaries", [])
    research_loops = state.get("research_loops", 0)
    original_query = state.get("query", "")

    # Safety check: if no summaries exist, cannot evaluate
    if not summaries:
        print("[CRITIC] No summaries to evaluate — forcing forward to Writer.")
        return {
            "critique":            "No summaries available to critique.",
            "critic_review":       None,
            "follow_up_queries":   [],
            "summary_scores":      [],
            "needs_more_research": False,
            "research_loops":      research_loops,
            "status_log": ["Critic: No summaries found — skipping review."]
        }

    # Loop limit guard: stop looping once maximum allowed passes are reached
    if research_loops >= MAX_RESEARCH_LOOPS:
        print(f"[CRITIC] Max research loops ({MAX_RESEARCH_LOOPS}) reached.")
        print("[CRITIC] Proceeding to Writer with notes on remaining limitations.")
        return {
            "critique": (
                f"Maximum research loops ({MAX_RESEARCH_LOOPS}) reached. "
                f"Proceeding to final report with available research. "
                f"Some gaps may remain."
            ),
            "critic_review":       None,
            "follow_up_queries":   [],
            "summary_scores":      state.get("summary_scores", ["PASS"] * len(summaries)),
            "needs_more_research": False,
            "research_loops":      research_loops,
            "status_log": [
                f"Critic: Max loops reached ({MAX_RESEARCH_LOOPS}). Proceeding to Writer."
            ]
        }

    formatted_summaries = _format_summaries_for_critique(summaries)

    system_prompt = SystemMessage(content="""You are a demanding, rigorous, skeptical research quality reviewer and editor at a high-standard publication. Your job is to reject mediocre research.

Your job is to evaluate each research summary and determine if they provide sufficient, accurate, and complete coverage of the original research query.

Evaluate each summary against this strict rubric:

CRITERION 1 — SPECIFICITY (most important)
FAIL if the summary uses vague phrases without specifics, makes claims without concrete evidence (numbers, names, dates), or contains generic filler.

CRITERION 2 — COMPLETENESS
FAIL if the summary does not actually answer its specific sub-question, or leaves major aspects unaddressed.

CRITERION 3 — RECENCY
FAIL if the summary cites only general background knowledge with no recent developments in a fast-moving topic.

CRITERION 4 — DEPTH
FAIL if the summary provides only surface-level definitions without mechanism or comparison, or is shorter than 3 substantial sentences.

SCORING RULES:
- If 2 or more summaries FAIL any criterion → verdict: NEEDS_MORE_RESEARCH
- If 1 summary has a MAJOR failure (Criterion 1 or 2) → verdict: NEEDS_MORE_RESEARCH
- If all summaries pass all criteria → verdict: APPROVED
- For every summary that FAILS, you MUST provide a concrete, targeted 'follow_up_query' that the researcher can execute on Google/DuckDuckGo to retrieve the missing information.
""")

    human_prompt = HumanMessage(content=f"""Original research query: {original_query}

Research summaries to evaluate:
{formatted_summaries}

Apply the strict rubric. For every summary that fails, provide a targeted follow-up search query to fill the gap.""")

    llm = get_llm(temperature=0.1)

    # Invoke with structured output backed by Pydantic
    try:
        structured_llm = llm.with_structured_output(CriticEvaluation)
        evaluation: CriticEvaluation = structured_llm.invoke([system_prompt, human_prompt])
    except Exception as e:
        print(f"[CRITIC] Structured output invocation failed: {e}. Attempting JSON fallback...")
        evaluation = _fallback_json_critique(llm, system_prompt, human_prompt, summaries)

    needs_more_research = (evaluation.verdict == "NEEDS_MORE_RESEARCH")
    summary_scores = [r.status for r in evaluation.reviews]
    fail_count = summary_scores.count("FAIL")

    # Extract targeted follow-up queries for failed questions
    follow_up_queries = []
    if needs_more_research:
        for r in evaluation.reviews:
            if r.status == "FAIL":
                query_str = r.follow_up_query or f"{original_query} {r.question_id} details"
                follow_up_queries.append({
                    "question_id": r.question_id,
                    "query": query_str,
                    "gaps": r.gaps,
                    "rationale": r.reason
                })

    critique_markdown = _format_critique_markdown(evaluation)

    print(f"[CRITIC] Verdict: {evaluation.verdict} ({fail_count}/{len(summaries)} failed)")
    if follow_up_queries:
        print(f"[CRITIC] Generated {len(follow_up_queries)} targeted follow-up queries:")
        for fq in follow_up_queries:
            print(f"  [{fq['question_id']}] {fq['query']}")

    return {
        "critique":            critique_markdown,
        "critic_review":       evaluation.model_dump(),
        "follow_up_queries":   follow_up_queries,
        "summary_scores":      summary_scores,
        "needs_more_research": needs_more_research,
        "research_loops":      research_loops,
        "status_log": [
            f"Critic: {evaluation.verdict} — {fail_count}/{len(summaries)} summaries failed "
            f"(loop {research_loops}/{MAX_RESEARCH_LOOPS})"
        ]
    }


def _format_summaries_for_critique(summaries: list[dict]) -> str:
    """Renders summaries into a clearly structured block labeled with question IDs."""
    formatted = []
    for i, entry in enumerate(summaries, 1):
        q_id = entry.get("question_id", f"Q{i}")
        question = entry.get("question", "Unknown question")
        summary  = entry.get("summary", "No summary available")
        has_gap  = entry.get("has_gap", False)

        gap_note = " [FLAGGED AS GAP BY SUMMARIZER]" if has_gap else ""
        formatted.append(
            f"SUMMARY {i} [ID: {q_id}]{gap_note}:\n"
            f"Sub-Question: {question}\n"
            f"Content Summary: {summary}"
        )
    return "\n\n".join(formatted)


def _format_critique_markdown(evaluation: CriticEvaluation) -> str:
    """Generates clean markdown report from structured Critic evaluation."""
    lines = [
        f"VERDICT: {evaluation.verdict}\n",
        "SUMMARY SCORES:"
    ]
    for r in evaluation.reviews:
        lines.append(f"- {r.question_id}: {r.status} — {r.reason}")

    issues = [f"- [{r.question_id}] {gap}" for r in evaluation.reviews for gap in r.gaps]
    lines.append("\nISSUES FOUND:")
    lines.append("\n".join(issues) if issues else "- None")

    follow_ups = [f"- [{r.question_id}] {r.follow_up_query}" for r in evaluation.reviews if r.follow_up_query and r.status == "FAIL"]
    lines.append("\nSPECIFIC GAPS TO ADDRESS:")
    lines.append("\n".join(follow_ups) if follow_ups else "- None")

    lines.append(f"\nOVERALL ASSESSMENT:\n{evaluation.overall_assessment}")
    return "\n".join(lines)


def _fallback_json_critique(llm, system_prompt, human_prompt, summaries: list[dict]) -> CriticEvaluation:
    """
    Fallback parser if structured output binding fails.
    Prompts the LLM for raw JSON and validates with Pydantic.
    """
    json_prompt = HumanMessage(content=human_prompt.content + "\n\nOutput your response strictly as valid JSON matching the CriticEvaluation schema.")
    response = llm.invoke([system_prompt, json_prompt])
    text = response.content.strip()

    # Clean markdown json fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        data = json.loads(text)
        return CriticEvaluation.model_validate(data)
    except Exception as parse_err:
        print(f"[CRITIC] JSON parsing failed ({parse_err}). Falling back to safe conservative evaluation.")
        reviews = []
        for i, s in enumerate(summaries, 1):
            qid = s.get("question_id", f"Q{i}")
            reviews.append(QuestionReview(
                question_id=qid,
                status="PASS" if not s.get("has_gap") else "FAIL",
                reason="Evaluated via fallback parser",
                gaps=["Review format error, manual verification suggested"] if s.get("has_gap") else [],
                follow_up_query=None
            ))
        verdict = "NEEDS_MORE_RESEARCH" if any(r.status == "FAIL" for r in reviews) else "APPROVED"
        return CriticEvaluation(
            verdict=verdict,
            reviews=reviews,
            overall_assessment="Automated review encountered formatting anomalies; proceeding with conservative scoring."
        )
