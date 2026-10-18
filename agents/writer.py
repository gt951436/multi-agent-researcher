from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm
from graph.state import ResearchState


def writer_agent(state: ResearchState) -> dict:
    """
    The Writer Agent — fifth and final node in the LangGraph pipeline.

    Responsibility:
        Compiles all verified summaries, source URLs, and metadata
        into a polished, structured markdown research report that
        is ready for the end user to read and download.

    Why it exists as a separate agent:
        Compilation is a distinct skill from research and summarization.
        The Writer's job is narrative structure, flow, and presentation —
        not fact-finding or quality control. Keeping it separate means
        each agent can be optimized independently.

    What makes this agent different from the others:
        - Uses the highest temperature (0.5) for fluent prose
        - Draws on ALL previous agent outputs simultaneously
        - Produces the system's only user-facing output
        - Adds metadata: timestamp, source count, confidence rating

    Reads from state:  query, sub_questions, summaries, sources,
                       critique, research_loops
    Writes to state:   final_report, status_log

    Args:
        state: The shared ResearchState

    Returns:
        dict with keys: final_report, status_log
    """
    print("\n[WRITER] Starting final report compilation...")    # --- Extract everything needed from state ---
    query          = state["query"]
    sub_questions  = state.get("sub_questions", [])
    summaries      = state.get("summaries", [])
    sources        = state.get("sources", [])
    critique       = state.get("critique", "")
    summary_scores = state.get("summary_scores", [])
    research_loops = state.get("research_loops", 0)

    # Deduplicate sources preserving insertion order
    unique_sources = list(dict.fromkeys(sources))
    source_count   = len(unique_sources)

    # --- Safety check ---
    if not summaries:
        print("[WRITER] No summaries available — generating minimal report.")
        fallback_report = _generate_fallback_report(query)
        return {
            "final_report": fallback_report,
            "confidence": "Low",
            "status_log": ["Writer: Generated fallback report (no summaries available)."]
        }

    # --- Build the structured content block ---
    # Map each question to its latest summary using question_id
    content_block = _format_content_for_writer(sub_questions, summaries)

    # --- Build the sources block ---
    # Format unique collected URLs as a numbered reference list.
    sources_block = _format_sources_block(unique_sources)

    # --- Determine confidence level ---
    confidence = _determine_confidence(research_loops, critique, summaries, summary_scores)

    # --- Generate timestamp ---
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")

    # --- System Prompt ---
    # Writer's role: architect of the final document.
    system_prompt = SystemMessage(content="""You are an expert research writer and report architect.

Your job is to compile a set of research summaries into a single, 
polished, well-structured research report in markdown format.

Report structure (follow this exactly):
1. A title (# heading) derived from the research query
2. An executive summary section (## Executive Summary) — 
   3-4 sentences giving the reader the key takeaway before they read details
3. One section per research sub-question (## numbered heading) —
   expand the provided summary into a well-written section with 
   smooth prose and clear transitions
4. A ## Key Takeaways section — 
   5-7 bullet points distilling the most important findings
5. A ## Conclusion section — 
   3-4 sentences synthesizing the overall picture and implications

Writing rules:
- Write in clear, professional but accessible prose
- Use markdown formatting: ## headings, **bold** for key terms, 
  bullet points where appropriate within sections
- Each section should flow naturally into the next
- Do NOT repeat the same information across sections
- Do NOT add information not present in the provided summaries
- Do NOT use the sub-question as the section heading verbatim — 
  convert it into a clean, readable heading
- Write the report as if it will be read by a knowledgeable but 
  non-specialist audience
""")

    human_prompt = HumanMessage(content=f"""Research query: {query}

Research content (sub-questions and their latest synthesized summaries):
{content_block}

Write the research report following the structure in your instructions.
Do NOT include the sources section — that will be appended separately.
Do NOT include a Research Notes section — that will be appended separately.
Just write sections 1 through 5 (title through conclusion).""")

    llm      = get_llm(temperature=0.5)
    response = llm.invoke([system_prompt, human_prompt])
    report_body = response.content.strip()

    print(f"[WRITER] Report body generated ({len(report_body)} chars)")

    # Assemble the complete report with synchronized source count
    final_report = _assemble_final_report(
        report_body    = report_body,
        sources_block  = sources_block,
        confidence     = confidence,
        research_loops = research_loops,
        critique       = critique,
        timestamp      = timestamp,
        source_count   = source_count
    )

    print(f"[WRITER] Final report assembled ({len(final_report)} chars)")
    print(f"[WRITER] Report contains {final_report.count(chr(10))} lines")

    return {
        "final_report": final_report,
        "confidence":   confidence,
        "status_log": [
            f"Writer: Report compiled successfully. "
            f"{len(sub_questions)} sections, {source_count} unique sources, "
            f"confidence: {confidence}."
        ]
    }


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def _format_content_for_writer(
    sub_questions: list,
    summaries: list[dict]
) -> str:
    """
    Pairs each sub-question with its latest summary using stable question IDs.
    Eliminates zip truncation and guarantees that follow-up summaries are consumed.
    """
    summary_map = {}
    for s in summaries:
        qid = s.get("question_id")
        if qid:
            summary_map[qid] = s

    blocks = []
    for i, item in enumerate(sub_questions, 1):
        if isinstance(item, dict):
            qid = item.get("id", f"Q{i}")
            question = item.get("question", str(item))
        else:
            qid = f"Q{i}"
            question = str(item)

        # Match by question_id; fallback to positional index if missing
        summary_entry = summary_map.get(qid)
        if not summary_entry and (i - 1) < len(summaries):
            summary_entry = summaries[i - 1]

        summary_text = summary_entry.get("summary", "No summary available.") if summary_entry else "No summary available."
        has_gap      = summary_entry.get("has_gap", False) if summary_entry else True

        gap_note = "\n[NOTE: This section has limited research coverage]" if has_gap else ""

        blocks.append(
            f"SUB-QUESTION {i} [{qid}]: {question}\n"
            f"SUMMARY: {summary_text}"
            f"{gap_note}"
        )

    return "\n\n---\n\n".join(blocks)


def _format_sources_block(sources: list[str]) -> str:
    """
    Formats unique source URLs into a clean numbered markdown reference list.
    """
    if not sources:
        return "_No sources were collected during research._"

    # Deduplicate while preserving insertion order
    unique_sources = list(dict.fromkeys(sources))

    lines = []
    for i, url in enumerate(unique_sources, 1):
        lines.append(f"{i}. {url}")

    return "\n".join(lines)


def _determine_confidence(
    research_loops: int,
    critique:       str,
    summaries:      list[dict],
    summary_scores: list[str] = None
) -> str:
    """
    Calculates an honest confidence rating using structured quality signals:
    1. Unresolved gaps flagged by Summarizer
    2. Critic's structured scores and verdict
    3. Number of re-research passes needed

    Scale:
        High   → 0 gaps + clean APPROVED + 0 loops + 0 fails
        Medium → approved after re-research loops OR minor non-critical issues
        Low    → >=2 gaps OR unapproved with remaining critical issues
    """
    upper = critique.upper()
    gap_count = sum(1 for s in summaries if s.get("has_gap", False))
    approved = ("NEEDS_MORE_RESEARCH" not in upper and "APPROVED" in upper)

    if summary_scores:
        fail_count = summary_scores.count("FAIL")
    else:
        fail_count = upper.count(": FAIL")

    if gap_count >= 2 or not approved or fail_count >= 3:
        return "Low"

    if approved and research_loops == 0 and gap_count == 0 and fail_count == 0:
        return "High"

    return "Medium"



def _assemble_final_report(
    report_body:    str,
    sources_block:  str,
    confidence:     str,
    research_loops: int,
    critique:       str,
    timestamp:      str,
    source_count:   int
) -> str:
    """
    Appends the programmatically-generated sections to the
    LLM-written report body.

    Why append instead of asking the LLM to generate these sections:
        Sources, metadata, and research notes have precise formatting
        requirements. Asking an LLM to format a numbered URL list
        reliably is harder than just doing it in Python. We generate
        the report body with the LLM (where prose quality matters)
        and append structured sections in code (where precision matters).

    Sections appended:
        - ## Sources — numbered reference list
        - ## Research Notes — methodology, confidence, limitations

    Args:
        report_body:    The LLM-generated main report text
        sources_block:  Pre-formatted numbered source list
        confidence:     "High" / "Medium" / "Low"
        research_loops: Number of research iterations completed
        critique:       Critic's final feedback text
        timestamp:      Human-readable generation time
        source_count:   Total number of unique sources

    Returns:
        The complete, final markdown report string
    """
    # --- Confidence badge line ---
    confidence_display = {
        "High":   "🟢 High",
        "Medium": "🟡 Medium",
        "Low":    "🔴 Low"
    }.get(confidence, "🟡 Medium")

    # --- Research methodology note ---
    # Transparent description of how this report was generated
    methodology = (
        f"This report was generated by an autonomous multi-agent research system. "
        f"A Planner agent decomposed the query into sub-questions, a Researcher agent "
        f"performed live web searches (DuckDuckGo) for each sub-question, a Summarizer "
        f"synthesized the findings, and a Critic reviewed quality"
    )

    if research_loops > 0:
        methodology += (
            f" — triggering {research_loops} additional research "
            f"loop{'s' if research_loops > 1 else ''} to address identified gaps"
        )

    methodology += f". {source_count} web sources were consulted."

    # --- Critic summary for notes section ---
    # Extract just the OVERALL ASSESSMENT from the critique
    # to show in Research Notes without dumping the full critique
    critic_summary = _extract_overall_assessment(critique)

    # --- Assemble everything ---
    report = f"""{report_body}

---

## Sources

{sources_block}

---

## Research Notes

| Field | Value |
|---|---|
| Generated | {timestamp} |
| Confidence | {confidence_display} |
| Research loops | {research_loops} |
| Sources consulted | {source_count} |
| System | Multi-Agent Research Assistant (LLaMA 3.3 70B via Groq) |

**Methodology:** {methodology}

**Quality Review:** {critic_summary}

_This report was compiled by an AI research assistant. All information 
is sourced from live web searches conducted at the time of generation. 
Verify critical claims with primary sources before use._
"""

    return report.strip()


def _extract_overall_assessment(critique: str) -> str:
    """
    Extracts the OVERALL ASSESSMENT section from the Critic's
    full critique text to use in Research Notes.

    If the section can't be found (edge case), returns a
    generic summary instead of crashing.

    Args:
        critique: Full critique text from Critic agent

    Returns:
        Just the overall assessment paragraph, or a fallback string
    """
    if not critique:
        return "Quality review was not completed."

    # Look for the OVERALL ASSESSMENT section header
    marker = "OVERALL ASSESSMENT:"
    upper  = critique.upper()
    idx    = upper.find(marker)

    if idx != -1:
        # Extract everything after the marker
        assessment = critique[idx + len(marker):].strip()
        # Clean up any trailing section headers that might follow
        # (in case the LLM added extra content after the assessment)
        return assessment.split("\n\n")[0].strip()

    # Fallback: return the last paragraph of the critique
    paragraphs = [p.strip() for p in critique.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else "See critique for details."


def _generate_fallback_report(query: str) -> str:
    """
    Minimal report for the edge case where the Writer
    is called but no summaries exist in state.

    This should never happen in a correctly wired graph,
    but defensive programming prevents a silent failure.
    """
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
    return f"""# Research Report: {query}

## Notice

This report could not be generated because no research summaries 
were available. This may indicate an issue with the research pipeline.

**Generated:** {timestamp}  
**Status:** Failed — no summaries available
"""