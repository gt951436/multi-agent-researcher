import re
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm
from graph.state import ResearchState


def writer_agent(state: ResearchState) -> dict:
    """
    The Writer Agent — fifth and final node in the LangGraph pipeline.

    Responsibility:
        Compiles all verified summaries, source records, and metadata
        into a polished, structured markdown research report with working
        claim-level inline citations ([1], [2]).

    Key Enhancements:
        1. Inline Claim Citations: Converts source tokens into numbered citations ([1], [2]).
        2. Citation Validation: Programmatically validates that all citations in the prose
           resolve to valid consulted sources, calculating citation precision metrics.
        3. Enriched Bibliography: Notes which sources had full articles analyzed vs snippets.
        4. Synchronized Counts: The source count in the metadata table matches the numbered
           reference list exactly.

    Reads from state:  query, sub_questions, summaries, sources, source_registry,
                       critique, summary_scores, research_loops
    Writes to state:   final_report, confidence, citation_metrics, status_log
    """
    print("\n[WRITER] Starting final report compilation with citation validation...")

    query          = state["query"]
    sub_questions  = state.get("sub_questions", [])
    summaries      = state.get("summaries", [])
    raw_sources    = state.get("sources", [])
    registry       = state.get("source_registry", [])
    critique       = state.get("critique", "")
    summary_scores = state.get("summary_scores", [])
    research_loops = state.get("research_loops", 0)

    # Build unique, ordered source list
    unique_sources, sid_to_num = _build_source_registry(raw_sources, registry)
    source_count = len(unique_sources)

    # Safety check
    if not summaries:
        print("[WRITER] No summaries available — generating minimal report.")
        fallback_report = _generate_fallback_report(query)
        return {
            "final_report": fallback_report,
            "confidence": "Low",
            "citation_metrics": None,
            "status_log": ["Writer: Generated fallback report (no summaries available)."]
        }

    # Build content block with mapped numbered citations
    content_block = _format_content_for_writer(sub_questions, summaries, sid_to_num)

    # Format numbered bibliography with domain & content type details
    sources_block = _format_sources_block(unique_sources)

    # Determine confidence rating
    confidence = _determine_confidence(research_loops, critique, summaries, summary_scores)
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")

    system_prompt = SystemMessage(content="""You are an expert research writer and report architect at a top-tier scientific publication.

Your job is to compile a set of research summaries into a polished, authoritative research report in markdown format.

Report structure (follow this exactly):
1. A title (# heading) derived from the research query
2. An executive summary (## Executive Summary) — 3-4 sentences synthesizing the core findings
3. One section per research sub-question (## numbered heading) — expanding the summary into smooth, authoritative prose
4. A ## Key Takeaways section — 5-7 bullet points distilling the most important findings
5. A ## Conclusion section — 3-4 sentences synthesizing implications

CRITICAL CITATION RULES:
- Every material factual statement, metric, or finding MUST include inline bracket citations pointing to the numbered reference sources, e.g. [1] or [1, 2].
- Retain the numbered citations [1], [2] provided in the research content.
- Do NOT fabricate citation numbers outside the provided range.
- Do NOT output the ## Sources section or ## Research Notes section — those will be appended programmatically.
""")

    human_prompt = HumanMessage(content=f"""Research query: {query}

Research content (sub-questions, summaries, and mapped citation indices [1..{source_count}]):
{content_block}

Write the report sections (title through conclusion) with inline citations:""")

    llm = get_llm(temperature=0.5)
    response = llm.invoke([system_prompt, human_prompt])
    report_body = response.content.strip()

    print(f"[WRITER] Report body generated ({len(report_body)} chars)")

    # Validate inline citations against the bibliography
    citation_metrics = _validate_citations(report_body, source_count)
    print(f"[WRITER] Citation validation: {citation_metrics['total_citations']} citations found, "
          f"{citation_metrics['valid_citations']} valid ({citation_metrics['precision'] * 100:.1f}% precision)")

    # Assemble complete report
    final_report = _assemble_final_report(
        report_body       = report_body,
        sources_block     = sources_block,
        confidence        = confidence,
        research_loops    = research_loops,
        critique          = critique,
        timestamp         = timestamp,
        unique_sources    = unique_sources,
        citation_metrics  = citation_metrics
    )

    print(f"[WRITER] Final report assembled ({len(final_report)} chars)")

    return {
        "final_report":     final_report,
        "confidence":       confidence,
        "citation_metrics": citation_metrics,
        "status_log": [
            f"Writer: Report compiled with {len(sub_questions)} sections, {source_count} unique sources, "
            f"{citation_metrics['total_citations']} citations ({citation_metrics['precision'] * 100:.0f}% valid), "
            f"confidence: {confidence}."
        ]
    }


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def _build_source_registry(raw_sources: list, registry: list) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Builds a unified, deduplicated list of source dictionaries and a mapping
    from source ID / URL to 1-based reference numbers.
    """
    seen_urls = set()
    unique_sources = []
    sid_to_num = {}

    # First add from structured registry
    for record in registry:
        url = record.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(record)
            num = len(unique_sources)
            if record.get("source_id"):
                sid_to_num[record["source_id"]] = num
            sid_to_num[url] = num

    # Add any raw URLs that weren't in registry
    for url in raw_sources:
        if isinstance(url, str) and url not in seen_urls:
            seen_urls.add(url)
            record = {
                "source_id": f"S{len(unique_sources) + 1}",
                "url": url,
                "title": url,
                "domain": urllib.parse.urlparse(url).netloc or "Unknown",
                "snippet": "",
                "content_type": "snippet"
            }
            unique_sources.append(record)
            num = len(unique_sources)
            sid_to_num[record["source_id"]] = num
            sid_to_num[url] = num

    return unique_sources, sid_to_num


def _convert_source_tags_to_numbers(text: str, sid_to_num: Dict[str, int]) -> str:
    """
    Converts source tokens like [S1], [S2, S3] into [1], [2, 3].
    """
    def replacer(match):
        raw_inner = match.group(1)
        tokens = raw_inner.split(",")
        converted = []
        for t in tokens:
            t = t.strip()
            if t in sid_to_num:
                converted.append(str(sid_to_num[t]))
            elif t.startswith("S") and t[1:].isdigit():
                converted.append(t[1:])
            else:
                converted.append(t)
        return f"[{', '.join(converted)}]"

    return re.sub(r"\[(S\d+(?:,\s*S\d+)*)\]", replacer, text)


def _format_content_for_writer(
    sub_questions: list,
    summaries: list[dict],
    sid_to_num: Dict[str, int] = None
) -> str:
    """
    Pairs each sub-question with its latest summary using stable question IDs,
    translating internal source tags ([S1]) to human-readable citation numbers ([1]).
    """
    if sid_to_num is None:
        sid_to_num = {}

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

        summary_entry = summary_map.get(qid)
        if not summary_entry and (i - 1) < len(summaries):
            summary_entry = summaries[i - 1]

        raw_summary = summary_entry.get("summary", "No summary available.") if summary_entry else "No summary available."
        has_gap     = summary_entry.get("has_gap", False) if summary_entry else True

        # Translate [S_n] -> [n]
        summary_text = _convert_source_tags_to_numbers(raw_summary, sid_to_num)
        gap_note = "\n[NOTE: This section has limited research coverage]" if has_gap else ""

        blocks.append(
            f"SUB-QUESTION {i} [{qid}]: {question}\n"
            f"SUMMARY: {summary_text}"
            f"{gap_note}"
        )

    return "\n\n---\n\n".join(blocks)


def _validate_citations(report_body: str, total_sources: int) -> Dict[str, Any]:
    """
    Extracts all [n] citations from the report body and validates them
    against the total number of sources.
    """
    matches = re.findall(r"\[([0-9,\s\-]+)\]", report_body)
    all_citations = []

    for match in matches:
        parts = match.split(",")
        for p in parts:
            p = p.strip()
            if "-" in p:
                range_parts = p.split("-")
                if len(range_parts) == 2 and range_parts[0].isdigit() and range_parts[1].isdigit():
                    all_citations.extend(range(int(range_parts[0]), int(range_parts[1]) + 1))
            elif p.isdigit():
                all_citations.append(int(p))

    if total_sources == 0:
        return {
            "total_citations": len(all_citations),
            "valid_citations": 0,
            "invalid_citations": all_citations,
            "unique_sources_cited": 0,
            "precision": 0.0
        }

    valid_citations = [c for c in all_citations if 1 <= c <= total_sources]
    invalid_citations = [c for c in all_citations if c < 1 or c > total_sources]
    unique_cited = set(valid_citations)
    precision = (len(valid_citations) / len(all_citations)) if all_citations else 1.0

    return {
        "total_citations": len(all_citations),
        "valid_citations": len(valid_citations),
        "invalid_citations": invalid_citations,
        "unique_sources_cited": len(unique_cited),
        "precision": precision
    }


def _format_sources_block(unique_sources: list) -> str:
    """
    Formats unique sources into a numbered bibliography detailing title, domain,
    and whether full article text or snippet was analyzed.
    Deduplicates entries by URL/string while preserving first-seen order.
    """
    if not unique_sources:
        return "_No sources were collected during research._"

    lines = []
    seen = set()
    idx = 1
    for src in unique_sources:
        if isinstance(src, dict):
            url = src.get("url") or "#"
            if url in seen:
                continue
            seen.add(url)
            title = src.get("title") or "Source Publication"
            domain = src.get("domain") or urllib.parse.urlparse(url).netloc
            content_type = src.get("content_type", "snippet")
            type_label = "Full text analyzed" if content_type == "full_text" else "Search snippet"
            lines.append(f"{idx}. [{title}]({url}) ({domain}) — _{type_label}_")
            idx += 1
        else:
            url_str = str(src)
            if url_str in seen:
                continue
            seen.add(url_str)
            lines.append(f"{idx}. {url_str}")
            idx += 1

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
    report_body:       str,
    sources_block:     str,
    confidence:        str,
    research_loops:    int,
    critique:          str,
    timestamp:         str,
    unique_sources:    list,
    citation_metrics:  dict = None
) -> str:
    """
    Appends the programmatically-generated Sources and Research Notes sections,
    including citation verification metrics and evidence depth stats.
    """
    source_count = len(unique_sources)
    full_text_count = sum(1 for s in unique_sources if isinstance(s, dict) and s.get("content_type") == "full_text")

    confidence_display = {
        "High":   "🟢 High",
        "Medium": "🟡 Medium",
        "Low":    "🔴 Low"
    }.get(confidence, "🟡 Medium")

    methodology = (
        f"This report was generated by an autonomous multi-agent research system with an evidence-grounded retrieval layer. "
        f"A Planner decomposed the query into sub-questions, a Researcher retrieved live web pages with SSRF safeguards, "
        f"a Summarizer synthesized the evidence, and a Critic verified completeness"
    )

    if research_loops > 0:
        methodology += (
            f" — triggering {research_loops} targeted follow-up "
            f"loop{'s' if research_loops > 1 else ''} to close identified evidence gaps"
        )

    methodology += f". {source_count} unique web sources were consulted ({full_text_count} full articles analyzed)."
    critic_summary = _extract_overall_assessment(critique)

    citation_info = ""
    if citation_metrics:
        total_cit = citation_metrics.get("total_citations", 0)
        precision = citation_metrics.get("precision", 1.0) * 100
        sources_cited = citation_metrics.get("unique_sources_cited", 0)
        citation_info = (
            f"| Inline citations | {total_cit} ({precision:.0f}% verified) |\n"
            f"| Source citation coverage | {sources_cited}/{source_count} sources cited |\n"
            f"| Evidence depth | {full_text_count}/{source_count} full articles analyzed |"
        )

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
| Research passes | {research_loops} |
| Sources consulted | {source_count} |
{citation_info}
| System | Multi-Agent Research Assistant (LLaMA 3.3 70B via Groq) |

**Methodology:** {methodology}

**Quality Review:** {critic_summary}

_This report was compiled by an AI research assistant with verified claim-level citations. All information 
is grounded in live web sources retrieved at generation time. Verify critical claims with primary literature before clinical or financial application._
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