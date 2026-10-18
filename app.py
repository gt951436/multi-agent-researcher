# app.py
import streamlit as st
from datetime import datetime
from config.settings import SETTINGS
from graph.workflow import build_graph
from graph.state import ResearchState

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Session state initialization
# Streamlit re-runs app.py on every interaction.
# We use session_state to persist data across re-runs
# and prevent duplicate graph executions.
# ─────────────────────────────────────────────
if "is_running"    not in st.session_state:
    st.session_state.is_running    = False
if "final_state"   not in st.session_state:
    st.session_state.final_state   = None
if "last_query"    not in st.session_state:
    st.session_state.last_query    = ""
if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.markdown("---")

    st.markdown("### How it works")
    st.markdown("""
5 specialized agents collaborate in a self-correcting pipeline:

1. 🗂️ **Planner** — Decomposes query into sub-questions
2. 🔍 **Researcher** — Live web search per sub-question
3. 📝 **Summarizer** — Synthesizes raw findings
4. ✅ **Critic** — Strict quality review (may loop back)
5. ✍️ **Writer** — Compiles final structured report
    """)

    st.markdown("---")
    st.markdown("### Critic rubric")
    st.markdown("""
The Critic evaluates each summary on:
- **Specificity** — concrete facts, not vague claims
- **Completeness** — sub-question fully answered
- **Recency** — current information
- **Depth** — meaningful analysis

Fails → triggers re-research loop (max 2×)
    """)

    st.markdown("---")
    st.markdown("### Tech stack")
    st.code(
        "LLM:    LLaMA 3.3 70B (Groq)\n"
        "Graph:  LangGraph\n"
        "Search: DuckDuckGo\n"
        "UI:     Streamlit",
        language="text"
    )

    st.markdown("---")
    st.markdown("### Example queries")
    examples = [
        "What is the current state of quantum computing?",
        "How does mRNA vaccine technology work?",
        "What are the latest developments in fusion energy?",
        "How are autonomous vehicles regulated globally in 2025?",
    ]
    for q in examples:
        if st.button(q, use_container_width=True, key=f"ex_{q}"):
            st.session_state.prefill_query = q
            st.rerun()

# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────
st.title("🔬 Multi-Agent Research Assistant")
st.markdown(
    "Type any research topic. Five AI agents will autonomously "
    "research, critique, and compile a structured report with live sources."
)

# ─────────────────────────────────────────────
# Query input
# ─────────────────────────────────────────────
query = st.text_input(
    label="Research query",
    placeholder="e.g. What is the current state of nuclear fusion research?",
    value=st.session_state.prefill_query,
    key="query_input",
    disabled=st.session_state.is_running
)

# Clear prefill after it's been used
if st.session_state.prefill_query and query == st.session_state.prefill_query:
    st.session_state.prefill_query = ""

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    run_button = st.button(
        "🚀 Research",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_running or not query.strip()
    )
with col2:
    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True,
        disabled=st.session_state.is_running
    )

if clear_button:
    st.session_state.final_state = None
    st.session_state.last_query  = ""
    st.rerun()

# ─────────────────────────────────────────────
# Research execution
# Only runs when button clicked AND not already running
# session_state.is_running prevents Streamlit re-runs
# from triggering the graph a second time
# ─────────────────────────────────────────────
if run_button and query.strip() and not st.session_state.is_running:

    st.session_state.is_running  = True
    st.session_state.final_state = None
    st.session_state.last_query  = query.strip()

    st.markdown("---")
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### 🤖 Agent Activity")

    with right_col:
        st.markdown("### 📄 Research Report")
        report_ph = st.empty()
        report_ph.info("⏳ Report will appear here when complete...")

    agent_display = {
        "planner":    ("🗂️", "Planner",    "Decomposing query..."),
        "researcher": ("🔍", "Researcher", "Searching the web..."),
        "summarizer": ("📝", "Summarizer", "Synthesizing findings..."),
        "critic":     ("✅", "Critic",     "Evaluating quality..."),
        "writer":     ("✍️", "Writer",     "Writing report..."),
    }

    completed_agents = []
    start_time       = datetime.now()

    graph = build_graph()

    initial_state: ResearchState = {
        "query":               query.strip(),
        "sub_questions":       [],
        "research_data":       [],
        "sources":             [],
        "source_registry":     [],
        "summaries":           [],
        "critique":            "",
        "critic_review":       None,
        "follow_up_queries":   [],
        "summary_scores":      [],
        "needs_more_research": False,
        "research_loops":      0,
        "final_report":        "",
        "citation_metrics":    None,
        "status_log":          []
    }

    accumulated_state = {k: v for k, v in initial_state.items()}

    with left_col:
        activity_box = st.empty()

    try:
        for step in graph.stream(initial_state, {"recursion_limit": 25}):
            node_name   = list(step.keys())[0]
            node_update = step[node_name]

            # Merge update into accumulated state
            for key, val in node_update.items():
                if key in ["research_data", "sources", "source_registry", "status_log"] \
                        and isinstance(val, list):
                    accumulated_state[key] = accumulated_state.get(key, []) + val
                else:
                    accumulated_state[key] = val


            completed_agents.append(node_name)

            # --- Rebuild activity log ---
            lines = []
            seen  = set()

            for agent in completed_agents:
                if agent in seen:
                    continue
                seen.add(agent)

                emoji, label, _ = agent_display.get(
                    agent, ("⚙️", agent.title(), "")
                )
                run_count  = completed_agents.count(agent)
                loop_badge = f" `×{run_count}`" if run_count > 1 else ""

                # Get last status log entry for this agent
                all_logs   = accumulated_state.get("status_log", [])
                agent_logs = [l for l in all_logs
                              if label.lower() in l.lower()]
                detail     = f"_{agent_logs[-1]}_" if agent_logs else ""

                lines.append(f"✅ **{emoji} {label}**{loop_badge}  \n{detail}")

            # Show what's running next
            next_map = {
                "planner":    "researcher",
                "researcher": "summarizer",
                "summarizer": "critic",
                "critic": (
                    "researcher"
                    if accumulated_state.get("needs_more_research")
                    else "writer"
                ),
                "writer": None
            }
            nxt = next_map.get(node_name)
            if nxt and nxt in agent_display:
                e, l, msg = agent_display[nxt]
                lines.append(f"\n⏳ **{e} {l}**  \n_{msg}_")

            # Critic verdict badge
            verdict = accumulated_state.get("needs_more_research", False)
            loops   = accumulated_state.get("research_loops", 0)
            if "critic" in completed_agents:
                badge = "🔴 Needs more research" if verdict else "🟢 Approved"
                lines.append(f"\n**Critic verdict:** {badge}  \n"
                             f"**Loops:** {loops}/{SETTINGS.MAX_RESEARCH_LOOPS}")

            elapsed = (datetime.now() - start_time).seconds
            lines.append(f"\n⏱️ _Elapsed: {elapsed}s_")

            activity_box.markdown("\n\n".join(lines))

            # Show report when Writer is done
            if node_name == "writer":
                report = accumulated_state.get("final_report", "")
                if report:
                    report_ph.empty()
                    with right_col:
                        dl_name = (
                            f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                        )
                        st.download_button(
                            "⬇️ Download Report (.md)",
                            data=report,
                            file_name=dl_name,
                            mime="text/markdown",
                            use_container_width=True
                        )
                        st.markdown(report)

        # Store result
        st.session_state.final_state = accumulated_state

    except Exception as e:
        st.error(f"⚠️ Research pipeline encountered an error: {e}")
    finally:
        # Always release the running lock — even if an error occurred
        st.session_state.is_running = False

    # --- Metrics bar ---
    elapsed_total = (datetime.now() - start_time).seconds
    unique_sources = list(dict.fromkeys(accumulated_state.get("sources", [])))
    source_count  = len(unique_sources)
    loops         = accumulated_state.get("research_loops", 0)
    report_len    = len(accumulated_state.get("final_report", ""))

    reg = accumulated_state.get("source_registry", [])
    full_text_count = sum(1 for s in reg if isinstance(s, dict) and s.get("content_type") == "full_text")
    cit_metrics = accumulated_state.get("citation_metrics")

    raw_conf = accumulated_state.get("confidence")
    if not raw_conf:
        if "| Confidence | 🟢 High" in accumulated_state.get("final_report", ""):
            raw_conf = "High"
        elif "| Confidence | 🔴 Low" in accumulated_state.get("final_report", ""):
            raw_conf = "Low"
        else:
            raw_conf = "Medium"

    confidence_display = {
        "High":   "🟢 High",
        "Medium": "🟡 Medium",
        "Low":    "🔴 Low"
    }.get(raw_conf, "🟡 Medium")

    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Time",        f"{elapsed_total}s")
    sources_label = f"{source_count}" if full_text_count == 0 else f"{source_count} ({full_text_count} full)"
    m2.metric("Sources",     sources_label)
    m3.metric("Loops",       f"{loops}/{SETTINGS.MAX_RESEARCH_LOOPS}")
    if cit_metrics and cit_metrics.get("total_citations", 0) > 0:
        prec = cit_metrics.get("precision", 1.0) * 100
        m4.metric("Citations", f"{cit_metrics['valid_citations']} ({prec:.0f}% verified)")
    else:
        m4.metric("Report",  f"{report_len} chars")
    m5.metric("Confidence",  confidence_display)


# Show previous result if no new run triggered
elif st.session_state.final_state and not st.session_state.is_running:

    st.markdown("---")
    st.info(
        f"Showing report for: **{st.session_state.last_query}**  \n"
        f"Run a new query above to generate a fresh report."
    )
    st.markdown(st.session_state.final_state.get("final_report", ""))