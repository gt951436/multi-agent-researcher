# streamlit UI

import streamlit as st
from datetime import datetime
from graph.workflow import build_graph
from graph.state import ResearchState

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Sidebar — About & Settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.markdown("---")

    st.markdown("### How it works")
    st.markdown("""
This system uses **5 AI agents** working in a coordinated pipeline:

1. 🗂️ **Planner** — Breaks your query into sub-questions
2. 🔍 **Researcher** — Searches the web for each sub-question
3. 📝 **Summarizer** — Synthesizes raw findings
4. ✅ **Critic** — Reviews quality, may trigger re-research
5. ✍️ **Writer** — Compiles the final report
    """)

    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown("""
- **LLM:** LLaMA 3.3 70B via Groq (free)
- **Orchestration:** LangGraph
- **Search:** DuckDuckGo (free, no key)
- **UI:** Streamlit
    """)

    st.markdown("---")
    st.markdown("### Example queries")
    example_queries = [
        "What is the current state of quantum computing?",
        "How does CRISPR gene editing work and what are its applications?",
        "What are the latest developments in renewable energy storage?",
        "How are autonomous vehicles being tested and regulated in 2025?",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill_query"] = q
            st.rerun()

# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────
st.title("🔬 Multi-Agent Research Assistant")
st.markdown(
    "Enter any research topic. Five specialized AI agents will "
    "autonomously research, evaluate, and compile a structured report."
)

# ─────────────────────────────────────────────
# Query input
# ─────────────────────────────────────────────
prefill = st.session_state.get("prefill_query", "")

query = st.text_input(
    label="Research query",
    placeholder="e.g. What is the current state of large language model research?",
    value=prefill,
    key="query_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    run_button = st.button(
        "🚀 Start Research",
        type="primary",
        use_container_width=True,
        disabled=not query.strip()
    )

# ─────────────────────────────────────────────
# Main research execution
# ─────────────────────────────────────────────
if run_button and query.strip():

    # Clear any prefill after use
    if "prefill_query" in st.session_state:
        del st.session_state["prefill_query"]

    st.markdown("---")

    # --- Layout: two columns ---
    # Left: live agent activity log
    # Right: final report (appears when done)
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### 🤖 Agent Activity")
        activity_container = st.container()

    with right_col:
        st.markdown("### 📄 Research Report")
        report_placeholder = st.empty()
        report_placeholder.info(
            "⏳ Report will appear here when all agents have finished..."
        )

    # --- Agent status tracker ---
    # Maps agent names to display emoji + label
    agent_display = {
        "planner":    ("🗂️", "Planner",    "Breaking query into sub-questions..."),
        "researcher": ("🔍", "Researcher", "Searching the web..."),
        "summarizer": ("📝", "Summarizer", "Synthesizing findings..."),
        "critic":     ("✅", "Critic",     "Reviewing research quality..."),
        "writer":     ("✍️", "Writer",     "Compiling final report..."),
    }

    # Track which agents have completed (for display)
    completed_agents = []
    loop_count       = 0
    start_time       = datetime.now()

    # --- Build the graph ---
    graph = build_graph()

    # Build initial state
    initial_state: ResearchState = {
        "query":               query.strip(),
        "sub_questions":       [],
        "research_data":       [],
        "sources":             [],
        "summaries":           [],
        "critique":            "",
        "needs_more_research": False,
        "research_loops":      0,
        "final_report":        "",
        "status_log":          []
    }

    # --- Stream the graph ---
    # graph.stream() yields a dict after EACH node completes.
    # The dict key is the node name, value is that node's state update.
    # This lets us update the UI in real time.
    final_state = initial_state.copy()

    with activity_container:
        status_box = st.empty()

    try:
        for step_output in graph.stream(initial_state, {"recursion_limit": 25}):
            # step_output = {"node_name": {state_updates_dict}}
            node_name   = list(step_output.keys())[0]
            node_update = step_output[node_name]

            # Merge into final_state to track accumulated state
            for key, value in node_update.items():
                if key in ["research_data", "sources", "status_log"] \
                        and isinstance(value, list):
                    existing = final_state.get(key, [])
                    final_state[key] = existing + value
                else:
                    final_state[key] = value

            # Track loop count
            if node_name == "researcher":
                loop_count = final_state.get("research_loops", 0)

            # Record completion
            completed_agents.append(node_name)

            # --- Rebuild activity log display ---
            activity_lines = []

            # Show completed agents
            seen = set()
            for agent in completed_agents:
                if agent in seen:
                    continue
                seen.add(agent)

                emoji, label, _ = agent_display.get(
                    agent, ("⚙️", agent.title(), "Working...")
                )

                # Count how many times this agent ran (for loop display)
                run_count = completed_agents.count(agent)
                loop_note = f" _(×{run_count})_" if run_count > 1 else ""

                # Pull the last status log entry for this agent
                all_logs   = final_state.get("status_log", [])
                agent_logs = [l for l in all_logs
                              if label.lower() in l.lower()]
                detail     = agent_logs[-1] if agent_logs else ""

                activity_lines.append(
                    f"✅ **{emoji} {label}**{loop_note}  \n"
                    f"_{detail}_"
                )

            # Show currently running agent (next one)
            next_agents = {
                "planner":    "researcher",
                "researcher": "summarizer",
                "summarizer": "critic",
                "critic":     "researcher" if final_state.get("needs_more_research") else "writer",
                "writer":     None
            }
            current_next = next_agents.get(node_name)
            if current_next and current_next in agent_display:
                e, l, msg = agent_display[current_next]
                activity_lines.append(f"⏳ **{e} {l}**  \n_{msg}_")

            # Elapsed time
            elapsed = (datetime.now() - start_time).seconds
            activity_lines.append(f"\n⏱️ _Elapsed: {elapsed}s_")

            # Update the activity display
            status_box.markdown("\n\n".join(activity_lines))

            # --- Show report when Writer completes ---
            if node_name == "writer":
                report = final_state.get("final_report", "")
                if report:
                    report_placeholder.empty()
                    with right_col:
                        # Download button
                        st.download_button(
                            label="⬇️ Download Report (.md)",
                            data=report,
                            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                        # Render the markdown report
                        st.markdown(report)

    except Exception as e:
        st.error(f"An error occurred during research: {e}")
        st.exception(e)

    # --- Final summary bar ---
    elapsed_total = (datetime.now() - start_time).seconds
    sources_count = len(set(final_state.get("sources", [])))
    loops_ran     = final_state.get("research_loops", 0)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total time",       f"{elapsed_total}s")
    m2.metric("Sources consulted", sources_count)
    m3.metric("Research loops",    loops_ran)
    m4.metric("Report length",
              f"{len(final_state.get('final_report', ''))} chars")