# test_writer.py
from graph.state import ResearchState
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.summarizer import summarizer_agent
from agents.critic import critic_agent
from agents.writer import writer_agent


def merge_state(base: dict, update: dict, additive_keys: list) -> dict:
    """Simulates LangGraph's state merging logic."""
    merged = {**base}
    for key, value in update.items():
        if key in additive_keys and isinstance(value, list):
            merged[key] = base.get(key, []) + value
        else:
            merged[key] = value
    return merged


ADDITIVE_KEYS = ["research_data", "sources", "status_log"]

# --- Initial state ---
initial_state: ResearchState = {
    "query": "What is the current state of large language model research up until April 2026?",
    "sub_questions": [],
    "research_data": [],
    "sources": [],
    "summaries": [],
    "critique": "",
    "needs_more_research": False,
    "research_loops": 0,
    "final_report": "",
    "status_log": []
}

print("=" * 60)
print("TESTING FULL 5-AGENT PIPELINE")
print("=" * 60)

# --- Run all 5 agents ---
print("\n[TEST] Step 1: Planner...")
result = planner_agent(initial_state)
state  = merge_state(initial_state, result, ADDITIVE_KEYS)
print(f"  Sub-questions: {len(state['sub_questions'])}")

print("\n[TEST] Step 2: Researcher...")
result = researcher_agent(state)
state  = merge_state(state, result, ADDITIVE_KEYS)
print(f"  Research entries: {len(state['research_data'])}")
print(f"  Sources: {len(state['sources'])}")

print("\n[TEST] Step 3: Summarizer...")
result = summarizer_agent(state)
state  = merge_state(state, result, ADDITIVE_KEYS)
print(f"  Summaries: {len(state['summaries'])}")

print("\n[TEST] Step 4: Critic...")
result = critic_agent(state)
state  = merge_state(state, result, ADDITIVE_KEYS)
print(f"  Verdict: {'NEEDS_MORE_RESEARCH' if state['needs_more_research'] else 'APPROVED'}")
print(f"  Loops: {state['research_loops']}")

# --- If Critic sent back to Researcher, run one more loop ---
# In the real graph this is handled by conditional edges.
# In this test we simulate it manually.
if state["needs_more_research"]:
    print("\n[TEST] Critic triggered loop — running Researcher again...")
    result = researcher_agent(state)
    state  = merge_state(state, result, ADDITIVE_KEYS)

    print("\n[TEST] Re-running Summarizer with new research...")
    result = summarizer_agent(state)
    state  = merge_state(state, result, ADDITIVE_KEYS)

    print("\n[TEST] Critic reviewing again...")
    result = critic_agent(state)
    state  = merge_state(state, result, ADDITIVE_KEYS)
    print(f"  Final verdict: {'NEEDS_MORE_RESEARCH' if state['needs_more_research'] else 'APPROVED'}")

print("\n[TEST] Step 5: Writer...")
result = writer_agent(state)
state  = merge_state(state, result, ADDITIVE_KEYS)

# --- Inspect the final report ---
print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(state["final_report"])

# --- Status log ---
print("\n" + "=" * 60)
print("COMPLETE STATUS LOG")
print("=" * 60)
for entry in state["status_log"]:
    print(f"  • {entry}")

# --- Save the report to a file ---
report_filename = "sample_report.md"
with open(report_filename, "w", encoding="utf-8") as f:
    f.write(state["final_report"])
print(f"\n[TEST] Report saved to: {report_filename}")

# --- Validate ---
print("\n--- VALIDATION ---")
report = state["final_report"]
assert isinstance(report, str),              "final_report must be a string"
assert len(report) > 500,                    "report must be substantial (>500 chars)"
assert "## Sources"          in report,      "report must have Sources section"
assert "## Research Notes"   in report,      "report must have Research Notes section"
assert "## Executive Summary" in report,     "report must have Executive Summary"
assert "## Conclusion"       in report,      "report must have Conclusion"
assert len(state["status_log"]) >= 5,        "all 5 agents must have logged status"
print("All validations passed!")
print(f"\nFinal report: {len(report)} chars, "
      f"{report.count(chr(10))} lines")