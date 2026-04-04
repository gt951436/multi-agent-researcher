from graph.state import ResearchState
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.summarizer import summarizer_agent


def merge_state(base: dict, update: dict, additive_keys: list) -> dict:
    """
    Simulates what LangGraph does internally:
    - For additive keys (operator.add fields): MERGE the lists
    - For all other keys: OVERWRITE with new value

    We use this in tests to manually replicate LangGraph's
    state merging logic without actually running the graph.
    """
    merged = {**base}
    for key, value in update.items():
        if key in additive_keys and isinstance(value, list):
            merged[key] = base.get(key, []) + value
        else:
            merged[key] = value
    return merged


# Keys that use operator.add in ResearchState (lists that accumulate)
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
print("TESTING PLANNER → RESEARCHER → SUMMARIZER PIPELINE")
print("=" * 60)

# --- Run Planner ---
print("\n[TEST] Step 1: Running Planner...")
planner_result   = planner_agent(initial_state)
state_after_planner = merge_state(initial_state, planner_result, ADDITIVE_KEYS)
print(f"  Sub-questions: {len(state_after_planner['sub_questions'])}")

# --- Run Researcher ---
print("\n[TEST] Step 2: Running Researcher (~30 seconds)...")
researcher_result      = researcher_agent(state_after_planner)
state_after_researcher = merge_state(state_after_planner, researcher_result, ADDITIVE_KEYS)
print(f"  Research entries: {len(state_after_researcher['research_data'])}")
print(f"  Sources: {len(state_after_researcher['sources'])}")

# --- Run Summarizer ---
print("\n[TEST] Step 3: Running Summarizer...")
summarizer_result      = summarizer_agent(state_after_researcher)
state_after_summarizer = merge_state(state_after_researcher, summarizer_result, ADDITIVE_KEYS)

# --- Inspect output ---
print("\n" + "=" * 60)
print("SUMMARIZER OUTPUT INSPECTION")
print("=" * 60)

summaries = state_after_summarizer["summaries"]
print(f"\nTotal summaries produced: {len(summaries)}")

gaps = [s for s in summaries if s.get("has_gap", False)]
print(f"Research gaps flagged:    {len(gaps)}")

print("\n--- FULL SUMMARY: Entry 1 ---")
if summaries:
    first = summaries[0]
    print(f"Question:\n  {first['question']}\n")
    print(f"Summary:\n  {first['summary']}\n")
    print(f"Has gap: {first['has_gap']}")

print("\n--- STATUS LOG (all agents so far) ---")
for entry in state_after_summarizer["status_log"]:
    print(f"  • {entry}")

# --- Validate ---
print("\n--- VALIDATION ---")
assert isinstance(summaries, list),              "summaries must be a list"
assert len(summaries) > 0,                       "must have at least 1 summary"
assert "question" in summaries[0],               "each summary needs 'question'"
assert "summary"  in summaries[0],               "each summary needs 'summary'"
assert "has_gap"  in summaries[0],               "each summary needs 'has_gap'"
assert len(summaries) == len(
    state_after_researcher["research_data"]),     "one summary per research entry"
assert isinstance(
    summarizer_result["status_log"], list),       "status_log must be a list"
print("All validations passed!")