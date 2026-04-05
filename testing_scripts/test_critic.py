# Planner → Researcher → Summarizer → Critic 

from graph.state import ResearchState
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.summarizer import summarizer_agent
from agents.critic import critic_agent


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
print("TESTING FULL 4-AGENT PIPELINE")
print("=" * 60)

# --- Run all agents in sequence ---
print("\n[TEST] Step 1: Planner...")
planner_result = planner_agent(initial_state)
state = merge_state(initial_state, planner_result, ADDITIVE_KEYS)
print(f"  Sub-questions: {len(state['sub_questions'])}")

print("\n[TEST] Step 2: Researcher (~30 seconds)...")
researcher_result = researcher_agent(state)
state = merge_state(state, researcher_result, ADDITIVE_KEYS)
print(f"  Research entries: {len(state['research_data'])}")

print("\n[TEST] Step 3: Summarizer...")
summarizer_result = summarizer_agent(state)
state = merge_state(state, summarizer_result, ADDITIVE_KEYS)
print(f"  Summaries: {len(state['summaries'])}")

print("\n[TEST] Step 4: Critic...")
critic_result = critic_agent(state)
state = merge_state(state, critic_result, ADDITIVE_KEYS)

# --- Inspect Critic output ---
print("\n" + "=" * 60)
print("CRITIC OUTPUT INSPECTION")
print("=" * 60)

print(f"\nVerdict — needs more research: {state['needs_more_research']}")
print(f"Research loops counter:        {state['research_loops']}")

print("\n--- FULL CRITIQUE ---")
print(state["critique"])

print("\n--- STATUS LOG (all 4 agents) ---")
for entry in state["status_log"]:
    print(f"  • {entry}")

# --- Validate structure ---
print("\n--- VALIDATION ---")
assert isinstance(state["critique"], str),           "critique must be a string"
assert len(state["critique"]) > 0,                   "critique must not be empty"
assert isinstance(state["needs_more_research"], bool),"needs_more_research must be bool"
assert isinstance(state["research_loops"], int),      "research_loops must be int"
assert isinstance(critic_result["status_log"], list), "status_log must be a list"

# Verdict string must appear in critique
upper = state["critique"].upper()
assert "APPROVED" in upper or "NEEDS" in upper, \
    "Critique must contain a clear APPROVED or NEEDS verdict"

print("All validations passed!")
print(f"\nRouting decision: → {'RESEARCHER (loop back)' if state['needs_more_research'] else 'WRITER (proceed)'}")