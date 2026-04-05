# test_planner.py
from graph.state import ResearchState
from agents.planner import planner_agent

# --- Simulate what LangGraph will pass to this agent ---
# We manually create a state dict with only the fields
# the Planner needs. The rest can be empty/default.
test_state: ResearchState = {
    "query": "What is the current state of artificial general intelligence research?",
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
print("TESTING PLANNER AGENT")
print("=" * 60)
print(f"Input query: {test_state['query']}\n")

# run the agent
result = planner_agent(test_state)

# inspect the output
print("\n--- PLANNER OUTPUT ---")
print(f"Number of sub-questions: {len(result['sub_questions'])}")
print("\nSub-questions:")
for i, q in enumerate(result['sub_questions'], 1):
    print(f"  {i}. {q}")

print("\nStatus log entry:")
print(f"  {result['status_log'][0]}")

# validate the output structure
print("\n--- VALIDATION ---")
assert isinstance(result["sub_questions"], list), "sub_questions must be a list"
assert 3 <= len(result["sub_questions"]) <= 5, f"Expected 3-5 questions, got {len(result['sub_questions'])}"
assert isinstance(result["status_log"], list), "status_log must be a list"
assert len(result["status_log"]) == 1, "should add exactly 1 status log entry"
print("All validations passed!")