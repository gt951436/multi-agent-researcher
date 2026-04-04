# test_researcher.py
from graph.state import ResearchState
from agents.planner import planner_agent
from agents.researcher import researcher_agent

# --- Step 1: Build initial state ---
initial_state: ResearchState = {
    "query": "What is the current state of large language model research up until april 2026?",
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
print("TESTING PLANNER → RESEARCHER PIPELINE")
print("=" * 60)

# --- Step 2: Run Planner ---
print("\n[TEST] Running Planner agent...")
planner_result = planner_agent(initial_state)

# Merge planner output into state (simulating what LangGraph does)
state_after_planner = {**initial_state, **planner_result}
# For operator.add fields, manually merge lists
state_after_planner["status_log"] = (
    initial_state["status_log"] + planner_result["status_log"]
)

print(f"\n[TEST] Planner produced {len(state_after_planner['sub_questions'])} sub-questions")

# --- Step 3: Run Researcher ---
print("\n[TEST] Running Researcher agent (takes some time due to web searches)...")
researcher_result = researcher_agent(state_after_planner)

# --- Step 4: Inspect Results ---
print("\n" + "=" * 60)
print("RESEARCHER OUTPUT INSPECTION")
print("=" * 60)

research_data = researcher_result["research_data"]
sources       = researcher_result["sources"]

print(f"\nTotal sub-questions researched: {len(research_data)}")
print(f"Total sources collected:        {len(sources)}")

print("\n--- SAMPLE: First research entry ---")
if research_data:
    first = research_data[0]
    print(f"Question: {first['question']}")
    print(f"Results found: {first['result_count']}")
    print(f"Key facts (first 400 chars):\n{first['key_facts'][:400]}...")

print("\n--- SOURCES (first 5) ---")
for i, url in enumerate(sources[:5], 1):
    print(f"  {i}. {url}")

# --- Step 5: Validate structure ---
print("\n--- VALIDATION ---")
assert isinstance(research_data, list),         "research_data must be a list"
assert len(research_data) > 0,                  "must have at least 1 research entry"
assert "question"  in research_data[0],         "each entry needs 'question'"
assert "key_facts" in research_data[0],         "each entry needs 'key_facts'"
assert isinstance(sources, list),               "sources must be a list"
assert isinstance(researcher_result["status_log"], list), "status_log must be a list"
print("All validations passed!")