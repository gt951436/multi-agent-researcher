import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.workflow import build_graph, run_research

print("Testing graph compilation...")
graph = build_graph()
print("Graph compiled successfully!")

# Optional: visualize the graph structure
try:
    # This works if you have graphviz installed
    png = graph.get_graph().draw_mermaid()
    print("\nGraph structure (Mermaid):")
    print(png)
except Exception:
    pass

print("\nRunning full pipeline via graph.invoke()...")

result = run_research(
    "What are the key differences between supervised and "
    "unsupervised machine learning?"
)

print("\n--- GRAPH RUN COMPLETE ---")
print(f"Final report length: {len(result['final_report'])} chars")
print(f"Research loops ran:  {result['research_loops']}")
print(f"Sources collected:   {len(result['sources'])}")
print("\nStatus log:")
for entry in result["status_log"]:
    print(f"  • {entry}")

print("\nFirst 500 chars of report:")
print(result["final_report"][:500])