from tools.search_tool import web_search, format_results_for_llm

print("Testing DuckDuckGo search tool...\n")
results = web_search("Latest breakthroughs in artificial general intelligence up until April 2026", max_results=3)

print(f"Got {len(results)} results\n")
print("--- FORMATTED FOR LLM ---")
print(format_results_for_llm(results))