from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm
from graph.state import ResearchState

def planner_agent(state: ResearchState) -> dict:
    """
    The Planner Agent — first node in the LangGraph pipeline.

    Responsibility:
        Takes the raw user query from state and decomposes it into
        3-5 focused sub-questions that guide the Researcher agent.

    Why this design:
        - Receives the full state but only READS 'query'
        - Only WRITES 'sub_questions' and 'status_log'
        - Returns a dict (not the full state) — LangGraph merges it

    Args:
        state: The shared ResearchState flowing through the graph

    Returns:
        dict with keys: sub_questions, status_log
    """
    print("\n[Planner] Starting research planning...")
    query = state["query"]
    
    # --- System Prompt ---
    # This defines the agent's ROLE and RULES.
    # We tell it exactly what format to output so parsing is reliable.
    system_prompt = SystemMessage(content="""You are an expert research strategist and planner.

Your job is to decompose a broad research query into focused, specific sub-questions.

Rules:
- Generate exactly 3 to 5 sub-questions
- Each sub-question must be specific and independently searchable on the web
- Sub-questions must be NON-overlapping — each should cover a distinct angle
- Sub-questions together must provide COMPLETE coverage of the original query
- Write each sub-question on its own line, starting with a number and period (e.g. "1. ")
- Do NOT include any explanation, preamble, or commentary — only the numbered list

Example output format:
1. What are the latest hardware breakthroughs in quantum computing as of 2025?
2. Which companies are leading quantum computing research and development?
3. What are the main technical obstacles to practical quantum computers?
4. What real-world problems can quantum computers solve that classical ones cannot?
5. How close are we to achieving quantum advantage in commercial applications?
""")
    
    # --- Human Prompt ---
    # This provides the actual task for this specific run.
    # We inject the user's query dynamically here.
    human_prompt = HumanMessage(content=f"""Research query: {query}
                                
Generate 3-5 focused sub-questions to guide comprehensive research on this topic.""")
    
    # --- Call the LLM ---
    # get_llm(0.1) = low temperature for structured, deterministic output
    # We want consistent formatting, not creative variation
    llm = get_llm(temperature=0.1)
    response = llm.invoke([system_prompt, human_prompt])
    
    # --- Parse the LLM response ---
    # The LLM returns a string like:
    # "1. What is...?\n2. How does...?\n3. Why are...?"
    # We split by newlines and extract each numbered line.
    raw_output = response.content.strip()
    parsed_questions = _parse_sub_questions(raw_output)

    # Assign stable question IDs (Q1, Q2, ...) so all downstream agents
    # (Researcher, Summarizer, Critic, Writer) reference questions by ID
    # rather than relying on brittle list positional indices.
    sub_questions = [
        {"id": f"Q{i}", "question": q}
        for i, q in enumerate(parsed_questions, 1)
    ]

    print(f"[PLANNER] Generated {len(sub_questions)} structured sub-questions:")
    for item in sub_questions:
        print(f"  [{item['id']}] {item['question']}")

    # Return structured sub-questions and status log
    return {
        "sub_questions": sub_questions,
        "status_log": [f"Planner: Generated {len(sub_questions)} sub-questions for query: '{query}'"]
    }


def _parse_sub_questions(raw_text: str) -> list[str]:
    """
    Parses the LLM's numbered list output into a clean Python list.

    Why a separate function:
        - Keeps the main agent function clean and readable
        - Easier to test and debug parsing logic in isolation
        - Easy to improve the parser without touching agent logic

    Handles edge cases:
        - Lines that don't start with a number are skipped
        - Strips "1. ", "1) ", "1 - " style prefixes
        - Removes empty lines
    """
    lines = raw_text.split("\n")
    questions = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Only process lines that start with a digit (our numbered questions)
        if line[0].isdigit():
            # Remove the numbering prefix: "1. ", "2. ", "1) ", etc.
            # Find the first space after the number/punctuation
            for i, char in enumerate(line):
                if char == " " and i > 0:
                    question = line[i:].strip()
                    if question:  # make sure we didn't strip to empty
                        questions.append(question)
                    break

    # Safety net: if parsing failed and we got nothing,
    # fall back to treating each non-empty line as a question
    if not questions:
        questions = [l.strip() for l in lines if l.strip()]

    return questions