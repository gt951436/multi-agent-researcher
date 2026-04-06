# Multi-Agent Research Assistant

> An autonomous AI research system built with **LangGraph** + **LLaMA 3.3 70B** that decomposes any research query into targeted sub-questions, searches the live web, self-evaluates research quality, and compiles a structured report — entirely free to run.

<br>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=flat-square)
![LLaMA](https://img.shields.io/badge/LLaMA_3.3_70B-Groq-F55036?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![Cost](https://img.shields.io/badge/API_Cost-Free-22C55E?style=flat-square)

---

## What It Does

Given a query like *"What is the current state of quantum computing?"*, the system:

1. **Decomposes** it into 3–5 focused, non-overlapping sub-questions
2. **Searches** the live web (DuckDuckGo) for each sub-question independently
3. **Synthesizes** raw search results into coherent prose summaries
4. **Evaluates** research quality against a strict rubric — loops back if gaps are found
5. **Compiles** a polished markdown report with cited sources and a confidence rating

**Output:** A 6,000–10,000 character structured research report with 20–50 live web sources, generated in ~60–90 seconds, at zero API cost.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│                  LangGraph StateGraph            │
│                                                  │
│  ┌──────────┐                                    │
│  │ Planner  │  Decomposes query → 3–5            │
│  │  Agent   │  focused sub-questions             │
│  └────┬─────┘                                    │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐  DuckDuckGo search per             │
│  │Researcher│  sub-question → LLM fact           │
│  │  Agent   │  extraction                        │
│  └────┬─────┘                                    │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐  Raw facts → coherent              │
│  │Summarizer│  prose summaries                   │
│  │  Agent   │  (one per sub-question)            │
│  └────┬─────┘                                    │
│       │                                          │
│       ▼                                          │
│  ┌──────────┐   gaps found (max 2×)              │
│  │  Critic  │ ──────────────────────► Researcher │
│  │  Agent   │                                    │
│  └────┬─────┘                                    │
│       │  approved                                │
│       ▼                                          │
│  ┌──────────┐  Summaries + sources →             │
│  │  Writer  │  structured markdown report        │
│  │  Agent   │  with confidence rating            │
│  └────┬─────┘                                    │
│       │                                          │
│       ▼                                          │
│   Final Report                                   │
└─────────────────────────────────────────────────┘
```

### LangGraph graph topology (Mermaid)

```
__start__ → planner → researcher → summarizer → critic -.→ researcher
                                                  critic -.→ writer → __end__
```

Dotted edges = conditional routing driven by the Critic's verdict.

---

## Key Design Patterns

| Pattern | Implementation | Why It Matters |
|---|---|---|
| **Shared typed state** | `ResearchState` TypedDict flows through all nodes | Agents communicate through state, never by calling each other directly |
| **Operator.add accumulation** | `Annotated[List, operator.add]` on list fields | Research data accumulates across loop-back iterations without overwriting |
| **Conditional routing** | `route_after_critic()` reads `needs_more_research` bool | The Critic's verdict drives graph routing — agents have zero routing logic |
| **Tool use** | Researcher calls DuckDuckGo as an external tool | Agents can act on live data, not just model training knowledge |
| **Self-correction loop** | Critic → Researcher → Summarizer → Critic (max 2×) | System evaluates its own output and autonomously decides whether to iterate |
| **Graceful degradation** | Max loop cap + empty result handlers | Pipeline always terminates — no infinite loops, no silent crashes |
| **Two-phase report generation** | LLM writes prose body, Python appends sources/metadata | Precision where formatting matters, fluency where prose matters |

---

## Agent Breakdown

### Planner Agent
- **Input:** Raw user query
- **Output:** 3–5 focused, non-overlapping sub-questions
- **Temperature:** `0.1` (deterministic structured output)
- **Why it exists:** Searching for one vague query returns shallow results. 5 targeted searches return deep, complementary information per angle.

### Researcher Agent
- **Input:** Sub-questions + (on loop-back) Critic's specific gap list
- **Output:** Raw research data with extracted key facts + source URLs
- **Temperature:** `0.1`
- **Tool:** DuckDuckGo (`ddgs` library — zero API key required)
- **Design:** Two-phase per sub-question — search first, LLM filters relevance second. 2s rate-limit between searches to avoid blocks.

### Summarizer Agent
- **Input:** Raw research data (all loop iterations accumulated)
- **Output:** Prose summaries, one per sub-question, with `has_gap` flag
- **Temperature:** `0.4` (fluent prose, still grounded in facts)
- **Why it exists:** Researcher outputs fragmented bullet points. Critic and Writer need coherent paragraphs. Synthesis ≠ extraction.

### Critic Agent
- **Input:** All summaries + original query
- **Output:** Structured verdict (APPROVED / NEEDS_MORE_RESEARCH) + per-summary PASS/FAIL scores + specific gap list
- **Temperature:** `0.1` (consistent, deterministic evaluation)
- **Rubric:** Specificity · Completeness · Recency · Depth
- **Loop safety:** After `MAX_RESEARCH_LOOPS` (default: 2), always approves — pipeline never hangs

### Writer Agent
- **Input:** Summaries + sources + critique + metadata
- **Output:** Full structured markdown report
- **Temperature:** `0.5` (highest — most linguistic freedom for prose variety)
- **Design:** LLM writes the 5 content sections. Python programmatically appends sources and Research Notes. Confidence rating derived from loop count + Critic verdict + gap count.

---

## Tech Stack

| Component | Technology |
|---|-
| LLM backbone | LLaMA 3.3 70B via [Groq API](https://console.groq.com)  (14,400 req/day free) 
| Agent orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) 
| Agent framework | [LangChain](https://python.langchain.com) 
| Web search | [ddgs](https://github.com/deedy5/ddgs) (DuckDuckGo) 
| UI | [Streamlit](https://streamlit.io)
| Language | Python 3.11+ 


---

## Project Structure

```
multi_agent_researcher/
│
├── agents/
│   ├── __init__.py
│   ├── planner.py          # Query → sub-questions
│   ├── researcher.py       # Sub-questions → web research + fact extraction
│   ├── summarizer.py       # Raw facts → prose summaries
│   ├── critic.py           # Quality gate + loop-back trigger
│   └── writer.py           # Summaries → final report
│
├── tools/
│   ├── __init__.py
│   └── search_tool.py      # DuckDuckGo search wrapper + LLM formatter
│
├── graph/
│   ├── __init__.py
│   ├── state.py            # ResearchState TypedDict (shared agent memory)
│   └── workflow.py         # LangGraph StateGraph + conditional routing
│
├── config/
│   ├── __init__.py
│   └── settings.py         # LLM factory, model name, loop limits
│
├── app.py                  # Streamlit web UI with live agent activity log
├── requirements.txt
├── .env                    # Your Groq API key (never commit this)
└── .gitignore
```

---

## Setup & Installation

### Prerequisites
- Python 3.11 or newer
- A free [Groq API key](https://console.groq.com)

### Step 1 — Clone the repository

```bash
git clone https://github.com/gt951436/multi-agent-researcher.git
cd multi-agent-researcher
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure your API key

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_key_here
```

### Step 5 — Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Sample Output

**Query:** *"What are the key differences between supervised and unsupervised machine learning?"*

```markdown
# Supervised vs Unsupervised Machine Learning: Key Differences

## Executive Summary
The distinction between supervised and unsupervised machine learning represents
a fundamental divide in how AI systems learn from data. Supervised learning relies
on labeled examples with known outcomes, while unsupervised learning discovers
hidden structure in unlabeled data...

## 1. Algorithmic Distinctions
Supervised learning algorithms — including linear regression, SVMs, and neural
networks — operate by minimizing a loss function computed against ground-truth
labels. Unsupervised methods such as k-means clustering, PCA, and autoencoders
instead optimize internal coherence metrics like inertia or reconstruction error...

## 2. Data Requirements and Training
...

## Key Takeaways
- Supervised learning requires labeled data; unsupervised does not
- Evaluation metrics differ fundamentally: accuracy/F1 vs. silhouette score/BIC
- Supervised methods dominate production deployments; unsupervised excels at
  exploratory analysis and dimensionality reduction
...

## Conclusion
...

---

## Sources
1. https://en.wikipedia.org/wiki/...
2. https://arxiv.org/...
[20 total sources]

## Research Notes
| Field             | Value                                      |
|-------------------|--------------------------------------------|
| Generated         | April 05, 2026 at 14:32                    |
| Confidence        | 🟢 High                                    |
| Research loops    | 0                                          |
| Sources consulted | 20                                         |
| System            | Multi-Agent Research Assistant (LLaMA 3.3) |
```

---

## How the Self-Correction Loop Works

```
First pass:
  Planner → Researcher → Summarizer → Critic
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                        NEEDS_MORE_RESEARCH        APPROVED
                              │                       │
                         [loop back]            → Writer → Report
                              │
                  Researcher (targeted re-search)
                          │
                    Summarizer
                          │
                       Critic
                          │
                   [approve or loop again — max 2 total loops]
```

The Critic evaluates each summary on 4 criteria:

| Criterion | Fails if... |
|---|---|
| **Specificity** | Vague phrases like "various methods" with no named examples |
| **Completeness** | Sub-question not actually answered, or "details not available" |
| **Recency** | Only background knowledge, no recent developments |
| **Depth** | Surface-level definitions only, no analytical insight |

If 2+ summaries fail → `NEEDS_MORE_RESEARCH` → loop back to Researcher with the specific gap list in context.

After `MAX_RESEARCH_LOOPS` (default: 2) → always proceed to Writer regardless, to guarantee termination.

---

## Confidence Rating

The Writer assigns one of three confidence levels to each report:

| Rating | Conditions |
|---|---|
| 🟢 **High** | Critic approved on first pass, 0 gaps, all summaries PASS |
| 🟡 **Medium** | Approved after 1+ research loops, or minor Critic issues |
| 🔴 **Low** | 2+ research gaps, or max loops reached without clean approval |

---

## Configuration

Edit `config/settings.py` to tune the system:

```python
# Switch the LLM model (any Groq-supported open model)
# Options: llama-3.3-70b-versatile, llama-3.1-8b-instant,
#          mixtral-8x7b-32768, gemma2-9b-it
model = "llama-3.3-70b-versatile"

# Max research loop-back iterations before forcing forward
MAX_RESEARCH_LOOPS = 2   # increase to 3 for deeper (slower) research

# Per-agent temperatures
# Planner/Researcher/Critic: 0.1  (deterministic, structured)
# Summarizer: 0.4                 (balanced prose)
# Writer: 0.5                     (fluent, varied prose)
```

---

## Limitations

- **DuckDuckGo snippet length:** Search results return 2–3 sentence snippets, not full articles. Deep technical topics may produce surface-level summaries even after 2 research loops.
- **Groq free tier rate limits:** 14,400 requests/day, 6,000 tokens/minute for LLaMA 3.3 70B. For high-volume usage, consider the paid tier or a smaller model.
- **Self-evaluation bias:** The Critic uses the same LLM that generated the summaries. LLMs have a known tendency to rate their own outputs favorably — the strict rubric mitigates but does not eliminate this.
- **No persistent memory:** Each research run starts fresh. There is no cross-session memory or document ingestion.

---

## Concepts Demonstrated

This project was built to demonstrate the following skills relevant to AI/ML engineering roles:

- **Multi-agent orchestration** with LangGraph `StateGraph`
- **Stateful agent pipelines** using typed shared state (`TypedDict`)
- **Conditional graph routing** with runtime decision functions
- **Tool use** — agents calling external functions (web search)
- **Self-correcting feedback loops** — system evaluates and improves its own output
- **Context engineering** — prompt design for deterministic structured output
- **Operator.add state accumulation** — safe list merging across agent loops
- **Graceful degradation** — max loop caps, empty result handlers, fallback paths
- **Production patterns** — single-responsibility agents, isolated testability, clean separation of concerns

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain: How and When to Build Multi-Agent Systems (2025)](https://blog.langchain.com/how-and-when-to-build-multi-agent-systems/)
- [Benchmarking Multi-Agent Architectures — LangChain Blog](https://blog.langchain.com/benchmarking-multi-agent-architectures/)
- [ReAct: Synergizing Reasoning and Acting in LLMs — Yao et al. (2022)](https://arxiv.org/abs/2210.03629)
- [Groq API — Free LLaMA 3.3 70B inference](https://console.groq.com)
- [ddgs — DuckDuckGo Search Python Library](https://github.com/deedy5/ddgs)

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Built by [GT](https://github.com/gt951436)

</div>