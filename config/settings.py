import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# --- Model & Runtime Configuration ---
# Central source of truth for runtime settings, overridable via environment variables.
DEFAULT_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MODEL_NAME: str = os.getenv("MODEL_DISPLAY_NAME", f"LLaMA 3.3 70B ({DEFAULT_MODEL})")
MAX_RESEARCH_LOOPS: int = int(os.getenv("MAX_RESEARCH_LOOPS", "2"))
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))

# Temperature presets:
# LOW (0.1)  : Analytical, deterministic nodes (Planner, Researcher, Critic)
# MED (0.4)  : Synthesis node (Summarizer)
# HIGH (0.5) : Expressive, stylistic node (Writer)
TEMP_FACTUAL: float = 0.1
TEMP_SYNTHESIS: float = 0.4
TEMP_CREATIVE: float = 0.5


def validate_config() -> dict[str, str]:
    """
    Validates essential environment settings at startup.
    Returns a dictionary of validated configurations.
    Raises ValueError with actionable instructions if critical credentials are missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "GROQ_API_KEY is not set. Please add your Groq API key to your .env file:\n"
            "GROQ_API_KEY=gsk_..."
        )
    return {
        "model": DEFAULT_MODEL,
        "max_research_loops": str(MAX_RESEARCH_LOOPS),
        "request_timeout": str(REQUEST_TIMEOUT),
    }


def get_llm(temperature: float = TEMP_FACTUAL, model: str = None) -> ChatGroq:
    """
    Factory function returning a configured Groq LLM instance.
    Centralizes client creation and guarantees standard timeouts, model selection,
    and API key injection across all agent nodes.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        model: Optional model override; defaults to DEFAULT_MODEL ("llama-3.3-70b-versatile")

    Returns:
        Configured ChatGroq client
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "GROQ_API_KEY is missing from environment. Ensure it is defined in .env or system environment."
        )

    return ChatGroq(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        api_key=api_key,
        max_tokens=2048,
        request_timeout=REQUEST_TIMEOUT,
    )