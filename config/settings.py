# LLM init, environment variables
# Every agent will import the llm from here instaed of each creating their own
# If we ever want to swiitch mpodel, we change here only
# This method follows Dependency Injection Pattern --> makes system easy to maintain and test

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

#---LLM Configs---
# 2 temperature settings:
# LOW (0.1) : factual agents : Planner, Researcher, Critic
# HIGH (0.4) : Creative agents : Summarizer, Writer

def get_llm(temperature: float=0.1) -> ChatGroq:
    """
    Factory fn that return a configured Groq LLM instance.
    Every agent calls this to get their LLM - single src of truth.
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=2048, # good enough for agent responses 
    )
    
MODEL_NAME = "LLaMA 3.3 70B (Groq)"
MAX_RESEARCH_LOOPS = 2 # Critic can trigger re-research at most 2 times