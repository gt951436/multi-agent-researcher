from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)

response = llm.invoke("In one sentence, what is a multi-agent system?")
print("LLM response:",response.content)
print("\nSetup completed! start building...")