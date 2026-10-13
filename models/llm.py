from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


llm=ChatGroq(
    model=os.getenv("GROQ_MODEL"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=4096

)

def get_llm()->ChatGroq:
    """
    Returns the shared Groq LLM instance.
    """
    return llm