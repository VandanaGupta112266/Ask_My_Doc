import os
from dotenv import load_dotenv
from src.generation.qa_pipeline import get_llm

load_dotenv()
llm = get_llm()
print("Sending test message...")
res = llm.invoke("Hi")
print("Response:", res.content)
