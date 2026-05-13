import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.generation.prompt_loader import load_prompt
from src.retrieval.vector_store import get_vector_store, get_hybrid_retriever
from src.retrieval.reranker import Reranker
from langfuse import observe

load_dotenv()

def get_llm():
    """Initializes the LLM via OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Warning: OPENROUTER_API_KEY is not set.")
        
    return ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

# Global reranker instance (initialized on first use)
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker

@observe()
def is_greeting_or_general(query: str) -> bool:
    """Uses a fast-pass check then an LLM to decide if a query is general."""
    # Fast-pass for common short greetings
    greetings = ["hi", "hello", "hey", "hola", "greetings", "morning", "afternoon"]
    clean_query = query.lower().strip().strip('?!.')
    if clean_query in greetings:
        return True
        
    llm = get_llm()
    routing_prompt = f"""Task: Categorize user query.
    Rules:
    - Respond 'GENERAL' if it's a greeting, pleasantry, or intro (e.g., 'how are you', 'who are you', 'help').
    - Respond 'SEARCH' if it's a question about specific facts or documents.
    
    Query: {query}
    
    Response (GENERAL/SEARCH):"""
    
    try:
        response = llm.invoke(routing_prompt)
        return "GENERAL" in response.content.upper()
    except Exception as e:
        print(f"Routing Error: {e}")
        return False

@observe()
def generate_answer(query: str, vector_store=None) -> tuple[str, list[Document]]:
    """Generates an answer using a Router + (Hybrid Search + Reranking)."""
    
    # 1. Intent Routing (Absolute First Step)
    print(f"\n--- New Request: {query} ---")
    print("Step 1: Routing Query...")
    if is_greeting_or_general(query):
        print("Result: General Conversation - Skipping Retrieval")
        llm = get_llm()
        response = llm.invoke(f"The user said: '{query}'. Provide a brief, friendly greeting.")
        return response.content, []
        
    print("Result: Technical Question - Proceeding to Retrieval")
    if vector_store is None:
        vector_store = get_vector_store()
        
    # 2. Hybrid Retrieval
    print("Step 2: Performing Hybrid Search...")
    retriever = get_hybrid_retriever(vector_store, k=15)
    retrieved_chunks = retriever.invoke(query)
    
    if not retrieved_chunks:
        return "I don't have enough information to answer this based on the provided documents.", []
        
    # 2. Re-ranking (narrow down to top 5 most relevant)
    print("Re-ranking results...")
    reranker = get_reranker()
    final_chunks = reranker.rerank(query, retrieved_chunks, top_n=5)
    
    llm = get_llm()
    
    # 3. Format the context
    context_text = ""
    for i, chunk in enumerate(final_chunks):
        chunk_id = chunk.metadata.get('chunk_id', f'chunk_{i}')
        context_text += f"\n--- Chunk ID: {chunk_id} ---\n{chunk.page_content}\n"
    
    # 4. Load prompt from YAML
    prompt = load_prompt("prompts/qa_prompts.yaml", "qa_task")
    
    chain = prompt | llm
    
    print("Generating answer with LLM...")
    try:
        response = chain.invoke({
            "context": context_text,
            "question": query
        })
        return response.content, final_chunks
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}", []

if __name__ == "__main__":
    print("QA Pipeline ready.")
