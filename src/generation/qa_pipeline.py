import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.generation.prompt_loader import load_prompt
from src.retrieval.vector_store import get_vector_store, get_hybrid_retriever
from src.retrieval.reranker import Reranker
from langfuse.decorators import observe, langfuse_context

load_dotenv()

def get_llm():
    """Initializes the LLM via OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Warning: OPENROUTER_API_KEY is not set.")
        
    return ChatOpenAI(
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        request_timeout=180  # Increased timeout for slow free-tier responses
    )

# Global reranker instance (initialized on first use)
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker

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
    
    # 1. Start Pipeline
    print(f"\n--- New Request: {query} ---")
    
    # Update trace with professional metadata
    langfuse_context.update_current_trace(
        name="Production RAG Query",
        tags=["development"],
        metadata={"model": "minimax-m2.5"}
    )
    
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
        
        # Capture model parameters and usage
        langfuse_context.update_current_observation(
            model="minimax/minimax-m2.5:free",
            usage={
                "input": len(context_text + query) // 4,
                "output": len(response.content) // 4
            },
            metadata={
                "top_rerank_score": float(final_chunks[0].metadata.get('re_score', 0)) if final_chunks else 0,
                "chunks_retrieved": len(final_chunks)
            }
        )
        
        return response.content, final_chunks
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}", []

if __name__ == "__main__":
    print("QA Pipeline ready.")
