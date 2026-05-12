import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

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
        temperature=0.1
    )

def generate_answer(query: str, retrieved_chunks: list[Document]) -> str:
    """Generates an answer using the retrieved chunks and LLM."""
    if not retrieved_chunks:
        return "I don't have enough information to answer this based on the provided documents."
        
    llm = get_llm()
    
    # Format the context
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        chunk_id = chunk.metadata.get('chunk_id', f'chunk_{i}')
        context_text += f"\n--- Chunk ID: {chunk_id} ---\n{chunk.page_content}\n"
    
    # Define the strict prompt
    system_prompt = """You are a helpful assistant. 
Answer questions using ONLY the provided context chunks. 
Cite each claim using the Chunk ID provided. Example: [Source: file.pdf, Chunk ID: file.pdf_p0_c1].
If the chunks do not support an answer to the question, strictly say:
"I don't have enough information to answer this."
Do not attempt to use outside knowledge."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:")
    ])
    
    chain = prompt | llm
    
    print("Generating answer...")
    try:
        response = chain.invoke({
            "context": context_text,
            "question": query
        })
        return response.content
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"

if __name__ == "__main__":
    print("QA Pipeline ready.")
