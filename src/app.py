import streamlit as st
import os
import sys

# Add the project root to sys.path so we can import from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.ingestion.loader import load_documents_from_directory
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import get_vector_store, add_documents_to_store, retrieve_top_k
from src.generation.qa_pipeline import generate_answer

# Load env variables
load_dotenv()

st.set_page_config(page_title="Ask My Docs", page_icon="📚")

st.title("📚 Ask My Docs - Production RAG")
st.markdown("Phase 1 implementation using ChromaDB, Qwen Embeddings, and OpenRouter LLMs.")

# Initialize Vector Store in session state
if "vector_store" not in st.session_state:
    with st.spinner("Loading Embedding Model..."):
        st.session_state.vector_store = get_vector_store()

# Sidebar for Ingestion
with st.sidebar:
    st.header("⚙️ System Control")
    
    st.divider()
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF, MD, or TXT", type=["pdf", "md", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("💾 Save & Ingest Uploaded Files"):
            with st.spinner("Saving and ingesting..."):
                os.makedirs("data/raw_docs", exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("data/raw_docs", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Trigger ingestion
                docs = load_documents_from_directory("data/raw_docs")
                if docs:
                    chunks = chunk_documents(docs)
                    add_documents_to_store(chunks, st.session_state.vector_store)
                    st.success(f"Successfully ingested {len(uploaded_files)} files!")

   
# Main QA Interface
st.header("💬 Ask a Question")
query = st.text_input("What would you like to know about your documents?")

if st.button("Search & Answer") and query:
    if not os.environ.get("OPENROUTER_API_KEY"):
        st.error("Please enter your OpenRouter API Key in the sidebar.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved_chunks = retrieve_top_k(query, st.session_state.vector_store)
            
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, retrieved_chunks)
            
        st.markdown("### Answer")
        st.markdown(answer)
        
        with st.expander("View Retrieved Sources"):
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {i+1}** (ID: `{chunk.metadata.get('chunk_id')}`)")
                st.text(chunk.page_content)
                st.markdown("---")
