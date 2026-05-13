import streamlit as st
import os
import sys
from datetime import datetime

# Add the project root to sys.path so we can import from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.ingestion.loader import load_documents_from_directory
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import get_vector_store, add_documents_to_store, retrieve_top_k, get_hybrid_retriever
from src.generation.qa_pipeline import generate_answer
from src.generation.chat_history import create_session, save_message, get_sessions, get_messages, update_session_title
import uuid

def get_time_ago(timestamp_str):
    if not timestamp_str:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(timestamp_str)
        delta = datetime.now() - dt
        seconds = delta.total_seconds()
        if seconds < 60:
            return "Just now"
        if seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds // 3600)}h ago"
        return f"{int(seconds // 86400)}d ago"
    except:
        return "Unknown"

# Load env variables
load_dotenv()

st.set_page_config(page_title="Ask My Docs", page_icon="📚")

st.title("📚 Ask My Docs - Production RAG")
st.markdown("Phase 1 implementation using ChromaDB, Qwen Embeddings, and OpenRouter LLMs.")

# Initialize Vector Store in session state
if "vector_store" not in st.session_state:
    with st.spinner("Loading Embedding Model..."):
        st.session_state.vector_store = get_vector_store()

# Initialize Session Management
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
    create_session(st.session_state.current_session_id, "Initial Chat")

# Sidebar for Ingestion and Sessions
with st.sidebar:
    st.header("💬 Chat Sessions")
    if st.button("➕ New Chat"):
        st.session_state.current_session_id = str(uuid.uuid4())
        create_session(st.session_state.current_session_id, f"Chat {datetime.now().strftime('%H:%M:%S')}")
        st.rerun()

    sessions = get_sessions()
    for sid, title, updated_at in sessions:
        time_ago = get_time_ago(updated_at)
        is_active = sid == st.session_state.current_session_id
        
        # Add a visual indicator and use 'primary' type for active session
        icon = "💬" if is_active else "📄"
        button_label = f"{icon} {title[:20]}... ({time_ago})" if len(title) > 20 else f"{icon} {title} ({time_ago})"
        
        if st.button(
            button_label, 
            key=sid, 
            use_container_width=True, 
            type="primary" if is_active else "secondary"
        ):
            st.session_state.current_session_id = sid
            st.rerun()

    st.divider()
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

   
# Main Chat Interface
st.header(f"💬 Conversation History")

# Load and Display Messages for Current Session
messages = get_messages(st.session_state.current_session_id)
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message
    save_message(st.session_state.current_session_id, "user", prompt)
    
    # If this is the first message, update the session title to the user's prompt
    current_messages = get_messages(st.session_state.current_session_id)
    if len(current_messages) == 1:
        # Limit title length
        new_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
        update_session_title(st.session_state.current_session_id, new_title)
        st.rerun()

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Processing request..."):
            answer, chunks = generate_answer(prompt, st.session_state.vector_store)
            st.markdown(answer)
            
            # Add collapsible section for sources
            if chunks:
                with st.expander("🔍 View Retrieved Chunks"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}** (ID: `{chunk.metadata.get('chunk_id', 'N/A')}`)")
                        st.text(chunk.page_content)
                        st.markdown("---")
            
            # Save assistant response
            save_message(st.session_state.current_session_id, "assistant", answer)
