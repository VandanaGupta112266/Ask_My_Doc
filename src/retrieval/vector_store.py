import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def get_embeddings():
    """Initializes the Qwen3 embedding model."""
    print("Loading Qwen3-Embedding-0.6B model. This might take a minute on first run...")
    return HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_vector_store(persist_directory: str = "data/chroma"):
    """Returns a Chroma vector store instance."""
    embeddings = get_embeddings()
    os.makedirs(persist_directory, exist_ok=True)
    
    vector_store = Chroma(
        collection_name="policy_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def add_documents_to_store(documents: list[Document], vector_store: Chroma):
    """Adds chunked documents to the vector store."""
    if not documents:
        print("No documents to add.")
        return
        
    print(f"Adding {len(documents)} chunks to Chroma vector store...")
    vector_store.add_documents(documents)
    print("Successfully added documents.")

def retrieve_top_k(query: str, vector_store: Chroma, k: int = 5) -> list[Document]:
    """Retrieves the top-k most relevant chunks for a given query."""
    print(f"Searching for: '{query}'")
    results = vector_store.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    # Test initialization
    store = get_vector_store()
    print("Vector store loaded successfully.")
