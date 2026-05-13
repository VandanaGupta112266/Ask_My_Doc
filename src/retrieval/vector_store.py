import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

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

def get_hybrid_retriever(vector_store: Chroma, k: int = 5):
    """Creates an EnsembleRetriever combining Vector Search and BM25."""
    # 1. Get all documents from Chroma to build BM25 index
    # Note: For very large datasets, you'd want to persist the BM25 index separately.
    all_docs = vector_store.get()["documents"]
    metadatas = vector_store.get()["metadatas"]
    
    # Reconstruct Document objects
    documents = [
        Document(page_content=doc, metadata=meta) 
        for doc, meta in zip(all_docs, metadatas)
    ]
    
    if not documents:
        print("Warning: No documents found in vector store to build BM25 index.")
        return vector_store.as_retriever(search_kwargs={"k": k})

    # 2. Initialize BM25
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    # 3. Initialize Vector Retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # 4. Combine them using Reciprocal Rank Fusion
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

if __name__ == "__main__":
    # Test initialization
    store = get_vector_store()
    print("Vector store loaded successfully.")
