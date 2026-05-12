from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents: list[Document], chunk_size: int = 600, chunk_overlap: int = 100) -> list[Document]:
    """Splits documents into smaller chunks for vectorization."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add a chunk_id to metadata to make citations easier later
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        # Extract filename from path for cleaner citations
        if "/" in source:
            source = source.split("/")[-1]
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = f"{source}_p{page}_c{i}"
        
    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # Test chunking
    from src.ingestion.loader import load_documents_from_directory
    docs = load_documents_from_directory("data/raw_docs")
    if docs:
        chunks = chunk_documents(docs)
        print(f"Sample chunk metadata: {chunks[0].metadata}")
