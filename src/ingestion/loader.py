import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader

def load_documents_from_directory(directory_path: str) -> list[Document]:
    """Loads all PDF documents from the given directory."""
    documents = []
    path = Path(directory_path)
    if not path.exists():
        print(f"Directory not found: {directory_path}")
        return documents

    # Load PDFs
    for file_path in path.glob("*.pdf"):
        print(f"Loading {file_path.name}...")
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        documents.extend(docs)
        print(f"Loaded {len(docs)} pages from {file_path.name}")

    # Load Markdown and Text

    for file_path in path.glob("*.md"):
        print(f"Loading {file_path.name}...")
        loader = UnstructuredMarkdownLoader(str(file_path))
        docs = loader.load()
        documents.extend(docs)

    # Load Text

    for file_path in path.glob("*.txt"):
        print(f"Loading {file_path.name}...")
        loader = TextLoader(str(file_path))
        docs = loader.load()
        documents.extend(docs)
        
    return documents

if __name__ == "__main__":
    # Test loading
    docs = load_documents_from_directory("data/raw_docs")
    print(f"Total pages loaded: {len(docs)}")
