from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        print(f"Loading Reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        """Reranks the documents based on the query and returns the top_n."""
        if not documents:
            return []
            
        # Prepare pairs for the CrossEncoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Combine documents with their scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score in descending order
        sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        # Return only the top_n documents and attach score to metadata
        print(f"Reranked {len(documents)} documents. Top score: {sorted_docs[0][1]:.4f}")
        
        final_docs = []
        for doc, score in sorted_docs[:top_n]:
            doc.metadata["re_score"] = float(score)
            final_docs.append(doc)
            
        return final_docs

if __name__ == "__main__":
    # Test reranker
    reranker = Reranker()
    test_query = "What is AI risk management?"
    test_docs = [
        Document(page_content="AI risk management is a process of identifying and mitigating risks."),
        Document(page_content="The weather today is sunny."),
        Document(page_content="NIST provides a framework for AI risk.")
    ]
    results = reranker.rerank(test_query, test_docs, top_n=2)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content}")
