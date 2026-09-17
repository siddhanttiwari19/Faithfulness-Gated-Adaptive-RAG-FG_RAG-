import pytest
from fg_arag.retrieval.hybrid import HybridRetriever

class MockVectorStore:
    def search(self, query: str, top_k: int = 10):
        # returns distance scores (lower is better)
        return [
            {"chunk_id": "c1", "score": 0.1, "text": "dense match 1"},
            {"chunk_id": "c2", "score": 0.5, "text": "dense match 2"},
            {"chunk_id": "c3", "score": 0.9, "text": "dense match 3"},
        ]

class MockBM25Indexer:
    def search(self, query: str, top_k: int = 10):
        # returns BM25 scores (higher is better)
        return [
            {"chunk_id": "c2", "score": 10.0, "text": "sparse match 1"},
            {"chunk_id": "c3", "score": 5.0, "text": "sparse match 2"},
            {"chunk_id": "c4", "score": 1.0, "text": "sparse match 3"},
        ]

def test_hybrid_retriever():
    vs = MockVectorStore()
    bm25 = MockBM25Indexer()
    
    retriever = HybridRetriever(vector_store=vs, bm25_indexer=bm25, alpha=0.5)
    results = retriever.retrieve("test query", top_k=2)
    
    assert len(results) == 2
    # c1: dense normalized score = 1.0 (inverted 0.1), sparse = 0.0 -> hybrid = 0.5
    # c2: dense normalized score = 0.5 (inverted 0.5), sparse = 1.0 -> hybrid = 0.75
    # c3: dense normalized score = 0.0 (inverted 0.9), sparse = 0.44 -> hybrid = 0.22
    # c4: dense normalized = 0.0, sparse = 0.0 -> hybrid = 0.0
    
    # c2 should be top (0.75), c1 next (0.50)
    assert results[0]["chunk_id"] == "c2"
    assert results[1]["chunk_id"] == "c1"
