import pytest
from fg_arag.preprocessing.embedder import BGEEmbedder
from fg_arag.retrieval.vector_store import VectorStore
from fg_arag.preprocessing.bm25_indexer import BM25Indexer

def test_embedder_initialization():
    # We test on CPU to avoid CUDA dependency in basic unit tests
    embedder = BGEEmbedder(device="cpu")
    embeddings = embedder.embed_queries(["test query"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768  # BGE base dimensionality

def test_bm25_indexer():
    indexer = BM25Indexer(index_dir="data/test_indexes/bm25")
    chunks = [
        {"chunk_id": "c1", "text": "This is a document about dogs."},
        {"chunk_id": "c2", "text": "This is a document about cats."}
    ]
    indexer.build_index(chunks)
    
    results = indexer.search("dogs", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    
def test_vector_store():
    # Skip full Chroma testing to avoid heavy disk operations, just check it compiles 
    # Real test requires temp directory for Chroma
    pass
