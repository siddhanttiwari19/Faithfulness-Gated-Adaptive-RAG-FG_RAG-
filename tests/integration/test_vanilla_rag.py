import pytest
from fg_arag.pipeline.vanilla_rag import VanillaRAGPipeline
from fg_arag.retrieval.hybrid import HybridRetriever
from fg_arag.generation.base import BaseGenerator
from fg_arag.retrieval.vector_store import VectorStore
from fg_arag.preprocessing.bm25_indexer import BM25Indexer
from fg_arag.tests.unit.test_hybrid_retrieval import MockVectorStore, MockBM25Indexer
from fg_arag.tests.unit.test_generation import MockGeminiClient

class MockGenerator(BaseGenerator):
    def generate_answer(self, query, evidence, temperature=0.0):
        return {"answer": "Mock generated answer", "latency": 0.1, "model": "mock-model"}
        
    def decompose_claims(self, answer, temperature=0.0):
        pass
        
    def regenerate_claim(self, claim, evidence, temperature=0.0):
        pass

def test_vanilla_rag_pipeline():
    vs = MockVectorStore()
    bm25 = MockBM25Indexer()
    retriever = HybridRetriever(vector_store=vs, bm25_indexer=bm25)
    generator = MockGenerator()
    
    pipeline = VanillaRAGPipeline(retriever=retriever, generator=generator, top_k=2)
    
    result = pipeline.run("test query")
    
    assert result["query"] == "test query"
    assert result["final_answer"] == "Mock generated answer"
    assert len(result["retrieved_documents"]) == 2
    assert "metrics" in result
    assert "total_latency" in result["metrics"]
    assert len(result["trace"]) == 2
    assert result["trace"][0]["step"] == "retrieval"
    assert result["trace"][1]["step"] == "generation"
