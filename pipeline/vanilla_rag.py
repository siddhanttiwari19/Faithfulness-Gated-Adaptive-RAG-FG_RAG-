from typing import Dict, Any, List
import time
from fg_arag.retrieval.hybrid import HybridRetriever
from fg_arag.generation.base import BaseGenerator

class VanillaRAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator: BaseGenerator, top_k: int = 10):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> Dict[str, Any]:
        """
        Executes the Vanilla RAG baseline pipeline.
        """
        trace = []
        start_time = time.time()
        
        # 1. Retrieve
        retrieval_start = time.time()
        retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k)
        retrieval_latency = time.time() - retrieval_start
        
        evidence_texts = [doc["text"] for doc in retrieved_docs]
        
        trace.append({
            "step": "retrieval",
            "query": query,
            "top_k": self.top_k,
            "retrieved_count": len(retrieved_docs),
            "latency": retrieval_latency
        })
        
        # 2. Generate
        generation_start = time.time()
        generation_result = self.generator.generate_answer(query, evidence_texts)
        generation_latency = time.time() - generation_start
        
        answer = generation_result["answer"]
        
        trace.append({
            "step": "generation",
            "latency": generation_latency,
            "model": generation_result.get("model", "unknown")
        })
        
        total_latency = time.time() - start_time
        
        return {
            "query": query,
            "final_answer": answer,
            "retrieved_documents": retrieved_docs,
            "metrics": {
                "total_latency": total_latency,
                "retrieval_latency": retrieval_latency,
                "generation_latency": generation_latency,
            },
            "trace": trace
        }
