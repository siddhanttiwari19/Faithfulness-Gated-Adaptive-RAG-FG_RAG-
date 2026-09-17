from typing import List, Dict, Any
from fg_arag.retrieval.vector_store import VectorStore
from fg_arag.preprocessing.bm25_indexer import BM25Indexer

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_indexer: BM25Indexer, alpha: float = 0.7):
        self.vector_store = vector_store
        self.bm25_indexer = bm25_indexer
        self.alpha = alpha

    def _normalize_scores(self, results: List[Dict[str, Any]], score_key: str, invert: bool = False) -> List[Dict[str, Any]]:
        """
        Min-max normalization of scores.
        If invert is True, smaller original scores become larger normalized scores (e.g. for distance metrics).
        """
        if not results:
            return results
            
        scores = [r[score_key] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            for r in results:
                r[f"{score_key}_normalized"] = 1.0
            return results
            
        for r in results:
            norm_score = (r[score_key] - min_score) / (max_score - min_score)
            if invert:
                norm_score = 1.0 - norm_score
            r[f"{score_key}_normalized"] = norm_score
            
        return results

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval combining Dense (VectorStore) and Sparse (BM25) results.
        """
        # Retrieve more from each to ensure good intersection/coverage before top_k
        dense_results = self.vector_store.search(query, top_k=top_k * 2)
        sparse_results = self.bm25_indexer.search(query, top_k=top_k * 2)
        
        # VectorStore returns distances (lower is better), so we invert for normalization
        dense_results = self._normalize_scores(dense_results, "score", invert=True)
        # BM25 returns scores (higher is better)
        sparse_results = self._normalize_scores(sparse_results, "score", invert=False)
        
        # Combine results
        combined_scores = {}
        document_metadata = {}
        
        # Process Dense
        for r in dense_results:
            chunk_id = r["chunk_id"]
            combined_scores[chunk_id] = self.alpha * r["score_normalized"]
            document_metadata[chunk_id] = r
            
        # Process Sparse
        for r in sparse_results:
            chunk_id = r["chunk_id"]
            sparse_score = (1.0 - self.alpha) * r["score_normalized"]
            
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += sparse_score
            else:
                combined_scores[chunk_id] = sparse_score
                document_metadata[chunk_id] = r
                
        # Sort and return top_k
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = []
        for chunk_id, hybrid_score in sorted_chunks:
            doc = document_metadata[chunk_id].copy()
            doc["hybrid_score"] = hybrid_score
            final_results.append(doc)
            
        return final_results
