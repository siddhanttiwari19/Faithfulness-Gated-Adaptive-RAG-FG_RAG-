import pickle
import os
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Indexer:
    def __init__(self, index_dir: str = "data/indexes/bm25"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.bm25 = None
        self.corpus_data = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Builds the BM25 index from a list of chunks.
        chunks expected format:
        [ { "chunk_id": "...", "text": "...", ...metadata... }, ... ]
        """
        self.corpus_data = chunks
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_index(self, filename: str = "bm25_index.pkl"):
        path = os.path.join(self.index_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump({"bm25": self.bm25, "corpus_data": self.corpus_data}, f)

    def load_index(self, filename: str = "bm25_index.pkl"):
        path = os.path.join(self.index_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index not found at {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus_data = data["corpus_data"]

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenization, lowercase
        # Advanced tokenization can be added if needed, but keeping it simple for BM25 MVP
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25:
            raise ValueError("BM25 index not loaded or built.")
            
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            result = self.corpus_data[idx].copy()
            result["score"] = doc_scores[idx]
            results.append(result)
            
        return results
