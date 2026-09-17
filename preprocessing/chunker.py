from typing import List, Dict, Any
from transformers import AutoTokenizer

class TokenAwareChunker:
    def __init__(self, tokenizer_name: str = "BAAI/bge-base-en-v1.5", chunk_size: int = 256, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # We load a fast tokenizer for accurate token counting aligned with the embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunks text into overlapping blocks of `chunk_size` tokens.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        if not tokens:
            return chunks
            
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # Move start index, considering overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
            
            # Break if we're not advancing (e.g. overlap >= chunk_size)
            if self.chunk_size <= self.chunk_overlap:
                break
                
        return chunks

    def process_document(self, doc_id: str, title: str, text: str, dataset_name: str = "hotpotqa") -> List[Dict[str, Any]]:
        """
        Processes a document into chunks with metadata.
        """
        chunks_text = self.chunk_text(text)
        processed_chunks = []
        
        for i, chunk in enumerate(chunks_text):
            token_count = len(self.tokenizer.encode(chunk, add_special_tokens=False))
            processed_chunks.append({
                "chunk_id": f"{doc_id}_chunk_{i}",
                "document_id": doc_id,
                "document_title": title,
                "chunk_index": i,
                "text": chunk,
                "token_count": token_count,
                "source_dataset": dataset_name
            })
            
        return processed_chunks
