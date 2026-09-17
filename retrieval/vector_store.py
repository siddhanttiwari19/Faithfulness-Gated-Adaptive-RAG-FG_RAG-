import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from fg_arag.preprocessing.embedder import BGEEmbedder

class VectorStore:
    def __init__(self, persist_directory: str = "data/indexes/chroma", collection_name: str = "fg_arag", embedder: Optional[BGEEmbedder] = None):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        self.collection_name = collection_name
        self.embedder = embedder
        
        # We handle embeddings externally so we don't rely on Chroma's default embedding function
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Adds chunks to the vector store.
        chunks expected format:
        [ { "chunk_id": "...", "text": "...", ...metadata... }, ... ]
        """
        if not self.embedder:
            raise ValueError("Embedder must be provided to add chunks.")
            
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            
            # Embed documents
            embeddings = self.embedder.embed_documents(texts)
            
            # Remove text and chunk_id from metadata
            metadatas = []
            for c in batch:
                meta = {k: v for k, v in c.items() if k not in ["text", "chunk_id"]}
                metadatas.append(meta)
                
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Searches for top_k most similar chunks.
        """
        if not self.embedder:
            raise ValueError("Embedder must be provided to search.")
            
        query_embedding = self.embedder.embed_queries([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "chunk_id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "score": results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0,
                    **results['metadatas'][0][i]
                })
                
        return formatted_results
