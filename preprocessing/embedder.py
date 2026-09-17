from typing import List
import torch
from transformers import AutoModel, AutoTokenizer

class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        BGE requires "Represent this sentence for searching relevant passages: " prefix for queries.
        """
        instruction = "Represent this sentence for searching relevant passages: "
        prefixed_queries = [instruction + q for q in queries]
        return self._embed(prefixed_queries)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Documents are embedded without the prefix.
        """
        return self._embed(documents)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
            
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy().tolist()
