import json
import os
from typing import List, Dict, Any
from datasets import load_dataset

class DatasetLoader:
    def __init__(self, dataset_name: str = "hotpotqa_distractor", cache_dir: str = "data/raw"):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_hotpotqa(self, split: str = "validation", max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Loads the HotpotQA distractor setting dataset.
        For MVP, we limit samples if max_samples is provided.
        """
        print(f"Loading HotpotQA (distractor) split: {split}...")
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, cache_dir=self.cache_dir, trust_remote_code=True)
        
        if max_samples:
            print(f"Limiting to {max_samples} samples.")
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        records = []
        for item in dataset:
            record = {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item["level"],
                "context": []
            }
            # HotpotQA context format: { 'title': [...], 'sentences': [...] }
            # Convert to list of dicts for easier processing
            titles = item["context"]["title"]
            sentences = item["context"]["sentences"]
            for title, sents in zip(titles, sentences):
                record["context"].append({
                    "title": title,
                    "sentences": sents
                })
            records.append(record)
            
        return records

    def load_dataset(self, split: str = "validation", max_samples: int = None) -> List[Dict[str, Any]]:
        if self.dataset_name == "hotpotqa_distractor":
            return self.load_hotpotqa(split, max_samples)
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} is not yet supported.")
