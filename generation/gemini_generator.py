import os
import json
import time
from typing import List, Dict, Any
from google import genai
from google.genai import types
from fg_arag.generation.base import BaseGenerator
from fg_arag.generation.prompts import QA_PROMPT, CLAIM_DECOMPOSITION_PROMPT, REGENERATE_CLAIM_PROMPT

class GeminiGenerator(BaseGenerator):
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        # Assumes GEMINI_API_KEY is in environment
        self.client = genai.Client()

    def generate_answer(self, query: str, evidence: List[str], temperature: float = 0.0) -> Dict[str, Any]:
        evidence_text = "\n\n".join(evidence)
        prompt = QA_PROMPT.format(evidence=evidence_text, query=query)
        
        start_time = time.time()
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
            )
        )
        
        latency = time.time() - start_time
        
        return {
            "answer": response.text.strip(),
            "latency": latency,
            "model": self.model_name,
            "query": query
        }

    def decompose_claims(self, answer: str, temperature: float = 0.0) -> Dict[str, Any]:
        prompt = CLAIM_DECOMPOSITION_PROMPT.format(answer=answer)
        
        start_time = time.time()
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            )
        )
        
        latency = time.time() - start_time
        
        try:
            result = json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback for repair logic could go here
            result = {"claims": []}
            
        return {
            "result": result,
            "latency": latency,
            "model": self.model_name
        }

    def regenerate_claim(self, claim: str, evidence: List[str], temperature: float = 0.0) -> Dict[str, Any]:
        evidence_text = "\n\n".join(evidence)
        prompt = REGENERATE_CLAIM_PROMPT.format(claim=claim, evidence=evidence_text)
        
        start_time = time.time()
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            )
        )
        
        latency = time.time() - start_time
        
        try:
            result = json.loads(response.text)
        except json.JSONDecodeError:
            result = {"status": "error", "claim": None}
            
        return {
            "result": result,
            "latency": latency,
            "model": self.model_name
        }
