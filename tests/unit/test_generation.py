import pytest
import json
from fg_arag.generation.gemini_generator import GeminiGenerator
from fg_arag.generation.base import BaseGenerator

class MockGeminiClient:
    class Models:
        def generate_content(self, model, contents, config=None):
            class MockResponse:
                def __init__(self, text):
                    self.text = text
            
            if "Decompose the following answer" in contents:
                # Mock claim decomposition
                return MockResponse('{"claims": [{"claim_id": "c1", "text": "test claim", "source_sentence": "test"}]}')
            elif "Given the original claim and new evidence" in contents:
                # Mock regenerate claim
                return MockResponse('{"status": "success", "claim": "new test claim"}')
            else:
                # Mock QA answer
                return MockResponse("This is a mock answer based on evidence.")
    
    def __init__(self):
        self.models = self.Models()

def test_gemini_generator(monkeypatch):
    # Mock genai.Client
    import google.genai as genai
    monkeypatch.setattr(genai, "Client", lambda: MockGeminiClient())
    
    generator = GeminiGenerator()
    
    # Test generate answer
    ans_res = generator.generate_answer("what is it?", ["evidence 1", "evidence 2"])
    assert ans_res["answer"] == "This is a mock answer based on evidence."
    assert "latency" in ans_res
    
    # Test decompose claims
    claims_res = generator.decompose_claims("This is a mock answer.")
    assert len(claims_res["result"]["claims"]) == 1
    assert claims_res["result"]["claims"][0]["claim_id"] == "c1"
    
    # Test regenerate claim
    regen_res = generator.regenerate_claim("test claim", ["new evidence"])
    assert regen_res["result"]["status"] == "success"
    assert regen_res["result"]["claim"] == "new test claim"
