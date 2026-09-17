from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseGenerator(ABC):
    @abstractmethod
    def generate_answer(self, query: str, evidence: List[str], temperature: float = 0.0) -> Dict[str, Any]:
        """
        Generates an answer based on the query and evidence.
        Returns a dict containing:
        - answer: str
        - token_usage: dict
        - latency: float
        """
        pass

    @abstractmethod
    def decompose_claims(self, answer: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Decomposes an answer into atomic claims.
        Returns a dict containing claims.
        """
        pass

    @abstractmethod
    def regenerate_claim(self, claim: str, evidence: List[str], temperature: float = 0.0) -> Dict[str, Any]:
        """
        Regenerates a specific claim based on new evidence.
        """
        pass
