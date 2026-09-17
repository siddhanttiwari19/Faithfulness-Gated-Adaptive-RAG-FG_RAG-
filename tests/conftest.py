import pytest
import os

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    monkeypatch.setenv("GEMINI_API_KEY", "mock_api_key_for_testing")
