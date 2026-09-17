import pytest
from fg_arag.preprocessing.cleaner import TextCleaner
from fg_arag.preprocessing.chunker import TokenAwareChunker

def test_text_cleaner():
    raw_text = "This is a\n\ntest\twith   multiple   spaces."
    cleaned = TextCleaner.normalize_text(raw_text)
    assert cleaned == "This is a test with multiple spaces."

def test_chunker_basic():
    # Use a small chunk size for testing
    chunker = TokenAwareChunker(chunk_size=10, chunk_overlap=2)
    # create a mock string of about 20 words
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15 word16 word17 word18 word19 word20"
    
    chunks = chunker.chunk_text(text)
    
    # We expect some chunks
    assert len(chunks) > 1
    # Check that they overlap conceptually
    
def test_chunker_document_processing():
    chunker = TokenAwareChunker(chunk_size=5, chunk_overlap=1)
    text = "This is a test document with some words."
    processed = chunker.process_document(doc_id="doc_123", title="Test Doc", text=text)
    
    assert len(processed) > 0
    assert processed[0]["document_id"] == "doc_123"
    assert processed[0]["document_title"] == "Test Doc"
    assert "token_count" in processed[0]
