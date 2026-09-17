import re
import unicodedata

class TextCleaner:
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Applies text cleaning and normalization:
        1. Unicode normalization (NFKC)
        2. Whitespace normalization
        3. Removes non-printable characters
        """
        if not text:
            return ""
            
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters (except standard whitespace)
        text = "".join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()
