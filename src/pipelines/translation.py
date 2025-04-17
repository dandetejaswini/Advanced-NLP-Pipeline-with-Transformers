from transformers import pipeline
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TranslationPipeline:
    """
    A pipeline for machine translation between multiple languages.
    
    Supports both generic and specialized translation models.
    """
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-de"):
        """
        Initialize the translation pipeline.
        
        Args:
            model_name: Name of the pre-trained translation model
        """
        self.model_name = model_name
        self.pipeline = pipeline("translation", model=model_name)
        logger.info(f"Initialized translation pipeline with model: {model_name}")
    
    def translate(self, text: str, src_lang: Optional[str] = None, tgt_lang: Optional[str] = None, **kwargs) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Input text to translate
            src_lang: Source language code (optional)
            tgt_lang: Target language code (optional)
            **kwargs: Additional arguments
            
        Returns:
            Translated text
        """
        if not text.strip():
            logger.error("Empty text provided for translation")
            raise ValueError("Input text cannot be empty")
            
        try:
            # For models that support multiple languages, we can specify src/tgt
            if src_lang and tgt_lang:
                kwargs.update({"src_lang": src_lang, "tgt_lang": tgt_lang})
            
            result = self.pipeline(text, **kwargs)
            return result[0]['translation_text']
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Translation failed: {str(e)}")
    
    def batch_translate(self, texts: List[str], **kwargs) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            **kwargs: Additional arguments
            
        Returns:
            List of translations
        """
        return [self.translate(text, **kwargs) for text in texts]

# Example usage
if __name__ == "__main__":
    # Initialize pipeline (English to German by default)
    translator = TranslationPipeline()
    
    # Sample text
    text = "Hello world, this is a test of the translation system."
    
    # Translate
    translation = translator.translate(text)
    print(f"Translation: {translation}")
    
    # Example with different language pair
    fr_de_translator = TranslationPipeline("Helsinki-NLP/opus-mt-fr-de")
    french_text = "Bonjour le monde, ceci est un test du système de traduction."
    german_translation = fr_de_translator.translate(french_text)
    print(f"French to German: {german_translation}")