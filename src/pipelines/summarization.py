from transformers import pipeline
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SummarizationPipeline:
    """
    A pipeline for text summarization using pre-trained transformer models.
    
    Supports both extractive and abstractive summarization.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarization pipeline.
        
        Args:
            model_name: Name of the pre-trained model to use for summarization.
        """
        self.model_name = model_name
        self.pipeline = pipeline("summarization", model=model_name)
        logger.info(f"Initialized summarization pipeline with model: {model_name}")
    
    def summarize(self, text: str, max_length: int = 130, min_length: int = 30, **kwargs) -> str:
        """
        Generate a summary of the input text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            **kwargs: Additional arguments to pass to the pipeline
            
        Returns:
            Generated summary
        """
        if not text.strip():
            logger.error("Empty text provided for summarization")
            raise ValueError("Input text cannot be empty")
            
        try:
            result = self.pipeline(text, max_length=max_length, min_length=min_length, **kwargs)
            return result[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise RuntimeError(f"Summarization failed: {str(e)}")
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[str]:
        """
        Summarize a batch of texts.
        
        Args:
            texts: List of texts to summarize
            **kwargs: Additional arguments to pass to the pipeline
            
        Returns:
            List of summaries
        """
        return [self.summarize(text, **kwargs) for text in texts]

# Example usage (for testing/documentation)
if __name__ == "__main__":
    # Initialize pipeline
    summarizer = SummarizationPipeline()
    
    # Generate synthetic article
    article = """
    The field of natural language processing has seen significant advances in recent years, 
    particularly with the advent of transformer models. These models, such as BERT and GPT, 
    have revolutionized how machines understand human language. They use attention mechanisms 
    to process words in relation to all other words in a sentence, rather than one-by-one in order.
    This allows for much better understanding of context and nuance in language.
    """
    
    # Generate summary
    summary = summarizer.summarize(article)
    print(f"Summary: {summary}")