from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Union, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TextClassificationPipeline:
    """
    A pipeline for text classification using transformer models.
    
    Supports both zero-shot classification and fine-tuned models.
    """
    
    def __init__(self, model_name: Optional[str] = None, zero_shot: bool = False):
        """
        Initialize the text classification pipeline.
        
        Args:
            model_name: Name of the pre-trained model (None for default)
            zero_shot: Whether to use zero-shot classification
        """
        self.zero_shot = zero_shot
        
        if zero_shot:
            self.model_name = "facebook/bart-large-mnli" if model_name is None else model_name
            self.pipeline = pipeline("zero-shot-classification", model=self.model_name)
        else:
            self.model_name = "distilbert-base-uncased-finetuned-sst-2-english" if model_name is None else model_name
            self.pipeline = pipeline("text-classification", model=self.model_name)
        
        logger.info(f"Initialized text classification pipeline (zero_shot={zero_shot}) with model: {self.model_name}")
    
    def classify(self, text: str, labels: Optional[List[str]] = None, **kwargs) -> Dict:
        """
        Classify the input text.
        
        Args:
            text: Input text to classify
            labels: For zero-shot, the candidate labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with classification results
        """
        if not text.strip():
            logger.error("Empty text provided for classification")
            raise ValueError("Input text cannot be empty")
            
        try:
            if self.zero_shot:
                if not labels:
                    logger.error("Labels must be provided for zero-shot classification")
                    raise ValueError("Labels must be provided for zero-shot classification")
                return self.pipeline(text, labels, **kwargs)
            else:
                return self.pipeline(text, **kwargs)
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            raise RuntimeError(f"Classification failed: {str(e)}")
    
    def batch_classify(self, texts: List[str], labels: Optional[List[str]] = None, **kwargs) -> List[Dict]:
        """
        Classify a batch of texts.
        
        Args:
            texts: List of texts to classify
            labels: For zero-shot, the candidate labels
            **kwargs: Additional arguments
            
        Returns:
            List of classification results
        """
        return [self.classify(text, labels, **kwargs) for text in texts]

# Example usage
if __name__ == "__main__":
    # Initialize zero-shot pipeline
    zero_shot_classifier = TextClassificationPipeline(zero_shot=True)
    
    # Sample text and labels
    text = "The new movie was amazing with incredible performances and stunning visuals."
    labels = ["positive", "negative", "neutral"]
    
    # Classify
    result = zero_shot_classifier.classify(text, labels)
    print(f"Classification result: {result}")
    
    # Initialize regular classifier
    classifier = TextClassificationPipeline()
    result = classifier.classify(text)
    print(f"Regular classification: {result}")