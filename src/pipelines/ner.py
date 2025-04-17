from transformers import pipeline
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class NERPipeline:
    """
    Named Entity Recognition pipeline using transformer models.
    Supports entity extraction with configurable model selection.
    """

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize the NER pipeline.
        
        Args:
            model_name: Name of the pre-trained NER model
                        Default: "dslim/bert-base-NER" (good balance of speed/accuracy)
        """
        try:
            self.model_name = model_name
            self.pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple"  # Groups subword tokens
            )
            logger.info(f"Initialized NER pipeline with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize NER pipeline: {str(e)}")
            raise RuntimeError(f"NER pipeline initialization failed: {str(e)}")

    def extract_entities(self, text: str, return_type: str = "dict") -> List[Dict]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to analyze
            return_type: Format of results ("dict" or "string")
            
        Returns:
            List of entities (format depends on return_type)
            
        Example:
            Input: "Apple is based in Cupertino."
            Output (dict): [{'entity_group': 'ORG', 'word': 'Apple', ...}]
            Output (string): "ORG: Apple | LOC: Cupertino"
        """
        if not text.strip():
            logger.error("Empty text provided for NER")
            raise ValueError("Input text cannot be empty")

        try:
            entities = self.pipeline(text)
            
            if return_type == "string":
                return " | ".join(
                    f"{e['entity_group']}: {e['word']}" 
                    for e in entities
                )
            return entities
            
        except Exception as e:
            logger.error(f"NER failed: {str(e)}")
            raise RuntimeError(f"Entity extraction failed: {str(e)}")

    def batch_extract(self, texts: List[str], **kwargs) -> List[List[Dict]]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of input texts
            **kwargs: Arguments passed to extract_entities()
            
        Returns:
            List of entity lists for each text
        """
        return [self.extract_entities(text, **kwargs) for text in texts]

# Example usage
if __name__ == "__main__":
    # Initialize with default model
    ner = NERPipeline()
    
    # Sample text with entities
    sample_text = """
        Apple was founded in 1976 by Steve Jobs in Cupertino, California. 
        Microsoft is headquartered in Redmond, Washington.
    """
    
    # Extract entities (dictionary format)
    print("Entities (detailed):")
    print(ner.extract_entities(sample_text))
    
    # Extract entities (readable string format)
    print("\nEntities (readable):")
    print(ner.extract_entities(sample_text, return_type="string"))