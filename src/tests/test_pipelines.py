import unittest
import logging
from src.pipelines import (
    SummarizationPipeline,
    TextClassificationPipeline,
    TranslationPipeline
)

logging.basicConfig(level=logging.ERROR)  # Reduce logging during tests

class TestSummarizationPipeline(unittest.TestCase):
    """Test cases for the SummarizationPipeline."""
    
    def setUp(self):
        """Initialize the pipeline for testing."""
        self.pipeline = SummarizationPipeline()
        self.sample_text = """
            The field of natural language processing has seen significant advances in recent years, 
            particularly with the advent of transformer models. These models, such as BERT and GPT, 
            have revolutionized how machines understand human language.
        """
    
    def test_summarize_basic(self):
        """Test basic summarization functionality."""
        summary = self.pipeline.summarize(self.sample_text)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
        self.assertTrue(len(summary) < len(self.sample_text))
    
    def test_empty_input(self):
        """Test handling of empty input."""
        with self.assertRaises(ValueError):
            self.pipeline.summarize("")
    
    def test_batch_summarize(self):
        """Test batch summarization."""
        texts = [self.sample_text, "Another sample text about machine learning."]
        summaries = self.pipeline.batch_summarize(texts)
        self.assertEqual(len(summaries), len(texts))
        for summary in summaries:
            self.assertIsInstance(summary, str)
            self.assertTrue(len(summary) > 0)

class TestTextClassificationPipeline(unittest.TestCase):
    """Test cases for the TextClassificationPipeline."""
    
    def setUp(self):
        """Initialize the pipeline for testing."""
        self.zero_shot_pipeline = TextClassificationPipeline(zero_shot=True)
        self.standard_pipeline = TextClassificationPipeline()
        self.sample_text = "This movie was absolutely fantastic!"
    
    def test_zero_shot_classification(self):
        """Test zero-shot classification."""
        labels = ["positive", "negative", "neutral"]
        result = self.zero_shot_pipeline.classify(self.sample_text, labels)
        
        self.assertIn("labels", result)
        self.assertIn("scores", result)
        self.assertEqual(len(result["labels"]), len(labels))
        self.assertEqual(len(result["scores"]), len(labels))
        
        # The sample text should be classified as positive
        self.assertEqual(result["labels"][0], "positive")
    
    def test_standard_classification(self):
        """Test standard classification."""
        result = self.standard_pipeline.classify(self.sample_text)
        self.assertIn("label", result)
        self.assertIn("score", result)
        self.assertTrue(result["score"] > 0.5)
    
    def test_missing_labels_zero_shot(self):
        """Test error handling when labels are missing in zero-shot mode."""
        with self.assertRaises(ValueError):
            self.zero_shot_pipeline.classify(self.sample_text)

class TestTranslationPipeline(unittest.TestCase):
    """Test cases for the TranslationPipeline."""
    
    def setUp(self):
        """Initialize the pipeline for testing."""
        self.pipeline = TranslationPipeline()
        self.sample_text = "Hello world, this is a test."
    
    def test_translation_basic(self):
        """Test basic translation functionality."""
        translation = self.pipeline.translate(self.sample_text)
        self.assertIsInstance(translation, str)
        self.assertTrue(len(translation) > 0)
        self.assertNotEqual(translation.lower(), self.sample_text.lower())
    
    def test_batch_translation(self):
        """Test batch translation."""
        texts = [self.sample_text, "Another sample text to translate."]
        translations = self.pipeline.batch_translate(texts)
        self.assertEqual(len(translations), len(texts))
        for translation in translations:
            self.assertIsInstance(translation, str)
            self.assertTrue(len(translation) > 0)

if __name__ == "__main__":
    unittest.main()