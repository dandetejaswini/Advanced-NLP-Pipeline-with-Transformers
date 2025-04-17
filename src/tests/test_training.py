import unittest
import tempfile
import shutil
import logging
from pathlib import Path
from src.training.train_classifier import train_text_classifier
from src.training.train_summarizer import train_summarization_model

# Disable excessive logging during tests
logging.basicConfig(level=logging.CRITICAL)

class TestTraining(unittest.TestCase):
    """Test training workflows with synthetic data."""
    
    @classmethod
    def setUpClass(cls):
        """Create temp directory for test models."""
        cls.test_dir = tempfile.mkdtemp()
        cls.classifier_dir = Path(cls.test_dir) / "classifier"
        cls.summarizer_dir = Path(cls.test_dir) / "summarizer"
        
    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory."""
        shutil.rmtree(cls.test_dir)
    
    def test_classifier_training(self):
        """Test text classifier training with synthetic data."""
        try:
            # Train with minimal settings (1 epoch)
            model, tokenizer = train_text_classifier(
                model_name="distilbert-base-uncased",
                output_dir=self.classifier_dir,
                num_epochs=1,
                batch_size=8,
                logging_steps=0  # Disable progress logging
            )
            
            # Verify outputs
            self.assertTrue(self.classifier_dir.exists())
            self.assertTrue((self.classifier_dir / "pytorch_model.bin").exists())
            self.assertTrue((self.classifier_dir / "config.json").exists())
            
        except Exception as e:
            self.fail(f"Classifier training failed: {str(e)}")
    
    def test_summarizer_training(self):
        """Test summarization model training with synthetic data."""
        try:
            # Train with minimal settings
            model, tokenizer = train_summarization_model(
                model_name="facebook/bart-large-cnn",
                output_dir=self.summarizer_dir,
                num_epochs=1,
                batch_size=4,
                logging_steps=0
            )
            
            # Verify outputs
            self.assertTrue(self.summarizer_dir.exists())
            self.assertTrue((self.summarizer_dir / "pytorch_model.bin").exists())
            
        except Exception as e:
            self.fail(f"Summarizer training failed: {str(e)}")
    
    def test_training_failure_handling(self):
        """Test invalid model handling."""
        with self.assertRaises(Exception):
            train_text_classifier(
                model_name="invalid-model-name",
                output_dir=self.classifier_dir,
                num_epochs=1
            )

if __name__ == "__main__":
    unittest.main()