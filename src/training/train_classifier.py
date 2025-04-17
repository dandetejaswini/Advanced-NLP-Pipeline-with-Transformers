import logging
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

logger = logging.getLogger(__name__)

def generate_synthetic_data(num_samples: int = 1000) -> DatasetDict:
    """
    Generate synthetic training data for text classification.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    # Generate synthetic reviews and labels
    positive_reviews = [
        f"This product is amazing! I've never used anything like it before. Sample {i}"
        for i in range(num_samples // 2)
    ]
    negative_reviews = [
        f"I'm very disappointed with this purchase. It broke after {i} days."
        for i in range(num_samples // 2)
    ]
    
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    
    # Split dataset
    train_testvalid = dataset.train_test_split(test_size=0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    
    return DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })

def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for evaluation.
    
    Args:
        p: EvalPrediction object
        
    Returns:
        Dictionary of metrics
    """
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted")
    }

def train_text_classifier(
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./models/fine_tuned/classifier",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    logging_steps: int = 10,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    **kwargs
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Fine-tune a transformer model for text classification.
    
    Args:
        model_name: Name of pre-trained model to fine-tune
        output_dir: Directory to save the fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Batch size for training/evaluation
        learning_rate: Learning rate
        logging_steps: Number of steps between logging
        evaluation_strategy: When to evaluate ("epoch", "steps", or "no")
        save_strategy: When to save checkpoints
        load_best_model_at_end: Whether to load best model at end of training
        **kwargs: Additional training arguments
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Generate synthetic data
    logger.info("Generating synthetic training data...")
    dataset = generate_synthetic_data()
    
    # Initialize tokenizer and model
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Binary classification for our synthetic data
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        **kwargs
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    logger.info(f"Test results: {test_results}")
    
    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, tokenizer = train_text_classifier()
    print("Training completed successfully!")