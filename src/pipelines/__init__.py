"""
Pipeline modules for various NLP tasks.

This package contains ready-to-use pipelines for:
- Text summarization
- Text classification
- Machine translation
"""

from .summarization import SummarizationPipeline
from .text_classification import TextClassificationPipeline
from .translation import TranslationPipeline

__all__ = [
    'SummarizationPipeline',
    'TextClassificationPipeline',
    'TranslationPipeline'
]

from .ner import NERPipeline
__all__ = [..., 'NERPipeline']