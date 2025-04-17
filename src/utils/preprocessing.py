import re
import string
import logging
from typing import List, Dict, Union, Optional, Callable
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import contractions
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Download NLTK resources (with error handling)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Advanced text preprocessing pipeline for NLP tasks with multiple configurable steps.
    
    Features:
    - HTML tag removal
    - Contraction expansion
    - Special character handling
    - Advanced tokenization options
    - Stemming/Lemmatization
    - Custom pipeline steps
    - Batch processing with progress bar
    
    Example:
    >>> preprocessor = TextPreprocessor(
            lowercase=True,
            remove_html=True,
            expand_contractions=True,
            remove_special_chars=True,
            lemmatize=True,
            custom_steps=[lambda text: re.sub(r'\d+', '', text)]
        )
    >>> processed_text = preprocessor.preprocess_text("<p>This is a sample text with numbers 123</p>")
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        expand_contractions: bool = True,
        remove_punct: bool = True,
        remove_special_chars: bool = True,
        remove_stopwords: bool = True,
        stopwords_lang: str = 'english',
        lemmatize: bool = True,
        stem: bool = False,
        min_token_length: int = 2,
        custom_steps: Optional[List[Callable[[str], str]]] = None,
        disable_progress_bar: bool = False
    ):
        """
        Initialize the text preprocessor with configuration options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_html: Remove HTML tags
            expand_contractions: Expand contractions (e.g., "don't" -> "do not")
            remove_punct: Remove punctuation
            remove_special_chars: Remove special characters
            remove_stopwords: Remove stopwords
            stopwords_lang: Language for stopwords
            lemmatize: Perform lemmatization
            stem: Perform stemming (overrides lemmatize if both are True)
            min_token_length: Minimum length of tokens to keep
            custom_steps: List of custom preprocessing functions
            disable_progress_bar: Disable progress bar for batch processing
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.expand_contractions = expand_contractions
        self.remove_punct = remove_punct
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(stopwords.words(stopwords_lang)) if remove_stopwords else set()
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_token_length = min_token_length
        self.custom_steps = custom_steps or []
        self.disable_progress_bar = disable_progress_bar
        
        # Initialize stemmer/lemmatizer
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = PorterStemmer() if stem else None
        
        # Pre-compile regex patterns for better performance
        self.punct_table = str.maketrans('', '', string.punctuation)
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def _apply_custom_steps(self, text: str) -> str:
        """Apply all custom preprocessing steps."""
        for step in self.custom_steps:
            text = step(text)
        return text
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags using BeautifulSoup."""
        return BeautifulSoup(text, 'html.parser').get_text()
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        return contractions.fix(text)
    
    def _tokenize_and_filter(self, text: str) -> List[str]:
        """Tokenize text and apply filtering."""
        tokens = word_tokenize(text)
        
        # Filter short tokens
        tokens = [token for token in tokens if len(token) >= self.min_token_length]
        
        # Filter stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stopwords]
        
        return tokens
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def _stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem tokens."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text: str, return_tokens: bool = False) -> Union[str, List[str]]:
        """
        Preprocess a single text document with all configured steps.
        
        Args:
            text: Input text to preprocess
            return_tokens: Whether to return tokens instead of joined string
            
        Returns:
            Processed text (as string or tokens)
        
        Raises:
            ValueError: If input text is empty after preprocessing
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("Received empty or non-string input")
            return [] if return_tokens else ""
        
        try:
            # Apply basic cleaning steps
            if self.remove_html:
                text = self._remove_html_tags(text)
            
            if self.expand_contractions:
                text = self._expand_contractions(text)
            
            if self.lowercase:
                text = text.lower()
            
            if self.remove_special_chars:
                text = self.special_char_pattern.sub(' ', text)
            
            if self.remove_punct:
                text = text.translate(self.punct_table)
            
            # Apply custom steps
            text = self._apply_custom_steps(text)
            
            # Normalize whitespace
            text = self.whitespace_pattern.sub(' ', text).strip()
            
            # Tokenize and filter
            tokens = self._tokenize_and_filter(text)
            
            # Apply stemming/lemmatization
            if self.stem and self.stemmer:
                tokens = self._stem_tokens(tokens)
            elif self.lemmatize and self.lemmatizer:
                tokens = self._lemmatize_tokens(tokens)
            
            if not tokens:
                raise ValueError("Text became empty after preprocessing")
            
            return tokens if return_tokens else ' '.join(tokens)
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}", exc_info=True)
            raise RuntimeError(f"Text preprocessing failed: {str(e)}")
    
    def batch_preprocess(
        self,
        texts: List[str],
        return_tokens: bool = False,
        n_jobs: int = 1
    ) -> List[Union[str, List[str]]]:
        """
        Preprocess a batch of texts with optional parallelization.
        
        Args:
            texts: List of texts to preprocess
            return_tokens: Whether to return tokens instead of joined strings
            n_jobs: Number of parallel jobs (currently supports 1)
            
        Returns:
            List of processed texts
        """
        if n_jobs != 1:
            logger.warning("Parallel processing not yet implemented, using sequential")
        
        processed_texts = []
        for text in tqdm(texts, disable=self.disable_progress_bar, desc="Preprocessing"):
            processed = self.preprocess_text(text, return_tokens=return_tokens)
            processed_texts.append(processed)
        
        return processed_texts

def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_html: bool = True,
    expand_contractions: bool = True,
    remove_punct: bool = True,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    return_tokens: bool = False
) -> Union[str, List[str]]:
    """
    Convenience function for one-off text preprocessing.
    
    Args:
        text: Input text
        lowercase: Whether to lowercase
        remove_html: Remove HTML tags
        expand_contractions: Expand contractions
        remove_punct: Remove punctuation
        remove_stopwords: Remove stopwords
        lemmatize: Lemmatize tokens
        return_tokens: Return tokens instead of joined string
        
    Returns:
        Processed text or tokens
    """
    preprocessor = TextPreprocessor(
        lowercase=lowercase,
        remove_html=remove_html,
        expand_contractions=expand_contractions,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        lemmatize=lemmatize
    )
    return preprocessor.preprocess_text(text, return_tokens=return_tokens)

def tokenize_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_token_length: int = 2
) -> List[str]:
    """
    Tokenize text with configurable preprocessing.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation
        remove_stopwords: Remove stopwords
        lemmatize: Lemmatize tokens
        min_token_length: Minimum token length to keep
        
    Returns:
        List of processed tokens
    """
    preprocessor = TextPreprocessor(
        lowercase=lowercase,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        lemmatize=lemmatize,
        min_token_length=min_token_length
    )
    return preprocessor.preprocess_text(text, return_tokens=True)

def create_preprocessing_pipeline(steps_config: List[Dict[str, Any]]) -> TextPreprocessor:
    """
    Create a preprocessing pipeline from a configuration dictionary.
    
    Args:
        steps_config: List of step configurations
            Example:
            [
                {"name": "lowercase", "params": {"enabled": True}},
                {"name": "remove_html", "params": {"enabled": True}},
                {"name": "custom", "params": {"function": lambda x: re.sub(r'\d+', '', x)}}
            ]
    
    Returns:
        Configured TextPreprocessor instance
    """
    kwargs = {}
    custom_steps = []
    
    for step in steps_config:
        name = step["name"]
        params = step.get("params", {})
        
        if name == "custom":
            custom_steps.append(params["function"])
        else:
            kwargs[name] = params.get("enabled", True)
    
    if custom_steps:
        kwargs["custom_steps"] = custom_steps
    
    return TextPreprocessor(**kwargs)

# Example usage
if __name__ == "__main__":
    # Sample text with various features
    sample_text = """
        <p>I've been working with NLP since 2020. The models' performance is amazing!
        Don't you think? Contact me at example@email.com or visit https://example.com</p>
    """
    
    print("Original text:", sample_text)
    
    # Basic preprocessing
    basic_processed = preprocess_text(sample_text)
    print("\nBasic preprocessing:", basic_processed)
    
    # Advanced preprocessing with tokens
    advanced_processed = preprocess_text(
        sample_text,
        remove_html=True,
        expand_contractions=True,
        remove_stopwords=True,
        lemmatize=True,
        return_tokens=True
    )
    print("\nAdvanced preprocessing (tokens):", advanced_processed)
    
    # Create pipeline from config
    pipeline_config = [
        {"name": "lowercase", "params": {"enabled": True}},
        {"name": "remove_html", "params": {"enabled": True}},
        {"name": "expand_contractions", "params": {"enabled": True}},
        {"name": "custom", "params": {"function": lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x)}},
        {"name": "custom", "params": {"function": lambda x: re.sub(r'\S+@\S+', '', x)}}
    ]
    
    custom_pipeline = create_preprocessing_pipeline(pipeline_config)
    custom_processed = custom_pipeline.preprocess_text(sample_text)
    print("\nCustom pipeline result:", custom_processed)