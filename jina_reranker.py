# jina_reranker.py

"""
Simple Jina Reranker implementation as Cohere replacement.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JinaReranker:
    """
    Jina AI Reranker v2 - Drop-in replacement for Cohere reranker.
    """
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: Optional[str] = None,
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 1024,
        cache_dir: Optional[str] = "./models"
    ):
        """
        Initialize Jina Reranker.
        
        Args:
            model_name: Jina model to use
            device: Device to run on (cuda/cpu/mps/auto)
            use_fp16: Use half precision for faster inference
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            cache_dir: Directory to cache downloaded models
        """
        # Set device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        logger.info(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        logger.info(f"Loading Jina model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("✓ Jina reranker loaded successfully")
        
        # Performance tracking
        self.total_queries = 0
        self.total_time = 0
    
    def rerank(
        self,
        model: Optional[str] = None,  # For Cohere compatibility (ignored)
        query: str = None,
        documents: Union[List[str], List[Dict]] = None,
        top_n: Optional[int] = None  # Cohere uses top_n instead of top_k
    ):
        """
        Rerank documents - Cohere-compatible interface.
        
        Args:
            model: Ignored (for Cohere compatibility)
            query: Query string
            documents: List of documents (strings or dicts)
            top_n: Number of top documents to return
            
        Returns:
            Object with .results attribute (Cohere format)
        """
        start_time = time.time()
        
        # Extract texts from documents
        if documents and isinstance(documents[0], dict):
            texts = [doc.get('text', doc.get('content', str(doc))) for doc in documents]
        else:
            texts = [str(doc) for doc in documents]
        
        # Calculate scores in batches
        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_scores = self._score_batch(query, batch_texts)
            scores.extend(batch_scores)
        
        # Create results
        results = []
        for idx, score in enumerate(scores):
            results.append(RerankResult(idx, float(score)))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply top_n if specified
        if top_n and top_n > 0:
            results = results[:top_n]
        
        # Update metrics
        elapsed = time.time() - start_time
        self.total_queries += 1
        self.total_time += elapsed
        
        logger.debug(f"Reranked {len(documents)} docs in {elapsed:.3f}s")
        
        # Return Cohere-compatible response
        return RerankResponse(results)
    
    def _score_batch(self, query: str, texts: List[str]) -> List[float]:
        """
        Score a batch of documents against query.
        
        Args:
            query: Query string
            texts: Batch of document texts
            
        Returns:
            List of scores
        """
        # Prepare pairs for Jina model
        pairs = [[query, text] for text in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get scores
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        return scores.tolist()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        avg_time = self.total_time / max(self.total_queries, 1)
        return {
            "total_queries": self.total_queries,
            "total_time": self.total_time,
            "average_time_per_query": avg_time,
            "device": self.device
        }


class RerankResult:
    """Cohere-compatible result object."""
    def __init__(self, index: int, score: float):
        self.index = index
        self.relevance_score = score


class RerankResponse:
    """Cohere-compatible response object."""
    def __init__(self, results: List[RerankResult]):
        self.results = results