"""
src/embeddings.py

A modular multilingual text embedding module powered by Sentence Transformers.

--------------------------------------------------------------------------------
CONCEPTUAL BEGINNER GUIDE:

1. What is an Embedding?
   An embedding is a numerical representation (a vector of floating-point numbers)
   that captures the semantic meaning of a text passage or question.
   Instead of matching exact keywords, embeddings allow computers to compare
   concepts: two sentences with similar meanings (even in different languages)
   will produce vectors that are close together in vector space.

2. Why do we use Multilingual Embeddings?
   Our dataset (MSMARCO-XI) contains text in both English and multiple Indic languages
   (Assamese, Hindi, Bengali, etc.). Multilingual models map sentences across different
   languages into a unified, shared vector space. This means an English query can find
   relevant Indic passages (or vice-versa) based on meaning.

3. Why do we Normalize Embeddings?
   Normalization scales each vector so its length (L2 norm) equals exactly 1.0.
   When vectors are normalized, computing Cosine Similarity (semantic similarity)
   reduces to a simple, ultra-fast Dot Product between vectors.
--------------------------------------------------------------------------------
"""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer

# Central configuration: default open-source multilingual model
# Model supports 50+ languages including major Indic scripts.
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalizes a 2D numpy array of embeddings so that each vector has an L2 norm (length) of 1.0.
    
    Formula: v_normalized = v / ||v||_2
    """
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        return embeddings if norm == 0 else embeddings / norm
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Prevent division by zero for empty/zero vectors
    norms[norms == 0] = 1.0
    return embeddings / norms


class MultilingualEmbedder:
    """
    Reusable embedder class to load models and generate normalized embeddings for text chunks and queries.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    def load_model(self) -> SentenceTransformer:
        """Loads the Sentence Transformer model lazily on first use."""
        if self._model is None:
            print(f"Loading embedding model: '{self.model_name}' ...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Model successfully loaded! Vector dimension: {self.embedding_dimension}")
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Returns the output vector dimension size (e.g., 384 for paraphrase-multilingual-MiniLM-L12-v2)."""
        model = self.load_model()
        if hasattr(model, 'get_embedding_dimension'):
            dim = model.get_embedding_dimension()
        else:
            dim = model.get_sentence_embedding_dimension()
        return dim if dim is not None else 384

    def embed_texts(self, texts: List[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        """
        Generates dense vector embeddings for a list of text strings (chunks or passages).
        
        Args:
            texts: List of text strings to embed.
            normalize: Whether to L2-normalize vectors to unit length.
            batch_size: Number of texts processed per forward pass.
            
        Returns:
            2D numpy array of shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        model = self.load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        if normalize:
            embeddings = normalize_embeddings(embeddings)

        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generates a 1D vector embedding for a single user search query.
        
        Args:
            query: User's question or search query string.
            normalize: Whether to L2-normalize the vector.
            
        Returns:
            1D numpy array of shape (embedding_dimension,)
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        model = self.load_model()
        vector = model.encode(query.strip(), convert_to_numpy=True)

        if normalize:
            vector = normalize_embeddings(vector)

        return vector.astype(np.float32)
