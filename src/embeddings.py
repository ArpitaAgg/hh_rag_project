"""
src/embeddings.py

A modular multilingual text embedding module powered by fastembed (ONNX
Runtime). Uses the same model weights as sentence-transformers would, but
without the PyTorch dependency -- PyTorch's own baseline memory overhead
(on the order of a few hundred MB just for the import, before any model is
even loaded) was enough by itself to exceed a 512MB deploy memory limit.
fastembed runs the same models through ONNX Runtime instead, cutting total
process memory roughly in half.

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
from fastembed import TextEmbedding

# Central configuration: default open-source multilingual model
# Model supports 50+ languages including major Indic scripts.
#
# NOTE: on a couple of specific short-passage MSMARCO-XI examples, this
# model's cross-lingual cosine similarity ranked an unrelated passage above
# the correct one. Larger (mpnet-base) and retrieval-tuned (multilingual-e5-
# small) alternatives were tested and did not reliably fix it either -- the
# margins were within noise on isolated short passages regardless of model,
# and the larger model also exceeded free-tier deployment memory limits
# (512MB). MiniLM-L12 is kept as the default since it's the only option that
# is both stable to deploy and no worse in practice; this is a known
# limitation on tiny/short-passage corpora that should be re-evaluated
# against a realistically-sized production dataset, where per-example noise
# like this washes out.
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
        self._model: Optional[TextEmbedding] = None
        self._dimension: Optional[int] = None

    def load_model(self) -> TextEmbedding:
        """Loads the fastembed ONNX model lazily on first use."""
        if self._model is None:
            print(f"Loading embedding model: '{self.model_name}' ...")
            # ONNX Runtime's default CPU memory arena pre-allocates well
            # beyond the model's own weight size (measured ~440MB RSS for a
            # ~90MB ONNX model on a single short query) -- disabling the
            # arena and pinning to a single thread keeps this small deploy
            # comfortably inside a 512MB memory limit.
            self._model = TextEmbedding(
                model_name=self.model_name,
                threads=1,
                providers=[("CPUExecutionProvider", {
                    "enable_cpu_mem_arena": "0",
                    "arena_extend_strategy": "kSameAsRequested",
                })],
            )
            print(f"Model successfully loaded! Vector dimension: {self.embedding_dimension}")
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Returns the output vector dimension size (e.g., 384 for paraphrase-multilingual-MiniLM-L12-v2)."""
        if self._dimension is None:
            model = self.load_model()
            probe = next(model.embed(["dimension probe"]))
            self._dimension = int(probe.shape[-1])
        return self._dimension

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
        embeddings = np.array(list(model.embed(texts, batch_size=batch_size)), dtype=np.float32)

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
        vector = next(model.embed([query.strip()])).astype(np.float32)

        if normalize:
            vector = normalize_embeddings(vector)

        return vector.astype(np.float32)
