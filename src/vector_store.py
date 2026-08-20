"""
src/vector_store.py

A modular FAISS-based vector database wrapper for storing text chunk embeddings
and executing fast semantic similarity retrieval.

--------------------------------------------------------------------------------
BEGINNER EXPLANATION:

1. What is FAISS?
   FAISS (Facebook AI Similarity Search) is an open-source library built by Meta
   specifically for fast dense-vector search. It allows us to search millions of
   vector embeddings in milliseconds.

2. What is a Vector Index?
   A vector index is an in-memory data structure optimized to organize mathematical
   vectors so we can efficiently find the "nearest neighbors" (most similar texts)
   for a given search query vector.

3. Why do we use Inner Product (IndexFlatIP) with Normalized Vectors?
   When vectors are L2-normalized (length = 1.0), their Inner Product (Dot Product)
   is mathematically identical to Cosine Similarity.
   A score of 1.0 means perfect semantic match; 0.0 means unrelated.

4. ID Mapping:
   FAISS stores vectors by numerical IDs (0, 1, 2, ... N-1). We maintain a parallel
   metadata list where `metadata[i]` corresponds to vector ID `i`.
--------------------------------------------------------------------------------
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional


class FAISSVectorStore:
    """
    Wrapper class around FAISS IndexFlatIP to manage vector indexing, search, and persistence.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # IndexFlatIP uses Inner Product (Dot Product)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store: List[Dict[str, Any]] = []

    @property
    def total_vectors(self) -> int:
        """Returns the total number of vectors currently stored in the index."""
        return self.index.ntotal

    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Adds normalized vector embeddings and their corresponding metadata to the index.

        Args:
            embeddings: 2D numpy array of shape (N, dimension), dtype float32
            metadata_list: List of N metadata dictionaries corresponding to each vector
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError(f"Embeddings count ({len(embeddings)}) does not match metadata count ({len(metadata_list)}).")

        if len(embeddings) == 0:
            print("Warning: Empty embeddings array provided. Nothing added.")
            return

        # Ensure correct shape, dimension, and float32 type for FAISS
        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        if embeddings_np.ndim == 1:
            embeddings_np = np.expand_dims(embeddings_np, axis=0)

        if embeddings_np.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {embeddings_np.shape[1]}")

        # Add vectors to FAISS index
        self.index.add(embeddings_np)
        # Store corresponding metadata
        self.metadata_store.extend(metadata_list)
        print(f"Successfully added {len(embeddings_np)} vectors. Index total: {self.total_vectors}")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index for the top_k most similar vectors to the query_vector.

        Args:
            query_vector: 1D or 2D numpy array of shape (dimension,) or (1, dimension)
            top_k: Number of nearest neighbors to return

        Returns:
            List of dictionaries containing score, vector_id, and chunk metadata.
        """
        if self.total_vectors == 0:
            print("Warning: FAISS index is empty. Returning 0 results.")
            return []

        # Prepare query vector
        query_np = np.asarray(query_vector, dtype=np.float32)
        if query_np.ndim == 1:
            query_np = np.expand_dims(query_np, axis=0)

        if query_np.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {query_np.shape[1]}")

        # Limit top_k to actual total stored vectors
        k = min(top_k, self.total_vectors)

        # Execute search in FAISS
        # scores matrix shape: (1, k), indices matrix shape: (1, k)
        scores, indices = self.index.search(query_np, k)

        results = []
        for rank in range(k):
            vector_id = int(indices[0][rank])
            similarity_score = float(scores[0][rank])

            if vector_id != -1 and vector_id < len(self.metadata_store):
                meta = self.metadata_store[vector_id]
                results.append({
                    "rank": rank + 1,
                    "score": similarity_score,
                    "vector_id": vector_id,
                    "metadata": meta
                })

        return results

    def save(self, dir_path: str):
        """
        Saves the FAISS binary index and metadata JSON to disk.
        """
        os.makedirs(dir_path, exist_ok=True)
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "metadata.json")

        # Save FAISS index
        faiss.write_index(self.index, index_file)

        # Save metadata store as JSON
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)

        print(f"Saved FAISS index ({self.total_vectors} vectors) to '{dir_path}'.")

    @classmethod
    def load(cls, dir_path: str) -> "FAISSVectorStore":
        """
        Loads a FAISS index and metadata store from disk.
        """
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "metadata.json")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"Cannot load index: missing '{index_file}' or '{meta_file}'")

        faiss_index = faiss.read_index(index_file)
        dimension = faiss_index.d

        store = cls(dimension=dimension)
        store.index = faiss_index

        with open(meta_file, "r", encoding="utf-8") as f:
            store.metadata_store = json.load(f)

        print(f"Successfully loaded FAISS index ({store.total_vectors} vectors) from '{dir_path}'.")
        return store
