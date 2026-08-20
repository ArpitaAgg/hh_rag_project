"""
src/bm25_store.py

A multilingual-friendly BM25 keyword search module powered by rank_bm25.

--------------------------------------------------------------------------------
BEGINNER EXPLANATION:

1. What is BM25?
   BM25 (Best Matching 25) is a classic, highly effective keyword-based retrieval algorithm.
   It counts exact word matches between a query and document chunks, while adjusting for:
   - Term Frequency (TF): Words appearing more frequently in a chunk receive higher weight.
   - Inverse Document Frequency (IDF): Rare words receive higher weight than common words.
   - Document Length Normalization: Penalizes overly long documents.

2. Why Multilingual Tokenization?
   Indic scripts (Assamese, Bengali, Hindi, etc.) do not always use English punctuation.
   Our tokenizer uses Unicode-aware regex matching (re.UNICODE) to extract words in any language safely.
--------------------------------------------------------------------------------
"""

import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


def multilingual_tokenize(text: str) -> List[str]:
    """
    Tokenizes multilingual text (English, Indic scripts, etc.) into lowercased word tokens.
    Uses Unicode word boundaries (re.UNICODE) to handle non-ASCII scripts safely.
    """
    if not text or not text.strip():
        return []
    
    # Extract word tokens (alphanumeric + all Unicode script characters)
    tokens = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
    return [t for t in tokens if len(t) > 0]


class BM25Store:
    """
    In-memory BM25 index for keyword search across text chunks.
    """
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Indexes a list of text chunks for BM25 keyword search.
        
        Args:
            chunks: List of chunk dictionaries (must contain 'text' field).
        """
        if not chunks:
            print("Warning: No chunks provided for BM25 indexing.")
            return

        self.chunks = chunks
        self.corpus_tokens = [multilingual_tokenize(c.get("text", "")) for c in chunks]
        
        # Build BM25Okapi index
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"BM25 Index successfully built for {len(self.chunks)} text chunks.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index using a text query.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
            
        Returns:
            List of result dictionaries containing rank, score, chunk_id, and metadata.
        """
        if not query or not query.strip() or self.total_chunks == 0 or self.bm25 is None:
            return []

        query_tokens = multilingual_tokenize(query)
        if not query_tokens:
            return []

        # Get raw BM25 scores for all corpus documents
        raw_scores = self.bm25.get_scores(query_tokens)

        # Get indices of top_k highest scores
        top_k_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:min(top_k, self.total_chunks)]

        results = []
        for rank, idx in enumerate(top_k_indices, start=1):
            score = float(raw_scores[idx])
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": score,
                "vector_id": idx,
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "metadata": chunk
            })

        return results
