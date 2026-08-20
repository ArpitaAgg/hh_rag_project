"""
src/hybrid_retriever.py

A modular hybrid retriever that combines FAISS semantic search (dense retrieval)
and BM25 keyword search (sparse retrieval) with score normalization.

--------------------------------------------------------------------------------
BEGINNER EXPLANATION OF SCORE NORMALIZATION & HYBRID FUSION:

1. Why Score Normalization is Required:
   FAISS Cosine Similarity scores range strictly between -1.0 and 1.0 (typically 0.0 to 0.9).
   BM25 scores are unbounded non-negative term frequency counts (typically 0.0 to 15.0+).
   Directly adding raw FAISS and BM25 scores would cause BM25 to completely overpower FAISS.

2. Min-Max Normalization Method:
   For every query, we extract top candidates from both FAISS and BM25.
   We scale raw FAISS scores into [0, 1] range:
       Norm_FAISS = (Score - Min_FAISS) / (Max_FAISS - Min_FAISS)
       
   We scale raw BM25 scores into [0, 1] range:
       Norm_BM25  = (Score - Min_BM25) / (Max_BM25 - Min_BM25)

3. Weighted Combination Formula:
   Combined_Score = (semantic_weight * Norm_FAISS) + (keyword_weight * Norm_BM25)
   
   Default weights: 70% Semantic (FAISS) + 30% Keyword (BM25).
--------------------------------------------------------------------------------
"""

import time
from typing import List, Dict, Any, Optional
from vector_store import FAISSVectorStore
from embeddings import MultilingualEmbedder
from bm25_store import BM25Store


def min_max_normalize(scores_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Applies Min-Max scaling to a dictionary of {item_id: raw_score}.
    Rescales all values into the range [0.0, 1.0].
    """
    if not scores_dict:
        return {}

    min_val = min(scores_dict.values())
    max_val = max(scores_dict.values())

    if max_val == min_val:
        # If all candidates have identical score
        return {k: (1.0 if max_val > 0 else 0.0) for k in scores_dict}

    return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}


class HybridRetriever:
    """
    Combines FAISS vector search and BM25 keyword search with normalized score fusion.
    """
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: MultilingualEmbedder,
        bm25_store: BM25Store,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        if not (0.0 <= semantic_weight <= 1.0) or not (0.0 <= keyword_weight <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")

        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_store = bm25_store
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes hybrid retrieval combining FAISS and BM25.

        Args:
            query: User's search query string.
            top_k: Number of final ranked results to return.

        Returns:
            Dictionary containing final results, intermediate FAISS/BM25 results, and latency breakdown.
        """
        if not query or not query.strip():
            return {"query": query, "results": [], "faiss_results": [], "bm25_results": [], "latency_ms": {}}

        # 1. FAISS Semantic Search Timing
        t0 = time.time()
        query_vec = self.embedder.embed_query(query, normalize=True)
        faiss_candidates = self.vector_store.search(query_vec, top_k=top_k * 2)
        t1 = time.time()
        faiss_latency = (t1 - t0) * 1000

        # 2. BM25 Keyword Search Timing
        t2 = time.time()
        bm25_candidates = self.bm25_store.search(query, top_k=top_k * 2)
        t3 = time.time()
        bm25_latency = (t3 - t2) * 1000

        # 3. Candidate Fusion & Score Normalization Timing
        t4 = time.time()
        
        # Maps chunk_id -> metadata
        candidate_map: Dict[str, Dict[str, Any]] = {}
        raw_faiss_scores: Dict[str, float] = {}
        raw_bm25_scores: Dict[str, float] = {}

        # Collect FAISS candidates
        for match in faiss_candidates:
            meta = match["metadata"]
            cid = meta.get("chunk_id", f"chunk_{match['vector_id']}")
            candidate_map[cid] = meta
            raw_faiss_scores[cid] = match["score"]

        # Collect BM25 candidates
        for match in bm25_candidates:
            meta = match["metadata"]
            cid = meta.get("chunk_id", f"chunk_{match['vector_id']}")
            candidate_map[cid] = meta
            raw_bm25_scores[cid] = match["score"]

        # Ensure all candidates are present in both score dicts (fill missing with min score or 0.0)
        all_chunk_ids = list(candidate_map.keys())
        for cid in all_chunk_ids:
            if cid not in raw_faiss_scores:
                raw_faiss_scores[cid] = 0.0
            if cid not in raw_bm25_scores:
                raw_bm25_scores[cid] = 0.0

        # Apply Min-Max Normalization
        norm_faiss_scores = min_max_normalize(raw_faiss_scores)
        norm_bm25_scores = min_max_normalize(raw_bm25_scores)

        # Compute Combined Weighted Score
        hybrid_candidates = []
        for cid in all_chunk_ids:
            raw_f = raw_faiss_scores[cid]
            raw_b = raw_bm25_scores[cid]
            norm_f = norm_faiss_scores[cid]
            norm_b = norm_bm25_scores[cid]

            combined_score = (self.semantic_weight * norm_f) + (self.keyword_weight * norm_b)
            # For cross-lingual/Hinglish queries where BM25 may be 0, preserve high dense semantic vector score
            if raw_f >= 0.50:
                combined_score = max(raw_f, combined_score)

            meta = candidate_map[cid]
            hybrid_candidates.append({
                "chunk_id": cid,
                "combined_score": combined_score,
                "semantic_score_raw": raw_f,
                "semantic_score_norm": norm_f,
                "keyword_score_raw": raw_b,
                "keyword_score_norm": norm_b,
                "is_selected": meta.get("is_selected", False),
                "text": meta.get("text", ""),
                "metadata": meta
            })

        # Sort combined candidates by combined_score descending
        hybrid_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        final_results = hybrid_candidates[:top_k]

        # Add ranks
        for r, res in enumerate(final_results, start=1):
            res["rank"] = r

        t5 = time.time()
        fusion_latency = (t5 - t4) * 1000
        total_latency = (t5 - t0) * 1000

        return {
            "query": query,
            "results": final_results,
            "faiss_candidates": faiss_candidates,
            "bm25_candidates": bm25_candidates,
            "latency_ms": {
                "faiss": faiss_latency,
                "bm25": bm25_latency,
                "fusion": fusion_latency,
                "total": total_latency
            }
        }
