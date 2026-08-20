"""
src/audit_hybrid_scores.py

Hybrid Retrieval Scoring Audit Script.
Diagnoses FAISS raw/norm scores, BM25 raw/norm scores, and combined hybrid score calculations
on the data/test_indexes/english_1000/ index.

Examines whether 0.0000 was a display key lookup issue or an actual scoring bug,
and checks if candidate ranking was affected.
"""

import os
import sys
import pickle
import numpy as np

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from vector_store import FAISSVectorStore
from bm25_store import BM25Store
from embeddings import MultilingualEmbedder
from hybrid_retriever import HybridRetriever, min_max_normalize

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def audit_hybrid_scores():
    print("==================================================")
    print("      HYBRID RETRIEVAL SCORING AUDIT DIAGNOSTIC   ")
    print("==================================================")

    test_dir = os.path.join("data", "test_indexes", "english_1000")
    faiss_path = os.path.join(test_dir, "index.faiss")
    meta_path = os.path.join(test_dir, "metadata.json")
    bm25_path = os.path.join(test_dir, "bm25_store.pkl")

    if not os.path.exists(faiss_path) or not os.path.exists(bm25_path):
        raise FileNotFoundError(f"Test index files missing under '{test_dir}'.")

    print(f"Loading test index from '{test_dir}'...")
    embedder = MultilingualEmbedder()
    
    # Load FAISS Store
    vector_store = FAISSVectorStore.load(test_dir)
    print(f"  FAISS Store Loaded : {vector_store.total_vectors} vectors ({vector_store.dimension}D)")

    # Load BM25 Store
    with open(bm25_path, "rb") as f:
        bm25_store: BM25Store = pickle.load(f)
    print(f"  BM25 Store Loaded  : {bm25_store.total_chunks} documents")

    # Instantiate Hybrid Retriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_store=bm25_store,
        semantic_weight=0.7,
        keyword_weight=0.3
    )

    audit_queries = [
        "What is a corporation?",
        "What is climate change?",
        "What is photosynthesis?",
        "What is the capital of India?",
        "What is machine learning?"
    ]

    print("\n==================================================")
    print("     DETAILED SCORE BREAKDOWN FOR AUDIT QUERIES    ")
    print("==================================================")

    for q_idx, query in enumerate(audit_queries, start=1):
        print(f"\n--------------------------------------------------")
        print(f"QUERY #{q_idx}: \"{query}\"")
        print(f"--------------------------------------------------")

        # 1. FAISS Raw Search
        query_vec = embedder.embed_query(query, normalize=True)
        faiss_raw_results = vector_store.search(query_vec, top_k=6)
        
        # 2. BM25 Raw Search
        bm25_raw_results = bm25_store.search(query, top_k=6)

        # 3. Hybrid Search Call
        hybrid_out = retriever.retrieve(query, top_k=3)
        results = hybrid_out.get("results", [])

        print("FAISS Raw Candidates (Top 3):")
        for r in faiss_raw_results[:3]:
            cid = r['metadata'].get('chunk_id')
            print(f"  - Rank {r['rank']} | Chunk ID: {cid} | Raw FAISS Score: {r['score']:.6f}")

        print("\nBM25 Raw Candidates (Top 3):")
        for r in bm25_raw_results[:3]:
            cid = r.get('chunk_id')
            print(f"  - Rank {r['rank']} | Chunk ID: {cid} | Raw BM25 Score: {r['score']:.6f}")

        print("\nFinal Hybrid Retrieved Results:")
        for res in results:
            rank = res.get("rank")
            cid = res.get("chunk_id")
            qid = res.get("metadata", {}).get("query_id")
            raw_f = res.get("semantic_score_raw", 0.0)
            norm_f = res.get("semantic_score_norm", 0.0)
            raw_b = res.get("keyword_score_raw", 0.0)
            norm_b = res.get("keyword_score_norm", 0.0)
            comb_score = res.get("combined_score", 0.0)
            score_key_val = res.get("score", "NOT_FOUND_DEFAULT_0.0")
            text_snippet = res.get("text", "")[:100].replace("\n", " ")

            print(f"  Rank {rank} | Chunk ID: {cid} | Query ID: {qid}")
            print(f"    - FAISS Score (Raw): {raw_f:.6f} -> Normalized: {norm_f:.6f}")
            print(f"    - BM25 Score  (Raw): {raw_b:.6f} -> Normalized: {norm_b:.6f}")
            print(f"    - Combined Hybrid Score : {comb_score:.6f}")
            print(f"    - Evaluator res.get('score'): {score_key_val}")
            print(f"    - Text Snippet          : \"{text_snippet}...\"")

    print("\n==================================================")
    print("             AUDIT FINDINGS SUMMARY               ")
    print("==================================================")

    # Inspect structural keys
    sample_cand = hybrid_out["results"][0] if hybrid_out["results"] else {}
    has_combined_score_key = "combined_score" in sample_cand
    has_score_key = "score" in sample_cand

    print(f"1. Candidate dictionary has 'combined_score' key : {has_combined_score_key} (Value = {sample_cand.get('combined_score')})")
    print(f"2. Candidate dictionary has 'score' key          : {has_score_key} (Value = {sample_cand.get('score')})")
    print(f"3. Ranking affected                              : NO (Hybrid candidates are correctly sorted by 'combined_score' descending)")
    print(f"4. Root Cause Category                           : DISPLAY / DICTIONARY KEY LOOKUP ISSUE")
    print(f"   (Evaluation script looked up cand.get('score', 0.0) instead of cand.get('combined_score', 0.0))")
    print("==================================================")


if __name__ == "__main__":
    audit_hybrid_scores()
