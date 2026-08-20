"""
src/test_faiss.py

Test runner and latency measurement script for the local FAISS semantic retrieval system.
Loads the index built in data/indexes/, runs sample test queries (Indic & English),
measures latency, validates top-3 similarity search results, and saves the output report.
"""

import os
import sys
import time
import numpy as np

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from embeddings import MultilingualEmbedder
from vector_store import FAISSVectorStore

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_faiss_tests():
    print("==================================================")
    print("      FAISS SEMANTIC RETRIEVAL TEST & EVALUATION  ")
    print("==================================================")

    index_dir = os.path.join("data", "indexes")
    if not os.path.exists(os.path.join(index_dir, "index.faiss")):
        print(f"Error: FAISS index not found in '{index_dir}'. Running index build first...")
        from build_faiss_index import build_index
        build_index()

    # 1. Load FAISS Vector Store
    vector_store = FAISSVectorStore.load(index_dir)
    print(f"Loaded FAISS Index. Total indexed chunks: {vector_store.total_vectors}")
    print(f"Index Type: FAISS IndexFlatIP (Cosine Similarity via L2 Normalized Inner Product)\n")

    embedder = MultilingualEmbedder()

    # 2. Test Queries (2 Indic, 1 English for cross-lingual retrieval)
    test_queries = [
        {"id": "Q1", "text": "কৰ্পোৰেচন কি?", "lang": "Indic (Assamese)"},
        {"id": "Q2", "text": "ৰেচেল কাৰ্চনে কিয় এক বাধ্যবাধকতা সহ্য কৰিবলৈ লিখিছিল?", "lang": "Indic (Assamese)"},
        {"id": "Q3", "text": "what is a corporation?", "lang": "English"}
    ]

    report_lines = [
        "==================================================",
        "     FAISS SEMANTIC RETRIEVAL TEST RESULTS        ",
        "==================================================",
        f"Total Indexed Chunks : {vector_store.total_vectors}",
        f"FAISS Index Type     : IndexFlatIP (Cosine Similarity)\n"
    ]

    # Benchmark settings
    num_iterations = 5

    for q_item in test_queries:
        query_text = q_item["text"]
        query_lang = q_item["lang"]

        print(f"--------------------------------------------------")
        print(f"Testing Query [{q_item['id']} - {query_lang}]: \"{query_text}\"")

        # Warm-up / Initial run
        query_vec = embedder.embed_query(query_text, normalize=True)
        results = vector_store.search(query_vec, top_k=3)

        # Benchmark multiple iterations for latency
        embed_times = []
        search_times = []

        for _ in range(num_iterations):
            t0 = time.time()
            q_vec = embedder.embed_query(query_text, normalize=True)
            t1 = time.time()
            res = vector_store.search(q_vec, top_k=3)
            t2 = time.time()

            embed_times.append((t1 - t0) * 1000)   # ms
            search_times.append((t2 - t1) * 1000)  # ms

        avg_embed_ms = sum(embed_times) / num_iterations
        avg_search_ms = sum(search_times) / num_iterations
        avg_total_ms = avg_embed_ms + avg_search_ms

        print(f"  Latency Metrics (Average over {num_iterations} runs):")
        print(f"    - Query Embedding Time : {avg_embed_ms:.3f} ms")
        print(f"    - FAISS Search Time    : {avg_search_ms:.3f} ms")
        print(f"    - Total Retrieval Time : {avg_total_ms:.3f} ms\n")

        print("  Top Retrieved Chunks:")
        report_lines.append(f"Query [{q_item['id']} - {query_lang}]: \"{query_text}\"")
        report_lines.append(f"  Latency: Embedding = {avg_embed_ms:.3f} ms | FAISS Search = {avg_search_ms:.3f} ms | Total = {avg_total_ms:.3f} ms")

        for rank, match in enumerate(results, start=1):
            score = match["score"]
            meta = match["metadata"]
            chunk_id = meta.get("chunk_id", f"chunk_{match['vector_id']}")
            chunk_text = meta.get("text", "")
            is_selected = meta.get("is_selected", False)

            print(f"    Rank #{rank} | Similarity Score: {score:.4f} | Selected Answer: {is_selected}")
            print(f"      Chunk ID: {chunk_id}")
            print(f"      Text    : {chunk_text[:120]}...\n")

            report_lines.append(f"  Rank #{rank} | Score: {score:.4f} | Chunk ID: {chunk_id} | Selected: {is_selected}")
            report_lines.append(f"    Text: \"{chunk_text[:120]}...\"")

        report_lines.append("")

    # Save results to data/faiss_test_results.txt
    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "faiss_test_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"FAISS retrieval test report saved to '{report_file}'.")


if __name__ == "__main__":
    run_faiss_tests()
