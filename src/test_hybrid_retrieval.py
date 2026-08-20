"""
src/test_hybrid_retrieval.py

Test runner and benchmark script for Hybrid Retrieval (FAISS Semantic + BM25 Keyword).
Loads local sample data, indexes BM25, loads FAISS vector store, tests 3 queries,
measures latency breakdown, displays top-3 hybrid results, and writes data/hybrid_retrieval_results.txt.
"""

import os
import sys
import json
import time

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from chunking import ChunkingPipeline
from embeddings import MultilingualEmbedder
from vector_store import FAISSVectorStore
from bm25_store import BM25Store
from hybrid_retriever import HybridRetriever

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_hybrid_tests():
    print("==================================================")
    print("      HYBRID RETRIEVAL TEST & EVALUATION         ")
    print("==================================================")

    # 1. Load local sample records and chunks
    sample_file = os.path.join("data", "sample_records.json")
    if not os.path.exists(sample_file):
        print(f"Error: Sample data file '{sample_file}' not found. Please complete Step 2 first.")
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    chunker_pipeline = ChunkingPipeline()
    chunks = chunker_pipeline.process_records(records, strategy_name="overlapping_window", use_translated=True)
    print(f"Loaded {len(records)} sample records -> Generated {len(chunks)} text chunks.")

    # 2. Initialize BM25 Store & Index Chunks
    bm25_store = BM25Store()
    bm25_store.index_chunks(chunks)

    # 3. Load FAISS Vector Store
    index_dir = os.path.join("data", "indexes")
    if not os.path.exists(os.path.join(index_dir, "index.faiss")):
        from build_faiss_index import build_index
        build_index()

    vector_store = FAISSVectorStore.load(index_dir)
    embedder = MultilingualEmbedder()

    # 4. Initialize Hybrid Retriever (70% FAISS, 30% BM25)
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_store=bm25_store,
        semantic_weight=0.7,
        keyword_weight=0.3
    )

    # 5. Test Queries
    test_queries = [
        {"id": "Q1", "text": "কৰ্পোৰেচন কি?", "lang": "Indic (Assamese)"},
        {"id": "Q2", "text": "ৰেচেল কাৰ্চনে কিয় এক বাধ্যবাধকতা সহ্য কৰিবলৈ লিখিছিল?", "lang": "Indic (Assamese)"},
        {"id": "Q3", "text": "what is a corporation?", "lang": "English"}
    ]

    report_lines = [
        "==================================================",
        "     HYBRID RETRIEVAL (FAISS + BM25) RESULTS     ",
        "==================================================",
        f"Total Chunks Indexed : {len(chunks)}",
        f"Semantic Weight (FAISS) : {hybrid_retriever.semantic_weight}",
        f"Keyword Weight (BM25)  : {hybrid_retriever.keyword_weight}\n"
    ]

    num_iterations = 5

    for q_item in test_queries:
        query_text = q_item["text"]
        query_lang = q_item["lang"]

        print("--------------------------------------------------")
        print(f"Testing Query [{q_item['id']} - {query_lang}]: \"{query_text}\"")

        # Benchmarking latency over multiple runs
        faiss_latencies = []
        bm25_latencies = []
        fusion_latencies = []
        total_latencies = []

        output_data = None
        for _ in range(num_iterations):
            output_data = hybrid_retriever.retrieve(query_text, top_k=3)
            l = output_data["latency_ms"]
            faiss_latencies.append(l["faiss"])
            bm25_latencies.append(l["bm25"])
            fusion_latencies.append(l["fusion"])
            total_latencies.append(l["total"])

        avg_faiss = sum(faiss_latencies) / num_iterations
        avg_bm25 = sum(bm25_latencies) / num_iterations
        avg_fusion = sum(fusion_latencies) / num_iterations
        avg_total = sum(total_latencies) / num_iterations

        print(f"  Latency Metrics (Average over {num_iterations} runs):")
        print(f"    - FAISS Search Time    : {avg_faiss:.3f} ms")
        print(f"    - BM25 Search Time     : {avg_bm25:.3f} ms")
        print(f"    - Score Fusion Time    : {avg_fusion:.3f} ms")
        print(f"    - Total Hybrid Latency : {avg_total:.3f} ms\n")

        print("  Final Hybrid Retrieved Chunks:")
        report_lines.append(f"Query [{q_item['id']} - {query_lang}]: \"{query_text}\"")
        report_lines.append(f"  Latency: FAISS={avg_faiss:.3f}ms | BM25={avg_bm25:.3f}ms | Fusion={avg_fusion:.3f}ms | Total={avg_total:.3f}ms")

        for match in output_data["results"]:
            rank = match["rank"]
            cid = match["chunk_id"]
            comb_score = match["combined_score"]
            sem_raw = match["semantic_score_raw"]
            key_raw = match["keyword_score_raw"]
            is_sel = match["is_selected"]
            text_snippet = match["text"][:120]

            print(f"    Rank #{rank} | Combined Score: {comb_score:.4f} (Semantic: {sem_raw:.4f}, Keyword: {key_raw:.4f}) | Selected: {is_sel}")
            print(f"      Chunk ID: {cid}")
            print(f"      Text    : {text_snippet}...\n")

            report_lines.append(f"  Rank #{rank} | Combined Score: {comb_score:.4f} | FAISS Score: {sem_raw:.4f} | BM25 Score: {key_raw:.4f} | Chunk ID: {cid} | Selected: {is_sel}")
            report_lines.append(f"    Text: \"{text_snippet}...\"")

        report_lines.append("")

    # Save to data/hybrid_retrieval_results.txt
    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "hybrid_retrieval_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Hybrid retrieval report saved to '{report_file}'.")


if __name__ == "__main__":
    run_hybrid_tests()
