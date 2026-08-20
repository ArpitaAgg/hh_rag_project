"""
src/test_generator.py

End-to-End RAG Pipeline Test & Latency Benchmark Script.
Integrates Step 6 Hybrid Retriever (FAISS + BM25) with Step 7 Answer Generator (Provider Independent).
Tests answerable Indic & English queries and insufficient context queries.
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
from generator import get_generator, BaseAnswerGenerator

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_generator_tests():
    print("==================================================")
    print("      END-TO-END RAG GENERATOR TEST & BENCHMARK   ")
    print("==================================================")

    # 1. Load local sample records and chunks
    sample_file = os.path.join("data", "sample_records.json")
    if not os.path.exists(sample_file):
        print(f"Error: '{sample_file}' not found. Please complete Step 2 first.")
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    chunker_pipeline = ChunkingPipeline()
    chunks = chunker_pipeline.process_records(records, strategy_name="overlapping_window", use_translated=True)
    
    # 2. Setup BM25 and FAISS Vector Stores
    bm25_store = BM25Store()
    bm25_store.index_chunks(chunks)

    index_dir = os.path.join("data", "indexes")
    if not os.path.exists(os.path.join(index_dir, "index.faiss")):
        from build_faiss_index import build_index
        build_index()

    vector_store = FAISSVectorStore.load(index_dir)
    embedder = MultilingualEmbedder()

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_store=bm25_store,
        semantic_weight=0.7,
        keyword_weight=0.3
    )

    # 3. Instantiate Provider-Independent Generator
    generator: BaseAnswerGenerator = get_generator()
    print(f"Active Answer Generator Provider: '{generator.provider_name}'\n")

    # 4. Test Queries (3 Answerable, 2 Insufficient Context)
    test_queries = [
        {
            "id": "Q1",
            "type": "Answerable (Indic)",
            "text": "কৰ্পোৰেচন কি?"
        },
        {
            "id": "Q2",
            "type": "Answerable (Indic)",
            "text": "ৰেচেল কাৰ্চনে কিয় এক বাধ্যবাধকতা সহ্য কৰিবলৈ লিখিছিল?"
        },
        {
            "id": "Q3",
            "type": "Answerable (English)",
            "text": "what is a corporation?"
        },
        {
            "id": "Q4",
            "type": "Insufficient Context",
            "text": "What is the speed of light in a vacuum?"
        },
        {
            "id": "Q5",
            "type": "Insufficient Context",
            "text": "Who won the FIFA World Cup in 2022?"
        }
    ]

    report_lines = [
        "==================================================",
        "      RAG PIPELINE GENERATION TEST RESULTS        ",
        "==================================================",
        f"Active Generator Provider : {generator.provider_name}",
        f"Total Sample Chunks       : {len(chunks)}\n"
    ]

    num_iterations = 5

    for q_item in test_queries:
        query_text = q_item["text"]
        q_type = q_item["type"]
        q_id = q_item["id"]

        print(f"--------------------------------------------------")
        print(f"Testing Query [{q_id} - {q_type}]: \"{query_text}\"")

        # Benchmark latency over multiple iterations
        retrieval_latencies = []
        generation_latencies = []
        total_latencies = []
        final_gen_result = None
        retrieval_data = None

        for _ in range(num_iterations):
            t_start = time.time()
            retrieval_data = hybrid_retriever.retrieve(query_text, top_k=3)
            t_retrieved = time.time()
            
            gen_result = generator.generate(query_text, retrieval_data["results"])
            t_end = time.time()

            ret_ms = (t_retrieved - t_start) * 1000
            gen_ms = (t_end - t_retrieved) * 1000
            tot_ms = (t_end - t_start) * 1000

            retrieval_latencies.append(ret_ms)
            generation_latencies.append(gen_ms)
            total_latencies.append(tot_ms)
            final_gen_result = gen_result

        avg_ret_ms = sum(retrieval_latencies) / num_iterations
        avg_gen_ms = sum(generation_latencies) / num_iterations
        avg_tot_ms = sum(total_latencies) / num_iterations

        answer = final_gen_result["answer"]
        status = final_gen_result["grounded_status"]
        provider = final_gen_result["model_provider"]
        error = final_gen_result["error"]

        print(f"  Grounded Status  : {status.upper()}")
        print(f"  Model Provider   : {provider}")
        print(f"  Latency Metrics  : Retrieval={avg_ret_ms:.2f}ms | Generation={avg_gen_ms:.2f}ms | Total={avg_tot_ms:.2f}ms")
        print(f"  Generated Answer : {answer[:140]}...\n")

        report_lines.append(f"Query [{q_id} - {q_type}]: \"{query_text}\"")
        report_lines.append(f"  - Grounded Status    : {status}")
        report_lines.append(f"  - Model Provider     : {provider}")
        report_lines.append(f"  - Retrieval Latency  : {avg_ret_ms:.2f} ms")
        report_lines.append(f"  - Generation Latency : {avg_gen_ms:.2f} ms")
        report_lines.append(f"  - Total RAG Latency  : {avg_tot_ms:.2f} ms")
        report_lines.append(f"  - Generated Answer   : \"{answer}\"")
        if error:
            report_lines.append(f"  - Error              : {error}")
        report_lines.append("")

    # Save to data/generation_test_results.txt
    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "generation_test_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Generation test results saved to '{report_file}'.")


if __name__ == "__main__":
    run_generator_tests()
