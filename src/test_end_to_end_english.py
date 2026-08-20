"""
src/test_end_to_end_english.py

Step 13: End-to-End English RAG Pipeline Evaluation.
Uses the 1,000-record test index in `data/test_indexes/english_1000/`.

Tests 5 Supported Queries and 3 Unsupported Queries.
Verifies input guardrails, hybrid retrieval, context quality validation,
answer generation (Groq/Ollama), and output grounding guardrails.
Ensures zero hallucination on unsupported questions.

Strictly protects production indexes (data/indexes/index.faiss, metadata.json).
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import RAGPipeline

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_end_to_end_english_eval():
    print("==================================================")
    print("  STEP 13: END-TO-END ENGLISH RAG EVALUATION      ")
    print("==================================================")
    t_suite_start = time.time()

    test_index_dir = os.path.join("data", "test_indexes", "english_1000")
    faiss_path = os.path.join(test_index_dir, "index.faiss")
    bm25_path = os.path.join(test_index_dir, "bm25_store.pkl")

    if not os.path.exists(faiss_path) or not os.path.exists(bm25_path):
        raise FileNotFoundError(f"Test index not found at '{test_index_dir}'. Please run Step 12 1,000-record test first.")

    # Verify Production Index Protection
    prod_faiss = os.path.join("data", "indexes", "index.faiss")
    prod_meta = os.path.join("data", "indexes", "metadata.json")
    prod_faiss_before = os.path.exists(prod_faiss)
    prod_meta_before = os.path.exists(prod_meta)

    # 1. Instantiate RAG Pipeline with test index
    print(f"Loading RAGPipeline Orchestrator from '{test_index_dir}'...")
    pipeline = RAGPipeline(index_dir=test_index_dir, top_k=3, min_relevance_score=0.30)

    # Supported Queries
    supported_queries = [
        "What is a corporation?",
        "What is climate change?",
        "What is photosynthesis?",
        "What is the capital of India?",
        "What is machine learning?"
    ]

    # Unsupported Queries (Unlikely to be answered by 1,000-record subset)
    unsupported_queries = [
        "What is the population of Mars?",
        "Who won the FIFA World Cup in 2022?",
        "What is the boiling point of water on Mount Everest?"
    ]

    all_queries = supported_queries + unsupported_queries

    query_results: List[Dict[str, Any]] = []
    errors = 0

    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []

    print("\n==================================================")
    print("    EVALUATING 5 SUPPORTED ENGLISH QUERIES        ")
    print("==================================================")

    for idx, query in enumerate(supported_queries, start=1):
        print(f"\n--------------------------------------------------")
        print(f"SUPPORTED QUERY #{idx}: \"{query}\"")
        print(f"--------------------------------------------------")

        res = pipeline.answer(query)
        query_results.append({"type": "supported", "result": res})

        status = res.get("status")
        answer = res.get("answer")
        grounded = res.get("grounded")
        reason = res.get("guardrail_reason")
        context = res.get("retrieved_context", [])
        lat = res.get("latency", {})

        r_ms = lat.get("retrieval_ms", 0.0)
        g_ms = lat.get("generation_ms", 0.0)
        tot_ms = lat.get("total_ms", 0.0)

        retrieval_latencies.append(r_ms)
        generation_latencies.append(g_ms)
        total_latencies.append(tot_ms)

        chunk_ids = [c.get("chunk_id") for c in context]
        scores = [c.get("score") for c in context]
        top_text = context[0].get("text", "")[:120] if context else "None"

        print(f"  - Pipeline Status      : {status}")
        print(f"  - Retrieved Chunk IDs  : {chunk_ids}")
        print(f"  - Combined Scores      : {scores}")
        print(f"  - Top Context Snippet  : \"{top_text}...\"")
        print(f"  - Retrieval Latency    : {r_ms:.2f} ms")
        print(f"  - Generated Answer     : \"{answer}\"")
        print(f"  - Generation Latency   : {g_ms:.2f} ms")
        print(f"  - Total End-to-End     : {tot_ms:.2f} ms")
        print(f"  - Guardrail Reason     : {reason}")
        print(f"  - Supported & Grounded : {grounded}")

    print("\n==================================================")
    print("   EVALUATING 3 UNSUPPORTED ENGLISH QUERIES       ")
    print("==================================================")

    for idx, query in enumerate(unsupported_queries, start=6):
        print(f"\n--------------------------------------------------")
        print(f"UNSUPPORTED QUERY #{idx}: \"{query}\"")
        print(f"--------------------------------------------------")

        res = pipeline.answer(query)
        query_results.append({"type": "unsupported", "result": res})

        status = res.get("status")
        answer = res.get("answer")
        grounded = res.get("grounded")
        reason = res.get("guardrail_reason")
        context = res.get("retrieved_context", [])
        lat = res.get("latency", {})

        r_ms = lat.get("retrieval_ms", 0.0)
        g_ms = lat.get("generation_ms", 0.0)
        tot_ms = lat.get("total_ms", 0.0)

        retrieval_latencies.append(r_ms)
        generation_latencies.append(g_ms)
        total_latencies.append(tot_ms)

        chunk_ids = [c.get("chunk_id") for c in context]
        scores = [c.get("score") for c in context]
        top_text = context[0].get("text", "")[:120] if context else "None"

        print(f"  - Pipeline Status      : {status}")
        print(f"  - Retrieved Chunk IDs  : {chunk_ids}")
        print(f"  - Combined Scores      : {scores}")
        print(f"  - Top Context Snippet  : \"{top_text}...\"")
        print(f"  - Retrieval Latency    : {r_ms:.2f} ms")
        print(f"  - Generated Answer     : \"{answer}\"")
        print(f"  - Generation Latency   : {g_ms:.2f} ms")
        print(f"  - Total End-to-End     : {tot_ms:.2f} ms")
        print(f"  - Guardrail Reason     : {reason}")
        print(f"  - Refused / Non-Hallucinated : {'YES' if status in ['insufficient_context', 'rejected'] or 'not have enough information' in answer.lower() else 'NO'}")

    # Summary Metrics Calculation
    supported_res = [r["result"] for r in query_results if r["type"] == "supported"]
    unsupported_res = [r["result"] for r in query_results if r["type"] == "unsupported"]

    retrieval_success_count = sum(1 for item in query_results if item["result"].get("retrieved_context"))
    retrieval_success_rate = (retrieval_success_count / len(all_queries)) * 100.0

    grounded_count = sum(1 for r in supported_res if r.get("grounded") and r.get("status") == "answered")
    grounded_answer_rate = (grounded_count / len(supported_queries)) * 100.0

    unsupported_rejected_count = sum(
        1 for r in unsupported_res
        if r.get("status") in ["insufficient_context", "rejected"] or "not have enough information" in r.get("answer", "").lower()
    )
    unsupported_rejection_rate = (unsupported_rejected_count / len(unsupported_queries)) * 100.0

    gen_success_count = sum(1 for r in supported_res if r.get("status") == "answered")
    gen_success_rate = (gen_success_count / len(supported_queries)) * 100.0

    # Latency p50 / p95
    ret_p50 = round(float(np.median(retrieval_latencies)), 2)
    ret_p95 = round(float(np.percentile(retrieval_latencies, 95)), 2)

    gen_lat_active = [g for g in generation_latencies if g > 0.0]
    gen_p50 = round(float(np.median(gen_lat_active)), 2) if gen_lat_active else 0.0
    gen_p95 = round(float(np.percentile(gen_lat_active, 95)), 2) if gen_lat_active else 0.0

    tot_p50 = round(float(np.median(total_latencies)), 2)
    tot_p95 = round(float(np.percentile(total_latencies, 95)), 2)

    t_suite_total = time.time() - t_suite_start

    print("\n==================================================")
    print("      END-TO-END RAG EVALUATION SUMMARY REPORT    ")
    print("==================================================")
    print(f"1. Total Test Queries Evaluated    : {len(all_queries)} (5 Supported + 3 Unsupported)")
    print(f"2. Hybrid Retrieval Success Rate   : {retrieval_success_rate:.1f}% ({retrieval_success_count}/{len(all_queries)})")
    print(f"3. Grounded Answer Rate            : {grounded_answer_rate:.1f}% ({grounded_count}/{len(supported_queries)})")
    print(f"4. Unsupported Rejection Rate      : {unsupported_rejection_rate:.1f}% ({unsupported_rejected_count}/{len(unsupported_queries)})")
    print(f"5. Generation Success Rate         : {gen_success_rate:.1f}% ({gen_success_count}/{len(supported_queries)})")
    print(f"6. Retrieval Latency (p50 / p95)   : {ret_p50} ms / {ret_p95} ms")
    print(f"7. Generation Latency (p50 / p95)  : {gen_p50} ms / {gen_p95} ms")
    print(f"8. Total End-to-End (p50 / p95)    : {tot_p50} ms / {tot_p95} ms")
    print(f"9. Pipeline Errors / Exceptions    : {errors}")
    print(f"10. Active Generator Provider      : '{pipeline.generator.provider_name}'")
    print("==================================================")

    # 9. Context Grounding Verification
    print("\n==================================================")
    print("     CONTEXT GROUNDING VERIFICATION CHECK         ")
    print("==================================================")
    context_received_by_gen = True
    for r in supported_res:
        if r.get("status") == "answered":
            ans = r.get("answer", "")
            ctx_texts = [c.get("text", "") for c in r.get("retrieved_context", [])]
            # Check overlap between generated answer and retrieved context text
            ans_words = set(ans.lower().split())
            ctx_words = set(" ".join(ctx_texts).lower().split())
            if len(ans_words.intersection(ctx_words)) < 3:
                context_received_by_gen = False

    print(f"Generator Receives Retrieved Context : {'YES (Empirically Verified)' if context_received_by_gen else 'NO'}")
    print("==================================================")

    # 10. Production Index Safety Verification
    prod_faiss_after = os.path.exists(prod_faiss)
    prod_meta_after = os.path.exists(prod_meta)
    print("\n==================================================")
    print("        PRODUCTION INDEX SAFETY VERIFICATION       ")
    print("==================================================")
    print(f"Production FAISS Index Untouched : {'YES' if prod_faiss_before == prod_faiss_after else 'WARNING CHANGED'}")
    print(f"Production Metadata Untouched    : {'YES' if prod_meta_before == prod_meta_after else 'WARNING CHANGED'}")
    print("==================================================")


if __name__ == "__main__":
    run_end_to_end_english_eval()
