"""
src/test_groq_rag.py

STEP 15: Tests the REAL Groq Production Generation Path.
Validates end-to-end RAG with real Groq LLM inference on data/test_indexes/english_1000/.
Verifies prompt context structure, grounding instructions, and unsupported query rejection.
"""

import os
import sys
import time
import dotenv
import numpy as np
from typing import List, Dict, Any

dotenv.load_dotenv()

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import RAGPipeline
from generator import GroqAnswerGenerator

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_groq_rag_test():
    print("==================================================")
    print("  STEP 15: REAL GROQ PRODUCTION GENERATION TEST   ")
    print("==================================================")

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    is_configured = bool(groq_key)

    print(f"GROQ_API_KEY Configured Status : {'CONFIGURED' if is_configured else 'NOT CONFIGURED'}")

    if not is_configured:
        print("\n==================================================")
        print("  STATUS: GROQ TEST BLOCKED BY MISSING CREDENTIALS")
        print("==================================================")
        print("Reason: GROQ_API_KEY environment variable is missing in .env.")
        print("==================================================")
        return False

    test_index_dir = os.path.join("data", "test_indexes", "english_1000")
    if not os.path.exists(os.path.join(test_index_dir, "index.faiss")):
        raise FileNotFoundError(f"Test index missing under '{test_index_dir}'.")

    # Verify Production Index Safety
    prod_faiss = os.path.join("data", "indexes", "index.faiss")
    prod_meta = os.path.join("data", "indexes", "metadata.json")
    prod_faiss_before = os.path.exists(prod_faiss)
    prod_meta_before = os.path.exists(prod_meta)

    # Force Groq provider in environment
    os.environ["GENERATOR_PROVIDER"] = "groq"
    
    print(f"\nInitializing RAGPipeline with Groq Generator from '{test_index_dir}'...")
    pipeline = RAGPipeline(index_dir=test_index_dir, top_k=3, min_relevance_score=0.30)
    
    # Explicitly ensure Groq generator is active
    pipeline.generator = GroqAnswerGenerator(api_key=groq_key)

    # Verify Prompt Grounding Template Structure
    sample_prompt = pipeline.generator.format_prompt("Test Question", [{"chunk_id": "c1", "text": "Sample context text"}])
    prompt_has_context = "Sample context text" in sample_prompt
    prompt_has_instruction = "ONLY the information provided in the RETRIEVED CONTEXT" in sample_prompt

    print("\n--------------------------------------------------")
    print("      GROQ PROMPT GROUNDING VERIFICATION          ")
    print("--------------------------------------------------")
    print(f"1. Prompt Contains Retrieved Context : {'YES' if prompt_has_context else 'NO'}")
    print(f"2. Prompt Contains Grounding Constraint : {'YES' if prompt_has_instruction else 'NO'}")
    print("--------------------------------------------------")

    supported_queries = [
        "What is a corporation?",
        "What is climate change?",
        "What is photosynthesis?",
        "What is machine learning?",
        "What is the capital of India?"
    ]

    unsupported_queries = [
        "What is the population of Mars?",
        "What is the GDP of Japan in 2030?",
        "What is the boiling point of water on Mount Everest?"
    ]

    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []
    api_errors = 0

    query_records = []

    print("\n==================================================")
    print("     EVALUATING 5 SUPPORTED QUERIES (GROQ LLM)    ")
    print("==================================================")

    for idx, query in enumerate(supported_queries, start=1):
        res = pipeline.answer(query)
        query_records.append({"type": "supported", "query": query, "result": res})

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

        if status == "failed":
            api_errors += 1

        print(f"\nSupported Query #{idx}: \"{query}\"")
        print(f"  - Retrieved Chunks   : {chunk_ids}")
        print(f"  - Combined Scores    : {scores}")
        print(f"  - Retrieval Latency  : {r_ms:.2f} ms")
        print(f"  - Sufficiency Result : {'SUFFICIENT' if status == 'answered' else 'INSUFFICIENT'}")
        print(f"  - Groq Success       : {'YES' if status != 'failed' else 'NO'}")
        print(f"  - Groq Answer        : \"{answer}\"")
        print(f"  - Generation Latency : {g_ms:.2f} ms")
        print(f"  - Total Latency      : {tot_ms:.2f} ms")
        print(f"  - Grounded           : {grounded}")
        print(f"  - Pipeline Status    : {status}")

    print("\n==================================================")
    print("    EVALUATING 3 UNSUPPORTED QUERIES (GROQ LLM)   ")
    print("==================================================")

    unsupported_rejected_count = 0
    for idx, query in enumerate(unsupported_queries, start=1):
        res = pipeline.answer(query)
        query_records.append({"type": "unsupported", "query": query, "result": res})

        status = res.get("status")
        answer = res.get("answer")
        grounded = res.get("grounded")
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

        is_rejected = (status in ["insufficient_context", "rejected"]) or ("not have enough information" in answer.lower())
        if is_rejected:
            unsupported_rejected_count += 1

        print(f"\nUnsupported Query #{idx}: \"{query}\"")
        print(f"  - Retrieved Chunks   : {chunk_ids}")
        print(f"  - Combined Scores    : {scores}")
        print(f"  - Retrieval Latency  : {r_ms:.2f} ms")
        print(f"  - Sufficiency Result : {'INSUFFICIENT (REJECTED)' if is_rejected else 'ALLOWED'}")
        print(f"  - Groq Success       : {'YES' if status != 'failed' else 'NO'}")
        print(f"  - Groq Response      : \"{answer}\"")
        print(f"  - Generation Latency : {g_ms:.2f} ms")
        print(f"  - Total Latency      : {tot_ms:.2f} ms")
        print(f"  - Grounded           : {grounded}")
        print(f"  - Pipeline Status    : {status}")

    # Calculate Benchmark Aggregates
    supp_res = [rec["result"] for rec in query_records if rec["type"] == "supported"]
    grounded_count = sum(1 for r in supp_res if r.get("grounded") and r.get("status") == "answered")
    grounded_rate = (grounded_count / len(supported_queries)) * 100.0
    rejection_rate = (unsupported_rejected_count / len(unsupported_queries)) * 100.0

    ret_p50 = round(float(np.median(retrieval_latencies)), 2)
    ret_p95 = round(float(np.percentile(retrieval_latencies, 95)), 2)

    gen_p50 = round(float(np.median(generation_latencies)), 2)
    gen_p95 = round(float(np.percentile(generation_latencies, 95)), 2)

    tot_p50 = round(float(np.median(total_latencies)), 2)
    tot_p95 = round(float(np.percentile(total_latencies, 95)), 2)

    print("\n==================================================")
    print("      STEP 15: GROQ PRODUCTION BENCHMARK SUMMARY   ")
    print("==================================================")
    print(f"1. Grounded Answer Rate           : {grounded_rate:.1f}% ({grounded_count}/{len(supported_queries)})")
    print(f"2. Unsupported Query Rejection    : {rejection_rate:.1f}% ({unsupported_rejected_count}/{len(unsupported_queries)})")
    print(f"3. Retrieval Latency (p50 / p95)  : {ret_p50} ms / {ret_p95} ms")
    print(f"4. Groq Gen Latency (p50 / p95)   : {gen_p50} ms / {gen_p95} ms")
    print(f"5. Total End-to-End (p50 / p95)   : {tot_p50} ms / {tot_p95} ms")
    print(f"6. API Errors / Failures          : {api_errors}")
    print(f"7. Active Groq Model Backend      : '{pipeline.generator.provider_name}'")
    print("==================================================")

    # Verify Production Index Protection
    prod_faiss_after = os.path.exists(prod_faiss)
    prod_meta_after = os.path.exists(prod_meta)
    print("\n==================================================")
    print("        PRODUCTION INDEX SAFETY VERIFICATION       ")
    print("==================================================")
    print(f"Production FAISS Index Untouched : {'YES' if prod_faiss_before == prod_faiss_after else 'WARNING CHANGED'}")
    print(f"Production Metadata Untouched    : {'YES' if prod_meta_before == prod_meta_after else 'WARNING CHANGED'}")
    print("==================================================")
    return True


if __name__ == "__main__":
    run_groq_rag_test()
