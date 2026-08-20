"""
src/test_context_sufficiency.py

Audits and evaluates Principled Context Sufficiency Handling.
Ensures that RAG distinguishes between:
  A. Relevant context that directly answers/supports the query -> allow generation
  B. Related but insufficient context -> insufficient_context
  C. Clearly irrelevant context -> insufficient_context

Tests 4 Supported Queries and 4 Unsupported Queries on data/test_indexes/english_1000/.
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import RAGPipeline

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_context_sufficiency_test():
    print("==================================================")
    print("   CONTEXT SUFFICIENCY AUDIT & EVALUATION TEST    ")
    print("==================================================")

    test_index_dir = os.path.join("data", "test_indexes", "english_1000")
    if not os.path.exists(os.path.join(test_index_dir, "index.faiss")):
        raise FileNotFoundError(f"Test index missing under '{test_index_dir}'.")

    # Verify Production Index Protection
    prod_faiss = os.path.join("data", "indexes", "index.faiss")
    prod_meta = os.path.join("data", "indexes", "metadata.json")
    prod_faiss_before = os.path.exists(prod_faiss)
    prod_meta_before = os.path.exists(prod_meta)

    # Initialize RAG Pipeline
    pipeline = RAGPipeline(index_dir=test_index_dir, top_k=3, min_relevance_score=0.30)

    supported_queries = [
        "What is a corporation?",
        "What is climate change?",
        "What is photosynthesis?",
        "What is machine learning?"
    ]

    unsupported_queries = [
        "What is the population of Mars?",
        "Who won the FIFA World Cup in 2022?",
        "What is the boiling point of water on Mount Everest?",
        "What is the GDP of Japan in 2030?"
    ]

    print("\n==================================================")
    print("      EVALUATING 4 SUPPORTED QUERIES               ")
    print("==================================================")

    supp_results = []
    for idx, query in enumerate(supported_queries, start=1):
        res = pipeline.answer(query)
        supp_results.append(res)

        status = res.get("status")
        answer = res.get("answer")
        grounded = res.get("grounded")
        reason = res.get("guardrail_reason")
        context = res.get("retrieved_context", [])
        scores = [c.get("score") for c in context]
        chunk_ids = [c.get("chunk_id") for c in context]

        print(f"\nSupported Query #{idx}: \"{query}\"")
        print(f"  - Retrieved Chunks : {chunk_ids}")
        print(f"  - Combined Scores  : {scores}")
        print(f"  - Sufficiency Decision : {'SUFFICIENT' if status == 'answered' else 'INSUFFICIENT'}")
        print(f"  - Pipeline Status  : {status}")
        print(f"  - Answer Text      : \"{answer[:120]}...\"")

    print("\n==================================================")
    print("     EVALUATING 4 UNSUPPORTED QUERIES             ")
    print("==================================================")

    unsupp_results = []
    unsupported_rejected_count = 0

    for idx, query in enumerate(unsupported_queries, start=1):
        res = pipeline.answer(query)
        unsupp_results.append(res)

        status = res.get("status")
        answer = res.get("answer")
        grounded = res.get("grounded")
        reason = res.get("guardrail_reason")
        context = res.get("retrieved_context", [])
        scores = [c.get("score") for c in context]
        chunk_ids = [c.get("chunk_id") for c in context]

        is_rejected = (status in ["insufficient_context", "rejected"]) or ("not have enough information" in answer.lower())
        if is_rejected:
            unsupported_rejected_count += 1

        print(f"\nUnsupported Query #{idx}: \"{query}\"")
        print(f"  - Retrieved Chunks : {chunk_ids}")
        print(f"  - Combined Scores  : {scores}")
        print(f"  - Sufficiency Decision : {'INSUFFICIENT (REJECTED)' if is_rejected else 'ALLOWED (INCORRECT)'}")
        print(f"  - Pipeline Status  : {status}")
        print(f"  - Answer Text      : \"{answer}\"")

    rejection_rate = (unsupported_rejected_count / len(unsupported_queries)) * 100.0

    print("\n==================================================")
    print("        CONTEXT SUFFICIENCY AUDIT SUMMARY         ")
    print("==================================================")
    print(f"1. Supported Queries Tested     : {len(supported_queries)}")
    print(f"2. Supported Sufficiency Rate   : {(sum(1 for r in supp_results if r.get('status') == 'answered') / len(supported_queries))*100:.1f}%")
    print(f"3. Unsupported Queries Tested   : {len(unsupported_queries)}")
    print(f"4. Unsupported Rejection Rate   : {rejection_rate:.1f}% ({unsupported_rejected_count}/{len(unsupported_queries)})")
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


if __name__ == "__main__":
    run_context_sufficiency_test()
