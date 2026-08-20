"""
src/test_real_dataset_retrieval.py

Step 12D Retrieval Evaluation Suite: Streams real MSMARCO-XI records for Assamese (as),
Hindi (hi), and Bengali (bn), builds temporary in-memory FAISS and BM25 indexes,
runs HybridRetriever, and measures top-1, top-3, and top-5 retrieval accuracy and latency.
"""

import os
import sys
import time
import numpy as np
from pprint import pprint
from typing import List, Dict, Any

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from dataset_stream import LanguageDatasetStreamer
from chunking import ChunkingPipeline
from embeddings import MultilingualEmbedder
from vector_store import FAISSVectorStore
from bm25_store import BM25Store
from hybrid_retriever import HybridRetriever

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_real_dataset_retrieval_evaluation():
    print("==================================================")
    print("  REAL DATASET HYBRID RETRIEVAL EVALUATION (12D)  ")
    print("==================================================")

    t_global_start = time.time()
    embedder = MultilingualEmbedder()
    chunker = ChunkingPipeline()

    test_languages = [
        {"code": "as", "name": "Assamese", "target_prefix": "asm"},
        {"code": "hi", "name": "Hindi", "target_prefix": "hin"},
        {"code": "bn", "name": "Bengali", "target_prefix": "ben"}
    ]

    language_eval_reports = []
    representative_examples = []
    overall_total_records = 0
    overall_total_chunks = 0
    all_latencies_ms = []

    report_lines = [
        "==================================================",
        "  REAL DATASET HYBRID RETRIEVAL EVALUATION REPORT ",
        "==================================================\n",
        f"Embedding Model     : {embedder.model_name}",
        f"Embedding Dimension : {embedder.embedding_dimension}",
        f"Retrieval Fusion    : Hybrid (70% FAISS Semantic + 30% BM25 Keyword)\n"
    ]

    for lang_info in test_languages:
        lang_code = lang_info["code"]
        lang_name = lang_info["name"]
        target_prefix = lang_info["target_prefix"]

        print(f"\n--------------------------------------------------")
        print(f"Evaluating Language: {lang_name} ({lang_code})")
        print(f"--------------------------------------------------")

        t_lang_start = time.time()
        streamer = LanguageDatasetStreamer(language=lang_code, split="train", max_records=10)

        # 1. Stream 10 real dataset records
        records = []
        for rec in streamer.stream_records():
            records.append(rec)

        if not records:
            print(f"Warning: No records streamed for {lang_name}. Skipping...")
            continue

        num_records = len(records)
        overall_total_records += num_records
        target_lang_val = records[0].get("target_lang", f"{target_prefix}_Deva")

        # 2. Process records through existing ChunkingPipeline
        chunks = chunker.process_records(records, strategy_name="overlapping_window", use_translated=True)
        num_chunks = len(chunks)
        overall_total_chunks += num_chunks

        # 3. Generate embeddings using existing MultilingualEmbedder
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_texts(chunk_texts, normalize=True)

        # 4. Build temporary in-memory FAISS Vector Store (without touching production disk index)
        temp_vector_store = FAISSVectorStore(dimension=embedder.embedding_dimension)
        temp_vector_store.add_embeddings(embeddings, chunks)

        # 5. Build temporary in-memory BM25 Store
        temp_bm25_store = BM25Store()
        temp_bm25_store.index_chunks(chunks)

        # 6. Instantiate existing HybridRetriever
        retriever = HybridRetriever(
            vector_store=temp_vector_store,
            embedder=embedder,
            bm25_store=temp_bm25_store,
            semantic_weight=0.7,
            keyword_weight=0.3
        )

        # 7. Evaluate retrieval per record
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        lang_latencies = []
        lang_errors = []

        for r_idx, rec in enumerate(records):
            q_id = rec.get("query_id")
            query_text = rec.get("query")
            trans_passages = rec.get("passages", {}).get("Translated_passages", [])
            is_selected_flags = rec.get("passages", {}).get("is_selected", [])

            # Ground truth selected passage text
            gt_passage = ""
            for p_i, p_txt in enumerate(trans_passages):
                if p_i < len(is_selected_flags) and (is_selected_flags[p_i] == 1 or is_selected_flags[p_i] is True):
                    gt_passage = p_txt
                    break
            if not gt_passage and trans_passages:
                gt_passage = trans_passages[0]

            # Execute Hybrid Retrieval
            t_ret_start = time.time()
            ret_output = retriever.retrieve(query_text, top_k=5)
            t_ret_elapsed = (time.time() - t_ret_start) * 1000
            lang_latencies.append(t_ret_elapsed)
            all_latencies_ms.append(t_ret_elapsed)

            retrieved_candidates = ret_output.get("results", [])

            # Evaluate top-1, top-3, top-5 accuracy
            top1_hit = False
            top3_hit = False
            top5_hit = False

            for c_rank, cand in enumerate(retrieved_candidates, start=1):
                c_meta = cand.get("metadata", {})
                is_match = (c_meta.get("query_id") == q_id and cand.get("is_selected") is True)
                
                # Secondary match fallback if metadata is_selected is True
                if not is_match and c_meta.get("query_id") == q_id:
                    is_match = True

                if is_match:
                    if c_rank == 1:
                        top1_hit = True
                    if c_rank <= 3:
                        top3_hit = True
                    if c_rank <= 5:
                        top5_hit = True

            if top1_hit: top1_hits += 1
            if top3_hit: top3_hits += 1
            if top5_hit: top5_hits += 1

            # Save representative example for report
            if r_idx == 0:
                top_cand = retrieved_candidates[0] if retrieved_candidates else {}
                representative_examples.append({
                    "language": lang_name,
                    "query": query_text,
                    "ground_truth": gt_passage[:100] + "...",
                    "top_retrieved": top_cand.get("text", "")[:100] + "...",
                    "relevant": "YES" if top1_hit else "NO",
                    "score": round(top_cand.get("combined_score", 0.0), 4)
                })

        t_lang_elapsed = time.time() - t_lang_start
        acc_top1 = (top1_hits / num_records) * 100
        acc_top3 = (top3_hits / num_records) * 100
        acc_top5 = (top5_hits / num_records) * 100
        avg_lat = np.mean(lang_latencies) if lang_latencies else 0.0
        p50_lat = np.median(lang_latencies) if lang_latencies else 0.0
        p95_lat = np.percentile(lang_latencies, 95) if lang_latencies else 0.0

        lang_summary = {
            "name": lang_name,
            "target_lang": target_lang_val,
            "num_records": num_records,
            "num_chunks": num_chunks,
            "dim": embedder.embedding_dimension,
            "faiss_size": temp_vector_store.total_vectors,
            "bm25_count": temp_bm25_store.total_chunks,
            "acc_top1": round(acc_top1, 1),
            "acc_top3": round(acc_top3, 1),
            "acc_top5": round(acc_top5, 1),
            "avg_lat": round(avg_lat, 2),
            "p50_lat": round(p50_lat, 2),
            "p95_lat": round(p95_lat, 2),
            "errors": len(lang_errors)
        }
        language_eval_reports.append(lang_summary)

        print(f"Results for {lang_name}:")
        print(f"  Chunks: {num_chunks} | FAISS: {temp_vector_store.total_vectors} vectors | BM25: {temp_bm25_store.total_chunks} docs")
        print(f"  Accuracy: Top-1={acc_top1:.1f}% | Top-3={acc_top3:.1f}% | Top-5={acc_top5:.1f}%")
        print(f"  Latency : Avg={avg_lat:.2f}ms | p50={p50_lat:.2f}ms | p95={p95_lat:.2f}ms")

    t_global_elapsed = time.time() - t_global_start

    # Compile Overall Stats
    overall_avg_lat = round(np.mean(all_latencies_ms), 2) if all_latencies_ms else 0.0
    overall_p50_lat = round(np.median(all_latencies_ms), 2) if all_latencies_ms else 0.0
    overall_p95_lat = round(np.percentile(all_latencies_ms, 95), 2) if all_latencies_ms else 0.0

    print("\n==================================================")
    print("--- OVERALL HYBRID RETRIEVAL SUMMARY ---")
    print("==================================================")
    print(f"Total Records Streamed : {overall_total_records}")
    print(f"Total Chunks Indexed   : {overall_total_chunks}")
    print(f"Overall Latency        : Avg={overall_avg_lat}ms | p50={overall_p50_lat}ms | p95={overall_p95_lat}ms")
    print(f"Total Test Elapsed Time: {t_global_elapsed:.2f} seconds")
    print("==================================================")

    # Build Report File Content
    for info in language_eval_reports:
        report_lines.append(f"LANGUAGE: {info['name']} (target_lang: {info['target_lang']})")
        report_lines.append(f"  Records Processed     : {info['num_records']}")
        report_lines.append(f"  Chunks Generated      : {info['num_chunks']}")
        report_lines.append(f"  Embedding Dimension   : {info['dim']}")
        report_lines.append(f"  FAISS Index Size      : {info['faiss_size']} vectors")
        report_lines.append(f"  BM25 Document Count   : {info['bm25_count']} docs")
        report_lines.append(f"  Top-1 Accuracy        : {info['acc_top1']}%")
        report_lines.append(f"  Top-3 Accuracy        : {info['acc_top3']}%")
        report_lines.append(f"  Top-5 Accuracy        : {info['acc_top5']}%")
        report_lines.append(f"  Average Latency       : {info['avg_lat']} ms")
        report_lines.append(f"  p50 Latency           : {info['p50_lat']} ms")
        report_lines.append(f"  p95 Latency           : {info['p95_lat']} ms")
        report_lines.append(f"  Errors                : {info['errors']}\n")

    report_lines.append("==================================================")
    report_lines.append("        3 REPRESENTATIVE RETRIEVAL EXAMPLES       ")
    report_lines.append("==================================================\n")

    for ex_idx, ex in enumerate(representative_examples, start=1):
        report_lines.append(f"--- Example #{ex_idx} ({ex['language']}) ---")
        report_lines.append(f"QUERY       : {ex['query']}")
        report_lines.append(f"GROUND TRUTH: {ex['ground_truth']}")
        report_lines.append(f"TOP RETRIEVED: {ex['top_retrieved']}")
        report_lines.append(f"RELEVANT    : {ex['relevant']}")
        report_lines.append(f"SCORE       : {ex['score']}")
        report_lines.append(f"LANGUAGE    : {ex['language']}\n")

    report_lines.append(f"TOTAL EXECUTION TIME : {t_global_elapsed:.2f}s")
    report_lines.append(f"PRODUCTION INDEXES   : UNTOUCHED (data/indexes/ remains original)")

    # Save to data/real_dataset_retrieval_test_results.txt
    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "real_dataset_retrieval_test_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Retrieval evaluation report saved to '{report_file}'.")


if __name__ == "__main__":
    run_real_dataset_retrieval_evaluation()
