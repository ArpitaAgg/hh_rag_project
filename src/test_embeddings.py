"""
src/test_embeddings.py

Test runner and validation script for the multilingual embeddings module.
Loads sample text chunks from Step 3, generates embeddings, checks vector
properties (dimension, normalization, non-NaN values), and logs statistics.
"""

import os
import sys
import json
import time
import numpy as np

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from chunking import ChunkingPipeline
from embeddings import MultilingualEmbedder, DEFAULT_MODEL_NAME

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_embedding_test():
    print("==================================================")
    print("      MULTILINGUAL EMBEDDINGS TEST & VALIDATION   ")
    print("==================================================")

    # 1. Load sample records and generate chunks from Step 3
    sample_file = os.path.join("data", "sample_records.json")
    if not os.path.exists(sample_file):
        print(f"Error: Sample data file '{sample_file}' not found. Please complete Step 2 first.")
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    pipeline = ChunkingPipeline()
    chunks = pipeline.process_records(records, strategy_name="overlapping_window", use_translated=True)
    
    print(f"Loaded {len(records)} sample records -> Generated {len(chunks)} text chunks for embedding.\n")

    # Extract chunk texts
    chunk_texts = [c["text"] for c in chunks]

    # 2. Initialize Embedder
    embedder = MultilingualEmbedder(model_name=DEFAULT_MODEL_NAME)
    
    # Measure model loading and embedding inference time
    start_time = time.time()
    embeddings = embedder.embed_texts(chunk_texts, normalize=True)
    elapsed_time = time.time() - start_time

    # 3. Verification & Metrics
    num_chunks = embeddings.shape[0]
    dimension = embedder.embedding_dimension
    matrix_shape = embeddings.shape

    # Check for NaN / Inf values
    has_nans = np.isnan(embeddings).any()
    has_infs = np.isinf(embeddings).any()
    is_valid_numbers = not (has_nans or has_infs)

    # Check normalization (L2 norm of each vector should be approx 1.0)
    vector_norms = np.linalg.norm(embeddings, axis=1)
    is_normalized = np.allclose(vector_norms, 1.0, atol=1e-5)

    print("--- Embedding Performance & Verification Results ---")
    print(f"Model Name           : {DEFAULT_MODEL_NAME}")
    print(f"Chunks Embedded      : {num_chunks}")
    print(f"Embedding Dimension  : {dimension}")
    print(f"Matrix Shape         : {matrix_shape}")
    print(f"Inference Time       : {elapsed_time:.4f} seconds ({elapsed_time/num_chunks:.4f} sec/chunk)")
    print(f"Valid Numerical Values (No NaN/Inf) : {'PASS' if is_valid_numbers else 'FAIL'}")
    print(f"L2 Normalized (Unit Length = 1.0)  : {'PASS' if is_normalized else 'FAIL'}\n")

    # 4. Test Single Query Embedding
    sample_query = "কৰ্পোৰেচন কি?"
    q_start = time.time()
    query_vector = embedder.embed_query(sample_query, normalize=True)
    q_elapsed = time.time() - q_start

    q_norm = np.linalg.norm(query_vector)
    q_valid = not (np.isnan(query_vector).any() or np.isinf(query_vector).any())

    print("--- Single Query Embedding Test ---")
    print(f"Sample Indic Query   : \"{sample_query}\"")
    print(f"Query Vector Shape   : {query_vector.shape}")
    print(f"Query Vector Norm    : {q_norm:.6f} (Expected ~1.0)")
    print(f"Query Inference Time : {q_elapsed:.4f} seconds")
    print(f"Valid Query Vector   : {'PASS' if q_valid else 'FAIL'}\n")

    # 5. Save results to data/embedding_test_results.txt
    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "embedding_test_results.txt")
    
    report_content = f"""==================================================
    EMBEDDINGS MODULE TEST RESULTS
==================================================
Model Name           : {DEFAULT_MODEL_NAME}
Chunks Embedded      : {num_chunks}
Embedding Dimension  : {dimension}
Matrix Shape         : {matrix_shape}
Inference Time       : {elapsed_time:.4f} seconds
Avg Time per Chunk   : {elapsed_time/num_chunks:.4f} seconds
Valid Numerical Values : {'PASS' if is_valid_numbers else 'FAIL'}
L2 Normalized Vector   : {'PASS' if is_normalized else 'FAIL'}

Single Query Test    : "{sample_query}"
Query Vector Shape   : {query_vector.shape}
Query Vector Norm    : {q_norm:.6f}
Query Valid          : {'PASS' if q_valid else 'FAIL'}
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Embedding test results saved to '{report_file}'.")


if __name__ == "__main__":
    run_embedding_test()
