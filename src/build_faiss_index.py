"""
src/build_faiss_index.py

Script to process local sample data, generate chunks, generate embeddings,
and store the resulting vector index and metadata into data/indexes/.
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

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def build_index():
    print("==================================================")
    print("      BUILDING LOCAL FAISS VECTOR INDEX           ")
    print("==================================================")

    # 1. Load local sample records
    sample_file = os.path.join("data", "sample_records.json")
    if not os.path.exists(sample_file):
        print(f"Error: '{sample_file}' not found. Please complete Step 2 first.")
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} sample records.")

    # 2. Generate chunks using Step 3 chunking module
    chunker_pipeline = ChunkingPipeline()
    chunks = chunker_pipeline.process_records(records, strategy_name="overlapping_window", use_translated=True)
    print(f"Generated {len(chunks)} text chunks using 'overlapping_window' strategy.")

    # 3. Generate embeddings using Step 4 embedder module
    embedder = MultilingualEmbedder()
    chunk_texts = [c["text"] for c in chunks]
    
    t0 = time.time()
    embeddings = embedder.embed_texts(chunk_texts, normalize=True)
    embed_time = time.time() - t0
    print(f"Generated embeddings shape {embeddings.shape} in {embed_time:.4f} seconds.")

    # 4. Build FAISS vector store
    vector_store = FAISSVectorStore(dimension=embedder.embedding_dimension)
    vector_store.add_embeddings(embeddings, chunks)

    # 5. Save index to data/indexes/
    output_dir = os.path.join("data", "indexes")
    vector_store.save(output_dir)

    print(f"FAISS index successfully built and saved to '{output_dir}'.\n")


if __name__ == "__main__":
    build_index()
