"""
src/ingest_dataset.py

One-time Offline Data Ingestion Pipeline for ai4bharat/MSMARCO-XI.
Downloads and caches target language Parquet files locally under `data/raw/msmarco_xi/<lang>/`.
Reads directly from local disk to process records, generate chunks, compute embeddings,
and build persisted FAISS + BM25 indexes under `data/indexes/<lang>/` (or custom output directory).

Decouples offline data ingestion completely from production FastAPI runtime.
"""

import os
import sys
import time
import gc
import json
import argparse
import requests
import pyarrow.parquet as pq
from typing import List, Dict, Any

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from chunking import ChunkingPipeline
from embeddings import MultilingualEmbedder
from vector_store import FAISSVectorStore
from bm25_store import BM25Store
from dataset_stream import normalize_record, LANGUAGE_MAP, DATASET_NAME

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def get_dir_size_mb(path: str) -> float:
    """Calculates total size of files in a directory in MB."""
    if not os.path.exists(path):
        return 0.0
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024.0 * 1024.0)
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024.0 * 1024.0)


def ensure_raw_parquet_file(language: str, split: str = "train") -> str:
    """
    Downloads and caches the single language Parquet file into `data/raw/msmarco_xi/<lang>/`.
    Skips download if file already exists locally.
    Uses huggingface_hub.hf_hub_download for one-time offline ingestion.
    """
    import huggingface_hub

    lang_info = LANGUAGE_MAP.get(language.lower())
    if not lang_info:
        raise ValueError(f"Unsupported language code: {language}")

    file_prefix = lang_info["file_prefix"]
    split_suffix = "train" if split == "train" else "val"
    filename = f"{file_prefix}{split_suffix}.parquet"
    repo_file_path = f"{split}/{filename}"

    raw_dir = os.path.join("data", "raw", "msmarco_xi", language.lower())
    os.makedirs(raw_dir, exist_ok=True)
    local_path = os.path.join(raw_dir, filename)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 100 * 1024 * 1024:
        file_size_mb = os.path.getsize(local_path) / (1024.0 * 1024.0)
        print(f"Local Parquet cache found at '{local_path}' ({file_size_mb:.2f} MB). Skipping download.")
        return local_path

    print(f"Downloading raw Parquet file for language '{language}' split '{split}' using huggingface_hub...")
    print(f"Dataset Repository : {DATASET_NAME}")
    print(f"Remote File        : {repo_file_path}")
    print(f"Destination Dir    : {raw_dir}")

    t0 = time.time()
    downloaded_path = huggingface_hub.hf_hub_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        filename=repo_file_path,
        local_dir=raw_dir
    )

    t_dl = time.time() - t0
    file_size_mb = os.path.getsize(downloaded_path) / (1024.0 * 1024.0)
    print(f"\nSuccessfully downloaded raw Parquet file ({file_size_mb:.2f} MB) to '{downloaded_path}' in {t_dl:.2f}s!")
    return downloaded_path


def load_records_from_local_parquet(local_path: str, max_records: int = 0) -> List[Dict[str, Any]]:
    """
    Reads records directly from local Parquet file using PyArrow.
    Loads up to `max_records` (0 = all records in file).
    """
    t0 = time.time()
    records = []
    
    pf = pq.ParquetFile(local_path)
    print(f"Reading records from local Parquet file '{local_path}' (Total rows in file: {pf.metadata.num_rows})...")

    target_count = max_records if max_records > 0 else pf.metadata.num_rows

    for batch in pf.iter_batches(batch_size=min(target_count, 1000)):
        pydict = batch.to_pydict()
        num_rows = len(pydict["query_id"])

        for i in range(num_rows):
            raw_ex = {
                "query_id": pydict["query_id"][i],
                "query": pydict["query"][i],
                "Answer": pydict["Answer"][i] if "Answer" in pydict else None,
                "query_type": pydict["query_type"][i] if "query_type" in pydict else None,
                "source_lang": pydict["source_lang"][i] if "source_lang" in pydict else None,
                "target_lang": pydict["target_lang"][i] if "target_lang" in pydict else None,
                "Eng_Query": pydict["Eng_Query"][i] if "Eng_Query" in pydict else None,
                "Eng_Answer": pydict["Eng_Answer"][i] if "Eng_Answer" in pydict else None,
                "passages": pydict["passages"][i] if "passages" in pydict else {}
            }
            records.append(normalize_record(raw_ex))

            if max_records > 0 and len(records) >= max_records:
                break

        if max_records > 0 and len(records) >= max_records:
            break

    t_elapsed = time.time() - t0
    print(f"Loaded {len(records)} normalized records from local Parquet in {t_elapsed:.2f}s.")
    return records


def ingest_dataset_pipeline(
    language: str = "as",
    split: str = "train",
    max_records: int = 100,
    output_dir: str = ""
):
    print("==================================================")
    print("  MSMARCO-XI ONE-TIME OFFLINE INGESTION PIPELINE  ")
    print("==================================================")
    t_start = time.time()

    # Determine Output Directory
    if not output_dir:
        dest_dir = os.path.join("data", "indexes", language.lower())
    else:
        dest_dir = output_dir

    os.makedirs(dest_dir, exist_ok=True)

    print(f"Target Language   : '{language}'")
    print(f"Dataset Split     : '{split}'")
    print(f"Requested Records : {max_records if max_records > 0 else 'ALL (0)'}")
    print(f"Output Index Dir  : '{dest_dir}'")
    print("--------------------------------------------------")

    # 1. Obtain Raw Local Parquet File
    local_parquet = ensure_raw_parquet_file(language=language, split=split)
    raw_file_size_mb = get_dir_size_mb(local_parquet)

    # 2. Read Normalized Records
    records = load_records_from_local_parquet(local_parquet, max_records=max_records)
    actual_records_count = len(records)
    if actual_records_count == 0:
        raise RuntimeError("No records loaded from local Parquet file.")

    total_passages_count = sum(len(r.get("passages", {}).get("Translated_passages", [])) for r in records)

    # 3. Chunking Stage
    print(f"\nChunking {actual_records_count} records using ChunkingPipeline...")
    t_chunk_start = time.time()
    chunker = ChunkingPipeline()
    chunks = chunker.process_records(records, strategy_name="overlapping_window", use_translated=True)
    t_chunk_elapsed = time.time() - t_chunk_start
    chunks_count = len(chunks)
    print(f"Generated {chunks_count} chunks from {total_passages_count} passages in {t_chunk_elapsed:.2f}s.")

    # 4. Embeddings Stage
    print(f"\nComputing embeddings for {chunks_count} chunks using MultilingualEmbedder...")
    t_embed_start = time.time()
    embedder = MultilingualEmbedder()
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts, normalize=True, batch_size=64)
    t_embed_elapsed = time.time() - t_embed_start
    embed_count = len(embeddings)
    print(f"Computed {embed_count} vector embeddings ({embedder.embedding_dimension}D) in {t_embed_elapsed:.2f}s.")

    # 5. FAISS Vector Store Indexing
    print("\nBuilding FAISS vector store index...")
    t_faiss_start = time.time()
    vector_store = FAISSVectorStore(dimension=embedder.embedding_dimension)
    vector_store.add_embeddings(embeddings, chunks)
    
    faiss_index_path = os.path.join(dest_dir, "index.faiss")
    faiss_meta_path = os.path.join(dest_dir, "metadata.json")
    vector_store.save_index(index_path=faiss_index_path, metadata_path=faiss_meta_path)
    t_faiss_elapsed = time.time() - t_faiss_start
    print(f"Saved FAISS index ({vector_store.total_vectors} vectors) to '{dest_dir}' in {t_faiss_elapsed:.2f}s.")

    # 6. BM25 Keyword Store Indexing
    print("\nBuilding BM25 keyword store index...")
    t_bm25_start = time.time()
    bm25_store = BM25Store()
    bm25_store.index_chunks(chunks)
    
    bm25_path = os.path.join(dest_dir, "bm25_store.pkl")
    bm25_store.save(bm25_path)
    t_bm25_elapsed = time.time() - t_bm25_start
    print(f"Saved BM25 store ({bm25_store.total_chunks} documents) to '{bm25_path}' in {t_bm25_elapsed:.2f}s.")

    t_total = time.time() - t_start
    index_size_mb = get_dir_size_mb(dest_dir)

    print("\n==================================================")
    print("      INGESTION PIPELINE EXECUTION SUMMARY        ")
    print("==================================================")
    print(f"Language              : {language}")
    print(f"Split                 : {split}")
    print(f"Source Parquet File   : {local_parquet}")
    print(f"Raw Parquet Disk Size : {raw_file_size_mb:.2f} MB")
    print(f"Records Processed     : {actual_records_count}")
    print(f"Passages Processed    : {total_passages_count}")
    print(f"Chunks Generated      : {chunks_count}")
    print(f"Embeddings Computed   : {embed_count}")
    print(f"FAISS Vector Count    : {vector_store.total_vectors}")
    print(f"BM25 Document Count   : {bm25_store.total_chunks}")
    print(f"Output Index Dir      : {dest_dir}")
    print(f"Persisted Index Size  : {index_size_mb:.2f} MB")
    print(f"Total Ingestion Time  : {t_total:.2f} seconds")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(description="One-time offline dataset ingestion pipeline for MSMARCO-XI.")
    parser.add_argument("--language", type=str, default="as", help="Language code (e.g. as, hi, bn, gu, kn, ml, mr, ta, te)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="Dataset split")
    parser.add_argument("--max-records", type=int, default=100, help="Max records to process (0 = process entire file)")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save generated indexes")

    args = parser.parse_args()
    ingest_dataset_pipeline(
        language=args.language,
        split=args.split,
        max_records=args.max_records,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
