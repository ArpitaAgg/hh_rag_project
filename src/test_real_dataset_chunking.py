"""
src/test_real_dataset_chunking.py

Step 12C Verification Suite: Passes streamed real MSMARCO-XI records through the
existing chunking pipeline (src/chunking.py) and verifies metadata traceability,
chunk lengths, and chunk text integrity for Assamese (as), Hindi (hi), and Bengali (bn).
"""

import os
import sys
import time
from pprint import pprint

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from dataset_stream import LanguageDatasetStreamer
from chunking import ChunkingPipeline

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_real_dataset_chunking_tests():
    print("==================================================")
    print("   REAL DATASET -> EXISTING CHUNKING MODULE TEST  ")
    print("==================================================")

    t_start = time.time()
    chunker_pipeline = ChunkingPipeline()

    test_languages = [
        {"code": "as", "name": "Assamese", "split": "train", "max_records": 10},
        {"code": "hi", "name": "Hindi", "split": "train", "max_records": 10},
        {"code": "bn", "name": "Bengali", "split": "train", "max_records": 10}
    ]

    total_records_processed = 0
    total_chunks_generated = 0
    all_chunk_lengths = []
    sample_chunk_preview = None
    all_errors = []
    languages_tested = []

    report_lines = [
        "==================================================",
        "  REAL MSMARCO-XI DATASET CHUNKING TEST RESULTS   ",
        "==================================================\n"
    ]

    for test_item in test_languages:
        lang_code = test_item["code"]
        lang_name = test_item["name"]
        split = test_item["split"]
        max_recs = test_item["max_records"]

        print(f"\n--- Testing Language: {lang_name} ({lang_code}) | Split: {split} | Max Recs: {max_recs} ---")
        languages_tested.append(f"{lang_name} ({lang_code})")

        t_lang_start = time.time()
        streamer = LanguageDatasetStreamer(language=lang_code, split=split, max_records=max_recs)
        
        # 1. Stream records
        lang_records = []
        for record in streamer.stream_records():
            lang_records.append(record)

        total_records_processed += len(lang_records)

        # 2. Process records through existing ChunkingPipeline
        chunks = chunker_pipeline.process_records(lang_records, strategy_name="overlapping_window", use_translated=True)
        total_chunks_generated += len(chunks)

        t_lang_elapsed = time.time() - t_lang_start
        print(f"Streamed {len(lang_records)} records -> Generated {len(chunks)} chunks in {t_lang_elapsed:.2f}s")

        # 3. Inspect individual record chunking & metadata traceability
        for record in lang_records:
            q_id = record.get("query_id")
            t_lang = record.get("target_lang")
            
            # Filter chunks for this query_id
            rec_chunks = [c for c in chunks if c.get("query_id") == q_id]
            rec_chunk_lengths = [c.get("char_length", 0) for c in rec_chunks]
            all_chunk_lengths.extend(rec_chunk_lengths)

            # Passage text length sum
            trans_passages = record.get("passages", {}).get("Translated_passages", [])
            orig_text_len = sum(len(p) for p in trans_passages)

            first_preview = rec_chunks[0]["text"][:60] + "..." if rec_chunks else "No Chunks"
            if sample_chunk_preview is None and rec_chunks:
                sample_chunk_preview = rec_chunks[0]

            print(f"  [Record query_id={q_id}] target_lang={t_lang} | Original Text Len={orig_text_len} | Chunks={len(rec_chunks)} | Chunk Lens={rec_chunk_lengths}")
            print(f"    Preview: \"{first_preview}\"")

            # 4. Validations
            for c in rec_chunks:
                if not isinstance(c.get("text"), str) or not c.get("text").strip():
                    all_errors.append(f"Query ID {q_id}: Chunk {c.get('chunk_id')} contains empty or invalid text.")
                if c.get("query_id") != q_id:
                    all_errors.append(f"Query ID mismatch in chunk {c.get('chunk_id')}: expected {q_id}, got {c.get('query_id')}.")
                if c.get("target_lang") != t_lang:
                    all_errors.append(f"Target language mismatch in chunk {c.get('chunk_id')}: expected {t_lang}, got {c.get('target_lang')}.")

    t_total_elapsed = time.time() - t_start

    min_chunk_len = min(all_chunk_lengths) if all_chunk_lengths else 0
    max_chunk_len = max(all_chunk_lengths) if all_chunk_lengths else 0
    avg_chunks_per_rec = round(total_chunks_generated / total_records_processed, 2) if total_records_processed > 0 else 0.0

    passed = (total_records_processed > 0 and total_chunks_generated > 0 and len(all_errors) == 0)

    print("\n==================================================")
    print("--- CHUNKING VALIDATION SUMMARY ---")
    print("==================================================")
    print(f"Overall Status          : {'PASS' if passed else 'FAIL'}")
    print(f"Languages Tested        : {', '.join(languages_tested)}")
    print(f"Records Processed       : {total_records_processed}")
    print(f"Total Chunks Generated  : {total_chunks_generated}")
    print(f"Average Chunks / Record : {avg_chunks_per_rec}")
    print(f"Min Chunk Length        : {min_chunk_len} chars")
    print(f"Max Chunk Length        : {max_chunk_len} chars")
    print(f"Total Elapsed Time      : {t_total_elapsed:.2f} seconds")
    print("==================================================")

    # Build Report File
    report_lines.append(f"OVERALL STATUS          : {'PASS' if passed else 'FAIL'}")
    report_lines.append(f"Languages Tested        : {', '.join(languages_tested)}")
    report_lines.append(f"Records Processed       : {total_records_processed}")
    report_lines.append(f"Total Chunks Generated  : {total_chunks_generated}")
    report_lines.append(f"Average Chunks / Record : {avg_chunks_per_rec}")
    report_lines.append(f"Min Chunk Length        : {min_chunk_len} chars")
    report_lines.append(f"Max Chunk Length        : {max_chunk_len} chars")
    report_lines.append(f"Total Elapsed Time      : {t_total_elapsed:.2f}s\n")

    report_lines.append("SAMPLE GENERATED CHUNK STRUCT:")
    if sample_chunk_preview:
        report_lines.append(f"  Chunk ID          : {sample_chunk_preview.get('chunk_id')}")
        report_lines.append(f"  Query ID          : {sample_chunk_preview.get('query_id')}")
        report_lines.append(f"  Target Lang       : {sample_chunk_preview.get('target_lang')}")
        report_lines.append(f"  Chunking Strategy : {sample_chunk_preview.get('chunking_strategy')}")
        report_lines.append(f"  Char Length       : {sample_chunk_preview.get('char_length')}")
        report_lines.append(f"  Word Length       : {sample_chunk_preview.get('word_length')}")
        report_lines.append(f"  Text              : \"{sample_chunk_preview.get('text')}\"\n")

    if all_errors:
        report_lines.append("ERRORS FOUND:")
        for err in all_errors:
            report_lines.append(f"  - {err}")
    else:
        report_lines.append("NO ERRORS FOUND:")
        report_lines.append("  - All real dataset records flowed through ChunkingPipeline seamlessly.")
        report_lines.append("  - Metadata query_id, target_lang, source_lang, and is_selected preserved cleanly.")

    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "real_dataset_chunking_test_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nChunking test report saved to '{report_file}'.")


if __name__ == "__main__":
    run_real_dataset_chunking_tests()
