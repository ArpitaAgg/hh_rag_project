"""
src/test_chunking.py

Test runner and verification script for the modular chunking system.
Loads sample records, runs all 3 chunking strategies, verifies metadata,
prints statistics, and saves a summary report to data/chunking_results.txt.
"""

import os
import sys
import json
from chunking import ChunkingPipeline

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

REQUIRED_METADATA_FIELDS = [
    "chunk_id",
    "original_passage_id",
    "source_lang",
    "target_lang",
    "query_id",
    "query_type",
    "chunking_strategy",
    "passage_index",
    "text",
    "is_selected"
]

def calculate_stats(chunks):
    if not chunks:
        return {"count": 0, "avg_char": 0, "min_char": 0, "max_char": 0, "avg_word": 0, "min_word": 0, "max_word": 0}
    
    char_lens = [c["char_length"] for c in chunks]
    word_lens = [c["word_length"] for c in chunks]
    
    return {
        "count": len(chunks),
        "avg_char": sum(char_lens) / len(chunks),
        "min_char": min(char_lens),
        "max_char": max(char_lens),
        "avg_word": sum(word_lens) / len(chunks),
        "min_word": min(word_lens),
        "max_word": max(word_lens)
    }

def verify_metadata(chunks, strategy_name):
    missing_report = []
    for idx, chunk in enumerate(chunks):
        for field in REQUIRED_METADATA_FIELDS:
            if field not in chunk:
                missing_report.append(f"Chunk #{idx} missing field '{field}'")
    
    if missing_report:
        print(f"FAILED metadata verification for strategy '{strategy_name}': {missing_report[:3]}")
        return False
    return True

def run_tests():
    sample_file = os.path.join("data", "sample_records.json")
    if not os.path.exists(sample_file):
        print(f"Error: Sample data file '{sample_file}' not found. Please complete Step 2 first.")
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Count total passages across records
    total_passages = sum(
        len(r.get("passages", {}).get("Translated_passages", []))
        for r in records
    )

    pipeline = ChunkingPipeline()
    strategies = ["sentence", "overlapping_window", "adaptive"]
    results = {}

    print("==================================================")
    print("      CHUNKING SYSTEM TEST & EVALUATION           ")
    print("==================================================")
    print(f"Total Sample Records Loaded: {len(records)}")
    print(f"Total Original Passages     : {total_passages}\n")

    report_lines = [
        "==================================================",
        "     CHUNKING STRATEGIES COMPARISON REPORT        ",
        "==================================================",
        f"Total Sample Records : {len(records)}",
        f"Total Input Passages : {total_passages}\n"
    ]

    for strat in strategies:
        chunks = pipeline.process_records(records, strategy_name=strat, use_translated=True)
        stats = calculate_stats(chunks)
        results[strat] = {"chunks": chunks, "stats": stats}
        
        # Verify metadata
        metadata_valid = verify_metadata(chunks, strat)
        
        print(f"--- Strategy: {strat.upper()} ---")
        print(f"  Chunks Generated : {stats['count']}")
        print(f"  Avg Chunk Length : {stats['avg_char']:.1f} chars ({stats['avg_word']:.1f} words)")
        print(f"  Min Chunk Length : {stats['min_char']} chars ({stats['min_word']} words)")
        print(f"  Max Chunk Length : {stats['max_char']} chars ({stats['max_word']} words)")
        print(f"  Metadata Valid   : {'YES (All 10 required fields present)' if metadata_valid else 'NO'}\n")
        
        report_lines.append(f"--- Strategy: {strat.upper()} ---")
        report_lines.append(f"  - Total Chunks Produced : {stats['count']}")
        report_lines.append(f"  - Average Chunk Length  : {stats['avg_char']:.1f} chars ({stats['avg_word']:.1f} words)")
        report_lines.append(f"  - Min / Max Length      : {stats['min_char']} chars / {stats['max_char']} chars")
        report_lines.append(f"  - Metadata Completeness : {'PASS' if metadata_valid else 'FAIL'}")
        if chunks:
            report_lines.append(f"  - Sample Chunk Text     : \"{chunks[0]['text'][:100]}...\"")
        report_lines.append("")

        # Print Sample Chunk to Console
        if chunks:
            print("  Sample Chunk Output:")
            print(f"    [ID: {chunks[0]['chunk_id']}]")
            print(f"    [Strategy: {chunks[0]['chunking_strategy']}]")
            print(f"    [Selected: {chunks[0]['is_selected']}]")
            print(f"    Text: {chunks[0]['text'][:120]}...\n")

    # Save report to data/chunking_results.txt
    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "chunking_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Summary report successfully written to '{report_file}'.")

if __name__ == "__main__":
    run_tests()
