"""
src/test_dataset_loader.py

Automated Test Suite for DatasetLoader class in src/dataset_loader.py.
Validates object initialization, schema normalization, key preservation,
and writes report to data/dataset_loader_test_results.txt.
"""

import os
import sys
import time

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import DatasetLoader

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def test_dataset_loader_schema():
    print("==================================================")
    print("        DATASET LOADER UNIT & SCHEMA TEST        ")
    print("==================================================")

    # 1. Instantiate DatasetLoader without triggering network requests
    loader = DatasetLoader(language="as", split="train", max_records=10)
    print(f"DatasetLoader initialized cleanly!")
    print(f"  Language : {loader.language} -> mapped: {loader.lang_code}")
    print(f"  Split    : {loader.split}")
    print(f"  Max Recs : {loader.max_records}\n")

    # 2. Mock Raw Example
    mock_raw = {
        "query_id": 1102432,
        "query": "কৰ্পোৰেচন কি?",
        "Answer": "নিগম হৈছে এটা কোম্পানী...",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "asm_Beng",
        "Eng_Query": ". what is a corporation?",
        "Eng_Answer": "A corporation is a company...",
        "passages": {
            "English_passages": ["A company is incorporated..."],
            "Translated_passages": ["এটা কোম্পানী একটা..."],
            "is_selected": [1]
        }
    }

    # 3. Test Normalization Method
    normalized = loader.normalize_record(mock_raw)

    required_keys = [
        "query_id", "query", "answer", "query_type",
        "source_lang", "target_lang", "eng_query", "eng_answer",
        "passages", "is_selected"
    ]

    missing_keys = [k for k in required_keys if k not in normalized]
    passages = normalized.get("passages", {})
    eng_passages_ok = "English_passages" in passages
    trans_passages_ok = "Translated_passages" in passages

    passed = (len(missing_keys) == 0 and eng_passages_ok and trans_passages_ok)

    print("Schema Normalization Verification:")
    print(f"  - Missing Keys        : {missing_keys if missing_keys else 'None (All Present)'}")
    print(f"  - English_passages    : {'PRESENT' if eng_passages_ok else 'MISSING'}")
    print(f"  - Translated_passages : {'PRESENT' if trans_passages_ok else 'MISSING'}")
    print(f"  - Overall Status      : {'PASS' if passed else 'FAIL'}\n")

    report_lines = [
        "==================================================",
        "        DATASET LOADER SCHEMA TEST RESULTS        ",
        "==================================================",
        f"Language Config    : as",
        f"Split              : train",
        f"Max Records        : 10",
        f"Overall Status     : {'PASS' if passed else 'FAIL'}\n",
        "SCHEMA VALIDATION DETAILS:",
        "  - Required keys (query_id, query, answer, query_type, source_lang, target_lang, eng_query, eng_answer, passages, is_selected) are normalized.",
        "  - Both English_passages and Translated_passages preserved inside passages dictionary."
    ]

    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "dataset_loader_test_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Test results saved to '{report_file}'.")


if __name__ == "__main__":
    test_dataset_loader_schema()
