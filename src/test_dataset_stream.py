"""
src/test_dataset_stream.py

Automated Multi-Language Test Suite for Step 12B LanguageDatasetStreamer.
Evaluates Assamese (as), Hindi (hi), Bengali (bn), Validation split, and Unsupported language handling.
Writes report to data/dataset_stream_test_results.txt.
"""

import os
import sys
import time

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from dataset_stream import LanguageDatasetStreamer, LANGUAGE_MAP

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Enable fast deterministic local dataset stream testing
os.environ["USE_LOCAL_DATASET"] = "1"


def run_dataset_stream_tests():
    print("==================================================")
    print("      DISCOVERED LANGUAGE MAP IN MSMARCO-XI      ")
    print("==================================================")
    print(f"Total Languages Mapped: {len(LANGUAGE_MAP)}")
    for lang, info in sorted(LANGUAGE_MAP.items()):
        print(f"  - '{lang}' -> parquet prefix: '{info['file_prefix']}', target_lang prefix: '{info['target_prefix']}'")

    print("\n==================================================")
    print("    GENERIC DATASET STREAMING MULTI-LANG TESTS    ")
    print("==================================================")

    test_configs = [
        {"name": "Assamese Train (as)", "lang": "as", "split": "train", "expected_prefix": "asm"},
        {"name": "Hindi Train (hi)", "lang": "hi", "split": "train", "expected_prefix": "hin"},
        {"name": "Bengali Train (bn)", "lang": "bn", "split": "train", "expected_prefix": "ben"},
        {"name": "Assamese Validation (as)", "lang": "as", "split": "validation", "expected_prefix": "asm"},
        {"name": "Unsupported Language (xx)", "lang": "xx", "split": "train", "expected_prefix": None}
    ]

    results = []
    report_lines = [
        "==================================================",
        "     DATASET STREAM MULTI-LANGUAGE TEST REPORT    ",
        "==================================================\n"
    ]

    for item in test_configs:
        t0 = time.time()
        c_name = item["name"]
        lang = item["lang"]
        split = item["split"]
        expected_prefix = item["expected_prefix"]

        print(f"\nTesting: {c_name}...")

        if lang == "xx":
            # Test Unsupported Language Exception
            try:
                streamer = LanguageDatasetStreamer(language=lang, split=split, max_records=10)
                t_elapsed = (time.time() - t0) * 1000
                results.append((c_name, False, f"Failed: Did not raise ValueError for unsupported language '{lang}'"))
                print(f"  Result: FAIL | Expected ValueError not raised.")
            except ValueError as ve:
                t_elapsed = (time.time() - t0) * 1000
                results.append((c_name, True, f"Passed: Caught expected ValueError ('{ve}')"))
                print(f"  Result: PASS | Correctly caught expected error: {ve}")
            continue

        # Valid Language Test
        try:
            streamer = LanguageDatasetStreamer(language=lang, split=split, max_records=10)
            records = []
            for rec in streamer.stream_records():
                records.append(rec)

            t_elapsed = time.time() - t0

            passed = True
            errors = []

            if len(records) == 0:
                passed = False
                errors.append("No records returned from streamer.")
            elif len(records) > 10:
                passed = False
                errors.append(f"Returned {len(records)} records, exceeding max_records=10.")
            else:
                for idx, rec in enumerate(records, start=1):
                    if not rec.get("query_id"):
                        passed = False; errors.append(f"Rec #{idx} missing query_id.")
                    if not rec.get("query"):
                        passed = False; errors.append(f"Rec #{idx} missing query.")
                    if not rec.get("answer"):
                        passed = False; errors.append(f"Rec #{idx} missing answer.")

                    target_lang = rec.get("target_lang", "")
                    if expected_prefix and not target_lang.startswith(expected_prefix):
                        passed = False
                        errors.append(f"Rec #{idx} target_lang '{target_lang}' does not start with '{expected_prefix}'.")

                    passages = rec.get("passages", {})
                    if "English_passages" not in passages:
                        passed = False; errors.append(f"Rec #{idx} missing English_passages.")
                    if "Translated_passages" not in passages:
                        passed = False; errors.append(f"Rec #{idx} missing Translated_passages.")
                    if "is_selected" not in passages:
                        passed = False; errors.append(f"Rec #{idx} missing is_selected.")

            detail_msg = f"Records: {len(records)} | Elapsed: {t_elapsed:.2f}s | Target Parquet: {streamer.parquet_path}"
            results.append((c_name, passed, detail_msg if passed else "; ".join(errors)))
            print(f"  Result: {'PASS' if passed else 'FAIL'} | {detail_msg}")

        except Exception as e:
            results.append((c_name, False, f"Unhandled exception: {e}"))
            print(f"  Result: FAIL | Exception: {e}")

    # Compile Summary
    passed_count = sum(1 for name, ok, desc in results if ok)
    report_lines.append(f"TOTAL TESTS EVALUATED : {len(results)}")
    report_lines.append(f"PASSED                : {passed_count} / {len(results)}\n")

    for idx, (name, ok, desc) in enumerate(results, start=1):
        report_lines.append(f"Case #{idx}: {name}")
        report_lines.append(f"  Status : {'PASS' if ok else 'FAIL'}")
        report_lines.append(f"  Details: {desc}\n")

    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "dataset_stream_test_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nMulti-language dataset stream report saved to '{report_file}'.")


if __name__ == "__main__":
    run_dataset_stream_tests()
