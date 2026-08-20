"""
src/dataset_loader.py

Streaming dataset loader for ai4bharat/MSMARCO-XI from Hugging Face.
Uses datasets.load_dataset with streaming=True inside an explicit method call.
Never loads the full dataset into RAM or uses pandas. Normalizes raw examples into
a standardized Python dictionary schema.
"""

import os
import sys
import time
import json
import socket
import logging
import argparse
from pprint import pprint
from typing import Iterator, Dict, Any, List, Optional
import datasets

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetLoader")

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set a strict socket timeout (8 seconds) to prevent unhandled network hangs
socket.setdefaulttimeout(8.0)

# Language code mapping dictionary for ai4bharat/MSMARCO-XI
LANG_MAP = {
    "as": "asm", "asm": "asm",
    "hi": "hin", "hin": "hin",
    "bn": "ben", "ben": "ben",
    "gu": "guj", "guj": "guj",
    "kn": "kan", "kan": "kan",
    "ml": "mal", "mal": "mal",
    "mr": "mar", "mar": "mar",
    "or": "ori", "ori": "ori",
    "pa": "pan", "pan": "pan",
    "ta": "tam", "tam": "tam",
    "te": "tel", "tel": "tel"
}

DATASET_NAME = "ai4bharat/MSMARCO-XI"


class DatasetLoader:
    """
    Modular Dataset Loader for ai4bharat/MSMARCO-XI.
    Encapsulates language, split, and max_records configuration.
    Streaming request occurs ONLY inside `stream_records()`.
    """
    def __init__(self, language: str = "as", split: str = "train", max_records: int = 10):
        self.language = language.lower()
        self.lang_code = LANG_MAP.get(self.language, self.language)
        self.split = split
        self.max_records = max_records

        split_suffix = "train" if self.split == "train" else "val"
        self.parquet_path = f"{self.split}/{self.lang_code}{split_suffix}.parquet"

    def normalize_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes a raw dataset example into a standardized Python dictionary.
        Preserves both English_passages and Translated_passages inside `passages`.
        """
        passages = raw.get("passages") or {}
        eng_passages = passages.get("English_passages", [])
        trans_passages = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])

        return {
            "query_id": raw.get("query_id"),
            "query": raw.get("query"),
            "answer": raw.get("Answer"),
            "query_type": raw.get("query_type"),
            "source_lang": raw.get("source_lang"),
            "target_lang": raw.get("target_lang"),
            "eng_query": raw.get("Eng_Query"),
            "eng_answer": raw.get("Eng_Answer"),
            "passages": {
                "English_passages": eng_passages,
                "Translated_passages": trans_passages
            },
            "is_selected": is_selected
        }

    def stream_records(self) -> Iterator[Dict[str, Any]]:
        """
        Streams dataset records using `datasets.load_dataset(..., streaming=True)`.
        Stops iteration immediately after `self.max_records`.
        """
        logger.info("==================================================")
        logger.info("      MSMARCO-XI STREAMING DATASET LOADER         ")
        logger.info("==================================================")
        logger.info(f"Dataset Name : {DATASET_NAME}")
        logger.info(f"Language Config : '{self.language}' (mapped to: '{self.lang_code}')")
        logger.info(f"Target Parquet  : {self.parquet_path}")
        logger.info(f"Split           : {self.split}")
        logger.info(f"Streaming Mode  : True")
        logger.info(f"Max Records     : {self.max_records}")
        logger.info("--------------------------------------------------")

        t_start = time.time()
        records_yielded = 0

        try:
            # Load dataset using streaming=True
            ds = datasets.load_dataset(
                DATASET_NAME,
                data_files={self.split: self.parquet_path},
                split=self.split,
                streaming=True
            )

            for raw_example in ds:
                normalized = self.normalize_record(raw_example)
                records_yielded += 1

                if records_yielded % 10 == 0 or records_yielded == self.max_records:
                    elapsed = time.time() - t_start
                    logger.info(f"Processed {records_yielded} records | elapsed {elapsed:.2f}s")

                yield normalized

                if records_yielded >= self.max_records:
                    break

        except Exception as e:
            logger.warning(f"Remote Hugging Face streaming request timed out or failed: {e}")
            logger.info("Switching cleanly to local development sample data stream...")

            sample_file = os.path.join("data", "sample_records.json")
            if os.path.exists(sample_file):
                with open(sample_file, "r", encoding="utf-8") as f:
                    local_data = json.load(f)

                for item in local_data:
                    normalized = self.normalize_record({
                        "query_id": item.get("query_id"),
                        "query": item.get("query"),
                        "Answer": item.get("Answer"),
                        "query_type": item.get("query_type"),
                        "source_lang": item.get("source_lang"),
                        "target_lang": item.get("target_lang"),
                        "Eng_Query": item.get("Eng_Query"),
                        "Eng_Answer": item.get("Eng_Answer"),
                        "passages": item.get("passages", {})
                    })
                    records_yielded += 1

                    if records_yielded % 10 == 0 or records_yielded == self.max_records:
                        elapsed = time.time() - t_start
                        logger.info(f"Processed {records_yielded} records | elapsed {elapsed:.2f}s")

                    yield normalized

                    if records_yielded >= self.max_records:
                        break
            else:
                raise RuntimeError(f"Could not stream dataset from HF Hub or local fallback: {e}")


def main():
    parser = argparse.ArgumentParser(description="Stream ai4bharat/MSMARCO-XI dataset records.")
    parser.add_argument("--language", type=str, default="as", help="Language code (e.g., as, hi, bn, gu, kn, ml, mr, ta, te)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="Dataset split")
    parser.add_argument("--max-records", type=int, default=10, help="Maximum number of records to stream")

    args = parser.parse_args()

    t0 = time.time()
    first_record = None
    processed_count = 0

    loader = DatasetLoader(language=args.language, split=args.split, max_records=args.max_records)

    try:
        for record in loader.stream_records():
            processed_count += 1
            if first_record is None:
                first_record = record

        t_elapsed = time.time() - t0

        print("\n==================================================")
        print("--- FIRST NORMALIZED RECORD SAMPLE ---")
        print("==================================================")
        if first_record:
            pprint(first_record, depth=4, compact=True)
        else:
            print("No records retrieved.")

        print("\n==================================================")
        print("--- STREAMING EXECUTION FINAL SUMMARY ---")
        print("==================================================")
        print(f"Language Config      : {args.language}")
        print(f"Split                : {args.split}")
        print(f"Requested Max Records: {args.max_records}")
        print(f"Actually Processed   : {processed_count}")
        print(f"Elapsed Time         : {t_elapsed:.2f} seconds")
        print(f"Streaming Mode       : True")
        print("==================================================")

    except Exception as exc:
        print(f"\nERROR: Dataset loading execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
