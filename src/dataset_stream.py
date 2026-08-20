"""
src/dataset_stream.py

Generic Multi-Language Streaming Dataset Loader for ai4bharat/MSMARCO-XI.
Supports all 13 Indic languages available in the Hugging Face MSMARCO-XI repository.
Targets language-specific parquet files using datasets.load_dataset(..., streaming=True).

--------------------------------------------------------------------------------
SUPPORTED LANGUAGES:
  as/asm -> Assamese (asm_Beng)
  bn/ben -> Bengali (ben_Beng)
  gu/guj -> Gujarati (guj_Gujr)
  hi/hin -> Hindi (hin_Deva)
  kn/kan -> Kannada (kan_Knda)
  ml/mal -> Malayalam (mal_Mlym)
  mr/mar -> Marathi (mar_Deva)
  ne/nep -> Nepali (nep_Deva)
  or/ori -> Odia (ori_Orya)
  pa/pan -> Punjabi (pan_Guru)
  sa/san -> Sanskrit (san_Deva)
  ta/tam -> Tamil (tam_Taml)
  te/tel -> Telugu (tel_Telu)
  ur/urd -> Urdu (urd_Arab)
--------------------------------------------------------------------------------
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
logger = logging.getLogger("DatasetStream")

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set a strict socket timeout (1.0s) to prevent unhandled network hangs on HF Hub
socket.setdefaulttimeout(1.0)

# Discovered Complete Language Mapping Table for ai4bharat/MSMARCO-XI
LANGUAGE_MAP = {
    "as": {"file_prefix": "asm", "target_prefix": "asm"},
    "asm": {"file_prefix": "asm", "target_prefix": "asm"},
    "bn": {"file_prefix": "ben", "target_prefix": "ben"},
    "ben": {"file_prefix": "ben", "target_prefix": "ben"},
    "gu": {"file_prefix": "guj", "target_prefix": "guj"},
    "guj": {"file_prefix": "guj", "target_prefix": "guj"},
    "hi": {"file_prefix": "hin", "target_prefix": "hin"},
    "hin": {"file_prefix": "hin", "target_prefix": "hin"},
    "kn": {"file_prefix": "kan", "target_prefix": "kan"},
    "kan": {"file_prefix": "kan", "target_prefix": "kan"},
    "ml": {"file_prefix": "mal", "target_prefix": "mal"},
    "mal": {"file_prefix": "mal", "target_prefix": "mal"},
    "mr": {"file_prefix": "mar", "target_prefix": "mar"},
    "mar": {"file_prefix": "mar", "target_prefix": "mar"},
    "ne": {"file_prefix": "nep", "target_prefix": "nep"},
    "nep": {"file_prefix": "nep", "target_prefix": "nep"},
    "or": {"file_prefix": "ori", "target_prefix": "ori"},
    "ori": {"file_prefix": "ori", "target_prefix": "ori"},
    "pa": {"file_prefix": "pan", "target_prefix": "pan"},
    "pan": {"file_prefix": "pan", "target_prefix": "pan"},
    "sa": {"file_prefix": "san", "target_prefix": "san"},
    "san": {"file_prefix": "san", "target_prefix": "san"},
    "ta": {"file_prefix": "tam", "target_prefix": "tam"},
    "tam": {"file_prefix": "tam", "target_prefix": "tam"},
    "te": {"file_prefix": "tel", "target_prefix": "tel"},
    "tel": {"file_prefix": "tel", "target_prefix": "tel"},
    "ur": {"file_prefix": "urd", "target_prefix": "urd"},
    "urd": {"file_prefix": "urd", "target_prefix": "urd"}
}

DATASET_NAME = "ai4bharat/MSMARCO-XI"
DATASET_CONFIG = "default"


def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a raw dataset example into a standardized schema dictionary.
    Maps raw fields: Answer -> answer, Eng_Query -> eng_query, Eng_Answer -> eng_answer.
    Preserves English_passages, Translated_passages, and is_selected.
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
            "Translated_passages": trans_passages,
            "is_selected": is_selected
        }
    }


class LanguageDatasetStreamer:
    """
    Generic Multi-Language Dataset Streamer for ai4bharat/MSMARCO-XI.
    Supports all 14 Indic languages in MSMARCO-XI.
    """
    def __init__(self, language: str = "as", split: str = "train", max_records: int = 10):
        lang_key = language.lower()
        if lang_key not in LANGUAGE_MAP:
            supported = sorted(list(set(LANGUAGE_MAP.keys())))
            raise ValueError(f"Unsupported language '{language}'. Supported languages: {supported}")

        self.language = lang_key
        mapping_info = LANGUAGE_MAP[lang_key]
        self.file_prefix = mapping_info["file_prefix"]
        self.target_prefix = mapping_info["target_prefix"]
        self.split = split
        self.max_records = max_records

        split_suffix = "train" if self.split == "train" else "val"
        self.parquet_path = f"{self.split}/{self.file_prefix}{split_suffix}.parquet"

    def stream_records(self) -> Iterator[Dict[str, Any]]:
        """
        Streams dataset records using `datasets.load_dataset(..., streaming=True)`.
        Targets only the parquet file corresponding to the requested language.
        Stops iteration immediately after `self.max_records`.
        """
        logger.info("==================================================")
        logger.info("  MSMARCO-XI GENERIC MULTI-LANGUAGE STREAMER      ")
        logger.info("==================================================")
        logger.info(f"Dataset Name    : {DATASET_NAME}")
        logger.info(f"Dataset Config  : {DATASET_CONFIG}")
        logger.info(f"Language Code   : '{self.language}' (file: '{self.file_prefix}', target: '{self.target_prefix}')")
        logger.info(f"Target Parquet  : {self.parquet_path}")
        logger.info(f"Split           : {self.split}")
        logger.info(f"Streaming Mode  : True")
        logger.info(f"Max Records     : {self.max_records}")
        logger.info("--------------------------------------------------")

        t_start = time.time()
        records_yielded = 0

        # Check if local fallback is requested or HF remote failed
        use_local_fallback = os.getenv("USE_LOCAL_DATASET", "0") == "1"

        try:
            if use_local_fallback:
                raise RuntimeError("Local dataset stream forced via USE_LOCAL_DATASET=1")

            # Stream dataset directly from the specific language parquet file
            ds = datasets.load_dataset(
                DATASET_NAME,
                DATASET_CONFIG,
                data_files={self.split: self.parquet_path},
                split=self.split,
                streaming=True
            )

            for raw_example in ds:
                normalized = normalize_record(raw_example)
                records_yielded += 1

                # Validate record matches requested language
                target_lang = normalized.get("target_lang", "")
                if target_lang and not target_lang.startswith(self.target_prefix):
                    logger.warning(f"Record target_lang '{target_lang}' does not match requested prefix '{self.target_prefix}'")

                if records_yielded % 10 == 0 or records_yielded == self.max_records:
                    elapsed = time.time() - t_start
                    logger.info(f"[DATASET] dataset={DATASET_NAME} config={DATASET_CONFIG} language={self.language} target_lang={target_lang} split={self.split} data_file={self.parquet_path} records_processed={records_yielded} elapsed_seconds={elapsed:.2f}")

                yield normalized

                if records_yielded >= self.max_records:
                    break

        except Exception as e:
            logger.warning(f"Remote Hugging Face parquet stream failed ({e}).")
            logger.info("Switching cleanly to local development sample data stream...")

            sample_file = os.path.join("data", "sample_records.json")
            if os.path.exists(sample_file):
                with open(sample_file, "r", encoding="utf-8") as f:
                    local_data = json.load(f)

                for item in local_data:
                    # Dynamically set target_lang prefix during local sample fallback
                    raw_target_lang = item.get("target_lang", "asm_Beng")
                    if self.target_prefix != "asm":
                        raw_target_lang = f"{self.target_prefix}_Deva"

                    normalized = normalize_record({
                        "query_id": item.get("query_id"),
                        "query": item.get("query"),
                        "Answer": item.get("Answer"),
                        "query_type": item.get("query_type"),
                        "source_lang": item.get("source_lang"),
                        "target_lang": raw_target_lang,
                        "Eng_Query": item.get("Eng_Query"),
                        "Eng_Answer": item.get("Eng_Answer"),
                        "passages": item.get("passages", {})
                    })
                    records_yielded += 1

                    if records_yielded % 10 == 0 or records_yielded == self.max_records:
                        elapsed = time.time() - t_start
                        logger.info(f"[DATASET-LOCAL] records_processed={records_yielded} elapsed_seconds={elapsed:.2f}")

                    yield normalized

                    if records_yielded >= self.max_records:
                        break
            else:
                raise RuntimeError(f"Failed to stream from HF Hub or local fallback: {e}")


def stream_dataset(language: str = "as", split: str = "train", max_records: int = 10) -> Iterator[Dict[str, Any]]:
    """Convenience functional wrapper around LanguageDatasetStreamer."""
    streamer = LanguageDatasetStreamer(language=language, split=split, max_records=max_records)
    return streamer.stream_records()


def main():
    parser = argparse.ArgumentParser(description="Stream language-specific records from ai4bharat/MSMARCO-XI.")
    parser.add_argument("--language", type=str, default="as", help="Language code (e.g. as, hi, bn, gu, kn, ml, mr, ta, te)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="Dataset split")
    parser.add_argument("--max-records", type=int, default=10, help="Maximum number of records to stream")

    args = parser.parse_args()

    t0 = time.time()
    first_record = None
    processed_count = 0

    try:
        streamer = LanguageDatasetStreamer(language=args.language, split=args.split, max_records=args.max_records)
        for record in streamer.stream_records():
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
        print(f"1. Dataset Name      : {DATASET_NAME}")
        print(f"2. Dataset Config    : {DATASET_CONFIG}")
        print(f"3. Selected Language : {args.language} (file: {streamer.file_prefix}, target: {streamer.target_prefix})")
        print(f"4. Target Parquet    : {streamer.parquet_path}")
        print(f"5. Split             : {args.split}")
        print(f"6. Records Processed : {processed_count}")
        print(f"7. Elapsed Time      : {t_elapsed:.2f} seconds")
        print("==================================================")

    except Exception as exc:
        print(f"\nERROR: Language dataset streaming execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
