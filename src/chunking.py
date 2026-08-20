"""
src/chunking.py

A modular, reusable chunking library for splitting text passages into smaller,
retrievable units while preserving metadata for RAG pipelines.
"""

import re
from typing import List, Dict, Any, Optional

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using standard English punctuation (. ! ?)
    as well as Indic sentence delimiters such as Purna Viram (।).
    Handles multilingual UTF-8 text cleanly.
    """
    if not text or not text.strip():
        return []
    
    # Regex pattern matching standard sentence terminators (. ! ?) and Indic Purna Viram (।)
    # followed by whitespace or end of string.
    raw_sentences = re.split(r'(?<=[.!?।॥])\s+', text.strip())
    
    # Filter out empty or whitespace-only strings
    sentences = [s.strip() for s in raw_sentences if s and s.strip()]
    
    # Fallback if no sentence boundary was found
    return sentences if sentences else [text.strip()]


class SentenceChunker:
    """
    Strategy 1: Sentence-based Chunking
    Splits a passage into individual sentences.
    """
    def __init__(self, strategy_name: str = "sentence"):
        self.strategy_name = strategy_name

    def chunk_passage(self, passage_text: str, metadata: Dict[str, Any], passage_idx: int) -> List[Dict[str, Any]]:
        if not passage_text or not passage_text.strip():
            return []

        sentences = split_into_sentences(passage_text)
        chunks = []
        
        for s_idx, sentence in enumerate(sentences):
            chunk_id = f"chunk_{metadata.get('query_id', 0)}_p{passage_idx}_s{s_idx}"
            chunk = {
                "chunk_id": chunk_id,
                "original_passage_id": f"p_{passage_idx}",
                "passage_index": passage_idx,
                "source_lang": metadata.get("source_lang", ""),
                "target_lang": metadata.get("target_lang", ""),
                "query_id": metadata.get("query_id"),
                "query_type": metadata.get("query_type", ""),
                "chunking_strategy": self.strategy_name,
                "text": sentence,
                "is_selected": bool(metadata.get("is_selected", False)),
                "char_length": len(sentence),
                "word_length": len(sentence.split())
            }
            chunks.append(chunk)
            
        return chunks


class OverlappingWindowChunker:
    """
    Strategy 2: Overlapping Sentence-Window Chunking
    Groups multiple sentences into sliding windows with configurable window size and overlap.
    """
    def __init__(self, window_size: int = 3, overlap: int = 1, strategy_name: str = "overlapping_window"):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if overlap >= window_size:
            raise ValueError("overlap must be strictly less than window_size")
            
        self.window_size = window_size
        self.overlap = overlap
        self.strategy_name = strategy_name

    def chunk_passage(self, passage_text: str, metadata: Dict[str, Any], passage_idx: int) -> List[Dict[str, Any]]:
        if not passage_text or not passage_text.strip():
            return []

        sentences = split_into_sentences(passage_text)
        if not sentences:
            return []

        chunks = []
        step = self.window_size - self.overlap
        chunk_count = 0

        for i in range(0, len(sentences), step):
            window_sentences = sentences[i : i + self.window_size]
            combined_text = " ".join(window_sentences)
            
            chunk_id = f"chunk_{metadata.get('query_id', 0)}_p{passage_idx}_w{chunk_count}"
            chunk = {
                "chunk_id": chunk_id,
                "original_passage_id": f"p_{passage_idx}",
                "passage_index": passage_idx,
                "source_lang": metadata.get("source_lang", ""),
                "target_lang": metadata.get("target_lang", ""),
                "query_id": metadata.get("query_id"),
                "query_type": metadata.get("query_type", ""),
                "chunking_strategy": self.strategy_name,
                "text": combined_text,
                "is_selected": bool(metadata.get("is_selected", False)),
                "char_length": len(combined_text),
                "word_length": len(combined_text.split()),
                "window_size": self.window_size,
                "overlap": self.overlap
            }
            chunks.append(chunk)
            chunk_count += 1
            
            # Stop if the current window reached the end of the sentence list
            if i + self.window_size >= len(sentences):
                break

        return chunks


class AdaptiveChunker:
    """
    Strategy 3: Adaptive Chunking
    Dynamically selects chunking behavior based on passage length (character length):
    - Very short passages (< short_threshold): Remain as a single chunk.
    - Medium passages (< medium_threshold): Group into small 2-sentence units.
    - Long passages (>= medium_threshold): Use overlapping sentence-window chunking.
    """
    def __init__(
        self,
        short_threshold: int = 150,
        medium_threshold: int = 400,
        strategy_name: str = "adaptive"
    ):
        self.short_threshold = short_threshold
        self.medium_threshold = medium_threshold
        self.strategy_name = strategy_name
        
        # Sub-chunkers for internal processing
        self.sentence_chunker = SentenceChunker(strategy_name="adaptive_sentence")
        self.medium_window_chunker = OverlappingWindowChunker(window_size=2, overlap=0, strategy_name="adaptive_medium")
        self.long_window_chunker = OverlappingWindowChunker(window_size=3, overlap=1, strategy_name="adaptive_long")

    def chunk_passage(self, passage_text: str, metadata: Dict[str, Any], passage_idx: int) -> List[Dict[str, Any]]:
        if not passage_text or not passage_text.strip():
            return []

        text_len = len(passage_text.strip())

        # 1. Short Passage Rule -> Return single intact chunk
        if text_len < self.short_threshold:
            chunk_id = f"chunk_{metadata.get('query_id', 0)}_p{passage_idx}_adapt_single"
            return [{
                "chunk_id": chunk_id,
                "original_passage_id": f"p_{passage_idx}",
                "passage_index": passage_idx,
                "source_lang": metadata.get("source_lang", ""),
                "target_lang": metadata.get("target_lang", ""),
                "query_id": metadata.get("query_id"),
                "query_type": metadata.get("query_type", ""),
                "chunking_strategy": f"{self.strategy_name}_short",
                "text": passage_text.strip(),
                "is_selected": bool(metadata.get("is_selected", False)),
                "char_length": text_len,
                "word_length": len(passage_text.strip().split())
            }]

        # 2. Medium Passage Rule -> 2-sentence non-overlapping chunks
        elif text_len < self.medium_threshold:
            chunks = self.medium_window_chunker.chunk_passage(passage_text, metadata, passage_idx)
            for c in chunks:
                c["chunking_strategy"] = f"{self.strategy_name}_medium"
            return chunks

        # 3. Long Passage Rule -> 3-sentence overlapping window chunks
        else:
            chunks = self.long_window_chunker.chunk_passage(passage_text, metadata, passage_idx)
            for c in chunks:
                c["chunking_strategy"] = f"{self.strategy_name}_long"
            return chunks


class ChunkingPipeline:
    """
    Main Chunking Pipeline manager.
    Runs chunking across records and collects chunks with metadata.
    """
    def __init__(self):
        self.strategies = {
            "sentence": SentenceChunker(),
            "overlapping_window": OverlappingWindowChunker(window_size=3, overlap=1),
            "adaptive": AdaptiveChunker(short_threshold=150, medium_threshold=400)
        }

    def process_records(self, records: List[Dict[str, Any]], strategy_name: str, use_translated: bool = True) -> List[Dict[str, Any]]:
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {list(self.strategies.keys())}")
        
        chunker = self.strategies[strategy_name]
        all_chunks = []

        for record in records:
            passages_dict = record.get("passages", {})
            passage_list = passages_dict.get("Translated_passages", []) if use_translated else passages_dict.get("English_passages", [])
            is_selected_flags = passages_dict.get("is_selected", [])

            for p_idx, p_text in enumerate(passage_list):
                selected_flag = is_selected_flags[p_idx] if p_idx < len(is_selected_flags) else 0
                
                metadata = {
                    "query_id": record.get("query_id"),
                    "query_type": record.get("query_type"),
                    "source_lang": record.get("source_lang"),
                    "target_lang": record.get("target_lang"),
                    "is_selected": bool(selected_flag == 1 or selected_flag is True)
                }

                chunks = chunker.chunk_passage(p_text, metadata, p_idx)
                all_chunks.extend(chunks)

        return all_chunks
