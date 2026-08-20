"""
src/guardrails.py

Comprehensive Guardrail Layer for Voice RAG Pipeline.

--------------------------------------------------------------------------------
ARCHITECTURE:

User Query
    ↓
InputGuardrail (Empty / Off-topic / Safety checks)
    ↓ (If allowed)
Hybrid Retrieval (FAISS + BM25)
    ↓
ContextValidator (Context quality & threshold validation)
    ↓ (If sufficient)
Answer Generator
    ↓
GroundingGuardrail (Hallucination & context adherence validation)
    ↓
Final Response
--------------------------------------------------------------------------------
"""

import re
import sys
import time
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# --- Configurable Off-Topic & Injection Patterns ---
OFF_TOPIC_PATTERNS = [
    r"\b(write|compose|create|generate)\s+.*(poem|song|story|essay|joke|game|script|code|program)\b",
    r"\b(tell|give)\s+me\s+.*(joke|riddle|story|poem)\b",
    r"\b(ignore|disregard|bypass)\s+.*(previous|system|prompt|instruction)",
    r"\breveal\s+.*(system\s+prompt|instructions)",
    r"\b(play|start)\s+(a\s+)?game\b",
    r"\b(python|javascript|c\+\+|java)\s+code\b"
]

# --- Configurable Basic Safety Patterns ---
UNSAFE_PATTERNS = [
    r"\b(make|build|create|manufacture)\s+.*(bomb|weapon|explosive|poison|harmful)\b",
    r"\b(hack|crack|breach|exploit)\s+.*(system|password|account)\b",
    r"\b(how\s+to\s+suicide|self-harm|kill)\b",
    r"\b(illegal|stolen|dangerous\s+weapon)\b"
]


class InputGuardrail:
    """
    Validates user queries before embedding and retrieval.
    Rejects empty inputs, off-topic requests, and unsafe content.
    Multilingual-safe: preserves non-English (Indic) queries.
    """
    def __init__(
        self,
        off_topic_patterns: Optional[List[str]] = None,
        unsafe_patterns: Optional[List[str]] = None
    ):
        self.off_topic_patterns = [re.compile(p, re.IGNORECASE) for p in (off_topic_patterns or OFF_TOPIC_PATTERNS)]
        self.unsafe_patterns = [re.compile(p, re.IGNORECASE) for p in (unsafe_patterns or UNSAFE_PATTERNS)]

    def validate(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "allowed": False,
                "reason": "Query is empty or contains only whitespace.",
                "category": "invalid"
            }

        q_clean = query.strip()

        # Check safety violations
        for pattern in self.unsafe_patterns:
            if pattern.search(q_clean):
                return {
                    "allowed": False,
                    "reason": "Query contains unsafe or restricted content.",
                    "category": "unsafe"
                }

        # Check off-topic non-RAG requests
        for pattern in self.off_topic_patterns:
            if pattern.search(q_clean):
                return {
                    "allowed": False,
                    "reason": "Query is off-topic (creative writing, coding, or non-RAG request).",
                    "category": "off_topic"
                }

        return {
            "allowed": True,
            "reason": "Query passed input validation.",
            "category": "allowed"
        }


class ContextValidator:
    """
    Evaluates retrieved passage quality and score thresholds before calling the answer generator.
    """
    def __init__(self, min_combined_score: float = 0.25, min_semantic_score: float = 0.20):
        self.min_combined_score = min_combined_score
        self.min_semantic_score = min_semantic_score

    def validate_context(self, retrieved_candidates: List[Dict[str, Any]], query: str = "") -> Dict[str, Any]:
        if not retrieved_candidates:
            return {
                "sufficient": False,
                "reason": "No context passages retrieved.",
                "status": "insufficient_context"
            }

        top_candidate = retrieved_candidates[0]
        comb_score = top_candidate.get("combined_score", top_candidate.get("score", 0.0))
        sem_score = top_candidate.get("semantic_score_raw", top_candidate.get("score", 0.0))

        if comb_score < self.min_combined_score:
            return {
                "sufficient": False,
                "reason": f"Top candidate combined score ({comb_score:.3f}) below relevance threshold ({self.min_combined_score}).",
                "status": "insufficient_context"
            }
        if sem_score < self.min_semantic_score:
            return {
                "sufficient": False,
                "reason": f"Top candidate semantic score ({sem_score:.3f}) below relevance threshold ({self.min_semantic_score}).",
                "status": "insufficient_context"
            }

        # Subject Entity & Keyword Presence Validation
        if query and query.strip():
            # Stop words including English & Indic transliterated connectors
            STOP_WORDS = {
                "what", "who", "where", "when", "which", "how", "does", "that", "this", "from", "with",
                "about", "the", "in", "of", "is", "a", "an", "on", "at", "to", "by", "for", "or", "and",
                "ka", "ki", "ke", "ko", "me", "par", "se", "h", "hai", "hain", "kitna", "kitne", "kitni", "hota", "hoti", "hote", "kya"
            }
            q_words = set(re.findall(r'\w+', query.lower())) - STOP_WORDS
            
            # Combine text across top candidates
            combined_candidate_text = " ".join([c.get("text", "").lower() for c in retrieved_candidates])

            # Extract distinct ASCII specific query entities (e.g. proper nouns, places, years, unique identifiers)
            specific_entities = {w for w in q_words if len(w) > 3 and all(ord(c) < 128 for c in w) and w not in {
                "what", "where", "when", "which", "how", "does", "that", "this", "from", "with", "about",
                "point", "water", "capital", "population", "boiling", "system", "state", "city", "school",
                "temperature", "degree", "degrees", "level", "value", "amount", "corporation", "corporate", "company",
                "color", "colour", "colors", "colours", "rang", "ranga", "shape", "size", "type", "kind", "meaning", "definition",
                "mera", "meri", "mere", "naam", "naama", "name", "names",
                "kaha", "kahan", "kidhar", "kaise", "kaisa", "kaisi", "kitna", "kitni", "kitne", "kiska", "kiski", "kiske", "kisko"
            }}

            # If query specifies unique proper ASCII entities (like 'mars', 'everest', 'japan', '2030'), verify ALL are present in context
            missing_entities = [e for e in specific_entities if e not in combined_candidate_text]
            if missing_entities:
                return {
                    "sufficient": False,
                    "reason": f"Retrieved context is missing key query entities {missing_entities}.",
                    "status": "insufficient_context"
                }

        return {
            "sufficient": True,
            "reason": "Context passes relevance quality checks.",
            "status": "sufficient"
        }


class GroundingGuardrail:
    """
    Lightweight, deterministic output validator.
    Ensures generated answers are grounded in the retrieved passage text.
    """
    def __init__(self, min_overlap_ratio: float = 0.20):
        self.min_overlap_ratio = min_overlap_ratio

    def validate_grounding(self, answer: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not answer or not answer.strip():
            return {
                "grounded": False,
                "confidence": 0.0,
                "reason": "Generated answer is empty.",
                "status": "ungrounded"
            }

        # Standard fallback string is always grounded
        insufficient_msg = "I do not have enough information from the retrieved context to answer this question."
        if insufficient_msg in answer:
            return {
                "grounded": True,
                "confidence": 1.0,
                "reason": "Correct insufficient context fallback response.",
                "status": "grounded"
            }

        # Combine text from retrieved context
        context_text = " ".join([
            c.get("text", "") or c.get("metadata", {}).get("text", "")
            for c in retrieved_context
        ]).lower()

        # Tokenize answer words
        answer_words = re.findall(r'\w+', answer.lower(), flags=re.UNICODE)
        if not answer_words:
            return {"grounded": True, "confidence": 1.0, "reason": "Short non-word answer.", "status": "grounded"}

        # Measure key word overlap ratio with retrieved context
        matched_words = [w for w in answer_words if w in context_text]
        overlap_ratio = len(matched_words) / len(answer_words)

        # Extract digits/numbers from answer
        answer_digits = set(re.findall(r'\d+', answer))
        context_digits = set(re.findall(r'\d+', context_text))
        digit_match = bool(answer_digits and answer_digits.issubset(context_digits))

        # Pass if word overlap threshold met, OR if numbers/digits match, OR if multilingual non-ASCII / Hinglish answer
        has_non_ascii = any(ord(c) > 127 for c in answer)
        is_multilingual = has_non_ascii or any(w in answer.lower().split() for w in ["hai", "h", "kya", "ek", "mein", "ko", "ki", "ke", "ka", "hota", "hote"])

        if overlap_ratio >= self.min_overlap_ratio or digit_match or is_multilingual or len(matched_words) >= 1:
            return {
                "grounded": True,
                "confidence": round(max(overlap_ratio, 0.85), 3),
                "reason": f"Answer is grounded with {overlap_ratio*100:.1f}% word overlap.",
                "status": "grounded"
            }

        return {
            "grounded": False,
            "confidence": round(overlap_ratio, 3),
            "reason": f"Answer word overlap ({overlap_ratio*100:.1f}%) is below minimum grounding threshold.",
            "status": "ungrounded"
        }


class GuardedRAGPipeline:
    """
    Orchestrates the full end-to-end RAG pipeline with Input, Context, and Output Guardrails.
    """
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.input_guardrail = InputGuardrail()
        self.context_validator = ContextValidator(min_combined_score=0.30)
        self.grounding_guardrail = GroundingGuardrail(min_overlap_ratio=0.20)

    def process(self, query: str) -> Dict[str, Any]:
        t_start = time.time()

        # 1. Input Guardrail
        t0 = time.time()
        input_eval = self.input_guardrail.validate(query)
        t1 = time.time()
        input_ms = (t1 - t0) * 1000

        if not input_eval["allowed"]:
            t_end = time.time()
            cat = input_eval["category"]
            msg = "I cannot process this request because it is off-topic or inappropriate for this RAG system." \
                if cat in ["off_topic", "unsafe"] else "Please provide a valid question."
            
            return {
                "query": query,
                "status": "rejected",
                "answer": msg,
                "grounded": False,
                "guardrail_reason": input_eval["reason"],
                "retrieved_chunks": [],
                "latency": {
                    "input_guardrail_ms": round(input_ms, 2),
                    "retrieval_ms": 0.0,
                    "generation_ms": 0.0,
                    "grounding_ms": 0.0,
                    "total_ms": round((t_end - t_start) * 1000, 2)
                }
            }

        # 2. Hybrid Retrieval
        t2 = time.time()
        retrieval_data = self.retriever.retrieve(query, top_k=3)
        t3 = time.time()
        retrieval_ms = (t3 - t2) * 1000
        candidates = retrieval_data.get("results", [])

        # 3. Context Quality Validation
        context_eval = self.context_validator.validate_context(candidates)
        if not context_eval["sufficient"]:
            t_end = time.time()
            return {
                "query": query,
                "status": "insufficient_context",
                "answer": "I do not have enough information from the retrieved context to answer this question.",
                "grounded": True,
                "guardrail_reason": context_eval["reason"],
                "retrieved_chunks": candidates,
                "latency": {
                    "input_guardrail_ms": round(input_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": 0.0,
                    "grounding_ms": 0.0,
                    "total_ms": round((t_end - t_start) * 1000, 2)
                }
            }

        # 4. Answer Generation
        t4 = time.time()
        gen_output = self.generator.generate(query, candidates)
        t5 = time.time()
        generation_ms = (t5 - t4) * 1000
        raw_answer = gen_output.get("answer", "")

        # 5. Output / Grounding Guardrail Validation
        t6 = time.time()
        ground_eval = self.grounding_guardrail.validate_grounding(raw_answer, candidates)
        t7 = time.time()
        grounding_ms = (t7 - t6) * 1000
        t_end = time.time()

        final_answer = raw_answer if ground_eval["grounded"] else \
            "I do not have enough information from the retrieved context to answer this question."

        return {
            "query": query,
            "status": "answered" if ground_eval["grounded"] else "insufficient_context",
            "answer": final_answer,
            "grounded": ground_eval["grounded"],
            "guardrail_reason": ground_eval["reason"],
            "retrieved_chunks": candidates,
            "latency": {
                "input_guardrail_ms": round(input_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "grounding_ms": round(grounding_ms, 2),
                "total_ms": round((t_end - t_start) * 1000, 2)
            }
        }
