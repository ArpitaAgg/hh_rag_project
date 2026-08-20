"""
src/voice_rag_pipeline.py

Voice-Enabled End-to-End RAG Orchestration Layer.
Integrates Speech-to-Text (STT) voice input with the unified RAGPipeline.

--------------------------------------------------------------------------------
ARCHITECTURE FLOW:

AUDIO FILE (.wav / .mp3 / .webm / .m4a / .ogg)
    ↓
BaseSpeechToText (Sarvam AI STT)
    ↓
Transcribed Text Query
    ↓
RAGPipeline.answer(query)
    ↓
Guardrails + Hybrid Retrieval + Answer Generator
    ↓
Structured Voice RAG Output
--------------------------------------------------------------------------------
"""

import time
import sys
import os
from typing import Dict, Any, Optional

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from speech_to_text import BaseSpeechToText, SarvamSpeechToText
from rag_pipeline import RAGPipeline

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class VoiceRAGPipeline:
    """
    End-to-End Voice RAG Pipeline Orchestrator.
    Accepts audio input, transcribes via STT, and returns grounded RAG answers.
    """
    def __init__(
        self,
        stt_provider: Optional[BaseSpeechToText] = None,
        rag_pipeline: Optional[RAGPipeline] = None
    ):
        self.stt = stt_provider or SarvamSpeechToText()
        self.rag_pipeline = rag_pipeline or RAGPipeline()

    def answer_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Processes an audio file and returns a complete grounded RAG answer.

        Args:
            audio_file_path: Path to the recorded audio file.

        Returns:
            Structured result containing transcript, answer, grounded status, and latencies.
        """
        t_start = time.time()

        # 1. Speech-to-Text Transcription
        t0 = time.time()
        stt_result = self.stt.transcribe(audio_file_path)
        t1 = time.time()
        stt_ms = round((t1 - t0) * 1000, 2)

        # Handle STT failure
        if stt_result["status"] != "success" or not stt_result["text"].strip():
            t_end = time.time()
            err_msg = stt_result.get("error") or "Speech-to-Text transcription failed or returned empty text."
            
            return {
                "status": "stt_failed",
                "transcript": "",
                "answer": f"Speech-to-Text Error: {err_msg}",
                "language": stt_result.get("language", "unknown"),
                "grounded": False,
                "guardrail_reason": f"STT Failure: {err_msg}",
                "latency": {
                    "stt_ms": stt_ms,
                    "rag_ms": 0.0,
                    "total_ms": round((t_end - t_start) * 1000, 2)
                },
                "error": err_msg,
                "rag_details": {}
            }

        transcript_text = stt_result["text"].strip()
        detected_lang = stt_result.get("language", "unknown")

        # 2. RAG Pipeline Execution
        t2 = time.time()
        rag_result = self.rag_pipeline.answer(transcript_text)
        t3 = time.time()
        rag_ms = round((t3 - t2) * 1000, 2)
        t_end = time.time()

        return {
            "status": rag_result.get("status", "failed"),
            "transcript": transcript_text,
            "answer": rag_result.get("answer", ""),
            "language": detected_lang,
            "grounded": rag_result.get("grounded", False),
            "guardrail_reason": rag_result.get("guardrail_reason", ""),
            "latency": {
                "stt_ms": stt_ms,
                "rag_ms": rag_ms,
                "total_ms": round((t_end - t_start) * 1000, 2)
            },
            "error": None,
            "rag_details": rag_result
        }
