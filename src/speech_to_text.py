"""
src/speech_to_text.py

Provider-Independent Speech-to-Text (STT) Layer with Sarvam AI Integration.

--------------------------------------------------------------------------------
ARCHITECTURE:

  BaseSpeechToText (Abstract Interface)
            │
            ▼
   SarvamSpeechToText (Sarvam AI STT API)

The RAG pipeline depends ONLY on BaseSpeechToText, keeping Sarvam completely
decoupled and pluggable.

SUPPORTED AUDIO FORMATS:
  .wav, .mp3, .m4a, .webm, .ogg, .flac

SECURITY:
  API key is loaded strictly from environment variable `SARVAM_API_KEY`
  (or via python-dotenv). Never hardcoded.
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Load environment variables if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class BaseSpeechToText(ABC):
    """
    Abstract Base Class for Speech-to-Text Providers.
    """
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    @abstractmethod
    def transcribe(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribes an audio file into text.

        Returns structured result:
            {
                "text": str,
                "provider": str,
                "language": str,
                "latency_ms": float,
                "status": "success" | "failed",
                "error": Optional[str]
            }
        """
        pass

    def validate_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        Validates audio file existence, non-emptiness, format, and size limit.
        Returns error string if invalid, or None if valid.
        """
        if not audio_file_path or not isinstance(audio_file_path, str):
            return "Audio file path must be a non-empty string."

        if not os.path.exists(audio_file_path):
            return f"Audio file not found at path: '{audio_file_path}'"

        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            return f"Audio file at '{audio_file_path}' is empty (0 bytes)."

        if file_size > MAX_FILE_SIZE_BYTES:
            return f"Audio file size ({file_size / (1024*1024):.2f} MB) exceeds maximum limit of 25 MB."

        ext = os.path.splitext(audio_file_path)[1].lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            return f"Unsupported audio format '{ext}'. Supported formats: {sorted(list(SUPPORTED_AUDIO_EXTENSIONS))}"

        return None


class SarvamSpeechToText(BaseSpeechToText):
    """
    Sarvam AI Speech-to-Text API Implementation.
    """
    SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: Optional[str] = None, model: str = "saaras:v3", language_code: str = "unknown"):
        super().__init__(provider_name="sarvam-stt")
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "").strip()
        self.model = model
        self.language_code = language_code
        

    def transcribe(self, audio_file_path: str) -> Dict[str, Any]:
        t0 = time.time()

        # 1. Audio File Validation (runs first)
        val_error = self.validate_audio_file(audio_file_path)
        if val_error:
            t1 = time.time()
            return {
                "text": "",
                "provider": self.provider_name,
                "language": "unknown",
                "latency_ms": round((t1 - t0) * 1000, 2),
                "status": "failed",
                "error": val_error
            }

        # 2. API Key Check
        if not self.api_key:
            t1 = time.time()
            return {
                "text": "",
                "provider": self.provider_name,
                "language": "unknown",
                "latency_ms": round((t1 - t0) * 1000, 2),
                "status": "failed",
                "error": "SARVAM_API_KEY environment variable is not configured."
            }

        # 3. Call Sarvam AI STT HTTP API
        try:
            headers = {
                "api-subscription-key": self.api_key
            }
            data = {
                "model": self.model,
                "language_code": self.language_code
            }

            with open(audio_file_path, "rb") as f:
                files = {"file": (os.path.basename(audio_file_path), f, "audio/wav")}
                response = requests.post(self.SARVAM_STT_URL, headers=headers, data=data, files=files, timeout=30)

            t1 = time.time()
            latency = round((t1 - t0) * 1000, 2)

            if response.status_code != 200:
                return {
                    "text": "",
                    "provider": self.provider_name,
                    "language": "unknown",
                    "latency_ms": latency,
                    "status": "failed",
                    "error": f"Sarvam API error (HTTP {response.status_code}): {response.text[:200]}"
                }

            result_json = response.json()
            transcript = result_json.get("transcript", "").strip()
            detected_lang = result_json.get("language_code", self.language_code)

            if not transcript:
                return {
                    "text": "",
                    "provider": self.provider_name,
                    "language": detected_lang,
                    "latency_ms": latency,
                    "status": "failed",
                    "error": "Sarvam API returned an empty transcription."
                }

            return {
                "text": transcript,
                "provider": self.provider_name,
                "language": detected_lang,
                "latency_ms": latency,
                "status": "success",
                "error": None
            }

        except Exception as e:
            t1 = time.time()
            return {
                "text": "",
                "provider": self.provider_name,
                "language": "unknown",
                "latency_ms": round((t1 - t0) * 1000, 2),
                "status": "failed",
                "error": f"Network/HTTP Exception during Sarvam STT call: {str(e)}"
            }
