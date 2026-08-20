"""
src/test_api.py

Automated Test Suite for FastAPI Web Backend (/health, /api/text, /api/voice).
Uses FastAPI TestClient to evaluate endpoints, error handling, request ID tracing,
and static frontend serving. Writes report to data/api_test_results.txt.
"""

import os
import sys
import json
import time
from unittest.mock import patch

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from api import app

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_api_tests():
    print("==================================================")
    print("      FASTAPI WEB BACKEND EVALUATION & TEST      ")
    print("==================================================")

    client = TestClient(app)
    test_results = []
    report_lines = [
        "==================================================",
        "      FASTAPI WEB BACKEND TEST RESULTS            ",
        "==================================================\n"
    ]

    # --- Test 1: GET /health ---
    print("Test 1: Health Check Endpoint (GET /health)...")
    res1 = client.get("/health")
    t1_pass = (res1.status_code == 200 and res1.json().get("status") == "ok")
    test_results.append(("GET /health", t1_pass, f"HTTP {res1.status_code} | Body: {res1.json()}"))
    print(f"  Result: {'PASS' if t1_pass else 'FAIL'} | Status: {res1.status_code} | Output: {res1.json()}\n")

    # --- Test 2: POST /api/text (Valid Query) ---
    print("Test 2: Text Q&A Endpoint (POST /api/text - Valid Query)...")
    res2 = client.post("/api/text", json={"query": "কৰ্পোৰেচন কি?"})
    body2 = res2.json()
    t2_pass = (res2.status_code == 200 and body2.get("status") == "answered")
    test_results.append(("POST /api/text (Valid)", t2_pass, f"Status: {body2.get('status')} | Answer: '{body2.get('answer')[:60]}...'"))
    print(f"  Result: {'PASS' if t2_pass else 'FAIL'} | Status: {body2.get('status')}")
    print(f"  Request ID Header: {res2.headers.get('X-Request-ID')}\n")

    # --- Test 3: POST /api/text (Empty Query) ---
    print("Test 3: Text Q&A Endpoint (POST /api/text - Empty Query)...")
    res3 = client.post("/api/text", json={"query": ""})
    body3 = res3.json()
    t3_pass = (res3.status_code == 200 and body3.get("status") == "rejected")
    test_results.append(("POST /api/text (Empty)", t3_pass, f"Status: {body3.get('status')}"))
    print(f"  Result: {'PASS' if t3_pass else 'FAIL'} | Status: {body3.get('status')}\n")

    # --- Test 4: POST /api/voice (Invalid Audio Upload) ---
    print("Test 4: Voice Q&A Endpoint (POST /api/voice - Invalid Audio File)...")
    files4 = {"file": ("test.txt", b"invalid text data", "text/plain")}
    res4 = client.post("/api/voice", files=files4)
    body4 = res4.json()
    t4_pass = (res4.status_code in [200, 500] and body4.get("status") == "stt_failed")
    test_results.append(("POST /api/voice (Invalid File)", t4_pass, f"Status: {body4.get('status')} | Error: {body4.get('error')}"))
    print(f"  Result: {'PASS' if t4_pass else 'FAIL'} | Status: {body4.get('status')} | Error: {body4.get('error')}\n")

    # --- Test 5: POST /api/voice (Mocked STT Flow) ---
    print("Test 5: Voice Q&A Endpoint (POST /api/voice - Mocked STT Flow)...")
    mock_stt_result = {
        "text": "কৰ্পোৰেচন কি?",
        "provider": "sarvam-stt-mocked",
        "language": "as-IN",
        "latency_ms": 120.0,
        "status": "success",
        "error": None
    }
    with patch("speech_to_text.SarvamSpeechToText.transcribe", return_value=mock_stt_result):
        files5 = {"file": ("recording.webm", b"dummy audio content", "audio/webm")}
        res5 = client.post("/api/voice", files=files5)
        body5 = res5.json()
        t5_pass = (res5.status_code == 200 and body5.get("status") == "answered" and body5.get("transcript") == "কৰ্পোৰেচন কি?")
        test_results.append(("POST /api/voice (Mocked STT)", t5_pass, f"Status: {body5.get('status')} | Transcript: '{body5.get('transcript')}'"))
        print(f"  Result: {'PASS' if t5_pass else 'FAIL'} | Status: {body5.get('status')} | Transcript: '{body5.get('transcript')}'\n")

    # --- Test 6: Static Frontend Serving (GET /) ---
    print("Test 6: Static Frontend Serving (GET /)...")
    res6 = client.get("/")
    t6_pass = (res6.status_code == 200 and "<title>Voice-Enabled Multilingual RAG System</title>" in res6.text)
    test_results.append(("GET / (Frontend Page)", t6_pass, f"HTTP {res6.status_code}"))
    print(f"  Result: {'PASS' if t6_pass else 'FAIL'} | Status: {res6.status_code}\n")

    # Compile Summary & Write Report
    passed_count = sum(1 for name, ok, desc in test_results if ok)
    report_lines.append(f"TOTAL TESTS RUN : {len(test_results)}")
    report_lines.append(f"PASSED          : {passed_count} / {len(test_results)}\n")

    for idx, (name, ok, desc) in enumerate(test_results, start=1):
        report_lines.append(f"Case #{idx}: {name}")
        report_lines.append(f"  Status : {'PASS' if ok else 'FAIL'}")
        report_lines.append(f"  Details: {desc}\n")

    os.makedirs("data", exist_ok=True)
    report_file = os.path.join("data", "api_test_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"API test report saved to '{report_file}'.")


if __name__ == "__main__":
    run_api_tests()
