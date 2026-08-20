"""
src/test_live_multilingual.py

Live Multilingual RAG Test Suite.
Tests native-script queries across multiple Indic languages (Hindi, Bengali, Assamese,
Gujarati, Marathi, Tamil) against the live FastAPI server at http://127.0.0.1:8000/api/text.
"""

import os
import sys
import time
import requests

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Native-script test queries for "What is a corporation?" across Indic languages
MULTILINGUAL_QUERIES = [
    {"lang": "Hindi", "code": "hi", "query": "निगम क्या है?"},
    {"lang": "Bengali", "code": "bn", "query": "কর্পোরেশন কি?"},
    {"lang": "Assamese", "code": "as", "query": "নিগম কি?"},
    {"lang": "Gujarati", "code": "gu", "query": "કોર્પોરેશન શું છે?"},
    {"lang": "Marathi", "code": "mr", "query": "महामंडळ म्हणजे काय?"},
    {"lang": "Tamil", "code": "ta", "query": "கார்பரேஷன் என்றால் என்ன?"},
]


def test_live_multilingual():
    print("==================================================")
    print("      LIVE MULTILINGUAL RAG API VERIFICATION      ")
    print("==================================================")
    
    server_url = "http://127.0.0.1:8000/api/text"
    all_passed = True

    for item in MULTILINGUAL_QUERIES:
        lang = item["lang"]
        q = item["query"]
        print(f"\nTesting Language: {lang} ({item['code']})")
        print(f"  Native Query: \"{q}\"")

        t0 = time.time()
        try:
            resp = requests.post(server_url, json={"query": q}, timeout=30.0)
            t_elapsed = (time.time() - t0) * 1000

            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                answer = data.get("answer", "")
                reason = data.get("guardrail_reason", "")
                grounded = data.get("grounded", False)

                print(f"  HTTP Status : 200 OK")
                print(f"  RAG Status  : {status}")
                print(f"  Grounded    : {grounded}")
                print(f"  Answer      : \"{answer[:120]}...\"")
                print(f"  Latency     : {t_elapsed:.2f} ms")

                if status == "answered":
                    print(f"  Result      : ✅ PASSED (Multilingual Answer Grounded)")
                else:
                    print(f"  Result      : ⚠️ REFUSED / INSUFFICIENT CONTEXT")
                    all_passed = False
            else:
                print(f"  HTTP Error  : {resp.status_code} - {resp.text}")
                all_passed = False
        except Exception as e:
            print(f"  Exception   : {e}")
            all_passed = False

    print("\n==================================================")
    print(f"  MULTILINGUAL AUDIT RESULT: {'ALL INDIC LANGUAGES WORKING ✅' if all_passed else 'SOME LANGUAGES REFUSED ⚠️'}")
    print("==================================================")


if __name__ == "__main__":
    test_live_multilingual()
