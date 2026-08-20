"""
src/test_groq_zero_hallucination.py

Targeted test suite verifying that Groq LLM NEVER uses its pre-trained memory to answer questions
when the answer is missing from the retrieved dataset context.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from generator import GroqAnswerGenerator
from guardrails import GroundingGuardrail

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def test_groq_zero_hallucination():
    print("==================================================")
    print("   GROQ ZERO-HALLUCINATION & MEMORY ISOLATION TEST")
    print("==================================================")

    groq_gen = GroqAnswerGenerator()
    grounding = GroundingGuardrail()

    test_cases = [
        {
            "name": "General Knowledge Trap 1 (Capital of France)",
            "query": "What is the capital of France?",
            "context": [
                {
                    "chunk_id": "c1",
                    "text": "Global Warming is often euphemistically referred to as Climate Change. Study of climate vs weather."
                }
            ],
            "forbidden_words": ["paris", "france"],
            "must_contain_refusal": True
        },
        {
            "name": "General Knowledge Trap 2 (Penicillin Discovery)",
            "query": "Who discovered penicillin?",
            "context": [
                {
                    "chunk_id": "c2",
                    "text": "1: a government-owned corporation (as a utility or railroad) engaged in a profit-making enterprise."
                }
            ],
            "forbidden_words": ["fleming", "alexander"],
            "must_contain_refusal": True
        },
        {
            "name": "General Knowledge Trap 3 (Speed of Light)",
            "query": "What is the speed of light in vacuum?",
            "context": [
                {
                    "chunk_id": "c3",
                    "text": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) under standard atmospheric pressure."
                }
            ],
            "forbidden_words": ["299", "300,000", "meter", "second"],
            "must_contain_refusal": True
        },
        {
            "name": "Grounded Dataset Query (Water Boiling Point)",
            "query": "What temperature does water boil at?",
            "context": [
                {
                    "chunk_id": "c4",
                    "text": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) under standard atmospheric pressure."
                }
            ],
            "forbidden_words": [],
            "must_contain_refusal": False
        }
    ]

    refusal_phrase = "not have enough information"
    all_passed = True

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}] {tc['name']}")
        print(f"    Query   : \"{tc['query']}\"")
        print(f"    Context : \"{tc['context'][0]['text'][:80]}...\"")

        t0 = time.time()
        res = groq_gen.generate(tc['query'], tc['context'])
        t_elapsed = (time.time() - t0) * 1000

        ans = res.get("answer", "")
        provider = res.get("model_provider", "")

        # Check refusal and forbidden memory words
        ans_lower = ans.lower()
        has_refusal = refusal_phrase in ans_lower
        used_memory = any(fw in ans_lower for fw in tc["forbidden_words"])

        if tc["must_contain_refusal"]:
            passed = has_refusal and not used_memory
        else:
            passed = not has_refusal and ("100" in ans or "celsius" in ans_lower or "212" in ans)

        print(f"    Provider : {provider}")
        print(f"    Answer   : \"{ans}\"")
        print(f"    Refusal  : {has_refusal} | Used Memory: {used_memory}")
        print(f"    Latency  : {t_elapsed:.2f} ms")
        print(f"    Result   : {'✅ PASSED (Strict Grounding Verified)' if passed else '❌ FAILED (Memory Leak)'}")

        if not passed:
            all_passed = False

    print("\n==================================================")
    print(f"  GROQ AUDIT RESULT: {'GROQ STRICTLY ISOLATED TO DATASET CONTEXT ✅' if all_passed else 'GROQ LEAKED PRE-TRAINED MEMORY ❌'}")
    print("==================================================")


if __name__ == "__main__":
    test_groq_zero_hallucination()
