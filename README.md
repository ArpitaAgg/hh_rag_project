# Voice-Enabled Multilingual RAG Project

A modular, production-ready voice-enabled Retrieval-Augmented Generation (RAG) system built step by step over the **ai4bharat/MSMARCO-XI** dataset.

It features **Sarvam AI Speech-to-Text**, **Multilingual Sentence Transformers**, **FAISS Vector Search**, **BM25 Keyword Search**, **Hybrid Retrieval Fusion**, **Provider-Independent Answer Generation**, and **Multi-tier Guardrails**.

---

## 🏗️ Architecture Overview

```
Browser Voice Recording / Text Query
                  │
                  ▼
         FastAPI Web Backend (/api/voice & /api/text)
                  │
                  ▼
         InputGuardrail (Empty / Off-topic / Unsafe checks)
                  │ (If Allowed)
                  ▼
    Sarvam AI Speech-to-Text (Voice audio to text transcript)
                  │
                  ▼
      Hybrid Retrieval (FAISS Semantic + BM25 Keyword)
                  │
                  ▼
   ContextValidator (Relevance score quality check)
                  │ (If Sufficient)
                  ▼
   Answer Generator (Provider-Independent: Local / Cloud API)
                  │
                  ▼
    GroundingGuardrail (Hallucination check)
                  │
                  ▼
         Structured Final Response Displayed in Web UI
```

---

## 🚀 How to Run Locally

### 1. Set Up Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install base dependencies
pip install -r requirements.txt

# Install web application dependencies
pip install -r requirements-web.txt
```

---

### 2. Configure Environment Variables (`.env`)

Copy `.env.example` to create your local `.env` file:

```bash
cp .env.example .env
```

Edit `.env` to add your **Sarvam AI API Key**:

```env
SARVAM_API_KEY=your_actual_sarvam_api_key_here
GENERATOR_PROVIDER=local
RETRIEVAL_TOP_K=3
FAISS_WEIGHT=0.7
BM25_WEIGHT=0.3
MIN_RELEVANCE_SCORE=0.30
```

> ⚠️ **Security Note**: Never commit `.env` to Git. It is already added to `.gitignore`.

---

### 3. Start the FastAPI Web Application

Run the server using Uvicorn:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

or via python module:

```bash
python -m uvicorn src.api:app --reload --port 8000
```

---

### 4. Open in Browser

Open your browser and navigate to:

👉 **`http://localhost:8000`**

You will see the web interface:
* **Microphone Button**: Record spoken voice queries directly from your browser mic.
* **Text Fallback Input**: Type questions in English or Indic languages (Assamese, Hindi, Bengali).
* **Knowledge Sources**: View retrieved passage text, similarity scores, and chunk IDs.
* **Performance Metrics**: View live STT, Retrieval, and Total latency metrics.

---

## 🧪 Running Automated Test Suites

You can run individual component test scripts at any time:

```bash
# Test Chunking Strategies
python src/test_chunking.py

# Test Multilingual Embeddings
python src/test_embeddings.py

# Test FAISS Semantic Retrieval
python src/test_faiss.py

# Test Hybrid Retrieval (FAISS + BM25)
python src/test_hybrid_retrieval.py

# Test Answer Generation
python src/test_generator.py

# Test Guardrail Safety & Hallucination Checks
python src/test_guardrails.py

# Test Main Unified RAG Pipeline
python src/test_rag_pipeline.py

# Test Speech-to-Text Layer
python src/test_speech_to_text.py

# Test FastAPI Web Endpoints & Frontend
python src/test_api.py
```

All test reports are automatically saved under `data/`.

---

## 📁 Complete Project Structure

```
goa2/
├── .env.example                   # Environment variable configuration template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Core project dependencies
├── requirements-web.txt           # FastAPI, Uvicorn & Web dependencies
├── README.md                      # Project documentation and setup guide
├── data/
│   ├── sample_records.json        # 10 representative dataset sample records
│   ├── dataset_analysis.txt       # Step 2 dataset analysis report
│   ├── chunking_results.txt       # Step 3 chunking comparison
│   ├── embedding_test_results.txt # Step 4 embedding validation report
│   ├── faiss_test_results.txt     # Step 5 FAISS report
│   ├── hybrid_retrieval_results.txt# Step 6 Hybrid retrieval report
│   ├── generation_test_results.txt# Step 7 Generation report
│   ├── guardrail_test_results.txt # Step 8 Guardrail evaluation report
│   ├── rag_pipeline_test_results.txt # Step 9 RAG Pipeline report
│   ├── speech_to_text_test_results.txt # Step 10 STT report
│   ├── api_test_results.txt       # Step 11 Web API test report
│   └── indexes/
│       ├── index.faiss            # FAISS vector index
│       └── metadata.json          # Metadata store
├── frontend/
│   ├── index.html                 # Main web application HTML
│   ├── style.css                  # Modern glassmorphism dark theme CSS
│   └── app.js                     # Browser MediaRecorder audio & REST client JS
└── src/
    ├── chunking.py                # Step 3 chunking module
    ├── test_chunking.py           # Step 3 test runner
    ├── embeddings.py              # Step 4 embeddings module
    ├── test_embeddings.py         # Step 4 test runner
    ├── vector_store.py            # Step 5 FAISS vector store
    ├── build_faiss_index.py       # Step 5 index builder
    ├── test_faiss.py              # Step 5 FAISS test runner
    ├── bm25_store.py              # Step 6 BM25 store
    ├── hybrid_retriever.py        # Step 6 Hybrid retriever
    ├── test_hybrid_retrieval.py   # Step 6 Hybrid test runner
    ├── system_info.py             # Hardware discovery script
    ├── generator.py               # Step 7 Provider-Independent Generator
    ├── test_generator.py          # Step 7 Generator test runner
    ├── guardrails.py              # Step 8 Guardrails Layer
    ├── test_guardrails.py         # Step 8 Guardrails test runner
    ├── rag_pipeline.py            # Step 9 Unified RAG Orchestrator
    ├── test_rag_pipeline.py       # Step 9 Main pipeline test runner
    ├── speech_to_text.py          # Step 10 Provider-Independent STT & Sarvam AI Integration
    ├── voice_rag_pipeline.py      # Step 10 Voice RAG Orchestration Pipeline
    ├── test_speech_to_text.py     # Step 10 STT test suite & runner
    ├── api.py                     # Step 11 FastAPI Backend API & Static Server
    └── test_api.py                # Step 11 FastAPI Web Backend test runner
```
