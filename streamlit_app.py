"""
streamlit_app.py

Streamlit Web Interface for Voice-Enabled Multilingual RAG Application.
Deployable on Streamlit Community Cloud (https://share.streamlit.io) for 100% FREE 1GB RAM hosting.
"""

import sys
import os
import time
import streamlit as st

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Hacker House Goa 2026 — Multilingual Voice RAG",
    page_icon="⚡",
    layout="wide"
)

# Custom Hacker House Goa Dark Emerald Glassmorphism Theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #071510 0%, #0d281e 50%, #040e0b 100%);
        color: #e2e8f0;
    }
    .header-banner {
        background: rgba(13, 40, 30, 0.6);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(12px);
        margin-bottom: 24px;
    }
    .result-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-banner">
    <h1 style="color: #34d399; margin-bottom: 4px;">⚡ Hacker House Goa 2026</h1>
    <h3 style="color: #fbbf24; margin-top: 0;">Multilingual RAG Engine (Sarvam STT + FAISS + BM25 + Groq LLM)</h3>
    <p style="color: #94a3b8;">Grounded Zero-Hallucination Q&A System · Sub-200ms Target Latency</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

try:
    with st.spinner("Loading FAISS Index & Embeddings..."):
        pipeline = load_pipeline()
    st.success("RAG Pipeline Engine Ready!")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# Query Input
user_query = st.text_input(
    "Ask a question in English, Hindi, Hinglish, Bengali, Tamil, etc.:",
    placeholder="e.g. what is the boiling point of water / corporation kya h / निगम क्या है?"
)

col1, col2 = st.columns([1, 4])
with col1:
    submit_btn = st.button("🔍 Submit Query", type="primary", use_container_width=True)

if submit_btn and user_query.strip():
    t0 = time.time()
    with st.spinner("Searching RAG Knowledge Base & Generating Grounded Answer..."):
        result = pipeline.answer(user_query.strip())
    elapsed_ms = round((time.time() - t0) * 1000, 2)

    st.markdown("---")
    st.markdown(f"### 💬 Answer (Response Latency: `{elapsed_ms} ms`)")
    
    st.markdown(f"""
    <div class="result-card">
        <h4 style="color: #34d399; margin-top: 0;">Grounded Answer:</h4>
        <p style="font-size: 1.15rem; color: #f8fafc; line-height: 1.6;">{result.get('answer', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📚 View Retrieved Source Chunks & Metadata"):
        st.json(result)
