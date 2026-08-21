"""
streamlit_app.py

Streamlit Web Interface for Voice-Enabled Multilingual RAG Application.
Deployable on Streamlit Community Cloud (https://share.streamlit.io) for 100% FREE 1GB RAM hosting.
Includes Live Microphone Recording (Sarvam AI STT), Audio File Upload, and Text Query Mode.
"""

import sys
import os
import time
import tempfile
import streamlit as st

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag_pipeline import RAGPipeline
from voice_rag_pipeline import VoiceRAGPipeline
from speech_to_text import SarvamSpeechToText

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
    .transcript-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #fbbf24;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-banner">
    <h1 style="color: #34d399; margin-bottom: 4px;">⚡ Hacker House Goa 2026</h1>
    <h3 style="color: #fbbf24; margin-top: 0;">Multilingual Voice RAG Engine (Sarvam STT + FAISS + BM25 + Groq LLM)</h3>
    <p style="color: #94a3b8;">Grounded Zero-Hallucination Q&A System · Voice & Text Multilingual Pipeline</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipelines():
    rag_pipe = RAGPipeline()
    stt_provider = SarvamSpeechToText()
    voice_pipe = VoiceRAGPipeline(stt_provider=stt_provider, rag_pipeline=rag_pipe)
    return rag_pipe, voice_pipe

try:
    with st.spinner("Loading FAISS Vector Index, BM25 Store & Sarvam Voice Engine..."):
        rag_pipeline, voice_pipeline = load_pipelines()
    st.success("Voice RAG Pipeline & Sarvam STT Engine Ready!")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# Tabs for Voice & Text Query
tab_voice, tab_text = st.tabs(["🎙️ Voice Input (Microphone & Audio Upload)", "💬 Text Input Mode"])

# --- TAB 1: VOICE INPUT ---
with tab_voice:
    st.subheader("🎙️ Voice Query Input")
    st.caption("Record your query using your microphone or upload an audio recording (.wav, .mp3, .webm, .m4a).")
    
    col_rec, col_up = st.columns(2)
    
    audio_bytes = None
    file_ext = ".webm"
    
    with col_rec:
        st.markdown("**Option A: Live Microphone Recording**")
        if hasattr(st, "audio_input"):
            rec_audio = st.audio_input("Click microphone to record voice query:")
            if rec_audio:
                audio_bytes = rec_audio.read()
                file_ext = ".wav"
        else:
            st.info("Live microphone widget available in Streamlit 1.38+. Use audio upload on the right.")

    with col_up:
        st.markdown("**Option B: Upload Audio Recording**")
        uploaded_file = st.file_uploader("Choose an audio file:", type=["wav", "mp3", "webm", "m4a", "ogg"])
        if uploaded_file and not audio_bytes:
            audio_bytes = uploaded_file.read()
            file_ext = os.path.splitext(uploaded_file.name)[1] or ".webm"

    if audio_bytes:
        st.audio(audio_bytes, format=f"audio/{file_ext.replace('.', '')}")
        if st.button("🚀 Process Voice Query", type="primary", key="btn_voice"):
            t0 = time.time()
            with st.spinner("Transcribing voice via Sarvam AI & searching RAG Knowledge Base..."):
                # Save temp audio file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                try:
                    result = voice_pipeline.answer_audio(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            elapsed_ms = round((time.time() - t0) * 1000, 2)
            rag_res = result.get("rag_details", {})
            
            st.markdown("---")
            st.markdown(f"### 🎙️ Transcribed Voice Result (Total Latency: `{elapsed_ms} ms`)")
            
            st.markdown(f"""
            <div class="transcript-box">
                <strong style="color: #fbbf24;">Transcribed Voice Query:</strong>
                <p style="font-size: 1.1rem; color: #f8fafc; margin: 4px 0 0 0;">"{result.get('transcript', '')}"</p>
                <small style="color: #94a3b8;">Detected Language: <code>{result.get('language', 'unknown')}</code> | STT Latency: <code>{result.get('latency', {}).get('stt_ms', 0)} ms</code></small>
            </div>
            <div class="result-card">
                <h4 style="color: #34d399; margin-top: 0;">Grounded Answer:</h4>
                <p style="font-size: 1.15rem; color: #f8fafc; line-height: 1.6;">{result.get('answer', '')}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📚 View Full RAG Context Details"):
                st.json(result)

# --- TAB 2: TEXT INPUT ---
with tab_text:
    st.subheader("💬 Text Query Input")
    user_query = st.text_input(
        "Ask a question in English, Hindi, Hinglish, Bengali, Tamil, etc.:",
        placeholder="e.g. what is the boiling point of water / corporation kya h / निगम क्या है?",
        key="input_text_query"
    )

    if st.button("🔍 Submit Text Query", type="primary", key="btn_text") and user_query.strip():
        t0 = time.time()
        with st.spinner("Searching RAG Knowledge Base & Generating Grounded Answer..."):
            result = rag_pipeline.answer(user_query.strip())
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
