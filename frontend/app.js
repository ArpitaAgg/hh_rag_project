/**
 * frontend/app.js
 * Browser Client JavaScript for Voice-Enabled Multilingual RAG Application.
 * Handles browser microphone recording via MediaRecorder API, text input fallback,
 * API requests to FastAPI backend (/api/voice, /api/text), and dynamic UI rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const timerDisplay = document.getElementById('timer');
    const textForm = document.getElementById('textForm');
    const queryInput = document.getElementById('queryInput');
    const submitTextBtn = document.getElementById('submitTextBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingMsg = document.getElementById('loadingMsg');
    const resultsSection = document.getElementById('resultsSection');

    // Result Display Elements
    const statusBadge = document.getElementById('statusBadge');
    const groundedBadge = document.getElementById('groundedBadge');
    const providerBadge = document.getElementById('providerBadge');
    const latStt = document.getElementById('latStt');
    const latRet = document.getElementById('latRet');
    const latTotal = document.getElementById('latTotal');
    const transcriptBox = document.getElementById('transcriptBox');
    const transcriptText = document.getElementById('transcriptText');
    const answerText = document.getElementById('answerText');
    const sourceCount = document.getElementById('sourceCount');
    const sourcesContainer = document.getElementById('sourcesContainer');

    // MediaRecorder Variables
    let mediaRecorder = null;
    let audioChunks = [];
    let startTime = 0;
    let timerInterval = null;

    // --- 1. Microphone Recording Logic ---
    recordBtn.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            
            // Prefer webm or wav format
            let mimeType = 'audio/webm';
            if (!MediaRecorder.isTypeSupported('audio/webm')) {
                mimeType = MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '';
            }

            mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                // Stop audio tracks
                stream.getTracks().forEach(track => track.stop());

                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                await sendVoiceQuery(audioBlob);
            };

            mediaRecorder.start();
            
            // UI Updates
            recordBtn.disabled = true;
            stopBtn.disabled = false;
            recordingStatus.classList.remove('hidden');

            // Timer
            startTime = Date.now();
            timerInterval = setInterval(() => {
                const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
                const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
                const secs = String(elapsedSec % 60).padStart(2, '0');
                timerDisplay.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            alert(`Microphone access error: ${err.message}. Please check browser permissions.`);
        }
    });

    stopBtn.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            clearInterval(timerInterval);
            recordBtn.disabled = false;
            stopBtn.disabled = true;
            recordingStatus.classList.add('hidden');
        }
    });

    // --- 2. Voice API Request (/api/voice) ---
    async function sendVoiceQuery(audioBlob) {
        showLoading("Transcribing voice & searching RAG knowledge base...");
        hideResults();

        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.webm');

        try {
            const response = await fetch('/api/voice', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            hideLoading();
            renderVoiceResult(data);

        } catch (err) {
            hideLoading();
            showError("Network error: Could not reach the voice API server.");
        }
    }

    // --- 3. Text API Form Submit (/api/text) ---
    textForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        showLoading("Processing text query through RAG pipeline...");
        hideResults();

        try {
            const response = await fetch('/api/text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();
            hideLoading();
            if (!response.ok) {
                showError(data.detail || data.guardrail_reason || `Server returned status ${response.status}`);
            } else {
                renderTextResult(data);
            }

        } catch (err) {
            hideLoading();
            showError(`Network error: ${err.message || "Could not reach API server. Please check if Render server is still building or waking up."}`);
        }
    });

    // Sample Query Pills Click Handler
    document.querySelectorAll('.sample-pill').forEach(pill => {
        pill.addEventListener('click', () => {
            const sampleQ = pill.getAttribute('data-query');
            if (sampleQ) {
                queryInput.value = sampleQ;
                textForm.dispatchEvent(new Event('submit'));
            }
        });
    });

    // --- 4. Render Voice Results ---
    function renderVoiceResult(data) {
        resultsSection.classList.remove('hidden');

        // Status & Grounded Badges
        updateStatusBadges(data.status, data.grounded, data.language);

        // Latencies
        const lat = data.latency || {};
        latStt.textContent = `${lat.stt_ms || 0}ms`;
        latRet.textContent = `${lat.rag_ms || 0}ms`;
        latTotal.textContent = `${lat.total_ms || 0}ms`;

        // Transcript
        if (data.transcript) {
            transcriptBox.classList.remove('hidden');
            transcriptText.textContent = data.transcript;
        } else {
            transcriptBox.classList.add('hidden');
        }

        // Answer
        answerText.textContent = data.answer || "No answer generated.";

        // Retrieved Sources
        const rag = data.rag_details || {};
        renderSources(rag.retrieved_context || []);
    }

    // --- 5. Render Text Results ---
    function renderTextResult(data) {
        resultsSection.classList.remove('hidden');
        transcriptBox.classList.add('hidden');

        // Status & Grounded Badges
        updateStatusBadges(data.status, data.grounded, data.metadata?.generator_provider);

        // Latencies
        const lat = data.latency || {};
        latStt.textContent = `0ms (Text Query)`;
        latRet.textContent = `${lat.retrieval_ms || 0}ms`;
        latTotal.textContent = `${lat.total_ms || 0}ms`;

        // Answer
        answerText.textContent = data.answer || "No answer generated.";

        // Retrieved Sources
        renderSources(data.retrieved_context || []);
    }

    // --- Helper Functions ---
    function updateStatusBadges(status, grounded, provider) {
        statusBadge.textContent = (status || 'UNKNOWN').toUpperCase();
        statusBadge.className = `badge badge-status ${status}`;

        groundedBadge.textContent = grounded ? 'GROUNDED' : 'UNGROUNDED / FALLBACK';
        groundedBadge.className = `badge badge-grounded ${grounded}`;

        providerBadge.textContent = `Info: ${provider || 'Local'}`;
    }

    function renderSources(sources) {
        sourcesContainer.innerHTML = '';
        sourceCount.textContent = sources.length;

        if (!sources || sources.length === 0) {
            sourcesContainer.innerHTML = '<p class="subtitle">No knowledge sources retrieved for this query.</p>';
            return;
        }

        sources.forEach((src, idx) => {
            const card = document.createElement('div');
            card.className = 'source-card';
            card.innerHTML = `
                <div class="source-card-header">
                    <span>Rank #${idx + 1} • Chunk ID: ${src.chunk_id || 'N/A'}</span>
                    <span>Similarity Score: ${src.score ? src.score.toFixed(4) : 'N/A'}</span>
                </div>
                <p class="source-card-text">${src.text}</p>
            `;
            sourcesContainer.appendChild(card);
        });
    }

    function showLoading(msg) {
        loadingMsg.textContent = msg;
        loadingOverlay.classList.remove('hidden');
    }

    function hideLoading() {
        loadingOverlay.classList.add('hidden');
    }

    function hideResults() {
        resultsSection.classList.add('hidden');
    }

    function showError(errMsg) {
        resultsSection.classList.remove('hidden');
        statusBadge.textContent = "ERROR";
        statusBadge.className = "badge badge-status rejected";
        answerText.textContent = errMsg;
        sourcesContainer.innerHTML = '';
        sourceCount.textContent = '0';
    }
});
