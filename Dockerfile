FROM python:3.11-slim

# Install ffmpeg for Sarvam voice STT audio processing and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Expose port (default 7860 for HF Spaces, $PORT for Render)
EXPOSE 7860 10000

# Start Uvicorn on $PORT (or 7860 fallback)
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
