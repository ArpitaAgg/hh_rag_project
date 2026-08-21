FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

COPY . .

# Seed the demo sample data and pre-build the FAISS index + embedding model
# cache at image build time, so the first live request doesn't pay for it.
RUN python extract_samples_and_analysis.py && \
    python src/build_faiss_index.py

# Hugging Face Spaces (Docker SDK) requires a world-writable app directory
# since the container runs as a non-root user.
RUN mkdir -p data/indexes data/temp_uploads && chmod -R 777 /app

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
