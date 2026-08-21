"""
app.py

Entry point for Hugging Face Spaces (Gradio / Python SDK).
Launches the FastAPI application on port 7860.
"""

import uvicorn
from src.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
