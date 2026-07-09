# GemmaJudge - container image (AMD Developer Hackathon ACT II, Track 3).
# Provides a reproducible one-command run for the submitted app.
#
# Builds the Streamlit demo (app.py). The engine talks to Gemma over an
# OpenAI-compatible endpoint - managed service for uptime or
# self-hosted vLLM on AMD + ROCm - configured entirely via environment variables
# (see .env.example). With no
# engine configured, the app still renders the committed real runs and the clearly
# labelled simulated demo, so the container is always demonstrable.
#
#   docker build -t gemmajudge .
#   docker run --rm -p 8501:8501 gemmajudge
#   # optional real backend: add --env-file .env
#   # then open http://localhost:8501
#
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Config/secrets come from the environment at run time - never baked into the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
