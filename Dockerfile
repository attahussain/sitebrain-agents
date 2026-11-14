FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SITEBRAIN_DOCS=/app/data/docs \
    SITEBRAIN_DB=/app/data/lancedb \
    SITEBRAIN_CHECKPOINT_DB=/app/data/state/checkpoints.sqlite \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app
RUN useradd --create-home --uid 1000 sitebrain

COPY pyproject.toml ./
COPY sitebrain ./sitebrain
RUN pip install --no-cache-dir .

# Bake the embedding model into the image so the container starts without network access.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY data/docs ./data/docs
RUN mkdir -p /app/data/state && chown -R sitebrain:sitebrain /app
USER sitebrain

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "sitebrain.api:app", "--host", "0.0.0.0", "--port", "8000"]
