FROM node:20-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts

COPY frontend/src/ ./src/
COPY frontend/index.html frontend/vite.config.ts frontend/tsconfig.json frontend/tsconfig.app.json frontend/tsconfig.node.json ./
RUN npm run build


FROM python:3.11-slim AS app

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install dependencies first so code changes don't invalidate this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-build

COPY app/ ./app/
COPY data/sample_kb/ ./data/sample_kb/
COPY --from=frontend-build /frontend/dist /app/frontend/dist
ENV PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# Bake the embedding model and keyword corpus into the image so a cold start
# does not download or rebuild them. Keep in sync with EMBEDDING_MODEL.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}')" \
    && python -m app.build_bm25

# Only the writable paths need to change owner; chowning the whole image is
# slow on large layers and is not required for read-only app files.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/uploads \
    && chown -R appuser:appuser /app/data /app/uploads /app/.cache/fastembed
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/api/ready',timeout=3)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
