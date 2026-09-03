# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim AS builder

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ------------------------------------------------------------
# Install uv
# ------------------------------------------------------------
RUN pip install --no-cache-dir uv

# ------------------------------------------------------------
# Copy project metadata
# ------------------------------------------------------------
COPY pyproject.toml uv.lock README.md ./

# ------------------------------------------------------------
# Install production dependencies only (project not installed yet)
# This layer is cached even when src/ changes.
# ------------------------------------------------------------
RUN uv sync --frozen --no-install-project --no-dev

# ------------------------------------------------------------
# Copy application source
# ------------------------------------------------------------
COPY src/ ./src/

# ------------------------------------------------------------
# Now install the project itself (src/ exists -> uv_build works)
# ------------------------------------------------------------
RUN uv sync --frozen --no-dev

# ------------------------------------------------------------
# Download NLTK data
# ------------------------------------------------------------
RUN /app/.venv/bin/python -m nltk.downloader \
    -d /usr/share/nltk_data \
    stopwords \
    punkt \
    punkt_tab


# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# ------------------------------------------------------------
# Create non-root user
# ------------------------------------------------------------
RUN useradd \
    --create-home \
    --uid 1000 \
    --shell /usr/sbin/nologin \
    appuser

# ------------------------------------------------------------
# Copy virtual environment
# ------------------------------------------------------------
COPY --from=builder /app/.venv /app/.venv

# ------------------------------------------------------------
# Copy NLTK data
# ------------------------------------------------------------
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data

# ------------------------------------------------------------
# Copy application source
# ------------------------------------------------------------
COPY --from=builder /app/src /app/src

# ------------------------------------------------------------
# Copy trained models
# ------------------------------------------------------------
COPY models/ /app/models/

# ------------------------------------------------------------
# Set permissions
# ------------------------------------------------------------
RUN chown -R appuser:appuser /app /usr/share/nltk_data

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA="/usr/share/nltk_data"

# ------------------------------------------------------------
# Run as non-root
# ------------------------------------------------------------
USER appuser

# ------------------------------------------------------------
# Application port
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ------------------------------------------------------------
# Start FastAPI
# ------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "src.nlp_disaster_tweets.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
