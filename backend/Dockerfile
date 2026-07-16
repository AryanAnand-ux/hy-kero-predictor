FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build-essential needed for some scikit-learn builds if no wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first to leverage Docker cache
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend codebase, training src directory, and data directories
COPY backend/ ./backend
COPY src/ ./src
COPY data/ ./data
COPY *.xlsx ./

# Expose FastAPI port
EXPOSE 8000

ENV PYTHONPATH=/app/src
ENV PORT=8000

# Create non-root user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Run FastAPI using uvicorn
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
