# Dockerfile for NetSentry-FL: Production Federated Intrusion Detection System
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and directories
COPY src/ /app/src/
COPY app/ /app/app/
COPY data/ /app/data/
COPY experiments/ /app/experiments/
COPY reports/ /app/reports/
COPY tests/ /app/tests/

# Default port for Streamlit interactive dashboard
EXPOSE 8501 8080

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
