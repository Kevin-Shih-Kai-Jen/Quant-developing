# =============================================================================
# Nexus Quant OS — Production Dockerfile
# =============================================================================
# Base  : python:3.11-slim (Debian Bookworm, ARM64-compatible)
# Target: Colima on Mac M1 (linux/arm64 VM)
# =============================================================================

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────
# gcc & friends needed for hmmlearn / scikit-learn C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ── Python env ────────────────────────────────────────────────────────
WORKDIR /app

# Step 1: Install PyTorch CPU-only from dedicated index
#         This avoids pulling 2 GB CUDA binaries
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining deps from PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ─────────────────────────────────────────────────
# (In docker-compose we mount .:/app, so this is for standalone builds)
COPY . .

# ── Runtime defaults ─────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default entrypoint: run the full DAG pipeline
CMD ["python", "main.py"]
