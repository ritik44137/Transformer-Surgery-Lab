# Transformer Surgery Lab — development image (GPU-capable host)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

# OS packages for building Python extensions and basic tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer cached until requirements change)
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

# Install PyTorch with CUDA 12.1 wheels (adjust index if your host differs)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY . .

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
