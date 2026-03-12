ARG CUDA_VERSION=12.6.3
ARG CUDNN_VERSION=9
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

LABEL maintainer="marksverdhei"
LABEL description="Bakery — Where LLMs go to get baked"
LABEL org.opencontainers.image.source="https://github.com/marksverdhei/bakery"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

# Python 3.12 ships with Ubuntu 24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install PyTorch with CUDA support first (large layer, cached separately)
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install bakery and all dependencies
WORKDIR /opt/bakery
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/
RUN uv pip install ".[qlora]"

# Verify installation
RUN python3 -c "import bakery; import torch; print(f'bakery OK, torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

# Default working directory for training runs
WORKDIR /workspace

ENTRYPOINT ["bakery"]
