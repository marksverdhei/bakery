FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

LABEL maintainer="marksverdhei"
LABEL description="Bakery — Where LLMs go to get baked"
LABEL org.opencontainers.image.source="https://github.com/marksverdhei/bakery"

ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install bakery and all dependencies (torch already in base image)
WORKDIR /opt/bakery
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/
RUN uv pip install ".[qlora]"

# Verify installation
RUN python3 -c "import bakery; import torch; print(f'bakery OK, torch={torch.__version__}, cuda_built={torch.version.cuda}')"

# Default working directory for training runs
WORKDIR /workspace

ENTRYPOINT ["bakery"]
