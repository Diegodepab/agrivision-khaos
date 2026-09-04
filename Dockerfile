FROM python:3.13-slim

ARG TORCH_EXTRA=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/opt/venv/bin:${PATH}"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra "${TORCH_EXTRA}"

COPY . .
RUN uv sync --frozen --no-dev --extra "${TORCH_EXTRA}"

CMD ["uv", "run", "--no-sync", "agrivision-quality", "--workers", "4"]
