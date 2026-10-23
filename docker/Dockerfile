FROM python:3.12-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements-serve.txt ./
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF" CMAKE_BUILD_PARALLEL_LEVEL=2
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements-serve.txt

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 nexus && useradd --uid 10001 --gid nexus --create-home nexus
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels
WORKDIR /app
COPY --chown=nexus:nexus src/ ./src/
RUN mkdir -p /var/data && chown nexus:nexus /var/data
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    NEXUS_MODEL_BACKEND=gguf \
    NEXUS_KEYS_DB=/var/data/nexus_keys.db HF_HOME=/var/data/huggingface \
    NEXUS_WEB_INDEX_DB=/var/data/nexus_web_index.db NEXUS_ENABLE_CRAWLER=0
USER nexus
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/ready', timeout=4)"
CMD ["python", "-m", "src.inference.server"]
