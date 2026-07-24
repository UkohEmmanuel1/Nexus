FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 as builder

ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip \
    git build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY pyproject.toml .
RUN python -m pip install -e ".[all]"

COPY . .

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages

WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["python", "-m", "src.inference.cli"]
CMD ["--help"]
