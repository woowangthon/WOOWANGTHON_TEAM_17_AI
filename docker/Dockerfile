FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace
COPY requirements-serving.txt /workspace/requirements-serving.txt
COPY requirements-training.txt /workspace/requirements-training.txt
COPY requirements-kernels.txt /workspace/requirements-kernels.txt
RUN python -m pip install --upgrade pip==25.3 \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 \
    && python -m pip install -r /workspace/requirements-serving.txt -r /workspace/requirements-training.txt \
    && python -m pip install -r /workspace/requirements-kernels.txt

COPY . /workspace
RUN python -m pip install --no-deps -e /workspace

EXPOSE 18117
ENTRYPOINT ["python"]
