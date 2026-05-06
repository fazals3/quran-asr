FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-setuptools \
      python3-wheel \
      ffmpeg \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip setuptools wheel
COPY transcriber/requirements.txt /app/requirements.txt

# CUDA-enabled torch is needed for GPU embeddings (SentenceTransformers / transformers).
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
RUN python3 -m pip install -r /app/requirements.txt

COPY transcriber/app /app/app

EXPOSE 9000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
