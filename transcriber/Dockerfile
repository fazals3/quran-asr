FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-venv \
      python3-dev \
      build-essential \
      ffmpeg \
      libsndfile1 \
      ca-certificates \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv && pip install --upgrade pip setuptools wheel Cython packaging
COPY transcriber/requirements.txt /app/requirements.txt

# CUDA 12.8 torch build matching the tested NeMo 2.6.1 environment.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchaudio==2.10.0
RUN pip install -r /app/requirements.txt

COPY transcriber/app /app/app

EXPOSE 9000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
