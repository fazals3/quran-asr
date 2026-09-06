# Quran ASR

Rust-backed Quran recitation transcription and ayah alignment.

`quran-asr` accepts uploaded audio, transcribes Arabic Quran recitation with an NVIDIA NeMo FastConformer ASR service, then force-matches the transcript to Quran verses with a Rust alignment pipeline. The API returns the transcript, likely surah/ayah range, multi-span alignment, jump detection, and timing metrics.

## What It Does

- Runs ASR in a Python FastAPI service using NVIDIA NeMo and the
  [`Muno459/fastconformer-quran`](https://huggingface.co/Muno459/fastconformer-quran)
  FastConformer hybrid RNNT/CTC model (Linux/GPU). The CTC head produces frame-accurate
  word timestamps and per-word confidences; long audio is chunked at detected silences.
- On macOS, a native Swift [WhisperKit](https://github.com/argmaxinc/WhisperKit) CoreML
  service still runs the Tarteel Whisper model (the FastConformer checkpoint has no CoreML build).
- Runs Quran search, verse guessing, and alignment in Rust for lower CPU latency.
- Supports batch job transcription over HTTP.
- Supports experimental low-latency streaming over WebSockets.
- Keeps generated audio jobs, Hugging Face caches, and model files out of git.
- Automatically cleans old `/data/jobs` artifacts with configurable retention.

## Quick start on macOS (Apple Silicon, CoreML)

Runs fully native (no Docker/CUDA) using a CoreML conversion of the Tarteel model on the
Apple Neural Engine. The model, Quran DB, and alignment assets are bundled via **git-lfs**,
so setup is two commands:

```bash
# Prereqs: Xcode (Swift 6), Rust, Node, ffmpeg, git-lfs   (see docs/mac.md)
git lfs install
git clone <repo-url> quran-asr && cd quran-asr   # LFS pulls model + db + assets
./scripts/setup-mac.sh        # one-time: verify tools + build everything
./scripts/run-mac.sh          # start transcriber + API + demo, then open localhost:3000
```

Full details, architecture, and verification: [`docs/mac.md`](docs/mac.md).
The Docker/CUDA instructions below remain the path for Linux/GPU deployments.

## Repository Layout

```text
api/              Rust API, alignment, guessing, streaming, and job processing
transcriber/      Python FastAPI ASR service
demo/             Standalone web demo frontend (React + Vite)
tools/            Local simulation, A/B comparison, and Quran DB build helpers
data/             Quran source JSON plus local runtime data mount
docs/             Additional operational notes
```

## Requirements

- Docker with Compose.
- NVIDIA Container Toolkit for GPU inference.
- A CUDA-capable GPU for the default `docker-compose.yml`.
- Python 3 if you want to build `data/quran.db` locally.

For CPU-only smoke tests, use `docker-compose.cpu.yml`; expect much slower ASR.

## Setup

1. Create the SQLite Quran database:

```bash
python3 tools/build_quran_db.py --input data/quran.json --output data/quran.db
```

2. Add v2 alignment assets under:

```text
data/alignment/v2/<asset-version>/meta.json
```

`AYAH_ALIGN_V2_ASSETS_DIR=/data/alignment/v2` can point either at the parent directory or directly at one asset version. The Rust API picks the newest child directory that contains `meta.json`.

3. Configure secrets:

```bash
cp .env.example .env
```

Set at least `API_KEY`, `STREAM_JWT_SECRET`, and the GPU/throughput knobs that match your machine.

4. Start the stack:

```bash
docker compose up -d --build
```

The transcriber downloads the `.nemo` checkpoint from Hugging Face into `data/hf` on first
start. To use a local copy instead, place it under `data/` and set `MODEL_PATH` in `.env`
to its path inside the container, e.g. `MODEL_PATH=/data/models/fastconformer-quran.nemo`.

## Transcriber Tuning

The Python service exposes a few knobs in `.env`:

- `DECODER_TYPE=ctc` selects the CTC head (default, frame-accurate timestamps) or `rnnt`.
- `CHUNK_MAX_S=15` / `CHUNK_OVERLAP_S=3.0` control how long recordings are split. Splits prefer
  detected silences (`CHUNK_SILENCE_NOISE_DB`, `CHUNK_SILENCE_MIN_S`); hard splits overlap and are
  deduplicated at the overlap midpoint.
- `SEGMENT_GAP_S=1.0` groups words into output segments at pauses longer than this.
- `BATCH_SIZE=8` is the number of chunks per forward pass; `BATCH_SIZE_CAP` ratchets down on CUDA OOM.
- `WORD_CONFIDENCE=true` attaches a `probability` to each word from NeMo's max-prob estimator.

The `/v1/transcribe` response keeps the same shape as before (`segments[].words[]` with
`start_s`/`end_s`), so the Rust API and tools are unchanged. Whisper-era query parameters
(`beam_size`, `patience`, `log_prob_threshold`, `no_speech_threshold`) are accepted and echoed
under `params.ignored` but have no effect on the CTC decoder.

## API

Health:

```bash
curl -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8001/health
```

Upload and wait for a result:

```bash
curl -X POST "http://127.0.0.1:8001/v1/jobs/transcribe?wait=true&wait_timeout_s=120" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@recitation.mp3"
```

Create a streaming session:

```bash
curl -X POST http://127.0.0.1:8001/v1/sessions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"window_s":8,"hop_s":2,"buffer_s":24}'
```

The streaming WebSocket accepts binary PCM16LE mono audio at 16 kHz.

## Demo Mode

An optional public demo UI (similar to Hugging Face Spaces) lets anyone try the system without an API key. It supports file upload, in-browser recording, and real-time streaming — all capped at 1 minute of audio.

Demo mode is **off by default**. To enable:

```bash
# In .env
DEMO_ENABLED=true
```

This mounts `/demo/*` endpoints on the API with CORS and per-IP rate limiting. The frontend is a standalone React app in `demo/` that can run on a separate machine from the API.

Quick start:

```bash
docker compose -f docker-compose.demo.yml up -d --build
```

Or run the frontend locally for development:

```bash
cd demo && npm install && npm run dev
```

See `docs/demo.md` for a full walkthrough.

## Job Cleanup

The Rust API periodically deletes old job artifact directories from `/data/jobs`.

- `JOB_CLEANUP_ENABLED=true` enables the cleanup loop.
- `JOB_RETENTION_S=86400` keeps completed or failed jobs for 24 hours by default.
- `JOB_CLEANUP_INTERVAL_S=3600` runs cleanup hourly after startup.
- `JOB_CLEANUP_STARTUP=true` also runs one cleanup pass when the API boots.
- `JOB_CLEANUP_MAX_DELETE_PER_RUN=200` caps deletes per pass so cleanup cannot monopolize disk I/O.

Queued and running jobs tracked by the API are never deleted. Job directories left over from previous API runs are cleaned by directory modification time.

## Model Credits

This project uses Quran-focused ASR work from the Hugging Face community:

- [`Muno459/fastconformer-quran`](https://huggingface.co/Muno459/fastconformer-quran), a NeMo
  FastConformer hybrid RNNT/CTC model fine-tuned for Quranic recitation from
  [`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0).
  It is the default Linux/GPU transcriber. Check the model card for its license terms before redistributing.
- [`tarteel-ai/whisper-base-ar-quran`](https://huggingface.co/tarteel-ai/whisper-base-ar-quran), an Apache-2.0 Quran fine-tuned Whisper base model from Tarteel AI, used by the macOS CoreML path.
- [`OdyAsh/faster-whisper-base-ar-quran`](https://huggingface.co/OdyAsh/faster-whisper-base-ar-quran), an Apache-2.0 CTranslate2/faster-whisper conversion of the Tarteel model, used by earlier releases of this project.

Project cleanup, repository organization, and README drafting were assisted by Codex GPT-5.2.

## License

No project license has been selected yet. Add a `LICENSE` file before publishing if you want others to have explicit reuse rights.
