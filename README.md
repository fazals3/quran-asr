# Quran ASR

Rust-backed Quran recitation transcription and ayah alignment.

`quran-asr` accepts uploaded audio, transcribes Arabic Quran recitation with a faster-whisper ASR service, then force-matches the transcript to Quran verses with a Rust alignment pipeline. The API returns the transcript, likely surah/ayah range, multi-span alignment, jump detection, and timing metrics.

## What It Does

- Runs ASR in a Python FastAPI service using `faster-whisper` (Linux/GPU), or a native
  Swift [WhisperKit](https://github.com/argmaxinc/WhisperKit) CoreML service on macOS.
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

This project uses Quran-focused Whisper ASR work from the Hugging Face community:

- [`tarteel-ai/whisper-base-ar-quran`](https://huggingface.co/tarteel-ai/whisper-base-ar-quran), an Apache-2.0 Quran fine-tuned Whisper base model from Tarteel AI.
- [`OdyAsh/faster-whisper-base-ar-quran`](https://huggingface.co/OdyAsh/faster-whisper-base-ar-quran), an Apache-2.0 CTranslate2/faster-whisper conversion of the Tarteel model.

Project cleanup, repository organization, and README drafting were assisted by Codex GPT-5.2.

## License

No project license has been selected yet. Add a `LICENSE` file before publishing if you want others to have explicit reuse rights.
