# Quran ASR

Rust-backed Quran recitation transcription and ayah alignment.

`quran-asr` accepts uploaded audio, transcribes Arabic Quran recitation with a faster-whisper ASR service, then force-matches the transcript to Quran verses with a Rust alignment pipeline. The API returns the transcript, likely surah/ayah range, multi-span alignment, jump detection, and timing metrics.

## What It Does

- Runs ASR in a Python FastAPI service using `faster-whisper`.
- Runs Quran search, verse guessing, and alignment in Rust for lower CPU latency.
- Supports batch job transcription over HTTP.
- Supports experimental low-latency streaming over WebSockets.
- Keeps generated audio jobs, Hugging Face caches, and model files out of git.
- Automatically cleans old `/data/jobs` artifacts with configurable retention.

## Repository Layout

```text
api/              Rust API, alignment, guessing, streaming, and job processing
transcriber/      Python FastAPI ASR service
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
