# Architecture

`quran-asr` is split into two runtime services:

- `transcriber`: Python FastAPI wrapper around faster-whisper. It owns model loading, batched transcription, conservative repair passes, and optional embeddings.
- `api`: Rust Axum service. It owns auth, queueing, file intake, Quran DB lookup, surah guessing, forced ayah alignment, multi-span/jump handling, and streaming session state.

The services communicate over the Docker Compose network. Uploaded files and generated job JSON are stored in `/data/jobs`, which is intentionally ignored by git.

## Batch Flow

1. Client submits `POST /v1/jobs/transcribe` with multipart audio or a JSON `source_url`.
2. Rust writes `request.json` and queues a job.
3. Rust sends the audio to the Python transcriber.
4. Rust runs Quran guessing and v2 alignment in parallel blocking tasks.
5. Rust writes `result.json` and returns it from `GET /v1/jobs/:job_id`.

Completed and failed job directories are retained for `JOB_RETENTION_S`, then removed by a background cleanup task. At startup, the same cleanup also removes stale job directories left behind by previous API runs.

## Streaming Flow

1. Client creates a session with `POST /v1/sessions`.
2. Client opens `/v1/sessions/:session_id/stream`.
3. Client sends PCM16LE mono 16 kHz frames.
4. Rust buffers windows, calls the transcriber, aligns each window, and emits `ayah_update` events.

## Local Runtime Data

The repo tracks only small source data. Runtime data is local:

- `data/quran.db`: generated from `data/quran.json`.
- `data/alignment/v2`: precomputed alignment assets.
- `data/hf`: downloaded model cache.
- `data/jobs`: uploaded audio and outputs.
