# Architecture

`quran-asr` is split into two runtime services:

- `transcriber`: Python FastAPI wrapper around NVIDIA NeMo running the `Muno459/fastconformer-quran` FastConformer hybrid RNNT/CTC checkpoint. It owns model loading, ffmpeg decoding, silence-aware chunking, batched CTC decoding with word timestamps and confidences, and optional embeddings.
- `api`: Rust Axum service. It owns auth, queueing, file intake, Quran DB lookup, surah guessing, forced ayah alignment, multi-span/jump handling, and streaming session state.

The services communicate over the Docker Compose network. Uploaded files and generated job JSON are stored in `/data/jobs`, which is intentionally ignored by git.

## Batch Flow

1. Client submits `POST /v1/jobs/transcribe` with multipart audio or a JSON `source_url`.
2. Rust writes `request.json` and queues a job.
3. Rust sends the audio to the Python transcriber.
4. Rust runs Quran guessing and v2 alignment in parallel blocking tasks.
5. Rust writes `result.json` and returns it from `GET /v1/jobs/:job_id`.

Completed and failed job directories are retained for `JOB_RETENTION_S`, then removed by a background cleanup task. At startup, the same cleanup also removes stale job directories left behind by previous API runs.

## Transcriber Internals

1. ffmpeg decodes the upload to 16 kHz mono float32 and reports silences (`silencedetect`) in the same pass.
2. Audio longer than `CHUNK_MAX_S` is split at silence midpoints; runs without silence are hard-split with `CHUNK_OVERLAP_S` of overlap.
3. Chunks are decoded in batches with NeMo's greedy batched CTC decoder (`timestamps=True`, max-prob word confidence).
4. Word timestamps are shifted back onto the global timeline; words from overlapped regions are kept from one side of the overlap midpoint only.
5. Words are grouped into segments at pauses longer than `SEGMENT_GAP_S`.

Streaming windows (a few seconds each) fit in a single chunk, so they skip the chunking step entirely.

## Streaming Flow

1. Client creates a session with `POST /v1/sessions`.
2. Client opens `/v1/sessions/:session_id/stream`.
3. Client sends PCM16LE mono 16 kHz frames.
4. Rust buffers windows, calls the transcriber, aligns each window, and emits `ayah_update` events.

## Demo Mode

When `DEMO_ENABLED=true`, the Rust API mounts a `/demo/*` route namespace with:

- `POST /demo/v1/transcribe` — upload and transcribe (no API key, rate limited per IP).
- `POST /demo/v1/sessions` — create a streaming session.
- `GET /demo/v1/sessions/:id/stream` — WebSocket (no auth).
- `POST /demo/v1/sessions/:id/stop` — stop a session.
- `GET /demo/v1/jobs/:id` — poll job result.

These endpoints use the same job queue and streaming infrastructure as the authenticated API. CORS headers are applied so the demo frontend (a standalone React app in `demo/`) can connect from any origin.

Rate limiting is in-memory, per-IP, sliding window (configurable via `DEMO_RATE_LIMIT_PER_MIN`). Upload size and audio duration are capped separately from the main API (`DEMO_MAX_UPLOAD_BYTES`, `DEMO_MAX_AUDIO_DURATION_S`).

## Local Runtime Data

The repo tracks only small source data. Runtime data is local:

- `data/quran.db`: generated from `data/quran.json`.
- `data/alignment/v2`: precomputed alignment assets.
- `data/hf`: downloaded model cache.
- `data/jobs`: uploaded audio and outputs.
