# Running on macOS (Apple Silicon, CoreML)

On Apple Silicon the project runs **fully native** — no Docker, no CUDA, no Python ASR
service. The Python `faster-whisper` transcriber is replaced by a small Swift service
built on [WhisperKit](https://github.com/argmaxinc/WhisperKit) that runs a CoreML
conversion of the Tarteel Quran model on the Apple Neural Engine (~30–40× realtime).

The rest of the system is unchanged: the Rust API still does Quran search, surah
guessing, v2 ayah alignment, and streaming (alignment is pure Rust and needs no
embedder at runtime), and the React demo is the same.

```
┌────────────────────┐   multipart /v1/transcribe   ┌──────────────────────────┐
│  Rust API (8001)   │ ───────────────────────────► │  Swift WhisperKit (9000) │
│  align · guess ·   │ ◄─────────────────────────── │  CoreML on the ANE       │
│  streaming · jobs  │     transcription JSON        └──────────────────────────┘
└─────────┬──────────┘
          │  React + Vite demo (3000)
          ▼
       browser
```

## What's bundled (git-lfs)

A fresh clone already contains everything needed to run — no downloads, no asset
hunting. These are stored with Git LFS (so install `git-lfs` before cloning):

- `data/quran.db` — prebuilt SQLite Quran DB (regenerated automatically if absent).
- `data/alignment/<version>/` — v2 alignment assets (`.npy` + `meta.json`).
- `mac/models/ultra-fast-tarteel-coreml/` — the CoreML model (WhisperKit format).
- `mac/models/whisper-base-tokenizer/` — the Whisper tokenizer (offline, no HF fetch).

## Prerequisites

Install once (Homebrew + Xcode toolchain):

```bash
xcode-select --install              # or install Xcode (Swift 6 toolchain)
brew install git-lfs ffmpeg node rust
git lfs install
```

You need: **Swift 6** (Xcode 16+), **Rust/cargo**, **Node 18+**, **ffmpeg**,
**python3** (only to (re)build the DB), and **git-lfs**.

## Run it (two commands)

```bash
git clone <repo-url> quran-asr && cd quran-asr   # LFS pulls model/db/assets
./scripts/setup-mac.sh        # one-time: verify tools, build everything
./scripts/run-mac.sh          # start transcriber + API + demo (Ctrl-C to stop)
```

Then open <http://localhost:3000> and upload / record / stream a recitation.

`run-mac.sh` is idempotent: it builds anything not yet built, so you can also skip
`setup-mac.sh` and just run it. `make mac` / `make setup` are equivalents.

Optional config: `cp .env.mac.example .env.mac` and edit. By default the API key is
`localdev`, demo mode is on, and all paths point at the checkout.

## How transcription works

1. The Rust API POSTs audio (multipart `file`) to the Swift service at `:9000`,
   exactly as it did to the Python transcriber.
2. The Swift service decodes to 16 kHz mono with `ffmpeg`, splits long recitations on
   silence (the Tarteel model emits ~one ayah per decode), runs WhisperKit per chunk
   with word timestamps, and returns the **same JSON shape** the Rust API expects:
   `{ "transcription": { language, duration_s, segments[ start_s, end_s, text,
   words[ word, start_s, end_s, probability ] ] }, "text", "params" }`.
3. The Rust API runs guessing + v2 alignment as before.

## Endpoints (transcriber)

- `GET /health` → `{"ok":true,"backend":"whisperkit-coreml","model_loaded":true}`
- `POST /v1/transcribe` (multipart `file`) → transcription JSON

Env vars (set by `run-mac.sh`): `MODEL_DIR`, `TOKENIZER_DIR`, `PORT` (9000),
`API_KEY`, `LANGUAGE` (ar), `WORD_TIMESTAMPS`.

## Verified

End-to-end on real recitation:

- Al-Ikhlas (112): guess → 112, alignment → 112:1–112:4 (conf ~0.96).
- An-Nas (114, 50 s): guess → 114, alignment → 114:1–114:6 (conf 1.0),
  ~1.3 s to transcribe 50 s of audio.

## Notes & limits

- First transcriber start compiles the CoreML model for the ANE (~30 s once; cached
  afterward, ~1 s).
- The CUDA/Docker stack (`docker-compose.yml`, `transcriber/`) is untouched and still
  works for Linux/GPU deployments — this is a parallel, Mac-native path.
- Streaming uses the same Swift service (each window is a `window.wav` POST).
