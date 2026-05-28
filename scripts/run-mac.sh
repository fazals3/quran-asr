#!/usr/bin/env bash
# Run the full Quran-ASR stack natively on macOS (Apple Silicon):
#   - Swift WhisperKit transcriber (CoreML, ANE-accelerated)
#   - Rust API (alignment, guessing, streaming, jobs)
#   - React/Vite demo frontend
#
# One command. Ctrl-C stops everything.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- Load optional .env.mac, then apply repo-relative defaults --------------
if [ -f .env.mac ]; then
  set -a; . ./.env.mac; set +a
fi

export DATA_DIR="${DATA_DIR:-$ROOT/data}"
export QURAN_DB_PATH="${QURAN_DB_PATH:-$ROOT/data/quran.db}"
export AYAH_ALIGN_MODE="${AYAH_ALIGN_MODE:-v2}"
export AYAH_ALIGN_V2_ASSETS_DIR="${AYAH_ALIGN_V2_ASSETS_DIR:-$ROOT/data/alignment}"
export TRANSCRIBER_URLS="${TRANSCRIBER_URLS:-http://127.0.0.1:${TRANSCRIBER_PORT:-9000}}"
export API_KEY="${API_KEY:-localdev}"
export DEMO_ENABLED="${DEMO_ENABLED:-true}"

export MODEL_DIR="${MODEL_DIR:-$ROOT/mac/models/ultra-fast-tarteel-coreml}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-$ROOT/mac/models/whisper-base-tokenizer}"
export PORT="${TRANSCRIBER_PORT:-9000}"        # consumed by the Swift transcriber
export HOST="127.0.0.1"

log() { printf '\033[1;36m[run-mac]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[run-mac] %s\033[0m\n' "$*" >&2; exit 1; }

# ---- Preflight: required tools ---------------------------------------------
for tool in swift cargo node npm ffmpeg python3; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool (see docs/mac.md)"
done

# ---- Ensure the Quran DB exists (regenerate if the LFS file wasn't pulled) --
if [ ! -f "$QURAN_DB_PATH" ] || ! file "$QURAN_DB_PATH" | grep -qi sqlite; then
  log "building quran.db from data/quran.json ..."
  python3 tools/build_quran_db.py --input "$ROOT/data/quran.json" --output "$QURAN_DB_PATH"
fi

# ---- Sanity-check bundled model came down via git-lfs -----------------------
enc="$MODEL_DIR/AudioEncoder.mlmodelc/weights/weight.bin"
if [ ! -f "$enc" ] || [ "$(wc -c < "$enc")" -lt 1000000 ]; then
  die "CoreML model weights missing/too small at $MODEL_DIR.
Run 'git lfs install && git lfs pull' to fetch the bundled model."
fi

# ---- Build (incremental; fast after first run) ------------------------------
log "building Swift transcriber (release) ..."
( cd mac/transcriber-swift && swift build -c release )
TRANSCRIBER_BIN="$ROOT/mac/transcriber-swift/.build/release/tarteel-transcriber"

log "building Rust API (release) ..."
( cd api && cargo build --release )
API_BIN="$ROOT/api/target/release/quran-asr-api"

if [ ! -d demo/node_modules ]; then
  log "installing demo dependencies ..."
  ( cd demo && npm install )
fi

# ---- Launch all three, clean up on exit ------------------------------------
pids=()
cleanup() {
  log "shutting down ..."
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "starting transcriber on http://127.0.0.1:$PORT (loading CoreML model) ..."
"$TRANSCRIBER_BIN" & pids+=($!)

# Wait for the transcriber to load the model and answer /health.
log "waiting for transcriber to load model ..."
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "transcriber ready."
    break
  fi
  [ "$i" -eq 60 ] && die "transcriber did not become ready in time (check logs above)"
  sleep 1
done

log "starting Rust API on http://127.0.0.1:8001 ..."
"$API_BIN" & pids+=($!)

log "starting demo on http://localhost:3000 ..."
( cd demo && npm run dev ) & pids+=($!)

log "all services up. Open http://localhost:3000  (Ctrl-C to stop)"
wait
