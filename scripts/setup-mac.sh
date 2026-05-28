#!/usr/bin/env bash
# One-time setup for the macOS (Apple Silicon) native stack.
# Verifies tools, pulls the git-lfs assets (model + DB + alignment), ensures the
# Quran DB exists, and pre-builds the Swift transcriber, Rust API, and demo deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf '\033[1;36m[setup-mac]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup-mac] %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[setup-mac] %s\033[0m\n' "$*" >&2; exit 1; }

# ---- Required tools --------------------------------------------------------
log "checking required tools ..."
missing=0
for tool in swift cargo node npm ffmpeg python3 git git-lfs; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '  ok   %s\n' "$tool"
  else
    printf '  MISS %s\n' "$tool"; missing=1
  fi
done
[ "$missing" -eq 0 ] || die "install the missing tools above, then re-run. See docs/mac.md."

case "$(uname -sm)" in
  "Darwin arm64") : ;;
  *) warn "expected Darwin arm64 (Apple Silicon); CoreML/ANE acceleration needs Apple Silicon." ;;
esac

# ---- Git LFS assets (CoreML model, Quran DB, alignment .npy) ----------------
log "fetching git-lfs assets (model + db + alignment) ..."
git lfs install >/dev/null 2>&1 || true
git lfs pull || warn "git lfs pull failed — if you didn't clone with LFS, install it and retry."

# ---- Quran DB (regenerate if the LFS file wasn't materialized) --------------
DB="$ROOT/data/quran.db"
if [ ! -f "$DB" ] || ! file "$DB" | grep -qi sqlite; then
  log "building data/quran.db from data/quran.json ..."
  python3 tools/build_quran_db.py --input "$ROOT/data/quran.json" --output "$DB"
fi

# ---- Model sanity ----------------------------------------------------------
ENC="$ROOT/mac/models/ultra-fast-tarteel-coreml/AudioEncoder.mlmodelc/weights/weight.bin"
if [ ! -f "$ENC" ] || [ "$(wc -c < "$ENC")" -lt 1000000 ]; then
  die "CoreML model weights are missing/too small.
Run 'git lfs install && git lfs pull' to fetch the bundled model."
fi

# ---- Builds ----------------------------------------------------------------
log "building Swift transcriber (release) — first build downloads WhisperKit/Vapor ..."
( cd mac/transcriber-swift && swift build -c release )

log "building Rust API (release) ..."
( cd api && cargo build --release )

log "installing demo dependencies ..."
( cd demo && npm install )

log "setup complete. Start everything with:  ./scripts/run-mac.sh   (or: make mac)"
