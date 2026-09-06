"""FastConformer (NVIDIA NeMo) transcriber service.

Wraps the `Muno459/fastconformer-quran` hybrid RNNT/CTC checkpoint behind the
same HTTP contract the Rust API already speaks:

    POST /v1/transcribe  multipart `file` -> {"transcription": {...}, "text": ..., ...}
    POST /v1/embed       {"text": ...}    -> float16 embedding (unchanged)
    GET  /health

Long audio is split at detected silences (hard splits with overlap when no
silence is found), each chunk is decoded with the CTC head and greedy batched
decoding, and per-chunk word timestamps are stitched back onto the global
timeline. Word confidences come from NeMo's max-prob confidence estimator.
"""

import asyncio
import base64
import copy
import gc
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

SAMPLE_RATE = 16000
_UNK_WORDS = {"<unk>", "⁇"}


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None else str(v)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(float(seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


@dataclass(frozen=True)
class Settings:
    api_key: str
    model_id: str
    model_path: str
    model_revision: str
    language: str
    device: str
    decoder_type: str

    batch_size: int
    batch_size_cap: int

    chunk_max_s: float
    chunk_overlap_s: float
    chunk_silence_noise_db: float
    chunk_silence_min_s: float
    segment_gap_s: float
    word_confidence: bool

    embed_model_id: str
    embed_device: str
    warm_embedder: bool

    max_upload_bytes: int
    max_concurrent_transcribes: int


def load_settings() -> Settings:
    return Settings(
        api_key=_env("API_KEY", "").strip(),
        model_id=_env("MODEL_ID", "Muno459/fastconformer-quran").strip(),
        model_path=_env("MODEL_PATH", "").strip(),
        model_revision=_env("MODEL_REVISION", "").strip(),
        language=_env("LANGUAGE", "ar"),
        device=_env("DEVICE", "cuda").strip().lower(),
        decoder_type=_env("DECODER_TYPE", "ctc").strip().lower(),
        batch_size=_env_int("BATCH_SIZE", 8),
        batch_size_cap=_env_int("BATCH_SIZE_CAP", 32),
        chunk_max_s=_env_float("CHUNK_MAX_S", 15.0),
        chunk_overlap_s=_env_float("CHUNK_OVERLAP_S", 3.0),
        chunk_silence_noise_db=_env_float("CHUNK_SILENCE_NOISE_DB", -32.0),
        chunk_silence_min_s=_env_float("CHUNK_SILENCE_MIN_S", 0.35),
        segment_gap_s=_env_float("SEGMENT_GAP_S", 1.0),
        word_confidence=_env_bool("WORD_CONFIDENCE", True),
        embed_model_id=_env("EMBEDDING_MODEL_ID", "").strip(),
        embed_device=_env("EMBEDDING_DEVICE", "cuda").strip(),
        warm_embedder=_env_bool("WARM_EMBEDDER", False),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 200 * 1024 * 1024),
        max_concurrent_transcribes=_env_int("MAX_CONCURRENT_TRANSCRIBES", 1),
    )


settings = load_settings()
app = FastAPI(title="Rust-backed Transcriber", version="2.0")

_model: Any = None
_model_source: str = ""
_model_lock = threading.Lock()
_embedder: Any = None
_sem = asyncio.Semaphore(max(1, int(settings.max_concurrent_transcribes)))
_batch_size_cap = max(1, int(settings.batch_size_cap))
_batch_size_lock = threading.Lock()


def _require_internal_access(request: Request) -> None:
    if not settings.api_key:
        return
    auth = request.headers.get("authorization") or ""
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token or token != settings.api_key:
        raise HTTPException(status_code=401, detail="unauthorized")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_checkpoint() -> str:
    """Return a local `.nemo` path: `MODEL_PATH` if set, else download from Hugging Face."""
    if settings.model_path:
        if not os.path.isfile(settings.model_path):
            raise RuntimeError(f"MODEL_PATH does not exist: {settings.model_path}")
        return settings.model_path

    from huggingface_hub import hf_hub_download, list_repo_files

    revision = settings.model_revision or None
    files = list_repo_files(settings.model_id, revision=revision)
    nemo_files = sorted(f for f in files if f.endswith(".nemo"))
    if not nemo_files:
        raise RuntimeError(f"no .nemo checkpoint found in {settings.model_id}: {files}")
    return hf_hub_download(settings.model_id, nemo_files[0], revision=revision)


def _lazy_load_model() -> Any:
    global _model
    global _model_source
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model

        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.utils import logging as nemo_logging
        from omegaconf import open_dict

        nemo_logging.setLevel(nemo_logging.WARNING)

        device = settings.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        ckpt = _resolve_checkpoint()
        model = nemo_asr.models.ASRModel.restore_from(ckpt, map_location=device)
        model.eval()

        # Hybrid RNNT+CTC checkpoint: the CTC head gives frame-accurate word
        # timestamps and avoids RNNT CUDA-graph decoding.
        if hasattr(model, "change_decoding_strategy"):
            decoder_type = settings.decoder_type if settings.decoder_type in {"ctc", "rnnt"} else "ctc"
            if decoder_type == "ctc" and hasattr(model.cfg, "aux_ctc"):
                decoding_cfg = copy.deepcopy(model.cfg.aux_ctc.decoding)
            else:
                decoding_cfg = copy.deepcopy(model.cfg.decoding)
            with open_dict(decoding_cfg):
                decoding_cfg.strategy = "greedy_batch"
                decoding_cfg.compute_timestamps = True
                decoding_cfg.preserve_alignments = True
                if settings.word_confidence:
                    decoding_cfg.confidence_cfg = {
                        "preserve_frame_confidence": True,
                        "preserve_token_confidence": True,
                        "preserve_word_confidence": True,
                        "exclude_blank": True,
                        "aggregation": "min",
                        "method_cfg": {"name": "max_prob"},
                    }
            try:
                model.change_decoding_strategy(decoding_cfg=decoding_cfg, decoder_type=decoder_type)
            except TypeError:
                model.change_decoding_strategy(decoding_cfg=decoding_cfg)

        _model_source = ckpt
        _model = model
        return _model


def _lazy_load_embedder() -> Any:
    global _embedder
    if _embedder is not None:
        return _embedder
    if not settings.embed_model_id:
        raise RuntimeError("EMBEDDING_MODEL_ID is not set")
    from sentence_transformers import SentenceTransformer

    _embedder = SentenceTransformer(settings.embed_model_id, device=settings.embed_device, trust_remote_code=True)
    return _embedder


@app.on_event("startup")
async def _startup() -> None:
    _lazy_load_model()
    if settings.warm_embedder and settings.embed_model_id:
        try:
            emb = _lazy_load_embedder().encode(["الحمد لله"], normalize_embeddings=True, show_progress_bar=False)
            _ = np.asarray(emb, dtype=np.float32)
        except Exception:
            pass


@app.get("/health")
def health(_=Depends(_require_internal_access)) -> Dict[str, Any]:
    return {
        "ok": True,
        "time": time.time(),
        "backend": "nemo-fastconformer",
        "model_id": settings.model_id,
        "model_path": settings.model_path or None,
        "model_loaded": _model is not None,
        "device": settings.device,
        "decoder_type": settings.decoder_type,
        "batch_size": int(settings.batch_size),
        "batch_size_cap": int(_batch_size_cap),
        "chunk_max_s": float(settings.chunk_max_s),
        "chunk_overlap_s": float(settings.chunk_overlap_s),
        "max_concurrent_transcribes": int(settings.max_concurrent_transcribes),
        "embedding_model_id": settings.embed_model_id or None,
        "embedding_device": settings.embed_device or None,
    }


def _safe_stream_copy(src, dst_path: str, *, max_bytes: int) -> int:
    total = 0
    with open(dst_path, "wb") as f:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > int(max_bytes):
                raise ValueError(f"upload too large ({total} bytes > {max_bytes})")
            f.write(chunk)
    return int(total)


# ---------------------------------------------------------------------------
# Audio decode + silence-aware chunking
# ---------------------------------------------------------------------------

_SIL_RE = re.compile(r"silence_(start|end): ([0-9.]+)")


def decode_audio(path: str, *, noise_db: float, min_sil_s: float) -> Tuple[np.ndarray, List[float]]:
    """Decode any ffmpeg-readable file to 16 kHz mono float32 and detect silences in one pass."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-af",
        f"silencedetect=noise={float(noise_db)}dB:d={float(min_sil_s)}",
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise ValueError("ffmpeg decode failed: " + " | ".join(tail))
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    duration_s = float(audio.shape[0]) / SAMPLE_RATE

    events = [(kind, float(val)) for kind, val in _SIL_RE.findall(proc.stderr.decode("utf-8", "replace"))]
    cuts: List[float] = []
    start: Optional[float] = None
    for kind, t in events:
        if kind == "start":
            start = t
        elif kind == "end" and start is not None:
            # Cut near the start of the silence so the pause and the next words land in the next chunk.
            cuts.append(min((start + t) / 2.0, start + 0.6))
            start = None
    cuts = sorted(c for c in cuts if 0.0 < c < duration_s)
    return audio, cuts


@dataclass(frozen=True)
class Chunk:
    start_s: float
    end_s: float
    keep_from_s: float  # drop stitched words starting before this
    keep_to_s: float  # drop stitched words starting at/after this


def build_chunks(duration_s: float, cuts: List[float], *, max_chunk_s: float, overlap_s: float) -> List[Chunk]:
    if duration_s <= 0.0:
        return []
    overlap_s = max(0.0, min(float(overlap_s), float(max_chunk_s) / 2.0))
    bounds: List[Tuple[float, float, bool]] = []  # (start, end, hard_split_end)
    pos = 0.0
    while pos < duration_s - 0.05:
        limit = pos + max_chunk_s
        if limit >= duration_s:
            bounds.append((pos, duration_s, False))
            break
        candidates = [c for c in cuts if pos + 1.0 < c <= limit]
        if candidates:
            bounds.append((pos, candidates[-1], False))
            pos = candidates[-1]
        else:
            bounds.append((pos, limit, True))
            pos = limit - overlap_s
    if not bounds:
        bounds.append((0.0, duration_s, False))
    chunks: List[Chunk] = []
    for i, (s, e, hard) in enumerate(bounds):
        keep_from = s
        if i > 0:
            _prev_s, prev_e, prev_hard = bounds[i - 1]
            if prev_hard:
                keep_from = prev_e - overlap_s / 2.0
        keep_to = e
        if hard:
            keep_to = e - overlap_s / 2.0
        chunks.append(Chunk(start_s=s, end_s=e, keep_from_s=keep_from, keep_to_s=keep_to))
    return chunks


def _slice(audio: np.ndarray, chunk: Chunk) -> np.ndarray:
    a = int(round(chunk.start_s * SAMPLE_RATE))
    b = int(round(chunk.end_s * SAMPLE_RATE))
    return np.ascontiguousarray(audio[a:b], dtype=np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _words_from_hypothesis(hyp: Any, offset_s: float) -> List[Dict[str, Any]]:
    ts = getattr(hyp, "timestamp", None) or {}
    raw_words = ts.get("word", []) if isinstance(ts, dict) else []
    confs = getattr(hyp, "word_confidence", None)
    use_conf = isinstance(confs, (list, tuple)) and len(confs) == len(raw_words)

    out: List[Dict[str, Any]] = []
    for i, w in enumerate(raw_words):
        text = str(w.get("word") or "").strip()
        if not text or text in _UNK_WORDS:
            continue
        start = w.get("start")
        if start is None:
            continue
        end = w.get("end")
        ws = float(start) + offset_s
        we = (float(end) if end is not None else float(start)) + offset_s
        if we < ws:
            we = ws
        item: Dict[str, Any] = {
            "start_s": ws,
            "end_s": we,
            "start_ts": format_timestamp(ws),
            "end_ts": format_timestamp(we),
            "word": text,
        }
        if use_conf:
            try:
                item["probability"] = float(confs[i])
            except Exception:
                pass
        out.append(item)
    return out


def _group_segments(words: List[Dict[str, Any]], *, gap_s: float) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []
    for w in words:
        if cur and (float(w["start_s"]) - float(cur[-1]["end_s"])) > float(gap_s):
            segments.append(_make_segment(cur))
            cur = []
        cur.append(w)
    if cur:
        segments.append(_make_segment(cur))
    return segments


def _make_segment(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    start_s = float(words[0]["start_s"])
    end_s = float(max(float(w["end_s"]) for w in words))
    probs = [float(w["probability"]) for w in words if "probability" in w]
    seg: Dict[str, Any] = {
        "start_s": start_s,
        "end_s": end_s,
        "start_ts": format_timestamp(start_s),
        "end_ts": format_timestamp(end_s),
        "text": " ".join(str(w["word"]) for w in words),
        "words": words,
    }
    if probs:
        seg["avg_probability"] = float(sum(probs) / len(probs))
    return seg


def _is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda failed" in msg or "cublas_status_alloc_failed" in msg


def _run_model(model: Any, clips: List[np.ndarray], *, batch_size: int) -> List[Any]:
    import torch

    with torch.inference_mode():
        hyps = model.transcribe(
            clips,
            batch_size=int(batch_size),
            return_hypotheses=True,
            timestamps=True,
            num_workers=0,
            verbose=False,
        )
    if isinstance(hyps, tuple):  # some NeMo versions return (best, n-best)
        hyps = hyps[0]
    return list(hyps)


def _transcribe_file(*, audio_path: str, batch_size: int) -> Tuple[Dict[str, Any], str, int]:
    global _batch_size_cap
    model = _lazy_load_model()

    audio, cuts = decode_audio(
        audio_path,
        noise_db=float(settings.chunk_silence_noise_db),
        min_sil_s=float(settings.chunk_silence_min_s),
    )
    duration_s = float(audio.shape[0]) / SAMPLE_RATE
    chunks = build_chunks(
        duration_s,
        cuts,
        max_chunk_s=float(settings.chunk_max_s),
        overlap_s=float(settings.chunk_overlap_s),
    )
    clips = [_slice(audio, c) for c in chunks]
    # Drop chunks too short for the mel front-end (they carry no speech anyway).
    keep = [i for i, clip in enumerate(clips) if clip.shape[0] >= SAMPLE_RATE // 10]
    chunks = [chunks[i] for i in keep]
    clips = [clips[i] for i in keep]

    with _batch_size_lock:
        bs = min(max(1, int(batch_size)), max(1, int(_batch_size_cap)))

    while True:
        try:
            words: List[Dict[str, Any]] = []
            with _model_lock:
                for i in range(0, len(clips), bs):
                    hyps = _run_model(model, clips[i : i + bs], batch_size=bs)
                    for hyp, ch in zip(hyps, chunks[i : i + bs]):
                        for w in _words_from_hypothesis(hyp, ch.start_s):
                            if ch.keep_from_s - 1e-6 <= float(w["start_s"]) < ch.keep_to_s + 1e-6:
                                words.append(w)
            break
        except Exception as e:  # noqa: BLE001 - we only retry on CUDA OOM
            if not _is_oom(e) or bs <= 1:
                raise
            bs = max(1, bs // 2)
            with _batch_size_lock:
                _batch_size_cap = min(max(1, int(_batch_size_cap)), bs)
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    words.sort(key=lambda w: (float(w["start_s"]), float(w["end_s"])))
    segments = _group_segments(words, gap_s=float(settings.segment_gap_s))
    full_text = " ".join(s["text"] for s in segments).strip()

    transcription: Dict[str, Any] = {
        "language": settings.language,
        "language_probability": 1.0,
        "duration_s": duration_s,
        "segments": segments,
        "chunking": {
            "chunks": len(chunks),
            "silence_cuts": len(cuts),
            "max_chunk_s": float(settings.chunk_max_s),
            "overlap_s": float(settings.chunk_overlap_s),
        },
    }
    return transcription, full_text, int(bs)


@app.post("/v1/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = None,
    batch_size: Optional[int] = None,
    word_timestamps: Optional[bool] = None,
    # Accepted for backwards compatibility with Whisper-era clients; not used by the CTC decoder.
    beam_size: Optional[int] = None,
    patience: Optional[float] = None,
    log_prob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
    _=Depends(_require_internal_access),
) -> JSONResponse:
    filename = os.path.basename(file.filename or "") or "upload.m4a"
    with tempfile.TemporaryDirectory(prefix="transcriber_") as td:
        path = os.path.join(td, filename)
        try:
            _safe_stream_copy(file.file, path, max_bytes=int(settings.max_upload_bytes))
        except ValueError as e:
            raise HTTPException(status_code=413, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"upload failed: {e}")

        bs = int(batch_size) if batch_size is not None else int(settings.batch_size)
        lang = str(language) if language is not None else settings.language

        t0 = time.time()
        try:
            async with _sem:
                transcription, text, bs_used = await asyncio.to_thread(
                    _transcribe_file,
                    audio_path=path,
                    batch_size=bs,
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        t_transcribe = time.time() - t0

        transcription["language"] = lang
        ignored = {
            k: v
            for k, v in {
                "beam_size": beam_size,
                "patience": patience,
                "log_prob_threshold": log_prob_threshold,
                "no_speech_threshold": no_speech_threshold,
            }.items()
            if v is not None
        }
        params: Dict[str, Any] = {
            "model_id": settings.model_id,
            "backend": "nemo-fastconformer",
            "decoder_type": settings.decoder_type,
            "language": lang,
            "word_timestamps": True,
            "batch_size": bs_used,
            "chunk_max_s": float(settings.chunk_max_s),
            "chunk_overlap_s": float(settings.chunk_overlap_s),
        }
        if word_timestamps is False:
            params["word_timestamps_requested"] = False
        if ignored:
            params["ignored"] = ignored

        return JSONResponse(
            {
                "input": {"filename": filename, "content_type": file.content_type},
                "transcription": transcription,
                "text": text,
                "params": params,
                "timing": {"transcribe_s": float(t_transcribe)},
            }
        )


@app.post("/v1/embed")
async def embed(
    payload: Dict[str, Any],
    request: Request,
    _=Depends(_require_internal_access),
) -> JSONResponse:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="missing text")
    try:
        model = _lazy_load_embedder()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    t0 = time.time()
    vec = await asyncio.to_thread(
        model.encode,
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 1:
        raise HTTPException(status_code=500, detail="unexpected embedding shape")
    v = np.asarray(arr[0], dtype=np.float16)
    raw = v.tobytes(order="C")
    b64 = base64.b64encode(raw).decode("ascii")
    return JSONResponse(
        {
            "model_id": settings.embed_model_id or None,
            "device": settings.embed_device,
            "dim": int(v.shape[0]),
            "dtype": "float16",
            "embedding_f16_base64": b64,
            "timing": {"embed_s": float(time.time() - t0)},
        }
    )
