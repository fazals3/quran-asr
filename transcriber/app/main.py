import asyncio
import base64
import math
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse


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


def _env_float_optional(name: str, default: Optional[float]) -> Optional[float]:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    s = str(v).strip().lower()
    if s in {"none", "null", "off", "false", "disabled"}:
        return None
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
    language: str
    device: str
    compute_type: str
    transcribe_backend: str

    beam_size: int
    patience: float
    word_timestamps: bool
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    batch_size: int
    batch_size_cap: int

    embed_model_id: str
    embed_device: str
    warm_embedder: bool

    enable_repair: bool
    repair_max_windows: int
    repair_context_s: float
    repair_gap_trigger_s: float
    repair_low_words_duration_s: float
    repair_low_words_max_words: int

    enable_dedupe: bool
    dedupe_min_words: int
    dedupe_max_words: int
    dedupe_max_gap_s: float

    max_upload_bytes: int
    max_concurrent_transcribes: int


def load_settings() -> Settings:
    return Settings(
        api_key=_env("API_KEY", "").strip(),
        model_id=_env("MODEL_ID", "OdyAsh/faster-whisper-base-ar-quran"),
        language=_env("LANGUAGE", "ar"),
        device=_env("DEVICE", "cuda"),
        compute_type=_env("COMPUTE_TYPE", "float16"),
        transcribe_backend=_env("TRANSCRIBE_BACKEND", "batched").strip().lower(),
        beam_size=_env_int("BEAM_SIZE", 5),
        patience=_env_float("PATIENCE", 1.2),
        word_timestamps=_env_bool("WORD_TIMESTAMPS", True),
        log_prob_threshold=_env_float_optional("WHISPER_LOG_PROB_THRESHOLD", -0.15),
        no_speech_threshold=_env_float_optional("WHISPER_NO_SPEECH_THRESHOLD", 0.0),
        batch_size=_env_int("BATCH_SIZE", 32),
        batch_size_cap=_env_int("BATCH_SIZE_CAP", 128),
        embed_model_id=_env("EMBEDDING_MODEL_ID", "").strip(),
        embed_device=_env("EMBEDDING_DEVICE", "cuda").strip(),
        warm_embedder=_env_bool("WARM_EMBEDDER", False),
        enable_repair=_env_bool("TRANSCRIBE_ENABLE_REPAIR", True),
        repair_max_windows=_env_int("TRANSCRIBE_REPAIR_MAX_WINDOWS", 3),
        repair_context_s=_env_float("TRANSCRIBE_REPAIR_CONTEXT_S", 3.0),
        repair_gap_trigger_s=_env_float("TRANSCRIBE_REPAIR_GAP_TRIGGER_S", 6.0),
        repair_low_words_duration_s=_env_float("TRANSCRIBE_REPAIR_LOW_WORDS_DURATION_S", 12.0),
        repair_low_words_max_words=_env_int("TRANSCRIBE_REPAIR_LOW_WORDS_MAX_WORDS", 6),
        enable_dedupe=_env_bool("TRANSCRIBE_ENABLE_DEDUPE", True),
        dedupe_min_words=_env_int("TRANSCRIBE_DEDUPE_MIN_WORDS", 4),
        dedupe_max_words=_env_int("TRANSCRIBE_DEDUPE_MAX_WORDS", 14),
        dedupe_max_gap_s=_env_float("TRANSCRIBE_DEDUPE_MAX_GAP_S", 0.8),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 200 * 1024 * 1024),
        max_concurrent_transcribes=_env_int("MAX_CONCURRENT_TRANSCRIBES", 1),
    )


settings = load_settings()
app = FastAPI(title="Rust-backed Transcriber", version="1.0")

_model: Any = None
_pipeline: Any = None
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


def _lazy_load_model_and_pipeline() -> Tuple[Any, Any]:
    global _model
    global _pipeline
    if _model is not None and _pipeline is not None:
        return _model, _pipeline

    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import BatchedInferencePipeline

    _model = WhisperModel(
        settings.model_id,
        device=settings.device,
        compute_type=settings.compute_type,
    )
    _pipeline = BatchedInferencePipeline(_model)
    return _model, _pipeline


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
    _lazy_load_model_and_pipeline()
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
        "model_id": settings.model_id,
        "device": settings.device,
        "compute_type": settings.compute_type,
        "transcribe_backend": settings.transcribe_backend,
        "batch_size": int(settings.batch_size),
        "batch_size_cap": int(settings.batch_size_cap),
        "max_concurrent_transcribes": int(settings.max_concurrent_transcribes),
        "repair_enabled": bool(settings.enable_repair),
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


def _transcribe_file(
    *,
    audio_path: str,
    language: str,
    beam_size: int,
    patience: float,
    word_timestamps: bool,
    log_prob_threshold: Optional[float],
    no_speech_threshold: Optional[float],
    batch_size: int,
) -> Tuple[Dict[str, Any], str, int]:
    global _batch_size_cap
    _lazy_load_model_and_pipeline()
    assert _model is not None
    assert _pipeline is not None

    vad_params: Dict[str, Any] = dict(
        threshold=0.15,
        min_speech_duration_ms=150,
        min_silence_duration_ms=1400,
        speech_pad_ms=700,
        # Note: BatchedInferencePipeline overrides max_speech_duration_s to `chunk_length` (default Whisper=30s).
        max_speech_duration_s=30,
    )

    base_kwargs: Dict[str, Any] = dict(
        language=str(language),
        temperature=0.0,
        beam_size=int(beam_size),
        patience=float(patience),
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=vad_params,
        word_timestamps=bool(word_timestamps),
    )
    if log_prob_threshold is not None:
        base_kwargs["log_prob_threshold"] = float(log_prob_threshold)
    if no_speech_threshold is not None:
        base_kwargs["no_speech_threshold"] = float(no_speech_threshold)

    bs_req = max(1, int(batch_size))
    with _batch_size_lock:
        bs = min(bs_req, max(1, int(_batch_size_cap)))

    def to_json(segments_iter: Any) -> Tuple[list, list]:
        segments_out = []
        text_parts = []
        for seg in segments_iter:
            seg_text = (seg.text or "").strip()
            start_s = float(seg.start)
            end_s = float(seg.end)
            item: Dict[str, Any] = {
                "start_s": start_s,
                "end_s": end_s,
                "start_ts": format_timestamp(start_s),
                "end_ts": format_timestamp(end_s),
                "text": seg_text,
            }

            for k in ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature"):
                v = getattr(seg, k, None)
                if v is not None:
                    try:
                        item[k] = float(v)
                    except Exception:
                        pass

            if word_timestamps and getattr(seg, "words", None):
                words_out = []
                for w in seg.words:
                    word_text = (w.word or "").strip()
                    if not word_text:
                        continue
                    ws = float(w.start)
                    we = float(w.end)
                    words_out.append(
                        {
                            "start_s": ws,
                            "end_s": we,
                            "start_ts": format_timestamp(ws),
                            "end_ts": format_timestamp(we),
                            "word": word_text,
                            "probability": float(getattr(w, "probability", math.nan)),
                        }
                    )
                item["words"] = words_out

            segments_out.append(item)
            if seg_text:
                text_parts.append(seg_text)
        return segments_out, text_parts

    def ffmpeg_slice(src: str, start_s: float, end_s: float, dst: str) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{float(start_s):.3f}",
            "-to",
            f"{float(end_s):.3f}",
            "-i",
            src,
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            dst,
        ]
        subprocess.check_call(cmd)

    def transcribe_standard(path: str, *, vad_filter_override: Optional[bool] = None) -> Tuple[Dict[str, Any], str]:
        assert _model is not None
        kwargs = dict(base_kwargs)
        if vad_filter_override is not None:
            kwargs["vad_filter"] = bool(vad_filter_override)
        segments_iter, info = _model.transcribe(path, **kwargs)
        segs, parts = to_json(segments_iter)
        transcription: Dict[str, Any] = {
            "language": str(info.language),
            "language_probability": float(info.language_probability),
            "duration_s": float(info.duration),
            "segments": segs,
        }
        return transcription, " ".join(parts).strip()

    _ARABIC_DIACRITIC_RANGES = [
        (0x0610, 0x061A),
        (0x064B, 0x065F),
        (0x06D6, 0x06DC),
        (0x06DD, 0x06E4),
        (0x06E7, 0x06E8),
        (0x06EA, 0x06ED),
    ]
    _ARABIC_DIACRITIC_SINGLE = {0x0670}
    _ARABIC_TATWEEL = 0x0640
    _ARABIC_TRANSLATE = {
        0x0623: 0x0627,  # أ -> ا
        0x0625: 0x0627,  # إ -> ا
        0x0622: 0x0627,  # آ -> ا
        0x0671: 0x0627,  # ٱ -> ا
        0x0624: 0x0648,  # ؤ -> و
        0x0626: 0x064A,  # ئ -> ي
        0x0649: 0x064A,  # ى -> ي
        0x0629: 0x0647,  # ة -> ه
    }

    def _is_arabic_diacritic(cp: int) -> bool:
        if cp in _ARABIC_DIACRITIC_SINGLE:
            return True
        for a, b in _ARABIC_DIACRITIC_RANGES:
            if a <= cp <= b:
                return True
        return False

    def _normalize_word_for_dedupe(w: str) -> str:
        # Small, deterministic normalizer (mirrors quran_text.normalize_token contract).
        w = str(w or "").strip()
        if not w:
            return ""
        out = []
        for ch in w:
            cp = ord(ch)
            if cp == _ARABIC_TATWEEL:
                continue
            if _is_arabic_diacritic(cp):
                continue
            cp = _ARABIC_TRANSLATE.get(cp, cp)
            out.append(chr(cp))
        s = "".join(out)
        # Keep only word-ish chars and remove spaces.
        s2 = []
        for ch in s:
            if ch.isspace():
                continue
            if ch.isalnum() or ch == "_" or ("\u0660" <= ch <= "\u0669"):
                s2.append(ch)
                continue
            # Drop punctuation/symbols.
        return "".join(s2)

    def dedupe_overlapping_prefixes(
        segs: list,
        *,
        min_words: int,
        max_words: int,
        max_gap_s: float,
    ) -> Tuple[list, Dict[str, Any]]:
        # Removes duplicated prefixes when a chunk boundary repeats the last few words of the previous segment.
        out = []
        meta: Dict[str, Any] = {"removed_total_words": 0, "pairs": 0}

        ss = [s for s in segs if isinstance(s, dict)]
        ss.sort(key=lambda x: float(x.get("start_s") or 0.0))
        for seg in ss:
            if not out:
                out.append(seg)
                continue
            prev = out[-1]
            try:
                gap = float(seg.get("start_s") or 0.0) - float(prev.get("end_s") or 0.0)
            except Exception:
                gap = 999.0
            if gap > float(max_gap_s):
                out.append(seg)
                continue

            prev_words = prev.get("words")
            cur_words = seg.get("words")
            if not isinstance(prev_words, list) or not isinstance(cur_words, list) or not prev_words or not cur_words:
                out.append(seg)
                continue

            prev_norm = [_normalize_word_for_dedupe(w.get("word") or "") for w in prev_words if isinstance(w, dict)]
            cur_norm = [_normalize_word_for_dedupe(w.get("word") or "") for w in cur_words if isinstance(w, dict)]
            if not prev_norm or not cur_norm:
                out.append(seg)
                continue

            max_k = min(int(max_words), len(prev_norm), len(cur_norm))
            best_k = 0
            for k in range(max_k, int(min_words) - 1, -1):
                if prev_norm[-k:] == cur_norm[:k]:
                    best_k = k
                    break
            if best_k <= 0:
                out.append(seg)
                continue

            trimmed = cur_words[best_k:]
            if not trimmed:
                meta["removed_total_words"] += best_k
                meta["pairs"] += 1
                continue

            seg["words"] = trimmed
            seg["text"] = " ".join([str(w.get("word") or "").strip() for w in trimmed if isinstance(w, dict)]).strip()
            try:
                ws = float(trimmed[0].get("start_s") or float(seg.get("start_s") or 0.0))
                seg["start_s"] = ws
                seg["start_ts"] = format_timestamp(ws)
            except Exception:
                pass

            meta["removed_total_words"] += best_k
            meta["pairs"] += 1
            out.append(seg)

        # Keep non-dict entries (shouldn't exist, but be safe).
        for s in segs:
            if not isinstance(s, dict):
                out.append(s)
        out.sort(key=lambda x: float(x.get("start_s") or 0.0) if isinstance(x, dict) else 0.0)
        return out, meta

    def pick_repair_windows(
        segs: list, *, duration_s: float, gap_trigger_s: float, low_words_duration_s: float, low_words_max_words: int
    ) -> list[tuple[float, float, str]]:
        # Returns list of (core_start_s, core_end_s, kind).
        if not segs:
            return []

        # Collect candidate windows with a severity score, then repair only the top-K worst windows.
        # This avoids missing a late large gap just because earlier smaller windows filled the budget.
        items: list[tuple[float, float, str, float]] = []
        ss = sorted([s for s in segs if isinstance(s, dict)], key=lambda x: float(x.get("start_s") or 0.0))
        prev_end = None
        for s in ss:
            st = float(s.get("start_s") or 0.0)
            en = float(s.get("end_s") or st)
            if prev_end is not None:
                gap = st - prev_end
                if gap >= float(gap_trigger_s):
                    a = max(0.0, float(prev_end))
                    b = min(float(duration_s), float(st))
                    score = max(0.0, b - a)
                    items.append((a, b, "gap", score))
            prev_end = en

        for s in ss:
            st = float(s.get("start_s") or 0.0)
            en = float(s.get("end_s") or st)
            dur = max(0.0, en - st)
            if dur < float(low_words_duration_s):
                continue
            words = s.get("words")
            wc = len(words) if isinstance(words, list) else len(str(s.get("text") or "").strip().split())
            if wc <= int(low_words_max_words):
                a = max(0.0, st)
                b = min(float(duration_s), en)
                score = max(0.0, b - a) * (1.0 + float(int(low_words_max_words) - int(wc)))
                items.append((a, b, "low_words", score))

        # Merge overlaps (conservative) and cap count.
        items.sort(key=lambda t: (t[0], t[1]))
        merged: list[tuple[float, float, str, float]] = []
        for a, b, kind, score in items:
            if b <= a:
                continue
            if not merged:
                merged.append((a, b, kind, float(score)))
                continue
            pa, pb, pk, ps = merged[-1]
            if a <= pb + 0.25:
                merged[-1] = (pa, max(pb, b), (pk if pk == kind else "mixed"), max(ps, float(score)))
            else:
                merged.append((a, b, kind, float(score)))

        k = max(0, int(settings.repair_max_windows))
        if k <= 0 or not merged:
            return []

        # Select by severity, then process in time order for stable behavior.
        merged.sort(key=lambda t: (-t[3], t[0], t[1]))
        chosen = merged[:k]
        chosen.sort(key=lambda t: (t[0], t[1]))
        return [(a, b, kind) for a, b, kind, _score in chosen]

    def repair_segments(
        segs: list, *, duration_s: float, context_s: float, gap_trigger_s: float, low_words_duration_s: float, low_words_max_words: int
    ) -> Tuple[list, Dict[str, Any]]:
        windows = pick_repair_windows(
            segs,
            duration_s=float(duration_s),
            gap_trigger_s=float(gap_trigger_s),
            low_words_duration_s=float(low_words_duration_s),
            low_words_max_words=int(low_words_max_words),
        )
        if not windows:
            return segs, {"windows": [], "applied": 0}

        out = [s for s in segs if isinstance(s, dict)]
        meta = {"windows": [], "applied": 0}

        for core_start, core_end, kind in windows:
            if core_end - core_start < 0.5:
                continue

            pad = float(max(0.0, context_s))
            pad_start = max(0.0, float(core_start) - pad)
            pad_end = min(float(duration_s), float(core_end) + pad)
            if pad_end - pad_start < 0.5:
                continue

            with tempfile.TemporaryDirectory(prefix="repair_") as td:
                clip_path = os.path.join(td, "clip.wav")
                ffmpeg_slice(audio_path, pad_start, pad_end, clip_path)

                # For repair clips, disable VAD so we don't accidentally drop the very region we're trying to recover.
                clip_tx, _ = transcribe_standard(clip_path, vad_filter_override=False)
                clip_segs = clip_tx.get("segments") if isinstance(clip_tx, dict) else None
                if not isinstance(clip_segs, list) or not clip_segs:
                    meta["windows"].append(
                        {
                            "kind": kind,
                            "core_start_s": core_start,
                            "core_end_s": core_end,
                            "pad_start_s": pad_start,
                            "pad_end_s": pad_end,
                            "status": "no_segments",
                        }
                    )
                    continue

                repaired: list[Dict[str, Any]] = []
                for s in clip_segs:
                    if not isinstance(s, dict):
                        continue
                    st = float(s.get("start_s") or 0.0) + pad_start
                    en = float(s.get("end_s") or st) + pad_start
                    if en <= core_start or st >= core_end:
                        continue
                    st2 = max(float(core_start), st)
                    en2 = min(float(core_end), en)
                    if en2 <= st2:
                        continue
                    item = dict(s)
                    item["start_s"] = st2
                    item["end_s"] = en2
                    item["start_ts"] = format_timestamp(st2)
                    item["end_ts"] = format_timestamp(en2)
                    words = item.get("words")
                    if isinstance(words, list) and words:
                        kept = []
                        for w in words:
                            if not isinstance(w, dict):
                                continue
                            ws = float(w.get("start_s") or 0.0) + pad_start
                            we = float(w.get("end_s") or ws) + pad_start
                            if we <= core_start or ws >= core_end:
                                continue
                            ws2 = max(float(core_start), ws)
                            we2 = min(float(core_end), we)
                            if we2 <= ws2:
                                continue
                            ww = dict(w)
                            ww["start_s"] = ws2
                            ww["end_s"] = we2
                            ww["start_ts"] = format_timestamp(ws2)
                            ww["end_ts"] = format_timestamp(we2)
                            kept.append(ww)
                        item["words"] = kept
                    repaired.append(item)

                if not repaired:
                    meta["windows"].append(
                        {
                            "kind": kind,
                            "core_start_s": core_start,
                            "core_end_s": core_end,
                            "pad_start_s": pad_start,
                            "pad_end_s": pad_end,
                            "status": "no_overlap_after_clip",
                        }
                    )
                    continue

                # Remove any original segments intersecting the core window, then insert repaired.
                kept = []
                for s in out:
                    st = float(s.get("start_s") or 0.0)
                    en = float(s.get("end_s") or st)
                    if en <= core_start or st >= core_end:
                        kept.append(s)
                out = kept + repaired
                out.sort(key=lambda x: float(x.get("start_s") or 0.0))
                meta["applied"] += 1
                meta["windows"].append(
                    {
                        "kind": kind,
                        "core_start_s": core_start,
                        "core_end_s": core_end,
                        "pad_start_s": pad_start,
                        "pad_end_s": pad_end,
                        "status": "applied",
                        "segments_added": len(repaired),
                    }
                )

        return out, meta

    while True:
        kwargs = dict(base_kwargs)
        kwargs["batch_size"] = int(bs)
        try:
            backend = str(settings.transcribe_backend or "batched").strip().lower()
            if backend in {"standard", "whisper"}:
                transcription, full_text = transcribe_standard(audio_path)
                return transcription, full_text, int(bs)

            segments_iter, info = _pipeline.transcribe(audio_path, **kwargs)
            segments_out, text_parts = to_json(segments_iter)

            duration_s = float(info.duration)
            repair_meta: Dict[str, Any] = {"windows": [], "applied": 0}
            dedupe_meta: Dict[str, Any] = {"pairs": 0, "removed_total_words": 0}
            if settings.enable_dedupe:
                segments_out, dedupe_meta = dedupe_overlapping_prefixes(
                    segments_out,
                    min_words=int(settings.dedupe_min_words),
                    max_words=int(settings.dedupe_max_words),
                    max_gap_s=float(settings.dedupe_max_gap_s),
                )

            # Repair after dedupe: overlap trimming can expose true gaps that need to be re-transcribed.
            if settings.enable_repair:
                segments_out, repair_meta = repair_segments(
                    segments_out,
                    duration_s=duration_s,
                    context_s=float(settings.repair_context_s),
                    gap_trigger_s=float(settings.repair_gap_trigger_s),
                    low_words_duration_s=float(settings.repair_low_words_duration_s),
                    low_words_max_words=int(settings.repair_low_words_max_words),
                )

            # A second dedupe pass cleans up any overlaps introduced by the repair window insertion.
            if settings.enable_dedupe:
                segments_out, dedupe_meta2 = dedupe_overlapping_prefixes(
                    segments_out,
                    min_words=int(settings.dedupe_min_words),
                    max_words=int(settings.dedupe_max_words),
                    max_gap_s=float(settings.dedupe_max_gap_s),
                )
                dedupe_meta = {
                    "pairs": int(dedupe_meta.get("pairs") or 0) + int(dedupe_meta2.get("pairs") or 0),
                    "removed_total_words": int(dedupe_meta.get("removed_total_words") or 0)
                    + int(dedupe_meta2.get("removed_total_words") or 0),
                }

            transcription = {
                "language": str(info.language),
                "language_probability": float(info.language_probability),
                "duration_s": duration_s,
                "segments": segments_out,
            }
            if settings.enable_repair:
                transcription["repair"] = repair_meta
            if settings.enable_dedupe:
                transcription["dedupe"] = dedupe_meta

            full_text = " ".join([str(s.get("text") or "").strip() for s in segments_out if isinstance(s, dict)]).strip()
            return transcription, full_text, int(bs)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda failed" not in msg:
                raise
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            with _batch_size_lock:
                _batch_size_cap = min(max(1, int(_batch_size_cap)), bs)
            try:
                import gc

                gc.collect()
            except Exception:
                pass


@app.post("/v1/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = None,
    beam_size: Optional[int] = None,
    patience: Optional[float] = None,
    word_timestamps: Optional[bool] = None,
    log_prob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
    batch_size: Optional[int] = None,
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

        lang = str(language) if language is not None else settings.language
        b = int(beam_size) if beam_size is not None else int(settings.beam_size)
        p = float(patience) if patience is not None else float(settings.patience)
        wt = bool(word_timestamps) if word_timestamps is not None else bool(settings.word_timestamps)
        lpt = float(log_prob_threshold) if log_prob_threshold is not None else settings.log_prob_threshold
        nst = float(no_speech_threshold) if no_speech_threshold is not None else settings.no_speech_threshold
        bs = int(batch_size) if batch_size is not None else int(settings.batch_size)

        t0 = time.time()
        async with _sem:
            transcription, text, bs_used = await asyncio.to_thread(
                _transcribe_file,
                audio_path=path,
                language=lang,
                beam_size=b,
                patience=p,
                word_timestamps=wt,
                log_prob_threshold=lpt,
                no_speech_threshold=nst,
                batch_size=bs,
            )
        t_transcribe = time.time() - t0

        return JSONResponse(
            {
                "input": {"filename": filename, "content_type": file.content_type},
                "transcription": transcription,
                "text": text,
                "params": {
                    "model_id": settings.model_id,
                    "language": lang,
                    "beam_size": b,
                    "patience": p,
                    "word_timestamps": wt,
                    "log_prob_threshold": lpt,
                    "no_speech_threshold": nst,
                    "batch_size": bs_used,
                    "temperature": 0.0,
                },
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
