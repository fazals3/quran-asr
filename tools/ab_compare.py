#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def _post_transcribe_job(*, base_url: str, api_key: str, audio_path: Path, wait_timeout_s: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/jobs/transcribe"
    with audio_path.open("rb") as f:
        resp = requests.post(
            url,
            params={"wait": "true", "wait_timeout_s": str(int(wait_timeout_s))},
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio_path.name, f, "audio/mpeg")},
            timeout=wait_timeout_s + 60,
        )
    resp.raise_for_status()
    return resp.json()


def _summarize(resp: Dict[str, Any]) -> Dict[str, Any]:
    result = resp.get("result") or {}
    timing = result.get("timing") or {}
    align = result.get("ayah_alignment") if isinstance(result.get("ayah_alignment"), dict) else {}
    start = (align.get("start") or {}).get("ayah_key") if isinstance(align.get("start"), dict) else None
    end = (align.get("end") or {}).get("ayah_key") if isinstance(align.get("end"), dict) else None
    conf = align.get("confidence")
    spans = result.get("ayah_alignment_spans")
    jumps = result.get("ayah_alignment_jumps")
    guess = result.get("guess") if isinstance(result.get("guess"), dict) else {}
    return {
        "status": resp.get("status"),
        "guess_surah": guess.get("predicted_surah_id"),
        "start": start,
        "end": end,
        "conf": conf,
        "spans": len(spans) if isinstance(spans, list) else None,
        "active_span": result.get("ayah_alignment_active_span_index"),
        "jumps": len(jumps) if isinstance(jumps, list) else None,
        "transcribe_s": timing.get("transcribe_s"),
        "guess_s": timing.get("guess_s"),
        "align_s": timing.get("align_s"),
        "total_s": timing.get("total_s"),
    }


def _ffmpeg(*args: str) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *args]
    subprocess.check_call(cmd)


def _make_multispan_fixture(tmp: Path, recordings_dir: Path) -> Path:
    # 20s from 078 + 20s from 079 + 20s from 105
    a = tmp / "078_20.mp3"
    b = tmp / "079_20.mp3"
    c = tmp / "105_20.mp3"
    out = tmp / "multispan_078_079_105.mp3"
    _ffmpeg("-y", "-i", str(recordings_dir / "078.mp3"), "-t", "20", "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", str(a))
    _ffmpeg("-y", "-i", str(recordings_dir / "079.mp3"), "-t", "20", "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", str(b))
    _ffmpeg("-y", "-i", str(recordings_dir / "105.mp3"), "-t", "20", "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", str(c))
    concat_list = tmp / "concat_list.txt"
    concat_list.write_text(f"file '{a}'\nfile '{b}'\nfile '{c}'\n", encoding="utf-8")
    _ffmpeg("-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(out))
    return out


def _find_ayah_start(result: Dict[str, Any], ayah_key: str) -> Optional[float]:
    align = result.get("ayah_alignment")
    if not isinstance(align, dict):
        return None
    segs = align.get("segments")
    if not isinstance(segs, list):
        return None
    for s in segs:
        if not isinstance(s, dict):
            continue
        if s.get("ayah_key") == ayah_key:
            v = s.get("start_s")
            try:
                return float(v) if v is not None else None
            except Exception:
                return None
    return None


def _make_skip_within_surah_fixture(
    *,
    tmp: Path,
    recordings_dir: Path,
    api_url_for_timestamps: str,
    api_key: str,
) -> Path:
    # Build a "Shams ayah1 then ayah3" clip by cutting surah 091 based on inferred timestamps.
    src = recordings_dir / "091.mp3"
    inf = _post_transcribe_job(base_url=api_url_for_timestamps, api_key=api_key, audio_path=src, wait_timeout_s=600)
    res = inf.get("result") or {}

    t1 = _find_ayah_start(res, "91:1")
    t2 = _find_ayah_start(res, "91:2")
    t3 = _find_ayah_start(res, "91:3")
    t4 = _find_ayah_start(res, "91:4")
    if t1 is None:
        t1 = 0.0
    if None in (t2, t3, t4):
        raise RuntimeError(f"could not extract ayah starts for 91:2-4 (got {t1},{t2},{t3},{t4})")

    clip1 = tmp / "shams_1.mp3"
    clip3 = tmp / "shams_3.mp3"
    out = tmp / "shams_ayah1_then_ayah3.mp3"

    _ffmpeg("-y", "-i", str(src), "-ss", f"{t1:.3f}", "-to", f"{t2:.3f}", "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", str(clip1))
    _ffmpeg("-y", "-i", str(src), "-ss", f"{t3:.3f}", "-to", f"{t4:.3f}", "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", str(clip3))
    concat_list = tmp / "concat_list_skip.txt"
    concat_list.write_text(f"file '{clip1}'\nfile '{clip3}'\n", encoding="utf-8")
    _ffmpeg("-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(out))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--old", default=os.environ.get("OLD_URL", "http://100.124.164.68:8000"))
    p.add_argument("--new", default=os.environ.get("NEW_URL", "http://100.124.164.68:8001"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    p.add_argument("--recordings-dir", default=str(Path(__file__).resolve().parents[2] / "recordings"))
    p.add_argument("--include-long", action="store_true")
    p.add_argument("--assert", dest="do_assert", action="store_true")
    args = p.parse_args()

    if not args.api_key:
        print("missing --api-key (or API_KEY env var)", file=sys.stderr)
        return 2

    recordings_dir = Path(args.recordings_dir).resolve()
    if not recordings_dir.is_dir():
        print(f"recordings dir not found: {recordings_dir}", file=sys.stderr)
        return 2

    tests: list[Path] = [
        recordings_dir / "112.mp3",
    ]
    if args.include_long:
        tests.append(recordings_dir / "007.mp3")

    with tempfile.TemporaryDirectory(prefix="ab_infer_") as td:
        tmp = Path(td)
        multispan = _make_multispan_fixture(tmp, recordings_dir)
        skip = (
            _make_skip_within_surah_fixture(
                tmp=tmp,
                recordings_dir=recordings_dir,
                api_url_for_timestamps=args.new,
                api_key=args.api_key,
            )
        )
        tests.append(multispan)
        tests.append(skip)

        rows = []
        for f in tests:
            if not f.exists():
                continue
            print(f"\n=== {f.name} ===")
            t0 = time.time()
            old = _post_transcribe_job(base_url=args.old, api_key=args.api_key, audio_path=f, wait_timeout_s=10800)
            new = _post_transcribe_job(base_url=args.new, api_key=args.api_key, audio_path=f, wait_timeout_s=10800)
            wall_s = time.time() - t0
            so = _summarize(old)
            sn = _summarize(new)
            rows.append({"file": f.name, "old": so, "new": sn, "wall_s": wall_s})
            print("old:", json.dumps(so, ensure_ascii=False))
            print("new:", json.dumps(sn, ensure_ascii=False))

            if args.do_assert:
                if f.name == "112.mp3":
                    assert sn.get("start", "").startswith("112:"), sn
                    assert sn.get("end", "").startswith("112:"), sn
                if f.name == multispan.name:
                    assert (sn.get("spans") or 0) >= 2, sn
                    assert str(sn.get("start", "")).startswith("105:"), sn
                    assert sn.get("active_span") in (1, 2, 3), sn
                if f.name == skip.name:
                    assert sn.get("start") == "91:1", sn
                    assert sn.get("end") == "91:3", sn

        print("\n=== summary ===")
        print(json.dumps(rows, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
