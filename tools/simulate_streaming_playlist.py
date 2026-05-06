#!/usr/bin/env python3

import argparse
import asyncio
import datetime as dt
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import requests
import websockets


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def create_session(http_base: str, api_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    url = http_base.rstrip("/") + "/v1/sessions"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.post(url, headers=headers, json=cfg, timeout=30)
    resp.raise_for_status()
    return resp.json()


def stop_session(http_base: str, api_key: str, session_id: str) -> None:
    url = http_base.rstrip("/") + f"/v1/sessions/{session_id}/stop"
    headers = {"Authorization": f"Bearer {api_key}"}
    requests.post(url, headers=headers, timeout=10).raise_for_status()


async def recv_events(ws: websockets.ClientConnection, out_path: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_first_update: float | None = None
    n = 0
    timings: list[dict[str, float]] = []
    error: str | None = None

    with out_path.open("w", encoding="utf-8") as f:
        try:
            async for msg in ws:
                now = time.monotonic()
                if isinstance(msg, bytes):
                    continue
                f.write(msg)
                f.write("\n")
                n += 1

                try:
                    evt = json.loads(msg)
                except Exception:
                    continue

                if evt.get("type") == "ayah_update":
                    if t_first_update is None:
                        t_first_update = now
                    timing = evt.get("timing")
                    if isinstance(timing, dict):
                        t = {}
                        for k in ("transcribe_s", "align_s", "loop_s"):
                            v = timing.get(k)
                            if isinstance(v, (int, float)):
                                t[k] = float(v)
                        if t:
                            timings.append(t)
        except Exception as e:
            error = str(e)

    return {
        "events_total": n,
        "t_first_update": t_first_update,
        "timings": timings,
        "error": error,
    }


async def _send_pcm_frames(
    ws: websockets.ClientConnection,
    *,
    pcm_iter: Any,
    speed: float,
    frame_ms: int,
) -> dict[str, Any]:
    if speed <= 0:
        raise ValueError("speed must be > 0")

    bytes_per_s = 16_000 * 1 * 2
    frame_bytes = max(2, int(bytes_per_s * (frame_ms / 1000.0)))
    frame_bytes -= frame_bytes % 2
    frame_s = frame_bytes / bytes_per_s

    t_first_send: float | None = None
    bytes_sent = 0
    frames_sent = 0
    next_deadline = time.monotonic()

    async for chunk in pcm_iter(frame_bytes):
        if not chunk:
            break
        if t_first_send is None:
            t_first_send = time.monotonic()
        await ws.send(chunk)
        bytes_sent += len(chunk)
        frames_sent += 1

        next_deadline += frame_s / speed
        sleep_s = next_deadline - time.monotonic()
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

    return {
        "t_first_send": t_first_send,
        "bytes_sent": bytes_sent,
        "frames_sent": frames_sent,
        "approx_audio_s": bytes_sent / bytes_per_s,
        "frame_bytes": frame_bytes,
        "frame_ms": frame_ms,
        "speed": speed,
    }


async def stream_audio_ffmpeg(
    ws: websockets.ClientConnection,
    audio_path: Path,
    *,
    speed: float,
    frame_ms: int,
    start_s: float | None,
    max_s: float | None,
) -> dict[str, Any]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if start_s is not None and start_s > 0:
        cmd += ["-ss", str(start_s)]
    cmd += ["-i", str(audio_path)]
    if max_s is not None and max_s > 0:
        cmd += ["-t", str(max_s)]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"]

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)
    assert proc.stdout is not None

    async def _pcm_iter(frame_bytes: int):
        try:
            while True:
                try:
                    chunk = await proc.stdout.readexactly(frame_bytes)
                except asyncio.IncompleteReadError as e:
                    chunk = e.partial
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    stats = await _send_pcm_frames(ws, pcm_iter=_pcm_iter, speed=speed, frame_ms=frame_ms)
    stats["audio_path"] = str(audio_path)
    stats["start_s"] = start_s
    stats["max_s"] = max_s
    stats["kind"] = "audio"
    return stats


async def stream_silence(
    ws: websockets.ClientConnection,
    silence_s: float,
    *,
    speed: float,
    frame_ms: int,
) -> dict[str, Any]:
    bytes_per_s = 16_000 * 1 * 2
    if silence_s <= 0:
        return {"kind": "silence", "silence_s": silence_s, "bytes_sent": 0, "frames_sent": 0}
    total_bytes = int(bytes_per_s * silence_s)
    total_bytes -= total_bytes % 2
    silence_bytes = b"\x00\x00" * 1024

    async def _pcm_iter(frame_bytes: int):
        remaining = total_bytes
        while remaining > 0:
            n = min(frame_bytes, remaining)
            # Keep sample alignment.
            n -= n % 2
            if n <= 0:
                break
            if n <= len(silence_bytes):
                yield silence_bytes[:n]
            else:
                yield b"\x00" * n
            remaining -= n

    stats = await _send_pcm_frames(ws, pcm_iter=_pcm_iter, speed=speed, frame_ms=frame_ms)
    stats["kind"] = "silence"
    stats["silence_s"] = silence_s
    return stats


def load_playlist(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("playlist must be a JSON array")
    items: list[dict[str, Any]] = []
    for i, it in enumerate(obj):
        if not isinstance(it, dict):
            raise ValueError(f"playlist[{i}] must be an object")
        if "audio" in it:
            items.append(
                {
                    "kind": "audio",
                    "audio": str(it["audio"]),
                    "start_s": float(it.get("start_s")) if it.get("start_s") is not None else None,
                    "max_s": float(it.get("max_s")) if it.get("max_s") is not None else None,
                    "label": it.get("label"),
                }
            )
        elif "silence_s" in it:
            items.append(
                {
                    "kind": "silence",
                    "silence_s": float(it["silence_s"]),
                    "label": it.get("label"),
                }
            )
        else:
            raise ValueError(f"playlist[{i}] must contain either 'audio' or 'silence_s'")
    return items


async def run(args: argparse.Namespace) -> int:
    http_base = args.http_base.rstrip("/")
    ws_base = args.ws_base.rstrip("/")

    env = {}
    if args.env_file:
        env = load_env_file(Path(args.env_file))

    api_key = args.api_key or os.environ.get("API_KEY") or env.get("API_KEY", "")
    if not api_key:
        raise SystemExit("missing API key (pass --api-key or set API_KEY or --env-file)")

    cfg: dict[str, Any] = {}
    for k in ("window_s", "hop_s", "buffer_s", "min_process_s"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    playlist_path = Path(args.playlist)
    playlist = load_playlist(playlist_path)

    session = create_session(http_base, api_key, cfg)
    session_id = session["session_id"]
    ws_url = session.get("ws_url") or (ws_base + session["ws_path"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    events_path = out_dir / "events.jsonl"

    meta_path.write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
                "http_base": http_base,
                "ws_url": ws_url,
                "cfg": cfg,
                "session": session,
                "playlist_path": str(playlist_path.resolve()),
                "playlist": playlist,
                "speed": args.speed,
                "frame_ms": args.frame_ms,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    stop_flag = asyncio.Event()

    def _handle_sigint(*_: Any) -> None:
        stop_flag.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    headers = {"Authorization": f"Bearer {api_key}"}
    async with websockets.connect(
        ws_url,
        additional_headers=headers,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        recv_task = asyncio.create_task(recv_events(ws, events_path))

        t_first_send: float | None = None
        send_items: list[dict[str, Any]] = []
        try:
            for item in playlist:
                if stop_flag.is_set():
                    break
                if item["kind"] == "silence":
                    stats = await stream_silence(ws, item["silence_s"], speed=args.speed, frame_ms=args.frame_ms)
                else:
                    audio_path = Path(item["audio"])
                    stats = await stream_audio_ffmpeg(
                        ws,
                        audio_path,
                        speed=args.speed,
                        frame_ms=args.frame_ms,
                        start_s=item.get("start_s"),
                        max_s=item.get("max_s"),
                    )
                if t_first_send is None and stats.get("t_first_send") is not None:
                    t_first_send = float(stats["t_first_send"])
                send_items.append({"item": item, "stats": stats})

            try:
                await ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            await asyncio.sleep(0.25)
        finally:
            try:
                await ws.close()
            except Exception:
                pass

        try:
            recv_stats = await asyncio.wait_for(recv_task, timeout=10)
        except Exception:
            recv_task.cancel()
            recv_stats = None

    try:
        stop_session(http_base, api_key, session_id)
    except Exception:
        pass

    summary: dict[str, Any] = {
        "session_id": session_id,
        "send": {
            "t_first_send": t_first_send,
            "items": send_items,
        },
        "recv": recv_stats,
    }

    if recv_stats and t_first_send and recv_stats.get("t_first_update"):
        summary["ttfa_s"] = float(recv_stats["t_first_update"] - t_first_send)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {events_path} and {out_dir/'summary.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Simulate streaming a playlist of audio/silence into inference WS.")
    p.add_argument("--playlist", required=True, help="Path to a JSON playlist (audio/silence items).")
    p.add_argument("--http-base", default="http://100.124.164.68:8002", help="Dev inference HTTP base URL.")
    p.add_argument("--ws-base", default="ws://100.124.164.68:8002", help="Dev inference WS base URL.")
    p.add_argument("--api-key", default=None, help="Bearer token (or set API_KEY / use --env-file).")
    p.add_argument("--env-file", default=str(Path(__file__).resolve().parents[1] / ".env.dev"))
    p.add_argument("--out-dir", default=None, help="Output directory (default: outputs/streaming_playlist_sim_<ts>).")
    p.add_argument("--speed", type=float, default=1.0, help="Speed factor: 1.0=real-time, 4.0=faster.")
    p.add_argument("--frame-ms", type=int, default=20, help="PCM frame size in milliseconds.")
    p.add_argument("--window-s", type=float, default=None, help="Override session window_s.")
    p.add_argument("--hop-s", type=float, default=None, help="Override session hop_s.")
    p.add_argument("--buffer-s", type=float, default=None, help="Override session buffer_s.")
    p.add_argument("--min-process-s", type=float, default=None, help="Override session min_process_s.")

    args = p.parse_args()
    if args.out_dir is None:
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(Path("outputs") / f"streaming_playlist_sim_{ts}")

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

