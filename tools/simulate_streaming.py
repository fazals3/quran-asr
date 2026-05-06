#!/usr/bin/env python3

import argparse
import asyncio
import datetime as dt
import json
import os
import signal
import subprocess
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


async def recv_events(ws: websockets.ClientConnection, out_path: Path, print_every: int) -> dict[str, Any]:
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

                    if print_every > 0 and (n % print_every) == 0:
                        align = evt.get("ayah_alignment") or {}
                        start = align.get("start")
                        end = align.get("end")
                        conf = align.get("confidence")
                        print(f"[evt {n}] at_s={evt.get('at_s')} range={start}..{end} conf={conf}")
        except Exception as e:
            error = str(e)

    return {
        "events_total": n,
        "t_first_update": t_first_update,
        "timings": timings,
        "error": error,
    }


async def stream_file(
    ws: websockets.ClientConnection,
    audio_path: Path,
    *,
    speed: float,
    frame_ms: int,
    start_s: float | None,
    max_s: float | None,
) -> dict[str, Any]:
    if speed <= 0:
        raise ValueError("speed must be > 0")

    bytes_per_s = 16_000 * 1 * 2
    frame_bytes = max(2, int(bytes_per_s * (frame_ms / 1000.0)))
    frame_bytes -= frame_bytes % 2
    frame_s = frame_bytes / bytes_per_s

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
    cmd += [
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)
    assert proc.stdout is not None

    t_first_send: float | None = None
    bytes_sent = 0
    frames_sent = 0
    next_deadline = time.monotonic()

    try:
        while True:
            try:
                chunk = await proc.stdout.readexactly(frame_bytes)
            except asyncio.IncompleteReadError as e:
                chunk = e.partial

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
    finally:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass

    return {
        "t_first_send": t_first_send,
        "bytes_sent": bytes_sent,
        "frames_sent": frames_sent,
        "approx_audio_s": bytes_sent / bytes_per_s,
        "frame_bytes": frame_bytes,
        "frame_ms": frame_ms,
        "speed": speed,
    }


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
                "audio_path": str(Path(args.audio).resolve()),
                "cfg": cfg,
                "session": session,
                "speed": args.speed,
                "frame_ms": args.frame_ms,
                "start_s": args.start_s,
                "max_s": args.max_s,
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
        recv_task = asyncio.create_task(recv_events(ws, events_path, args.print_every))
        send_task = asyncio.create_task(
            stream_file(
                ws,
                Path(args.audio),
                speed=args.speed,
                frame_ms=args.frame_ms,
                start_s=args.start_s,
                max_s=args.max_s,
            )
        )
        stop_task = asyncio.create_task(stop_flag.wait())

        send_stats = None
        recv_stats = None
        try:
            done, _ = await asyncio.wait({send_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            if stop_task in done and stop_flag.is_set():
                print("Stop requested; stopping session…")
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
            else:
                send_stats = await send_task

            try:
                await ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass

            # Give the server a moment to flush any last events.
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

    summary = {
        "session_id": session_id,
        "send": send_stats,
        "recv": recv_stats,
    }

    if send_stats and recv_stats:
        t_first_send = send_stats.get("t_first_send")
        t_first_update = recv_stats.get("t_first_update")
        if t_first_send and t_first_update:
            summary["ttfa_s"] = float(t_first_update - t_first_send)

        timings = recv_stats.get("timings") or []
        if timings:
            for k in ("transcribe_s", "align_s", "loop_s"):
                vals = [t.get(k) for t in timings if k in t]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if vals:
                    summary[f"avg_{k}"] = sum(vals) / len(vals)
                    summary[f"p90_{k}"] = sorted(vals)[int(0.9 * (len(vals) - 1))]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {events_path} and {out_dir/'summary.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Simulate live streaming into the quran-asr WebSocket API.")
    p.add_argument("--audio", required=True, help="Path to an audio file (mp3/m4a/wav).")
    p.add_argument("--http-base", default="http://100.124.164.68:8002", help="Dev inference HTTP base URL.")
    p.add_argument("--ws-base", default="ws://100.124.164.68:8002", help="Dev inference WS base URL.")
    p.add_argument("--api-key", default=None, help="Bearer token (or set API_KEY / use --env-file).")
    p.add_argument("--env-file", default=str(Path(__file__).resolve().parents[1] / ".env.dev"))
    p.add_argument("--out-dir", default=None, help="Output directory (default: outputs/streaming_sim_<ts>).")
    p.add_argument("--speed", type=float, default=1.0, help="Speed factor: 1.0=real-time, 4.0=faster.")
    p.add_argument("--frame-ms", type=int, default=20, help="PCM frame size in milliseconds.")
    p.add_argument("--start-s", type=float, default=None, help="Start offset into the file (seconds).")
    p.add_argument("--max-s", type=float, default=None, help="Max duration to stream (seconds).")
    p.add_argument("--print-every", type=int, default=10, help="Print every N events (0=quiet).")
    p.add_argument("--window-s", type=float, default=None, help="Override session window_s.")
    p.add_argument("--hop-s", type=float, default=None, help="Override session hop_s.")
    p.add_argument("--buffer-s", type=float, default=None, help="Override session buffer_s.")
    p.add_argument("--min-process-s", type=float, default=None, help="Override session min_process_s.")

    args = p.parse_args()
    if args.out_dir is None:
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(Path("outputs") / f"streaming_sim_{ts}")

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
