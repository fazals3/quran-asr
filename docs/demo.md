# Demo Mode Tutorial

This guide walks through enabling and running the public demo interface for `quran-asr`. The demo gives anyone a browser-based way to try Quran recitation transcription without needing an API key — similar to a Hugging Face Spaces deployment.

## Overview

The demo consists of two parts:

1. **Backend** — A set of `/demo/*` API routes on the Rust server (no auth, rate-limited, CORS-enabled).
2. **Frontend** — A standalone React web app (`demo/`) that connects to the backend over HTTP/WebSocket.

The frontend and backend can run on completely different machines. For example, you might run the GPU inference stack on a cloud server and serve the frontend from a CDN or static host.

## Enabling Demo Mode

Demo mode is **disabled by default**. To enable it, set one environment variable:

```bash
DEMO_ENABLED=true
```

Add this to your `.env` file (or pass it directly to the container).

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_ENABLED` | `false` | Master switch for demo endpoints |
| `DEMO_ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated, or `*` for any) |
| `DEMO_MAX_AUDIO_DURATION_S` | `60` | Maximum audio duration in seconds |
| `DEMO_MAX_UPLOAD_BYTES` | `26214400` (25 MB) | Maximum upload file size |
| `DEMO_RATE_LIMIT_PER_MIN` | `10` | Max requests per IP per minute |

## Running with Docker Compose

The easiest way to run everything together:

```bash
docker compose -f docker-compose.demo.yml up -d --build
```

This starts:
- `transcriber` + `transcriber2` — GPU ASR workers
- `api` — Rust API with `DEMO_ENABLED=true`
- `demo-frontend` — Nginx serving the built React app on port 3000

Open http://localhost:3000 to use the demo.

## Running the Frontend Separately

If you want to develop the frontend or host it on a different machine:

```bash
cd demo
cp .env.example .env
# Edit .env — set VITE_API_URL to your API server address
npm install
npm run dev
```

The dev server starts at http://localhost:3000. You can also change the API URL at runtime using the configuration bar in the UI.

### Production Build

```bash
cd demo
npm run build
```

Output goes to `demo/dist/`. Deploy these static files with any web server:

```bash
# Example with a simple static server
npx serve dist
```

Or build and run the Docker image:

```bash
docker build -t quran-asr-demo ./demo
docker run -p 3000:3000 quran-asr-demo
```

## Using the Demo

The demo UI has three modes:

### 1. Upload Audio

Upload a pre-recorded audio file (mp3, m4a, wav, webm, etc.) of Quran recitation. The file must be 1 minute or shorter.

1. Click the drop zone or drag a file onto it.
2. Click **Transcribe**.
3. Wait for the result — you'll see the Arabic transcription text, detected surah/ayah range, alignment confidence, and timing breakdown.

### 2. Record

Record directly in the browser using your microphone.

1. Click **Start Recording**.
2. Recite for up to 60 seconds (recording auto-stops at the limit).
3. Click **Stop** when finished.
4. Click **Transcribe** to send the recording to the server.

The mic level meter shows real-time audio levels so you can verify your microphone is working.

> **Note:** Most browsers require HTTPS for microphone access (localhost is an exception). If you're accessing the demo over plain HTTP on a non-localhost domain, the browser will block mic access.

### 3. Live Stream

Real-time streaming transcription — results appear as you recite.

1. Click **Start Streaming**.
2. Grant microphone permission when prompted.
3. Begin reciting — the UI shows live ayah alignment updates as audio flows to the server.
4. The stream auto-stops after 60 seconds. Click **Stop** to end early.

The streaming mode uses a WebSocket connection and sends PCM16LE audio at 16 kHz. The server responds with `ayah_update` events containing the current alignment state.

## API Endpoints Reference

All demo endpoints are nested under `/demo` and require no authentication.

### Health Check

```
GET /demo/health
```

Returns `{"ok": true, "demo": true}`.

### Transcribe (Upload)

```
POST /demo/v1/transcribe?wait=true&wait_timeout_s=120
Content-Type: multipart/form-data

file: <audio file>
```

Returns the full transcription and alignment result when `wait=true`. The `wait_timeout_s` parameter controls how long the server holds the connection before returning a queued status.

### Job Status

```
GET /demo/v1/jobs/:job_id
```

Poll for job completion if you didn't use `wait=true`.

### Create Streaming Session

```
POST /demo/v1/sessions
Content-Type: application/json

{
  "sample_rate_hz": 16000,
  "channels": 1,
  "window_s": 12,
  "hop_s": 2,
  "buffer_s": 60
}
```

Returns `session_id` and `ws_path` for the WebSocket connection.

### Streaming WebSocket

```
GET /demo/v1/sessions/:session_id/stream
Upgrade: websocket
```

Send binary PCM16LE mono 16 kHz frames. Receive JSON text messages with alignment results.

### Stop Session

```
POST /demo/v1/sessions/:session_id/stop
```

## Security Considerations

- Demo endpoints have **no API key requirement** — they rely on rate limiting and upload size caps.
- Rate limiting is per-IP, in-memory (resets on server restart).
- If you're behind a reverse proxy (Cloudflare, nginx), make sure `X-Forwarded-For` is properly handled so rate limiting applies to the real client IP.
- Set `DEMO_ALLOWED_ORIGINS` to your frontend's domain in production instead of `*`.
- The demo shares the same job queue and GPU resources as the main API. If you need isolation, run a separate stack for the demo.

## Troubleshooting

**"rate limit exceeded"** — Wait 60 seconds or increase `DEMO_RATE_LIMIT_PER_MIN`.

**"queue full, try again later"** — The transcription queue is saturated. Either wait or increase `MAX_QUEUE_LENGTH` / `WORKER_CONCURRENCY`.

**"streaming not available on this instance"** — The server needs v2 alignment assets loaded (`AYAH_ALIGN_MODE=v2` and valid `AYAH_ALIGN_V2_ASSETS_DIR`).

**Microphone not working** — Ensure you're on HTTPS or localhost. Check browser permissions.

**CORS errors** — Verify `DEMO_ALLOWED_ORIGINS` includes your frontend's origin, or set it to `*`.

**WebSocket connection fails** — If the frontend and API are on different domains, ensure the API allows WebSocket upgrades through any reverse proxy in between.
