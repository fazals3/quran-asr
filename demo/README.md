# Quran ASR Demo Frontend

A standalone web UI for testing Quran recitation transcription and ayah alignment. Connects to the `quran-asr` API server's demo endpoints.

## Features

- **Upload** — Drag-and-drop or browse for an audio file (max 1 minute).
- **Record** — Record directly in the browser using your microphone (max 1 minute).
- **Live Stream** — Real-time streaming transcription via WebSocket as you recite.

## Prerequisites

The API server must have `DEMO_ENABLED=true` set. The demo endpoints are mounted at `/demo/*` and do not require an API key (they use IP-based rate limiting instead).

The frontend and API can run on completely different machines — configure the API URL in the UI or via environment variable.

## Development

```bash
cp .env.example .env
# Edit .env to point VITE_API_URL at your API server

npm install
npm run dev
```

Opens at http://localhost:3000.

## Production

```bash
npm run build
```

Static files are output to `dist/`. Serve them with any static file server (nginx, Caddy, Cloudflare Pages, etc).

Or use the Dockerfile:

```bash
docker build -t quran-asr-demo .
docker run -p 3000:3000 quran-asr-demo
```

## Docker Compose

Use `docker-compose.demo.yml` from the project root to run the full stack with demo mode enabled:

```bash
docker compose -f docker-compose.demo.yml up -d --build
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8001` | API server base URL (set at build time) |

The API URL can also be changed at runtime in the UI's configuration bar.
