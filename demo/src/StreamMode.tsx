import { useState, useRef, useCallback, useEffect } from "react";

interface Props {
  baseUrl: string;
  apiUrl: string;
}

const MAX_DURATION_S = 60;

interface StreamEvent {
  type: string;
  data: any;
  ts: number;
}

export default function StreamMode({ baseUrl, apiUrl }: Props) {
  const [status, setStatus] = useState<"idle" | "connecting" | "streaming" | "error">("idle");
  const [elapsed, setElapsed] = useState(0);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [micLevel, setMicLevel] = useState(0);
  const [latestAyah, setLatestAyah] = useState<any>(null);
  const [latestText, setLatestText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [bytesSent, setBytesSent] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);
  const timerRef = useRef<number>(0);
  const startTimeRef = useRef(0);
  const sessionIdRef = useRef<string | null>(null);
  const bytesSentRef = useRef(0);
  const eventsEndRef = useRef<HTMLDivElement>(null);

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      try { wsRef.current.close(1000, "stop"); } catch {}
      wsRef.current = null;
    }
    if (workletRef.current) {
      try { workletRef.current.disconnect(); } catch {}
      workletRef.current = null;
    }
    if (audioCtxRef.current) {
      try { audioCtxRef.current.close(); } catch {}
      audioCtxRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = 0;
    }
    analyserRef.current = null;
    setMicLevel(0);

    if (sessionIdRef.current) {
      fetch(`${baseUrl}/v1/sessions/${sessionIdRef.current}/stop`, {
        method: "POST",
      }).catch(() => {});
      sessionIdRef.current = null;
    }
  }, [baseUrl]);

  useEffect(() => {
    return () => { cleanup(); };
  }, [cleanup]);

  const buildWorkletUrl = () => {
    const code = `
class Pcm16Worklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.pos = 0;
    this.carry = new Float32Array(0);
    this.chunkSamples = 320;
    this.pending = [];
  }
  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input || input.length === 0) return true;
    const src = new Float32Array(this.carry.length + input.length);
    src.set(this.carry, 0);
    src.set(input, this.carry.length);
    const ratio = sampleRate / this.targetRate;
    const out = [];
    while (this.pos + 1 < src.length) {
      const i0 = Math.floor(this.pos);
      const i1 = i0 + 1;
      const frac = this.pos - i0;
      out.push(src[i0] + (src[i1] - src[i0]) * frac);
      this.pos += ratio;
    }
    const consumed = Math.max(0, Math.floor(this.pos) - 1);
    this.pos = Math.max(0, this.pos - consumed);
    this.carry = src.slice(consumed);
    for (let i = 0; i < out.length; i++) {
      const x = Math.max(-1, Math.min(1, out[i]));
      this.pending.push(x < 0 ? Math.round(x * 32768) : Math.round(x * 32767));
      if (this.pending.length >= this.chunkSamples) {
        const chunk = this.pending.splice(0, this.chunkSamples);
        const i16 = new Int16Array(chunk);
        this.port.postMessage(i16.buffer, [i16.buffer]);
      }
    }
    return true;
  }
}
registerProcessor('pcm16-worklet', Pcm16Worklet);
`;
    return URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
  };

  const start = async () => {
    setError(null);
    setEvents([]);
    setLatestAyah(null);
    setLatestText("");
    setBytesSent(0);
    bytesSentRef.current = 0;
    setElapsed(0);
    setStatus("connecting");

    try {
      const resp = await fetch(`${baseUrl}/v1/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_rate_hz: 16000,
          channels: 1,
          window_s: 12,
          hop_s: 2,
          buffer_s: 60,
          min_process_s: 2,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `Failed (${resp.status})`);

      const sessionId = data.session_id;
      sessionIdRef.current = sessionId;

      // Build ws URL using the demo path (no auth required)
      const apiBase = apiUrl.replace(/\/+$/, "");
      const wsProto = apiBase.startsWith("https") ? "wss:" : "ws:";
      const apiHost = apiBase.replace(/^https?:\/\//, "");
      const wsPath = `/demo/v1/sessions/${sessionId}/stream`;
      const wsUrl = `${wsProto}//${apiHost}${wsPath}`;

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = async () => {
        setStatus("streaming");
        startTimeRef.current = Date.now();
        timerRef.current = window.setInterval(() => {
          const s = (Date.now() - startTimeRef.current) / 1000;
          setElapsed(s);
          if (s >= MAX_DURATION_S) {
            stop();
          }
        }, 100);

        // Start audio capture
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
        streamRef.current = stream;

        const audioCtx = new AudioContext();
        audioCtxRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);

        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 1024;
        source.connect(analyser);
        analyserRef.current = analyser;

        const buf = new Float32Array(analyser.fftSize);
        const tick = () => {
          if (!analyserRef.current) return;
          analyserRef.current.getFloatTimeDomainData(buf);
          let sumSq = 0;
          for (let i = 0; i < buf.length; i++) sumSq += buf[i] * buf[i];
          const rms = Math.sqrt(sumSq / buf.length);
          const db = rms > 0 ? 20 * Math.log10(rms) : -100;
          setMicLevel(Math.max(0, Math.min(1, (db + 60) / 40)));
          animFrameRef.current = requestAnimationFrame(tick);
        };
        animFrameRef.current = requestAnimationFrame(tick);

        const workletUrl = buildWorkletUrl();
        await audioCtx.audioWorklet.addModule(workletUrl);
        URL.revokeObjectURL(workletUrl);

        const worklet = new AudioWorkletNode(audioCtx, "pcm16-worklet", {
          numberOfInputs: 1,
          numberOfOutputs: 0,
        });
        workletRef.current = worklet;

        worklet.port.onmessage = (ev) => {
          const buf = ev.data as ArrayBuffer;
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(buf);
            bytesSentRef.current += buf.byteLength;
            setBytesSent(bytesSentRef.current);
          }
        };

        source.connect(worklet);
      };

      ws.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        try {
          const obj = JSON.parse(ev.data);
          const event: StreamEvent = { type: obj.type || "unknown", data: obj, ts: Date.now() };
          setEvents((prev) => [...prev.slice(-99), event]);

          if (obj.type === "ayah_update") {
            const align = obj.ayah_alignment;
            if (align) setLatestAyah(align);
            const tx = obj.transcription;
            if (tx && tx.text_tail) setLatestText(tx.text_tail);
          } else if (obj.type === "session_end") {
            stop();
          }
        } catch {}
      };

      ws.onerror = () => {
        setStatus("error");
        setError("WebSocket connection error");
      };

      ws.onclose = (ev) => {
        if (status !== "idle") {
          setStatus("idle");
        }
      };
    } catch (e: any) {
      setError(e.message || "Failed to start stream");
      setStatus("error");
      cleanup();
    }
  };

  const stop = useCallback(() => {
    cleanup();
    setStatus("idle");
  }, [cleanup]);

  useEffect(() => {
    if (eventsEndRef.current) {
      eventsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [events]);

  const pct = (elapsed / MAX_DURATION_S) * 100;

  return (
    <div>
      <div className="card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
          <div className="actions-row">
            {status === "idle" || status === "error" ? (
              <button className="btn btn-primary" onClick={start}>
                Start Streaming
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stop}>
                Stop
              </button>
            )}
          </div>
          <div className="status-pill">
            <span className={`dot ${status === "streaming" ? "active" : status === "connecting" ? "loading" : status === "error" ? "error" : "idle"}`} />
            {status === "idle" && "Idle"}
            {status === "connecting" && "Connecting..."}
            {status === "streaming" && "Streaming"}
            {status === "error" && "Error"}
          </div>
        </div>

        {(status === "streaming" || status === "connecting") && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>
              <span>{elapsed.toFixed(1)}s / {MAX_DURATION_S}s</span>
              <span>{(bytesSent / 1024).toFixed(0)} KB sent</span>
            </div>
            <div className="progress-bar">
              <div className="fill recording" style={{ width: `${pct}%` }} />
            </div>
            <div style={{ marginTop: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>
                <span>Mic Level</span>
              </div>
              <div className="meter">
                <div className="fill" style={{ width: `${Math.round(micLevel * 100)}%` }} />
              </div>
            </div>
          </>
        )}

        {error && <div className="error-msg">{error}</div>}
      </div>

      {(latestText || latestAyah) && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Live Result</h3>
          {latestText && (
            <div className="ayah-text">{latestText}</div>
          )}
          {latestAyah && (
            <div className="kv-grid" style={{ marginTop: 12 }}>
              {latestAyah.start && latestAyah.end && (
                <>
                  <div className="label">Range</div>
                  <div className="value">
                    {latestAyah.start.surah_id}:{latestAyah.start.ayah_num} &rarr;{" "}
                    {latestAyah.end.surah_id}:{latestAyah.end.ayah_num}
                  </div>
                </>
              )}
              {latestAyah.confidence != null && (
                <>
                  <div className="label">Confidence</div>
                  <div className="value">{(latestAyah.confidence * 100).toFixed(0)}%</div>
                </>
              )}
            </div>
          )}
        </div>
      )}

      {events.length > 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <h3 style={{ fontSize: 14, fontWeight: 700 }}>Events ({events.length})</h3>
            <button className="btn" style={{ padding: "4px 10px", fontSize: 12 }} onClick={() => setEvents([])}>
              Clear
            </button>
          </div>
          <div className="stream-events">
            {events.map((ev, i) => (
              <div key={i} style={{ marginBottom: 4 }}>
                <span style={{ color: "var(--brand)" }}>[{ev.type}]</span>{" "}
                {ev.type === "ayah_update" && ev.data.ayah_alignment?.start
                  ? `${ev.data.ayah_alignment.start.surah_id}:${ev.data.ayah_alignment.start.ayah_num} → ${ev.data.ayah_alignment.end?.surah_id}:${ev.data.ayah_alignment.end?.ayah_num}`
                  : ev.type === "silence"
                  ? `silent ${Number(ev.data.silent_s || 0).toFixed(1)}s`
                  : JSON.stringify(ev.data).slice(0, 120)}
              </div>
            ))}
            <div ref={eventsEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}
