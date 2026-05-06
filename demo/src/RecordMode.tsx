import { useState, useRef, useCallback, useEffect } from "react";
import ResultDisplay from "./ResultDisplay";

interface Props {
  baseUrl: string;
}

const MAX_DURATION_S = 60;

export default function RecordMode({ baseUrl }: Props) {
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [micLevel, setMicLevel] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number>(0);
  const startTimeRef = useRef(0);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = 0;
    }
    setRecording(false);
    setMicLevel(0);
  }, []);

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  const startRecording = async () => {
    setError(null);
    setResult(null);
    setBlob(null);
    setElapsed(0);
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
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
        const level = Math.max(0, Math.min(1, (db + 60) / 40));
        setMicLevel(level);
        animFrameRef.current = requestAnimationFrame(tick);
      };
      animFrameRef.current = requestAnimationFrame(tick);

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const b = new Blob(chunksRef.current, { type: mimeType });
        setBlob(b);
      };

      recorder.start(250);
      setRecording(true);
      startTimeRef.current = Date.now();

      timerRef.current = window.setInterval(() => {
        const s = (Date.now() - startTimeRef.current) / 1000;
        setElapsed(s);
        if (s >= MAX_DURATION_S) {
          stopRecording();
        }
      }, 100);
    } catch (e: any) {
      setError(
        e.name === "NotAllowedError"
          ? "Microphone permission denied. Please allow mic access."
          : e.message || "Failed to start recording"
      );
    }
  };

  const submit = async () => {
    if (!blob) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(10);

    try {
      const form = new FormData();
      form.append("file", blob, "recording.webm");

      const progressTimer = setInterval(() => {
        setProgress((p) => Math.min(p + 5, 90));
      }, 1000);

      const resp = await fetch(
        `${baseUrl}/v1/transcribe?wait=true&wait_timeout_s=120`,
        { method: "POST", body: form }
      );

      clearInterval(progressTimer);
      setProgress(100);

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || `Server error (${resp.status})`);
      }
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
      setTimeout(() => setProgress(0), 500);
    }
  };

  const pct = (elapsed / MAX_DURATION_S) * 100;

  return (
    <div>
      <div className="card">
        <div className="actions-row">
          {!recording && !blob && (
            <button className="btn btn-primary" onClick={startRecording}>
              Start Recording
            </button>
          )}
          {recording && (
            <>
              <button className="btn btn-danger" onClick={stopRecording}>
                Stop
              </button>
              <span className="recording-indicator">
                <span className="dot" />
                Recording...
              </span>
              <span
                style={{ marginLeft: "auto", color: "var(--text-muted)", fontSize: 13 }}
              >
                {elapsed.toFixed(1)}s / {MAX_DURATION_S}s
              </span>
            </>
          )}
          {blob && !recording && (
            <>
              <button
                className="btn btn-primary"
                onClick={submit}
                disabled={loading}
              >
                {loading ? "Processing..." : "Transcribe"}
              </button>
              <button
                className="btn"
                onClick={() => {
                  setBlob(null);
                  setResult(null);
                  setError(null);
                  setElapsed(0);
                }}
              >
                Discard & Re-record
              </button>
              <span style={{ color: "var(--text-muted)", fontSize: 13 }}>
                {elapsed.toFixed(1)}s recorded
              </span>
            </>
          )}
        </div>

        {recording && (
          <>
            <div className="progress-bar" style={{ marginTop: 16 }}>
              <div className="fill recording" style={{ width: `${pct}%` }} />
            </div>
            <div style={{ marginTop: 12 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 12,
                  color: "var(--text-muted)",
                  marginBottom: 4,
                }}
              >
                <span>Mic Level</span>
              </div>
              <div className="meter">
                <div
                  className="fill"
                  style={{ width: `${Math.round(micLevel * 100)}%` }}
                />
              </div>
            </div>
          </>
        )}

        {progress > 0 && !recording && (
          <div className="progress-bar">
            <div className="fill" style={{ width: `${progress}%` }} />
          </div>
        )}

        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && <ResultDisplay data={result} />}
    </div>
  );
}
