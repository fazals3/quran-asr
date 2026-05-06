import { useState, useRef, useCallback } from "react";
import ResultDisplay from "./ResultDisplay";

interface Props {
  baseUrl: string;
}

const MAX_DURATION_S = 60;

export default function UploadMode({ baseUrl }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(10);

    try {
      const form = new FormData();
      form.append("file", file);

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

  return (
    <div>
      <div className="card">
        <input
          ref={inputRef}
          type="file"
          accept="audio/*"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />

        <div
          className={`drop-zone ${dragging ? "dragging" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <div className="icon">&#x1F3A4;</div>
          <p>
            Drop an audio file here, or click to browse.
            <br />
            Max {MAX_DURATION_S}s of Quran recitation.
          </p>
        </div>

        {file && (
          <div className="file-info">
            <div>
              <div className="name">{file.name}</div>
              <div className="meta">
                {(file.size / 1024 / 1024).toFixed(2)} MB &middot;{" "}
                {file.type || "audio"}
              </div>
            </div>
          </div>
        )}

        {progress > 0 && (
          <div className="progress-bar">
            <div className="fill" style={{ width: `${progress}%` }} />
          </div>
        )}

        <div className="actions-row" style={{ marginTop: 16 }}>
          <button
            className="btn btn-primary"
            disabled={!file || loading}
            onClick={submit}
          >
            {loading ? "Processing..." : "Transcribe"}
          </button>
          {file && !loading && (
            <button
              className="btn"
              onClick={() => {
                setFile(null);
                setResult(null);
                setError(null);
              }}
            >
              Clear
            </button>
          )}
        </div>

        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && <ResultDisplay data={result} />}
    </div>
  );
}
