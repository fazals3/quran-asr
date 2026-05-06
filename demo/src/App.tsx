import { useState, useCallback } from "react";
import UploadMode from "./UploadMode";
import RecordMode from "./RecordMode";
import StreamMode from "./StreamMode";

type Mode = "upload" | "record" | "stream";

const DEFAULT_API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

export default function App() {
  const [mode, setMode] = useState<Mode>("upload");
  const [apiUrl, setApiUrl] = useState(() => {
    return localStorage.getItem("quran-asr-demo-api-url") || DEFAULT_API_URL;
  });

  const handleApiUrlChange = useCallback((val: string) => {
    setApiUrl(val);
    localStorage.setItem("quran-asr-demo-api-url", val);
  }, []);

  const baseUrl = apiUrl.replace(/\/+$/, "") + "/demo";

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Quran ASR Demo</h1>
          <div className="subtitle">
            Upload, record, or stream Quran recitation for transcription and
            ayah alignment.
          </div>
        </div>
      </header>

      <div className="config-bar">
        <label>API Server</label>
        <input
          value={apiUrl}
          onChange={(e) => handleApiUrlChange(e.target.value)}
          placeholder="http://your-server:8001"
          spellCheck={false}
        />
      </div>

      <div className="main">
        <div className="tabs">
          <button
            className={`tab ${mode === "upload" ? "active" : ""}`}
            onClick={() => setMode("upload")}
          >
            Upload Audio
          </button>
          <button
            className={`tab ${mode === "record" ? "active" : ""}`}
            onClick={() => setMode("record")}
          >
            Record
          </button>
          <button
            className={`tab ${mode === "stream" ? "active" : ""}`}
            onClick={() => setMode("stream")}
          >
            Live Stream
          </button>
        </div>

        {mode === "upload" && <UploadMode baseUrl={baseUrl} />}
        {mode === "record" && <RecordMode baseUrl={baseUrl} />}
        {mode === "stream" && <StreamMode baseUrl={baseUrl} apiUrl={apiUrl} />}
      </div>
    </div>
  );
}
