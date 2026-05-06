interface ResultProps {
  data: any;
}

export default function ResultDisplay({ data }: ResultProps) {
  if (!data) return null;

  const result = data.result || data;
  const transcription = result.transcription || {};
  const guess = result.guess || {};
  const alignment = result.ayah_alignment || {};
  const timing = result.timing || {};

  const text = transcription.text || "";
  const durationS = transcription.duration_s;
  const surahId = guess.predicted_surah_id;
  const alignedRange = guess.aligned_range;

  const startRef = alignment.start;
  const endRef = alignment.end;
  const confidence = alignment.confidence;

  return (
    <div className="result">
      <div className="result-header">
        <h3>Result</h3>
        <div style={{ display: "flex", gap: 6 }}>
          {confidence != null && (
            <span
              className={`chip ${confidence > 0.7 ? "good" : confidence > 0.4 ? "warn" : "bad"}`}
            >
              Confidence: {(confidence * 100).toFixed(0)}%
            </span>
          )}
          {timing.total_s != null && (
            <span className="chip brand">
              {Number(timing.total_s).toFixed(2)}s
            </span>
          )}
        </div>
      </div>

      {text && <div className="ayah-text">{text}</div>}

      <div className="card" style={{ marginTop: 12 }}>
        <div className="kv-grid">
          {surahId != null && (
            <>
              <div className="label">Predicted Surah</div>
              <div className="value">{surahId}</div>
            </>
          )}
          {startRef && endRef && (
            <>
              <div className="label">Aligned Range</div>
              <div className="value">
                {startRef.surah_id}:{startRef.ayah_num} &rarr;{" "}
                {endRef.surah_id}:{endRef.ayah_num}
              </div>
            </>
          )}
          {durationS != null && (
            <>
              <div className="label">Audio Duration</div>
              <div className="value">{Number(durationS).toFixed(1)}s</div>
            </>
          )}
          {timing.transcribe_s != null && (
            <>
              <div className="label">Transcription</div>
              <div className="value">
                {Number(timing.transcribe_s).toFixed(3)}s
              </div>
            </>
          )}
          {timing.align_s != null && (
            <>
              <div className="label">Alignment</div>
              <div className="value">{Number(timing.align_s).toFixed(3)}s</div>
            </>
          )}
          {timing.guess_s != null && (
            <>
              <div className="label">Guess</div>
              <div className="value">{Number(timing.guess_s).toFixed(3)}s</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
