import Vapor

// Response DTOs. Field names intentionally mirror the Python transcriber's output
// so the Rust API (api/src/alignment_v2.rs::extract_timed_tokens_from_transcription
// and process_job) consumes them unchanged.

struct WordDTO: Content {
    let word: String
    let start_s: Double
    let end_s: Double
    let start_ts: String
    let end_ts: String
    let probability: Double
}

struct SegmentDTO: Content {
    let start_s: Double
    let end_s: Double
    let start_ts: String
    let end_ts: String
    let text: String
    let avg_logprob: Double
    let no_speech_prob: Double
    let compression_ratio: Double
    let temperature: Double
    let words: [WordDTO]
}

// The Rust API (api/src/main.rs process_job) expects the segments nested under a
// "transcription" object, with "text" and "params" at the top level.
struct TranscriptionDTO: Content {
    let language: String
    let language_probability: Double
    let duration_s: Double
    let segments: [SegmentDTO]
}

struct TranscribeResponse: Content {
    let transcription: TranscriptionDTO
    let text: String
    let params: [String: String]
    let backend: String
}

struct HealthResponse: Content {
    let ok: Bool
    let backend: String
    let model_loaded: Bool
}

struct ErrorResponse: Content {
    let error: String
}
