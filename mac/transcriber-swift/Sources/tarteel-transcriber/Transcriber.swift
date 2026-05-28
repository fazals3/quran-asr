import Foundation
import WhisperKit

enum TranscriberError: Error, CustomStringConvertible {
    case notLoaded
    var description: String {
        switch self {
        case .notLoaded: return "model not loaded"
        }
    }
}

/// Owns the WhisperKit instance and turns its output into the Python-compatible JSON.
actor Transcriber {
    private let config: Config
    private var whisperKit: WhisperKit?

    init(config: Config) {
        self.config = config
    }

    var isLoaded: Bool { whisperKit != nil }

    /// Load the CoreML model + tokenizer fully from disk (no network).
    func load() async throws {
        if whisperKit != nil { return }
        let modelURL = URL(fileURLWithPath: config.modelDir, isDirectory: true)
        let tokenizerURL = URL(fileURLWithPath: config.tokenizerDir, isDirectory: true)
        let wkConfig = WhisperKitConfig(
            modelFolder: modelURL.path,
            tokenizerFolder: tokenizerURL,
            download: false
        )
        whisperKit = try await WhisperKit(wkConfig)
    }

    /// One WhisperKit decode pass over a (sub-)clip; timestamps are shifted by `offset`
    /// seconds so chunked results line up on the original audio timeline.
    private func runWhisper(on wavPath: String, offset: Double, wk: WhisperKit) async throws
        -> (segments: [SegmentDTO], language: String?)
    {
        var options = DecodingOptions()
        options.task = .transcribe
        options.language = config.language
        options.temperature = 0.0
        options.wordTimestamps = config.wordTimestamps

        let results = try await wk.transcribe(audioPath: wavPath, decodeOptions: options)

        var out: [SegmentDTO] = []
        var lang: String?
        for result in results {
            if !result.language.isEmpty { lang = result.language }
            for seg in result.segments {
                let text = stripSpecialTokens(seg.text)
                if text.isEmpty { continue }
                var words: [WordDTO] = []
                for w in seg.words ?? [] {
                    let trimmed = w.word.trimmingCharacters(in: .whitespacesAndNewlines)
                    if trimmed.isEmpty { continue }
                    let ws = Double(w.start) + offset
                    let we = Double(w.end) + offset
                    words.append(WordDTO(
                        word: trimmed,
                        start_s: ws, end_s: we,
                        start_ts: formatTimestamp(ws), end_ts: formatTimestamp(we),
                        probability: Double(w.probability)
                    ))
                }
                let ss = Double(seg.start) + offset
                let se = Double(seg.end) + offset
                out.append(SegmentDTO(
                    start_s: ss, end_s: se,
                    start_ts: formatTimestamp(ss), end_ts: formatTimestamp(se),
                    text: text,
                    avg_logprob: Double(seg.avgLogprob),
                    no_speech_prob: Double(seg.noSpeechProb),
                    compression_ratio: Double(seg.compressionRatio),
                    temperature: Double(seg.temperature),
                    words: words
                ))
            }
        }
        return (out, lang)
    }

    func transcribe(wavPath: String, durationS: Double) async throws -> TranscribeResponse {
        guard let wk = whisperKit else { throw TranscriberError.notLoaded }

        // The Tarteel model is fine-tuned on single ayahs and emits <|endoftext|> after one,
        // so a single decode pass stops early on multi-ayah audio. Split the recitation on
        // silence (ffmpeg) and transcribe each speech region, offsetting timestamps. This
        // mirrors the Python transcriber's VAD approach and keeps long recitations complete.
        let speech = Audio.detectSpeechSegments(wavPath: wavPath, totalDuration: durationS)

        var segments: [SegmentDTO] = []
        var detectedLanguage = config.language

        if speech.count <= 1 {
            let (segs, lang) = try await runWhisper(on: wavPath, offset: 0, wk: wk)
            segments = segs
            if let lang { detectedLanguage = lang }
        } else {
            let tmp = FileManager.default.temporaryDirectory
            for (i, sp) in speech.enumerated() {
                let clip = tmp.appendingPathComponent("chunk-\(UUID().uuidString)-\(i).wav").path
                defer { try? FileManager.default.removeItem(atPath: clip) }
                do {
                    try Audio.extractClip(input: wavPath, start: sp.start, end: sp.end, output: clip)
                } catch { continue }
                let (segs, lang) = try await runWhisper(on: clip, offset: sp.start, wk: wk)
                segments.append(contentsOf: segs)
                if let lang { detectedLanguage = lang }
            }
        }

        let fullText = segments
            .map { $0.text }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return TranscribeResponse(
            transcription: TranscriptionDTO(
                language: detectedLanguage,
                language_probability: 1.0,
                duration_s: durationS,
                segments: segments
            ),
            text: fullText,
            params: [
                "backend": "whisperkit-coreml",
                "language": detectedLanguage,
                "word_timestamps": config.wordTimestamps ? "true" : "false",
            ],
            backend: "whisperkit-coreml"
        )
    }
}
