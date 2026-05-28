import Foundation

/// Runtime configuration, read from environment variables (set by scripts/run-mac.sh).
struct Config {
    let modelDir: String
    let tokenizerDir: String
    let apiKey: String
    let port: Int
    let host: String
    let language: String
    let wordTimestamps: Bool

    static func fromEnvironment() -> Config {
        let env = ProcessInfo.processInfo.environment
        return Config(
            modelDir: env["MODEL_DIR"] ?? "../models/ultra-fast-tarteel-coreml",
            tokenizerDir: env["TOKENIZER_DIR"] ?? "../models/whisper-base-tokenizer",
            apiKey: env["API_KEY"] ?? "",
            port: Int(env["PORT"] ?? "9000") ?? 9000,
            host: env["HOST"] ?? "127.0.0.1",
            language: env["LANGUAGE"] ?? "ar",
            wordTimestamps: (env["WORD_TIMESTAMPS"] ?? "true").lowercased() != "false"
        )
    }
}

enum AudioError: Error, CustomStringConvertible {
    case ffmpegFailed(String)
    case ffmpegNotFound

    var description: String {
        switch self {
        case .ffmpegFailed(let s): return "ffmpeg failed: \(s)"
        case .ffmpegNotFound: return "ffmpeg not found on PATH"
        }
    }
}

enum Audio {
    /// Decode an arbitrary audio file to 16 kHz mono signed-16 PCM WAV using ffmpeg,
    /// matching the Python transcriber's preprocessing. Returns the wav path.
    static func decodeToWav16kMono(input: String, output: String) throws {
        guard let ffmpeg = which("ffmpeg") else { throw AudioError.ffmpegNotFound }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: ffmpeg)
        proc.arguments = [
            "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            "-f", "wav", output,
        ]
        let err = Pipe()
        proc.standardError = err
        try proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            let data = err.fileHandleForReading.readDataToEndOfFile()
            throw AudioError.ffmpegFailed(String(data: data, encoding: .utf8) ?? "exit \(proc.terminationStatus)")
        }
    }

    struct SpeechSegment { let start: Double; let end: Double }

    /// Detect speech regions by running ffmpeg `silencedetect` and complementing the
    /// reported silence intervals. Used to chunk long recitations (the model emits one
    /// ayah per decode). Returns a single full-span segment if detection finds nothing.
    static func detectSpeechSegments(wavPath: String, totalDuration: Double) -> [SpeechSegment] {
        let whole = [SpeechSegment(start: 0, end: max(totalDuration, 0.1))]
        guard let ffmpeg = which("ffmpeg"), totalDuration > 0 else { return whole }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: ffmpeg)
        proc.arguments = [
            "-nostdin", "-hide_banner", "-i", wavPath,
            "-af", "silencedetect=noise=-32dB:d=0.35", "-f", "null", "-",
        ]
        let err = Pipe()
        proc.standardError = err
        proc.standardOutput = Pipe()
        do { try proc.run() } catch { return whole }
        let data = err.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        let log = String(data: data, encoding: .utf8) ?? ""

        // Parse "silence_start: X" / "silence_end: Y" lines into silence intervals.
        var silences: [(Double, Double)] = []
        var pendingStart: Double?
        for line in log.split(separator: "\n") {
            if let r = line.range(of: "silence_start:") {
                pendingStart = Double(line[r.upperBound...].trimmingCharacters(in: .whitespaces)
                    .split(separator: " ").first.map(String.init) ?? "")
            } else if let r = line.range(of: "silence_end:") {
                let rest = line[r.upperBound...].trimmingCharacters(in: .whitespaces)
                let endStr = rest.split(separator: " ").first.map(String.init) ?? ""
                if let s = pendingStart, let e = Double(endStr) { silences.append((s, e)) }
                pendingStart = nil
            }
        }
        if let s = pendingStart { silences.append((s, totalDuration)) }
        if silences.isEmpty { return whole }

        // Complement silences into speech segments, padding slightly and merging tiny gaps.
        let pad = 0.15
        var segments: [SpeechSegment] = []
        var cursor = 0.0
        for (s, e) in silences {
            if s - cursor > 0.25 {
                segments.append(SpeechSegment(
                    start: max(0, cursor - pad), end: min(totalDuration, s + pad)))
            }
            cursor = max(cursor, e)
        }
        if totalDuration - cursor > 0.25 {
            segments.append(SpeechSegment(start: max(0, cursor - pad), end: totalDuration))
        }
        return segments.isEmpty ? whole : segments
    }

    /// Extract [start, end] of a wav into a new 16 kHz mono wav.
    static func extractClip(input: String, start: Double, end: Double, output: String) throws {
        guard let ffmpeg = which("ffmpeg") else { throw AudioError.ffmpegNotFound }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: ffmpeg)
        proc.arguments = [
            "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", String(format: "%.3f", max(0, start)),
            "-to", String(format: "%.3f", end),
            "-i", input,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", output,
        ]
        let e = Pipe(); proc.standardError = e
        try proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            let d = e.fileHandleForReading.readDataToEndOfFile()
            throw AudioError.ffmpegFailed(String(data: d, encoding: .utf8) ?? "exit \(proc.terminationStatus)")
        }
    }

    /// Duration in seconds of a 16 kHz mono PCM16 wav, from its byte length.
    static func wavDurationSeconds(path: String) -> Double {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? Int, size > 44 else { return 0 }
        return Double(size - 44) / Double(16000 * 2)
    }

    private static func which(_ tool: String) -> String? {
        // Common locations first (Homebrew on Apple Silicon, Intel, system).
        for p in ["/opt/homebrew/bin/\(tool)", "/usr/local/bin/\(tool)", "/usr/bin/\(tool)"] {
            if FileManager.default.isExecutableFile(atPath: p) { return p }
        }
        // Fall back to PATH lookup.
        let env = ProcessInfo.processInfo.environment
        for dir in (env["PATH"] ?? "").split(separator: ":") {
            let p = "\(dir)/\(tool)"
            if FileManager.default.isExecutableFile(atPath: p) { return p }
        }
        return nil
    }
}

/// Remove Whisper special tokens like <|startoftranscript|>, <|ar|>, <|0.00|>,
/// <|endoftext|> that appear in the raw decoded segment text, then tidy whitespace.
func stripSpecialTokens(_ s: String) -> String {
    let stripped = s.replacingOccurrences(
        of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
    let collapsed = stripped.replacingOccurrences(
        of: "\\s+", with: " ", options: .regularExpression)
    return collapsed.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// Format seconds as HH:MM:SS.mmm (matches the Python transcriber's *_ts fields).
func formatTimestamp(_ seconds: Double) -> String {
    let s = max(0, seconds)
    let h = Int(s) / 3600
    let m = (Int(s) % 3600) / 60
    let sec = Int(s) % 60
    let ms = Int((s - Double(Int(s))) * 1000.0 + 0.5)
    return String(format: "%02d:%02d:%02d.%03d", h, m, sec, min(ms, 999))
}
