import Foundation
import Vapor

@main
enum Entrypoint {
    static func main() async throws {
        let env = try Environment.detect()
        let app = try await Application.make(env)
        let config = Config.fromEnvironment()
        let transcriber = Transcriber(config: config)

        app.http.server.configuration.hostname = config.host
        app.http.server.configuration.port = config.port
        app.routes.defaultMaxBodySize = "250mb"

        app.logger.info("loading WhisperKit model from \(config.modelDir)")
        do {
            try await transcriber.load()
            app.logger.info("WhisperKit model loaded; listening on \(config.host):\(config.port)")
        } catch {
            // Keep serving so /health can report not-loaded and the operator sees the error.
            app.logger.error("failed to load WhisperKit model: \(String(reflecting: error))")
        }

        try routes(app, transcriber: transcriber, config: config)

        do {
            try await app.execute()
        } catch {
            try? await app.asyncShutdown()
            throw error
        }
        try await app.asyncShutdown()
    }
}

struct UploadForm: Content {
    var file: File
}

func routes(_ app: Application, transcriber: Transcriber, config: Config) throws {
    app.get("health") { _ async -> HealthResponse in
        HealthResponse(ok: true, backend: "whisperkit-coreml", model_loaded: await transcriber.isLoaded)
    }

    app.on(.POST, "v1", "transcribe", body: .collect(maxSize: "250mb")) { req async throws -> Response in
        if !config.apiKey.isEmpty {
            guard let bearer = req.headers.bearerAuthorization, bearer.token == config.apiKey else {
                return try await ErrorResponse(error: "unauthorized")
                    .encodeResponse(status: .unauthorized, for: req)
            }
        }

        let form: UploadForm
        do {
            form = try req.content.decode(UploadForm.self)
        } catch {
            return try await ErrorResponse(error: "missing audio file (multipart field 'file')")
                .encodeResponse(status: .badRequest, for: req)
        }

        let data = Data(buffer: form.file.data)
        let tmp = FileManager.default.temporaryDirectory
        let id = UUID().uuidString
        let ext = (form.file.filename as NSString).pathExtension
        let inPath = tmp.appendingPathComponent("in-\(id).\(ext.isEmpty ? "bin" : ext)").path
        // Distinct from inPath so ffmpeg never sees identical input/output (e.g. .wav uploads,
        // including streaming window.wav frames).
        let wavPath = tmp.appendingPathComponent("out-\(id).wav").path
        defer {
            try? FileManager.default.removeItem(atPath: inPath)
            try? FileManager.default.removeItem(atPath: wavPath)
        }

        do {
            try data.write(to: URL(fileURLWithPath: inPath))
            try Audio.decodeToWav16kMono(input: inPath, output: wavPath)
        } catch {
            return try await ErrorResponse(error: "audio decode failed: \(error)")
                .encodeResponse(status: .badRequest, for: req)
        }

        let duration = Audio.wavDurationSeconds(path: wavPath)
        do {
            let result = try await transcriber.transcribe(wavPath: wavPath, durationS: duration)
            return try await result.encodeResponse(for: req)
        } catch {
            return try await ErrorResponse(error: "transcription failed: \(error)")
                .encodeResponse(status: .internalServerError, for: req)
        }
    }
}
