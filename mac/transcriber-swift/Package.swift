// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "tarteel-transcriber",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        // Argmax open-source SDK (provides the WhisperKit product). v1.0.0 graduated
        // WhisperKit into the argmax-oss-swift package.
        .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", from: "1.0.0"),
        // Lightweight HTTP server + multipart handling.
        .package(url: "https://github.com/vapor/vapor.git", from: "4.115.0"),
    ],
    targets: [
        .executableTarget(
            name: "tarteel-transcriber",
            dependencies: [
                .product(name: "WhisperKit", package: "argmax-oss-swift"),
                .product(name: "Vapor", package: "vapor"),
            ],
            path: "Sources/tarteel-transcriber"
        )
    ]
)
