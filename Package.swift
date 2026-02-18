// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "zimage.swift",
  platforms: [.macOS(.v14), .iOS(.v16)],
  products: [
    .library(name: "ZImage", targets: ["ZImage"]),
    .executable(name: "ZImageCLI", targets: ["ZImageCLI"]),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.29.1")),
    .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.7.0"),
    .package(
      url: "https://github.com/huggingface/swift-transformers",
      .upToNextMinor(from: "0.1.24")
    ),
    .package(url: "https://github.com/apple/swift-log.git", from: "1.6.4"),
  ],
  targets: [
    .target(
      name: "ZImage",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXOptimizers", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
        .product(name: "Transformers", package: "swift-transformers"),
        .product(name: "Logging", package: "swift-log"),
      ],
      path: "Sources/ZImage"
    ),
    .executableTarget(
      name: "ZImageCLI",
      dependencies: ["ZImage"],
      path: "Sources/ZImageCLI"
    ),
    .testTarget(
      name: "ZImageTests",
      dependencies: [
        "ZImage",
        .product(name: "MLX", package: "mlx-swift"),
      ],
      path: "Tests/ZImageTests"
    ),
    .testTarget(
      name: "ZImageIntegrationTests",
      dependencies: [
        "ZImage",
        .product(name: "MLX", package: "mlx-swift"),
      ],
      path: "Tests/ZImageIntegrationTests",
      resources: [
        .copy("Resources"),
      ]
    ),
    .testTarget(
      name: "ZImageE2ETests",
      dependencies: ["ZImage"],
      path: "Tests/ZImageE2ETests"
    ),
  ]
)
