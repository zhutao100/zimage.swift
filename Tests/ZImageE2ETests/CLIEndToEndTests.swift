import XCTest
import Foundation

/// End-to-end tests for ZImageCLI command line interface.
/// These tests build and run the actual CLI executable.
/// Run with: xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -only-testing:ZImageE2ETests -parallel-testing-enabled NO
final class CLIEndToEndTests: XCTestCase {

  private var cliPath: String?
  private static let cliDependencyTimestamp: Date = newestCLIDependencyTimestamp()

  /// Project root directory (derived from test file location)
  private static let projectRoot: URL = {
    // Go up from Tests/ZImageE2ETests/CLIEndToEndTests.swift to project root
    URL(fileURLWithPath: #file)
      .deletingLastPathComponent()  // Remove CLIEndToEndTests.swift
      .deletingLastPathComponent()  // Remove ZImageE2ETests
      .deletingLastPathComponent()  // Remove Tests -> project root
  }()

  /// Output directory for test-generated images (inside project)
  private static let outputDir: URL = {
    let url = projectRoot
      .appendingPathComponent("Tests")
      .appendingPathComponent("ZImageE2ETests")
      .appendingPathComponent("Resources")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }()

  override func setUp() async throws {
    try await super.setUp()
    // Build CLI if not already built
    cliPath = try await buildCLI()
  }

  override class func tearDown() {
    // Keep Resources directory for visual inspection
    // try? FileManager.default.removeItem(at: outputDir)
    super.tearDown()
  }

  // MARK: - Basic CLI Tests

  func testHelpCommand() async throws {
    try skipIfNoCLI()

    let (stdout, stderr, exitCode) = try await runCLI(["--help"])

    XCTAssertEqual(exitCode, 0, "Help should exit with code 0")
    XCTAssertTrue(stdout.contains("USAGE") || stdout.contains("usage") || stdout.contains("Usage") || stderr.contains("USAGE"),
                  "Help output should contain usage information")
  }

  // MARK: - Text-to-Image Generation Tests

  func testBasicTextToImageGeneration() async throws {
    try skipIfNoCLI()
    try skipIfNoGPU()

    let outputPath = Self.outputDir.appendingPathComponent("e2e_basic.png").path

    let (_, stderr, exitCode) = try await runCLI([
      "-p", "a simple test image of a red apple",
      "-o", outputPath,
      "-W", "512",
      "-H", "512",
      "-s", "9",
      "-m", "mzbac/z-image-turbo-8bit"
    ], timeout: 300)

    if exitCode != 0 {
      print("CLI stderr: \(stderr)")
    }

    XCTAssertEqual(exitCode, 0, "Generation should succeed")
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputPath), "Output file should exist")
  }

  // MARK: - LoRA Generation Tests

  func testLoRAGeneration() async throws {
    try skipIfNoCLI()
    try skipIfNoGPU()

    let outputPath = Self.outputDir.appendingPathComponent("e2e_lora.png").path

    let (_, stderr, exitCode) = try await runCLI([
      "-p", "a lion",
      "--lora", "ostris/z_image_turbo_childrens_drawings",
      "--lora-scale", "1.0",
      "-W", "512",
      "-H", "512",
      "-s", "9",
      "-o", outputPath,
      "-m", "mzbac/z-image-turbo-8bit"
    ], timeout: 300)

    if exitCode != 0 {
      print("CLI stderr: \(stderr)")
    }

    XCTAssertEqual(exitCode, 0, "LoRA generation should succeed")
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputPath), "Output file should exist")
  }

  // MARK: - ControlNet Generation Tests

  func testControlNetWithCanny() async throws {
    try skipIfNoCLI()
    try skipIfNoGPU()

    let controlImagePath = getCannyImagePath()
    let outputPath = Self.outputDir.appendingPathComponent("e2e_controlnet_canny.png").path

    let (_, stderr, exitCode) = try await runCLI([
      "control",
      "-p", "A hyper-realistic close-up portrait of a leopard face hiding behind dense green jungle leaves, camouflaged, direct eye contact, intricate fur detail, bright yellow eyes, cinematic lighting, soft shadows, National Geographic photography, 8k, sharp focus, depth of field",
      "-c", controlImagePath,
      "--cw", "mzbac/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8bit",
      "--cs", "0.75",
      "-W", "512",
      "-H", "512",
      "-s", "9",
      "-o", outputPath,
      "-m", "mzbac/Z-Image-Turbo-8bit"
    ], timeout: 300)

    if exitCode != 0 {
      print("CLI stderr: \(stderr)")
    }

    XCTAssertEqual(exitCode, 0, "ControlNet with Canny should succeed")
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputPath), "Output file should exist")
  }

  func testControlNetWithDepth() async throws {
    try skipIfNoCLI()
    try skipIfNoGPU()

    let controlImagePath = getDepthImagePath()
    let outputPath = Self.outputDir.appendingPathComponent("e2e_controlnet_depth.png").path

    let (_, stderr, exitCode) = try await runCLI([
      "control",
      "-p", "A hyperrealistic architectural photograph of a spacious, minimalist modern hallway interior. Large floor-to-ceiling windows on the right wall fill the space with bright natural daylight. A light gray sectional sofa and a low, modern coffee table are placed in the foreground on a light wood floor. A large potted plant is visible further down the hallway. White walls, clean lines, serene atmosphere, highly detailed, 8k resolution, cinematic lighting",
      "-c", controlImagePath,
      "--cw", "mzbac/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8bit",
      "--cs", "0.75",
      "-W", "512",
      "-H", "512",
      "-s", "9",
      "-o", outputPath,
      "-m", "mzbac/Z-Image-Turbo-8bit"
    ], timeout: 300)

    if exitCode != 0 {
      print("CLI stderr: \(stderr)")
    }

    XCTAssertEqual(exitCode, 0, "ControlNet with Depth should succeed")
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputPath), "Output file should exist")
  }

  func testControlNetWithHed() async throws {
    try skipIfNoCLI()
    try skipIfNoGPU()

    let controlImagePath = getHedImagePath()
    let outputPath = Self.outputDir.appendingPathComponent("e2e_controlnet_hed.png").path

    let (_, stderr, exitCode) = try await runCLI([
      "control",
      "-p", "A photorealistic film still of a man in a dark shirt sitting at a dining table in a modern kitchen at night, looking down at a bowl of soup. A glass bottle and a glass of white wine are in the foreground. Warm, low, cinematic lighting, soft shadows, shallow depth of field, contemplative atmosphere, highly detailed.",
      "-c", controlImagePath,
      "--cw", "mzbac/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8bit",
      "--cs", "0.75",
      "-W", "512",
      "-H", "512",
      "-s", "9",
      "-o", outputPath,
      "-m", "mzbac/Z-Image-Turbo-8bit"
    ], timeout: 300)

    if exitCode != 0 {
      print("CLI stderr: \(stderr)")
    }

    XCTAssertEqual(exitCode, 0, "ControlNet with HED should succeed")
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputPath), "Output file should exist")
  }

  func testControlNetWithPose() async throws {
    try skipIfNoCLI()
    try skipIfNoGPU()

    let controlImagePath = getPoseImagePath()
    let outputPath = Self.outputDir.appendingPathComponent("e2e_controlnet_pose.png").path

    let (_, stderr, exitCode) = try await runCLI([
      "control",
      "-p", "一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动",
      "-c", controlImagePath,
      "--cw", "mzbac/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8bit",
      "--cs", "0.75",
      "-W", "512",
      "-H", "512",
      "-s", "9",
      "-o", outputPath,
      "-m", "mzbac/Z-Image-Turbo-8bit"
    ], timeout: 300)

    if exitCode != 0 {
      print("CLI stderr: \(stderr)")
    }

    XCTAssertEqual(exitCode, 0, "ControlNet with Pose should succeed")
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputPath), "Output file should exist")
  }

  // MARK: - Error Handling Tests

  func testMissingPrompt() async throws {
    try skipIfNoCLI()

    let outputPath = Self.outputDir.appendingPathComponent("e2e_no_prompt.png").path

    let (stdout, stderr, exitCode) = try await runCLI([
      "-o", outputPath
    ])

    // Should fail or show usage when prompt is missing
    let output = stdout + stderr
    XCTAssertTrue(exitCode != 0 || output.contains("Usage") || output.contains("prompt"),
                  "Should fail or show usage without prompt")
  }

  func testControlNetMissingControlImage() async throws {
    try skipIfNoCLI()

    let outputPath = Self.outputDir.appendingPathComponent("e2e_cn_no_image.png").path

    let (stdout, stderr, exitCode) = try await runCLI([
      "control",
      "-p", "test",
      "--cw", "mzbac/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8bit",
      "-o", outputPath
    ])

    // Should fail when control image is missing
    let output = stdout + stderr
    XCTAssertTrue(exitCode != 0 || output.contains("control-image") || output.contains("required"),
                  "Should fail without control image")
  }

  func testControlNetMissingWeights() async throws {
    try skipIfNoCLI()

    let controlImagePath = getCannyImagePath()
    let outputPath = Self.outputDir.appendingPathComponent("e2e_cn_no_weights.png").path

    let (stdout, stderr, exitCode) = try await runCLI([
      "control",
      "-p", "test",
      "-c", controlImagePath,
      "-o", outputPath
    ])

    // Should fail when controlnet weights are missing
    let output = stdout + stderr
    XCTAssertTrue(exitCode != 0 || output.contains("controlnet-weights") || output.contains("required"),
                  "Should fail without controlnet weights")
  }

  // MARK: - Helper Functions

  private func buildCLI() async throws -> String {
    // Use project-local .build folder for xcodebuild outputs
    let buildDir = Self.projectRoot.appendingPathComponent(".build")

    // Check if CLI already exists in local build folder
    let releasePath = buildDir.appendingPathComponent("Build/Products/Release/ZImageCLI")
    let debugPath = buildDir.appendingPathComponent("Build/Products/Debug/ZImageCLI")
    if let existing = [releasePath, debugPath].first(where: { FileManager.default.fileExists(atPath: $0.path) }),
       Self.isCLIBinaryUpToDate(existing) {
      return existing.path
    }

    // Try to build with xcodebuild to local build folder (proper Metal library bundling)
    let buildProcess = Process()
    buildProcess.executableURL = URL(fileURLWithPath: "/usr/bin/xcodebuild")
    buildProcess.arguments = [
      "build",
      "-scheme", "ZImageCLI",
      "-configuration", "Release",
      "-destination", "platform=macOS",
      "-derivedDataPath", buildDir.path
    ]
    buildProcess.currentDirectoryURL = Self.projectRoot

    let pipe = Pipe()
    buildProcess.standardOutput = pipe
    buildProcess.standardError = pipe

    try buildProcess.run()
    buildProcess.waitUntilExit()

    if buildProcess.terminationStatus == 0 {
      // Check for the built CLI
      if FileManager.default.fileExists(atPath: releasePath.path) {
        return releasePath.path
      }
      if FileManager.default.fileExists(atPath: debugPath.path) {
        return debugPath.path
      }
    }

    // If build fails, return empty string (tests will be skipped)
    return ""
  }

  private func runCLI(_ arguments: [String], timeout: TimeInterval = 60) async throws -> (stdout: String, stderr: String, exitCode: Int32) {
    guard let path = cliPath, !path.isEmpty else {
      throw CLITestError.cliNotBuilt
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: path)
    process.arguments = arguments
    process.currentDirectoryURL = Self.projectRoot

    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    try process.run()

    // Wait with timeout
    let deadline = Date().addingTimeInterval(timeout)
    while process.isRunning && Date() < deadline {
      try await Task.sleep(nanoseconds: 100_000_000) // 0.1 second
    }

    if process.isRunning {
      process.terminate()
      throw CLITestError.timeout
    }

    let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

    let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
    let stderr = String(data: stderrData, encoding: .utf8) ?? ""

    return (stdout, stderr, process.terminationStatus)
  }

  private func skipIfNoCLI() throws {
    guard let path = cliPath, !path.isEmpty, FileManager.default.fileExists(atPath: path) else {
      throw XCTSkip("CLI not built. Run 'swift build -c release --product ZImageCLI' first.")
    }
  }

  private func skipIfNoGPU() throws {
    if ProcessInfo.processInfo.environment["CI"] != nil {
      throw XCTSkip("Skipping GPU-intensive test in CI environment")
    }
  }

  // MARK: - Control Image Paths (using existing images in ./images directory)

  private func getCannyImagePath() -> String {
    Self.projectRoot.appendingPathComponent("images").appendingPathComponent("canny.jpg").path
  }

  private func getDepthImagePath() -> String {
    Self.projectRoot.appendingPathComponent("images").appendingPathComponent("depth.jpg").path
  }

  private func getPoseImagePath() -> String {
    Self.projectRoot.appendingPathComponent("images").appendingPathComponent("pose.jpg").path
  }

  private func getHedImagePath() -> String {
    Self.projectRoot.appendingPathComponent("images").appendingPathComponent("hed.jpg").path
  }

  enum CLITestError: Error {
    case cliNotBuilt
    case timeout
    case executionFailed(String)
  }

  private static func isCLIBinaryUpToDate(_ url: URL) -> Bool {
    guard let binaryDate = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate else {
      return false
    }
    return binaryDate >= cliDependencyTimestamp
  }

  private static func newestCLIDependencyTimestamp() -> Date {
    let fm = FileManager.default
    var newest = Date.distantPast

    func consider(_ url: URL) {
      guard let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate else { return }
      if date > newest { newest = date }
    }

    consider(projectRoot.appendingPathComponent("Package.swift"))
    consider(projectRoot.appendingPathComponent("Package.resolved"))

    let sourcesDir = projectRoot.appendingPathComponent("Sources")
    if let enumerator = fm.enumerator(
      at: sourcesDir,
      includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
      options: [.skipsHiddenFiles]
    ) {
      for case let fileURL as URL in enumerator {
        let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .contentModificationDateKey])
        guard values?.isRegularFile == true, let date = values?.contentModificationDate else { continue }
        if date > newest { newest = date }
      }
    }

    return newest
  }
}
