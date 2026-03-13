import Foundation
import XCTest

final class ServeEndToEndTests: XCTestCase {
  private var servePath: String?

  private static let projectRoot: URL = {
    URL(fileURLWithPath: #file)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }()

  override func setUpWithError() throws {
    try super.setUpWithError()
    try requireE2ETestsEnabled()
    let serveURL = try resolveSwiftPMExecutable(named: "ZImageServe", for: type(of: self))
    try ensureMLXMetalLibraryAdjacent(to: serveURL)
    servePath = serveURL.path
  }

  func testHelpCommand() async throws {
    try skipIfNoServe()

    let (stdout, stderr, exitCode) = try await runServe(["--help"])
    let output = stdout + stderr

    XCTAssertEqual(exitCode, 0)
    XCTAssertTrue(output.contains("ZImageServe"))
    XCTAssertTrue(output.contains("serve"))
  }

  func testServeHelpCommand() async throws {
    try skipIfNoServe()

    let (stdout, stderr, exitCode) = try await runServe(["serve", "--help"])
    let output = stdout + stderr

    XCTAssertEqual(exitCode, 0)
    XCTAssertTrue(output.contains("staging daemon"))
    XCTAssertTrue(output.contains("--socket"))
  }

  func testDaemonCreatesSocketFile() throws {
    try skipIfNoServe()

    let socketURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("zimage-serve-\(UUID().uuidString).sock")
    let process = try startDaemon(socketURL: socketURL)
    defer {
      stopDaemon(process, socketURL: socketURL)
    }

    XCTAssertTrue(FileManager.default.fileExists(atPath: socketURL.path))
  }

  func testStatusCommandReportsIdleDaemon() async throws {
    try skipIfNoServe()

    let socketURL = makeSocketURL()
    let process = try startDaemon(socketURL: socketURL)
    defer {
      stopDaemon(process, socketURL: socketURL)
    }

    let (stdout, stderr, exitCode) = try await runServe(["--socket", socketURL.path, "status"])
    let output = stdout + stderr

    XCTAssertEqual(exitCode, 0)
    XCTAssertTrue(output.contains("Socket: \(socketURL.path)"))
    XCTAssertTrue(output.contains("Executing: no"))
    XCTAssertTrue(output.contains("Resident worker: none"))
  }

  func testShutdownCommandStopsIdleDaemon() async throws {
    try skipIfNoServe()

    let socketURL = makeSocketURL()
    let process = try startDaemon(socketURL: socketURL)

    let (stdout, stderr, exitCode) = try await runServe(["--socket", socketURL.path, "shutdown"])
    let output = stdout + stderr

    XCTAssertEqual(exitCode, 0)
    XCTAssertTrue(output.contains("Shutdown acknowledged"))

    let deadline = Date().addingTimeInterval(10)
    while process.isRunning && Date() < deadline {
      try await Task.sleep(nanoseconds: 100_000_000)
    }

    XCTAssertFalse(process.isRunning, "Daemon should exit after an idle shutdown request")
    XCTAssertFalse(FileManager.default.fileExists(atPath: socketURL.path))
  }

  func testBatchCommandSubmitsStructuredManifest() async throws {
    try skipIfNoServe()

    let socketURL = makeSocketURL()
    let process = try startDaemon(socketURL: socketURL)
    defer {
      stopDaemon(process, socketURL: socketURL)
    }

    let tempDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: tempDirectory) }

    let missingModelPath = tempDirectory.appendingPathComponent("missing-model").path
    let outputPath = tempDirectory.appendingPathComponent("batch.png").path
    let manifestURL = tempDirectory.appendingPathComponent("jobs.json")
    try """
      {
        "version": 1,
        "jobs": [
          {
            "id": "batch-1",
            "kind": "text",
            "prompt": "a mountain lake",
            "model": "\(missingModelPath)",
            "outputPath": "\(outputPath)"
          }
        ]
      }
      """.write(to: manifestURL, atomically: true, encoding: .utf8)

    let (stdout, stderr, exitCode) = try await runServe(
      ["--socket", socketURL.path, "batch", manifestURL.path],
      timeout: 120
    )
    let output = stdout + stderr

    XCTAssertNotEqual(exitCode, 0)
    XCTAssertTrue(output.contains("batch-1"))
    XCTAssertTrue(output.contains("failed staged job"))
    XCTAssertTrue(process.isRunning)
  }

  func testMarkdownCommandSubmitsFencedInvocation() async throws {
    try skipIfNoServe()

    let socketURL = makeSocketURL()
    let process = try startDaemon(socketURL: socketURL)
    defer {
      stopDaemon(process, socketURL: socketURL)
    }

    let tempDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: tempDirectory) }

    let missingModelPath = tempDirectory.appendingPathComponent("missing-model").path
    let outputPathTemplate = tempDirectory.appendingPathComponent("markdown-$(printf runtime).png").path
    let markdownURL = tempDirectory.appendingPathComponent("prompts.md")
    try """
      ```bash
      ZImageServe --prompt "a forest path" --model \(missingModelPath) --output "\(outputPathTemplate)"
      ```
      """.write(to: markdownURL, atomically: true, encoding: .utf8)

    let (stdout, stderr, exitCode) = try await runServe(
      ["--socket", socketURL.path, "markdown", markdownURL.path],
      timeout: 120
    )
    let output = stdout + stderr

    XCTAssertNotEqual(exitCode, 0)
    XCTAssertTrue(output.contains("markdown-1"))
    XCTAssertTrue(output.contains("failed staged job"))
    XCTAssertFalse(output.contains("Command substitution failed"))
    XCTAssertTrue(process.isRunning)
  }

  private func runServe(_ arguments: [String], timeout: TimeInterval = 60) async throws -> (
    stdout: String, stderr: String, exitCode: Int32
  ) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: try requireServePath())
    process.arguments = arguments
    process.currentDirectoryURL = Self.projectRoot

    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    try process.run()

    let deadline = Date().addingTimeInterval(timeout)
    while process.isRunning && Date() < deadline {
      try await Task.sleep(nanoseconds: 100_000_000)
    }

    if process.isRunning {
      process.terminate()
      throw ServeTestError.timeout
    }

    let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    return (stdout, stderr, process.terminationStatus)
  }

  private func makeSocketURL() -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("zimage-serve-\(UUID().uuidString).sock")
  }

  private func makeTemporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("zimage-serve-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
  }

  private func startDaemon(socketURL: URL) throws -> Process {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: try requireServePath())
    process.arguments = ["serve", "--socket", socketURL.path]
    process.currentDirectoryURL = Self.projectRoot
    process.standardOutput = Pipe()
    process.standardError = Pipe()

    try process.run()
    try waitForSocket(at: socketURL)
    return process
  }

  private func stopDaemon(_ process: Process, socketURL: URL) {
    if process.isRunning {
      process.terminate()
      process.waitUntilExit()
    }
    try? FileManager.default.removeItem(at: socketURL)
  }

  private func waitForSocket(at socketURL: URL, timeout: TimeInterval = 10) throws {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
      if FileManager.default.fileExists(atPath: socketURL.path) {
        return
      }
      Thread.sleep(forTimeInterval: 0.1)
    }

    throw ServeTestError.timeout
  }

  private func skipIfNoServe() throws {
    guard let servePath, !servePath.isEmpty, FileManager.default.fileExists(atPath: servePath) else {
      throw XCTSkip("Serve executable is unavailable after SwiftPM test preparation.")
    }
  }

  private func requireServePath() throws -> String {
    try skipIfNoServe()
    guard let servePath else {
      throw ServeTestError.serveNotBuilt
    }
    return servePath
  }

  enum ServeTestError: Error {
    case serveNotBuilt
    case timeout
  }
}
