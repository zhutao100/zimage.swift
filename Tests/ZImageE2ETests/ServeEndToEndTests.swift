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
    let process = Process()
    process.executableURL = URL(fileURLWithPath: try requireServePath())
    process.arguments = ["serve", "--socket", socketURL.path]
    process.currentDirectoryURL = Self.projectRoot
    process.standardOutput = Pipe()
    process.standardError = Pipe()

    try process.run()
    defer {
      if process.isRunning {
        process.terminate()
        process.waitUntilExit()
      }
      try? FileManager.default.removeItem(at: socketURL)
    }

    let deadline = Date().addingTimeInterval(10)
    while Date() < deadline {
      if FileManager.default.fileExists(atPath: socketURL.path) {
        return
      }
      Thread.sleep(forTimeInterval: 0.1)
    }

    XCTFail("Serve daemon did not create socket at \(socketURL.path)")
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
