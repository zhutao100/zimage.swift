import Foundation
import XCTest

private let truthyEnvironmentValues: Set<String> = ["1", "true", "yes", "on"]

func integrationTestsEnabled() -> Bool {
  guard
    let enabled =
      ProcessInfo.processInfo.environment["ZIMAGE_RUN_INTEGRATION_TESTS"]?
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
  else {
    return false
  }
  return truthyEnvironmentValues.contains(enabled)
}

func requireIntegrationTestsEnabled() throws {
  guard integrationTestsEnabled() else {
    throw XCTSkip("Set ZIMAGE_RUN_INTEGRATION_TESTS=1 to enable integration tests.")
  }
}

func ensureMLXMetalLibraryColocated(for testCase: AnyClass) throws {
  guard let executableURL = Bundle(for: testCase).executableURL else {
    throw XCTSkip("Cannot determine test executable location for colocating mlx.metallib.")
  }

  let binaryDir = executableURL.deletingLastPathComponent()
  let colocated = binaryDir.appendingPathComponent("mlx.metallib")
  if FileManager.default.fileExists(atPath: colocated.path) { return }

  // Expected layout for SwiftPM:
  //   <bin>/zimage.swiftPackageTests.xctest/Contents/MacOS/zimage.swiftPackageTests
  // and scripts/build_mlx_metallib.sh writes:
  //   <bin>/mlx.metallib
  let binRoot =
    binaryDir
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
  let built = binRoot.appendingPathComponent("mlx.metallib")
  guard FileManager.default.fileExists(atPath: built.path) else {
    throw XCTSkip("mlx.metallib not found at \(built.path). Run scripts/build_mlx_metallib.sh first.")
  }

  _ = try? FileManager.default.removeItem(at: colocated)
  try FileManager.default.copyItem(at: built, to: colocated)
}
