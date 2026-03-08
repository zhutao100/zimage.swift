import Foundation
import XCTest

private let truthyEnvironmentValues: Set<String> = ["1", "true", "yes", "on"]

func requireE2ETestsEnabled() throws {
  let enabled =
    ProcessInfo.processInfo.environment["ZIMAGE_RUN_E2E_TESTS"]?
    .trimmingCharacters(in: .whitespacesAndNewlines)
    .lowercased()
  guard let enabled, truthyEnvironmentValues.contains(enabled) else {
    throw XCTSkip("Set ZIMAGE_RUN_E2E_TESTS=1 to enable CLI end-to-end tests.")
  }
}

func resolveSwiftPMExecutable(named executableName: String, for testCase: AnyClass) throws -> URL {
  guard let executableURL = Bundle(for: testCase).executableURL else {
    throw XCTSkip("Cannot determine the SwiftPM test executable location.")
  }

  let binaryDir = executableURL.deletingLastPathComponent()
  let productsDir =
    binaryDir
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
  let productURL = productsDir.appendingPathComponent(executableName)
  guard FileManager.default.fileExists(atPath: productURL.path) else {
    throw XCTSkip(
      "\(executableName) not found at \(productURL.path). Run `swift build --product \(executableName)` first."
    )
  }

  return productURL
}

func ensureMLXMetalLibraryAdjacent(to executableURL: URL) throws {
  let metalLibraryURL = executableURL.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
  guard FileManager.default.fileExists(atPath: metalLibraryURL.path) else {
    throw XCTSkip("mlx.metallib not found at \(metalLibraryURL.path). Run scripts/build_mlx_metallib.sh first.")
  }
}
