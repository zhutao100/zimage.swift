import Foundation
import XCTest

enum TestFixtures {
  static var fixturesRoot: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appending(path: "Fixtures")
  }

  static func snapshot(named name: String) -> URL {
    fixturesRoot
      .appending(path: "Snapshots")
      .appending(path: name)
  }

  static func withTemporaryCopy(of source: URL, _ body: (URL) throws -> Void) throws {
    let fm = FileManager.default
    let tempRoot = fm.temporaryDirectory.appendingPathComponent("zimage_tests_\(UUID().uuidString)")
    try fm.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: tempRoot) }

    let dest = tempRoot.appendingPathComponent(source.lastPathComponent)
    try fm.copyItem(at: source, to: dest)
    try body(dest)
  }

  static func createEmptyFile(at url: URL) throws {
    let dir = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try Data().write(to: url, options: [.atomic])
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
