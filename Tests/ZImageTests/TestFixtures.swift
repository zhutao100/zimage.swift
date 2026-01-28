import Foundation

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
