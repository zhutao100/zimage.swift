import Foundation

enum HuggingFaceCacheTestSupport {
  static func withTemporarySnapshot<T>(
    repoID: String,
    revision: String = "main",
    files: [String],
    perform: (_ modelSpec: String, _ snapshot: URL) async throws -> T
  ) async throws -> T {
    let fm = FileManager.default
    let tempCache = fm.temporaryDirectory.appendingPathComponent("hf_cache_\(UUID().uuidString)")
    let repoRoot = tempCache.appendingPathComponent("models--\(repoID.replacingOccurrences(of: "/", with: "--"))")
    let commit = "0123456789abcdef0123456789abcdef01234567"
    let snapshot = repoRoot.appendingPathComponent("snapshots").appendingPathComponent(commit)
    let blobs = repoRoot.appendingPathComponent("blobs")
    let refs = repoRoot.appendingPathComponent("refs")

    try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
    try fm.createDirectory(at: refs, withIntermediateDirectories: true)
    try Data(commit.utf8).write(to: refs.appendingPathComponent(revision))

    for file in files {
      let blobURL = blobs.appendingPathComponent(UUID().uuidString)
      try Data("test".utf8).write(to: blobURL)

      let snapshotFile = snapshot.appendingPathComponent(file)
      try fm.createDirectory(at: snapshotFile.deletingLastPathComponent(), withIntermediateDirectories: true)
      try fm.createSymbolicLink(atPath: snapshotFile.path, withDestinationPath: blobURL.path)
    }

    let previousHFHubCache = ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
    let previousHFEndpoint = ProcessInfo.processInfo.environment["HF_ENDPOINT"]
    setenv("HF_HUB_CACHE", tempCache.path, 1)
    setenv("HF_ENDPOINT", "http://127.0.0.1:1", 1)

    defer {
      if let previousHFHubCache {
        setenv("HF_HUB_CACHE", previousHFHubCache, 1)
      } else {
        unsetenv("HF_HUB_CACHE")
      }

      if let previousHFEndpoint {
        setenv("HF_ENDPOINT", previousHFEndpoint, 1)
      } else {
        unsetenv("HF_ENDPOINT")
      }

      try? fm.removeItem(at: tempCache)
    }

    return try await perform("\(repoID):\(revision)", snapshot)
  }
}
