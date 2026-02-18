import Foundation
import HuggingFace

#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

enum HuggingFaceHubError: Error, LocalizedError {
  case invalidRepoId(String)
  case snapshotNotFound(String)
  case noFilesMatched(repoId: String, patterns: [String])

  var errorDescription: String? {
    switch self {
    case let .invalidRepoId(repoId):
      return "Invalid Hugging Face repo id: \(repoId)"
    case let .snapshotNotFound(repoId):
      return "Snapshot not found for: \(repoId)"
    case let .noFilesMatched(repoId, patterns):
      return "No files matched for '\(repoId)' (patterns: \(patterns.joined(separator: ", ")))"
    }
  }
}

enum HuggingFaceHub {
  static func cacheDirectory() -> URL {
    HubCache.default.cacheDirectory
  }

  static func resolveSnapshotDirectory(
    cache: HubCache,
    repo: Repo.ID,
    kind: Repo.Kind = .model,
    revision: String?
  ) -> URL? {
    let snapshotsRoot = cache.snapshotsDirectory(repo: repo, kind: kind)
    let fm = FileManager.default

    if let revision, !revision.isEmpty {
      let trimmed = revision.trimmingCharacters(in: .whitespacesAndNewlines)
      let commitHash: String?
      if isCommitHash(trimmed) {
        commitHash = trimmed
      } else {
        commitHash = cache.resolveRevision(repo: repo, kind: kind, ref: trimmed)
      }

      if let commitHash {
        let candidate = snapshotsRoot.appendingPathComponent(commitHash)
        if fm.fileExists(atPath: candidate.path) {
          return candidate
        }
      }
    }

    guard fm.fileExists(atPath: snapshotsRoot.path) else {
      return nil
    }

    if let snapshots = try? fm.contentsOfDirectory(at: snapshotsRoot, includingPropertiesForKeys: nil) {
      for snapshot in snapshots {
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: snapshot.path, isDirectory: &isDir), isDir.boolValue {
          return snapshot
        }
      }
    }

    return nil
  }

  static func ensureSnapshot(
    repoId: String,
    kind: Repo.Kind = .model,
    revision: String = "main",
    matching patterns: [String] = [],
    cacheDirectory: URL? = nil,
    hfToken: String? = nil,
    offline: Bool = false,
    useBackgroundSession: Bool = false,
    progressHandler: (@Sendable (Progress) -> Void)? = nil
  ) async throws -> URL {
    guard let repo = Repo.ID(rawValue: repoId) else {
      throw HuggingFaceHubError.invalidRepoId(repoId)
    }

    let cache = cacheDirectory.map { HubCache(cacheDirectory: $0) } ?? .default

    if offline {
      guard let cached = resolveSnapshotDirectory(cache: cache, repo: repo, kind: kind, revision: revision) else {
        throw HuggingFaceHubError.snapshotNotFound(repoId)
      }
      return cached
    }

    let tokenProvider: TokenProvider = hfToken.map { .fixed(token: $0) } ?? .environment

    let session = makeSession(useBackgroundSession: useBackgroundSession)
    let client = HubClient(
      session: session,
      host: detectHost(),
      tokenProvider: tokenProvider,
      cache: cache
    )

    let entries = try await client.listFiles(in: repo, kind: kind, revision: revision, recursive: true)
    let filePaths = entries
      .filter { $0.type == .file }
      .map(\.path)
      .filter { filePath in
        guard !patterns.isEmpty else { return true }
        return patterns.contains { fnmatch($0, filePath, 0) == 0 }
      }

    guard !filePaths.isEmpty else {
      throw HuggingFaceHubError.noFilesMatched(repoId: repoId, patterns: patterns)
    }

    let progress = Progress(totalUnitCount: Int64(filePaths.count))
    progressHandler?(progress)

    let tempRoot = FileManager.default.temporaryDirectory
      .appendingPathComponent("zimage.swift-hub", isDirectory: true)
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    for filePath in filePaths {
      if cache.cachedFilePath(repo: repo, kind: kind, revision: revision, filename: filePath) != nil {
        progress.completedUnitCount += 1
        progressHandler?(progress)
        continue
      }

      let destination = tempRoot.appendingPathComponent(filePath, isDirectory: false)
      try FileManager.default.createDirectory(
        at: destination.deletingLastPathComponent(),
        withIntermediateDirectories: true
      )

      _ = try await client.downloadFile(
        at: filePath,
        from: repo,
        to: destination,
        kind: kind,
        revision: revision,
        progress: nil
      )
      try? FileManager.default.removeItem(at: destination)

      progress.completedUnitCount += 1
      progressHandler?(progress)
    }

    guard let snapshotURL = resolveSnapshotDirectory(cache: cache, repo: repo, kind: kind, revision: revision) else {
      throw HuggingFaceHubError.snapshotNotFound(repoId)
    }

    progressHandler?(progress)
    return snapshotURL
  }

  private static func isCommitHash(_ revision: String) -> Bool {
    revision.count == 40 && revision.allSatisfy { $0.isHexDigit }
  }

  private static func detectHost() -> URL {
    if let endpoint = ProcessInfo.processInfo.environment["HF_ENDPOINT"],
       let url = URL(string: endpoint)
    {
      return url
    }
    return HubClient.defaultHost
  }

  private static func makeSession(useBackgroundSession: Bool) -> URLSession {
    guard useBackgroundSession else {
      return URLSession(configuration: .default)
    }

    #if canImport(FoundationNetworking)
    return URLSession(configuration: .default)
    #else
    return URLSession(configuration: .background(withIdentifier: "zimage.swift.hub.\(UUID().uuidString)"))
    #endif
  }
}
