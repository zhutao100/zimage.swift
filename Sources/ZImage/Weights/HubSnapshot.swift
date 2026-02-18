import Foundation
import HuggingFace

public enum HubRepoType: String, Sendable {
  case models
  case datasets
  case spaces

  var kind: Repo.Kind {
    switch self {
    case .models:
      return .model
    case .datasets:
      return .dataset
    case .spaces:
      return .space
    }
  }
}

public enum HubSnapshotError: Error, LocalizedError, Sendable {
  case fileNotFound(String)

  public var errorDescription: String? {
    switch self {
    case let .fileNotFound(path):
      return "File not found in snapshot: \(path)"
    }
  }
}

public struct HubSnapshotOptions {
  public var repoId: String
  public var revision: String
  public var repoType: HubRepoType
  public var patterns: [String]
  public var cacheDirectory: URL?
  public var hfToken: String?
  public var offline: Bool
  public var useBackgroundSession: Bool

  public init(
    repoId: String,
    revision: String = "main",
    repoType: HubRepoType = .models,
    patterns: [String] = [],
    cacheDirectory: URL? = nil,
    hfToken: String? = nil,
    offline: Bool = false,
    useBackgroundSession: Bool = false
  ) {
    self.repoId = repoId
    self.revision = revision
    self.repoType = repoType
    self.patterns = patterns
    self.cacheDirectory = cacheDirectory
    self.hfToken = hfToken
    self.offline = offline
    self.useBackgroundSession = useBackgroundSession
  }
}

public struct HubSnapshotProgress: Sendable {
  public let fractionCompleted: Double
  public let completedUnitCount: Int64
  public let totalUnitCount: Int64
  public let estimatedSpeedBytesPerSecond: Double?

  init(progress: Progress, speed: Double?) {
    fractionCompleted = progress.totalUnitCount > 0
      ? Double(progress.completedUnitCount) / Double(progress.totalUnitCount)
      : 0
    completedUnitCount = progress.completedUnitCount
    totalUnitCount = progress.totalUnitCount
    estimatedSpeedBytesPerSecond = speed
  }
}

public actor HubSnapshot {
  public typealias ProgressHandler = @Sendable (HubSnapshotProgress) -> Void

  private let options: HubSnapshotOptions
  private var cachedSnapshotURL: URL?
  private let resolvedCacheDirectory: URL

  public init(
    options: HubSnapshotOptions
  ) throws {
    self.options = options

    resolvedCacheDirectory = try HubSnapshot.resolveCacheDirectory(
      requested: options.cacheDirectory,
      fileManager: FileManager.default
    )
  }

  public func prepare(progressHandler: ProgressHandler? = nil) async throws -> URL {
    if let cachedSnapshotURL,
       FileManager.default.fileExists(atPath: cachedSnapshotURL.path)
    {
      return cachedSnapshotURL
    }

    let snapshotURL = try await HuggingFaceHub.ensureSnapshot(
      repoId: options.repoId,
      kind: options.repoType.kind,
      revision: options.revision,
      matching: options.patterns,
      cacheDirectory: resolvedCacheDirectory,
      hfToken: options.hfToken,
      offline: options.offline,
      useBackgroundSession: options.useBackgroundSession,
      progressHandler: { progress in
        progressHandler?(HubSnapshotProgress(progress: progress, speed: nil))
      }
    )
    cachedSnapshotURL = snapshotURL
    return snapshotURL
  }

  public func fileURL(
    for relativePath: String,
    progressHandler: ProgressHandler? = nil
  ) async throws -> URL {
    let snapshot = try await prepare(progressHandler: progressHandler)
    let url = snapshot.appending(path: relativePath)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw HubSnapshotError.fileNotFound(relativePath)
    }
    return url
  }

  public func invalidateCache() {
    cachedSnapshotURL = nil
  }

  private static func resolveCacheDirectory(
    requested: URL?,
    fileManager: FileManager
  ) throws -> URL {
    if let explicit = requested {
      try fileManager.createDirectory(at: explicit, withIntermediateDirectories: true, attributes: nil)
      return explicit
    }

    if let caches = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first {
      let directory = caches.appending(path: "qwen-image/hub")
      try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
      return directory
    }

    let fallback = fileManager.temporaryDirectory.appending(path: "qwen-image/hub")
    try fileManager.createDirectory(at: fallback, withIntermediateDirectories: true, attributes: nil)
    return fallback
  }
}
