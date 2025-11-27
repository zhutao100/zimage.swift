import Foundation
import Hub

public struct HubSnapshotOptions {
  public var repoId: String
  public var revision: String
  public var repoType: Hub.RepoType
  public var patterns: [String]
  public var cacheDirectory: URL?
  public var hfToken: String?
  public var offline: Bool
  public var useBackgroundSession: Bool

  public init(
    repoId: String,
    revision: String = "main",
    repoType: Hub.RepoType = .models,
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
    self.fractionCompleted = progress.totalUnitCount > 0
      ? Double(progress.completedUnitCount) / Double(progress.totalUnitCount)
      : 0
    self.completedUnitCount = progress.completedUnitCount
    self.totalUnitCount = progress.totalUnitCount
    self.estimatedSpeedBytesPerSecond = speed
  }
}

public actor HubSnapshot {
  public typealias ProgressHandler = @Sendable (HubSnapshotProgress) -> Void

  private let options: HubSnapshotOptions
  private let hubApi: HubApi
  private var cachedSnapshotURL: URL?

  public init(
    options: HubSnapshotOptions,
    hubApi: HubApi? = nil
  ) throws {
    self.options = options

    let cacheDirectory = try HubSnapshot.resolveCacheDirectory(
      requested: options.cacheDirectory,
      fileManager: FileManager.default
    )

    let api = hubApi ?? HubApi(
      downloadBase: cacheDirectory,
      hfToken: options.hfToken,
      useBackgroundSession: options.useBackgroundSession,
      useOfflineMode: options.offline ? true : nil
    )

    self.hubApi = api
  }

  public func prepare(progressHandler: ProgressHandler? = nil) async throws -> URL {
    if let cachedSnapshotURL,
      FileManager.default.fileExists(atPath: cachedSnapshotURL.path) {
      return cachedSnapshotURL
    }

    let repo = Hub.Repo(id: options.repoId, type: options.repoType)
    let patterns = options.patterns
    let snapshotURL = try await hubApi.snapshot(
      from: repo,
      revision: options.revision,
      matching: patterns,
      progressHandler: { progress, speed in
        progressHandler?(HubSnapshotProgress(progress: progress, speed: speed))
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
      throw Hub.HubClientError.fileNotFound(relativePath)
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
