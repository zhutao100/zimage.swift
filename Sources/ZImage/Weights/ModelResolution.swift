import Foundation
import HuggingFace

public enum ModelResolutionError: Error, LocalizedError {
  case modelNotFound(String)
  case downloadFailed(String, Error)
  case networkUnavailable
  case authorizationRequired(String)

  public var errorDescription: String? {
    switch self {
    case let .modelNotFound(spec):
      return "Model not found: \(spec)"
    case let .downloadFailed(modelId, error):
      return "Failed to download '\(modelId)': \(error.localizedDescription)"
    case .networkUnavailable:
      return "No internet connection. Please check your network or use a local model path."
    case let .authorizationRequired(modelId):
      return "Model '\(modelId)' not found or requires authentication"
    }
  }
}

public enum ModelResolution {
  public static func isHuggingFaceModelId(_ modelSpec: String) -> Bool {
    if modelSpec.hasPrefix("/") || modelSpec.hasPrefix("./") || modelSpec.hasPrefix("../") {
      return false
    }

    let url = URL(fileURLWithPath: modelSpec).standardizedFileURL
    if FileManager.default.fileExists(atPath: url.path) {
      return false
    }

    let baseSpec = String(modelSpec.split(separator: ":")[0])
    let parts = baseSpec.split(separator: "/")

    guard parts.count == 2 else {
      return false
    }

    let org = String(parts[0])
    let repo = String(parts[1])

    guard !org.isEmpty && !repo.isEmpty else {
      return false
    }

    let pathIndicators = ["models", "model", "weights", "data", "datasets", "checkpoints", "output", "tmp", "temp", "cache"]
    if pathIndicators.contains(org.lowercased()) {
      return false
    }

    if org.filter({ $0 == "." }).count > 1 {
      return false
    }

    return true
  }

  public static func resolve(
    modelSpec: String,
    defaultModelId _: String = ZImageRepository.id,
    defaultRevision: String = ZImageRepository.revision,
    filePatterns: [String] = ["*.safetensors", "*.json", "tokenizer/*"],
    requireWeights: Bool = true,
    snapshotValidator: (@Sendable (URL) -> Bool)? = nil,
    progressHandler: (@Sendable (Progress) -> Void)? = nil
  ) async throws -> URL {
    let localURL = URL(fileURLWithPath: modelSpec).standardizedFileURL
    if FileManager.default.fileExists(atPath: localURL.path) {
      return localURL
    }

    if !isHuggingFaceModelId(modelSpec) {
      throw ModelResolutionError.modelNotFound(modelSpec)
    }

    let parts = modelSpec.split(separator: ":", maxSplits: 1)
    let modelId = String(parts[0])
    let revision = parts.count > 1 ? String(parts[1]) : defaultRevision

    if let cachedURL = findCachedModel(
      modelId: modelId,
      revision: revision,
      requireWeights: requireWeights,
      snapshotValidator: snapshotValidator
    ) {
      return cachedURL
    }

    return try await downloadModel(
      modelId: modelId,
      revision: revision,
      filePatterns: filePatterns,
      progressHandler: progressHandler
    )
  }

  public static func resolveOrDefault(
    modelSpec: String?,
    defaultModelId: String = ZImageRepository.id,
    defaultRevision: String = ZImageRepository.revision,
    filePatterns: [String] = ["*.safetensors", "*.json", "tokenizer/*"],
    requireWeights: Bool = true,
    snapshotValidator: (@Sendable (URL) -> Bool)? = nil,
    progressHandler: (@Sendable (Progress) -> Void)? = nil
  ) async throws -> URL {
    if let spec = modelSpec {
      return try await resolve(
        modelSpec: spec,
        defaultModelId: defaultModelId,
        defaultRevision: defaultRevision,
        filePatterns: filePatterns,
        requireWeights: requireWeights,
        snapshotValidator: snapshotValidator,
        progressHandler: progressHandler
      )
    }

    if let cachedURL = findCachedModel(
      modelId: defaultModelId,
      revision: defaultRevision,
      requireWeights: requireWeights,
      snapshotValidator: snapshotValidator
    ) {
      return cachedURL
    }

    return try await downloadModel(
      modelId: defaultModelId,
      revision: defaultRevision,
      filePatterns: filePatterns,
      progressHandler: progressHandler
    )
  }

  private static func getHuggingFaceCacheDirectory() -> URL {
    HuggingFaceHub.cacheDirectory()
  }

  private static func findCachedModel(
    modelId: String,
    revision: String?,
    requireWeights: Bool = true,
    snapshotValidator: (@Sendable (URL) -> Bool)? = nil
  ) -> URL? {
    let fm = FileManager.default
    let cacheDir = getHuggingFaceCacheDirectory()

    func directoryHasSafetensors(_ directory: URL) -> Bool {
      let contents = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
      if contents.contains(where: { $0.pathExtension == "safetensors" }) {
        return true
      }

      let subDirs = contents.filter { url in
        var isDir: ObjCBool = false
        return fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
      }

      for subDir in subDirs {
        if let subContents = try? fm.contentsOfDirectory(at: subDir, includingPropertiesForKeys: nil) {
          if subContents.contains(where: { $0.pathExtension == "safetensors" }) {
            return true
          }
        }
      }

      return false
    }

    func directoryHasModelIndexOrConfig(_ directory: URL) -> Bool {
      let modelIndex = directory.appendingPathComponent("model_index.json")
      let configFile = directory.appendingPathComponent("config.json")
      return fm.fileExists(atPath: modelIndex.path) || fm.fileExists(atPath: configFile.path)
    }

    func isValidCacheDirectory(_ directory: URL) -> Bool {
      if requireWeights {
        guard directoryHasSafetensors(directory) else { return false }
      } else {
        guard directoryHasModelIndexOrConfig(directory) else { return false }
      }

      if let snapshotValidator {
        return snapshotValidator(directory)
      }
      return true
    }

    // HuggingFace CLI / huggingface_hub cache layout:
    // ~/.cache/huggingface/hub/models--ORG--REPO/snapshots/<commit>/
    let repoCacheRoot = cacheDir
      .appendingPathComponent("models--\(modelId.replacingOccurrences(of: "/", with: "--"))")
    let snapshotsRoot = repoCacheRoot.appendingPathComponent("snapshots")

    if fm.fileExists(atPath: snapshotsRoot.path) {
      var preferredSnapshot: URL?

      if let revision, !revision.isEmpty {
        let trimmed = revision.trimmingCharacters(in: .whitespacesAndNewlines)
        let isCommitHash = trimmed.count == 40 && trimmed.allSatisfy { $0.isHexDigit }
        if isCommitHash {
          let candidate = snapshotsRoot.appendingPathComponent(trimmed)
          if fm.fileExists(atPath: candidate.path), isValidCacheDirectory(candidate) {
            preferredSnapshot = candidate
          }
        } else {
          let refFile = repoCacheRoot.appendingPathComponent("refs").appendingPathComponent(trimmed)
          if let commit = try? String(contentsOf: refFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !commit.isEmpty
          {
            let candidate = snapshotsRoot.appendingPathComponent(commit)
            if fm.fileExists(atPath: candidate.path), isValidCacheDirectory(candidate) {
              preferredSnapshot = candidate
            }
          }
        }
      }

      if let preferredSnapshot {
        return preferredSnapshot
      }

      if let snapshots = try? fm.contentsOfDirectory(at: snapshotsRoot, includingPropertiesForKeys: nil) {
        for snapshot in snapshots where isValidCacheDirectory(snapshot) {
          return snapshot
        }
      }
    }

    // swift-transformers HubApi local layout:
    // <downloadBase>/models/ORG/REPO/
    let swiftTransformersPath = cacheDir.appendingPathComponent("models").appendingPathComponent(modelId)
    if fm.fileExists(atPath: swiftTransformersPath.path), isValidCacheDirectory(swiftTransformersPath) {
      return swiftTransformersPath
    }

    return nil
  }

  private static func downloadModel(
    modelId: String,
    revision: String,
    filePatterns: [String],
    progressHandler: (@Sendable (Progress) -> Void)?
  ) async throws -> URL {
    do {
      return try await HuggingFaceHub.ensureSnapshot(
        repoId: modelId,
        revision: revision,
        matching: filePatterns,
        progressHandler: progressHandler
      )
    } catch {
      if let clientError = error as? HTTPClientError,
         case let .responseError(response, _) = clientError,
         response.statusCode == 401 || response.statusCode == 403
      {
        throw ModelResolutionError.authorizationRequired(modelId)
      }

      if let urlError = error as? URLError,
         urlError.code == .notConnectedToInternet
      {
        throw ModelResolutionError.networkUnavailable
      }
      throw ModelResolutionError.downloadFailed(modelId, error)
    }
  }
}
