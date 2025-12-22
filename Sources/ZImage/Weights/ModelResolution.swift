import Foundation
import Hub

public enum ModelResolutionError: Error, LocalizedError {
  case modelNotFound(String)
  case downloadFailed(String, Error)
  case networkUnavailable
  case authorizationRequired(String)

  public var errorDescription: String? {
    switch self {
    case .modelNotFound(let spec):
      return "Model not found: \(spec)"
    case .downloadFailed(let modelId, let error):
      return "Failed to download '\(modelId)': \(error.localizedDescription)"
    case .networkUnavailable:
      return "No internet connection. Please check your network or use a local model path."
    case .authorizationRequired(let modelId):
      return "Model '\(modelId)' not found or requires authentication"
    }
  }
}

public struct ModelResolution {

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
    defaultModelId: String = ZImageRepository.id,
    defaultRevision: String = ZImageRepository.revision,
    filePatterns: [String] = ["*.safetensors", "*.json", "tokenizer/*"],
    requireWeights: Bool = true,
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

    if let cachedURL = findCachedModel(modelId: modelId, requireWeights: requireWeights) {
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
    progressHandler: (@Sendable (Progress) -> Void)? = nil
  ) async throws -> URL {
    if let spec = modelSpec {
      return try await resolve(
        modelSpec: spec,
        defaultModelId: defaultModelId,
        defaultRevision: defaultRevision,
        filePatterns: filePatterns,
        requireWeights: requireWeights,
        progressHandler: progressHandler
      )
    }

    if let cachedURL = findCachedModel(modelId: defaultModelId, requireWeights: requireWeights) {
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
    let env = ProcessInfo.processInfo.environment

    if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
      return URL(fileURLWithPath: hubCache)
    }

    if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
      return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
    }

    let homeDir = FileManager.default.homeDirectoryForCurrentUser
    return homeDir.appendingPathComponent(".cache/huggingface/hub")
  }

  private static func createHubApi() -> HubApi {
    let hfCacheDir = getHuggingFaceCacheDirectory()
    try? FileManager.default.createDirectory(at: hfCacheDir, withIntermediateDirectories: true)
    return HubApi(downloadBase: hfCacheDir)
  }

  private static func findCachedModel(modelId: String, requireWeights: Bool = true) -> URL? {
    let fm = FileManager.default
    let cacheDir = getHuggingFaceCacheDirectory()

    let modelCachePath = cacheDir
      .appendingPathComponent("models--\(modelId.replacingOccurrences(of: "/", with: "--"))")
      .appendingPathComponent("snapshots")

    guard fm.fileExists(atPath: modelCachePath.path) else {
      return nil
    }

    guard let snapshots = try? fm.contentsOfDirectory(at: modelCachePath, includingPropertiesForKeys: nil) else {
      return nil
    }

    for snapshot in snapshots {
      let modelIndex = snapshot.appendingPathComponent("model_index.json")
      let configFile = snapshot.appendingPathComponent("config.json")

      let hasModelIndex = fm.fileExists(atPath: modelIndex.path)
      let hasConfig = fm.fileExists(atPath: configFile.path)

      guard hasModelIndex || hasConfig else {
        continue
      }

      if requireWeights {
        let contents = (try? fm.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: nil)) ?? []
        let hasSafetensors = contents.contains { $0.pathExtension == "safetensors" }

        let subDirs = contents.filter { url in
          var isDir: ObjCBool = false
          return fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
        }

        var hasNestedSafetensors = false
        for subDir in subDirs {
          if let subContents = try? fm.contentsOfDirectory(at: subDir, includingPropertiesForKeys: nil) {
            if subContents.contains(where: { $0.pathExtension == "safetensors" }) {
              hasNestedSafetensors = true
              break
            }
          }
        }

        if !hasSafetensors && !hasNestedSafetensors {
          continue
        }
      }

      return snapshot
    }

    return nil
  }

  private static func downloadModel(
    modelId: String,
    revision: String,
    filePatterns: [String],
    progressHandler: (@Sendable (Progress) -> Void)?
  ) async throws -> URL {
    let hub = createHubApi()

    do {
      let repo = Hub.Repo(id: modelId)
      return try await hub.snapshot(
        from: repo,
        revision: revision,
        matching: filePatterns,
        progressHandler: progressHandler ?? { _ in }
      )
    } catch Hub.HubClientError.authorizationRequired {
      throw ModelResolutionError.authorizationRequired(modelId)
    } catch {
      let nserror = error as NSError
      if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
        throw ModelResolutionError.networkUnavailable
      }
      throw ModelResolutionError.downloadFailed(modelId, error)
    }
  }
}
