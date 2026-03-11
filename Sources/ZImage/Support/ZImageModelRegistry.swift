import Foundation

public enum ZImageKnownModel: String, CaseIterable, Sendable {
  case zImageTurbo = "Tongyi-MAI/Z-Image-Turbo"
  case zImage = "Tongyi-MAI/Z-Image"
  case zImageTurbo8bit = "mzbac/z-image-turbo-8bit"

  public var id: String { rawValue }
}

public enum ZImageModelRegistry {
  private struct ModelIndexMetadata: Decodable {
    let modelType: String?

    private enum CodingKeys: String, CodingKey {
      case modelType = "model_type"
    }
  }

  private struct SchedulerMetadata: Decodable {
    let shift: Float
  }

  public static func normalizedModelId(from modelSpec: String) -> String {
    let trimmed = modelSpec.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.split(separator: ":", maxSplits: 1).first.map(String.init) ?? trimmed
  }

  public static func areZImageVariants(_ model1: String, _ model2: String) -> Bool {
    isZImageVariant(model1) && isZImageVariant(model2)
  }

  private static func isZImageVariant(_ modelSpec: String) -> Bool {
    if knownModel(for: modelSpec) != nil { return true }
    if detectedKnownModel(for: modelSpec) != nil { return true }
    return false
  }

  static func detectedKnownModel(for modelSpec: String?) -> ZImageKnownModel? {
    guard let modelSpec else { return .zImageTurbo }
    if let known = knownModel(for: modelSpec) {
      return known
    }
    if let inspected = inspectedKnownModel(for: modelSpec) {
      return inspected
    }
    return heuristicKnownModel(for: modelSpec)
  }

  static func knownModel(for modelSpec: String) -> ZImageKnownModel? {
    let normalized = normalizedModelId(from: modelSpec)
    if normalized == ZImageKnownModel.zImageTurbo.id { return .zImageTurbo }
    if normalized == ZImageKnownModel.zImage.id { return .zImage }
    if normalized == ZImageKnownModel.zImageTurbo8bit.id { return .zImageTurbo8bit }
    if normalized == "mzbac/Z-Image-Turbo-8bit" { return .zImageTurbo8bit }  // common alternative capitalization
    return nil
  }

  private static func inspectedKnownModel(for modelSpec: String) -> ZImageKnownModel? {
    for directory in candidateSnapshotDirectories(for: modelSpec) {
      if let inspected = inspectSnapshotDirectory(directory) {
        return inspected
      }
    }

    let normalized = normalizedModelId(from: modelSpec)
    guard ModelResolution.isHuggingFaceModelId(normalized) else { return nil }

    let parts = normalized.split(separator: ":", maxSplits: 1)
    let modelId = String(parts[0])
    let revision = parts.count > 1 ? String(parts[1]) : ZImageRepository.revision
    guard let cachedSnapshot = ModelResolution.findCachedModel(
      modelId: modelId,
      revision: revision,
      requireWeights: false
    ) else {
      return nil
    }
    return inspectSnapshotDirectory(cachedSnapshot)
  }

  private static func candidateSnapshotDirectories(for modelSpec: String) -> [URL] {
    let expandedPath = ModelResolution.expandedLocalPath(from: modelSpec)
    let url = URL(fileURLWithPath: expandedPath).standardizedFileURL
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
      return []
    }

    if isDirectory.boolValue {
      return [url]
    }

    let parent = url.deletingLastPathComponent()
    let grandparent = parent.deletingLastPathComponent()
    var directories: [URL] = []
    for directory in [parent, grandparent] {
      let path = directory.standardizedFileURL.path
      if !directories.contains(where: { $0.standardizedFileURL.path == path }) {
        directories.append(directory)
      }
    }
    return directories
  }

  private static func inspectSnapshotDirectory(_ directory: URL) -> ZImageKnownModel? {
    let modelIndexURL = directory.appending(path: ZImageFiles.modelIndex)
    if let data = try? Data(contentsOf: modelIndexURL),
      let metadata = try? JSONDecoder().decode(ModelIndexMetadata.self, from: data),
      let modelType = metadata.modelType?.lowercased()
    {
      switch modelType {
      case "z-image":
        return .zImage
      case "z-image-turbo":
        return .zImageTurbo
      default:
        break
      }
    }

    let schedulerURL = directory.appending(path: ZImageFiles.schedulerConfig)
    if let data = try? Data(contentsOf: schedulerURL),
      let metadata = try? JSONDecoder().decode(SchedulerMetadata.self, from: data)
    {
      if metadata.shift >= 5.0 {
        return .zImage
      }
      if metadata.shift <= 3.5 {
        return .zImageTurbo
      }
    }

    return nil
  }

  private static func heuristicKnownModel(for modelSpec: String) -> ZImageKnownModel? {
    let normalized = normalizedModelId(from: modelSpec).lowercased().replacingOccurrences(of: "_", with: "-")
    guard normalized.contains("z-image") || normalized.contains("zimage") else {
      return nil
    }
    if normalized.contains("turbo") || normalized.contains("8bit") {
      return .zImageTurbo
    }
    return .zImage
  }
}

public struct ZImagePreset: Sendable, Equatable {
  public let width: Int
  public let height: Int
  public let steps: Int
  public let guidanceScale: Float
  public let maxSequenceLength: Int
  public let negativePrompt: String?

  public static func defaults(for modelSpec: String?) -> ZImagePreset {
    guard let knownModel = ZImageModelRegistry.detectedKnownModel(for: modelSpec) else {
      return .zImageTurbo
    }
    return defaults(for: knownModel.id)
  }

  public static func defaults(for modelId: String) -> ZImagePreset {
    switch ZImageModelRegistry.knownModel(for: modelId) ?? ZImageModelRegistry.detectedKnownModel(for: modelId) {
    case .zImage?:
      return .zImage
    case .zImageTurbo?, .zImageTurbo8bit?:
      return .zImageTurbo
    default:
      return .zImageTurbo
    }
  }

  public func applying(
    width: Int? = nil,
    height: Int? = nil,
    steps: Int? = nil,
    guidanceScale: Float? = nil,
    maxSequenceLength: Int? = nil
  ) -> ZImagePreset {
    ZImagePreset(
      width: width ?? self.width,
      height: height ?? self.height,
      steps: steps ?? self.steps,
      guidanceScale: guidanceScale ?? self.guidanceScale,
      maxSequenceLength: maxSequenceLength ?? self.maxSequenceLength,
      negativePrompt: negativePrompt
    )
  }

  public static func resolved(
    for modelSpec: String?,
    width: Int? = nil,
    height: Int? = nil,
    steps: Int? = nil,
    guidanceScale: Float? = nil,
    maxSequenceLength: Int? = nil
  ) -> ZImagePreset {
    defaults(for: modelSpec).applying(
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidanceScale,
      maxSequenceLength: maxSequenceLength
    )
  }

  public static let zImageTurbo = ZImagePreset(
    width: ZImageModelMetadata.recommendedWidth,
    height: ZImageModelMetadata.recommendedHeight,
    steps: ZImageModelMetadata.recommendedInferenceSteps,
    guidanceScale: ZImageModelMetadata.recommendedGuidanceScale,
    maxSequenceLength: 512,
    negativePrompt: nil
  )

  public static let zImage = ZImagePreset(
    width: 1024,
    height: 1024,
    steps: 50,
    guidanceScale: 4.0,
    maxSequenceLength: 512,
    negativePrompt: nil
  )
}
