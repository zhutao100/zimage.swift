import Foundation

public enum ZImageKnownModel: String, CaseIterable, Sendable {
  case zImageTurbo = "Tongyi-MAI/Z-Image-Turbo"
  case zImage = "Tongyi-MAI/Z-Image"
  case zImageTurbo8bit = "mzbac/z-image-turbo-8bit"

  public var id: String { rawValue }
}

public enum ZImageModelRegistry {
  public static func normalizedModelId(from modelSpec: String) -> String {
    let trimmed = modelSpec.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.split(separator: ":", maxSplits: 1).first.map(String.init) ?? trimmed
  }

  public static func areZImageVariants(_ model1: String, _ model2: String) -> Bool {
    isZImageVariant(model1) && isZImageVariant(model2)
  }

  private static func isZImageVariant(_ modelSpec: String) -> Bool {
    let normalized = normalizedModelId(from: modelSpec)
    if normalized == ZImageKnownModel.zImageTurbo.id { return true }
    if normalized == ZImageKnownModel.zImage.id { return true }
    if normalized == ZImageKnownModel.zImageTurbo8bit.id { return true }
    if normalized == "mzbac/Z-Image-Turbo-8bit" { return true }  // common alternative capitalization
    return false
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
    guard let modelSpec else { return defaults(for: ZImageKnownModel.zImageTurbo.id) }
    return defaults(for: modelSpec)
  }

  public static func defaults(for modelId: String) -> ZImagePreset {
    switch ZImageModelRegistry.normalizedModelId(from: modelId) {
    case ZImageKnownModel.zImage.id:
      return .zImage
    case ZImageKnownModel.zImageTurbo.id:
      return .zImageTurbo
    case ZImageKnownModel.zImageTurbo8bit.id, "mzbac/Z-Image-Turbo-8bit":
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
