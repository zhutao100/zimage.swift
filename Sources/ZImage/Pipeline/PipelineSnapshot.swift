import Foundation
import Logging

enum PipelineSnapshot {

  static let configAndTokenizerFilePatterns: [String] = [
    ZImageFiles.modelIndex,
    ZImageFiles.schedulerConfig,
    ZImageFiles.transformerConfig,
    ZImageFiles.textEncoderConfig,
    ZImageFiles.vaeConfig,
    "tokenizer/*",
  ]

  static let vaeOnlyFilePatterns: [String] = [
    ZImageFiles.modelIndex,
    ZImageFiles.vaeConfig,
    "vae/*.safetensors",
  ]

  static func prepare(
    model: String?,
    defaultModelId: String = ZImageRepository.id,
    defaultRevision: String = ZImageRepository.revision,
    filePatterns: [String]? = nil,
    logger: Logger
  ) async throws -> URL {
    let patterns = filePatterns ?? ["*.safetensors", "*.json", "tokenizer/*"]
    let requireWeights = patterns.contains(where: { $0.localizedCaseInsensitiveContains("safetensors") })
    let resolvedURL = try await ModelResolution.resolveOrDefault(
      modelSpec: model,
      defaultModelId: defaultModelId,
      defaultRevision: defaultRevision,
      filePatterns: patterns,
      requireWeights: requireWeights,
      progressHandler: { [logger] progress in
        let completed = progress.completedUnitCount
        let total = progress.totalUnitCount
        let percent = Int(progress.fractionCompleted * 100)
        logger.info("Downloading: \(completed)/\(total) files (\(percent)%)")
      }
    )

    logger.info("Using model at \(resolvedURL.path)")
    return resolvedURL
  }
}
