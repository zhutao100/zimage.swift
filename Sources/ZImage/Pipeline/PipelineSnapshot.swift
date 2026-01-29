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

  static func vaeOnlyFilePatterns(weightsVariant: String?) -> [String] {
    var patterns: [String] = [
      ZImageFiles.modelIndex,
      ZImageFiles.vaeConfig,
    ]

    if let weightsVariant, !weightsVariant.isEmpty {
      let variant = weightsVariant.trimmingCharacters(in: .whitespacesAndNewlines)
      if !variant.isEmpty {
        patterns.append("vae/*.\(variant).safetensors")
        patterns.append("vae/*\(variant)*.safetensors")
        patterns.append("vae/*.\(variant).safetensors.index.json")
        patterns.append("vae/*\(variant)*.safetensors.index.json")
        patterns.append("vae/*.safetensors.index.json")
        return patterns
      }
    }

    patterns.append("vae/*.safetensors")
    patterns.append("vae/*.safetensors.index.json")
    return patterns
  }

  static func modelFilePatterns(weightsVariant: String?) -> [String] {
    if let weightsVariant, !weightsVariant.isEmpty {
      let variant = weightsVariant.trimmingCharacters(in: .whitespacesAndNewlines)
      if !variant.isEmpty {
        return [
          "*.\(variant).safetensors",
          "*\(variant)*.safetensors",
          "*.\(variant).safetensors.index.json",
          "*\(variant)*.safetensors.index.json",
          "*.safetensors.index.json",
          "*.json",
          "tokenizer/*",
        ]
      }
    }

    return ["*.safetensors", "*.json", "tokenizer/*"]
  }

  static func prepare(
    model: String?,
    defaultModelId: String = ZImageRepository.id,
    defaultRevision: String = ZImageRepository.revision,
    weightsVariant: String? = nil,
    filePatterns: [String]? = nil,
    snapshotValidator: (@Sendable (URL) -> Bool)? = nil,
    logger: Logger
  ) async throws -> URL {
    let patterns = filePatterns ?? modelFilePatterns(weightsVariant: weightsVariant)
    let requireWeights = patterns.contains(where: { $0.localizedCaseInsensitiveContains("safetensors") })
    let resolvedURL = try await ModelResolution.resolveOrDefault(
      modelSpec: model,
      defaultModelId: defaultModelId,
      defaultRevision: defaultRevision,
      filePatterns: patterns,
      requireWeights: requireWeights,
      snapshotValidator: snapshotValidator,
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
