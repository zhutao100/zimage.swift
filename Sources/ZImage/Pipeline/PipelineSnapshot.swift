import Foundation
import Logging

enum PipelineSnapshot {

  static func prepare(
    model: String?,
    defaultModelId: String = ZImageRepository.id,
    defaultRevision: String = ZImageRepository.revision,
    logger: Logger
  ) async throws -> URL {
    let resolvedURL = try await ModelResolution.resolveOrDefault(
      modelSpec: model,
      defaultModelId: defaultModelId,
      defaultRevision: defaultRevision,
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
