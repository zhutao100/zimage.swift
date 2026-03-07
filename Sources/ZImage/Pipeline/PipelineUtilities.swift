import Foundation
import MLX
import MLXNN

public enum PipelineUtilities {
  public enum UtilityError: Error {
    @available(*, deprecated, message: "No longer thrown by PipelineUtilities; kept for source compatibility.")
    case textEncoderFailed
    case emptyEmbeddings
  }

  public static func encodePrompt(
    _ prompt: String,
    tokenizer: QwenTokenizer,
    textEncoder: QwenTextEncoder,
    maxLength: Int
  ) throws -> (embeddings: MLXArray, mask: MLXArray) {
    let encoded = try tokenizer.encodeChat(prompts: [prompt], maxLength: maxLength)
    let embeddingsList = textEncoder.encodeForZImage(
      inputIds: encoded.inputIds,
      attentionMask: encoded.attentionMask
    )

    guard let firstEmbeds = embeddingsList.first else {
      throw UtilityError.emptyEmbeddings
    }

    let embedsBatch = firstEmbeds.expandedDimensions(axis: 0)
    let mask = MLX.ones([1, firstEmbeds.dim(0)], dtype: .int32)

    return (embedsBatch, mask)
  }

  public static func decodeLatents(
    _ latents: MLXArray,
    vae: VAEImageDecoding,
    height: Int,
    width: Int
  ) -> MLXArray {
    let input: MLXArray =
      if latents.dtype == vae.dtype {
        latents
      } else {
        latents.asType(vae.dtype)
      }

    let (decoded, _) = vae.decode(input, return_dict: false)
    var image = decoded
    if height != decoded.dim(2) || width != decoded.dim(3) {
      var nhwc = image.transposed(0, 2, 3, 1)
      let hScale = Float(height) / Float(decoded.dim(2))
      let wScale = Float(width) / Float(decoded.dim(3))
      nhwc = MLXNN.Upsample(scaleFactor: .array([hScale, wScale]), mode: .nearest)(nhwc)
      image = nhwc.transposed(0, 3, 1, 2)
    }

    image = QwenImageIO.denormalizeFromDecoder(image)
    return MLX.clip(image, min: 0, max: 1)
  }

  public static func calculateShift(
    imageSeqLen: Int,
    baseSeqLen: Int,
    maxSeqLen: Int,
    baseShift: Float,
    maxShift: Float
  ) -> Float {
    let m = (maxShift - baseShift) / Float(maxSeqLen - baseSeqLen)
    let b = baseShift - m * Float(baseSeqLen)
    return Float(imageSeqLen) * m + b
  }

  @available(*, deprecated, message: "Use ZImagePipeline.loadModel / PipelineSnapshot.prepare instead.")
  public static func prepareSnapshot(
    model: String?,
    defaultModelId: String,
    defaultRevision: String,
    weightsVariant: String? = nil,
    progressHandler: (@Sendable (Progress) -> Void)? = nil
  ) async throws -> URL {
    let normalizedWeightsVariant = ZImageFiles.normalizedWeightsVariant(weightsVariant)
    let patterns = PipelineSnapshot.modelFilePatterns(weightsVariant: normalizedWeightsVariant)
    let requireWeights = patterns.contains(where: { $0.localizedCaseInsensitiveContains("safetensors") })
    let snapshotValidator: (@Sendable (URL) -> Bool)? =
      if let normalizedWeightsVariant {
        { snapshot in
          !ZImageFiles.resolveTransformerWeights(at: snapshot, weightsVariant: normalizedWeightsVariant).isEmpty
            && !ZImageFiles.resolveTextEncoderWeights(at: snapshot, weightsVariant: normalizedWeightsVariant).isEmpty
            && !ZImageFiles.resolveVAEWeights(at: snapshot, weightsVariant: normalizedWeightsVariant).isEmpty
        }
      } else {
        nil
      }
    return try await ModelResolution.resolveOrDefault(
      modelSpec: model,
      defaultModelId: defaultModelId,
      defaultRevision: defaultRevision,
      filePatterns: patterns,
      requireWeights: requireWeights,
      snapshotValidator: snapshotValidator,
      progressHandler: progressHandler
    )
  }

  @available(*, deprecated, message: "Pipelines validate dimensions internally; kept for source compatibility.")
  public static func validateDimensions(
    width: Int,
    height: Int,
    vaeScale: Int = 16
  ) throws {
    if width % vaeScale != 0 {
      throw DimensionError.widthNotDivisible(width: width, scale: vaeScale)
    }
    if height % vaeScale != 0 {
      throw DimensionError.heightNotDivisible(height: height, scale: vaeScale)
    }
  }

  @available(*, deprecated, message: "Deprecated with validateDimensions.")
  public enum DimensionError: Error, LocalizedError {
    case widthNotDivisible(width: Int, scale: Int)
    case heightNotDivisible(height: Int, scale: Int)

    public var errorDescription: String? {
      switch self {
      case .widthNotDivisible(let width, let scale):
        "Width must be divisible by \(scale) (got \(width)). Please adjust to a multiple of \(scale)."
      case .heightNotDivisible(let height, let scale):
        "Height must be divisible by \(scale) (got \(height)). Please adjust to a multiple of \(scale)."
      }
    }
  }
}
