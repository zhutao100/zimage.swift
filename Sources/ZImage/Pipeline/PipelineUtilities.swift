import Foundation
import Logging
import MLX
import MLXNN

public enum PipelineUtilities {
  public enum UtilityError: Error {
    @available(*, deprecated, message: "No longer thrown by PipelineUtilities; kept for source compatibility.")
    case textEncoderFailed
    case emptyEmbeddings
  }

  public enum StabilityError: Error, LocalizedError, Sendable {
    case nonFiniteTensor(String)
    case excessiveTensorMagnitude(name: String, maxAbs: Float, threshold: Float)
    case fullyClippedImage(name: String, min: Float, max: Float)

    public var errorDescription: String? {
      switch self {
      case .nonFiniteTensor(let name):
        return "Non-finite values detected in \(name)."
      case .excessiveTensorMagnitude(let name, let maxAbs, let threshold):
        return
          "Numerical instability detected in \(name): abs max \(maxAbs) exceeded threshold \(threshold)."
      case .fullyClippedImage(let name, let min, let max):
        return
          "Decoded \(name) collapsed fully outside the displayable range before clipping (min=\(min), max=\(max))."
      }
    }
  }

  static let defaultTensorAbsMaxThreshold: Float = 10_000

  struct StandardSnapshotContext {
    let snapshot: URL
    let configs: ZImageModelConfigs
    let weightsMapper: ZImageWeightsMapper
    let quantizationManifest: ZImageQuantizationManifest?
    let weightsVariant: String?
  }

  static func makeTokenizer(from snapshot: URL) throws -> QwenTokenizer {
    try QwenTokenizer.load(from: snapshot.appending(path: "tokenizer"))
  }

  static func makeTextEncoder(config: ZImageTextEncoderConfig) -> QwenTextEncoder {
    QwenTextEncoder(
      configuration: .init(
        vocabSize: config.vocabSize,
        hiddenSize: config.hiddenSize,
        numHiddenLayers: config.numHiddenLayers,
        numAttentionHeads: config.numAttentionHeads,
        numKeyValueHeads: config.numKeyValueHeads,
        intermediateSize: config.intermediateSize,
        ropeTheta: config.ropeTheta,
        maxPositionEmbeddings: config.maxPositionEmbeddings,
        rmsNormEps: config.rmsNormEps,
        headDim: config.headDim
      )
    )
  }

  static func makeTransformer(config: ZImageTransformerConfig) -> ZImageTransformer2DModel {
    ZImageTransformer2DModel(configuration: config)
  }

  static func makeVAEConfiguration(from config: ZImageVAEConfig) -> VAEConfig {
    .init(
      inChannels: config.inChannels,
      outChannels: config.outChannels,
      latentChannels: config.latentChannels,
      scalingFactor: config.scalingFactor,
      shiftFactor: config.shiftFactor,
      blockOutChannels: config.blockOutChannels,
      layersPerBlock: config.layersPerBlock,
      normNumGroups: config.normNumGroups,
      sampleSize: config.sampleSize,
      midBlockAddAttention: config.midBlockAddAttention
    )
  }

  static func makeVAEEncoder(config: ZImageVAEConfig) -> AutoencoderEncoderOnly {
    AutoencoderEncoderOnly(configuration: makeVAEConfiguration(from: config))
  }

  static func makeVAEDecoder(config: ZImageVAEConfig) -> AutoencoderDecoderOnly {
    AutoencoderDecoderOnly(configuration: makeVAEConfiguration(from: config))
  }

  static func standardSnapshotValidator(weightsVariant: String?) -> (@Sendable (URL) -> Bool)? {
    guard let weightsVariant = ZImageFiles.normalizedWeightsVariant(weightsVariant) else {
      return nil
    }

    return { snapshot in
      !ZImageFiles.resolveTransformerWeights(at: snapshot, weightsVariant: weightsVariant).isEmpty
        && !ZImageFiles.resolveTextEncoderWeights(at: snapshot, weightsVariant: weightsVariant).isEmpty
        && !ZImageFiles.resolveVAEWeights(at: snapshot, weightsVariant: weightsVariant).isEmpty
    }
  }

  static func prepareStandardSnapshot(
    model: String?,
    weightsVariant: String?,
    logger: Logger
  ) async throws -> StandardSnapshotContext {
    let normalizedWeightsVariant = ZImageFiles.normalizedWeightsVariant(weightsVariant)
    let snapshot = try await PipelineSnapshot.prepare(
      model: model,
      weightsVariant: normalizedWeightsVariant,
      snapshotValidator: standardSnapshotValidator(weightsVariant: normalizedWeightsVariant),
      logger: logger
    )
    let configs = try ZImageModelConfigs.load(from: snapshot)
    let weightsMapper = ZImageWeightsMapper(
      snapshot: snapshot,
      weightsVariant: normalizedWeightsVariant,
      logger: logger
    )
    let quantizationManifest = weightsMapper.loadQuantizationManifest()
    if quantizationManifest == nil {
      try ZImageFiles.validateRequiredComponentWeights(
        at: snapshot,
        weightsVariant: normalizedWeightsVariant,
        logger: logger
      )
    }

    return StandardSnapshotContext(
      snapshot: snapshot,
      configs: configs,
      weightsMapper: weightsMapper,
      quantizationManifest: quantizationManifest,
      weightsVariant: normalizedWeightsVariant
    )
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

  public static func encodePromptPair(
    prompt: String,
    negativePrompt: String,
    tokenizer: QwenTokenizer,
    textEncoder: QwenTextEncoder,
    maxLength: Int
  ) throws -> (promptEmbeddings: MLXArray, negativeEmbeddings: MLXArray) {
    let encoded = try tokenizer.encodeChat(prompts: [prompt, negativePrompt], maxLength: maxLength)
    let embeddingsList = textEncoder.encodeForZImage(
      inputIds: encoded.inputIds,
      attentionMask: encoded.attentionMask
    )

    guard embeddingsList.count == 2 else {
      throw UtilityError.emptyEmbeddings
    }

    return (
      promptEmbeddings: embeddingsList[0].expandedDimensions(axis: 0),
      negativeEmbeddings: embeddingsList[1].expandedDimensions(axis: 0)
    )
  }

  public static func decodeLatents(
    _ latents: MLXArray,
    vae: VAEImageDecoding,
    height: Int,
    width: Int
  ) throws -> MLXArray {
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
    try validateDisplayImageRange(image, name: "image")
    return MLX.clip(image, min: 0, max: 1)
  }

  static func validateTensorStability(
    _ tensor: MLXArray,
    name: String,
    maxAbsThreshold: Float = defaultTensorAbsMaxThreshold
  ) throws {
    guard isFinite(tensor).all().item(Bool.self) else {
      throw StabilityError.nonFiniteTensor(name)
    }

    let maxAbs = abs(tensor).max().item(Float.self)
    guard maxAbs.isFinite else {
      throw StabilityError.nonFiniteTensor(name)
    }
    guard maxAbs <= maxAbsThreshold else {
      throw StabilityError.excessiveTensorMagnitude(
        name: name,
        maxAbs: maxAbs,
        threshold: maxAbsThreshold
      )
    }
  }

  static func validateDisplayImageRange(_ image: MLXArray, name: String) throws {
    guard isFinite(image).all().item(Bool.self) else {
      throw StabilityError.nonFiniteTensor(name)
    }

    let minValue = image.min().item(Float.self)
    let maxValue = image.max().item(Float.self)

    guard minValue.isFinite, maxValue.isFinite else {
      throw StabilityError.nonFiniteTensor(name)
    }
    if maxValue <= 0 || minValue >= 1 {
      throw StabilityError.fullyClippedImage(name: name, min: minValue, max: maxValue)
    }
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

  static func usesClassifierFreeGuidance(guidanceScale: Float) -> Bool {
    guidanceScale > 0
  }

  static func effectiveGuidanceScale(
    guidanceScale: Float,
    normalizedTimestep: Float,
    cfgTruncation: Float
  ) -> Float {
    guard usesClassifierFreeGuidance(guidanceScale: guidanceScale) else { return 0 }
    guard cfgTruncation <= 1, normalizedTimestep > cfgTruncation else {
      return guidanceScale
    }
    return 0
  }

  static func guidedNoisePrediction(
    positive: MLXArray,
    negative: MLXArray,
    guidanceScale: Float,
    cfgNormalization: Bool
  ) -> MLXArray {
    var prediction = add(positive, multiply(guidanceScale, subtract(positive, negative)))
    guard cfgNormalization else { return prediction }

    let positiveNorm = l2Norm(positive)
    let predictionNorm = l2Norm(prediction)
    let normalizationScale = cfgNormalizationScale(
      positiveNorm: positiveNorm,
      predictionNorm: predictionNorm,
      cfgNormalization: cfgNormalization
    )
    guard normalizationScale < 1 else {
      return prediction
    }

    prediction = multiply(normalizationScale, prediction)
    return prediction
  }

  static func cfgNormalizationScale(
    positiveNorm: Float,
    predictionNorm: Float,
    cfgNormalization: Bool
  ) -> Float {
    guard cfgNormalization, positiveNorm > 0, predictionNorm > positiveNorm else {
      return 1.0
    }
    return positiveNorm / predictionNorm
  }

  static func runtimeDType(for module: Module) -> DType? {
    for (_, parameter) in module.parameters().flattened() {
      switch parameter.dtype {
      case .float16, .bfloat16, .float32, .float64:
        return parameter.dtype
      default:
        continue
      }
    }
    return nil
  }

  static func castModelInputToRuntimeDTypeIfNeeded(_ input: MLXArray, module: Module) -> MLXArray {
    guard let dtype = runtimeDType(for: module), input.dtype != dtype else {
      return input
    }
    return input.asType(dtype)
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
    return try await ModelResolution.resolveOrDefault(
      modelSpec: model,
      defaultModelId: defaultModelId,
      defaultRevision: defaultRevision,
      filePatterns: patterns,
      requireWeights: requireWeights,
      snapshotValidator: standardSnapshotValidator(weightsVariant: normalizedWeightsVariant),
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

  private static func l2Norm(_ array: MLXArray) -> Float {
    sum(square(array.asType(.float32))).item(Float.self).squareRoot()
  }
}
