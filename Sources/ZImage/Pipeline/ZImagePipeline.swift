import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import Hub
import Dispatch

public struct ZImageGenerationRequest {
  public var prompt: String
  public var negativePrompt: String?
  public var width: Int
  public var height: Int
  public var steps: Int
  public var guidanceScale: Float
  public var seed: UInt64?
  public var outputPath: URL
  public var model: String?

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    width: Int = ZImageModelMetadata.recommendedWidth,
    height: Int = ZImageModelMetadata.recommendedHeight,
    steps: Int = ZImageModelMetadata.recommendedInferenceSteps,
    guidanceScale: Float = ZImageModelMetadata.recommendedGuidanceScale,
    seed: UInt64? = nil,
    outputPath: URL = URL(fileURLWithPath: "z-image.png"),
    model: String? = nil
  ) {
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.width = width
    self.height = height
    self.steps = steps
    self.guidanceScale = guidanceScale
    self.seed = seed
    self.outputPath = outputPath
    self.model = model
  }
}

public struct ZImagePipeline {
  public enum PipelineError: Error {
    case notImplemented
    case tokenizerNotLoaded
    case textEncoderNotLoaded
    case transformerNotLoaded
    case vaeNotLoaded
    case weightsMissing(String)
  }

  private var logger: Logger
  private let hubApi: HubApi
  private var tokenizer: QwenTokenizer?
  private var textEncoder: QwenTextEncoder?
  private var transformer: ZImageTransformer2DModel?
  private var vae: AutoencoderKL?

  public init(logger: Logger = Logger(label: "z-image.pipeline"), hubApi: HubApi = .shared) {
    self.logger = logger
    self.hubApi = hubApi
  }

  private func loadTokenizer(snapshot: URL) throws -> QwenTokenizer {
    let tokDir = snapshot.appending(path: "tokenizer")
    return try QwenTokenizer.load(from: tokDir, hubApi: hubApi)
  }

  private func loadTextEncoder(snapshot: URL, config: ZImageTextEncoderConfig) throws -> QwenTextEncoder {
    return QwenTextEncoder(
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

  private func loadTransformer(snapshot: URL, config: ZImageTransformerConfig) throws -> ZImageTransformer2DModel {
    return ZImageTransformer2DModel(configuration: config)
  }

  private func loadVAE(snapshot: URL, config: ZImageVAEConfig) throws -> AutoencoderKL {
    return AutoencoderKL(configuration: .init(
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
    ))
  }

  private func encodePrompt(_ prompt: String, tokenizer: QwenTokenizer, textEncoder: QwenTextEncoder, maxLength: Int = 2048) throws -> (MLXArray, MLXArray) {
    let encoded = try tokenizer.encodeChat(prompts: [prompt], maxLength: maxLength)
    let embeddingsList = textEncoder.encodeForZImage(inputIds: encoded.inputIds, attentionMask: encoded.attentionMask)

    guard let firstEmbeds = embeddingsList.first else {
      throw PipelineError.textEncoderNotLoaded
    }

    let embedsBatch = firstEmbeds.expandedDimensions(axis: 0)
    let mask = MLX.ones([1, firstEmbeds.dim(0)], dtype: .int32)

    return (embedsBatch, mask)
  }

  public func generate(_ request: ZImageGenerationRequest) async throws -> URL {
    logger.info("Requested Z-Image generation")

    let snapshot = try await prepareSnapshot(model: request.model)
    let modelConfigs = try ZImageModelConfigs.load(from: snapshot)
    let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, logger: logger)
    let quantManifest = weightsMapper.loadQuantizationManifest()

    if let manifest = quantManifest {
      logger.info("Loading quantized model (bits=\(manifest.bits), group_size=\(manifest.groupSize))")
    }

    // Phase 1: Load text encoder and encode prompts
    let promptEmbeds: MLXArray
    let negativeEmbeds: MLXArray

    let usePythonEmbeds = ProcessInfo.processInfo.environment["USE_PYTHON_EMBEDS"] != nil
    if usePythonEmbeds {
      var pe = try loadNumpyArray(from: "/tmp/python_prompt_embeds.npy")
      var ne = try loadNumpyArray(from: "/tmp/python_negative_embeds.npy")
      if pe.ndim == 2 { pe = pe.expandedDimensions(axis: 0) }
      if ne.ndim == 2 { ne = ne.expandedDimensions(axis: 0) }
      promptEmbeds = pe
      negativeEmbeds = ne
    } else {
      logger.info("Loading text encoder...")
      let tokenizer = try loadTokenizer(snapshot: snapshot)
      let textEncoder = try loadTextEncoder(snapshot: snapshot, config: modelConfigs.textEncoder)
      let textEncoderWeights = try weightsMapper.loadTextEncoder()
      ZImageWeightsMapping.applyTextEncoder(weights: textEncoderWeights, to: textEncoder, manifest: quantManifest, logger: logger)

      let (pe, _) = try encodePrompt(request.prompt, tokenizer: tokenizer, textEncoder: textEncoder)
      let (ne, _) = try encodePrompt(request.negativePrompt ?? "", tokenizer: tokenizer, textEncoder: textEncoder)
      promptEmbeds = pe
      negativeEmbeds = ne

      MLX.eval(promptEmbeds, negativeEmbeds)
      logger.info("Text encoding complete, clearing text encoder from memory")
    }
    GPU.clearCache()

    // Phase 2: Load transformer and run denoising loop
    logger.info("Loading transformer...")
    let transformer = try loadTransformer(snapshot: snapshot, config: modelConfigs.transformer)
    let transformerWeights = try weightsMapper.loadTransformer()
    ZImageWeightsMapping.applyTransformer(weights: transformerWeights, to: transformer, manifest: quantManifest, logger: logger)

    let vaeDivisor = modelConfigs.vae.latentDivisor
    let latentH = max(1, request.height / vaeDivisor)
    let latentW = max(1, request.width / vaeDivisor)
    let shape: [Int] = [1, ZImageModelMetadata.Transformer.inChannels, latentH, latentW]

    var latents: MLXArray
    if usePythonEmbeds, FileManager.default.fileExists(atPath: "/tmp/python_initial_latents.npy") {
      latents = try loadNumpyArray(from: "/tmp/python_initial_latents.npy")
    } else {
      let randomKey: RandomStateOrKey? = request.seed.map { MLXRandom.key($0) }
      latents = MLXRandom.normal(shape, loc: 0, scale: 1, key: randomKey)
    }

    let mu = calculateShift(
      imageSeqLen: latentH * latentW,
      baseSeqLen: modelConfigs.scheduler.baseImageSeqLen ?? 256,
      maxSeqLen: modelConfigs.scheduler.maxImageSeqLen ?? 4096,
      baseShift: modelConfigs.scheduler.baseShift ?? 0.5,
      maxShift: modelConfigs.scheduler.maxShift ?? 1.15
    )

    let scheduler = FlowMatchEulerScheduler(
      numInferenceSteps: request.steps,
      config: modelConfigs.scheduler,
      mu: modelConfigs.scheduler.useDynamicShifting ? mu : nil
    )

    let timestepsArray = scheduler.timesteps.asArray(Float.self)
    let doCFG = request.guidanceScale > 1.0

    logger.info("Running \(request.steps) denoising steps...")
    for stepIndex in 0..<request.steps {
      let timestep = timestepsArray[stepIndex]
      let normalizedTimestep = (1000.0 - timestep) / 1000.0
      let timestepArray = MLXArray([normalizedTimestep], [1])

      var modelLatents = latents
      var embeds = promptEmbeds
      if doCFG {
        modelLatents = MLX.concatenated([latents, latents], axis: 0)
        embeds = MLX.concatenated([promptEmbeds, negativeEmbeds], axis: 0)
      }

      let noisePred = transformer.forward(latents: modelLatents, timestep: timestepArray, promptEmbeds: embeds)
      var guidedNoise: MLXArray
      if doCFG {
        let batch = latents.dim(0)
        let positive = noisePred[0 ..< batch, 0..., 0..., 0...]
        let negative = noisePred[batch ..< batch * 2, 0..., 0..., 0...]
        guidedNoise = positive + request.guidanceScale * (positive - negative)
      } else {
        guidedNoise = noisePred
      }

      guidedNoise = -guidedNoise
      latents = scheduler.step(modelOutput: guidedNoise, timestepIndex: stepIndex, sample: latents)
      MLX.eval(latents)
    }

    // Phase 3: Clear transformer and load VAE for decoding
    logger.info("Denoising complete, loading VAE...")
    GPU.clearCache()

    let vae = try loadVAE(snapshot: snapshot, config: modelConfigs.vae)
    let vaeWeights = try weightsMapper.loadVAE()
    ZImageWeightsMapping.applyVAE(weights: vaeWeights, to: vae, manifest: quantManifest, logger: logger)

    let decoded = decodeLatents(latents, vae: vae, height: request.height, width: request.width)
    try QwenImageIO.saveImage(array: decoded, to: request.outputPath)
    logger.info("Wrote image to \(request.outputPath.path)")
    return request.outputPath
  }

  private func decodeLatents(_ latents: MLXArray, vae: AutoencoderKL, height: Int, width: Int) -> MLXArray {
    let (decoded, _) = vae.decode(latents)
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

  private func prepareSnapshot(model: String? = nil) async throws -> URL {
    let filePatterns = [
      "*.json",
      "*.safetensors",
      "tokenizer/*"
    ]

    let resolvedURL = try await ModelResolution.resolveOrDefault(
      modelSpec: model,
      defaultModelId: ZImageRepository.id,
      defaultRevision: ZImageRepository.revision,
      filePatterns: filePatterns,
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

  private func loadNumpyArray(from path: String) throws -> MLXArray {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)

    guard data.count > 10 else {
      throw PipelineError.weightsMissing("Invalid numpy file: \(path)")
    }

    let headerLen = Int(data[8]) | (Int(data[9]) << 8)
    let headerStart = 10
    let headerEnd = headerStart + headerLen

    let headerData = data[headerStart..<headerEnd]
    let headerStr = String(data: headerData, encoding: .ascii) ?? ""

    var shape: [Int] = []
    if let shapeStart = headerStr.range(of: "'shape': ("),
       let shapeEnd = headerStr.range(of: ")", range: shapeStart.upperBound..<headerStr.endIndex) {
      let shapeStr = String(headerStr[shapeStart.upperBound..<shapeEnd.lowerBound])
      shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    let dataStart = headerEnd
    let floatCount = (data.count - dataStart) / 4

    var floats = [Float](repeating: 0, count: floatCount)
    data.withUnsafeBytes { ptr in
      let floatPtr = ptr.baseAddress!.advanced(by: dataStart).assumingMemoryBound(to: Float.self)
      for i in 0..<floatCount {
        floats[i] = floatPtr[i]
      }
    }

    if shape.isEmpty {
      let hiddenDim = 2560
      let seqLen = floatCount / hiddenDim
      shape = [seqLen, hiddenDim]
    }

    return MLXArray(floats, shape)
  }

  private func calculateShift(
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

}
