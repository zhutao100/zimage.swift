import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import Hub
import Dispatch

public struct ZImageGenerationRequest: Sendable {
  public var prompt: String
  public var negativePrompt: String?
  public var width: Int
  public var height: Int
  public var steps: Int
  public var guidanceScale: Float
  public var seed: UInt64?
  public var outputPath: URL
  public var model: String?
  public var maxSequenceLength: Int

  public var lora: LoRAConfiguration?

  public var enhancePrompt: Bool

  public var enhanceMaxTokens: Int

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    width: Int = ZImageModelMetadata.recommendedWidth,
    height: Int = ZImageModelMetadata.recommendedHeight,
    steps: Int = ZImageModelMetadata.recommendedInferenceSteps,
    guidanceScale: Float = ZImageModelMetadata.recommendedGuidanceScale,
    seed: UInt64? = nil,
    outputPath: URL = URL(fileURLWithPath: "z-image.png"),
    model: String? = nil,
    maxSequenceLength: Int = 512,
    lora: LoRAConfiguration? = nil,
    enhancePrompt: Bool = false,
    enhanceMaxTokens: Int = 512
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
    self.maxSequenceLength = maxSequenceLength
    self.lora = lora
    self.enhancePrompt = enhancePrompt
    self.enhanceMaxTokens = enhanceMaxTokens
  }
}

public final class ZImagePipeline {
  public enum PipelineError: Error, Sendable {
    case notImplemented
    case tokenizerNotLoaded
    case invalidDimensions(String)
    case textEncoderNotLoaded
    case transformerNotLoaded
    case vaeNotLoaded
    case weightsMissing(String)
    case modelNotLoaded
    case loraError(LoRAError)
  }

  private var logger: Logger
  private let hubApi: HubApi
  private var tokenizer: QwenTokenizer?
  private var textEncoder: QwenTextEncoder?
  private var transformer: ZImageTransformer2DModel?
  private var vae: AutoencoderKL?
  private var modelConfigs: ZImageModelConfigs?
  private var quantManifest: ZImageQuantizationManifest?
  private var isModelLoaded: Bool = false
  private var loadedModelId: String?
  private var currentLoRA: LoRAWeights?
  private var currentLoRAConfig: LoRAConfiguration?
  private var modelSnapshot: URL?
  private var useDynamicLoRA: Bool = false

  public init(logger: Logger = Logger(label: "z-image.pipeline"), hubApi: HubApi = .shared) {
    self.logger = logger
    self.hubApi = hubApi
  }
  public var isLoaded: Bool {
    return isModelLoaded
  }
  public func unloadModel() {
    tokenizer = nil
    textEncoder = nil
    transformer = nil
    vae = nil
    modelConfigs = nil
    quantManifest = nil
    isModelLoaded = false
    loadedModelId = nil

    currentLoRA = nil
    currentLoRAConfig = nil
    modelSnapshot = nil
    useDynamicLoRA = false
    GPU.clearCache()
    logger.info("Model unloaded from memory")
  }

  public func unloadLoRA() {
    guard currentLoRA != nil else { return }

    if let trans = transformer {

      LoRAApplicator.clearDynamicLoRA(from: trans, logger: logger)
    }
    currentLoRA = nil
    currentLoRAConfig = nil
    useDynamicLoRA = false
    GPU.clearCache()
    logger.info("LoRA unloaded (instant)")
  }
  public func unloadTransformer() {
    transformer = nil

    currentLoRA = nil
    currentLoRAConfig = nil
    useDynamicLoRA = false

    GPU.clearCache()
    logger.info("Transformer unloaded for memory optimization")
  }

  private func getAvailableMemory() -> UInt64 {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &stats) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
      }
    }
    guard result == KERN_SUCCESS else { return 0 }
    let pageSize = UInt64(sysconf(_SC_PAGESIZE))
    return UInt64(stats.free_count) * pageSize
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

  private func encodePrompt(_ prompt: String, tokenizer: QwenTokenizer, textEncoder: QwenTextEncoder, maxLength: Int) throws -> (MLXArray, MLXArray) {
    do {
      let result = try PipelineUtilities.encodePrompt(prompt, tokenizer: tokenizer, textEncoder: textEncoder, maxLength: maxLength)
      return (result.embeddings, result.mask)
    } catch {
      throw PipelineError.textEncoderNotLoaded
    }
  }
  public struct GenerationProgress: Sendable {
    public let stage: Stage
    public let stepIndex: Int
    public let totalSteps: Int

    public enum Stage: String, Sendable {
      case loadingModel = "Loading model"
      case encodingText = "Encoding text"
      case loadingTransformer = "Loading transformer"
      case loadingLoRA = "Loading LoRA"
      case denoising = "Denoising"
      case loadingVAE = "Loading VAE"
      case decoding = "Decoding"
      case saving = "Saving"
    }

    public var fractionCompleted: Double {
      guard totalSteps > 0 else { return 0 }
      return Double(stepIndex) / Double(totalSteps)
    }

    public var percentComplete: Int {
      Int(fractionCompleted * 100)
    }
  }

  public typealias ProgressHandler = (GenerationProgress) -> Void
  public func loadModel(modelSpec: String? = nil, progressHandler: ProgressHandler? = nil) async throws {
    let modelId = modelSpec ?? ZImageRepository.id
    if isModelLoaded && loadedModelId == modelId {
      logger.info("Model already loaded, skipping load")
      return
    }
    let canPreserveSharedComponents = isModelLoaded
      && loadedModelId != modelId
      && areZImageVariants(loadedModelId ?? "", modelId)
    if isModelLoaded && loadedModelId != modelId {
      if canPreserveSharedComponents {
        logger.info("Switching Z-Image variant, preserving VAE and tokenizer")

        textEncoder = nil
        transformer = nil

        currentLoRA = nil
        currentLoRAConfig = nil
        useDynamicLoRA = false
      } else {
        logger.info("Different model requested, unloading current model")
        unloadModel()
      }
    }

    logger.info("Loading model: \(modelId)")
    progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))

    let snapshot = try await PipelineSnapshot.prepare(model: modelSpec, logger: logger)
    let configs = try ZImageModelConfigs.load(from: snapshot)
    let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, logger: logger)
    let manifest = weightsMapper.loadQuantizationManifest()

    if let m = manifest {
      logger.info("Loading quantized model (bits=\(m.bits), group_size=\(m.groupSize))")
    }
    if tokenizer == nil {
      progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
      logger.info("Loading tokenizer...")
      tokenizer = try loadTokenizer(snapshot: snapshot)
    } else {
      logger.info("Reusing cached tokenizer")
    }
    logger.info("Loading text encoder...")
    let te = try loadTextEncoder(snapshot: snapshot, config: configs.textEncoder)
    let textEncoderWeights = try weightsMapper.loadTextEncoder()
    ZImageWeightsMapping.applyTextEncoder(weights: textEncoderWeights, to: te, manifest: manifest, logger: logger)
    textEncoder = te
    progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
    logger.info("Loading transformer...")
    let trans = try loadTransformer(snapshot: snapshot, config: configs.transformer)
    let transformerWeights = try weightsMapper.loadTransformer()
    ZImageWeightsMapping.applyTransformer(weights: transformerWeights, to: trans, manifest: manifest, logger: logger)
    transformer = trans
    if vae == nil {
      progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
      logger.info("Loading VAE...")
      let v = try loadVAE(snapshot: snapshot, config: configs.vae)
      let vaeWeights = try weightsMapper.loadVAE()
      ZImageWeightsMapping.applyVAE(weights: vaeWeights, to: v, manifest: manifest, logger: logger)
      vae = v
    } else {
      logger.info("Reusing cached VAE")
    }

    modelConfigs = configs
    quantManifest = manifest
    modelSnapshot = snapshot
    isModelLoaded = true
    loadedModelId = modelId

    logger.info("Model loaded successfully and cached in memory")
  }
  public func loadLoRA(_ config: LoRAConfiguration, progressHandler: ProgressHandler? = nil) async throws {
    guard let trans = transformer else {
      throw PipelineError.transformerNotLoaded
    }
    if let currentConfig = currentLoRAConfig, currentConfig == config {
      logger.info("LoRA already loaded with same configuration, skipping")
      return
    }
    if currentLoRA != nil {
      logger.info("Unloading previous LoRA...")
      unloadLoRA()
    }

    progressHandler?(GenerationProgress(stage: .loadingLoRA, stepIndex: 0, totalSteps: 1))
    logger.info("Loading LoRA from \(config.source.displayName)...")

    do {

      let loraWeights = try await LoRAWeightLoader.load(from: config)
      logger.info("Loaded LoRA: rank=\(loraWeights.rank), alpha=\(loraWeights.alpha), layers=\(loraWeights.layerCount)")

      useDynamicLoRA = true
      LoRAApplicator.applyDynamically(to: trans, loraWeights: loraWeights, scale: config.scale, logger: logger)

      currentLoRA = loraWeights
      currentLoRAConfig = config

      logger.info("LoRA applied successfully with scale=\(config.scale)")
    } catch let error as LoRAError {
      throw PipelineError.loraError(error)
    }
  }
  public var hasLoRALoaded: Bool {
    return currentLoRA != nil
  }
  public var loadedLoRAConfig: LoRAConfiguration? {
    return currentLoRAConfig
  }

  public func generate(_ request: ZImageGenerationRequest, progressHandler: ProgressHandler? = nil) async throws -> URL {
    logger.info("Requested Z-Image generation")

    let decoded = try await generateCore(request, progressHandler: progressHandler)

    progressHandler?(GenerationProgress(stage: .saving, stepIndex: request.steps, totalSteps: request.steps))
    try QwenImageIO.saveImage(array: decoded, to: request.outputPath)
    logger.info("Wrote image to \(request.outputPath.path)")

    return request.outputPath
  }
  public func generateToMemory(_ request: ZImageGenerationRequest, progressHandler: ProgressHandler? = nil) async throws -> Data {
    logger.info("Requested Z-Image generation (to memory)")

    let decoded = try await generateCore(request, progressHandler: progressHandler)

    progressHandler?(GenerationProgress(stage: .saving, stepIndex: request.steps, totalSteps: request.steps))
    let imageData = try QwenImageIO.imageData(from: decoded)
    logger.info("Generated image data (\(imageData.count) bytes)")

    return imageData
  }
  private func generateCore(_ request: ZImageGenerationRequest, progressHandler: ProgressHandler? = nil) async throws -> MLXArray {

    let vaeScale = 16
    if request.width % vaeScale != 0 {
      throw PipelineError.invalidDimensions("Width must be divisible by \(vaeScale) (got \(request.width)). Please adjust to a multiple of \(vaeScale).")
    }
    if request.height % vaeScale != 0 {
      throw PipelineError.invalidDimensions("Height must be divisible by \(vaeScale) (got \(request.height)). Please adjust to a multiple of \(vaeScale).")
    }
    let requestedModelId = request.model ?? ZImageRepository.id
    if !isModelLoaded || loadedModelId != requestedModelId {
      try await loadModel(modelSpec: request.model, progressHandler: progressHandler)
    }

    guard let tokenizer = tokenizer,
          let textEncoder = textEncoder,
          let transformer = transformer,
          let vae = vae,
          let modelConfigs = modelConfigs else {
      throw PipelineError.modelNotLoaded
    }
    if let loraConfig = request.lora {

      if currentLoRAConfig != loraConfig {
        try await loadLoRA(loraConfig, progressHandler: progressHandler)
      }
    } else if currentLoRA != nil {

      unloadLoRA()
    }
    progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: request.steps))
    var finalPrompt = request.prompt
    if request.enhancePrompt {
      logger.info("Enhancing prompt using LLM (max tokens: \(request.enhanceMaxTokens))...")
      let enhanceConfig = PromptEnhanceConfig(
        maxNewTokens: request.enhanceMaxTokens,
        temperature: 0.7,
        topP: 0.9,
        repetitionPenalty: 1.05
      )
      let enhanced = try textEncoder.enhancePrompt(request.prompt, tokenizer: tokenizer, config: enhanceConfig)
      if enhanced.isEmpty {
        logger.warning("Prompt enhancement incomplete (need more tokens), using original prompt")
      } else {
        logger.info("Enhanced prompt: \(enhanced)")
        finalPrompt = enhanced
      }
      GPU.clearCache()
    }
    logger.info("Encoding prompts...")

    let promptEmbeds: MLXArray
    let negativeEmbeds: MLXArray?
    let doCFG = request.guidanceScale > 1.0

    let (pe, _) = try encodePrompt(finalPrompt, tokenizer: tokenizer, textEncoder: textEncoder, maxLength: request.maxSequenceLength)
    promptEmbeds = pe

    if doCFG {
      let (ne, _) = try encodePrompt(request.negativePrompt ?? "", tokenizer: tokenizer, textEncoder: textEncoder, maxLength: request.maxSequenceLength)
      negativeEmbeds = ne
      MLX.eval(promptEmbeds, ne)
    } else {
      negativeEmbeds = nil
      MLX.eval(promptEmbeds)
    }
    logger.info("Text encoding complete")

    let vaeDivisor = modelConfigs.vae.latentDivisor
    let latentH = max(1, request.height / vaeDivisor)
    let latentW = max(1, request.width / vaeDivisor)
    let shape: [Int] = [1, ZImageModelMetadata.Transformer.inChannels, latentH, latentW]
    let randomKey: RandomStateOrKey? = request.seed.map { MLXRandom.key($0) }
    var latents = MLXRandom.normal(shape, loc: 0, scale: 1, key: randomKey)

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

    logger.info("Running \(request.steps) denoising steps...")
    for stepIndex in 0..<request.steps {
      try Task.checkCancellation()
      progressHandler?(GenerationProgress(stage: .denoising, stepIndex: stepIndex, totalSteps: request.steps))
      let timestep = timestepsArray[stepIndex]
      let normalizedTimestep = (1000.0 - timestep) / 1000.0
      let timestepArray = MLXArray([normalizedTimestep], [1])

      var modelLatents = latents
      var embeds = promptEmbeds
      if doCFG, let ne = negativeEmbeds {
        modelLatents = MLX.concatenated([latents, latents], axis: 0)
        embeds = MLX.concatenated([promptEmbeds, ne], axis: 0)
      }

      let noisePred = transformer.forward(latents: modelLatents, timestep: timestepArray, promptEmbeds: embeds)
      var guidedNoise: MLXArray
      if doCFG, negativeEmbeds != nil {
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

    logger.info("Denoising complete, decoding with VAE...")
    progressHandler?(GenerationProgress(stage: .decoding, stepIndex: request.steps, totalSteps: request.steps))

    let decoded = decodeLatents(latents, vae: vae, height: request.height, width: request.width)
    MLX.eval(MLXArray([]))
    GPU.clearCache()

    return decoded
  }

  private func decodeLatents(_ latents: MLXArray, vae: AutoencoderKL, height: Int, width: Int) -> MLXArray {
    PipelineUtilities.decodeLatents(latents, vae: vae, height: height, width: width)
  }

  private func calculateShift(
    imageSeqLen: Int,
    baseSeqLen: Int,
    maxSeqLen: Int,
    baseShift: Float,
    maxShift: Float
  ) -> Float {
    PipelineUtilities.calculateShift(
      imageSeqLen: imageSeqLen,
      baseSeqLen: baseSeqLen,
      maxSeqLen: maxSeqLen,
      baseShift: baseShift,
      maxShift: maxShift
    )
  }

  private func areZImageVariants(_ model1: String, _ model2: String) -> Bool {
    let zImageIds: Set<String> = [
      "Tongyi-MAI/Z-Image-Turbo",
      "mzbac/Z-Image-Turbo-8bit"
    ]
    return zImageIds.contains(model1) && zImageIds.contains(model2)
  }

}
