import Dispatch
import Foundation
import Hub
import Logging
import MLX
import MLXNN
import MLXRandom
import Tokenizers

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
  public var forceTransformerOverrideOnly: Bool

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
    enhanceMaxTokens: Int = 512,
    forceTransformerOverrideOnly: Bool = false
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
    self.forceTransformerOverrideOnly = forceTransformerOverrideOnly
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
  private var vae: VAEImageDecoding?
  private var modelConfigs: ZImageModelConfigs?
  private var quantManifest: ZImageQuantizationManifest?
  private var isModelLoaded: Bool = false
  private var loadedModelId: String?
  private var currentLoRA: LoRAWeights?
  private var currentLoRAConfig: LoRAConfiguration?
  private var modelSnapshot: URL?
  private var useDynamicLoRA: Bool = false
  private var activeTransformerOverrideURL: URL?
  private var activeAIOCheckpointURL: URL?

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
    activeTransformerOverrideURL = nil
    activeAIOCheckpointURL = nil
    GPU.clearCache()
    logger.info("Model unloaded from memory")
  }

  public func unloadLoRA() {
    guard currentLoRA != nil else { return }

    if let trans = transformer {
      if let lora = currentLoRA, let config = currentLoRAConfig, lora.hasLoKr {
        LoRAApplicator.removeLoKr(from: trans, loraWeights: lora, scale: config.scale, logger: logger)
      }

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
    activeTransformerOverrideURL = nil

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

  private func loadTextEncoder(snapshot _: URL, config: ZImageTextEncoderConfig) throws -> QwenTextEncoder {
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

  private func loadTransformer(snapshot _: URL, config: ZImageTransformerConfig) throws -> ZImageTransformer2DModel {
    return ZImageTransformer2DModel(configuration: config)
  }

  private func auditModuleWeightShapeMismatches(
    module: Module,
    weights: [String: MLXArray],
    transpose4DTensors: Bool,
    logger: Logger,
    sample: Int = 10
  ) -> [String] {
    let params = module.parameters().flattened()
    var mismatches: [String] = []
    mismatches.reserveCapacity(8)

    for (key, param) in params {
      guard var tensor = weights[key] else { continue }
      if transpose4DTensors, tensor.ndim == 4 {
        tensor = tensor.transposed(0, 2, 3, 1)
      }
      if tensor.shape != param.shape {
        mismatches.append("\(key) expected \(param.shape) got \(tensor.shape)")
      }
    }

    if !mismatches.isEmpty {
      let sampleList = mismatches.prefix(max(0, sample)).joined(separator: "; ")
      let suffix = mismatches.count > sample ? "; ..." : ""
      logger.warning("Found \(mismatches.count) weight shape mismatches (sample: \(sampleList)\(suffix))")
    }

    return mismatches
  }

  private func loadVAEDecoder(snapshot _: URL, config: ZImageVAEConfig) throws -> AutoencoderDecoderOnly {
    AutoencoderDecoderOnly(configuration: .init(
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

  struct ModelSelection: Sendable {
    var baseModelSpec: String?
    var transformerOverrideURL: URL?
    var aioCheckpointURL: URL?
    var aioTextEncoderPrefix: String?
  }

  func resolveModelSelection(_ modelSpec: String?, forceTransformerOverrideOnly: Bool) -> ModelSelection {
    guard let modelSpec else { return .init(baseModelSpec: nil, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil) }

    let candidateURL = URL(fileURLWithPath: modelSpec)
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: candidateURL.path, isDirectory: &isDir) {
      if !isDir.boolValue && candidateURL.pathExtension == "safetensors" {
        return resolveLocalSafetensors(candidateURL, forceTransformerOverrideOnly: forceTransformerOverrideOnly)
      }
      if isDir.boolValue {
        let required = [
          ZImageFiles.transformerConfig,
          ZImageFiles.textEncoderConfig,
          ZImageFiles.vaeConfig,
        ]
        let hasStructure = required.allSatisfy { FileManager.default.fileExists(atPath: candidateURL.appending(path: $0).path) }
        if hasStructure {
          return .init(baseModelSpec: modelSpec, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
        }

        let contents = (try? FileManager.default.contentsOfDirectory(at: candidateURL, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        let safes = contents.filter { $0.pathExtension == "safetensors" }
        if safes.isEmpty {
          logger.warning("Model path is a directory without expected configs or safetensors: \(modelSpec). Falling back to default model.")
          return .init(baseModelSpec: nil, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
        }

        let preferred = safes.first(where: { $0.lastPathComponent.lowercased().contains("v2") })
          ?? safes.max(by: { a, b in
            let sa = (try? a.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            let sb = (try? b.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return sa < sb
          })

        guard let preferred else {
          return .init(baseModelSpec: nil, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
        }
        return resolveLocalSafetensors(preferred, forceTransformerOverrideOnly: forceTransformerOverrideOnly, sourceDirectory: candidateURL)
      }

      return .init(baseModelSpec: modelSpec, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
    }

    return .init(baseModelSpec: modelSpec, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
  }

  private func resolveLocalSafetensors(
    _ url: URL,
    forceTransformerOverrideOnly: Bool,
    sourceDirectory: URL? = nil
  ) -> ModelSelection {
    if !forceTransformerOverrideOnly {
      let inspection = ZImageAIOCheckpoint.inspect(fileURL: url)
      if inspection.isAIO, let prefix = inspection.textEncoderPrefix {
        if let sourceDirectory {
          logger.info("Detected AIO checkpoint in \(sourceDirectory.lastPathComponent): \(url.lastPathComponent). Bypassing base model weights.")
        } else {
          logger.info("Detected AIO checkpoint: \(url.lastPathComponent). Bypassing base model weights.")
        }
        return .init(baseModelSpec: nil, transformerOverrideURL: nil, aioCheckpointURL: url, aioTextEncoderPrefix: prefix)
      }
    }

    if let sourceDirectory {
      logger.info("Using transformer override file from directory: \(url.lastPathComponent)")
    } else {
      logger.info("Using transformer override file: \(url.lastPathComponent)")
    }
    return .init(baseModelSpec: nil, transformerOverrideURL: url, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
  }

  private func applyTransformerOverrideIfNeeded(_ overrideURL: URL?) throws {
    guard overrideURL != activeTransformerOverrideURL else { return }
    guard activeAIOCheckpointURL == nil else { return }
    guard let transformer, let snapshot = modelSnapshot, let configs = modelConfigs else { throw PipelineError.modelNotLoaded }

    let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, logger: logger)
    let baseTransformerWeights = try weightsMapper.loadTransformer()
    ZImageWeightsMapping.applyTransformer(weights: baseTransformerWeights, to: transformer, manifest: nil, logger: logger)

    activeTransformerOverrideURL = nil

    if let overrideURL {
      logger.info("Applying transformer override weights from: \(overrideURL.lastPathComponent)")
      var overrideWeights = try weightsMapper.loadTransformer(fromFile: overrideURL, dtype: .bfloat16)

      if let inferredDim = inferTransformerDim(from: overrideWeights), inferredDim != configs.transformer.dim {
        throw PipelineError.weightsMissing("Transformer override dim \(inferredDim) mismatches model dim \(configs.transformer.dim)")
      }

      overrideWeights = canonicalizeTransformerOverride(overrideWeights, dim: configs.transformer.dim, logger: logger)
      ZImageWeightsMapping.applyTransformer(weights: overrideWeights, to: transformer, manifest: nil, logger: logger)
      activeTransformerOverrideURL = overrideURL
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
  public func loadModel(
    modelSpec: String? = nil,
    aioCheckpointURL: URL? = nil,
    aioTextEncoderPrefix: String? = nil,
    progressHandler: ProgressHandler? = nil
  ) async throws {
    let modelId = modelSpec ?? ZImageRepository.id
    let normalizedAIOPath = aioCheckpointURL?.standardizedFileURL.path
    let currentAIOPath = activeAIOCheckpointURL?.standardizedFileURL.path
    let hasLoadedComponents = tokenizer != nil && textEncoder != nil && transformer != nil && vae != nil && modelConfigs != nil && modelSnapshot != nil
    if isModelLoaded, loadedModelId == modelId, normalizedAIOPath == currentAIOPath, hasLoadedComponents {
      logger.info("Model already loaded, skipping load")
      return
    }
    let canPreserveSharedComponents = isModelLoaded
      && loadedModelId != modelId
      && currentAIOPath == nil
      && normalizedAIOPath == nil
      && areZImageVariants(loadedModelId ?? "", modelId)
    if isModelLoaded, loadedModelId != modelId || normalizedAIOPath != currentAIOPath {
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

    let snapshotFilePatterns: [String]? = aioCheckpointURL == nil ? nil : PipelineSnapshot.configAndTokenizerFilePatterns
    let snapshot = try await PipelineSnapshot.prepare(model: modelSpec, filePatterns: snapshotFilePatterns, logger: logger)
    let configs = try ZImageModelConfigs.load(from: snapshot)
    if tokenizer == nil {
      progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
      logger.info("Loading tokenizer...")
      tokenizer = try loadTokenizer(snapshot: snapshot)
    } else {
      logger.info("Reusing cached tokenizer")
    }
    if let aioCheckpointURL {
      let textEncoderPrefix: String
      if let aioTextEncoderPrefix, !aioTextEncoderPrefix.isEmpty {
        textEncoderPrefix = aioTextEncoderPrefix
      } else {
        let inspection = ZImageAIOCheckpoint.inspect(fileURL: aioCheckpointURL)
        guard inspection.isAIO, let inferred = inspection.textEncoderPrefix else {
          let reason = inspection.diagnostics.isEmpty ? "unknown" : inspection.diagnostics.joined(separator: "; ")
          throw PipelineError.weightsMissing("Not a valid AIO checkpoint: \(aioCheckpointURL.lastPathComponent) (\(reason)). Use --force-transformer-override-only to treat it as transformer-only.")
        }
        textEncoderPrefix = inferred
      }

      logger.info("Loading AIO checkpoint weights from \(aioCheckpointURL.lastPathComponent)")
      let aio = try ZImageAIOCheckpoint.loadComponents(from: aioCheckpointURL, textEncoderPrefix: textEncoderPrefix, dtype: .bfloat16, logger: logger)

      logger.info("Loading text encoder...")
      let te = try loadTextEncoder(snapshot: snapshot, config: configs.textEncoder)
      ZImageWeightsMapping.applyTextEncoder(weights: aio.textEncoder, to: te, manifest: nil, logger: logger)
      textEncoder = te

      progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
      logger.info("Loading transformer...")
      let trans = try loadTransformer(snapshot: snapshot, config: configs.transformer)
      var transformerWeights = canonicalizeTransformerOverride(aio.transformer, dim: configs.transformer.dim, logger: logger)
      if let inferredDim = inferTransformerDim(from: transformerWeights), inferredDim != configs.transformer.dim {
        throw PipelineError.weightsMissing("AIO transformer dim \(inferredDim) mismatches model dim \(configs.transformer.dim)")
      }
      try validateStrictAIOTransformerWeights(transformerWeights, config: configs.transformer)
      try validateAIOTransformerCoverage(transformerWeights, transformer: trans)
      ZImageWeightsMapping.applyTransformer(weights: transformerWeights, to: trans, manifest: nil, logger: logger)
      transformer = trans

      activeTransformerOverrideURL = nil
      activeAIOCheckpointURL = aioCheckpointURL
      quantManifest = nil

      if vae == nil {
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        logger.info("Loading VAE...")
        let v = try loadVAEDecoder(snapshot: snapshot, config: configs.vae)
        let rawDecoderWeights = aio.vae.filter { $0.key.hasPrefix("decoder.") }
        let decoderWeights = ZImageAIOCheckpoint.canonicalizeVAEWeights(
          rawDecoderWeights,
          expectedUpBlocks: configs.vae.blockOutChannels.count,
          logger: logger
        )

        let audit = WeightsAudit.audit(module: v, weights: decoderWeights, logger: logger, sample: 10)
        let total = audit.matched + audit.missing.count
        let coverage = total > 0 ? Double(audit.matched) / Double(total) : 0.0
        let minimumCoverage = 0.99
        let mismatches = auditModuleWeightShapeMismatches(
          module: v,
          weights: decoderWeights,
          transpose4DTensors: true,
          logger: logger,
          sample: 10
        )

        if coverage >= minimumCoverage, mismatches.isEmpty {
          ZImageWeightsMapping.applyVAE(weights: decoderWeights, to: v, manifest: nil, logger: logger)
        } else {
          let percent = Int((coverage * 100.0).rounded())
          if mismatches.isEmpty {
            logger.warning("AIO VAE decoder weights coverage too low: matched \(audit.matched)/\(total) (\(percent)%). Falling back to base VAE weights.")
          } else {
            logger.warning("AIO VAE decoder weights have incompatible shapes (coverage \(percent)%), falling back to base VAE weights.")
          }

          let baseVAESnapshot = try await PipelineSnapshot.prepare(
            model: modelSpec,
            filePatterns: PipelineSnapshot.vaeOnlyFilePatterns,
            logger: logger
          )
          let weightsMapper = ZImageWeightsMapper(snapshot: baseVAESnapshot, logger: logger)
          let baseVAEWeights = try weightsMapper.loadVAE()
          let baseDecoderWeights = baseVAEWeights.filter { $0.key.hasPrefix("decoder.") }
          ZImageWeightsMapping.applyVAE(weights: baseDecoderWeights, to: v, manifest: nil, logger: logger)
        }
        vae = v
      } else {
        logger.info("Reusing cached VAE")
      }
    } else {
      let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, logger: logger)
      let manifest = weightsMapper.loadQuantizationManifest()

      if let m = manifest {
        logger.info("Loading quantized model (bits=\(m.bits), group_size=\(m.groupSize))")
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
      activeTransformerOverrideURL = nil
      activeAIOCheckpointURL = nil
      if vae == nil {
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        logger.info("Loading VAE...")
        let v = try loadVAEDecoder(snapshot: snapshot, config: configs.vae)
        let vaeWeights = try weightsMapper.loadVAE()
        let decoderWeights = vaeWeights.filter { $0.key.hasPrefix("decoder.") }
        ZImageWeightsMapping.applyVAE(weights: decoderWeights, to: v, manifest: manifest, logger: logger)
        vae = v
      } else {
        logger.info("Reusing cached VAE")
      }

      quantManifest = manifest
    }

    modelConfigs = configs
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
    let selection = resolveModelSelection(request.model, forceTransformerOverrideOnly: request.forceTransformerOverrideOnly)
    try await loadModel(
      modelSpec: selection.baseModelSpec,
      aioCheckpointURL: selection.aioCheckpointURL,
      aioTextEncoderPrefix: selection.aioTextEncoderPrefix,
      progressHandler: progressHandler
    )

    guard let vae = vae,
          let modelConfigs = modelConfigs
    else {
      throw PipelineError.modelNotLoaded
    }

    if selection.aioCheckpointURL == nil {
      try applyTransformerOverrideIfNeeded(selection.transformerOverrideURL)
    }

    if let loraConfig = request.lora {
      if currentLoRAConfig != loraConfig {
        try await loadLoRA(loraConfig, progressHandler: progressHandler)
      }
    } else if currentLoRA != nil {
      unloadLoRA()
    }
    progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: request.steps))
    logger.info("Encoding prompts...")

    let doCFG = request.guidanceScale > 1.0
    let promptEmbeds: MLXArray
    let negativeEmbeds: MLXArray?
    do {
      guard let tokenizer = tokenizer else {
        throw PipelineError.tokenizerNotLoaded
      }
      guard let textEncoder = textEncoder else {
        throw PipelineError.textEncoderNotLoaded
      }

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
    }
    logger.info("Text encoding complete")
    textEncoder = nil
    GPU.clearCache()

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
    do {
      guard let transformer = transformer else {
        throw PipelineError.transformerNotLoaded
      }
      for stepIndex in 0 ..< request.steps {
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
        let guidedNoise: MLXArray
        if doCFG, negativeEmbeds != nil {
          let batch = latents.dim(0)
          let positive = noisePred[0 ..< batch, 0..., 0..., 0...]
          let negative = noisePred[batch ..< batch * 2, 0..., 0..., 0...]
          guidedNoise = positive + request.guidanceScale * (positive - negative)
        } else {
          guidedNoise = noisePred
        }

        latents = scheduler.step(modelOutput: -guidedNoise, timestepIndex: stepIndex, sample: latents)
        MLX.eval(latents)
      }
      transformer.clearCache()
    }

    progressHandler?(GenerationProgress(stage: .denoising, stepIndex: request.steps, totalSteps: request.steps))
    unloadTransformer()
    logger.info("Denoising complete, decoding with VAE...")
    progressHandler?(GenerationProgress(stage: .decoding, stepIndex: request.steps, totalSteps: request.steps))

    let decoded = decodeLatents(latents, vae: vae, height: request.height, width: request.width)
    MLX.eval(MLXArray([]))
    GPU.clearCache()

    return decoded
  }

  private func decodeLatents(_ latents: MLXArray, vae: VAEImageDecoding, height: Int, width: Int) -> MLXArray {
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
    ZImageModelRegistry.areZImageVariants(model1, model2)
  }

  private func inferTransformerDim(from weights: [String: MLXArray]) -> Int? {
    // Try common norm vectors first
    if let w = weights["layers.0.attention_norm1.weight"], w.ndim == 1 { return w.dim(0) }
    if let w = weights["layers.0.ffn_norm1.weight"], w.ndim == 1 { return w.dim(0) }
    // Try attention projections
    if let w = weights["layers.0.attention.to_q.weight"], w.ndim == 2 { return w.dim(0) }
    if let w = weights["layers.0.attention.to_out.0.weight"], w.ndim == 2 { return w.dim(1) }
    // Scan for any norm weight
    if let (k, w) = weights.first(where: { $0.key.hasSuffix("attention_norm1.weight") && $0.value.ndim == 1 }) { _ = k; return w.dim(0) }
    if let (k, w) = weights.first(where: { $0.key.hasSuffix("ffn_norm1.weight") && $0.value.ndim == 1 }) { _ = k; return w.dim(0) }
    return nil
  }

  // Canonicalize override checkpoints so their tensor keys match our transformer module names.
  // Supports SD/ComfyUI-style exports that prefix keys with e.g. "model.diffusion_model.".
  func canonicalizeTransformerOverride(_ weights: [String: MLXArray], dim: Int, logger: Logger) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    for (k, v) in weights {
      // Strip common root prefixes from external checkpoints.
      var key = k
      for prefix in ["model.diffusion_model.", "diffusion_model.", "transformer.", "model."] {
        if key.hasPrefix(prefix) {
          key = String(key.dropFirst(prefix.count))
        }
      }

      // Some checkpoints use q_norm/k_norm naming; base Z-Image uses norm_q/norm_k.
      key = key.replacingOccurrences(of: ".attention.q_norm.weight", with: ".attention.norm_q.weight")
      key = key.replacingOccurrences(of: ".attention.k_norm.weight", with: ".attention.norm_k.weight")

      // Map attention.out.weight -> attention.to_out.0.weight
      if key.hasSuffix(".attention.out.weight") {
        let newKey = key.replacingOccurrences(of: ".attention.out.weight", with: ".attention.to_out.0.weight")
        out[newKey] = v
        continue
      }

      // Split attention.qkv.weight -> to_q.weight, to_k.weight, to_v.weight
      if key.hasSuffix(".attention.qkv.weight") {
        if v.ndim == 2, v.dim(0) == dim * 3, v.dim(1) == dim {
          let q = v[0 ..< dim, 0...]
          let kW = v[dim ..< 2 * dim, 0...]
          let vW = v[2 * dim ..< 3 * dim, 0...]
          let base = key.replacingOccurrences(of: ".attention.qkv.weight", with: "")
          out["\(base).attention.to_q.weight"] = q
          out["\(base).attention.to_k.weight"] = kW
          out["\(base).attention.to_v.weight"] = vW
        } else {
          logger.warning("Unexpected qkv shape for \(key): \(v.shape) (expected [\(dim * 3), \(dim)])")
        }
        continue
      }

      // Passthrough other keys
      var mapped = key
      // Remap final_layer.* -> all_final_layer.2-1.* so our loader can pick them up
      if mapped.hasPrefix("final_layer.") {
        mapped = mapped.replacingOccurrences(of: "final_layer.", with: "all_final_layer.2-1.")
      }
      // Remap x_embedder.* -> all_x_embedder.2-1.*
      if mapped.hasPrefix("x_embedder.") {
        mapped = mapped.replacingOccurrences(of: "x_embedder.", with: "all_x_embedder.2-1.")
      }
      out[mapped] = v
    }
    return out
  }

  func validateStrictAIOTransformerWeights(_ weights: [String: MLXArray], config: ZImageTransformerConfig) throws {
    var required: [String] = [
      "layers.0.attention.to_q.weight",
      "layers.0.attention.to_out.0.weight",
    ]

    if config.qkNorm {
      required.append(contentsOf: [
        "layers.0.attention.norm_q.weight",
        "layers.0.attention.norm_k.weight",
      ])
    }

    let missing = required.filter { weights[$0] == nil }
    if !missing.isEmpty {
      throw PipelineError.weightsMissing(
        "AIO checkpoint missing required transformer tensors after canonicalization: \(missing.joined(separator: ", ")). Use --force-transformer-override-only to treat it as transformer-only."
      )
    }
  }

  func validateAIOTransformerCoverage(
    _ weights: [String: MLXArray],
    transformer: ZImageTransformer2DModel,
    minimumCoverage: Double = 0.99
  ) throws {
    var auditWeights = weights
    if let w = weights["cap_embedder.0.weight"] { auditWeights["capEmbedNorm.weight"] = w }
    if let w = weights["cap_embedder.1.weight"] { auditWeights["capEmbedLinear.weight"] = w }
    if let w = weights["cap_embedder.1.bias"] { auditWeights["capEmbedLinear.bias"] = w }

    let audit = WeightsAudit.audit(module: transformer, weights: auditWeights, logger: logger, sample: 10)
    let total = audit.matched + audit.missing.count
    guard total > 0 else {
      throw PipelineError.weightsMissing("AIO transformer audit failed: transformer contains no parameters.")
    }

    let coverage = Double(audit.matched) / Double(total)
    guard coverage >= minimumCoverage else {
      let percent = Int((coverage * 100.0).rounded())
      let missingSample = audit.missing.prefix(10).joined(separator: ", ")
      let suffix = audit.missing.count > 10 ? ", ..." : ""
      throw PipelineError.weightsMissing("""
        AIO transformer weights coverage too low:
         matched \(audit.matched)/\(total) (\(percent)%).
         Missing (sample): \(missingSample)\(suffix).
         Use --force-transformer-override-only to treat it as transformer-only.
        """
      )
    }
  }
}
