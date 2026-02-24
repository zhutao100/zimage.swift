import Dispatch
import Foundation
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
  public var weightsVariant: String?
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
    weightsVariant: String? = nil,
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
    self.weightsVariant = weightsVariant
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
  private var tokenizer: QwenTokenizer?
  private var textEncoder: QwenTextEncoder?
  private var transformer: ZImageTransformer2DModel?
  private var vae: VAEImageDecoding?
  private var modelConfigs: ZImageModelConfigs?
  private var isModelLoaded: Bool = false
  private var loadedModelId: String?
  private var loadedWeightsVariant: String?
  private var currentLoRA: LoRAWeights?
  private var currentLoRAConfig: LoRAConfiguration?
  private var modelSnapshot: URL?
  private var activeTransformerOverrideURL: URL?
  private var activeAIOCheckpointURL: URL?

  public init(logger: Logger = Logger(label: "z-image.pipeline")) {
    self.logger = logger
  }

  public var isLoaded: Bool {
    isModelLoaded
  }

  public func unloadModel() {
    tokenizer = nil
    textEncoder = nil
    transformer = nil
    vae = nil
    modelConfigs = nil
    isModelLoaded = false
    loadedModelId = nil
    loadedWeightsVariant = nil

    currentLoRA = nil
    currentLoRAConfig = nil
    modelSnapshot = nil
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
    GPU.clearCache()
    logger.info("LoRA unloaded (instant)")
  }

  public func unloadTransformer() {
    transformer = nil

    currentLoRA = nil
    currentLoRAConfig = nil
    activeTransformerOverrideURL = nil

    GPU.clearCache()
    logger.info("Transformer unloaded for memory optimization")
  }

  private func loadTokenizer(snapshot: URL) throws -> QwenTokenizer {
    let tokDir = snapshot.appending(path: "tokenizer")
    return try QwenTokenizer.load(from: tokDir)
  }

  private func loadTextEncoder(snapshot _: URL, config: ZImageTextEncoderConfig) throws -> QwenTextEncoder {
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

  private func loadTransformer(snapshot _: URL, config: ZImageTransformerConfig) throws -> ZImageTransformer2DModel {
    ZImageTransformer2DModel(configuration: config)
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
      if !isDir.boolValue, candidateURL.pathExtension == "safetensors" {
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

    let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, weightsVariant: loadedWeightsVariant, logger: logger)
    let baseTransformerWeights = try weightsMapper.loadTransformer()
    ZImageWeightsMapping.applyTransformer(weights: baseTransformerWeights, to: transformer, manifest: nil, logger: logger)

    activeTransformerOverrideURL = nil

    if let overrideURL {
      logger.info("Applying transformer override weights from: \(overrideURL.lastPathComponent)")
      var overrideWeights = try weightsMapper.loadTransformer(fromFile: overrideURL, dtype: .bfloat16)

      if let inferredDim = ZImageTransformerOverride.inferDim(from: overrideWeights), inferredDim != configs.transformer.dim {
        throw PipelineError.weightsMissing("Transformer override dim \(inferredDim) mismatches model dim \(configs.transformer.dim)")
      }

      overrideWeights = ZImageTransformerOverride.canonicalize(overrideWeights, dim: configs.transformer.dim, logger: logger)
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
    weightsVariant: String? = nil,
    aioCheckpointURL: URL? = nil,
    aioTextEncoderPrefix: String? = nil,
    progressHandler: ProgressHandler? = nil
  ) async throws {
    let normalizedWeightsVariant = ZImageFiles.normalizedWeightsVariant(weightsVariant)

    let modelId = modelSpec ?? ZImageRepository.id
    let normalizedAIOPath = aioCheckpointURL?.standardizedFileURL.path
    let currentAIOPath = activeAIOCheckpointURL?.standardizedFileURL.path
    let hasLoadedComponents = tokenizer != nil && textEncoder != nil && transformer != nil && vae != nil && modelConfigs != nil && modelSnapshot != nil
    if isModelLoaded,
       loadedModelId == modelId,
       loadedWeightsVariant == normalizedWeightsVariant,
       normalizedAIOPath == currentAIOPath,
       hasLoadedComponents
    {
      logger.info("Model already loaded, skipping load")
      return
    }
    let canPreserveSharedComponents = isModelLoaded
      && loadedModelId != modelId
      && currentAIOPath == nil
      && normalizedAIOPath == nil
      && loadedWeightsVariant == normalizedWeightsVariant
      && areZImageVariants(loadedModelId ?? "", modelId)
    if isModelLoaded,
       loadedModelId != modelId || normalizedAIOPath != currentAIOPath || loadedWeightsVariant != normalizedWeightsVariant
    {
      if canPreserveSharedComponents {
        logger.info("Switching Z-Image variant, preserving VAE and tokenizer")

        textEncoder = nil
        transformer = nil

        currentLoRA = nil
        currentLoRAConfig = nil
      } else {
        logger.info("Different model requested, unloading current model")
        unloadModel()
      }
    }

    logger.info("Loading model: \(modelId)")
    progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))

    let snapshotFilePatterns: [String]? = aioCheckpointURL == nil ? nil : PipelineSnapshot.configAndTokenizerFilePatterns
    let shouldValidateWeightsForCache = snapshotFilePatterns == nil
    let snapshotValidator: (@Sendable (URL) -> Bool)? = if shouldValidateWeightsForCache, let normalizedWeightsVariant {
      { [logger] snapshot in
        !ZImageFiles.resolveTransformerWeights(at: snapshot, weightsVariant: normalizedWeightsVariant, logger: logger).isEmpty
          && !ZImageFiles.resolveTextEncoderWeights(at: snapshot, weightsVariant: normalizedWeightsVariant, logger: logger).isEmpty
          && !ZImageFiles.resolveVAEWeights(at: snapshot, weightsVariant: normalizedWeightsVariant, logger: logger).isEmpty
      }
    } else {
      nil
    }

    let snapshot = try await PipelineSnapshot.prepare(
      model: modelSpec,
      weightsVariant: normalizedWeightsVariant,
      filePatterns: snapshotFilePatterns,
      snapshotValidator: snapshotValidator,
      logger: logger
    )
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
      var transformerWeights = ZImageTransformerOverride.canonicalize(aio.transformer, dim: configs.transformer.dim, logger: logger)
      if let inferredDim = ZImageTransformerOverride.inferDim(from: transformerWeights), inferredDim != configs.transformer.dim {
        throw PipelineError.weightsMissing("AIO transformer dim \(inferredDim) mismatches model dim \(configs.transformer.dim)")
      }

      let missingStrictKeys = ZImageAIOTransformerValidation.missingStrictRequiredKeys(in: transformerWeights, config: configs.transformer)
      if !missingStrictKeys.isEmpty {
        throw PipelineError.weightsMissing(
          """
          AIO checkpoint missing required transformer tensors after canonicalization: \(missingStrictKeys.joined(separator: ", ")).
            Use --force-transformer-override-only to treat it as transformer-only.
          """
        )
      }

      let auditWeights = ZImageAIOTransformerValidation.coverageAuditWeights(transformerWeights)
      let audit = WeightsAudit.audit(module: trans, weights: auditWeights, logger: logger, sample: 10)
      let total = audit.matched + audit.missing.count
      guard total > 0 else {
        throw PipelineError.weightsMissing("AIO transformer audit failed: transformer contains no parameters.")
      }

      let minimumCoverage = 0.99
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
        """)
      }
      ZImageWeightsMapping.applyTransformer(weights: transformerWeights, to: trans, manifest: nil, logger: logger)
      transformer = trans

      activeTransformerOverrideURL = nil
      activeAIOCheckpointURL = aioCheckpointURL

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
        let mismatches = WeightsAudit.auditShapeMismatches(
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

          let vaeSnapshotValidator: (@Sendable (URL) -> Bool)? = if let normalizedWeightsVariant {
            { [logger] snapshot in
              !ZImageFiles.resolveVAEWeights(at: snapshot, weightsVariant: normalizedWeightsVariant, logger: logger).isEmpty
            }
          } else {
            nil
          }
          let baseVAESnapshot = try await PipelineSnapshot.prepare(
            model: modelSpec,
            weightsVariant: normalizedWeightsVariant,
            filePatterns: PipelineSnapshot.vaeOnlyFilePatterns(weightsVariant: normalizedWeightsVariant),
            snapshotValidator: vaeSnapshotValidator,
            logger: logger
          )
          if let normalizedWeightsVariant,
             ZImageFiles.resolveVAEWeights(at: baseVAESnapshot, weightsVariant: normalizedWeightsVariant, logger: logger).isEmpty
          {
            throw ZImageFiles.WeightsVariantError.missingRequiredComponentWeights(
              weightsVariant: normalizedWeightsVariant,
              missingComponents: ["vae"],
              snapshot: baseVAESnapshot
            )
          }
          let weightsMapper = ZImageWeightsMapper(snapshot: baseVAESnapshot, weightsVariant: normalizedWeightsVariant, logger: logger)
          let baseVAEWeights = try weightsMapper.loadVAE()
          let baseDecoderWeights = baseVAEWeights.filter { $0.key.hasPrefix("decoder.") }
          ZImageWeightsMapping.applyVAE(weights: baseDecoderWeights, to: v, manifest: nil, logger: logger)
        }
        vae = v
      } else {
        logger.info("Reusing cached VAE")
      }
    } else {
      let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, weightsVariant: normalizedWeightsVariant, logger: logger)
      let manifest = weightsMapper.loadQuantizationManifest()
      if manifest == nil {
        try ZImageFiles.validateRequiredComponentWeights(at: snapshot, weightsVariant: normalizedWeightsVariant, logger: logger)
      }

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
    }

    modelConfigs = configs
    modelSnapshot = snapshot
    isModelLoaded = true
    loadedModelId = modelId
    loadedWeightsVariant = normalizedWeightsVariant

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

      LoRAApplicator.applyDynamically(to: trans, loraWeights: loraWeights, scale: config.scale, logger: logger)

      currentLoRA = loraWeights
      currentLoRAConfig = config

      logger.info("LoRA applied successfully with scale=\(config.scale)")
    } catch let error as LoRAError {
      throw PipelineError.loraError(error)
    }
  }

  public var hasLoRALoaded: Bool {
    currentLoRA != nil
  }

  public var loadedLoRAConfig: LoRAConfiguration? {
    currentLoRAConfig
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
      weightsVariant: request.weightsVariant,
      aioCheckpointURL: selection.aioCheckpointURL,
      aioTextEncoderPrefix: selection.aioTextEncoderPrefix,
      progressHandler: progressHandler
    )

    guard let vae,
          let modelConfigs
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
      guard let tokenizer else {
        throw PipelineError.tokenizerNotLoaded
      }
      guard let textEncoder else {
        throw PipelineError.textEncoderNotLoaded
      }

      var finalPrompt = request.prompt
      if request.enhancePrompt {
        logger.info("Enhancing prompt using LLM (max tokens: \(request.enhanceMaxTokens))...")
        let enhanceConfig = PromptEnhanceConfig(maxNewTokens: request.enhanceMaxTokens)
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

    let mu = PipelineUtilities.calculateShift(
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
      guard let transformer else {
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

    let decoded = PipelineUtilities.decodeLatents(latents, vae: vae, height: request.height, width: request.width)
    MLX.eval(MLXArray([]))
    GPU.clearCache()

    return decoded
  }

  private func areZImageVariants(_ model1: String, _ model2: String) -> Bool {
    ZImageModelRegistry.areZImageVariants(model1, model2)
  }
}
