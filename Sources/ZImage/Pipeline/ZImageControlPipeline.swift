import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import Hub
import Dispatch

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

public struct ZImageControlGenerationRequest {
  public var prompt: String
  public var negativePrompt: String?
  public var controlImage: URL?
  public var controlContextScale: Float
  public var width: Int
  public var height: Int
  public var steps: Int
  public var guidanceScale: Float
  public var seed: UInt64?
  public var outputPath: URL
  public var model: String?
  public var controlnetWeights: String?
  public var maxSequenceLength: Int

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    controlImage: URL? = nil,
    controlContextScale: Float = 0.75,
    width: Int = ZImageModelMetadata.recommendedWidth,
    height: Int = ZImageModelMetadata.recommendedHeight,
    steps: Int = ZImageModelMetadata.recommendedInferenceSteps,
    guidanceScale: Float = ZImageModelMetadata.recommendedGuidanceScale,
    seed: UInt64? = nil,
    outputPath: URL = URL(fileURLWithPath: "z-image-control.png"),
    model: String? = nil,
    controlnetWeights: String? = nil,
    maxSequenceLength: Int = 512
  ) {
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.controlImage = controlImage
    self.controlContextScale = controlContextScale
    self.width = width
    self.height = height
    self.steps = steps
    self.guidanceScale = guidanceScale
    self.seed = seed
    self.outputPath = outputPath
    self.model = model
    self.controlnetWeights = controlnetWeights
    self.maxSequenceLength = maxSequenceLength
  }
}

public struct ZImageControlPipeline {
  public enum PipelineError: Error {
    case notImplemented
    case tokenizerNotLoaded
    case textEncoderNotLoaded
    case transformerNotLoaded
    case vaeNotLoaded
    case weightsMissing(String)
    case controlImageNotFound(URL)
    case controlImageLoadFailed(String)
  }

  private var logger: Logger
  private let hubApi: HubApi

  public init(logger: Logger = Logger(label: "z-image.control-pipeline"), hubApi: HubApi = .shared) {
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

  private func loadControlTransformer(snapshot: URL, config: ZImageTransformerConfig) throws -> ZImageControlTransformer2DModel {
    let controlConfig = ZImageControlTransformerConfig(
      inChannels: config.inChannels,
      dim: config.dim,
      nLayers: config.nLayers,
      nRefinerLayers: config.nRefinerLayers,
      nHeads: config.nHeads,
      nKVHeads: config.nKVHeads,
      normEps: config.normEps,
      qkNorm: config.qkNorm,
      capFeatDim: config.capFeatDim,
      ropeTheta: config.ropeTheta,
      tScale: config.tScale,
      axesDims: config.axesDims,
      axesLens: config.axesLens,
      controlLayersPlaces: [0, 5, 10, 15, 20, 25],
      controlInDim: 16
    )
    return ZImageControlTransformer2DModel(configuration: controlConfig)
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
    let encoded = try tokenizer.encodeChat(prompts: [prompt], maxLength: maxLength)
    let embeddingsList = textEncoder.encodeForZImage(inputIds: encoded.inputIds, attentionMask: encoded.attentionMask)

    guard let firstEmbeds = embeddingsList.first else {
      throw PipelineError.textEncoderNotLoaded
    }

    let embedsBatch = firstEmbeds.expandedDimensions(axis: 0)
    let mask = MLX.ones([1, firstEmbeds.dim(0)], dtype: .int32)

    return (embedsBatch, mask)
  }

  private func loadControlImage(
    url: URL,
    vae: AutoencoderKL,
    vaeConfig: ZImageVAEConfig,
    targetHeight: Int,
    targetWidth: Int
  ) throws -> MLXArray {
    #if canImport(CoreGraphics)
    guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
      throw PipelineError.controlImageLoadFailed("Failed to load image from \(url.path)")
    }

    let vaeDivisor = vaeConfig.latentDivisor
    let latentH = max(1, targetHeight / vaeDivisor)
    let latentW = max(1, targetWidth / vaeDivisor)

    let pixelH = latentH * vaeDivisor
    let pixelW = latentW * vaeDivisor

    let imageArray = try QwenImageIO.resizedPixelArray(
      from: cgImage,
      width: pixelW,
      height: pixelH,
      addBatchDimension: true,
      dtype: .float32
    )

    let normalized = QwenImageIO.normalizeForEncoder(imageArray)

    let encodedLatents = vae.encode(normalized)
    let latentChannels = vaeConfig.latentChannels
    let latents = encodedLatents[0..., 0..<latentChannels, 0..., 0...]

    let normalizedLatents = (latents - vaeConfig.shiftFactor) * vaeConfig.scalingFactor

    let controlContext = MLX.expandedDimensions(normalizedLatents, axis: 2)

    return controlContext
    #else
    throw PipelineError.controlImageLoadFailed("CoreGraphics not available on this platform")
    #endif
  }

  public func generate(
    _ request: ZImageControlGenerationRequest,
    progress: (@Sendable (_ completed: Int, _ total: Int) -> Void)? = nil
  ) async throws -> URL {
    logger.info("Requested Z-Image control generation")

    let snapshot = try await prepareSnapshot(model: request.model)
    let modelConfigs = try ZImageModelConfigs.load(from: snapshot)
    let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, logger: logger)
    let quantManifest = weightsMapper.loadQuantizationManifest()

    if let manifest = quantManifest {
      logger.info("Loading quantized model (bits=\(manifest.bits), group_size=\(manifest.groupSize))")
    }

    let doCFG = request.guidanceScale > 1.0

    logger.info("Loading text encoder...")
    var promptEmbeds: MLXArray
    var negativeEmbeds: MLXArray?
    do {
      let tokenizer = try loadTokenizer(snapshot: snapshot)
      let textEncoder = try loadTextEncoder(snapshot: snapshot, config: modelConfigs.textEncoder)
      let textEncoderWeights = try weightsMapper.loadTextEncoder()
      ZImageWeightsMapping.applyTextEncoder(weights: textEncoderWeights, to: textEncoder, manifest: quantManifest, logger: logger)

      let (pe, _) = try encodePrompt(request.prompt, tokenizer: tokenizer, textEncoder: textEncoder, maxLength: request.maxSequenceLength)

      if doCFG {
        let (ne, _) = try encodePrompt(request.negativePrompt ?? "", tokenizer: tokenizer, textEncoder: textEncoder, maxLength: request.maxSequenceLength)
        promptEmbeds = pe
        negativeEmbeds = ne
        MLX.eval(pe, ne)
      } else {
        promptEmbeds = pe
        negativeEmbeds = nil
        MLX.eval(pe)
      }
    }
    logger.info("Text encoding complete, clearing text encoder from memory")
    GPU.clearCache()

    logger.info("Loading VAE...")
    let vae = try loadVAE(snapshot: snapshot, config: modelConfigs.vae)
    // Load VAE weights as bfloat16 to keep decode compute in bf16
    let vaeWeights = try weightsMapper.loadVAE(dtype: .bfloat16)
    ZImageWeightsMapping.applyVAE(weights: vaeWeights, to: vae, manifest: quantManifest, logger: logger)

    var controlContext: MLXArray? = nil
    if let controlImageURL = request.controlImage {
      logger.info("Loading control image from \(controlImageURL.path)...")
      let context = try loadControlImage(
        url: controlImageURL,
        vae: vae,
        vaeConfig: modelConfigs.vae,
        targetHeight: request.height,
        targetWidth: request.width
      )
      MLX.eval(context)
      logger.info("Control image encoded, shape: \(context.shape)")
      controlContext = context
    }

    logger.info("Loading control transformer...")
    let transformer = try loadControlTransformer(snapshot: snapshot, config: modelConfigs.transformer)

    let transformerWeights = try weightsMapper.loadTransformer()
    ZImageControlWeightsMapping.applyControlTransformer(
      weights: transformerWeights,
      to: transformer,
      manifest: quantManifest,
      logger: logger
    )

    if let controlnetSpec = request.controlnetWeights {
      logger.info("Loading controlnet weights from \(controlnetSpec)...")
      let controlnetWeights = try await loadControlnetWeights(controlnetSpec: controlnetSpec)
      ZImageControlWeightsMapping.applyControlnetWeights(
        weights: controlnetWeights,
        to: transformer,
        logger: logger
      )
    }

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

    logger.info("Running \(request.steps) denoising steps with control_context_scale=\(request.controlContextScale)...")
    for stepIndex in 0..<request.steps {
      let timestep = timestepsArray[stepIndex]
      let normalizedTimestep = (1000.0 - timestep) / 1000.0
      let timestepArray = MLXArray([normalizedTimestep], [1])

      var modelLatents = latents
      var embeds = promptEmbeds
      if doCFG, let ne = negativeEmbeds {
        modelLatents = MLX.concatenated([latents, latents], axis: 0)
        embeds = MLX.concatenated([promptEmbeds, ne], axis: 0)
      }

      let noisePred = transformer.forward(
        latents: modelLatents,
        timestep: timestepArray,
        promptEmbeds: embeds,
        controlContext: controlContext,
        controlContextScale: request.controlContextScale
      )

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
      if let progress {
        progress(stepIndex + 1, request.steps)
      }
    }

    logger.info("Denoising complete, decoding latents...")
    GPU.clearCache()

    let decoded = decodeLatents(latents, vae: vae, height: request.height, width: request.width)
    try QwenImageIO.saveImage(array: decoded, to: request.outputPath)
    logger.info("Wrote image to \(request.outputPath.path)")
    return request.outputPath
  }

  private func decodeLatents(_ latents: MLXArray, vae: AutoencoderKL, height: Int, width: Int) -> MLXArray {
    // Ensure VAE decode runs in bf16 by casting inputs; post-processing remains float32/uint8 downstream
    let input = latents.asType(.bfloat16)
    let (decoded, _) = vae.decode(input)
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

  private func loadControlnetWeights(controlnetSpec: String, dtype: DType = .bfloat16) async throws -> [String: MLXArray] {
    let localURL = URL(fileURLWithPath: controlnetSpec)
    if FileManager.default.fileExists(atPath: localURL.path) && controlnetSpec.hasSuffix(".safetensors") {
      logger.info("Loading controlnet from local file: \(controlnetSpec)")
      return try loadSafetensorsFile(url: localURL, dtype: dtype)
    }

    if ModelResolution.isHuggingFaceModelId(controlnetSpec) {
      logger.info("Resolving controlnet from HuggingFace: \(controlnetSpec)")
      let snapshot = try await ModelResolution.resolve(
        modelSpec: controlnetSpec,
        filePatterns: ["*.safetensors"],
        progressHandler: { [logger] progress in
          let percent = Int(progress.fractionCompleted * 100)
          logger.info("Downloading controlnet: \(percent)%")
        }
      )

      let fm = FileManager.default
      let contents = try fm.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: nil)
      let safetensorsFiles = contents.filter { $0.pathExtension == "safetensors" }

      guard let safetensorsFile = safetensorsFiles.first else {
        throw PipelineError.weightsMissing("No .safetensors file found in controlnet model")
      }

      logger.info("Loading controlnet weights from \(safetensorsFile.lastPathComponent)")
      return try loadSafetensorsFile(url: safetensorsFile, dtype: dtype)
    }

    throw PipelineError.weightsMissing("Invalid controlnet spec: \(controlnetSpec). Provide a local .safetensors path or HuggingFace model ID (e.g., alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union)")
  }

  private func loadSafetensorsFile(url: URL, dtype: DType) throws -> [String: MLXArray] {
    var tensors: [String: MLXArray] = [:]
    let reader = try SafeTensorsReader(fileURL: url)
    for meta in reader.allMetadata() {
      var tensor = try reader.tensor(named: meta.name)
      if tensor.dtype != dtype {
        tensor = tensor.asType(dtype)
      }
      tensors[meta.name] = tensor
    }
    logger.info("Loaded \(tensors.count) controlnet tensors")
    return tensors
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

public enum ZImageControlWeightsMapping {

  private static func transformerMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      mapped["transformer.\(k)"] = v
    }
    return mapped
  }

  private static func applyToModule(_ module: Module, weights: [String: MLXArray], prefix: String, logger: Logger) {
    let params = module.parameters().flattened()
    var updates: [(String, MLXArray)] = []

    for (key, _) in params {
      let candidates = [key, "\(prefix).\(key)"]
      if let found = candidates.compactMap({ weights[$0] }).first {
        updates.append((key, found))
      }
    }

    for (weightKey, tensor) in weights {
      var paramKey = weightKey
      if weightKey.hasPrefix("\(prefix).") {
        paramKey = String(weightKey.dropFirst("\(prefix).".count))
      }

      if (paramKey.hasSuffix(".scales") || paramKey.hasSuffix(".biases")) {
        if !updates.contains(where: { $0.0 == paramKey }) {
          updates.append((paramKey, tensor))
        }
      }
    }

    if updates.isEmpty {
      logger.warning("\(prefix) received no matching weights; skipping apply.")
      return
    }

    do {
      let nd = ModuleParameters.unflattened(updates)
      try module.update(parameters: nd, verify: [.shapeMismatch])
    } catch {
      logger.error("Failed to apply weights to \(prefix): \(error)")
    }
  }

  public static func applyControlTransformer(
    weights: [String: MLXArray],
    to transformer: ZImageControlTransformer2DModel,
    manifest: ZImageQuantizationManifest?,
    logger: Logger
  ) {
    if let manifest = manifest {
      let availableKeys = Set(weights.keys)
      ZImageQuantizer.applyQuantization(
        to: transformer,
        manifest: manifest,
        availableKeys: availableKeys,
        tensorNameTransform: ZImageQuantizer.transformerTensorName
      )
    }

    let groupSize = manifest?.groupSize ?? 32
    let bits = manifest?.bits ?? 8

    let mapped = transformerMapping(weights)
    applyToModule(transformer, weights: mapped, prefix: "transformer", logger: logger)

    transformer.loadCapEmbedderWeights(from: weights)
    transformer.loadXEmbedderWeights(from: weights, groupSize: groupSize, bits: bits)
    transformer.loadFinalLayerWeights(from: weights, groupSize: groupSize, bits: bits)

    transformer.setPadTokens(xPad: weights["x_pad_token"], capPad: weights["cap_pad_token"])

    logger.info("Applied base transformer weights to control transformer")
  }

  public static func applyControlnetWeights(
    weights: [String: MLXArray],
    to transformer: ZImageControlTransformer2DModel,
    logger: Logger
  ) {
    transformer.loadControlXEmbedderWeights(from: weights)

    for (idx, block) in transformer.controlNoiseRefiner.enumerated() {
      applyTransformerBlockWeights(weights: weights, prefix: "control_noise_refiner.\(idx)", to: block)
    }

    for (idx, block) in transformer.controlLayers.enumerated() {
      applyControlTransformerBlockWeights(weights: weights, prefix: "control_layers.\(idx)", to: block)
    }

    logger.info("Applied controlnet weights")
  }

  private static func applyTransformerBlockWeights(
    weights: [String: MLXArray],
    prefix: String,
    to block: ZImageTransformerBlock
  ) {
    if let w = weights["\(prefix).attention.to_q.weight"] {
      block.attention.toQ.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_k.weight"] {
      block.attention.toK.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_v.weight"] {
      block.attention.toV.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_out.0.weight"] {
      block.attention.toOut[0].weight._updateInternal(w)
    }
    if let b = weights["\(prefix).attention.to_out.0.bias"] {
      block.attention.toOut[0].bias?._updateInternal(b)
    }

    if let w = weights["\(prefix).attention.norm_q.weight"] {
      block.attention.normQ?.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.norm_k.weight"] {
      block.attention.normK?.weight._updateInternal(w)
    }

    if let adaLN = block.adaLN, adaLN.count > 0 {
      if let w = weights["\(prefix).adaLN_modulation.0.weight"] {
        adaLN[0].weight._updateInternal(w)
      }
      if let b = weights["\(prefix).adaLN_modulation.0.bias"] {
        adaLN[0].bias?._updateInternal(b)
      }
    }

    if let w = weights["\(prefix).attention_norm1.weight"] {
      block.attentionNorm1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).ffn_norm1.weight"] {
      block.ffnNorm1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention_norm2.weight"] {
      block.attentionNorm2.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).ffn_norm2.weight"] {
      block.ffnNorm2.weight._updateInternal(w)
    }

    if let w = weights["\(prefix).feed_forward.w1.weight"] {
      block.feedForward.w1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).feed_forward.w2.weight"] {
      block.feedForward.w2.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).feed_forward.w3.weight"] {
      block.feedForward.w3.weight._updateInternal(w)
    }
  }

  private static func applyBaseTransformerBlockWeights(
    weights: [String: MLXArray],
    prefix: String,
    to block: BaseZImageTransformerBlock
  ) {
    if let w = weights["\(prefix).attention.to_q.weight"] {
      block.attention.toQ.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_k.weight"] {
      block.attention.toK.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_v.weight"] {
      block.attention.toV.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_out.0.weight"] {
      block.attention.toOut[0].weight._updateInternal(w)
    }
    if let b = weights["\(prefix).attention.to_out.0.bias"] {
      block.attention.toOut[0].bias?._updateInternal(b)
    }

    if let w = weights["\(prefix).attention.norm_q.weight"] {
      block.attention.normQ?.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.norm_k.weight"] {
      block.attention.normK?.weight._updateInternal(w)
    }

    if let adaLN = block.adaLN, adaLN.count > 0 {
      if let w = weights["\(prefix).adaLN_modulation.0.weight"] {
        adaLN[0].weight._updateInternal(w)
      } else if let w = weights["\(prefix).adaLN_modulation.1.weight"] {
        adaLN[0].weight._updateInternal(w)
      }
      if let b = weights["\(prefix).adaLN_modulation.0.bias"] {
        adaLN[0].bias?._updateInternal(b)
      } else if let b = weights["\(prefix).adaLN_modulation.1.bias"] {
        adaLN[0].bias?._updateInternal(b)
      }
    }

    if let w = weights["\(prefix).attention_norm1.weight"] {
      block.attentionNorm1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).ffn_norm1.weight"] {
      block.ffnNorm1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention_norm2.weight"] {
      block.attentionNorm2.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).ffn_norm2.weight"] {
      block.ffnNorm2.weight._updateInternal(w)
    }

    if let w = weights["\(prefix).feed_forward.w1.weight"] {
      block.feedForward.w1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).feed_forward.w2.weight"] {
      block.feedForward.w2.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).feed_forward.w3.weight"] {
      block.feedForward.w3.weight._updateInternal(w)
    }
  }

  private static func applyControlTransformerBlockWeights(
    weights: [String: MLXArray],
    prefix: String,
    to block: ZImageControlTransformerBlock
  ) {
    if let beforeProj = block.beforeProj {
      if let w = weights["\(prefix).before_proj.weight"] {
        beforeProj.weight._updateInternal(w)
      }
      if let b = weights["\(prefix).before_proj.bias"] {
        beforeProj.bias?._updateInternal(b)
      }
    }

    if let w = weights["\(prefix).after_proj.weight"] {
      block.afterProj.weight._updateInternal(w)
    }
    if let b = weights["\(prefix).after_proj.bias"] {
      block.afterProj.bias?._updateInternal(b)
    }

    if let w = weights["\(prefix).attention.to_q.weight"] {
      block.attention.toQ.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_k.weight"] {
      block.attention.toK.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_v.weight"] {
      block.attention.toV.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_out.0.weight"] {
      block.attention.toOut[0].weight._updateInternal(w)
    }
    if let b = weights["\(prefix).attention.to_out.0.bias"] {
      block.attention.toOut[0].bias?._updateInternal(b)
    }

    if let w = weights["\(prefix).attention.norm_q.weight"] {
      block.attention.normQ?.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.norm_k.weight"] {
      block.attention.normK?.weight._updateInternal(w)
    }

    if let adaLN = block.adaLN, adaLN.count > 0 {
      if let w = weights["\(prefix).adaLN_modulation.0.weight"] {
        adaLN[0].weight._updateInternal(w)
      } else if let w = weights["\(prefix).adaLN_modulation.1.weight"] {
        adaLN[0].weight._updateInternal(w)
      }
      if let b = weights["\(prefix).adaLN_modulation.0.bias"] {
        adaLN[0].bias?._updateInternal(b)
      } else if let b = weights["\(prefix).adaLN_modulation.1.bias"] {
        adaLN[0].bias?._updateInternal(b)
      }
    }

    if let w = weights["\(prefix).attention_norm1.weight"] {
      block.attentionNorm1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).ffn_norm1.weight"] {
      block.ffnNorm1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention_norm2.weight"] {
      block.attentionNorm2.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).ffn_norm2.weight"] {
      block.ffnNorm2.weight._updateInternal(w)
    }

    if let w = weights["\(prefix).feed_forward.w1.weight"] {
      block.feedForward.w1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).feed_forward.w2.weight"] {
      block.feedForward.w2.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).feed_forward.w3.weight"] {
      block.feedForward.w3.weight._updateInternal(w)
    }
  }
}
