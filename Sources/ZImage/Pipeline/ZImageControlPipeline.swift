// swiftlint:disable file_length

import Dispatch
import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom
import Tokenizers

#if canImport(CoreGraphics)
  import CoreGraphics
  import ImageIO
#endif
public struct ControlProgress: Sendable {
  public let stage: String
  public let stepIndex: Int
  public let totalSteps: Int
  public let fractionCompleted: Double
  public let enhancedPrompt: String?
  public init(stage: String, stepIndex: Int, totalSteps: Int, fractionCompleted: Double, enhancedPrompt: String? = nil)
  {
    self.stage = stage
    self.stepIndex = stepIndex
    self.totalSteps = totalSteps
    self.fractionCompleted = fractionCompleted
    self.enhancedPrompt = enhancedPrompt
  }
}

public typealias ControlProgressCallback = @Sendable (ControlProgress) -> Void
public struct ZImageControlGenerationRequest {
  public var prompt: String
  public var negativePrompt: String?
  public var controlImage: URL?
  #if canImport(CoreGraphics)
    public var controlImageCG: CGImage?
  #endif
  public var inpaintImage: URL?
  #if canImport(CoreGraphics)
    public var inpaintImageCG: CGImage?
  #endif
  public var maskImage: URL?
  #if canImport(CoreGraphics)
    public var maskImageCG: CGImage?
  #endif
  public var controlContextScale: Float
  public var width: Int
  public var height: Int
  public var steps: Int
  public var guidanceScale: Float
  public var seed: UInt64?
  public var outputPath: URL?
  public var model: String?
  public var weightsVariant: String?
  public var controlnetWeights: String?
  public var controlnetWeightsFile: String?
  public var maxSequenceLength: Int
  public var lora: LoRAConfiguration?
  public var progressCallback: ControlProgressCallback?
  public var enhancePrompt: Bool
  public var enhanceMaxTokens: Int
  public init(
    prompt: String,
    negativePrompt: String? = nil,
    controlImage: URL? = nil,
    inpaintImage: URL? = nil,
    maskImage: URL? = nil,
    controlContextScale: Float = 0.75,
    width: Int = ZImageModelMetadata.recommendedWidth,
    height: Int = ZImageModelMetadata.recommendedHeight,
    steps: Int = ZImageModelMetadata.recommendedInferenceSteps,
    guidanceScale: Float = ZImageModelMetadata.recommendedGuidanceScale,
    seed: UInt64? = nil,
    outputPath: URL? = nil,
    model: String? = nil,
    weightsVariant: String? = nil,
    controlnetWeights: String? = nil,
    controlnetWeightsFile: String? = nil,
    maxSequenceLength: Int = 512,
    lora: LoRAConfiguration? = nil,
    progressCallback: ControlProgressCallback? = nil,
    enhancePrompt: Bool = false,
    enhanceMaxTokens: Int = 512
  ) {
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.controlImage = controlImage
    self.inpaintImage = inpaintImage
    self.maskImage = maskImage
    self.controlContextScale = controlContextScale
    self.width = width
    self.height = height
    self.steps = steps
    self.guidanceScale = guidanceScale
    self.seed = seed
    self.outputPath = outputPath
    self.model = model
    self.weightsVariant = weightsVariant
    self.controlnetWeights = controlnetWeights
    self.controlnetWeightsFile = controlnetWeightsFile
    self.maxSequenceLength = maxSequenceLength
    self.lora = lora
    self.progressCallback = progressCallback
    self.enhancePrompt = enhancePrompt
    self.enhanceMaxTokens = enhanceMaxTokens
  }

  #if canImport(CoreGraphics)
    public init(
      prompt: String,
      negativePrompt: String? = nil,
      controlImageCG: CGImage?,
      inpaintImageCG: CGImage? = nil,
      maskImageCG: CGImage? = nil,
      controlContextScale: Float = 0.75,
      width: Int = ZImageModelMetadata.recommendedWidth,
      height: Int = ZImageModelMetadata.recommendedHeight,
      steps: Int = ZImageModelMetadata.recommendedInferenceSteps,
      guidanceScale: Float = ZImageModelMetadata.recommendedGuidanceScale,
      seed: UInt64? = nil,
      model: String? = nil,
      weightsVariant: String? = nil,
      controlnetWeights: String? = nil,
      controlnetWeightsFile: String? = nil,
      maxSequenceLength: Int = 512,
      lora: LoRAConfiguration? = nil,
      progressCallback: ControlProgressCallback? = nil,
      enhancePrompt: Bool = false,
      enhanceMaxTokens: Int = 512
    ) {
      self.prompt = prompt
      self.negativePrompt = negativePrompt
      controlImage = nil
      self.controlImageCG = controlImageCG
      inpaintImage = nil
      self.inpaintImageCG = inpaintImageCG
      maskImage = nil
      self.maskImageCG = maskImageCG
      self.controlContextScale = controlContextScale
      self.width = width
      self.height = height
      self.steps = steps
      self.guidanceScale = guidanceScale
      self.seed = seed
      outputPath = nil
      self.model = model
      self.weightsVariant = weightsVariant
      self.controlnetWeights = controlnetWeights
      self.controlnetWeightsFile = controlnetWeightsFile
      self.maxSequenceLength = maxSequenceLength
      self.lora = lora
      self.progressCallback = progressCallback
      self.enhancePrompt = enhancePrompt
      self.enhanceMaxTokens = enhanceMaxTokens
    }
  #endif
}

public class ZImageControlPipeline {
  public enum PipelineError: Error {
    case notImplemented
    case tokenizerNotLoaded
    case textEncoderNotLoaded
    case transformerNotLoaded
    case vaeNotLoaded
    case weightsMissing(String)
    case controlImageNotFound(URL)
    case controlImageLoadFailed(String)
    case outputPathRequired
  }

  private var logger: Logger
  private var tokenizer: QwenTokenizer?
  private var vae: AutoencoderKL?
  private var transformer: ZImageControlTransformer2DModel?
  private var modelConfigs: ZImageModelConfigs?
  private var quantManifest: ZImageQuantizationManifest?
  private var snapshot: URL?
  private var loadedModelId: String?
  private var loadedWeightsVariant: String?
  private var loadedControlnetWeightsId: String?
  private var currentLoRA: LoRAWeights?
  private var currentLoRAConfig: LoRAConfiguration?
  private struct CachedPromptEmbedding {
    let prompt: String
    let negativePrompt: String?
    let maxSequenceLength: Int
    let promptEmbeds: MLXArray
    let negativeEmbeds: MLXArray?
    let enhancePrompt: Bool
    let enhanceMaxTokens: Int
    let enhancedPrompt: String?
  }

  private var cachedPromptEmbedding: CachedPromptEmbedding?
  public init(logger: Logger = Logger(label: "z-image.control-pipeline")) {
    self.logger = logger
  }

  private func clearControlnetWeights() {
    guard let transformer else { return }
    logger.info("Clearing controlnet weights to free GPU memory...")
    for (_, linear) in transformer.controlAllXEmbedder {
      let zeroWeight = MLXArray.zeros(like: linear.weight)
      linear.weight._updateInternal(zeroWeight)
      if let bias = linear.bias {
        let zeroBias = MLXArray.zeros(like: bias)
        linear.bias?._updateInternal(zeroBias)
      }
    }
    for block in transformer.controlNoiseRefiner {
      zeroOutControlTransformerBlock(block)
    }
    for block in transformer.controlLayers {
      zeroOutControlTransformerBlock(block)
    }
    Memory.clearCache()
    logger.info("Controlnet weights cleared")
  }

  private func zeroOutTransformerBlock(_ block: ZImageTransformerBlock) {
    block.attention.toQ.weight._updateInternal(MLXArray.zeros(like: block.attention.toQ.weight))
    block.attention.toK.weight._updateInternal(MLXArray.zeros(like: block.attention.toK.weight))
    block.attention.toV.weight._updateInternal(MLXArray.zeros(like: block.attention.toV.weight))
    if block.attention.toOut.count > 0 {
      block.attention.toOut[0].weight._updateInternal(MLXArray.zeros(like: block.attention.toOut[0].weight))
      if let bias = block.attention.toOut[0].bias {
        block.attention.toOut[0].bias?._updateInternal(MLXArray.zeros(like: bias))
      }
    }
    if let normQ = block.attention.normQ {
      normQ.weight._updateInternal(MLXArray.zeros(like: normQ.weight))
    }
    if let normK = block.attention.normK {
      normK.weight._updateInternal(MLXArray.zeros(like: normK.weight))
    }
    if let adaLN = block.adaLN, adaLN.count > 0 {
      adaLN[0].weight._updateInternal(MLXArray.zeros(like: adaLN[0].weight))
      if let bias = adaLN[0].bias {
        adaLN[0].bias?._updateInternal(MLXArray.zeros(like: bias))
      }
    }
    block.attentionNorm1.weight._updateInternal(MLXArray.zeros(like: block.attentionNorm1.weight))
    block.ffnNorm1.weight._updateInternal(MLXArray.zeros(like: block.ffnNorm1.weight))
    block.attentionNorm2.weight._updateInternal(MLXArray.zeros(like: block.attentionNorm2.weight))
    block.ffnNorm2.weight._updateInternal(MLXArray.zeros(like: block.ffnNorm2.weight))
    block.feedForward.w1.weight._updateInternal(MLXArray.zeros(like: block.feedForward.w1.weight))
    block.feedForward.w2.weight._updateInternal(MLXArray.zeros(like: block.feedForward.w2.weight))
    block.feedForward.w3.weight._updateInternal(MLXArray.zeros(like: block.feedForward.w3.weight))
  }

  private func zeroOutControlTransformerBlock(_ block: ZImageControlTransformerBlock) {
    block.attention.toQ.weight._updateInternal(MLXArray.zeros(like: block.attention.toQ.weight))
    block.attention.toK.weight._updateInternal(MLXArray.zeros(like: block.attention.toK.weight))
    block.attention.toV.weight._updateInternal(MLXArray.zeros(like: block.attention.toV.weight))
    if block.attention.toOut.count > 0 {
      block.attention.toOut[0].weight._updateInternal(MLXArray.zeros(like: block.attention.toOut[0].weight))
      if let bias = block.attention.toOut[0].bias {
        block.attention.toOut[0].bias?._updateInternal(MLXArray.zeros(like: bias))
      }
    }
    if let normQ = block.attention.normQ {
      normQ.weight._updateInternal(MLXArray.zeros(like: normQ.weight))
    }
    if let normK = block.attention.normK {
      normK.weight._updateInternal(MLXArray.zeros(like: normK.weight))
    }
    if let adaLN = block.adaLN, adaLN.count > 0 {
      adaLN[0].weight._updateInternal(MLXArray.zeros(like: adaLN[0].weight))
      if let bias = adaLN[0].bias {
        adaLN[0].bias?._updateInternal(MLXArray.zeros(like: bias))
      }
    }
    block.attentionNorm1.weight._updateInternal(MLXArray.zeros(like: block.attentionNorm1.weight))
    block.ffnNorm1.weight._updateInternal(MLXArray.zeros(like: block.ffnNorm1.weight))
    block.attentionNorm2.weight._updateInternal(MLXArray.zeros(like: block.attentionNorm2.weight))
    block.ffnNorm2.weight._updateInternal(MLXArray.zeros(like: block.ffnNorm2.weight))
    block.feedForward.w1.weight._updateInternal(MLXArray.zeros(like: block.feedForward.w1.weight))
    block.feedForward.w2.weight._updateInternal(MLXArray.zeros(like: block.feedForward.w2.weight))
    block.feedForward.w3.weight._updateInternal(MLXArray.zeros(like: block.feedForward.w3.weight))
    if let beforeProj = block.beforeProj {
      beforeProj.weight._updateInternal(MLXArray.zeros(like: beforeProj.weight))
      if let bias = beforeProj.bias {
        beforeProj.bias?._updateInternal(MLXArray.zeros(like: bias))
      }
    }
    block.afterProj.weight._updateInternal(MLXArray.zeros(like: block.afterProj.weight))
    if let bias = block.afterProj.bias {
      block.afterProj.bias?._updateInternal(MLXArray.zeros(like: bias))
    }
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

  private func loadControlTransformer(snapshot _: URL, config: ZImageTransformerConfig) throws
    -> ZImageControlTransformer2DModel
  {
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
      axesLens: config.axesLens
    )
    return ZImageControlTransformer2DModel(configuration: controlConfig)
  }

  private func loadVAE(snapshot _: URL, config: ZImageVAEConfig) throws -> AutoencoderKL {
    AutoencoderKL(
      configuration: .init(
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

  private func encodePrompt(_ prompt: String, tokenizer: QwenTokenizer, textEncoder: QwenTextEncoder, maxLength: Int)
    throws -> (MLXArray, MLXArray)
  {
    do {
      let result = try PipelineUtilities.encodePrompt(
        prompt, tokenizer: tokenizer, textEncoder: textEncoder, maxLength: maxLength)
      return (result.embeddings, result.mask)
    } catch {
      throw PipelineError.textEncoderNotLoaded
    }
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
        let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
      else {
        throw PipelineError.controlImageLoadFailed("Failed to load image from \(url.path)")
      }
      return try loadControlImage(
        cgImage: cgImage,
        vae: vae,
        vaeConfig: vaeConfig,
        targetHeight: targetHeight,
        targetWidth: targetWidth
      )
    #else
      throw PipelineError.controlImageLoadFailed("CoreGraphics not available on this platform")
    #endif
  }

  #if canImport(CoreGraphics)
    private func loadCGImage(from url: URL) -> CGImage? {
      guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
        let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
      else {
        return nil
      }
      return cgImage
    }

    private func encodeImageToLatents(
      cgImage: CGImage,
      vae: AutoencoderKL,
      vaeConfig: ZImageVAEConfig,
      pixelH: Int,
      pixelW: Int
    ) throws -> MLXArray {
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
      return (latents - vaeConfig.shiftFactor) * vaeConfig.scalingFactor
    }

    private func convertToRGBA(_ image: CGImage) -> CGImage? {
      let width = image.width
      let height = image.height
      let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
      let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
      guard
        let context = CGContext(
          data: nil,
          width: width,
          height: height,
          bitsPerComponent: 8,
          bytesPerRow: width * 4,
          space: colorSpace,
          bitmapInfo: bitmapInfo
        )
      else {
        return nil
      }
      context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
      return context.makeImage()
    }

    private func processMaskToLatent(
      cgImage: CGImage,
      latentH: Int,
      latentW: Int
    ) throws -> MLXArray {
      guard let rgbaImage = convertToRGBA(cgImage) else {
        throw PipelineError.controlImageLoadFailed("Failed to convert mask to RGBA")
      }
      let maskArray = try QwenImageIO.resizedPixelArray(
        from: rgbaImage,
        width: latentW,
        height: latentH,
        addBatchDimension: true,
        dtype: .float32,
        interpolation: .none
      )
      guard maskArray.ndim == 4 else {
        throw PipelineError.controlImageLoadFailed("Mask array has wrong dimensions: \(maskArray.shape)")
      }
      let grayscale = MLX.mean(maskArray, axis: 1, keepDims: true)
      return MLX.where(grayscale .>= 0.5, MLXArray(Float(1.0)), MLXArray(Float(0.0)))
    }

    private func buildControlContext(
      controlImage: CGImage?,
      inpaintImage: CGImage?,
      maskImage: CGImage?,
      vae: AutoencoderKL,
      vaeConfig: ZImageVAEConfig,
      targetHeight: Int,
      targetWidth: Int
    ) throws -> MLXArray {
      let vaeDivisor = vaeConfig.latentDivisor
      let latentH = max(1, targetHeight / vaeDivisor)
      let latentW = max(1, targetWidth / vaeDivisor)
      let pixelH = latentH * vaeDivisor
      let pixelW = latentW * vaeDivisor
      let controlLatents: MLXArray =
        if let control = controlImage {
          try encodeImageToLatents(
            cgImage: control,
            vae: vae,
            vaeConfig: vaeConfig,
            pixelH: pixelH,
            pixelW: pixelW
          )
        } else {
          MLXArray.zeros([1, 16, latentH, latentW])
        }
      let pixelMask: MLXArray?
      if let mask = maskImage {
        guard let rgbaMask = convertToRGBA(mask) else {
          throw PipelineError.controlImageLoadFailed("Failed to convert mask to RGBA")
        }
        let maskPixels = try QwenImageIO.resizedPixelArray(
          from: rgbaMask,
          width: pixelW,
          height: pixelH,
          addBatchDimension: true,
          dtype: .float32,
          interpolation: .high
        )
        let grayscaleMask = MLX.mean(maskPixels, axis: 1, keepDims: true)
        pixelMask = MLX.where(grayscaleMask .>= 0.5, MLXArray(Float(1.0)), MLXArray(Float(0.0)))
      } else {
        pixelMask = nil
      }
      let inpaintLatents: MLXArray
      if let inpaint = inpaintImage {
        guard let rgbaInpaint = convertToRGBA(inpaint) else {
          throw PipelineError.controlImageLoadFailed("Failed to convert inpaint image to RGBA")
        }
        let inpaintPixels = try QwenImageIO.resizedPixelArray(
          from: rgbaInpaint,
          width: pixelW,
          height: pixelH,
          addBatchDimension: true,
          dtype: .float32
        )
        let normalized = (inpaintPixels * 2.0) - 1.0
        var maskedNormalized = normalized
        if let mask = pixelMask {
          let keepMask = MLX.less(mask, MLXArray(Float(0.5)))
          maskedNormalized = normalized * keepMask.asType(normalized.dtype)
        }
        MLX.eval(maskedNormalized)
        let shiftFactor = MLXArray(vaeConfig.shiftFactor)
        let scaleFactor = MLXArray(vaeConfig.scalingFactor)
        let latentChannels = vaeConfig.latentChannels
        let encoded = vae.encode(maskedNormalized)
        let latents = encoded[0..., 0..<latentChannels, 0..., 0...]
        inpaintLatents = (latents - shiftFactor) * scaleFactor
      } else {
        inpaintLatents = MLXArray.zeros([1, 16, latentH, latentW])
      }
      let maskCondition: MLXArray
      if let mask = pixelMask {
        let invertedMask = 1.0 - mask
        var nhwc = invertedMask.transposed(0, 2, 3, 1)
        let hScale = Float(latentH) / Float(pixelH)
        let wScale = Float(latentW) / Float(pixelW)
        nhwc = MLXNN.Upsample(scaleFactor: .array([hScale, wScale]), mode: .nearest)(nhwc)
        maskCondition = nhwc.transposed(0, 3, 1, 2)
      } else {
        maskCondition = MLXArray.zeros([1, 1, latentH, latentW])
      }
      let combined = MLX.concatenated([controlLatents, maskCondition, inpaintLatents], axis: 1)
      return MLX.expandedDimensions(combined, axis: 2)
    }

    private func loadControlImage(
      cgImage: CGImage,
      vae: AutoencoderKL,
      vaeConfig: ZImageVAEConfig,
      targetHeight: Int,
      targetWidth: Int
    ) throws -> MLXArray {
      try buildControlContext(
        controlImage: cgImage,
        inpaintImage: nil,
        maskImage: nil,
        vae: vae,
        vaeConfig: vaeConfig,
        targetHeight: targetHeight,
        targetWidth: targetWidth
      )
    }
  #endif
  private func applyLoRAIfNeeded(_ requestedConfig: LoRAConfiguration?) async throws {
    guard let transformer else {
      throw PipelineError.transformerNotLoaded
    }
    if let currentConfig = currentLoRAConfig, let requestedConfig, currentConfig == requestedConfig {
      logger.info("LoRA already loaded with same configuration, skipping")
      return
    }
    if currentLoRA != nil {
      logger.info("Clearing previous LoRA...")
      if let lora = currentLoRA, let config = currentLoRAConfig, lora.hasLoKr {
        LoRAApplicator.removeLoKr(from: transformer, loraWeights: lora, scale: config.scale, logger: logger)
      }
      LoRAApplicator.clearDynamicLoRA(from: transformer, logger: logger)
      currentLoRA = nil
      currentLoRAConfig = nil
    }
    if let config = requestedConfig {
      logger.info("Loading LoRA from \(config.source.displayName)...")
      let loraWeights = try await LoRAWeightLoader.load(from: config)
      logger.info("Loaded LoRA: rank=\(loraWeights.rank), alpha=\(loraWeights.alpha), layers=\(loraWeights.layerCount)")
      LoRAApplicator.applyDynamically(to: transformer, loraWeights: loraWeights, scale: config.scale, logger: logger)
      currentLoRA = loraWeights
      currentLoRAConfig = config
      logger.info("LoRA applied successfully with scale=\(config.scale)")
    }
  }

  public func unloadLoRA() {
    guard let transformer else { return }
    if currentLoRA != nil {
      if let lora = currentLoRA, let config = currentLoRAConfig, lora.hasLoKr {
        LoRAApplicator.removeLoKr(from: transformer, loraWeights: lora, scale: config.scale, logger: logger)
      }
      LoRAApplicator.clearDynamicLoRA(from: transformer, logger: logger)
      currentLoRA = nil
      currentLoRAConfig = nil
      Memory.clearCache()
      logger.info("LoRA unloaded")
    }
  }

  public func unloadTransformer() {
    transformer = nil
    currentLoRA = nil
    currentLoRAConfig = nil
    loadedControlnetWeightsId = nil
    Memory.clearCache()
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

  public var hasLoRALoaded: Bool {
    currentLoRA != nil
  }

  public var loadedLoRAConfig: LoRAConfiguration? {
    currentLoRAConfig
  }

  public func generate(_ request: ZImageControlGenerationRequest) async throws -> URL {
    logger.info("Requested Z-Image control generation")
    let decoded = try await generateCore(request)
    guard let outputPath = request.outputPath else {
      throw PipelineError.outputPathRequired
    }
    try QwenImageIO.saveImage(array: decoded, to: outputPath)
    logger.info("Wrote image to \(outputPath.path)")
    return outputPath
  }

  // swiftlint:disable:next cyclomatic_complexity
  private func generateCore(_ request: ZImageControlGenerationRequest) async throws -> MLXArray {
    let requestedModelId = request.model ?? ZImageRepository.id
    let requestedWeightsVariant = ZImageFiles.normalizedWeightsVariant(request.weightsVariant)
    let requestedControlnetId = request.controlnetWeights
    let needsModelReload = (loadedModelId != requestedModelId) || (loadedWeightsVariant != requestedWeightsVariant)
    let needsControlnetReload = (loadedControlnetWeightsId != requestedControlnetId)
    if needsModelReload {
      let canPreserveSharedComponents =
        loadedModelId != nil
        && loadedModelId != requestedModelId
        && loadedWeightsVariant == requestedWeightsVariant
        && ZImageModelRegistry.areZImageVariants(loadedModelId ?? "", requestedModelId)
      if canPreserveSharedComponents {
        logger.info("Switching Z-Image variant, preserving VAE and tokenizer")
        self.transformer = nil
        self.modelConfigs = nil
        self.quantManifest = nil
        self.snapshot = nil
        loadedWeightsVariant = nil
        currentLoRA = nil
        currentLoRAConfig = nil
        cachedPromptEmbedding = nil
        Memory.clearCache()
      } else {
        self.tokenizer = nil
        self.vae = nil
        self.transformer = nil
        self.modelConfigs = nil
        self.quantManifest = nil
        self.snapshot = nil
        loadedWeightsVariant = nil
        currentLoRA = nil
        currentLoRAConfig = nil
        cachedPromptEmbedding = nil
        Memory.clearCache()
      }
      logger.info("Loading model \(requestedModelId)...")
      let snapshotValidator: (@Sendable (URL) -> Bool)? =
        if let requestedWeightsVariant {
          { [logger] snapshot in
            !ZImageFiles.resolveTransformerWeights(
              at: snapshot, weightsVariant: requestedWeightsVariant, logger: logger
            ).isEmpty
              && !ZImageFiles.resolveTextEncoderWeights(
                at: snapshot, weightsVariant: requestedWeightsVariant, logger: logger
              ).isEmpty
              && !ZImageFiles.resolveVAEWeights(at: snapshot, weightsVariant: requestedWeightsVariant, logger: logger)
                .isEmpty
          }
        } else {
          nil
        }
      let snapshot = try await PipelineSnapshot.prepare(
        model: request.model,
        weightsVariant: requestedWeightsVariant,
        snapshotValidator: snapshotValidator,
        logger: logger
      )
      let modelConfigs = try ZImageModelConfigs.load(from: snapshot)
      let weightsMapper = ZImageWeightsMapper(
        snapshot: snapshot, weightsVariant: requestedWeightsVariant, logger: logger)
      let quantManifest = weightsMapper.loadQuantizationManifest()
      if quantManifest == nil {
        try ZImageFiles.validateRequiredComponentWeights(
          at: snapshot, weightsVariant: requestedWeightsVariant, logger: logger)
      }
      if let manifest = quantManifest {
        logger.info("Loading quantized model (bits=\(manifest.bits), group_size=\(manifest.groupSize))")
      }
      self.snapshot = snapshot
      self.modelConfigs = modelConfigs
      self.quantManifest = quantManifest
      if self.tokenizer == nil {
        logger.info("Loading tokenizer...")
        self.tokenizer = try loadTokenizer(snapshot: snapshot)
      } else {
        logger.info("Reusing cached tokenizer")
      }

      if self.vae == nil {
        logger.info("Loading VAE...")
        let vae = try loadVAE(snapshot: snapshot, config: modelConfigs.vae)
        let vaeWeights = try weightsMapper.loadVAE()
        ZImageWeightsMapping.applyVAE(weights: vaeWeights, to: vae, manifest: quantManifest, logger: logger)
        self.vae = vae
      } else {
        logger.info("Reusing cached VAE")
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
      self.transformer = transformer
      loadedModelId = requestedModelId
      loadedWeightsVariant = requestedWeightsVariant
      loadedControlnetWeightsId = nil
    } else if transformer == nil {
      logger.info("Reloading transformer (cache preserved)...")
      guard let snapshot = self.snapshot,
        let modelConfigs = self.modelConfigs
      else {
        throw PipelineError.transformerNotLoaded
      }
      let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, weightsVariant: loadedWeightsVariant, logger: logger)
      logger.info("Loading control transformer...")
      let transformer = try loadControlTransformer(snapshot: snapshot, config: modelConfigs.transformer)
      let transformerWeights = try weightsMapper.loadTransformer()
      ZImageControlWeightsMapping.applyControlTransformer(
        weights: transformerWeights,
        to: transformer,
        manifest: quantManifest,
        logger: logger
      )
      self.transformer = transformer
      loadedControlnetWeightsId = nil
    } else {
      logger.info("Reusing cached model \(requestedModelId)")
    }
    if needsControlnetReload || needsModelReload {
      if let controlnetSpec = requestedControlnetId {
        logger.info("Loading controlnet weights from \(controlnetSpec)...")
        let result = try await loadControlnetWeights(
          controlnetSpec: controlnetSpec,
          preferredFile: request.controlnetWeightsFile,
          progressCallback: request.progressCallback
        )
        ZImageControlWeightsMapping.applyControlnetWeights(
          weights: result.weights,
          to: transformer!,
          manifest: result.manifest,
          logger: logger
        )
        loadedControlnetWeightsId = controlnetSpec
      } else {
        if loadedControlnetWeightsId != nil {
          clearControlnetWeights()
        }
        loadedControlnetWeightsId = nil
      }
    } else if requestedControlnetId != nil {
      logger.info("Reusing cached controlnet weights")
    }
    try await applyLoRAIfNeeded(request.lora)
    guard let snapshot,
      let modelConfigs,
      let tokenizer,
      let vae
    else {
      throw PipelineError.transformerNotLoaded
    }
    let doCFG = request.guidanceScale > 1.0
    let promptEmbeds: MLXArray
    let negativeEmbeds: MLXArray?
    if let cached = cachedPromptEmbedding,
      cached.prompt == request.prompt,
      cached.negativePrompt == request.negativePrompt,
      cached.maxSequenceLength == request.maxSequenceLength,
      cached.enhancePrompt == request.enhancePrompt,
      cached.enhanceMaxTokens == request.enhanceMaxTokens
    {
      logger.info("Reusing cached prompt embeddings")
      promptEmbeds = cached.promptEmbeds
      negativeEmbeds = cached.negativeEmbeds
      if let enhancedPrompt = cached.enhancedPrompt {
        request.progressCallback?(
          ControlProgress(
            stage: "Prompt enhanced",
            stepIndex: 0, totalSteps: 0, fractionCompleted: 0,
            enhancedPrompt: enhancedPrompt
          ))
      }
    } else {
      if request.enhancePrompt, transformer != nil {
        let availableMemory = getAvailableMemory()
        let textEncoderSize: UInt64 = 6 * 1024 * 1024 * 1024
        if availableMemory < textEncoderSize {
          logger.info(
            "Low memory (\(availableMemory / 1024 / 1024)MB available), offloading transformer before enhancement...")
          unloadTransformer()
        } else {
          logger.info("Sufficient memory (\(availableMemory / 1024 / 1024)MB available), keeping transformer loaded")
        }
      }
      request.progressCallback?(
        ControlProgress(
          stage: "Loading text encoder",
          stepIndex: 0, totalSteps: 0, fractionCompleted: 0
        ))
      logger.info("Loading text encoder...")
      let textEncoder = try loadTextEncoder(snapshot: snapshot, config: modelConfigs.textEncoder)
      let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, weightsVariant: loadedWeightsVariant, logger: logger)
      let textEncoderWeights = try weightsMapper.loadTextEncoder()
      ZImageWeightsMapping.applyTextEncoder(
        weights: textEncoderWeights, to: textEncoder, manifest: quantManifest, logger: logger)
      var finalPrompt = request.prompt
      var enhancedPromptForCache: String? = nil
      if request.enhancePrompt {
        request.progressCallback?(
          ControlProgress(
            stage: "Enhancing prompt",
            stepIndex: 0, totalSteps: 0, fractionCompleted: 0
          ))
        logger.info("Enhancing prompt using LLM (max tokens: \(request.enhanceMaxTokens))...")
        let enhanceConfig = PromptEnhanceConfig(maxNewTokens: request.enhanceMaxTokens)
        let enhanced = try textEncoder.enhancePrompt(request.prompt, tokenizer: tokenizer, config: enhanceConfig)
        if enhanced.isEmpty {
          logger.warning("Prompt enhancement incomplete (need more tokens), using original prompt")
        } else {
          logger.info("Enhanced prompt: \(enhanced)")
          finalPrompt = enhanced
          enhancedPromptForCache = enhanced
          request.progressCallback?(
            ControlProgress(
              stage: "Prompt enhanced",
              stepIndex: 0, totalSteps: 0, fractionCompleted: 0,
              enhancedPrompt: enhanced
            ))
        }
        Memory.clearCache()
      }
      let (pe, _) = try encodePrompt(
        finalPrompt, tokenizer: tokenizer, textEncoder: textEncoder, maxLength: request.maxSequenceLength)
      promptEmbeds = pe
      if doCFG {
        let (ne, _) = try encodePrompt(
          request.negativePrompt ?? "", tokenizer: tokenizer, textEncoder: textEncoder,
          maxLength: request.maxSequenceLength)
        negativeEmbeds = ne
        MLX.eval(promptEmbeds, ne)
      } else {
        negativeEmbeds = nil
        MLX.eval(promptEmbeds)
      }
      cachedPromptEmbedding = CachedPromptEmbedding(
        prompt: request.prompt,
        negativePrompt: request.negativePrompt,
        maxSequenceLength: request.maxSequenceLength,
        promptEmbeds: promptEmbeds,
        negativeEmbeds: negativeEmbeds,
        enhancePrompt: request.enhancePrompt,
        enhanceMaxTokens: request.enhanceMaxTokens,
        enhancedPrompt: enhancedPromptForCache
      )
      logger.info("Text encoding complete, embeddings cached")
    }
    var controlContext: MLXArray? = nil
    #if canImport(CoreGraphics)
      let controlCG: CGImage? = request.controlImageCG ?? (request.controlImage.flatMap { loadCGImage(from: $0) })
      let inpaintCG: CGImage? = request.inpaintImageCG ?? (request.inpaintImage.flatMap { loadCGImage(from: $0) })
      let maskCG: CGImage? = request.maskImageCG ?? (request.maskImage.flatMap { loadCGImage(from: $0) })
      if controlCG != nil || inpaintCG != nil || maskCG != nil {
        logger.info(
          "Building control context (control=\(controlCG != nil), inpaint=\(inpaintCG != nil), mask=\(maskCG != nil))..."
        )
        let result = try buildControlContext(
          controlImage: controlCG,
          inpaintImage: inpaintCG,
          maskImage: maskCG,
          vae: vae,
          vaeConfig: modelConfigs.vae,
          targetHeight: request.height,
          targetWidth: request.width
        )
        MLX.eval(result)
        logger.info("Control context built, shape: \(result.shape)")
        controlContext = result.asType(.bfloat16)
      }
    #else
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
    #endif
    let vaeDivisor = modelConfigs.vae.latentDivisor
    let latentH = max(1, request.height / vaeDivisor)
    let latentW = max(1, request.width / vaeDivisor)
    let shape: [Int] = [1, ZImageModelMetadata.Transformer.inChannels, latentH, latentW]
    let randomKey: MLXArray? = request.seed.map { MLXRandom.key($0) }
    let initialNoise = MLXRandom.normal(shape, loc: 0, scale: 1, key: randomKey)
    var latents = initialNoise
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
    if transformer == nil {
      logger.info("Reloading transformer after prompt encoding...")
      let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, weightsVariant: loadedWeightsVariant, logger: logger)
      let transformerModel = try loadControlTransformer(snapshot: snapshot, config: modelConfigs.transformer)
      let transformerWeights = try weightsMapper.loadTransformer()
      ZImageControlWeightsMapping.applyControlTransformer(
        weights: transformerWeights,
        to: transformerModel,
        manifest: quantManifest,
        logger: logger
      )
      transformer = transformerModel
      loadedControlnetWeightsId = nil
      if let controlnetSpec = request.controlnetWeights {
        logger.info("Reloading controlnet weights...")
        let result = try await loadControlnetWeights(
          controlnetSpec: controlnetSpec,
          preferredFile: request.controlnetWeightsFile,
          progressCallback: request.progressCallback
        )
        ZImageControlWeightsMapping.applyControlnetWeights(
          weights: result.weights,
          to: transformer!,
          manifest: result.manifest,
          logger: logger
        )
        loadedControlnetWeightsId = controlnetSpec
      }
      if let loraConfig = request.lora {
        try await applyLoRAIfNeeded(loraConfig)
      }
    }
    logger.info("Running \(request.steps) denoising steps with control_context_scale=\(request.controlContextScale)...")
    do {
      guard let transformer else {
        throw PipelineError.transformerNotLoaded
      }
      for stepIndex in 0..<request.steps {
        try Task.checkCancellation()
        request.progressCallback?(
          ControlProgress(
            stage: "Denoising",
            stepIndex: stepIndex,
            totalSteps: request.steps,
            fractionCompleted: Double(stepIndex) / Double(request.steps)
          ))
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
        let guidedNoise: MLXArray
        if doCFG, negativeEmbeds != nil {
          let batch = latents.dim(0)
          let positive = noisePred[0..<batch, 0..., 0..., 0...]
          let negative = noisePred[batch..<batch * 2, 0..., 0..., 0...]
          let guidanceDelta = subtract(positive, negative)
          guidedNoise = add(positive, multiply(request.guidanceScale, guidanceDelta))
        } else {
          guidedNoise = noisePred
        }
        latents = scheduler.step(modelOutput: -guidedNoise, timestepIndex: stepIndex, sample: latents)
        MLX.eval(latents)
      }
      transformer.clearCache()
    }
    unloadTransformer()
    request.progressCallback?(
      ControlProgress(
        stage: "Denoising",
        stepIndex: request.steps,
        totalSteps: request.steps,
        fractionCompleted: 1.0
      ))
    request.progressCallback?(
      ControlProgress(
        stage: "Decoding",
        stepIndex: request.steps,
        totalSteps: request.steps,
        fractionCompleted: 1.0
      ))
    logger.info("Denoising complete, decoding latents...")
    return PipelineUtilities.decodeLatents(latents, vae: vae, height: request.height, width: request.width)
  }

  #if canImport(CoreGraphics)
    public func generateToMemory(_ request: ZImageControlGenerationRequest) async throws -> Data {
      logger.info("Requested Z-Image control generation (to memory)")
      let decoded = try await generateCore(request)
      let imageData = try QwenImageIO.imageData(from: decoded)
      logger.info("Generated image data (\(imageData.count) bytes)")
      return imageData
    }
  #endif

  private struct ControlnetWeightsResult {
    let weights: [String: MLXArray]
    let manifest: ZImageQuantizationManifest?
  }

  private func loadControlnetWeights(
    controlnetSpec: String,
    preferredFile: String? = nil,
    dtype: DType = .bfloat16,
    progressCallback: ControlProgressCallback? = nil
  ) async throws -> ControlnetWeightsResult {
    let fm = FileManager.default
    let localURL = URL(fileURLWithPath: controlnetSpec)
    if fm.fileExists(atPath: localURL.path), controlnetSpec.hasSuffix(".safetensors") {
      progressCallback?(
        ControlProgress(
          stage: "Loading ControlNet",
          stepIndex: 0, totalSteps: 0, fractionCompleted: 0
        ))
      logger.info("Loading controlnet from local file: \(controlnetSpec)")
      let weights = try loadSafetensorsFile(url: localURL, dtype: dtype)
      return ControlnetWeightsResult(weights: weights, manifest: nil)
    }
    var isDirectory: ObjCBool = false
    if fm.fileExists(atPath: localURL.path, isDirectory: &isDirectory), isDirectory.boolValue {
      progressCallback?(
        ControlProgress(
          stage: "Loading ControlNet",
          stepIndex: 0, totalSteps: 0, fractionCompleted: 0
        ))
      logger.info("Loading controlnet from local directory: \(controlnetSpec)")
      return try loadControlnetFromDirectory(localURL, dtype: dtype, preferredFile: preferredFile)
    }
    if ModelResolution.isHuggingFaceModelId(controlnetSpec) {
      logger.info("Resolving controlnet from HuggingFace: \(controlnetSpec)")
      let snapshot = try await ModelResolution.resolve(
        modelSpec: controlnetSpec,
        filePatterns: ["*.safetensors", "*.json"],
        progressHandler: { [logger] progress in
          let percent = Int(progress.fractionCompleted * 100)
          logger.info("Downloading controlnet: \(percent)%")
          progressCallback?(
            ControlProgress(
              stage: "Downloading ControlNet",
              stepIndex: 0,
              totalSteps: 0,
              fractionCompleted: progress.fractionCompleted
            ))
        }
      )
      return try loadControlnetFromDirectory(snapshot, dtype: dtype, preferredFile: preferredFile)
    }
    throw PipelineError.weightsMissing(
      """
      Invalid controlnet spec: \(controlnetSpec).
       Provide a local .safetensors path, directory,
      or HuggingFace model ID (e.g., alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1)
      """)
  }

  private func loadControlnetFromDirectory(_ directory: URL, dtype: DType, preferredFile: String? = nil) throws
    -> ControlnetWeightsResult
  {
    let fm = FileManager.default
    var manifest: ZImageQuantizationManifest? = nil
    if let loadedManifest = try ZImageQuantizer.loadControlnetManifest(from: directory) {
      logger.info(
        "Found quantized controlnet manifest: \(loadedManifest.bits)-bit, group_size=\(loadedManifest.groupSize)")
      manifest = loadedManifest
    }
    let contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    var safetensorsFiles = contents.filter { $0.pathExtension == "safetensors" }
    if let preferredFile {
      safetensorsFiles = safetensorsFiles.filter { $0.lastPathComponent == preferredFile }
      if safetensorsFiles.isEmpty {
        throw PipelineError.weightsMissing(
          "Specified controlnet file '\(preferredFile)' not found in directory: \(directory.path)")
      }
      logger.info("Using specified controlnet file: \(preferredFile)")
    }
    guard !safetensorsFiles.isEmpty else {
      throw PipelineError.weightsMissing("No .safetensors file found in controlnet directory: \(directory.path)")
    }
    let preserveQuantized = manifest != nil
    var allWeights: [String: MLXArray] = [:]
    for file in safetensorsFiles {
      logger.info("Loading controlnet weights from \(file.lastPathComponent)")
      let weights = try loadSafetensorsFile(url: file, dtype: dtype, preserveQuantized: preserveQuantized)
      for (key, value) in weights {
        allWeights[key] = value
      }
    }
    return ControlnetWeightsResult(weights: allWeights, manifest: manifest)
  }

  private func loadSafetensorsFile(url: URL, dtype: DType, preserveQuantized: Bool = false) throws -> [String: MLXArray]
  {
    var tensors: [String: MLXArray] = [:]
    let reader = try SafeTensorsReader(fileURL: url)
    for meta in reader.allMetadata() {
      var tensor = try reader.tensor(named: meta.name)
      let isQuantizedWeight = preserveQuantized && (tensor.dtype == .uint32)
      let isScalesOrBiases = preserveQuantized && (meta.name.hasSuffix(".scales") || meta.name.hasSuffix(".biases"))
      if !isQuantizedWeight, !isScalesOrBiases, tensor.dtype != dtype {
        tensor = tensor.asType(dtype)
      }
      tensors[meta.name] = tensor
    }
    logger.info("Loaded \(tensors.count) controlnet tensors")
    return tensors
  }
}
