import Foundation
import MLX
import MLXNN

public struct ZImageControlNetConfig {
  public var inChannels: Int = 16
  public var dim: Int = 3840
  public var nLayers: Int = 30
  public var nRefinerLayers: Int = 2
  public var nHeads: Int = 30
  public var nKVHeads: Int = 30
  public var normEps: Float = 1e-5
  public var qkNorm: Bool = true
  public var capFeatDim: Int = 2560
  public var ropeTheta: Float = 256.0
  public var tScale: Float = 1000.0
  public var axesDims: [Int] = [32, 48, 48]
  public var axesLens: [Int] = [1024, 512, 512]
  public var controlLayersPlaces: [Int] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
  public var controlRefinerLayersPlaces: [Int] = [0, 1]
  public var controlInDim: Int = 33
  public var addControlNoiseRefiner: Bool = true

  public init() {}

  public init(
    inChannels: Int = 16,
    dim: Int = 3840,
    nLayers: Int = 30,
    nRefinerLayers: Int = 2,
    nHeads: Int = 30,
    nKVHeads: Int = 30,
    normEps: Float = 1e-5,
    qkNorm: Bool = true,
    capFeatDim: Int = 2560,
    ropeTheta: Float = 256.0,
    tScale: Float = 1000.0,
    axesDims: [Int] = [32, 48, 48],
    axesLens: [Int] = [1024, 512, 512],
    controlLayersPlaces: [Int] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    controlRefinerLayersPlaces: [Int] = [0, 1],
    controlInDim: Int = 33,
    addControlNoiseRefiner: Bool = true
  ) {
    self.inChannels = inChannels
    self.dim = dim
    self.nLayers = nLayers
    self.nRefinerLayers = nRefinerLayers
    self.nHeads = nHeads
    self.nKVHeads = nKVHeads
    self.normEps = normEps
    self.qkNorm = qkNorm
    self.capFeatDim = capFeatDim
    self.ropeTheta = ropeTheta
    self.tScale = tScale
    self.axesDims = axesDims
    self.axesLens = axesLens
    self.controlLayersPlaces = controlLayersPlaces
    self.controlRefinerLayersPlaces = controlRefinerLayersPlaces
    self.controlInDim = controlInDim
    self.addControlNoiseRefiner = addControlNoiseRefiner
  }

  public init(transformerConfig: ZImageTransformerConfig) {
    self.init(
      inChannels: transformerConfig.inChannels,
      dim: transformerConfig.dim,
      nLayers: transformerConfig.nLayers,
      nRefinerLayers: transformerConfig.nRefinerLayers,
      nHeads: transformerConfig.nHeads,
      nKVHeads: transformerConfig.nKVHeads,
      normEps: transformerConfig.normEps,
      qkNorm: transformerConfig.qkNorm,
      capFeatDim: transformerConfig.capFeatDim,
      ropeTheta: transformerConfig.ropeTheta,
      tScale: transformerConfig.tScale,
      axesDims: transformerConfig.axesDims,
      axesLens: transformerConfig.axesLens
    )
  }
}

public typealias ZImageControlTransformerConfig = ZImageControlNetConfig

public final class ZImageControlNetModel: Module {
  public let configuration: ZImageControlNetConfig

  let tEmbedder: ZImageTimestepEmbedder
  let allXEmbedder: [String: Linear]
  let noiseRefiner: [ZImageTransformerBlock]
  let contextRefiner: [ZImageTransformerBlock]
  let capEmbedNorm: RMSNorm
  let capEmbedLinear: Linear
  let ropeEmbedder: ZImageRopeEmbedder
  let xPadToken: MLXArray?
  let capPadToken: MLXArray?

  let controlLayersPlaces: [Int]
  let controlRefinerLayersPlaces: [Int]
  let addControlNoiseRefiner: Bool

  @ModuleInfo(key: "control_all_x_embedder") var controlAllXEmbedder: [String: Linear]
  @ModuleInfo(key: "control_noise_refiner") public internal(set) var controlNoiseRefiner:
    [ZImageControlTransformerBlock]
  @ModuleInfo(key: "control_layers") public internal(set) var controlLayers: [ZImageControlTransformerBlock]

  private var cache: TransformerCache?
  private var cacheKey: TransformerCacheKey?

  public init(configuration: ZImageControlNetConfig, sharedTransformer transformer: ZImageTransformer2DModel) {
    self.configuration = configuration
    self.tEmbedder = transformer.tEmbedder
    self.allXEmbedder = transformer.allXEmbedder
    self.noiseRefiner = transformer.noiseRefiner
    self.contextRefiner = transformer.contextRefiner
    self.capEmbedNorm = transformer.capEmbedNorm
    self.capEmbedLinear = transformer.capEmbedLinear
    self.ropeEmbedder = transformer.ropeEmbedder
    self.xPadToken = transformer.sharedXPadToken
    self.capPadToken = transformer.sharedCapPadToken
    self.controlLayersPlaces = configuration.controlLayersPlaces
    self.controlRefinerLayersPlaces = configuration.controlRefinerLayersPlaces
    self.addControlNoiseRefiner = configuration.addControlNoiseRefiner

    let patchSize = 2
    let fPatchSize = 1
    let key = "\(patchSize)-\(fPatchSize)"

    var controlXEmbedder: [String: Linear] = [:]
    let controlInFeatures = fPatchSize * patchSize * patchSize * configuration.controlInDim
    controlXEmbedder[key] = Linear(controlInFeatures, configuration.dim, bias: true)
    self._controlAllXEmbedder.wrappedValue = controlXEmbedder

    var controlNoiseBlocks: [ZImageControlTransformerBlock] = []
    for idx in 0..<configuration.controlRefinerLayersPlaces.count {
      controlNoiseBlocks.append(
        ZImageControlTransformerBlock(
          blockId: idx,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm
        )
      )
    }
    self._controlNoiseRefiner.wrappedValue = controlNoiseBlocks

    var controlLayerBlocks: [ZImageControlTransformerBlock] = []
    for (idx, _) in configuration.controlLayersPlaces.enumerated() {
      controlLayerBlocks.append(
        ZImageControlTransformerBlock(
          blockId: idx,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm
        )
      )
    }
    self._controlLayers.wrappedValue = controlLayerBlocks

    super.init()
  }

  public func loadControlXEmbedderWeights(from weights: [String: MLXArray]) {
    let key = "2-1"
    let prefix = "control_all_x_embedder.\(key)"
    guard let linear = controlAllXEmbedder[key] else { return }

    if let w = weights["\(prefix).weight"] {
      linear.weight._updateInternal(w)
    }
    if let b = weights["\(prefix).bias"] {
      linear.bias?._updateInternal(b)
    }
  }

  public func clearCache() {
    cache = nil
    cacheKey = nil
  }

  private func getOrBuildCache(
    batch: Int,
    height: Int,
    width: Int,
    frames: Int,
    capOriLen: Int,
    patchSize: Int,
    fPatchSize: Int
  ) -> TransformerCache {
    let key = TransformerCacheKey(
      batch: batch,
      height: height,
      width: width,
      frames: frames,
      capOriLen: capOriLen
    )

    if let existingKey = cacheKey, let existingCache = cache, existingKey == key {
      return existingCache
    }

    let newCache = TransformerCacheBuilder.build(
      batch: batch,
      height: height,
      width: width,
      frames: frames,
      capOriLen: capOriLen,
      patchSize: patchSize,
      fPatchSize: fPatchSize,
      ropeEmbedder: ropeEmbedder
    )

    cache = newCache
    cacheKey = key

    return newCache
  }

  private func embedControlContext(
    controlContext: MLXArray,
    patchSize: Int = 2,
    fPatchSize: Int = 1
  ) -> MLXArray {
    let key = "\(patchSize)-\(fPatchSize)"
    guard let controlXEmbed = controlAllXEmbedder[key] else {
      fatalError("Control embedder not found for key: \(key)")
    }

    let batch = controlContext.dim(0)
    let channels = controlContext.dim(1)
    let frames = controlContext.dim(2)
    let height = controlContext.dim(3)
    let width = controlContext.dim(4)

    let fTokens = frames / fPatchSize
    let hTokens = height / patchSize
    let wTokens = width / patchSize
    let controlTokens = fTokens * hTokens * wTokens

    var controlImage =
      controlContext
      .reshaped(batch, channels, fTokens, fPatchSize, hTokens, patchSize, wTokens, patchSize)
      .transposed(0, 2, 4, 6, 3, 5, 7, 1)
      .reshaped(batch, controlTokens, patchSize * patchSize * fPatchSize * channels)

    let seqMultiOf = 32
    let controlPad = (seqMultiOf - (controlTokens % seqMultiOf)) % seqMultiOf
    if controlPad > 0 {
      controlImage = padSequenceByRepeatingLastToken(
        controlImage,
        validLength: controlTokens,
        padLength: controlPad
      )
    }

    var controlEmbed = controlXEmbed(controlImage)

    if let xPadToken, controlPad > 0 {
      let padDim = xPadToken.dim(xPadToken.ndim - 1)
      let controlSeqLen = controlTokens + controlPad
      let padMask1d = MLX.concatenated(
        [
          MLX.zeros([controlTokens], dtype: .bool),
          MLX.ones([controlPad], dtype: .bool),
        ],
        axis: 0
      )
      let padMask = MLX.broadcast(padMask1d.reshaped(1, controlSeqLen), to: [batch, controlSeqLen])
      let pad = MLX.broadcast(xPadToken.reshaped(1, 1, padDim), to: [batch, controlSeqLen, padDim])
      controlEmbed = MLX.where(MLX.expandedDimensions(padMask, axis: 2), pad, controlEmbed)
    }

    return controlEmbed
  }

  private func forwardControlRefiner(
    noiseStream: MLXArray,
    controlEmbed: MLXArray,
    imgFreqs: MLXArray,
    tEmb: MLXArray,
    conditioningScale: Float
  ) -> (refinerHints: ZImageControlBlockSamples, refinedControl: MLXArray) {
    guard !controlNoiseRefiner.isEmpty else {
      return ([:], controlEmbed)
    }

    let adalnInput = tEmb.asType(controlEmbed.dtype)

    var controlState = ZImageControlHintState(control: controlEmbed)
    for refinerBlock in controlNoiseRefiner {
      controlState = refinerBlock(
        controlState,
        x: noiseStream,
        attnMask: nil,
        freqsCis: imgFreqs,
        adalnInput: adalnInput
      )
    }

    return (
      controlState.scaledHints(
        layerPlaces: controlRefinerLayersPlaces,
        conditioningScale: conditioningScale
      ),
      controlState.control
    )
  }

  private func forwardControlLayers(
    unified: MLXArray,
    refinedControl: MLXArray,
    capFeats: MLXArray,
    unifiedFreqs: MLXArray,
    tEmb: MLXArray,
    conditioningScale: Float
  ) -> ZImageControlBlockSamples {
    guard !controlLayers.isEmpty else { return [:] }

    let adalnInput = tEmb.asType(unified.dtype)
    let controlUnified = MLX.concatenated([refinedControl, capFeats], axis: 1)
    var controlState = ZImageControlHintState(control: controlUnified)
    for controlLayer in controlLayers {
      controlState = controlLayer(
        controlState,
        x: unified,
        attnMask: nil,
        freqsCis: unifiedFreqs,
        adalnInput: adalnInput
      )
    }
    return controlState.scaledHints(layerPlaces: controlLayersPlaces, conditioningScale: conditioningScale)
  }

  public func forward(
    latents: MLXArray,
    timestep: MLXArray,
    promptEmbeds: MLXArray,
    controlContext: MLXArray,
    conditioningScale: Float = 1.0
  ) -> ZImageControlBlockSamples {
    let hasFrameDim = latents.ndim == 5
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let frames = hasFrameDim ? latents.dim(2) : 1
    let height = latents.dim(hasFrameDim ? 3 : 2)
    let width = latents.dim(hasFrameDim ? 4 : 3)

    let patchSize = 2
    let fPatchSize = 1
    let key = "\(patchSize)-\(fPatchSize)"
    guard let xEmbed = allXEmbedder[key] else {
      return [:]
    }

    let capOriLen = promptEmbeds.dim(1)
    let cached = getOrBuildCache(
      batch: batch,
      height: height,
      width: width,
      frames: frames,
      capOriLen: capOriLen,
      patchSize: patchSize,
      fPatchSize: fPatchSize
    )

    var latentsWithFrame = latents
    if !hasFrameDim {
      latentsWithFrame = MLX.expandedDimensions(latents, axis: 2)
    }

    let tScaled = timestep * MLXArray(configuration.tScale)
    var tEmb = tEmbedder(tScaled)

    var controlEmbed = embedControlContext(controlContext: controlContext)

    var image =
      latentsWithFrame
      .reshaped(batch, channels, cached.fTokens, fPatchSize, cached.hTokens, patchSize, cached.wTokens, patchSize)
      .transposed(0, 2, 4, 6, 3, 5, 7, 1)
      .reshaped(batch, cached.imageTokens, patchSize * patchSize * fPatchSize * channels)

    if cached.imgPad > 0 {
      image = padSequenceByRepeatingLastToken(
        image,
        validLength: cached.imageTokens,
        padLength: cached.imgPad
      )
    }

    image = xEmbed(image)
    tEmb = tEmb.asType(image.dtype)
    controlEmbed = controlEmbed.asType(image.dtype)

    if let xPadToken, let imgPadMask = cached.imgPadMask {
      let padDim = xPadToken.dim(xPadToken.ndim - 1)
      let pad = MLX.broadcast(xPadToken.reshaped(1, 1, padDim), to: [batch, cached.imgSeqLen, padDim])
      image = MLX.where(MLX.expandedDimensions(imgPadMask, axis: 2), pad, image)
    }

    var noiseStream = image
    var refinedControl = controlEmbed
    if addControlNoiseRefiner {
      let refinerResult = forwardControlRefiner(
        noiseStream: noiseStream,
        controlEmbed: controlEmbed,
        imgFreqs: cached.imgFreqs,
        tEmb: tEmb,
        conditioningScale: conditioningScale
      )
      let refinerHints = refinerResult.refinerHints
      refinedControl = refinerResult.refinedControl
      for (layerIdx, block) in noiseRefiner.enumerated() {
        noiseStream = block(
          noiseStream,
          attnMask: nil,
          freqsCis: cached.imgFreqs,
          adalnInput: tEmb
        )
        if let controlHint = refinerHints[layerIdx] {
          noiseStream = noiseStream + controlHint
        }
      }
    } else {
      for block in noiseRefiner {
        noiseStream = block(
          noiseStream,
          attnMask: nil,
          freqsCis: cached.imgFreqs,
          adalnInput: tEmb
        )
      }
    }

    var capFeat = promptEmbeds
    if cached.capPad > 0 {
      capFeat = padSequenceByRepeatingLastToken(
        promptEmbeds,
        validLength: capOriLen,
        padLength: cached.capPad
      )
    }
    capFeat = capEmbedLinear(capEmbedNorm(capFeat))

    if let capPadToken, let capPadMask = cached.capPadMask {
      let padDim = capPadToken.dim(capPadToken.ndim - 1)
      let pad = MLX.broadcast(capPadToken.reshaped(1, 1, padDim), to: [batch, cached.capSeqLen, padDim])
      capFeat = MLX.where(MLX.expandedDimensions(capPadMask, axis: 2), pad, capFeat)
    }

    var capStream = capFeat
    for block in contextRefiner {
      capStream = block(
        capStream,
        attnMask: nil,
        freqsCis: cached.capFreqs,
        adalnInput: nil
      )
    }

    let unified = MLX.concatenated([noiseStream, capStream], axis: 1)
    return forwardControlLayers(
      unified: unified,
      refinedControl: refinedControl,
      capFeats: capStream,
      unifiedFreqs: cached.unifiedFreqsCis,
      tEmb: tEmb,
      conditioningScale: conditioningScale
    )
  }
}

public typealias ZImageControlTransformer2DModel = ZImageControlNetModel
