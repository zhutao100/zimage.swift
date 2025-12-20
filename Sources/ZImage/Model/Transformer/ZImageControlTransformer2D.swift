import Foundation
import MLX
import MLXNN

public struct ZImageControlTransformerConfig {
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
}
public final class ZImageControlTransformer2DModel: Module {
  public let configuration: ZImageControlTransformerConfig

  let controlLayersPlaces: [Int]
  let controlLayersMapping: [Int: Int]
  let controlRefinerLayersPlaces: [Int]
  let controlRefinerLayersMapping: [Int: Int]
  let controlInDim: Int
  let addControlNoiseRefiner: Bool

  @ModuleInfo(key: "t_embedder") var tEmbedder: ZImageTimestepEmbedder
  @ModuleInfo(key: "all_x_embedder") var allXEmbedder: [String: Linear]
  @ModuleInfo(key: "all_final_layer") var allFinalLayer: [String: ZImageFinalLayer]
  @ModuleInfo(key: "noise_refiner") public internal(set) var noiseRefiner: [BaseZImageTransformerBlock]
  @ModuleInfo(key: "context_refiner") public internal(set) var contextRefiner: [ZImageTransformerBlock]

  @ModuleInfo(key: "layers") public internal(set) var layers: [BaseZImageTransformerBlock]

  @ModuleInfo(key: "control_all_x_embedder") var controlAllXEmbedder: [String: Linear]
  @ModuleInfo(key: "control_noise_refiner") public internal(set) var controlNoiseRefiner: [ZImageControlTransformerBlock]
  @ModuleInfo(key: "control_layers") public internal(set) var controlLayers: [ZImageControlTransformerBlock]

  var capEmbedNorm: RMSNorm
  var capEmbedLinear: Linear

  let ropeEmbedder: ZImageRopeEmbedder
  private var xPadToken: MLXArray?
  private var capPadToken: MLXArray?

  private var cache: TransformerCache?
  private var cacheKey: TransformerCacheKey?

  public init(configuration: ZImageControlTransformerConfig) {
    self.configuration = configuration
    self.controlLayersPlaces = configuration.controlLayersPlaces
    self.controlRefinerLayersPlaces = configuration.controlRefinerLayersPlaces
    self.controlInDim = configuration.controlInDim
    self.addControlNoiseRefiner = configuration.addControlNoiseRefiner
    var mapping: [Int: Int] = [:]
    for (idx, place) in configuration.controlLayersPlaces.enumerated() {
      mapping[place] = idx
    }
    self.controlLayersMapping = mapping
    var refinerMapping: [Int: Int] = [:]
    for (idx, place) in configuration.controlRefinerLayersPlaces.enumerated() {
      refinerMapping[place] = idx
    }
    self.controlRefinerLayersMapping = refinerMapping

    let outSize = min(configuration.dim, 256)
    self._tEmbedder.wrappedValue = ZImageTimestepEmbedder(outSize: outSize, midSize: 1024)

    let patchSize = 2
    let fPatchSize = 1
    let key = "\(patchSize)-\(fPatchSize)"

    var xEmbedder: [String: Linear] = [:]
    var finalLayers: [String: ZImageFinalLayer] = [:]
    let inFeatures = fPatchSize * patchSize * patchSize * configuration.inChannels
    xEmbedder[key] = Linear(inFeatures, configuration.dim, bias: true)
    finalLayers[key] = ZImageFinalLayer(
      hiddenSize: configuration.dim,
      outChannels: patchSize * patchSize * fPatchSize * configuration.inChannels
    )
    self._allXEmbedder.wrappedValue = xEmbedder
    self._allFinalLayer.wrappedValue = finalLayers

    var controlXEmbedder: [String: Linear] = [:]
    let controlInFeatures = fPatchSize * patchSize * patchSize * configuration.controlInDim
    controlXEmbedder[key] = Linear(controlInFeatures, configuration.dim, bias: true)
    self._controlAllXEmbedder.wrappedValue = controlXEmbedder

    self.capEmbedNorm = RMSNorm(dimensions: configuration.capFeatDim, eps: configuration.normEps)
    self.capEmbedLinear = Linear(configuration.capFeatDim, configuration.dim, bias: true)
    var noiseBlocks: [BaseZImageTransformerBlock] = []
    for layerId in 0..<configuration.nRefinerLayers {
      let blockId = configuration.addControlNoiseRefiner ? refinerMapping[layerId] : nil
      noiseBlocks.append(
        BaseZImageTransformerBlock(
          layerId: 1000 + layerId,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm,
          modulation: true,
          blockId: blockId
        )
      )
    }
    self._noiseRefiner.wrappedValue = noiseBlocks
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

    var contextBlocks: [ZImageTransformerBlock] = []
    for layerId in 0..<configuration.nRefinerLayers {
      contextBlocks.append(
        ZImageTransformerBlock(
          layerId: layerId,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm,
          modulation: false
        )
      )
    }
    self._contextRefiner.wrappedValue = contextBlocks

    var mainLayers: [BaseZImageTransformerBlock] = []
    for layerId in 0..<configuration.nLayers {
      let blockId = mapping[layerId]
      mainLayers.append(
        BaseZImageTransformerBlock(
          layerId: layerId,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm,
          modulation: true,
          blockId: blockId
        )
      )
    }
    self._layers.wrappedValue = mainLayers

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

    self.ropeEmbedder = ZImageRopeEmbedder(
      theta: configuration.ropeTheta,
      axesDims: configuration.axesDims,
      axesLens: configuration.axesLens
    )

    super.init()
  }

  public func loadCapEmbedderWeights(from weights: [String: MLXArray]) {
    if let normWeight = weights["cap_embedder.0.weight"] {
      capEmbedNorm.weight._updateInternal(normWeight)
    }
    if let linearWeight = weights["cap_embedder.1.weight"] {
      capEmbedLinear.weight._updateInternal(linearWeight)
    }
    if let linearBias = weights["cap_embedder.1.bias"] {
      capEmbedLinear.bias?._updateInternal(linearBias)
    }
  }

  public func loadXEmbedderWeights(from weights: [String: MLXArray], groupSize: Int = 32, bits: Int = 8) {
    let key = "2-1"
    let prefix = "all_x_embedder.\(key)"
    guard let linear = allXEmbedder[key] else { return }

    if let w = weights["\(prefix).weight"] {
      linear.weight._updateInternal(w)
    }
    if let b = weights["\(prefix).bias"] {
      linear.bias?._updateInternal(b)
    }
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

  public func loadFinalLayerWeights(from weights: [String: MLXArray], groupSize: Int = 32, bits: Int = 8) {
    let key = "2-1"
    let prefix = "all_final_layer.\(key)"
    guard let finalLayer = allFinalLayer[key] else { return }

    if let lin = finalLayer.linear as? Linear {
      if let w = weights["\(prefix).linear.weight"] { lin.weight._updateInternal(w) }
      if let b = weights["\(prefix).linear.bias"] { lin.bias?._updateInternal(b) }
    }

    if let lin = finalLayer.adaLN.linear as? Linear {
      if let w = weights["\(prefix).adaLN_modulation.1.weight"] { lin.weight._updateInternal(w) }
      if let b = weights["\(prefix).adaLN_modulation.1.bias"] { lin.bias?._updateInternal(b) }
    }
  }

  public func setPadTokens(xPad: MLXArray?, capPad: MLXArray?) {
    if let xPad {
      let padDim = xPad.dim(xPad.ndim - 1)
      self.xPadToken = xPad.reshaped(padDim)
    }
    if let capPad {
      let padDim = capPad.dim(capPad.ndim - 1)
      self.capPadToken = capPad.reshaped(padDim)
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

    self.cache = newCache
    self.cacheKey = key

    return newCache
  }
  private func embedControlContext(
    controlContext: MLXArray,
    tEmb: MLXArray,
    patchSize: Int = 2,
    fPatchSize: Int = 1
  ) -> (MLXArray, Int) {
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

    var controlImage = controlContext
      .reshaped(batch, channels, fTokens, fPatchSize, hTokens, patchSize, wTokens, patchSize)
      .transposed(0, 2, 4, 6, 3, 5, 7, 1)
      .reshaped(batch, controlTokens, patchSize * patchSize * fPatchSize * channels)
    let seqMultiOf = 32
    let controlPad = (seqMultiOf - (controlTokens % seqMultiOf)) % seqMultiOf
    if controlPad > 0 {
      let last = controlImage[0..., controlTokens - 1, 0...]
      let pad = MLX.broadcast(last, to: [batch, controlPad, controlImage.dim(2)])
      controlImage = MLX.concatenated([controlImage, pad], axis: 1)
    }

    var controlEmbed = controlXEmbed(controlImage)
    let adalnInput = tEmb.asType(controlEmbed.dtype)

    if let xPadToken, controlPad > 0 {
      let padDim = xPadToken.dim(xPadToken.ndim - 1)
      let controlSeqLen = controlTokens + controlPad
      let padMask1d = MLX.concatenated([
        MLX.zeros([controlTokens], dtype: .bool),
        MLX.ones([controlPad], dtype: .bool)
      ], axis: 0)
      let padMask = MLX.broadcast(padMask1d.reshaped(1, controlSeqLen), to: [batch, controlSeqLen])
      let pad = MLX.broadcast(xPadToken.reshaped(1, 1, padDim), to: [batch, controlSeqLen, padDim])
      controlEmbed = MLX.where(MLX.expandedDimensions(padMask, axis: 2), pad, controlEmbed)
    }

    return (controlEmbed, controlTokens)
  }

  private func forwardControlRefiner(
    noiseStream: MLXArray,
    controlEmbed: MLXArray,
    imgFreqs: MLXArray,
    tEmb: MLXArray
  ) -> (refinerHints: [MLXArray], refinedControl: MLXArray) {
    let adalnInput = tEmb.asType(controlEmbed.dtype)

    var c = controlEmbed
    for refinerBlock in controlNoiseRefiner {
      c = refinerBlock(
        c,
        x: noiseStream,
        attnMask: nil,
        freqsCis: imgFreqs,
        adalnInput: adalnInput
      )
    }

    let numHints = c.dim(0) - 1
    var hints: [MLXArray] = []
    for i in 0..<numHints {
      hints.append(c[i])
    }
    let refinedControl = c[numHints]

    return (hints, refinedControl)
  }

  private func forwardControlLayers(
    unified: MLXArray,
    refinedControl: MLXArray,
    capFeats: MLXArray,
    unifiedFreqs: MLXArray,
    tEmb: MLXArray
  ) -> [MLXArray] {
    let adalnInput = tEmb.asType(unified.dtype)
    let controlUnified = MLX.concatenated([refinedControl, capFeats], axis: 1)
    var c = controlUnified
    for controlLayer in controlLayers {
      c = controlLayer(
        c,
        x: unified,
        attnMask: nil,
        freqsCis: unifiedFreqs,
        adalnInput: adalnInput
      )
    }
    let numHints = c.dim(0) - 1
    var hints: [MLXArray] = []
    for i in 0..<numHints {
      hints.append(c[i])
    }

    return hints
  }

  public func forward(
    latents: MLXArray,
    timestep: MLXArray,
    promptEmbeds: MLXArray,
    controlContext: MLXArray? = nil,
    controlContextScale: Float = 1.0
  ) -> MLXArray {
    let hasFrameDim = latents.ndim == 5
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let frames = hasFrameDim ? latents.dim(2) : 1
    let height = latents.dim(hasFrameDim ? 3 : 2)
    let width = latents.dim(hasFrameDim ? 4 : 3)

    let patchSize = 2
    let fPatchSize = 1
    let key = "\(patchSize)-\(fPatchSize)"
    guard let xEmbed = allXEmbedder[key], let finalLayer = allFinalLayer[key] else {
      return MLX.zeros(latents.shape, dtype: latents.dtype)
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

    var capFeat = promptEmbeds
    if cached.capPad > 0 {
      let last = promptEmbeds[0..., capOriLen - 1, 0...]
      let pad = MLX.broadcast(last, to: [batch, cached.capPad, promptEmbeds.dim(2)])
      capFeat = MLX.concatenated([promptEmbeds, pad], axis: 1)
    }
    capFeat = capEmbedLinear(capEmbedNorm(capFeat))

    if let capPadToken, let capPadMask = cached.capPadMask {
      let padDim = capPadToken.dim(capPadToken.ndim - 1)
      let pad = MLX.broadcast(capPadToken.reshaped(1, 1, padDim), to: [batch, cached.capSeqLen, padDim])
      capFeat = MLX.where(MLX.expandedDimensions(capPadMask, axis: 2), pad, capFeat)
    }

    var image = latentsWithFrame
      .reshaped(batch, channels, cached.fTokens, fPatchSize, cached.hTokens, patchSize, cached.wTokens, patchSize)
      .transposed(0, 2, 4, 6, 3, 5, 7, 1)
      .reshaped(batch, cached.imageTokens, patchSize * patchSize * fPatchSize * channels)

    if cached.imgPad > 0 {
      let last = image[0..., cached.imageTokens - 1, 0...]
      let pad = MLX.broadcast(last, to: [batch, cached.imgPad, image.dim(2)])
      image = MLX.concatenated([image, pad], axis: 1)
    }

    image = xEmbed(image)
    tEmb = tEmb.asType(image.dtype)

    if let xPadToken, let imgPadMask = cached.imgPadMask {
      let padDim = xPadToken.dim(xPadToken.ndim - 1)
      let pad = MLX.broadcast(xPadToken.reshaped(1, 1, padDim), to: [batch, cached.imgSeqLen, padDim])
      image = MLX.where(MLX.expandedDimensions(imgPadMask, axis: 2), pad, image)
    }
    var refinerHints: [MLXArray]? = nil
    var refinedControlContext: MLXArray? = nil
    var noiseStream = image
    if let controlCtx = controlContext, addControlNoiseRefiner {

      let (controlEmbed, _) = embedControlContext(controlContext: controlCtx, tEmb: tEmb)
      let refinerResult = forwardControlRefiner(
        noiseStream: noiseStream,
        controlEmbed: controlEmbed,
        imgFreqs: cached.imgFreqs,
        tEmb: tEmb
      )
      refinerHints = refinerResult.refinerHints
      refinedControlContext = refinerResult.refinedControl
    }
    for block in noiseRefiner {
      noiseStream = block(
        noiseStream,
        attnMask: nil,
        freqsCis: cached.imgFreqs,
        adalnInput: tEmb,
        hints: refinerHints,
        contextScale: controlContextScale
      )
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
    var unified = MLX.concatenated([noiseStream, capStream], axis: 1)
    var layerHints: [MLXArray]? = nil
    if let refinedControl = refinedControlContext {

      layerHints = forwardControlLayers(
        unified: unified,
        refinedControl: refinedControl,
        capFeats: capStream,
        unifiedFreqs: cached.unifiedFreqsCis,
        tEmb: tEmb
      )
    }
    for block in layers {
      unified = block(
        unified,
        attnMask: nil,
        freqsCis: cached.unifiedFreqsCis,
        adalnInput: tEmb,
        hints: layerHints,
        contextScale: controlContextScale
      )
    }

    let imageOut = unified[0..., 0..<cached.imageTokens, 0...]
    let projected = finalLayer(imageOut, conditioning: tEmb)
    let outChannels = configuration.inChannels

    var reshaped = projected
      .reshaped(batch, cached.fTokens, cached.hTokens, cached.wTokens, fPatchSize, patchSize, patchSize, outChannels)
      .transposed(0, 7, 1, 4, 2, 5, 3, 6)
      .reshaped(batch, outChannels, cached.fTokens * fPatchSize, cached.hTokens * patchSize, cached.wTokens * patchSize)

    reshaped = reshaped[0..., 0..., 0, 0..., 0...]
    return reshaped
  }
}
