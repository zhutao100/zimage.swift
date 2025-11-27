import Foundation
import MLX
import MLXNN
import MLXFast

enum QwenTextEncoderError: Error {
  case visionTowerUnavailable
  case mismatchedVisionTokenCount
}

public struct QwenTextEncoderConfiguration {
  public var vocabSize: Int
  public var hiddenSize: Int
  public var numHiddenLayers: Int
  public var numAttentionHeads: Int
  public var numKeyValueHeads: Int
  public var intermediateSize: Int
  public var ropeTheta: Float
  public var maxPositionEmbeddings: Int
  public var rmsNormEps: Float
  public var promptDropIndex: Int
  public var headDim: Int

  public init(
    vocabSize: Int = 151_936,
    hiddenSize: Int = 2560,
    numHiddenLayers: Int = 36,
    numAttentionHeads: Int = 32,
    numKeyValueHeads: Int = 8,
    intermediateSize: Int = 9_728,
    ropeTheta: Float = 1_000_000.0,
    maxPositionEmbeddings: Int = 40_960,
    rmsNormEps: Float = 1e-6,
    promptDropIndex: Int = 0,
    headDim: Int = 128
  ) {
    self.vocabSize = vocabSize
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.intermediateSize = intermediateSize
    self.ropeTheta = ropeTheta
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.rmsNormEps = rmsNormEps
    self.promptDropIndex = promptDropIndex
    self.headDim = headDim
  }
}

public final class QwenTextEncoder: Module {

  public let configuration: QwenTextEncoderConfiguration
  @ModuleInfo(key: "encoder") var encoder: QwenEncoder
  private var visionTower: QwenVisionTower?

  public init(configuration: QwenTextEncoderConfiguration = .init()) {
    self.configuration = configuration
    self._encoder.wrappedValue = QwenEncoder(configuration: configuration)
  }

  func setVisionTower(_ tower: QwenVisionTower) {
    self.visionTower = tower
  }

  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    encode(inputIds: inputIds, attentionMask: attentionMask)
  }

  public func encode(
    inputIds: MLXArray,
    attentionMask: MLXArray?,
    keepFullSequence: Bool = false
  ) -> (MLXArray, MLXArray) {
    let result = encoder.forward(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputHiddenStates: false
    )
    let hiddenStates = result.lastHiddenState
    if keepFullSequence {
      let mask = attentionMask ?? MLX.ones([hiddenStates.dim(0), hiddenStates.dim(1)], dtype: .int32)
      return (hiddenStates, mask.asType(.int32))
    }
    let processed = QwenTextEncoder.processTextEmbeddings(
      hiddenStates: hiddenStates,
      attentionMask: attentionMask,
      dropIndex: configuration.promptDropIndex
    )
    return processed
  }

  public func forwardWithHiddenStates(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
    return encoder.forward(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputHiddenStates: true
    )
  }

  public func encodeForZImage(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) -> [MLXArray] {
    let result = encoder.forward(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputHiddenStates: true
    )

    guard let allHiddenStates = result.hiddenStates, allHiddenStates.count >= 2 else {
      return [result.lastHiddenState]
    }

    // Get second-to-last hidden state (before final norm)
    let secondToLast = allHiddenStates[allHiddenStates.count - 2]

    // Extract only the valid (non-padding) tokens for each batch item
    let batchSize = secondToLast.dim(0)
    var embeddingsList: [MLXArray] = []
    embeddingsList.reserveCapacity(batchSize)

    for i in 0..<batchSize {
      let batchEmbeds = secondToLast[i]

      if let mask = attentionMask {
        MLX.eval(mask)
        let batchMask = mask[i].asArray(Int32.self)
        let validCount = batchMask.filter { $0 != 0 }.count

        if validCount > 0 && validCount < batchMask.count {
          let validEmbeds = batchEmbeds[0..<validCount]
          embeddingsList.append(validEmbeds)
        } else {
          embeddingsList.append(batchEmbeds)
        }
      } else {
        embeddingsList.append(batchEmbeds)
      }
    }

    return embeddingsList
  }

  static func processTextEmbeddings(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    dropIndex: Int
  ) -> (MLXArray, MLXArray) {
    let batchSize = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)
    let hiddenDim = hiddenStates.dim(2)

    var mask: MLXArray
    if let attentionMask {
      mask = attentionMask
    } else {
      mask = MLX.ones([batchSize, seqLen], dtype: .int32)
    }
    if mask.dtype != .int32 {
      mask = mask.asType(.int32)
    }

    let trimmedStart = max(0, min(dropIndex, seqLen))

    // Pre-compute valid lengths to avoid .item() calls in loop
    let validLengthsArray = mask.sum(axis: 1)
    MLX.eval(validLengthsArray)
    let validLengths = validLengthsArray.asArray(Int.self)
    let trimmedLengths = validLengths.map { max(0, $0 - trimmedStart) }
    let maxTrimmedLength = trimmedLengths.max() ?? 0

    // Build padded embeddings and masks
    var paddedEmbeds: [MLXArray] = []
    paddedEmbeds.reserveCapacity(batchSize)
    var paddedMasks: [MLXArray] = []
    paddedMasks.reserveCapacity(batchSize)

    for batch in 0..<batchSize {
      let trimmedLength = trimmedLengths[batch]
      let sliceEnd = trimmedStart + trimmedLength

      var sampleEmbeds: MLXArray
      if trimmedLength > 0 {
        sampleEmbeds = hiddenStates[batch, trimmedStart..<sliceEnd, 0...]
      } else {
        sampleEmbeds = MLX.zeros([0, hiddenDim], dtype: hiddenStates.dtype)
      }

      if trimmedLength < maxTrimmedLength {
        let pad = MLX.zeros([maxTrimmedLength - trimmedLength, hiddenDim], dtype: hiddenStates.dtype)
        sampleEmbeds = MLX.concatenated([sampleEmbeds, pad], axis: 0)
      }
      paddedEmbeds.append(sampleEmbeds)

      let sampleMask: MLXArray
      if trimmedLength == 0 {
        sampleMask = MLX.zeros([maxTrimmedLength], dtype: .int32)
      } else if trimmedLength == maxTrimmedLength {
        sampleMask = MLX.ones([maxTrimmedLength], dtype: .int32)
      } else {
        let tailOnes = MLX.ones([trimmedLength], dtype: .int32)
        let leadZeros = MLX.zeros([maxTrimmedLength - trimmedLength], dtype: .int32)
        sampleMask = MLX.concatenated([tailOnes, leadZeros], axis: 0)
      }
      paddedMasks.append(sampleMask)
    }

    let promptEmbeds = MLX.stacked(paddedEmbeds, axis: 0)
    let encoderMask = MLX.stacked(paddedMasks, axis: 0)
    return (promptEmbeds, encoderMask)
  }

}

public final class QwenAttention: Module {
  let hiddenSize: Int
  let numAttentionHeads: Int
  let numKeyValueHeads: Int
  let headDim: Int
  let numKeyValueGroups: Int
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear
  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

  let rope: RoPE

  init(configuration: QwenTextEncoderConfiguration) {
    self.hiddenSize = configuration.hiddenSize
    self.numAttentionHeads = configuration.numAttentionHeads
    self.numKeyValueHeads = configuration.numKeyValueHeads
    self.headDim = configuration.headDim
    self.numKeyValueGroups = configuration.numAttentionHeads / configuration.numKeyValueHeads
    self.scale = pow(Float(configuration.headDim), -0.5)

    self._qProj.wrappedValue = Linear(hiddenSize, numAttentionHeads * headDim, bias: false)
    self._kProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
    self._vProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
    self._oProj.wrappedValue = Linear(numAttentionHeads * headDim, hiddenSize, bias: false)

    self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
    self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)

    self.rope = RoPE(
      dimensions: headDim,
      traditional: false,
      base: configuration.ropeTheta,
      scale: 1.0
    )
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
  ) -> MLXArray {
    let B = x.dim(0)
    let L = x.dim(1)

    var queries = qProj(x)
    var keys = kProj(x)
    var values = vProj(x)

    queries = qNorm(queries.reshaped(B, L, numAttentionHeads, headDim)).transposed(0, 2, 1, 3)
    keys = kNorm(keys.reshaped(B, L, numKeyValueHeads, headDim)).transposed(0, 2, 1, 3)
    values = values.reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

    queries = rope(queries, offset: 0)
    keys = rope(keys, offset: 0)

    if numKeyValueHeads != numAttentionHeads {
      keys = expandKeyValue(keys, repeats: numKeyValueGroups)
      values = expandKeyValue(values, repeats: numKeyValueGroups)
    }

    var output = MLXFast.scaledDotProductAttention(
      queries: queries,
      keys: keys,
      values: values,
      scale: scale,
      mask: mask
    )

    output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)

    return oProj(output)
  }

  private func expandKeyValue(_ x: MLXArray, repeats: Int) -> MLXArray {
    guard repeats > 1 else { return x }
    var expanded = MLX.expandedDimensions(x, axis: 2)
    expanded = MLX.repeated(expanded, count: repeats, axis: 2)
    let shape = x.shape
    return expanded.reshaped(shape[0], shape[1] * repeats, shape[2], shape[3])
  }
}

public final class QwenMLP: Module {
  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear

  init(dimensions: Int, hiddenDimensions: Int) {
    self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    downProj(silu(gateProj(x)) * upProj(x))
  }
}

public final class QwenEncoderLayer: Module {
  @ModuleInfo(key: "self_attn") var selfAttention: QwenAttention
  let mlp: QwenMLP

  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(configuration: QwenTextEncoderConfiguration) {
    self._selfAttention.wrappedValue = QwenAttention(configuration: configuration)
    self.mlp = QwenMLP(dimensions: configuration.hiddenSize, hiddenDimensions: configuration.intermediateSize)
    self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
  ) -> MLXArray {
    let normed = inputLayerNorm(x)
    let r = selfAttention(normed, mask: mask)
    let h = x + r

    let postNormed = postAttentionLayerNorm(h)
    let mlpOut = mlp(postNormed)

    return h + mlpOut
  }
}

public final class QwenEncoder: Module {

  public let configuration: QwenTextEncoderConfiguration
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "layers") var layers: [QwenEncoderLayer]
  @ModuleInfo(key: "norm") var norm: RMSNorm

  public init(configuration: QwenTextEncoderConfiguration) {
    self.configuration = configuration
    self._embedTokens.wrappedValue = Embedding(
      embeddingCount: configuration.vocabSize, dimensions: configuration.hiddenSize)
    self._layers.wrappedValue = (0..<configuration.numHiddenLayers).map { _ in
      QwenEncoderLayer(configuration: configuration)
    }
    self._norm.wrappedValue = RMSNorm(
      dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
  }

  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) -> MLXArray {
    forward(inputIds: inputIds, attentionMask: attentionMask).lastHiddenState
  }

  public func forward(
    inputIds: MLXArray,
    attentionMask: MLXArray?,
    outputHiddenStates: Bool = false
  ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }

    var h = embedTokens(tokenIds)

    let mask = createAttentionMask(h: h, attentionMask: attentionMask)

    var allHiddenStates: [MLXArray]? = outputHiddenStates ? [h] : nil

    for layer in layers {
      h = layer(h, mask: mask)
      if outputHiddenStates {
        allHiddenStates?.append(h)
      }
    }

    h = norm(h)

    if outputHiddenStates, var states = allHiddenStates, !states.isEmpty {
      states[states.count - 1] = h
      allHiddenStates = states
    }

    return (h, allHiddenStates)
  }

  private func createAttentionMask(h: MLXArray, attentionMask: MLXArray?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let L = h.dim(1)

    let causalMask = MLXFast.ScaledDotProductAttentionMaskMode.causal

    if let attentionMask = attentionMask {
      let paddingMask = attentionMask.asType(h.dtype)
      let zeros = MLX.zeros(paddingMask.shape, dtype: h.dtype)
      let negInf = MLXArray(-Float.infinity).asType(h.dtype)
      let keepMask = paddingMask .== MLXArray(1).asType(h.dtype)
      var additivePaddingMask = MLX.where(keepMask, zeros, zeros + negInf)

      additivePaddingMask = additivePaddingMask.reshaped(additivePaddingMask.dim(0), 1, 1, L)

      let idx = MLXArray(0..<L)
      let rows = idx.reshaped(L, 1)
      let cols = idx.reshaped(1, L)
      let causalBool = cols .> rows
      var causalAdditive = MLX.zeros([L, L], dtype: h.dtype)
      causalAdditive = MLX.where(causalBool, causalAdditive + negInf, causalAdditive)
      causalAdditive = causalAdditive.reshaped(1, 1, L, L)

      let combinedMask = causalAdditive + additivePaddingMask

      return .array(combinedMask)
    }

    return causalMask
  }
}

extension QwenEncoder {
  public func embed(inputIds: MLXArray) -> MLXArray {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }
    return embedTokens(tokenIds)
  }
}

extension QwenTextEncoder {
  public func encodeJoint(
    inputIds: MLXArray,
    attentionMask: MLXArray?,
    imageTokenId: Int,
    visionStartTokenId: Int,
    placeholderGridTHW: [(Int, Int, Int)],
    spatialMergeSize: Int,
    replacements: [MLXArray],
    dropIndex dropIndexOverride: Int? = nil
  ) -> (MLXArray, MLXArray) {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }
    var hiddenStates = encoder.embed(inputIds: tokenIds)
    let dropIndexValue = dropIndexOverride ?? configuration.promptDropIndex

    if !replacements.isEmpty {
      hiddenStates = replaceVisionTokens(
        hiddenStates: hiddenStates,
        inputIds: tokenIds,
        imageTokenId: imageTokenId,
        replacements: replacements
      )
    }

    let attentionMaskUpdated: MLXArray
    if let attentionMask {
      attentionMaskUpdated = attentionMask.asType(.int32)
    } else {
      attentionMaskUpdated = MLX.ones([hiddenStates.dim(0), hiddenStates.dim(1)], dtype: .int32)
    }

    // For joint encoding, use the standard forward path
    let result = encoder.forward(
      inputIds: inputIds,
      attentionMask: attentionMaskUpdated,
      outputHiddenStates: false
    )

    let processed = QwenTextEncoder.processTextEmbeddings(
      hiddenStates: result.lastHiddenState,
      attentionMask: attentionMaskUpdated,
      dropIndex: dropIndexValue
    )
    return processed
  }

  private func replaceVisionTokens(
    hiddenStates: MLXArray,
    inputIds: MLXArray,
    imageTokenId: Int,
    replacements: [MLXArray]
  ) -> MLXArray {
    let batch = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)
    let hiddenDim = hiddenStates.dim(2)

    guard let first = replacements.first else {
      return hiddenStates
    }
    var replacementTensor = replacements.count == 1 ? first : MLX.concatenated(replacements, axis: 0)
    if replacementTensor.dtype != .float32 {
      replacementTensor = replacementTensor.asType(.float32)
    }
    MLX.eval(replacementTensor)
    let replacementValues = replacementTensor.asArray(Float32.self)
    let replacementCount = replacementTensor.dim(0)

    let tokenArray = inputIds.asType(.int32)
    MLX.eval(tokenArray)
    let tokenValues = tokenArray.asArray(Int32.self)

    var updatedRows: [MLXArray] = []
    updatedRows.reserveCapacity(batch)
    for row in 0..<batch {
      let rowEmb = hiddenStates[row, 0..., 0...].asType(.float32)
      MLX.eval(rowEmb)
      var rowValues = rowEmb.asArray(Float32.self)
      var cursor = 0
      for position in 0..<seqLen where tokenValues[row * seqLen + position] == Int32(imageTokenId) {
        precondition(cursor < replacementCount, "[QwenTextEncoder] placeholder mismatch in row \(row)")
        let dest = position * hiddenDim
        let src = cursor * hiddenDim
        rowValues.withUnsafeMutableBufferPointer { destPtr in
          replacementValues.withUnsafeBufferPointer { srcPtr in
            memcpy(
              destPtr.baseAddress! + dest,
              srcPtr.baseAddress! + src,
              hiddenDim * MemoryLayout<Float32>.size
            )
          }
        }
        cursor += 1
      }
      precondition(cursor == replacementCount, "[QwenTextEncoder] row \(row) consumed \(cursor) tokens, expected \(replacementCount)")
      let rowArray = MLXArray(rowValues, [seqLen, hiddenDim]).asType(hiddenStates.dtype)
      updatedRows.append(rowArray)
    }
    return MLX.stacked(updatedRows, axis: 0)
  }
}
