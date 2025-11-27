import Foundation
import MLX
import MLXNN

public final class ZImageTransformerBlock: Module {
  let dim: Int
  let nHeads: Int
  let nKvHeads: Int
  let modulation: Bool

  @ModuleInfo(key: "attention") var attention: ZImageSelfAttention
  @ModuleInfo(key: "adaLN_modulation") var adaLN: [Linear]?
  @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
  @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
  @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
  @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
  @ModuleInfo(key: "feed_forward") var feedForward: ZImageFeedForward

  init(
    layerId: Int,
    dim: Int,
    nHeads: Int,
    nKvHeads: Int,
    normEps: Float,
    qkNorm: Bool,
    modulation: Bool
  ) {
    self.dim = dim
    self.nHeads = nHeads
    self.nKvHeads = nKvHeads
    self.modulation = modulation

    self._attention.wrappedValue = ZImageSelfAttention(dim: dim, heads: nHeads, normEps: normEps, qkNorm: qkNorm)
    if modulation {
      self._adaLN.wrappedValue = [Linear(min(dim, 256), 4 * dim, bias: true)]
    }
    self._attentionNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._ffnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._attentionNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._ffnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)

    let hiddenDim = Int(Float(dim) / 3.0 * 8.0)
    self._feedForward.wrappedValue = ZImageFeedForward(dim: dim, hiddenDim: hiddenDim)

    super.init()
  }

  func callAsFunction(
    _ x: MLXArray,
    attnMask: MLXArray? = nil,
    freqsCis: MLXArray? = nil,
    adalnInput: MLXArray? = nil
  ) -> MLXArray {
    var out = x

    var scaleMsa: MLXArray = MLXArray(1.0)
    var gateMsa: MLXArray = MLXArray(1.0)
    var scaleMlp: MLXArray = MLXArray(1.0)
    var gateMlp: MLXArray = MLXArray(1.0)

    if modulation {
      guard let c = adalnInput else {
        fatalError("adalnInput required when modulation is enabled")
      }
      guard let adaLNModule = adaLN else {
        fatalError("adaLN module should exist when modulation is enabled")
      }
      let mod = adaLNModule[0](c)
      let chunks = mod.split(parts: 4, axis: -1)
      scaleMsa = MLXArray(1.0) + chunks[0]
      gateMsa = MLX.tanh(chunks[1])
      scaleMlp = MLXArray(1.0) + chunks[2]
      gateMlp = MLX.tanh(chunks[3])
    }

    func expand(_ t: MLXArray) -> MLXArray {
      if t.ndim == 0 {
        return t
      }
      return MLX.expandedDimensions(t, axis: 1)
    }

    let attnScale = expand(scaleMsa)
    let attnGate = expand(gateMsa)
    let mlpScale = expand(scaleMlp)
    let mlpGate = expand(gateMlp)

    let attnInput = attentionNorm1(out) * attnScale
    let attnOut = attention(attnInput, attnMask: attnMask, freqsCis: freqsCis)
    out = out + attnGate * attentionNorm2(attnOut)

    let ffnInput = ffnNorm1(out) * mlpScale
    let ffnOut = feedForward(ffnInput)
    out = out + mlpGate * ffnNorm2(ffnOut)

    return out
  }
}
