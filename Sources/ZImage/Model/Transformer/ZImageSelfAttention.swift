import Foundation
import MLX
import MLXFast
import MLXNN

final class ZImageSelfAttention: Module {
  let dim: Int
  let heads: Int
  let headDim: Int
  let useQKNorm: Bool

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: [Linear]
  @ModuleInfo(key: "norm_q") var normQ: RMSNorm?
  @ModuleInfo(key: "norm_k") var normK: RMSNorm?

  init(dim: Int, heads: Int, normEps: Float, qkNorm: Bool) {
    self.dim = dim
    self.heads = heads
    self.headDim = dim / heads
    self.useQKNorm = qkNorm

    self._toQ.wrappedValue = Linear(dim, dim, bias: false)
    self._toK.wrappedValue = Linear(dim, dim, bias: false)
    self._toV.wrappedValue = Linear(dim, dim, bias: false)
    self._toOut.wrappedValue = [Linear(dim, dim, bias: false)]
    if qkNorm {
      self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: normEps)
      self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: normEps)
    }
    super.init()
  }

  func callAsFunction(
    _ x: MLXArray,
    attnMask: MLXArray? = nil,
    freqsCis: MLXArray? = nil
  ) -> MLXArray {
    let batch = x.dim(0)
    let seqLen = x.dim(1)

    var q = toQ(x)
    var k = toK(x)
    var v = toV(x)

    q = q.reshaped(batch, seqLen, heads, headDim)
    k = k.reshaped(batch, seqLen, heads, headDim)
    v = v.reshaped(batch, seqLen, heads, headDim)

    if useQKNorm {
      if let normQ {
        q = normQ(q)
      }
      if let normK {
        k = normK(k)
      }
    }

    if let freqsCis {
      let rotated = ZImageAttentionUtils.applyComplexRoPEBLHD(query: q, key: k, freqsCis: freqsCis)
      q = rotated.0
      k = rotated.1
    }

    q = q.transposed(0, 2, 1, 3)
    k = k.transposed(0, 2, 1, 3)
    v = v.transposed(0, 2, 1, 3)

    let scale = Float(1.0) / sqrt(Float(headDim))

    var attn = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: attnMask
    )

    attn = attn.transposed(0, 2, 1, 3).reshaped(batch, seqLen, dim)
    return toOut[0](attn)
  }
}
