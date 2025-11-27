import Foundation
import MLX

enum ZImageAttentionUtils {

  static func applyComplexRoPEBLHD(
    query: MLXArray,
    key: MLXArray,
    freqsCis: MLXArray
  ) -> (MLXArray, MLXArray) {
    func applyRotary(_ x: MLXArray, _ freqs: MLXArray) -> MLXArray {
      let shape = x.shape
      let newShape = Array(shape.dropLast()) + [shape.last! / 2, 2]
      let xReshaped = x.reshaped(newShape)

      let xReal = xReshaped[0..., 0..., 0..., 0..., 0]
      let xImag = xReshaped[0..., 0..., 0..., 0..., 1]

      var freqsCos: MLXArray
      var freqsSin: MLXArray

      if freqs.ndim == 3 && freqs.dim(-1) == 2 {
        freqsCos = freqs[0..., 0..., 0]
        freqsSin = freqs[0..., 0..., 1]
      } else {
        let halfDim = freqs.dim(-1) / 2
        freqsCos = freqs[0..., 0..<halfDim]
        freqsSin = freqs[0..., halfDim...]
      }

      freqsCos = freqsCos[.newAxis, 0..., .newAxis, 0...]
      freqsSin = freqsSin[.newAxis, 0..., .newAxis, 0...]

      let outReal = xReal * freqsCos - xImag * freqsSin
      let outImag = xReal * freqsSin + xImag * freqsCos

      let stacked = MLX.stacked([outReal, outImag], axis: -1)
      return stacked.reshaped(shape)
    }

    return (applyRotary(query, freqsCis), applyRotary(key, freqsCis))
  }

  static func applyComplexRoPE(
    query: MLXArray,
    key: MLXArray,
    freqsCis: MLXArray
  ) -> (MLXArray, MLXArray) {
    func applyRotary(_ x: MLXArray, _ freqs: MLXArray) -> MLXArray {
      let shape = x.shape
      let newShape = Array(shape.dropLast()) + [shape.last! / 2, 2]
      let xReshaped = x.reshaped(newShape)

      let xReal = xReshaped[0..., 0..., 0..., 0..., 0]
      let xImag = xReshaped[0..., 0..., 0..., 0..., 1]

      var freqsCos: MLXArray
      var freqsSin: MLXArray

      if freqs.ndim == 3 && freqs.dim(-1) == 2 {
        freqsCos = freqs[0..., 0..., 0]
        freqsSin = freqs[0..., 0..., 1]
      } else {
        let halfDim = freqs.dim(-1) / 2
        freqsCos = freqs[0..., 0..<halfDim]
        freqsSin = freqs[0..., halfDim...]
      }

      freqsCos = freqsCos[.newAxis, .newAxis, 0..., 0...]
      freqsSin = freqsSin[.newAxis, .newAxis, 0..., 0...]

      let outReal = xReal * freqsCos - xImag * freqsSin
      let outImag = xReal * freqsSin + xImag * freqsCos

      let stacked = MLX.stacked([outReal, outImag], axis: -1)
      return stacked.reshaped(shape)
    }

    return (applyRotary(query, freqsCis), applyRotary(key, freqsCis))
  }

  static func applyRoPEBSHD(
    query: MLXArray,
    key: MLXArray,
    cos: MLXArray,
    sin: MLXArray
  ) -> (MLXArray, MLXArray) {
    let computeType = cos.dtype
    let queryTensor = query.asType(computeType)
    let keyTensor = key.asType(computeType)
    let cosPrepared = cos.asType(computeType)
    let sinPrepared = sin.asType(computeType)

    let cosExpanded = cosPrepared.reshaped(1, cosPrepared.dim(0), 1, cosPrepared.dim(1))
    let sinExpanded = sinPrepared.reshaped(1, cosPrepared.dim(0), 1, cosPrepared.dim(1))

    func mix(_ tensor: MLXArray) -> MLXArray {
      let lastDim = tensor.dim(tensor.ndim - 1)
      let reshapeShape = Array(tensor.shape.dropLast() + [lastDim / 2, 2])
      let reshaped = tensor.reshaped(reshapeShape)
      let real = reshaped[.ellipsis, 0]
      let imag = reshaped[.ellipsis, 1]
      let out0 = real * cosExpanded + (-imag) * sinExpanded
      let out1 = imag * cosExpanded + real * sinExpanded
      return MLX.stacked([out0, out1], axis: -1).reshaped(tensor.shape)
    }

    return (
      mix(queryTensor).asType(query.dtype),
      mix(keyTensor).asType(key.dtype)
    )
  }
}
