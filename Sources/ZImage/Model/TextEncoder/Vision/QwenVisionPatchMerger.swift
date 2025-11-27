import Foundation
import MLX
import MLXNN

final class QwenVisionPatchMerger: Module {
  let spatialMergeSize: Int
  let contextDim: Int
  let hiddenDim: Int
  let outputDim: Int

  private let mergeUnit: Int

  @ModuleInfo(key: "ln_q") private var norm: LayerNorm
  @ModuleInfo(key: "mlp_0") private var mlpInput: Linear
  @ModuleInfo(key: "mlp_2") private var mlpOutput: Linear

  init(contextDim: Int, outputDim: Int, spatialMergeSize: Int) {
    self.contextDim = contextDim
    self.outputDim = outputDim
    self.spatialMergeSize = spatialMergeSize
    self.mergeUnit = spatialMergeSize * spatialMergeSize
    self.hiddenDim = contextDim * mergeUnit

    self._norm.wrappedValue = LayerNorm(dimensions: contextDim, eps: 1e-6, affine: true)
    self._mlpInput.wrappedValue = Linear(hiddenDim, hiddenDim)
    self._mlpOutput.wrappedValue = Linear(hiddenDim, outputDim)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    precondition(hiddenStates.ndim == 2, "Expected 2D input [N, D]")
    let targetType = hiddenStates.dtype
    let features = hiddenStates.dim(1)

    if features == contextDim {
      let tokens = hiddenStates.dim(0)
      precondition(tokens % mergeUnit == 0, "Token count must be divisible by merge unit \(mergeUnit)")
      var normed = norm(hiddenStates)
      normed = normed.reshaped(tokens / mergeUnit, hiddenDim)
      var merged = mlpInput(normed)
      merged = MLXNN.gelu(merged)
      merged = mlpOutput(merged)
      return merged.asType(targetType)
    } else if features == hiddenDim {
      let windows = hiddenStates.dim(0)
      var reshaped = hiddenStates.reshaped(windows * mergeUnit, contextDim)
      reshaped = norm(reshaped)
      reshaped = reshaped.reshaped(windows, hiddenDim)
      var merged = mlpInput(reshaped)
      merged = MLXNN.gelu(merged)
      merged = mlpOutput(merged)
      return merged.asType(targetType)
    } else {
      preconditionFailure("Unexpected feature dimension: got \(features); expected \(contextDim) or \(hiddenDim)")
    }
  }
}
