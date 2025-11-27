import Foundation
import MLX
import MLXNN

final class QwenVisionMLP: Module {
  @ModuleInfo(key: "gate") private var gate: Linear
  @ModuleInfo(key: "up") private var up: Linear
  @ModuleInfo(key: "down") private var down: Linear

  private let activation: QwenVisionConfiguration.Activation

  init(dim: Int, hiddenDim: Int, activation: QwenVisionConfiguration.Activation) {
    self.activation = activation
    self._gate.wrappedValue = Linear(dim, hiddenDim)
    self._up.wrappedValue = Linear(dim, hiddenDim)
    self._down.wrappedValue = Linear(hiddenDim, dim)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    let originalType = hiddenStates.dtype
    let input = hiddenStates
    var gated = gate(input)
    switch activation {
    case .geluApproximate:
      gated = MLXNN.geluFastApproximate(gated)
    case .silu:
      gated = MLXNN.silu(gated)
    }
    let upProjected = up(input)
    var hidden = gated * upProjected
    hidden = down(hidden)
    return hidden.asType(originalType)
  }

}
