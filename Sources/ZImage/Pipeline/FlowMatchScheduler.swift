import Foundation
import MLX

public struct FlowMatchEulerScheduler {
  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int

  public init(
    numInferenceSteps: Int,
    config: ZImageSchedulerConfig,
    mu: Float? = nil
  ) {
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    precondition(config.numTrainTimesteps > 0, "numTrainTimesteps must be positive")

    let numTrainTimesteps = Float(config.numTrainTimesteps)
    let shift = config.shift

    let initSigmaMin: Float = 1.0 / numTrainTimesteps

    let shiftedSigmaMin = shift * initSigmaMin / (1 + (shift - 1) * initSigmaMin)
    let sigmaMax: Float = 1.0
    let sigmaMin: Float = shiftedSigmaMin

    var timesteps = Self.linspace(sigmaMax * numTrainTimesteps, sigmaMin * numTrainTimesteps, count: numInferenceSteps)

    var sigmas = timesteps.map { $0 / numTrainTimesteps }

    if config.useDynamicShifting, let mu {
      sigmas = sigmas.map { sigma in
        let shifted = exp(mu) / (exp(mu) + pow(1 / sigma - 1, 1.0))
        return shifted
      }
    } else if abs(shift - 1.0) > Float.ulpOfOne {
      sigmas = sigmas.map { sigma in
        let numerator = shift * sigma
        let denominator = 1 + (shift - 1) * sigma
        return denominator > 0 ? numerator / denominator : sigma
      }
    }

    timesteps = sigmas.map { $0 * numTrainTimesteps }

    sigmas.append(0.0)

    self.sigmas = MLXArray(sigmas, [sigmas.count])
    self.timesteps = MLXArray(timesteps, [timesteps.count])
    self.numInferenceSteps = numInferenceSteps
  }

  public func step(modelOutput: MLXArray, timestepIndex: Int, sample: MLXArray) -> MLXArray {
    precondition(timestepIndex >= 0 && timestepIndex + 1 < sigmas.dim(0), "invalid timestep index")
    let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(sample.dtype)
    return sample + modelOutput * dt
  }

  private static func linspace(_ start: Float, _ end: Float, count: Int) -> [Float] {
    guard count > 1 else { return [start] }
    let step = (end - start) / Float(count - 1)
    return (0..<count).map { idx in start + Float(idx) * step }
  }
}
