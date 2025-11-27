import Foundation

public struct QwenVisionConfiguration {
  public var depth: Int
  public var embedDim: Int
  public var mlpHiddenDim: Int
  public var hiddenAct: Activation
  public var numHeads: Int
  public var eps: Float
  public var patchSize: Int
  public var temporalPatchSize: Int
  public var spatialMergeSize: Int
  public var inChannels: Int
  public var outHiddenDim: Int
  public var windowSize: Int
  public var fullAttentionBlockIndices: [Int]

  public enum Activation {
    case geluApproximate
    case silu
  }

  public init(
    depth: Int = 32,
    embedDim: Int = 1_280,
    mlpHiddenDim: Int = 3_420,
    hiddenAct: Activation = .silu,
    numHeads: Int = 16,
    eps: Float = 1e-6,
    patchSize: Int = 14,
    temporalPatchSize: Int = 2,
    spatialMergeSize: Int = 2,
    inChannels: Int = 3,
    outHiddenDim: Int = 3_584,
    windowSize: Int = 112,
    fullAttentionBlockIndices: [Int] = [7, 15, 23, 31]
  ) {
    self.depth = depth
    self.embedDim = embedDim
    self.mlpHiddenDim = mlpHiddenDim
    self.hiddenAct = hiddenAct
    self.numHeads = numHeads
    self.eps = eps
    self.patchSize = patchSize
    self.temporalPatchSize = temporalPatchSize
    self.spatialMergeSize = spatialMergeSize
    self.inChannels = inChannels
    self.outHiddenDim = outHiddenDim
    self.windowSize = windowSize
    self.fullAttentionBlockIndices = fullAttentionBlockIndices
  }
}
