import Foundation
import MLX
import MLXFast
import MLXNN

public struct VAEConfig {
  public let inChannels: Int
  public let outChannels: Int
  public let latentChannels: Int
  public let scalingFactor: Float
  public let shiftFactor: Float
  public let blockOutChannels: [Int]
  public let layersPerBlock: Int
  public let normNumGroups: Int
  public let sampleSize: Int
  public let midBlockAddAttention: Bool

  public init(
    inChannels: Int = 3,
    outChannels: Int = 3,
    latentChannels: Int = ZImageModelMetadata.VAE.latentChannels,
    scalingFactor: Float = ZImageModelMetadata.VAE.scalingFactor,
    shiftFactor: Float = ZImageModelMetadata.VAE.shiftFactor,
    blockOutChannels: [Int] = ZImageModelMetadata.VAE.blockOutChannels,
    layersPerBlock: Int = 2,
    normNumGroups: Int = 32,
    sampleSize: Int = ZImageModelMetadata.VAE.sampleSize,
    midBlockAddAttention: Bool = true
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.latentChannels = latentChannels
    self.scalingFactor = scalingFactor
    self.shiftFactor = shiftFactor
    self.blockOutChannels = blockOutChannels
    self.layersPerBlock = layersPerBlock
    self.normNumGroups = normNumGroups
    self.sampleSize = sampleSize
    self.midBlockAddAttention = midBlockAddAttention
  }

  public var vaeScaleFactor: Int {
    max(1, 1 << max(0, blockOutChannels.count - 1))
  }

  public var latentDivisor: Int {
    vaeScaleFactor * 2
  }
}

private final class VAESelfAttention: Module {
  @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: [Linear]

  init(channels: Int, normGroups: Int) {
    self._groupNorm.wrappedValue = GroupNorm(
      groupCount: normGroups, dimensions: channels, pytorchCompatible: true)
    self._toQ.wrappedValue = Linear(channels, channels)
    self._toK.wrappedValue = Linear(channels, channels)
    self._toV.wrappedValue = Linear(channels, channels)
    self._toOut.wrappedValue = [Linear(channels, channels)]
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (b, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])

    var hidden = groupNorm(x)

    let queries = toQ(hidden).reshaped(b, h * w, c).expandedDimensions(axis: 1)
    let keys = toK(hidden).reshaped(b, h * w, c).expandedDimensions(axis: 1)
    let values = toV(hidden).reshaped(b, h * w, c).expandedDimensions(axis: 1)

    let scale = 1 / sqrt(Float(c))

    let attn = MLXFast.scaledDotProductAttention(
      queries: queries,
      keys: keys,
      values: values,
      scale: scale,
      mask: nil
    )

    hidden = attn.squeezed(axis: 1).reshaped(b, h, w, c)

    hidden = toOut[0](hidden)
    return x + hidden
  }
}

private final class VAEResnetBlock2D: Module {
  @ModuleInfo(key: "norm1") var norm1: GroupNorm
  @ModuleInfo(key: "norm2") var norm2: GroupNorm
  @ModuleInfo(key: "conv1") var conv1: Conv2d
  @ModuleInfo(key: "conv2") var conv2: Conv2d
  @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

  let isConvShortcut: Bool

  init(
    inChannels: Int,
    outChannels: Int,
    normGroups: Int,
    eps: Float = 1e-6
  ) {
    self._norm1.wrappedValue = GroupNorm(
      groupCount: normGroups, dimensions: inChannels, eps: eps, affine: true, pytorchCompatible: true)
    self._norm2.wrappedValue = GroupNorm(
      groupCount: normGroups, dimensions: outChannels, eps: eps, affine: true, pytorchCompatible: true)
    self._conv1.wrappedValue = Conv2d(
      inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
    self._conv2.wrappedValue = Conv2d(
      inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)

    self.isConvShortcut = inChannels != outChannels
    if isConvShortcut {
      self._convShortcut.wrappedValue = Conv2d(
        inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, stride: 1)
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = silu(norm1(x))
    hidden = conv1(hidden)
    hidden = silu(norm2(hidden))
    hidden = conv2(hidden)
    let residual = isConvShortcut ? convShortcut!(x) : x
    let out = residual + hidden
    MLX.eval(out)
    return out
  }
}

private final class VAEUpSampler: Module {
  @ModuleInfo(key: "conv") var conv: Conv2d

  init(channels: Int) {
    self._conv.wrappedValue = Conv2d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let upscaled = VAEUpSampler.upSampleNearest(x)
    let y = conv(upscaled)
    MLX.eval(y)
    return y
  }

  static func upSampleNearest(_ x: MLXArray, scale: Int = 2) -> MLXArray {
    precondition(x.ndim == 4)
    // Use MLXNN.Upsample to avoid large broadcast+reshape intermediates
    return MLXNN.Upsample(scaleFactor: .array([Float(scale), Float(scale)]), mode: .nearest)(x)
  }
}

private final class VAEDownSampler: Module {
  @ModuleInfo(key: "conv") var conv: Conv2d

  init(channels: Int) {
    self._conv.wrappedValue = Conv2d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: 3,
      stride: 2
    )
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = padded(x, widths: [[0, 0], [0, 1], [0, 1], [0, 0]])
    hidden = conv(hidden)
    return hidden
  }
}

private final class VAEUpBlock: Module {
  @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock2D]
  @ModuleInfo(key: "upsamplers") var upsamplers: [VAEUpSampler]

  init(
    inChannels: Int,
    outChannels: Int,
    blockCount: Int,
    hasUpsampler: Bool,
    normGroups: Int
  ) {
    self._resnets.wrappedValue = (0..<blockCount).map { index in
      let isFirst = index == 0
      let resnetIn = isFirst ? inChannels : outChannels
      return VAEResnetBlock2D(
        inChannels: resnetIn,
        outChannels: outChannels,
        normGroups: normGroups
      )
    }

    if hasUpsampler {
      self._upsamplers.wrappedValue = [VAEUpSampler(channels: outChannels)]
    } else {
      self._upsamplers.wrappedValue = []
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for resnet in resnets {
      hidden = resnet(hidden)
    }
    if !upsamplers.isEmpty {
      hidden = upsamplers[0](hidden)
    }
    return hidden
  }
}

private final class VAEDownBlock: Module {
  @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock2D]
  @ModuleInfo(key: "downsamplers") var downsamplers: [VAEDownSampler]

  init(
    inChannels: Int,
    outChannels: Int,
    blockCount: Int,
    hasDownsampler: Bool,
    normGroups: Int
  ) {
    self._resnets.wrappedValue = (0..<blockCount).map { index in
      let isFirst = index == 0
      let resnetIn = isFirst ? inChannels : outChannels
      return VAEResnetBlock2D(
        inChannels: resnetIn,
        outChannels: outChannels,
        normGroups: normGroups
      )
    }

    if hasDownsampler {
      self._downsamplers.wrappedValue = [VAEDownSampler(channels: outChannels)]
    } else {
      self._downsamplers.wrappedValue = []
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for resnet in resnets {
      hidden = resnet(hidden)
    }
    if !downsamplers.isEmpty {
      hidden = downsamplers[0](hidden)
    }
    return hidden
  }
}

private final class VAEMidBlock: Module {
  @ModuleInfo(key: "attentions") var attentions: [VAESelfAttention]
  @ModuleInfo(key: "resnets") var resnets: [VAEResnetBlock2D]

  init(channels: Int, normGroups: Int, addAttention: Bool) {
    self._attentions.wrappedValue = addAttention ? [VAESelfAttention(channels: channels, normGroups: normGroups)] : []
    self._resnets.wrappedValue = [
      VAEResnetBlock2D(inChannels: channels, outChannels: channels, normGroups: normGroups),
      VAEResnetBlock2D(inChannels: channels, outChannels: channels, normGroups: normGroups)
    ]
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = resnets[0](x)
    if !attentions.isEmpty {
      hidden = attentions[0](hidden)
    }
    hidden = resnets[1](hidden)
    return hidden
  }
}

final class VAEDecoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "mid_block") fileprivate var midBlock: VAEMidBlock
  @ModuleInfo(key: "up_blocks") fileprivate var upBlocks: [VAEUpBlock]
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d

  init(config: VAEConfig) {
    let channels = config.blockOutChannels
    self._convIn.wrappedValue = Conv2d(
      inputChannels: config.latentChannels,
      outputChannels: channels.last ?? 512,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._midBlock.wrappedValue = VAEMidBlock(
      channels: channels.last ?? 512,
      normGroups: config.normNumGroups,
      addAttention: config.midBlockAddAttention
    )

    var up: [VAEUpBlock] = []
    let reversed = channels.reversed()
    for (index, channel) in reversed.enumerated() {
      let isLast = index == channels.count - 1
      let prevOut = index == 0 ? channel : (reversed[reversed.index(reversed.startIndex, offsetBy: index - 1)])
      up.append(
        VAEUpBlock(
          inChannels: prevOut,
          outChannels: channel,
          blockCount: config.layersPerBlock + 1,
          hasUpsampler: !isLast,
          normGroups: config.normNumGroups
        )
      )
    }
    self._upBlocks.wrappedValue = up
    self._convNormOut.wrappedValue = GroupNorm(
      groupCount: config.normNumGroups,
      dimensions: channels.first ?? 128,
      eps: 1e-6,
      affine: true,
      pytorchCompatible: true
    )
    self._convOut.wrappedValue = Conv2d(
      inputChannels: channels.first ?? 128,
      outputChannels: config.outChannels,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    super.init()
  }

  func callAsFunction(_ latents: MLXArray) -> MLXArray {
    var hidden = convIn(latents)
    hidden = midBlock(hidden)
    for block in upBlocks {
      hidden = block(hidden)
    }
    hidden = silu(convNormOut(hidden))
    hidden = convOut(hidden)
    return hidden
  }
}

private final class VAEEncoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "mid_block") var midBlock: VAEMidBlock
  @ModuleInfo(key: "down_blocks") var downBlocks: [VAEDownBlock]
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d

  init(config: VAEConfig) {
    let channels = config.blockOutChannels
    self._convIn.wrappedValue = Conv2d(
      inputChannels: config.inChannels,
      outputChannels: channels.first ?? 128,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )

    var downs: [VAEDownBlock] = []
    for (index, channel) in channels.enumerated() {
      let isLast = index == channels.count - 1
      let inCh = index == 0 ? channels.first! : channels[index - 1]
      downs.append(
        VAEDownBlock(
          inChannels: inCh,
          outChannels: channel,
          blockCount: config.layersPerBlock,
          hasDownsampler: !isLast,
          normGroups: config.normNumGroups
        )
      )
    }
    self._downBlocks.wrappedValue = downs
    self._midBlock.wrappedValue = VAEMidBlock(
      channels: channels.last ?? 512,
      normGroups: config.normNumGroups,
      addAttention: config.midBlockAddAttention
    )
    self._convNormOut.wrappedValue = GroupNorm(
      groupCount: config.normNumGroups,
      dimensions: channels.last ?? 512,
      eps: 1e-6,
      affine: true,
      pytorchCompatible: true
    )
    self._convOut.wrappedValue = Conv2d(
      inputChannels: channels.last ?? 512,
      outputChannels: config.latentChannels * 2,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = convIn(x)
    for block in downBlocks {
      hidden = block(hidden)
    }
    hidden = midBlock(hidden)
    hidden = silu(convNormOut(hidden))
    hidden = convOut(hidden)
    return hidden
  }
}

public final class AutoencoderKL: Module {
  public let configuration: VAEConfig
  @ModuleInfo(key: "decoder") private var decoder: VAEDecoder
  @ModuleInfo(key: "encoder") private var encoder: VAEEncoder

  public init(configuration: VAEConfig = .init()) {
    self.configuration = configuration
    self._decoder.wrappedValue = VAEDecoder(config: configuration)
    self._encoder.wrappedValue = VAEEncoder(config: configuration)
    super.init()
  }

  public func decode(_ latents: MLXArray, return_dict: Bool = false) -> (MLXArray, Any) {
    var x = latents
    x = x.transposed(0, 2, 3, 1)
    x = (x / MLXArray(configuration.scalingFactor)) + MLXArray(configuration.shiftFactor)
    x = decoder(x)
    x = x.transposed(0, 3, 1, 2)
    return (x, [:] as [String: Int])
  }

  public func encode(_ images: MLXArray) -> MLXArray {
    var hidden = images.transposed(0, 2, 3, 1)
    hidden = encoder(hidden)
    hidden = hidden.transposed(0, 3, 1, 2)
    return hidden
  }
}
