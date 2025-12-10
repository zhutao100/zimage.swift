import Foundation
import MLX
import MLXNN

public protocol VAEImageDecoding {
  func decode(_ latents: MLXArray, return_dict: Bool) -> (MLXArray, Any)
}

extension AutoencoderKL: VAEImageDecoding {}

/// Decoder-only variant of the VAE for image generation.
/// It builds only the decoder subgraph and exposes the same `decode` API.
public final class AutoencoderDecoderOnly: Module, VAEImageDecoding {
  public let configuration: VAEConfig
  @ModuleInfo(key: "decoder") private var decoder: VAEDecoder

  public init(configuration: VAEConfig) {
    self.configuration = configuration
    self._decoder.wrappedValue = VAEDecoder(config: configuration)
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
}

