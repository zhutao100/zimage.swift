import Foundation
import Logging
import MLX
import MLXNN

public enum ZImageWeightsMapping {
  public struct Partition {
    public let transformer: [String: MLXArray]
    public let textEncoder: [String: MLXArray]
    public let vae: [String: MLXArray]
    public let unassigned: [String: MLXArray]
  }

  public static func partition(weights: [String: MLXArray], logger _: Logger? = nil) -> Partition {
    var transformer: [String: MLXArray] = [:]
    var textEncoder: [String: MLXArray] = [:]
    var vae: [String: MLXArray] = [:]
    var unassigned: [String: MLXArray] = [:]

    for (key, tensor) in weights {
      if key.hasPrefix("transformer.") {
        transformer[String(key.dropFirst("transformer.".count))] = tensor
      } else if key.hasPrefix("text_encoder.") {
        textEncoder[String(key.dropFirst("text_encoder.".count))] = tensor
      } else if key.hasPrefix("vae.") {
        vae[String(key.dropFirst("vae.".count))] = tensor
      } else {
        unassigned[key] = tensor
      }
    }

    return Partition(
      transformer: transformer,
      textEncoder: textEncoder,
      vae: vae,
      unassigned: unassigned
    )
  }

  private static func transformerMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      mapped["transformer.\(k)"] = v
    }
    return mapped
  }

  private static func textEncoderMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      if k.hasPrefix("model.") {
        let remainder = String(k.dropFirst("model.".count))
        mapped["text_encoder.encoder.\(remainder)"] = v
      } else {
        mapped["text_encoder.\(k)"] = v
      }
    }
    return mapped
  }

  private static func vaeMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      var tensor = v
      if tensor.ndim == 4 {
        tensor = tensor.transposed(0, 2, 3, 1)
      }
      mapped["vae.\(k)"] = tensor
    }
    return mapped
  }

  public static func applyTransformer(
    weights: [String: MLXArray],
    to model: ZImageTransformer2DModel,
    manifest: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) {
    if weights.isEmpty {
      logger.warning("Transformer weights empty; nothing to apply.")
      return
    }

    if let manifest {
      let availableKeys = Set(weights.keys)
      ZImageQuantizer.applyQuantization(
        to: model,
        manifest: manifest,
        availableKeys: availableKeys,
        tensorNameTransform: ZImageQuantizer.transformerTensorName
      )
    }

    let mapped = transformerMapping(weights)
    ZImageModuleWeightsApplier.applyToModule(model, weights: mapped, prefix: "transformer", logger: logger)

    let groupSize = manifest?.groupSize ?? 32
    let bits = manifest?.bits ?? 8
    model.loadCapEmbedderWeights(from: weights)
    model.loadXEmbedderWeights(from: weights, groupSize: groupSize, bits: bits)
    model.loadFinalLayerWeights(from: weights, groupSize: groupSize, bits: bits)

    model.setPadTokens(xPad: weights["x_pad_token"], capPad: weights["cap_pad_token"])
  }

  public static func applyTextEncoder(
    weights: [String: MLXArray],
    to model: QwenTextEncoder,
    manifest: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) {
    if weights.isEmpty {
      logger.warning("Text encoder weights empty; nothing to apply.")
      return
    }

    if let manifest {
      let availableKeys = Set(weights.keys)
      ZImageQuantizer.applyQuantization(
        to: model,
        manifest: manifest,
        availableKeys: availableKeys,
        tensorNameTransform: ZImageQuantizer.textEncoderTensorName
      )
    }

    let mapped = textEncoderMapping(weights)
    ZImageModuleWeightsApplier.applyToModule(model, weights: mapped, prefix: "text_encoder", logger: logger)
  }

  public static func applyVAE(
    weights: [String: MLXArray],
    to model: Module,
    manifest _: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) {
    if weights.isEmpty {
      logger.warning("VAE weights empty; nothing to apply.")
      return
    }

    let mapped = vaeMapping(weights)
    ZImageModuleWeightsApplier.applyToModule(model, weights: mapped, prefix: "vae", logger: logger)
  }
}
