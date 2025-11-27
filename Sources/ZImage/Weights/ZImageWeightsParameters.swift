import Foundation
import MLX
import MLXNN

public enum ZImageWeightsParameters {

  public static func transformerParameters(
    from tensors: [String: MLXArray],
    manifest: ZImageQuantizationManifest?,
    dtype: DType? = nil
  ) throws -> ModuleParameters {
    var source = tensors
    var flat: [String: MLXArray] = [:]

    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "t_embedder.mlp.0", tensorBase: "t_embedder.mlp.0",
      hasBias: true, dtype: dtype, manifest: manifest
    )
    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "t_embedder.mlp.2", tensorBase: "t_embedder.mlp.2",
      hasBias: true, dtype: dtype, manifest: manifest
    )

    if source["cap_embedder.0.weight"] != nil {
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "capEmbedNorm", tensorBase: "cap_embedder.0",
        hasBias: false, dtype: dtype, manifest: manifest
      )
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "capEmbedLinear", tensorBase: "cap_embedder.1",
        hasBias: true, dtype: dtype, manifest: manifest
      )
    }

    let layerIndices = collectIndices(from: source.keys, prefix: "layers.")
    for i in layerIndices {
      try assignTransformerBlock(
        flat: &flat, source: &source,
        modulePath: "layers.\(i)", tensorBase: "layers.\(i)",
        dtype: dtype, manifest: manifest
      )
    }

    let noiseRefinerIndices = collectIndices(from: source.keys, prefix: "noise_refiner.")
    for i in noiseRefinerIndices {
      try assignTransformerBlock(
        flat: &flat, source: &source,
        modulePath: "noise_refiner.\(i)", tensorBase: "noise_refiner.\(i)",
        dtype: dtype, manifest: manifest
      )
    }

    let contextRefinerIndices = collectIndices(from: source.keys, prefix: "context_refiner.")
    for i in contextRefinerIndices {
      try assignTransformerBlock(
        flat: &flat, source: &source,
        modulePath: "context_refiner.\(i)", tensorBase: "context_refiner.\(i)",
        dtype: dtype, manifest: manifest
      )
    }

    if source["all_x_embedder.2-1.weight"] != nil {
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "all_x_embedder.2-1", tensorBase: "all_x_embedder.2-1",
        hasBias: true, dtype: dtype, manifest: manifest
      )
    }

    if source["all_final_layer.2-1.linear.weight"] != nil {
      if let w = source.removeValue(forKey: "all_final_layer.2-1.norm_final.weight") {
        flat["all_final_layer.2-1.norm_final.weight"] = dtypeAdjusted(w, dtype: dtype)
      }
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "all_final_layer.2-1.linear", tensorBase: "all_final_layer.2-1.linear",
        hasBias: true, dtype: dtype, manifest: manifest
      )
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "all_final_layer.2-1.adaLN_modulation.1", tensorBase: "all_final_layer.2-1.adaLN_modulation.1",
        hasBias: true, dtype: dtype, manifest: manifest
      )
    }

    _ = source.removeValue(forKey: "x_pad_token")
    _ = source.removeValue(forKey: "cap_pad_token")

    return ModuleParameters.unflattened(flat)
  }

  private static func assignTransformerBlock(
    flat: inout [String: MLXArray],
    source: inout [String: MLXArray],
    modulePath: String,
    tensorBase: String,
    dtype: DType?,
    manifest: ZImageQuantizationManifest?
  ) throws {
    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "\(modulePath).adaLN_modulation.0", tensorBase: "\(tensorBase).adaLN_modulation.0",
      hasBias: true, dtype: dtype, manifest: manifest
    )

    if let w = source.removeValue(forKey: "\(tensorBase).norm1.weight") {
      flat["\(modulePath).norm1.weight"] = dtypeAdjusted(w, dtype: dtype)
    }
    if let w = source.removeValue(forKey: "\(tensorBase).norm2.weight") {
      flat["\(modulePath).norm2.weight"] = dtypeAdjusted(w, dtype: dtype)
    }

    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "\(modulePath).attention.to_q", tensorBase: "\(tensorBase).attention.to_q",
      hasBias: false, dtype: dtype, manifest: manifest
    )
    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "\(modulePath).attention.to_k", tensorBase: "\(tensorBase).attention.to_k",
      hasBias: false, dtype: dtype, manifest: manifest
    )
    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "\(modulePath).attention.to_v", tensorBase: "\(tensorBase).attention.to_v",
      hasBias: false, dtype: dtype, manifest: manifest
    )
    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "\(modulePath).attention.to_out.0", tensorBase: "\(tensorBase).attention.to_out.0",
      hasBias: true, dtype: dtype, manifest: manifest
    )

    if let w = source.removeValue(forKey: "\(tensorBase).attention.norm_q.weight") {
      flat["\(modulePath).attention.norm_q.weight"] = dtypeAdjusted(w, dtype: dtype)
    }
    if let w = source.removeValue(forKey: "\(tensorBase).attention.norm_k.weight") {
      flat["\(modulePath).attention.norm_k.weight"] = dtypeAdjusted(w, dtype: dtype)
    }

    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "\(modulePath).feed_forward.w1", tensorBase: "\(tensorBase).feed_forward.w1",
      hasBias: false, dtype: dtype, manifest: manifest
    )
    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "\(modulePath).feed_forward.w2", tensorBase: "\(tensorBase).feed_forward.w2",
      hasBias: false, dtype: dtype, manifest: manifest
    )
    try assignLinear(
      flat: &flat, source: &source,
      modulePath: "\(modulePath).feed_forward.w3", tensorBase: "\(tensorBase).feed_forward.w3",
      hasBias: false, dtype: dtype, manifest: manifest
    )
  }

  public static func textEncoderParameters(
    from tensors: [String: MLXArray],
    manifest: ZImageQuantizationManifest?,
    dtype: DType? = nil
  ) throws -> ModuleParameters {
    var source = tensors
    var flat: [String: MLXArray] = [:]

    if let w = source.removeValue(forKey: "model.embed_tokens.weight") {
      flat["encoder.embed_tokens.weight"] = dtypeAdjusted(w, dtype: dtype)
    }

    if let w = source.removeValue(forKey: "model.norm.weight") {
      flat["encoder.norm.weight"] = dtypeAdjusted(w, dtype: dtype)
    }

    let layerIndices = collectIndices(from: source.keys, prefix: "model.layers.")
    for i in layerIndices {
      let srcPrefix = "model.layers.\(i)"
      let dstPrefix = "encoder.layers.\(i)"

      if let w = source.removeValue(forKey: "\(srcPrefix).input_layernorm.weight") {
        flat["\(dstPrefix).input_layernorm.weight"] = dtypeAdjusted(w, dtype: dtype)
      }
      if let w = source.removeValue(forKey: "\(srcPrefix).post_attention_layernorm.weight") {
        flat["\(dstPrefix).post_attention_layernorm.weight"] = dtypeAdjusted(w, dtype: dtype)
      }

      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "\(dstPrefix).self_attn.q_proj", tensorBase: "\(srcPrefix).self_attn.q_proj",
        hasBias: true, dtype: dtype, manifest: manifest
      )
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "\(dstPrefix).self_attn.k_proj", tensorBase: "\(srcPrefix).self_attn.k_proj",
        hasBias: true, dtype: dtype, manifest: manifest
      )
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "\(dstPrefix).self_attn.v_proj", tensorBase: "\(srcPrefix).self_attn.v_proj",
        hasBias: true, dtype: dtype, manifest: manifest
      )
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "\(dstPrefix).self_attn.o_proj", tensorBase: "\(srcPrefix).self_attn.o_proj",
        hasBias: false, dtype: dtype, manifest: manifest
      )

      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "\(dstPrefix).mlp.gate_proj", tensorBase: "\(srcPrefix).mlp.gate_proj",
        hasBias: false, dtype: dtype, manifest: manifest
      )
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "\(dstPrefix).mlp.up_proj", tensorBase: "\(srcPrefix).mlp.up_proj",
        hasBias: false, dtype: dtype, manifest: manifest
      )
      try assignLinear(
        flat: &flat, source: &source,
        modulePath: "\(dstPrefix).mlp.down_proj", tensorBase: "\(srcPrefix).mlp.down_proj",
        hasBias: false, dtype: dtype, manifest: manifest
      )
    }

    return ModuleParameters.unflattened(flat)
  }

  public static func vaeParameters(
    from tensors: [String: MLXArray],
    dtype: DType? = nil
  ) throws -> ModuleParameters {
    var flat: [String: MLXArray] = [:]

    for (key, value) in tensors {
      if key.contains(".conv") && key.hasSuffix(".weight") && value.ndim == 4 {
        let transposed = value.transposed(0, 2, 3, 1)
        flat[key] = dtypeAdjusted(transposed, dtype: dtype)
      } else {
        flat[key] = dtypeAdjusted(value, dtype: dtype)
      }
    }

    return ModuleParameters.unflattened(flat)
  }

  private static func assignLinear(
    flat: inout [String: MLXArray],
    source: inout [String: MLXArray],
    modulePath: String,
    tensorBase: String,
    hasBias: Bool,
    dtype: DType?,
    manifest: ZImageQuantizationManifest?
  ) throws {
    let quantized = isQuantizedLayer(tensorBase: tensorBase, source: source, manifest: manifest)

    if let w = source.removeValue(forKey: "\(tensorBase).weight") {
      flat["\(modulePath).weight"] = quantized ? w : dtypeAdjusted(w, dtype: dtype)
    }

    if hasBias, let b = source.removeValue(forKey: "\(tensorBase).bias") {
      flat["\(modulePath).bias"] = dtypeAdjusted(b, dtype: dtype)
    }

    if quantized {
      if let s = source.removeValue(forKey: "\(tensorBase).scales") {
        flat["\(modulePath).scales"] = s
      }
      if let bs = source.removeValue(forKey: "\(tensorBase).biases") {
        flat["\(modulePath).biases"] = bs
      }
    }
  }

  private static func isQuantizedLayer(
    tensorBase: String,
    source: [String: MLXArray],
    manifest: ZImageQuantizationManifest?
  ) -> Bool {
    guard manifest != nil else { return false }
    return source["\(tensorBase).scales"] != nil
  }

  private static func dtypeAdjusted(_ tensor: MLXArray, dtype: DType?) -> MLXArray {
    guard let dtype, tensor.dtype != dtype else { return tensor }
    return tensor.asType(dtype)
  }

  private static func collectIndices(from keys: Dictionary<String, MLXArray>.Keys, prefix: String) -> [Int] {
    var indices = Set<Int>()
    for key in keys {
      guard key.hasPrefix(prefix) else { continue }
      let suffix = key.dropFirst(prefix.count)
      if let dotIndex = suffix.firstIndex(of: "."),
         let value = Int(suffix[..<dotIndex]) {
        indices.insert(value)
      }
    }
    return indices.sorted()
  }
}
