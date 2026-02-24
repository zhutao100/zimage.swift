import Logging
import MLX

enum ZImageTransformerOverride {
  static func inferDim(from weights: [String: MLXArray]) -> Int? {
    // Try common norm vectors first
    if let w = weights["layers.0.attention_norm1.weight"], w.ndim == 1 { return w.dim(0) }
    if let w = weights["layers.0.ffn_norm1.weight"], w.ndim == 1 { return w.dim(0) }
    // Try attention projections
    if let w = weights["layers.0.attention.to_q.weight"], w.ndim == 2 { return w.dim(0) }
    if let w = weights["layers.0.attention.to_out.0.weight"], w.ndim == 2 { return w.dim(1) }
    // Scan for any norm weight
    if let (_, w) = weights.first(where: { $0.key.hasSuffix("attention_norm1.weight") && $0.value.ndim == 1 }) { return w.dim(0) }
    if let (_, w) = weights.first(where: { $0.key.hasSuffix("ffn_norm1.weight") && $0.value.ndim == 1 }) { return w.dim(0) }
    return nil
  }

  /// Canonicalize override checkpoints so their tensor keys match our transformer module names.
  /// Supports SD/ComfyUI-style exports that prefix keys with e.g. "model.diffusion_model.".
  static func canonicalize(_ weights: [String: MLXArray], dim: Int, logger: Logger) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    for (k, v) in weights {
      // Strip common root prefixes from external checkpoints.
      var key = k
      for prefix in ["model.diffusion_model.", "diffusion_model.", "transformer.", "model."] {
        if key.hasPrefix(prefix) {
          key = String(key.dropFirst(prefix.count))
        }
      }

      // Some checkpoints use q_norm/k_norm naming; base Z-Image uses norm_q/norm_k.
      key = key.replacingOccurrences(of: ".attention.q_norm.weight", with: ".attention.norm_q.weight")
      key = key.replacingOccurrences(of: ".attention.k_norm.weight", with: ".attention.norm_k.weight")

      // Map attention.out.weight -> attention.to_out.0.weight
      if key.hasSuffix(".attention.out.weight") {
        let newKey = key.replacingOccurrences(of: ".attention.out.weight", with: ".attention.to_out.0.weight")
        out[newKey] = v
        continue
      }

      // Split attention.qkv.weight -> to_q.weight, to_k.weight, to_v.weight
      if key.hasSuffix(".attention.qkv.weight") {
        if v.ndim == 2, v.dim(0) == dim * 3, v.dim(1) == dim {
          let q = v[0 ..< dim, 0...]
          let kW = v[dim ..< 2 * dim, 0...]
          let vW = v[2 * dim ..< 3 * dim, 0...]
          let base = key.replacingOccurrences(of: ".attention.qkv.weight", with: "")
          out["\(base).attention.to_q.weight"] = q
          out["\(base).attention.to_k.weight"] = kW
          out["\(base).attention.to_v.weight"] = vW
        } else {
          logger.warning("Unexpected qkv shape for \(key): \(v.shape) (expected [\(dim * 3), \(dim)])")
        }
        continue
      }

      // Passthrough other keys
      var mapped = key
      // Remap final_layer.* -> all_final_layer.2-1.* so our loader can pick them up
      if mapped.hasPrefix("final_layer.") {
        mapped = mapped.replacingOccurrences(of: "final_layer.", with: "all_final_layer.2-1.")
      }
      // Remap x_embedder.* -> all_x_embedder.2-1.*
      if mapped.hasPrefix("x_embedder.") {
        mapped = mapped.replacingOccurrences(of: "x_embedder.", with: "all_x_embedder.2-1.")
      }
      out[mapped] = v
    }
    return out
  }
}

enum ZImageAIOTransformerValidation {
  static func missingStrictRequiredKeys(in weights: [String: MLXArray], config: ZImageTransformerConfig) -> [String] {
    var required: [String] = [
      "layers.0.attention.to_q.weight",
      "layers.0.attention.to_out.0.weight",
    ]

    if config.qkNorm {
      required.append(contentsOf: [
        "layers.0.attention.norm_q.weight",
        "layers.0.attention.norm_k.weight",
      ])
    }

    return required.filter { weights[$0] == nil }
  }

  static func coverageAuditWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var auditWeights = weights
    if let w = weights["cap_embedder.0.weight"] { auditWeights["capEmbedNorm.weight"] = w }
    if let w = weights["cap_embedder.1.weight"] { auditWeights["capEmbedLinear.weight"] = w }
    if let w = weights["cap_embedder.1.bias"] { auditWeights["capEmbedLinear.bias"] = w }
    return auditWeights
  }
}
