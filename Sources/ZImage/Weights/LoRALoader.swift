import Foundation
import Hub
import Logging
import MLX
import MLXNN

public struct LoRALoader {
  private let logger: Logger
  private let hubApi: HubApi

  public init(logger: Logger = Logger(label: "z-image.lora"), hubApi: HubApi = .shared) {
    self.logger = logger
    self.hubApi = hubApi
  }

  public func loadLoRAWeights(from loraPath: String, dtype: DType = .bfloat16) async throws -> [String: MLXArray] {
    // Resolve local file/dir vs remote repo
    if FileManager.default.fileExists(atPath: loraPath) {
      let url = URL(fileURLWithPath: loraPath)
      let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
      if !isDir && url.pathExtension == "safetensors" {
        logger.info("Loading LoRA from file: \(url.lastPathComponent)")
        return try Self.loadLoRAWeights(fileURL: url, dtype: dtype, logger: logger)
      } else {
        logger.info("Loading LoRA from directory: \(url.path)")
        return try Self.loadLoRAWeights(directory: url, dtype: dtype, logger: logger)
      }
    }

    // Otherwise, treat as a HuggingFace repo id
    logger.info("Downloading LoRA from HuggingFace: \(loraPath)")
    let repo = Hub.Repo(id: loraPath)
    let loraDirectory = try await hubApi.snapshot(
      from: repo,
      matching: ["*.safetensors"]
    ) { progress in
      let percent = Int(progress.fractionCompleted * 100)
      if percent % 20 == 0 {
        self.logger.info("LoRA download: \(percent)%")
      }
    }

    return try Self.loadLoRAWeights(directory: loraDirectory, dtype: dtype, logger: logger)
  }

  public static func loadLoRAWeights(directory: URL, dtype: DType, logger: Logger) throws -> [String: MLXArray] {
    var loraWeights = [String: MLXArray]()

    guard let enumerator = FileManager.default.enumerator(
      at: directory, includingPropertiesForKeys: [.isRegularFileKey]
    ) else {
      throw LoRAError.directoryNotFound(directory.path)
    }

    for case let url as URL in enumerator {
      if url.pathExtension == "safetensors" {
        logger.info("Loading LoRA weights from: \(url.lastPathComponent)")
        let fileWeights = try loadLoRAWeights(fileURL: url, dtype: dtype, logger: logger)
        for (k, v) in fileWeights { loraWeights[k] = v }
      }
    }

    logger.info("Loaded \(loraWeights.count) LoRA tensors")
    return loraWeights
  }

  private static func loadLoRAWeights(fileURL: URL, dtype: DType, logger: Logger) throws -> [String: MLXArray] {
    // Prefer SafeTensorsReader for robustness and to inspect keys for variant detection
    let reader = try SafeTensorsReader(fileURL: fileURL)
    let names = reader.tensorNames

    // Detect format
    let hasLycoris = names.contains { $0.contains("lycoris_") || $0.contains(".lokr_w1") || $0.contains(".lokr_w2") }
    let hasStandard = names.contains { $0.contains(".lora_A.weight") || $0.contains(".lora_B.weight") || $0.contains(".lora_down.weight") || $0.contains(".lora_up.weight") }

    if hasLycoris {
      logger.info("Detected LoRA format: LyCORIS/LoKr in \(fileURL.lastPathComponent)")
    } else if hasStandard {
      logger.info("Detected LoRA format: Standard LoRA in \(fileURL.lastPathComponent)")
    } else {
      logger.warning("Unknown LoRA format for \(fileURL.lastPathComponent); attempting generic load")
    }

    var results: [String: MLXArray] = [:]

    if hasLycoris {
      // Remap LyCORIS keys to module paths and normalize dtype
      for name in names {
        // Only keep lokr matrices and optional alpha scalars
        if name.hasSuffix(".lokr_w1") || name.hasSuffix(".lokr_w2") || name.hasSuffix(".alpha") {
          let tensor = try reader.tensor(named: name)
          let newKey = remapWeightKey(name)
          let value = tensor.dtype == dtype ? tensor : tensor.asType(dtype)
          results[newKey] = value
        }
      }
      logger.info("Remapped LyCORIS tensors: \(results.count)")
      return results
    }

    // Fallback: standard LoRA-style keys
    for name in names {
      let tensor = try reader.tensor(named: name)
      let newKey = remapWeightKey(name)
      let value = tensor.dtype == dtype ? tensor : tensor.asType(dtype)
      results[newKey] = value
    }
    return results
  }

  internal static func remapWeightKey(_ key: String) -> String {
    var newKey = key

    // Handle "lora_unet_" prefix common in Diffusers/PEFT format
    if newKey.hasPrefix("lora_unet_") {
      newKey = String(newKey.dropFirst("lora_unet_".count))
    }

    // Handle "diffusion_model." prefix common in Z-Image LoRA format
    if newKey.hasPrefix("diffusion_model.") {
      newKey = String(newKey.dropFirst("diffusion_model.".count))
    }

    // Handle ".ff." or ".ff_context." for feed-forward layers (Flux format)
    if newKey.contains(".ff.") || newKey.contains(".ff_context.") {
      let components = newKey.components(separatedBy: ".")
      if components.count >= 5 {
        let blockIndex = components[1]
        let ffType = components[2]
        let netIndex = components[4]

        if netIndex == "0" {
          return "transformer_blocks.\(blockIndex).\(ffType).linear1.\(components.last ?? "")"
        } else if netIndex == "2" {
          return "transformer_blocks.\(blockIndex).\(ffType).linear2.\(components.last ?? "")"
        }
      }
    }

    // Handle LyCORIS LoKr keyspace -> Z-Image transformer modules
    // Patterns we handle:
    //   lycoris_transformer_blocks_{i}_attn_to_{q|k|v}.lokr_w{1|2}
    //   lycoris_transformer_blocks_{i}_attn_to_out_0.lokr_w{1|2}
    if newKey.hasPrefix("lycoris_transformer_blocks_") {
      // Extract layer index
      let prefix = "lycoris_transformer_blocks_"
      let remainder = String(newKey.dropFirst(prefix.count))
      // remainder example: "0_attn_to_q.lokr_w1"
      if let underscoreIndex = remainder.firstIndex(of: "_") {
        let layerStr = String(remainder[..<underscoreIndex])
        let after = String(remainder[remainder.index(after: underscoreIndex)...])
        if let layerIdx = Int(layerStr) {
          // Attention projections
          if after.hasPrefix("attn_to_q") {
            let suffix = String(after.dropFirst("attn_to_q".count)) // e.g. ".lokr_w1"
            return "layers.\(layerIdx).attention.to_q\(suffix)"
          } else if after.hasPrefix("attn_to_k") {
            let suffix = String(after.dropFirst("attn_to_k".count))
            return "layers.\(layerIdx).attention.to_k\(suffix)"
          } else if after.hasPrefix("attn_to_v") {
            let suffix = String(after.dropFirst("attn_to_v".count))
            return "layers.\(layerIdx).attention.to_v\(suffix)"
          } else if after.hasPrefix("attn_to_out_0") {
            let suffix = String(after.dropFirst("attn_to_out_0".count))
            return "layers.\(layerIdx).attention.to_out.0\(suffix)"
          }
        }
      }
    }

    return newKey
  }
}

public func applyLoRAWeights(
  to transformer: ZImageTransformer2DModel,
  loraWeights: [String: MLXArray],
  loraScale: Float = 1.0,
  logger: Logger
) {
  var layerUpdates: [String: MLXArray] = [:]
  var appliedCount = 0
  var appliedLoKrCount = 0

  for (key, module) in transformer.namedModules() {
    // Try different key patterns for LoRA weights
    // The LoRA keys after remapping should match the transformer module keys
    let keyPatterns = [
      key,
      "transformer.\(key)",
      key.replacingOccurrences(of: ".", with: "_")
    ]

    for pattern in keyPatterns {
      // Standard LoRA keys
      let loraAKey = "\(pattern).lora_A.weight"
      let loraBKey = "\(pattern).lora_B.weight"
      let loraAKeyAlt = "\(pattern).lora_down.weight"
      let loraBKeyAlt = "\(pattern).lora_up.weight"

      let loraA = loraWeights[loraAKey] ?? loraWeights[loraAKeyAlt]
      let loraB = loraWeights[loraBKey] ?? loraWeights[loraBKeyAlt]

      if let loraA = loraA, let loraB = loraB {
        // Fast shape guard before heavy ops
        let outDim = loraB.dim(max(0, loraB.ndim - 2)) // rows of B
        let inDim = loraA.dim(max(0, loraA.ndim - 1))  // cols of A
        var targetOut: Int = -1
        var targetIn: Int = -1
        if let qlin = module as? QuantizedLinear {
          targetOut = qlin.weight.dim(max(0, qlin.weight.ndim - 2))
          targetIn = qlin.weight.dim(max(0, qlin.weight.ndim - 1))
        } else if let lin = module as? Linear {
          targetOut = lin.weight.dim(max(0, lin.weight.ndim - 2))
          targetIn = lin.weight.dim(max(0, lin.weight.ndim - 1))
        }
        if targetOut > 0 && targetIn > 0 && (outDim != targetOut || inDim != targetIn) {
          logger.debug("Skipping LoRA pair for \(key): delta (\(outDim)x\(inDim)) vs weight (\(targetOut)x\(targetIn))")
          continue
        }
        if let quantizedLinear = module as? QuantizedLinear {
          logger.debug("Applying LoRA to quantized layer: \(key)")

          let dequantizedWeight = dequantized(
            quantizedLinear.weight,
            scales: quantizedLinear.scales,
            biases: quantizedLinear.biases,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits
          )

          let loraDelta = matmul(loraB, loraA)
          let fusedWeight = dequantizedWeight + loraScale * loraDelta

          let fusedLinear = Linear(
            weight: fusedWeight,
            bias: quantizedLinear.bias
          )

          let requantized = QuantizedLinear(
            fusedLinear,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits
          )

          layerUpdates["\(key).weight"] = requantized.weight
          layerUpdates["\(key).scales"] = requantized.scales
          layerUpdates["\(key).biases"] = requantized.biases
          appliedCount += 1

        } else if let linear = module as? Linear {
          logger.debug("Applying LoRA to linear layer: \(key)")
          let loraDelta = matmul(loraB, loraA)
          let currentWeight = linear.weight
          let newWeight = currentWeight + loraScale * loraDelta
          layerUpdates["\(key).weight"] = newWeight
          appliedCount += 1
        }

        break
      }

      // LyCORIS LoKr keys
      let lokrW1Key = "\(pattern).lokr_w1"
      let lokrW2Key = "\(pattern).lokr_w2"
      if let w1 = loraWeights[lokrW1Key], let w2 = loraWeights[lokrW2Key] {
        // Optional alpha per-module
        let alphaKey = "\(pattern).alpha"
        var alphaScale: Float = 1.0
        if let alpha = loraWeights[alphaKey] {
          if let v = alpha.asArray(Float.self).first {
              alphaScale = v
          }
        }

        // Compute Kronecker product delta with layout [a*c, b*d]
        func kron2D(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
          precondition(a.ndim == 2 && b.ndim == 2, "LoKr expects 2D matrices for Linear layers")
          let a0 = a.dim(0), a1 = a.dim(1)
          let b0 = b.dim(0), b1 = b.dim(1)
          var aExp = a.reshaped(a0, 1, a1, 1)
          var bExp = b.reshaped(1, b0, 1, b1)
          // Broadcast multiply then reshape
          let prod = aExp * bExp
          return prod.reshaped(a0 * b0, a1 * b1)
        }

        // Fast shape guard against module weight before computing kron
        let outDim = w1.dim(0) * w2.dim(0)
        let inDim = w1.dim(1) * w2.dim(1)
        var targetOut: Int = -1
        var targetIn: Int = -1
        if let qlin = module as? QuantizedLinear {
          targetOut = qlin.weight.dim(max(0, qlin.weight.ndim - 2))
          targetIn = qlin.weight.dim(max(0, qlin.weight.ndim - 1))
        } else if let lin = module as? Linear {
          targetOut = lin.weight.dim(max(0, lin.weight.ndim - 2))
          targetIn = lin.weight.dim(max(0, lin.weight.ndim - 1))
        }
        if targetOut > 0 && targetIn > 0 && (outDim != targetOut || inDim != targetIn) {
          logger.debug("Skipping LoKr for \(key): kron (\(outDim)x\(inDim)) vs weight (\(targetOut)x\(targetIn))")
          continue
        }

        if let quantizedLinear = module as? QuantizedLinear {
          logger.debug("Applying LoKr to quantized layer: \(key)")
          let dequantizedWeight = dequantized(
            quantizedLinear.weight,
            scales: quantizedLinear.scales,
            biases: quantizedLinear.biases,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits
          )

          var delta = kron2D(w1, w2)
          if delta.dtype != dequantizedWeight.dtype { delta = delta.asType(dequantizedWeight.dtype) }
          let fusedWeight = dequantizedWeight + (loraScale * alphaScale) * delta

          let fusedLinear = Linear(weight: fusedWeight, bias: quantizedLinear.bias)
          let requantized = QuantizedLinear(
            fusedLinear,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits
          )
          layerUpdates["\(key).weight"] = requantized.weight
          layerUpdates["\(key).scales"] = requantized.scales
          layerUpdates["\(key).biases"] = requantized.biases
          appliedLoKrCount += 1
        } else if let linear = module as? Linear {
          logger.debug("Applying LoKr to linear layer: \(key)")
          var delta = kron2D(w1, w2)
          if delta.dtype != linear.weight.dtype { delta = delta.asType(linear.weight.dtype) }
          let newWeight = linear.weight + (loraScale * alphaScale) * delta
          layerUpdates["\(key).weight"] = newWeight
          appliedLoKrCount += 1
        }

        break
      }
    }
  }

  if !layerUpdates.isEmpty {
    do {
      try transformer.update(parameters: ModuleParameters.unflattened(layerUpdates), verify: [.shapeMismatch])
      if appliedLoKrCount > 0 {
        logger.info("Applied LoRA weights to \(appliedCount) layers; LoKr applied to \(appliedLoKrCount) layers")
      } else {
        logger.info("Applied LoRA weights to \(appliedCount) layers")
      }
    } catch {
      logger.error("Failed to apply LoRA weights: \(error)")
    }
  } else {
    logger.warning("No matching LoRA weights found for transformer layers")
  }
}

public enum LoRAError: Error, LocalizedError {
  case directoryNotFound(String)
  case weightsNotFound(String)
  case applicationFailed(String)

  public var errorDescription: String? {
    switch self {
    case .directoryNotFound(let path):
      return "LoRA directory not found: \(path)"
    case .weightsNotFound(let path):
      return "LoRA weights not found at: \(path)"
    case .applicationFailed(let reason):
      return "Failed to apply LoRA: \(reason)"
    }
  }
}
