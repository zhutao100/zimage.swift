import Foundation
import Logging
import MLX

enum ZImageAIOCheckpoint {
  struct Inspection: Sendable {
    let isAIO: Bool
    let textEncoderPrefix: String?
    let diagnostics: [String]
  }

  struct Components {
    let transformer: [String: MLXArray]
    let textEncoder: [String: MLXArray]
    let vae: [String: MLXArray]
  }

  static func inspect(fileURL: URL) -> Inspection {
    do {
      let reader = try SafeTensorsReader(fileURL: fileURL)
      return inspect(reader: reader)
    } catch {
      return Inspection(isAIO: false, textEncoderPrefix: nil, diagnostics: ["failed to read safetensors header: \(error)"])
    }
  }

  static func inspect(reader: SafeTensorsReader) -> Inspection {
    let names = Set(reader.tensorNames)

    let hasDiffusionModel = names.contains(where: { $0.hasPrefix("model.diffusion_model.") || $0.hasPrefix("diffusion_model.") })
    let textEncoderPrefix = findTextEncoderPrefix(in: names)
    let hasVAE = names.contains(where: { $0.hasPrefix("vae.") })
    let hasVAEDecoder = hasRecognizedVAEDecoder(in: names)

    var diagnostics: [String] = []
    if !hasDiffusionModel { diagnostics.append("missing transformer tensors (model.diffusion_model.*)") }
    if textEncoderPrefix == nil { diagnostics.append("missing text encoder tensors (text_encoders.*.transformer.model.*)") }
    if !hasVAE { diagnostics.append("missing VAE tensors (vae.*)") }
    if hasVAE && !hasVAEDecoder {
      diagnostics.append("missing VAE decoder tensors (expected Diffusers `vae.decoder.mid_block.*`/`vae.decoder.up_blocks.*` or ComfyUI `vae.decoder.mid.*`/`vae.decoder.up.*`)")
    }

    if let textEncoderPrefix {
      if !names.contains("\(textEncoderPrefix)model.embed_tokens.weight") {
        diagnostics.append("missing text encoder embed_tokens weight")
      }
    }

    if !(names.contains("model.diffusion_model.cap_embedder.0.weight") || names.contains("diffusion_model.cap_embedder.0.weight")) {
      diagnostics.append("missing transformer cap_embedder tensors")
    }

    return Inspection(isAIO: diagnostics.isEmpty, textEncoderPrefix: textEncoderPrefix, diagnostics: diagnostics)
  }

  static func loadComponents(
    from fileURL: URL,
    textEncoderPrefix: String,
    dtype: DType? = .bfloat16,
    logger: Logger?
  ) throws -> Components {
    let reader = try SafeTensorsReader(fileURL: fileURL)

    var transformer: [String: MLXArray] = [:]
    var textEncoder: [String: MLXArray] = [:]
    var vae: [String: MLXArray] = [:]

    for meta in reader.allMetadata() {
      let name = meta.name
      if name.hasPrefix("model.diffusion_model.") || name.hasPrefix("diffusion_model.") {
        var tensor = try reader.tensor(named: name)
        if let dtype, tensor.dtype != dtype { tensor = tensor.asType(dtype) }
        transformer[name] = tensor
        continue
      }

      if name.hasPrefix(textEncoderPrefix) {
        let stripped = String(name.dropFirst(textEncoderPrefix.count))
        var tensor = try reader.tensor(named: name)
        if let dtype, tensor.dtype != dtype { tensor = tensor.asType(dtype) }
        textEncoder[stripped] = tensor
        continue
      }

      if name.hasPrefix("vae.") {
        let stripped = String(name.dropFirst("vae.".count))
        var tensor = try reader.tensor(named: name)
        if let dtype, tensor.dtype != dtype { tensor = tensor.asType(dtype) }
        vae[stripped] = tensor
        continue
      }
    }

    logger?.info("Loaded AIO checkpoint components (transformer=\(transformer.count), text_encoder=\(textEncoder.count), vae=\(vae.count))")
    return Components(transformer: transformer, textEncoder: textEncoder, vae: vae)
  }

  /// Canonicalize ComfyUI-style VAE decoder tensor names to Diffusers-style names expected by our VAE modules.
  ///
  /// Civitai "AIO" checkpoints frequently store VAE tensors under ComfyUI naming like:
  /// - `decoder.mid.attn_1.*`, `decoder.mid.block_1.*`, `decoder.up.<i>.block.<j>.*`, `decoder.norm_out.*`
  ///
  /// Diffusers-style names used by the base Z-Image-Turbo VAE are:
  /// - `decoder.mid_block.attentions.0.*`, `decoder.mid_block.resnets.*`, `decoder.up_blocks.*`, `decoder.conv_norm_out.*`
  ///
  /// This function expects keys WITHOUT the leading `vae.` prefix (matching `loadComponents` output).
  static func canonicalizeVAEWeights(
    _ weights: [String: MLXArray],
    expectedUpBlocks: Int,
    logger: Logger? = nil
  ) -> [String: MLXArray] {
    guard !weights.isEmpty else { return weights }

    let keys = weights.keys
    let looksDiffusers = keys.contains(where: { $0.contains("decoder.mid_block.") || $0.contains("decoder.up_blocks.") })
    let looksComfy = keys.contains(where: {
      $0.hasPrefix("decoder.mid.attn_1.")
        || $0.hasPrefix("decoder.mid.block_")
        || $0.hasPrefix("decoder.up.")
        || $0.hasPrefix("decoder.norm_out.")
    })

    let upBlocks = max(1, expectedUpBlocks)
    var out: [String: MLXArray] = [:]
    out.reserveCapacity(weights.count)

    var rewritten = 0
    var squeezed = 0
    for (key, tensor) in weights {
      let mapped = looksComfy ? mapComfyVAEKeyToDiffusers(key, upBlocks: upBlocks) : key
      if mapped != key { rewritten += 1 }

      let canonicalTensor = canonicalizeVAEDecoderTensor(for: mapped, tensor: tensor)
      if canonicalTensor.ndim != tensor.ndim { squeezed += 1 }

      out[mapped] = canonicalTensor
    }

    if looksComfy, !looksDiffusers {
      logger?.info("Canonicalized ComfyUI VAE keys -> rewritten \(rewritten)/\(weights.count) tensors (upBlocks=\(upBlocks), squeezed=\(squeezed))")
    } else if squeezed > 0 {
      logger?.info("Canonicalized VAE decoder tensor shapes -> squeezed \(squeezed)/\(weights.count) tensors")
    }
    return out
  }

  private static func findTextEncoderPrefix(in names: Set<String>) -> String? {
    // Look for: text_encoders.<name>.transformer.model.*
    for name in names {
      guard name.hasPrefix("text_encoders.") else { continue }
      let parts = name.split(separator: ".", omittingEmptySubsequences: false)
      guard parts.count >= 4 else { continue }
      guard parts[0] == "text_encoders" else { continue }
      guard parts[2] == "transformer" else { continue }
      let encoderName = parts[1]
      return "text_encoders.\(encoderName).transformer."
    }
    return nil
  }

  private static func hasRecognizedVAEDecoder(in names: Set<String>) -> Bool {
    guard names.contains("vae.decoder.conv_in.weight"),
          names.contains("vae.decoder.conv_out.weight") else { return false }

    let hasDiffusersMid = names.contains(where: { $0.hasPrefix("vae.decoder.mid_block.") })
    let hasDiffusersUp = names.contains(where: { $0.hasPrefix("vae.decoder.up_blocks.") })
    if hasDiffusersMid && hasDiffusersUp { return true }

    let hasComfyMid = names.contains(where: { $0.hasPrefix("vae.decoder.mid.attn_1.") || $0.hasPrefix("vae.decoder.mid.block_") })
    let hasComfyUp = names.contains(where: { $0.hasPrefix("vae.decoder.up.") })
    return hasComfyMid && hasComfyUp
  }

  private static func canonicalizeVAEDecoderTensor(for key: String, tensor: MLXArray) -> MLXArray {
    // Some ComfyUI exports store VAE mid attention projections as 1x1 conv weights [C, C, 1, 1].
    // Our VAE attention uses Linear layers which expect [C, C]. Convert safely by squeezing.
    guard key.hasPrefix("decoder.mid_block.attentions.") else { return tensor }
    guard key.hasSuffix(".weight") else { return tensor }

    let isProjectionWeight =
      key.hasSuffix(".to_q.weight")
        || key.hasSuffix(".to_k.weight")
        || key.hasSuffix(".to_v.weight")
        || key.hasSuffix(".to_out.0.weight")
    guard isProjectionWeight else { return tensor }

    guard tensor.ndim == 4 else { return tensor }

    // Handle PyTorch conv layout: [out, in, kH, kW]
    if tensor.dim(2) == 1 && tensor.dim(3) == 1 {
      return tensor.squeezed(axis: 3).squeezed(axis: 2)
    }

    // Handle MLX conv layout: [out, kH, kW, in]
    if tensor.dim(1) == 1 && tensor.dim(2) == 1 {
      return tensor.squeezed(axis: 2).squeezed(axis: 1)
    }

    return tensor
  }

  private static func mapComfyVAEKeyToDiffusers(_ key: String, upBlocks: Int) -> String {
    let parts = key.split(separator: ".", omittingEmptySubsequences: false)
    guard parts.count >= 3 else { return key }
    guard parts[0] == "decoder" else { return key }

    // decoder.norm_out.{weight,bias} -> decoder.conv_norm_out.{weight,bias}
    if parts[1] == "norm_out", parts.count == 3 {
      return ["decoder", "conv_norm_out", String(parts[2])].joined(separator: ".")
    }

    // decoder.mid.* -> decoder.mid_block.*
    if parts[1] == "mid", parts.count >= 4 {
      // decoder.mid.attn_1.{q,k,v,norm,proj_out}.{weight,bias}
      if parts[2] == "attn_1", parts.count == 5 {
        let leaf = parts[3]
        let suffix = String(parts[4])
        switch leaf {
        case "norm":
          return ["decoder", "mid_block", "attentions", "0", "group_norm", suffix].joined(separator: ".")
        case "q":
          return ["decoder", "mid_block", "attentions", "0", "to_q", suffix].joined(separator: ".")
        case "k":
          return ["decoder", "mid_block", "attentions", "0", "to_k", suffix].joined(separator: ".")
        case "v":
          return ["decoder", "mid_block", "attentions", "0", "to_v", suffix].joined(separator: ".")
        case "proj_out":
          return ["decoder", "mid_block", "attentions", "0", "to_out", "0", suffix].joined(separator: ".")
        default:
          return key
        }
      }

      // decoder.mid.block_1.* -> decoder.mid_block.resnets.0.*
      // decoder.mid.block_2.* -> decoder.mid_block.resnets.1.*
      if parts[2] == "block_1" || parts[2] == "block_2", parts.count >= 5 {
        let resnetIndex = parts[2] == "block_1" ? "0" : "1"
        let tail = parts[3...].map(String.init).joined(separator: ".")
        return ["decoder", "mid_block", "resnets", resnetIndex, tail].joined(separator: ".")
      }
    }

    // decoder.up.<src>.block.<j>.* -> decoder.up_blocks.<dst>.resnets.<j>.*
    // decoder.up.<src>.upsample.conv.* -> decoder.up_blocks.<dst>.upsamplers.0.conv.*
    if parts[1] == "up", parts.count >= 6, let srcIndex = Int(parts[2]) {
      let dstIndex = String(max(0, upBlocks - 1 - srcIndex))
      let section = parts[3]
      if section == "upsample", parts.count >= 6, parts[4] == "conv" {
        let tail = parts[5...].map(String.init).joined(separator: ".")
        return ["decoder", "up_blocks", dstIndex, "upsamplers", "0", "conv", tail].joined(separator: ".")
      }

      if section == "block", let resnetIndex = Int(parts[4]) {
        var tailParts = parts[5...].map(String.init)
        if let first = tailParts.first, first == "nin_shortcut" {
          tailParts[0] = "conv_shortcut"
        }
        let tail = tailParts.joined(separator: ".")
        return ["decoder", "up_blocks", dstIndex, "resnets", String(resnetIndex), tail].joined(separator: ".")
      }
    }

    return key
  }
}
