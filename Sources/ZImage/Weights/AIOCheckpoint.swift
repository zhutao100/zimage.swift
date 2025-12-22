import Foundation
import Logging
import MLX

enum ZImageAIOCheckpoint {

  struct Inspection: Sendable {
    let isAIO: Bool
    let textEncoderPrefix: String?
    let diagnostics: [String]
  }

  struct Components: Sendable {
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

    var diagnostics: [String] = []
    if !hasDiffusionModel { diagnostics.append("missing transformer tensors (model.diffusion_model.*)") }
    if textEncoderPrefix == nil { diagnostics.append("missing text encoder tensors (text_encoders.*.transformer.model.*)") }
    if !hasVAE { diagnostics.append("missing VAE tensors (vae.*)") }

    if let textEncoderPrefix {
      if !names.contains("\(textEncoderPrefix)model.embed_tokens.weight") {
        diagnostics.append("missing text encoder embed_tokens weight")
      }
    }

    if !names.contains("vae.decoder.conv_in.weight") {
      diagnostics.append("missing VAE decoder tensors (vae.decoder.*)")
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
}

