import Foundation
import MLX

public enum LoRAWeightLoader {
  private static let loraPatterns: [(down: String, up: String)] = [
    (".lora_down.", ".lora_up."),
    (".lora_A.", ".lora_B."),
  ]
  private enum LoKrSuffix: String {
    case w1 = ".lokr_w1"
    case w2 = ".lokr_w2"
    case alpha = ".alpha"
  }

  private static let prefixesToRemove = [
    "base_model.model.",
    "diffusion_model.",
    "lora_unet_",
    "lora_te_",
    "transformer.",
    "text_encoder.",
  ]

  public static func load(from config: LoRAConfiguration) async throws -> LoRAWeights {
    let url = try await resolveSource(config.source)
    return try load(from: url)
  }

  public static func load(from url: URL) throws -> LoRAWeights {
    let fm = FileManager.default
    var isDirectory: ObjCBool = false
    guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
      throw LoRAError.fileNotFound(url.path)
    }

    let safetensorFiles: [URL]
    let configDirectory: URL
    if isDirectory.boolValue {
      safetensorFiles = try findSafetensorFiles(in: url)
      configDirectory = url
    } else {
      safetensorFiles = [url]
      configDirectory = url.deletingLastPathComponent()
    }

    var loraWeights: [String: (down: MLXArray, up: MLXArray)] = [:]
    var lokrW1: [String: MLXArray] = [:]
    var lokrW2: [String: MLXArray] = [:]
    var lokrAlpha: [String: Float] = [:]

    for fileURL in safetensorFiles {
      let partial = try loadSafetensorFile(fileURL)
      for (k, v) in partial.loraPairs {
        loraWeights[k] = v
      }
      for (k, v) in partial.lokrW1 {
        lokrW1[k] = v
      }
      for (k, v) in partial.lokrW2 {
        lokrW2[k] = v
      }
      for (k, v) in partial.lokrAlpha {
        lokrAlpha[k] = v
      }
    }

    var lokrWeights: [String: LoKrWeights] = [:]
    lokrWeights.reserveCapacity(min(lokrW1.count, lokrW2.count))
    for (key, w1) in lokrW1 {
      guard let w2 = lokrW2[key] else { continue }
      lokrWeights[key] = LoKrWeights(w1: w1, w2: w2, alpha: lokrAlpha[key])
    }

    guard !loraWeights.isEmpty || !lokrWeights.isEmpty else {
      throw LoRAError.invalidFormat("No valid LoRA weight pairs found. Expected keys with .lora_down/.lora_up, .lora_A/.lora_B, or LyCORIS LoKr (.lokr_w1/.lokr_w2).")
    }

    let rank = inferRank(from: loraWeights)
    let alpha = loadAlpha(from: configDirectory)

    return LoRAWeights(weights: loraWeights, lokrWeights: lokrWeights, rank: rank, alpha: alpha)
  }

  public static func resolveSource(_ source: LoRASource) async throws -> URL {
    switch source {
    case let .local(url):
      guard FileManager.default.fileExists(atPath: url.path) else {
        throw LoRAError.fileNotFound(url.path)
      }
      return url

    case let .huggingFace(modelId, filename):
      return try await downloadFromHuggingFace(modelId: modelId, filename: filename)
    }
  }

  public static func validate(at url: URL) throws -> LoRAValidationResult {
    guard FileManager.default.fileExists(atPath: url.path) else {
      return .invalid("File not found: \(url.path)")
    }

    do {
      let reader = try SafeTensorsReader(fileURL: url)
      let keys = reader.tensorNames

      let hasDownWeights = keys.contains { key in loraPatterns.contains { key.contains($0.down) } }
      let hasUpWeights = keys.contains { key in loraPatterns.contains { key.contains($0.up) } }
      let hasLoKr = keys.contains { $0.hasSuffix(LoKrSuffix.w1.rawValue) || $0.hasSuffix(LoKrSuffix.w2.rawValue) }

      guard (hasDownWeights && hasUpWeights) || hasLoKr else {
        return .invalid("Not a valid LoRA file: missing lora_down/lora_up weight pairs or lokr_w1/lokr_w2 tensors")
      }

      var targetLayers: [String] = []
      var rank = 0

      for key in keys {
        let matchedDownPattern = loraPatterns.first { key.contains($0.down) }?.down
        guard let downPattern = matchedDownPattern,
              let tensor = try? reader.tensor(named: key) else { continue }

        let layerName = extractBaseKey(key, pattern: downPattern) ?? key
        targetLayers.append(layerName)

        if tensor.ndim == 2 {
          rank = max(rank, min(tensor.dim(0), tensor.dim(1)))
        }
      }

      let estimatedMemoryMB = (rank * 3840 * 2 * 4 * targetLayers.count) / (1024 * 1024)

      return LoRAValidationResult(
        isValid: true,
        rank: rank,
        targetLayers: targetLayers,
        estimatedMemoryMB: estimatedMemoryMB
      )
    } catch {
      return .invalid("Failed to read LoRA file: \(error.localizedDescription)")
    }
  }

  private static func resolveKeyPair(_ key: String) -> (downKey: String, upKey: String, baseKey: String)? {
    for (downPattern, upPattern) in loraPatterns {
      if key.contains(downPattern) {
        guard let base = extractBaseKey(key, pattern: downPattern) else { return nil }
        let upKey = key.replacingOccurrences(of: downPattern, with: upPattern)
        return (key, upKey, base)
      } else if key.contains(upPattern) {
        guard let base = extractBaseKey(key, pattern: upPattern) else { return nil }
        let downKey = key.replacingOccurrences(of: upPattern, with: downPattern)
        return (downKey, key, base)
      }
    }
    return nil
  }

  private static func extractBaseKey(_ key: String, pattern: String) -> String? {
    guard let range = key.range(of: pattern) else { return nil }
    var base = String(key[..<range.lowerBound])

    for prefix in prefixesToRemove {
      if base.hasPrefix(prefix) {
        base = String(base.dropFirst(prefix.count))
        break
      }
    }

    return base
  }

  private static func inferRank(from weights: [String: (down: MLXArray, up: MLXArray)]) -> Int {
    for (_, pair) in weights {
      let downShape = pair.down.shape
      if downShape.count == 2 {
        return min(downShape[0], downShape[1])
      }
    }
    return 16
  }

  private static func loadAlpha(from directory: URL) -> Float? {
    let configPath = directory.appendingPathComponent("adapter_config.json")

    guard FileManager.default.fileExists(atPath: configPath.path),
          let data = try? Data(contentsOf: configPath),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let alpha = json["lora_alpha"] as? NSNumber
    else {
      return nil
    }

    return alpha.floatValue
  }

  private static func downloadFromHuggingFace(modelId: String, filename: String?) async throws -> URL {
    let filePatterns = filename.map { [$0] } ?? ["*.safetensors"]

    do {
      let snapshotURL = try await ModelResolution.resolve(
        modelSpec: modelId,
        filePatterns: filePatterns
      )

      let fm = FileManager.default
      let contents = try fm.contentsOfDirectory(at: snapshotURL, includingPropertiesForKeys: nil)

      if let filename = filename {
        let targetURL = snapshotURL.appendingPathComponent(filename)
        if fm.fileExists(atPath: targetURL.path) {
          return targetURL
        }
      }

      if let safetensorFile = contents.first(where: { $0.pathExtension == "safetensors" }) {
        return safetensorFile
      }

      throw LoRAError.noSafetensorsFound(snapshotURL)
    } catch let error as LoRAError {
      throw error
    } catch {
      throw LoRAError.downloadFailed(modelId, error)
    }
  }

  private struct PartialLoRAWeights {
    let loraPairs: [String: (down: MLXArray, up: MLXArray)]
    let lokrW1: [String: MLXArray]
    let lokrW2: [String: MLXArray]
    let lokrAlpha: [String: Float]
  }

  private static func loadSafetensorFile(_ url: URL) throws -> PartialLoRAWeights {
    let reader = try SafeTensorsReader(fileURL: url)
    let keys = reader.tensorNames

    var processedKeys = Set<String>()
    var loraPairs: [String: (down: MLXArray, up: MLXArray)] = [:]
    var lokrW1: [String: MLXArray] = [:]
    var lokrW2: [String: MLXArray] = [:]
    var lokrAlpha: [String: Float] = [:]

    for key in keys {
      if processedKeys.contains(key) { continue }

      if let (moduleKey, suffix) = mapLoKrModuleKey(key) {
        switch suffix {
        case .w1:
          lokrW1[moduleKey] = try reader.tensor(named: key)
        case .w2:
          lokrW2[moduleKey] = try reader.tensor(named: key)
        case .alpha:
          let tensor = try reader.tensor(named: key)
          if let value = tensor.asArray(Float.self).first {
            lokrAlpha[moduleKey] = value
          }
        }
        continue
      }

      guard let (downKey, upKey, baseKey) = resolveKeyPair(key) else { continue }
      guard reader.contains(downKey), reader.contains(upKey) else { continue }

      let downWeight = try reader.tensor(named: downKey)
      let upWeight = try reader.tensor(named: upKey)

      let mappedKey = LoRAKeyMapper.mapToZImageKey(baseKey)
      loraPairs[mappedKey] = (down: downWeight, up: upWeight)

      processedKeys.insert(downKey)
      processedKeys.insert(upKey)
    }

    return PartialLoRAWeights(
      loraPairs: loraPairs,
      lokrW1: lokrW1,
      lokrW2: lokrW2,
      lokrAlpha: lokrAlpha
    )
  }

  private static func findSafetensorFiles(in directory: URL) throws -> [URL] {
    let fm = FileManager.default
    guard let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey]) else {
      throw LoRAError.noSafetensorsFound(directory)
    }

    var results: [URL] = []
    for case let url as URL in enumerator {
      if url.pathExtension == "safetensors" {
        results.append(url)
      }
    }
    if results.isEmpty {
      throw LoRAError.noSafetensorsFound(directory)
    }
    return results.sorted(by: { $0.path < $1.path })
  }

  private static func mapLoKrModuleKey(_ key: String) -> (moduleKey: String, suffix: LoKrSuffix)? {
    let suffix: LoKrSuffix
    if key.hasSuffix(LoKrSuffix.w1.rawValue) {
      suffix = .w1
    } else if key.hasSuffix(LoKrSuffix.w2.rawValue) {
      suffix = .w2
    } else if key.hasSuffix(LoKrSuffix.alpha.rawValue) {
      suffix = .alpha
    } else {
      return nil
    }

    if key.hasPrefix("lycoris_transformer_blocks_") {
      let prefix = "lycoris_transformer_blocks_"
      let remainder = String(key.dropFirst(prefix.count))
      guard let underscoreIndex = remainder.firstIndex(of: "_") else { return nil }
      let layerStr = String(remainder[..<underscoreIndex])
      guard let layerIdx = Int(layerStr) else { return nil }

      let after = String(remainder[remainder.index(after: underscoreIndex)...])
      if after.hasPrefix("attn_to_q") {
        return ("layers.\(layerIdx).attention.to_q", suffix)
      }
      if after.hasPrefix("attn_to_k") {
        return ("layers.\(layerIdx).attention.to_k", suffix)
      }
      if after.hasPrefix("attn_to_v") {
        return ("layers.\(layerIdx).attention.to_v", suffix)
      }
      if after.hasPrefix("attn_to_out_0") {
        return ("layers.\(layerIdx).attention.to_out.0", suffix)
      }
      return nil
    }

    let base = String(key.dropLast(suffix.rawValue.count))
    let mapped = LoRAKeyMapper.mapToZImageKey(base)
    guard mapped.hasSuffix(".weight") else { return nil }
    return (String(mapped.dropLast(".weight".count)), suffix)
  }
}
