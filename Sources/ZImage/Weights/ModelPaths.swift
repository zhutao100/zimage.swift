import Foundation
import Logging

/// Canonical locations for the Hugging Face snapshot. Kept small and explicit
/// so the rest of the pipeline can assemble download/caching steps.
public enum ZImageRepository {
  public static let id = ZImageKnownModel.zImageTurbo.id
  public static let revision = "main"

  public static func defaultCacheDirectory(base: URL = URL(fileURLWithPath: "models")) -> URL {
    defaultCacheDirectory(for: id, base: base)
  }

  public static func defaultCacheDirectory(for modelId: String, base: URL = URL(fileURLWithPath: "models")) -> URL {
    switch ZImageModelRegistry.normalizedModelId(from: modelId) {
    case ZImageKnownModel.zImage.id:
      return base.appendingPathComponent("z-image")
    case ZImageKnownModel.zImageTurbo.id, ZImageKnownModel.zImageTurbo8bit.id, "mzbac/Z-Image-Turbo-8bit":
      return base.appendingPathComponent("z-image-turbo")
    default:
      return base.appendingPathComponent(sanitizeModelIdForCachePath(modelId))
    }
  }

  private static func sanitizeModelIdForCachePath(_ modelId: String) -> String {
    let normalized = ZImageModelRegistry.normalizedModelId(from: modelId)
    let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))

    let replaced = normalized
      .replacingOccurrences(of: "/", with: "--")
      .unicodeScalars
      .map { allowed.contains($0) ? Character($0) : Character("-") }
    let cleaned = String(replaced).trimmingCharacters(in: CharacterSet(charactersIn: "-"))

    return cleaned.isEmpty ? "model" : cleaned.lowercased()
  }
}

public enum ZImageFiles {
  public static let modelIndex = "model_index.json"
  public static let schedulerConfig = "scheduler/scheduler_config.json"
  public static let transformerConfig = "transformer/config.json"
  // Legacy defaults for current snapshot; dynamic resolvers should be preferred.
  public static let transformerWeights = [
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
  ]
  public static let transformerIndex = "transformer/diffusion_pytorch_model.safetensors.index.json"

  public static let textEncoderConfig = "text_encoder/config.json"
  // Legacy defaults for current snapshot; dynamic resolvers should be preferred.
  public static let textEncoderWeights = [
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
  ]
  public static let textEncoderIndex = "text_encoder/model.safetensors.index.json"

  public static let tokenizerFiles = [
    "tokenizer/merges.txt",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
  ]

  public static let vaeConfig = "vae/config.json"
  public static let vaeWeights = ["vae/diffusion_pytorch_model.safetensors"]

  // MARK: - Dynamic weight resolution

  /// Resolve text encoder shard paths relative to the snapshot root.
  /// Prefers index.json when present, otherwise discovers shards by filename patterns.
  public static func resolveTextEncoderWeights(
    at snapshot: URL,
    weightsVariant: String? = nil,
    logger: Logger? = nil
  ) -> [String] {
    resolveWeights(
      at: snapshot,
      componentDir: "text_encoder",
      indexRelativePath: textEncoderIndex,
      preferredPrefixes: ["model-"],
      singleFileCandidates: ["model.safetensors"],
      weightsVariant: weightsVariant,
      logger: logger
    )
  }

  /// Resolve transformer shard paths relative to the snapshot root.
  /// Prefers index.json when present, otherwise discovers shards by filename patterns.
  public static func resolveTransformerWeights(
    at snapshot: URL,
    weightsVariant: String? = nil,
    logger: Logger? = nil
  ) -> [String] {
    resolveWeights(
      at: snapshot,
      componentDir: "transformer",
      indexRelativePath: transformerIndex,
      preferredPrefixes: ["diffusion_pytorch_model-"],
      singleFileCandidates: ["diffusion_pytorch_model.safetensors"],
      weightsVariant: weightsVariant,
      logger: logger
    )
  }

  /// Resolve VAE weight paths relative to the snapshot root.
  /// VAE is typically a single-file safetensors weight but can also appear in precision variants.
  public static func resolveVAEWeights(
    at snapshot: URL,
    weightsVariant: String? = nil,
    logger: Logger? = nil
  ) -> [String] {
    resolveWeights(
      at: snapshot,
      componentDir: "vae",
      indexRelativePath: "vae/diffusion_pytorch_model.safetensors.index.json",
      preferredPrefixes: ["diffusion_pytorch_model-"],
      singleFileCandidates: ["diffusion_pytorch_model.safetensors"],
      weightsVariant: weightsVariant,
      logger: logger
    )
  }

  public enum WeightsVariantError: Error, LocalizedError, Sendable {
    case missingRequiredComponentWeights(weightsVariant: String, missingComponents: [String], snapshot: URL)

    public var errorDescription: String? {
      switch self {
      case let .missingRequiredComponentWeights(weightsVariant, missingComponents, snapshot):
        let components = missingComponents.sorted().joined(separator: ", ")
        return """
        Requested weightsVariant '\(weightsVariant)' but missing required component weights: \(components).
        Snapshot: \(snapshot.path)
        """
      }
    }
  }

  /// Guardrail: when `weightsVariant` is provided, ensure all required components have matching weights.
  public static func validateRequiredComponentWeights(
    at snapshot: URL,
    weightsVariant: String?,
    logger: Logger? = nil
  ) throws {
    guard let weightsVariant, !weightsVariant.isEmpty else { return }

    var missing: [String] = []
    if resolveTransformerWeights(at: snapshot, weightsVariant: weightsVariant, logger: logger).isEmpty {
      missing.append("transformer")
    }
    if resolveTextEncoderWeights(at: snapshot, weightsVariant: weightsVariant, logger: logger).isEmpty {
      missing.append("text_encoder")
    }
    if resolveVAEWeights(at: snapshot, weightsVariant: weightsVariant, logger: logger).isEmpty {
      missing.append("vae")
    }

    if !missing.isEmpty {
      throw WeightsVariantError.missingRequiredComponentWeights(
        weightsVariant: weightsVariant,
        missingComponents: missing,
        snapshot: snapshot
      )
    }
  }

  // MARK: - Helpers

  private struct SafetensorsIndex: Decodable {
    let weight_map: [String: String]?
  }

  private static func matchesWeightsVariant(filename: String, weightsVariant: String) -> Bool {
    let needle = weightsVariant.lowercased()
    guard !needle.isEmpty else { return false }

    let haystack = filename.lowercased()
    guard haystack.contains(needle) else { return false }

    let hayChars = Array(haystack)
    let needleChars = Array(needle)
    guard needleChars.count <= hayChars.count else { return false }

    func isBoundary(_ c: Character) -> Bool {
      !c.isLetter && !c.isNumber
    }

    // Find any occurrence of `needle` that is bounded by non-alnum chars (or string boundaries).
    for i in 0 ... (hayChars.count - needleChars.count) {
      if Array(hayChars[i ..< i + needleChars.count]) != needleChars { continue }

      let before = i > 0 ? hayChars[i - 1] : nil
      let afterIndex = i + needleChars.count
      let after = afterIndex < hayChars.count ? hayChars[afterIndex] : nil

      let beforeOK = before == nil || isBoundary(before!)
      let afterOK = after == nil || isBoundary(after!)
      if beforeOK && afterOK { return true }
    }

    return false
  }

  private static func shardGroupKey(for fileName: String) -> String {
    guard fileName.hasSuffix(".safetensors") else { return fileName }
    let stem = String(fileName.dropLast(".safetensors".count))

    // Strip "-00001-of-00003" style suffix but preserve any trailing variant (e.g. ".fp16").
    guard let ofRange = stem.range(of: "-of-", options: .backwards) else { return stem }
    guard let lastDash = stem[..<ofRange.lowerBound].lastIndex(of: "-") else { return stem }

    let shardIndex = stem[stem.index(after: lastDash) ..< ofRange.lowerBound]
    guard !shardIndex.isEmpty, shardIndex.allSatisfy({ $0.isNumber }) else { return stem }

    let afterOf = stem[ofRange.upperBound...]
    var countEnd = afterOf.startIndex
    while countEnd < afterOf.endIndex, afterOf[countEnd].isNumber {
      countEnd = afterOf.index(after: countEnd)
    }
    guard countEnd != afterOf.startIndex else { return stem }

    let trailing = afterOf[countEnd...]
    return String(stem[..<lastDash]) + String(trailing)
  }

  private static func selectDeterministicGroup(
    files: [String],
    expectedBaseNames: [String],
    weightsVariant: String?,
    componentDir: String,
    logger: Logger?
  ) -> [String] {
    guard !files.isEmpty else { return [] }

    let filtered: [String]
    if let weightsVariant, !weightsVariant.isEmpty {
      filtered = files.filter {
        matchesWeightsVariant(filename: ($0 as NSString).lastPathComponent, weightsVariant: weightsVariant)
      }
    } else {
      filtered = files
    }

    guard !filtered.isEmpty else { return [] }

    let grouped = Dictionary(grouping: filtered) { file in
      shardGroupKey(for: (file as NSString).lastPathComponent)
    }

    if grouped.count == 1 {
      return grouped.values.first?.sorted(by: shardAwareLess) ?? []
    }

    if let weightsVariant, !weightsVariant.isEmpty {
      // Prefer groups whose key matches the variant token if there are multiple candidates.
      let matchingKeys = grouped.keys.filter { matchesWeightsVariant(filename: $0, weightsVariant: weightsVariant) }
      let keysToConsider = matchingKeys.isEmpty ? Array(grouped.keys) : matchingKeys
      let chosenKey = keysToConsider.sorted().first
      if let chosenKey, let selected = grouped[chosenKey] {
        return selected.sorted(by: shardAwareLess)
      }
      return []
    }

    // No explicit variant: prefer the non-variant base names if present.
    for expected in expectedBaseNames {
      if let selected = grouped[expected] {
        logger?.warning(
          "Multiple weight variants detected under '\(componentDir)', choosing non-variant group '\(expected)'."
        )
        return selected.sorted(by: shardAwareLess)
      }
    }

    let chosenKey = grouped.keys.sorted().first
    if let chosenKey, let selected = grouped[chosenKey] {
      logger?.warning(
        "Multiple weight variants detected under '\(componentDir)', choosing '\(chosenKey)'."
      )
      return selected.sorted(by: shardAwareLess)
    }

    return []
  }

  private static func variantIndexRelativePath(for indexRelativePath: String, weightsVariant: String) -> String? {
    let suffix = ".safetensors.index.json"
    guard indexRelativePath.hasSuffix(suffix) else { return nil }
    let prefix = indexRelativePath.dropLast(suffix.count)
    return "\(prefix).\(weightsVariant)\(suffix)"
  }

  private static func candidateIndexFiles(
    at snapshot: URL,
    componentDir: String,
    indexRelativePath: String,
    weightsVariant: String?
  ) -> [URL] {
    let fm = FileManager.default
    var candidates: [URL] = []

    if let weightsVariant, !weightsVariant.isEmpty {
      if let variantPath = variantIndexRelativePath(for: indexRelativePath, weightsVariant: weightsVariant) {
        let url = snapshot.appending(path: variantPath)
        if fm.fileExists(atPath: url.path) {
          candidates.append(url)
        }
      }

      let dirURL = snapshot.appending(path: componentDir)
      if let contents = try? fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil) {
        let matching = contents.filter { url in
          let name = url.lastPathComponent
          return name.hasSuffix(".safetensors.index.json")
            && matchesWeightsVariant(filename: name, weightsVariant: weightsVariant)
        }
        candidates.append(contentsOf: matching.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }))
      }

      let fallback = snapshot.appending(path: indexRelativePath)
      if fm.fileExists(atPath: fallback.path) {
        candidates.append(fallback)
      }
    } else {
      let preferred = snapshot.appending(path: indexRelativePath)
      if fm.fileExists(atPath: preferred.path) {
        candidates.append(preferred)
      } else {
        let dirURL = snapshot.appending(path: componentDir)
        if let contents = try? fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil) {
          let indexFiles = contents.filter { $0.lastPathComponent.hasSuffix(".safetensors.index.json") }
          if !indexFiles.isEmpty {
            candidates.append(contentsOf: indexFiles.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }))
          }
        }
      }
    }

    // De-duplicate while preserving order
    var seen: Set<String> = []
    var unique: [URL] = []
    unique.reserveCapacity(candidates.count)
    for url in candidates where seen.insert(url.path).inserted {
      unique.append(url)
    }
    return unique
  }

  private static func resolveWeights(
    at snapshot: URL,
    componentDir: String,
    indexRelativePath: String,
    preferredPrefixes: [String],
    singleFileCandidates: [String],
    weightsVariant: String?,
    logger: Logger?
  ) -> [String] {
    let fm = FileManager.default

    // 1) Try reading the safetensors index to enumerate shard files
    let expectedBaseNames = preferredPrefixes.map { prefix in
      prefix.hasSuffix("-") ? String(prefix.dropLast()) : prefix
    }
    let indexCandidates = candidateIndexFiles(
      at: snapshot,
      componentDir: componentDir,
      indexRelativePath: indexRelativePath,
      weightsVariant: weightsVariant
    )
    for indexURL in indexCandidates {
      guard let data = try? Data(contentsOf: indexURL),
            let idx = try? JSONDecoder().decode(SafetensorsIndex.self, from: data),
            let weightMap = idx.weight_map
      else { continue }

      let uniqueFiles = Array(Set(weightMap.values))
      var relative = uniqueFiles
        .map { file in file.contains("/") ? file : "\(componentDir)/\(file)" }
        .filter { fm.fileExists(atPath: snapshot.appending(path: $0).path) }

      if let weightsVariant, !weightsVariant.isEmpty {
        let indexHasVariant = matchesWeightsVariant(filename: indexURL.lastPathComponent, weightsVariant: weightsVariant)
        if !indexHasVariant {
          relative = relative.filter {
            matchesWeightsVariant(filename: ($0 as NSString).lastPathComponent, weightsVariant: weightsVariant)
          }
        }
      }

      let selected = selectDeterministicGroup(
        files: relative,
        expectedBaseNames: expectedBaseNames,
        weightsVariant: weightsVariant,
        componentDir: componentDir,
        logger: logger
      )
      if !selected.isEmpty { return selected }
    }

    // 2) Discover shards via directory scan with preferred filename prefixes
    let dirURL = snapshot.appending(path: componentDir)
    if let contents = try? fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil) {
      let safetensors = contents.filter { $0.pathExtension == "safetensors" }
      // Prefer files matching our expected base names; fall back to any safetensors.
      let preferred = safetensors.filter { url in
        let name = url.lastPathComponent
        return expectedBaseNames.contains { base in
          guard name.count > base.count else { return false }
          guard name.hasPrefix(base) else { return false }
          let nextIndex = name.index(name.startIndex, offsetBy: base.count)
          let next = name[nextIndex]
          return !next.isLetter && !next.isNumber
        }
      }
      let candidates = preferred.isEmpty ? safetensors : preferred

      var relativeAll = candidates
        .map { "\(componentDir)/\($0.lastPathComponent)" }
      let selected = selectDeterministicGroup(
        files: relativeAll,
        expectedBaseNames: expectedBaseNames,
        weightsVariant: weightsVariant,
        componentDir: componentDir,
        logger: logger
      )

      // 3) If no candidates via discovery, try known single-file names explicitly
      if selected.isEmpty, weightsVariant == nil {
        for single in singleFileCandidates {
          let path = "\(componentDir)/\(single)"
          if fm.fileExists(atPath: snapshot.appending(path: path).path) {
            return [path]
          }
        }
      }

      if !selected.isEmpty { return selected }
    }

    // 4) Fallback to legacy lists (may be stale for newer snapshots)
    if weightsVariant != nil { return [] }
    if componentDir == "text_encoder" { return textEncoderWeights }
    if componentDir == "transformer" { return transformerWeights }
    if componentDir == "vae" { return vaeWeights }
    return []
  }

  /// Comparator that sorts shard files numerically when the filename contains
  /// the pattern "-00001-of-000NN.safetensors", otherwise falls back to lexicographic.
  private static func shardAwareLess(_ a: String, _ b: String) -> Bool {
    func shardIndex(_ name: String) -> Int? {
      // Extract the integer between the last '-' before "-of-" and the "-of-" marker.
      guard let ofRange = name.range(of: "-of-") else { return nil }
      if let lastDash = name[..<ofRange.lowerBound].lastIndex(of: "-") {
        let start = name.index(after: lastDash)
        let idxStr = String(name[start ..< ofRange.lowerBound])
        return Int(idxStr)
      }
      return nil
    }
    let ia = shardIndex((a as NSString).lastPathComponent)
    let ib = shardIndex((b as NSString).lastPathComponent)
    switch (ia, ib) {
    case let (xa?, xb?):
      return xa < xb
    case (nil, nil):
      return a.localizedCompare(b) == .orderedAscending
    case (_?, nil):
      // Prefer shard-numbered files to non-numbered ones
      return true
    case (nil, _?):
      return false
    }
  }
}
