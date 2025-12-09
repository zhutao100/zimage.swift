import Foundation

/// Canonical locations for the Hugging Face snapshot. Kept small and explicit
/// so the rest of the pipeline can assemble download/caching steps.
public enum ZImageRepository {
  public static let id = "Tongyi-MAI/Z-Image-Turbo"
  public static let revision = "main"

  public static func defaultCacheDirectory(base: URL = URL(fileURLWithPath: "models")) -> URL {
    base.appendingPathComponent("z-image-turbo")
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
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
  ]
  public static let transformerIndex = "transformer/diffusion_pytorch_model.safetensors.index.json"

  public static let textEncoderConfig = "text_encoder/config.json"
  // Legacy defaults for current snapshot; dynamic resolvers should be preferred.
  public static let textEncoderWeights = [
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors"
  ]
  public static let textEncoderIndex = "text_encoder/model.safetensors.index.json"

  public static let tokenizerFiles = [
    "tokenizer/merges.txt",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json"
  ]

  public static let vaeConfig = "vae/config.json"
  public static let vaeWeights = ["vae/diffusion_pytorch_model.safetensors"]

  // MARK: - Dynamic weight resolution

  /// Resolve text encoder shard paths relative to the snapshot root.
  /// Prefers index.json when present, otherwise discovers shards by filename patterns.
  public static func resolveTextEncoderWeights(at snapshot: URL) -> [String] {
    resolveWeights(
      at: snapshot,
      componentDir: "text_encoder",
      indexRelativePath: textEncoderIndex,
      preferredPrefixes: ["model-"],
      singleFileCandidates: ["model.safetensors"]
    )
  }

  /// Resolve transformer shard paths relative to the snapshot root.
  /// Prefers index.json when present, otherwise discovers shards by filename patterns.
  public static func resolveTransformerWeights(at snapshot: URL) -> [String] {
    resolveWeights(
      at: snapshot,
      componentDir: "transformer",
      indexRelativePath: transformerIndex,
      preferredPrefixes: ["diffusion_pytorch_model-"],
      singleFileCandidates: ["diffusion_pytorch_model.safetensors"]
    )
  }

  // MARK: - Helpers

  private struct SafetensorsIndex: Decodable {
    let weight_map: [String: String]?
  }

  private static func resolveWeights(
    at snapshot: URL,
    componentDir: String,
    indexRelativePath: String,
    preferredPrefixes: [String],
    singleFileCandidates: [String]
  ) -> [String] {
    let fm = FileManager.default

    // 1) Try reading the safetensors index to enumerate shard files
    let indexURL = snapshot.appending(path: indexRelativePath)
    if fm.fileExists(atPath: indexURL.path),
       let data = try? Data(contentsOf: indexURL),
       let idx = try? JSONDecoder().decode(SafetensorsIndex.self, from: data),
       let weightMap = idx.weight_map {
      let uniqueFiles = Array(Set(weightMap.values))
      let relative = uniqueFiles
        .map { file in file.contains("/") ? file : "\(componentDir)/\(file)" }
        .sorted(by: shardAwareLess)
        .filter { fm.fileExists(atPath: snapshot.appending(path: $0).path) }
      if !relative.isEmpty { return relative }
    }

    // 2) Discover shards via directory scan with preferred filename prefixes
    let dirURL = snapshot.appending(path: componentDir)
    if let contents = try? fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil) {
      let safetensors = contents.filter { $0.pathExtension == "safetensors" }
      // Prefer files matching our expected prefixes; fall back to any safetensors
      let preferred = safetensors.filter { url in
        let name = url.lastPathComponent
        return preferredPrefixes.contains(where: { name.hasPrefix($0) })
      }
      let candidates = preferred.isEmpty ? safetensors : preferred

      var relative = candidates
        .map { "\(componentDir)/\($0.lastPathComponent)" }
        .sorted(by: shardAwareLess)

      // 3) If no candidates via prefixes, try known single-file names explicitly
      if relative.isEmpty {
        for single in singleFileCandidates {
          let path = "\(componentDir)/\(single)"
          if fm.fileExists(atPath: snapshot.appending(path: path).path) {
            relative = [path]
            break
          }
        }
      }

      if !relative.isEmpty { return relative }
    }

    // 4) Fallback to legacy lists (may be stale for newer snapshots)
    if componentDir == "text_encoder" { return textEncoderWeights }
    if componentDir == "transformer" { return transformerWeights }
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
        let idxStr = String(name[start..<ofRange.lowerBound])
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
