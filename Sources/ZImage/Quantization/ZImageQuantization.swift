import Foundation
import MLX
import MLXNN

public enum ZImageQuantizationMode: String, Codable, Sendable {
  case affine
  case mxfp4

  public var mlxMode: QuantizationMode {
    switch self {
    case .affine: return .affine
    case .mxfp4: return .mxfp4
    }
  }
}

public struct ZImageQuantizationSpec: Codable, Sendable {
  public var groupSize: Int
  public var bits: Int
  public var mode: ZImageQuantizationMode

  public init(groupSize: Int = 32, bits: Int = 8, mode: ZImageQuantizationMode = .affine) {
    self.groupSize = groupSize
    self.bits = bits
    self.mode = mode
  }

  enum CodingKeys: String, CodingKey {
    case groupSize = "group_size"
    case bits
    case mode
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 32
    self.bits = try container.decodeIfPresent(Int.self, forKey: .bits) ?? 8
    let modeStr = try container.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
    self.mode = ZImageQuantizationMode(rawValue: modeStr) ?? .affine
  }
}

public struct ZImageQuantizationManifest: Codable {
  public var modelId: String?
  public var revision: String?
  public var groupSize: Int
  public var bits: Int
  public var mode: String
  public var layers: [QuantizedLayerInfo]

  public struct QuantizedLayerInfo: Codable {
    public var name: String
    public var shape: [Int]
    public var inDim: Int
    public var outDim: Int
    public var file: String
    public var quantFile: String?
    public var groupSize: Int?
    public var bits: Int?
    public var mode: String?

    enum CodingKeys: String, CodingKey {
      case name
      case shape
      case inDim = "in_dim"
      case outDim = "out_dim"
      case file
      case quantFile = "quant_file"
      case groupSize = "group_size"
      case bits
      case mode
    }
  }

  enum CodingKeys: String, CodingKey {
    case modelId = "model_id"
    case revision
    case groupSize = "group_size"
    case bits
    case mode
    case layers
  }

  public static func load(from url: URL) throws -> ZImageQuantizationManifest {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(ZImageQuantizationManifest.self, from: data)
  }
}

public enum ZImageQuantizationError: Error, LocalizedError {
  case noSafetensorsFound(URL)
  case invalidGroupSize(Int)
  case invalidBits(Int)
  case quantizationFailed(String)
  case outputDirectoryCreationFailed(URL)

  public var errorDescription: String? {
    switch self {
    case .noSafetensorsFound(let url):
      return "No safetensors files found in \(url.path)"
    case .invalidGroupSize(let size):
      return "Invalid group size: \(size). Supported sizes: 32, 64, 128"
    case .invalidBits(let bits):
      return "Invalid bits: \(bits). Supported values: 4, 8"
    case .quantizationFailed(let reason):
      return "Quantization failed: \(reason)"
    case .outputDirectoryCreationFailed(let url):
      return "Failed to create output directory: \(url.path)"
    }
  }
}

public struct ZImageQuantizer {
  public static let supportedGroupSizes: Set<Int> = [32, 64, 128]
  public static let supportedBits: Set<Int> = [4, 8]

  public static func quantizeAndSave(
    from sourceURL: URL,
    to outputURL: URL,
    spec: ZImageQuantizationSpec,
    modelId: String? = nil,
    revision: String? = nil,
    verbose: Bool = false
  ) throws {
    guard supportedGroupSizes.contains(spec.groupSize) else {
      throw ZImageQuantizationError.invalidGroupSize(spec.groupSize)
    }
    guard supportedBits.contains(spec.bits) else {
      throw ZImageQuantizationError.invalidBits(spec.bits)
    }

    let fm = FileManager.default

    do {
      try fm.createDirectory(at: outputURL, withIntermediateDirectories: true)
    } catch {
      throw ZImageQuantizationError.outputDirectoryCreationFailed(outputURL)
    }

    let resolvedSourceURL = sourceURL.resolvingSymlinksInPath()

    var allQuantizedLayers: [ZImageQuantizationManifest.QuantizedLayerInfo] = []
    var totalQuantizedCount = 0

    let components = ["transformer", "text_encoder"]
    for component in components {
      let componentSourceDir = resolvedSourceURL.appendingPathComponent(component)
      let componentOutputDir = outputURL.appendingPathComponent(component)

      guard fm.fileExists(atPath: componentSourceDir.path) else {
        continue
      }

      try fm.createDirectory(at: componentOutputDir, withIntermediateDirectories: true)

      let contents = try fm.contentsOfDirectory(at: componentSourceDir, includingPropertiesForKeys: nil)
      let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

      if safetensorFiles.isEmpty {
        continue
      }

      var componentWeights: [String: MLXArray] = [:]
      for file in safetensorFiles {
        let weights = try MLX.loadArrays(url: file)
        for (key, value) in weights {
          componentWeights[key] = value
        }
      }

      var quantizedWeights: [String: MLXArray] = [:]
      var quantizedLayers: [ZImageQuantizationManifest.QuantizedLayerInfo] = []
      var quantizedCount = 0

      for (key, tensor) in componentWeights {
        let isEmbedding = key.contains("embed") || key.contains("embedding")
        let isNorm = key.contains("norm") || key.contains("layernorm")
        let isDictionaryModule = key.hasPrefix("all_x_embedder") || key.hasPrefix("all_final_layer")

        if key.hasSuffix(".weight") && tensor.ndim == 2 && !isEmbedding && !isNorm && !isDictionaryModule {
          let outDim = tensor.dim(0)
          let inDim = tensor.dim(1)

          if inDim % spec.groupSize == 0 {
            let base = String(key.dropLast(".weight".count))

            var f = tensor
            if f.dtype != .float32 {
              f = f.asType(.float32)
            }

            let (wq, scales, biases) = MLX.quantized(
              f,
              groupSize: spec.groupSize,
              bits: spec.bits,
              mode: spec.mode.mlxMode
            )

            quantizedWeights[key] = wq
            quantizedWeights["\(base).scales"] = scales
            if let b = biases {
              quantizedWeights["\(base).biases"] = b
            }

            quantizedLayers.append(.init(
              name: "\(component).\(base)",
              shape: [outDim, inDim],
              inDim: inDim,
              outDim: outDim,
              file: "",
              quantFile: nil,
              groupSize: spec.groupSize,
              bits: spec.bits,
              mode: spec.mode.rawValue
            ))

            quantizedCount += 1
          } else {
            quantizedWeights[key] = tensor
          }
        } else {
          quantizedWeights[key] = tensor
        }
      }

      totalQuantizedCount += quantizedCount
      allQuantizedLayers.append(contentsOf: quantizedLayers)

      try saveComponentWeights(
        quantizedWeights,
        to: componentOutputDir,
        component: component,
        layers: &quantizedLayers,
        verbose: verbose
      )

      let configURL = componentSourceDir.appendingPathComponent("config.json")
      if let data = try? Data(contentsOf: configURL) {
        let destConfig = componentOutputDir.appendingPathComponent("config.json")
        try data.write(to: destConfig)
      }
    }

    let vaeSourceDir = resolvedSourceURL.appendingPathComponent("vae")
    let vaeOutputDir = outputURL.appendingPathComponent("vae")
    if fm.fileExists(atPath: vaeSourceDir.path) {
      try fm.createDirectory(at: vaeOutputDir, withIntermediateDirectories: true)
      let vaeContents = try fm.contentsOfDirectory(at: vaeSourceDir, includingPropertiesForKeys: nil)
      for file in vaeContents {
        let destFile = vaeOutputDir.appendingPathComponent(file.lastPathComponent)
        if let data = try? Data(contentsOf: file) {
          try data.write(to: destFile)
        }
      }
    }

    let manifest = ZImageQuantizationManifest(
      modelId: modelId,
      revision: revision,
      groupSize: spec.groupSize,
      bits: spec.bits,
      mode: spec.mode.rawValue,
      layers: allQuantizedLayers
    )

    let manifestURL = outputURL.appendingPathComponent("quantization.json")
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let manifestData = try encoder.encode(manifest)
    try manifestData.write(to: manifestURL)

    try copyAncillaryFiles(from: resolvedSourceURL, to: outputURL, verbose: verbose)
  }

  private static func saveComponentWeights(
    _ weights: [String: MLXArray],
    to outputDir: URL,
    component: String,
    layers: inout [ZImageQuantizationManifest.QuantizedLayerInfo],
    verbose: Bool
  ) throws {
    let maxShardBytes = 4_500_000_000

    func bytes(of array: MLXArray) -> Int {
      array.shape.reduce(1, *) * array.dtype.size
    }

    let ordered = weights.keys.sorted().map { ($0, weights[$0]!) }
    var chunks: [[(String, MLXArray)]] = []
    var current: [(String, MLXArray)] = []
    var currentBytes = 0

    for (k, v) in ordered {
      let sz = bytes(of: v)
      if currentBytes > 0 && currentBytes + sz > maxShardBytes {
        chunks.append(current)
        current = []
        currentBytes = 0
      }
      current.append((k, v))
      currentBytes += sz
    }
    if !current.isEmpty {
      chunks.append(current)
    }

    let total = max(1, chunks.count)

    for (i, chunk) in chunks.enumerated() {
      let shardName: String
      if total == 1 {
        shardName = "model.safetensors"
      } else {
        shardName = String(format: "model-%05d-of-%05d.safetensors", i + 1, total)
      }
      let dstURL = outputDir.appendingPathComponent(shardName)

      var dict: [String: MLXArray] = [:]
      for (k, v) in chunk {
        dict[k] = v
      }

      try MLX.save(arrays: dict, metadata: [:], url: dstURL)

      let keys = Set(chunk.map { $0.0 })
      for i in 0..<layers.count {
        let layerBase = layers[i].name.hasPrefix("\(component).")
          ? String(layers[i].name.dropFirst("\(component).".count))
          : layers[i].name
        if keys.contains("\(layerBase).weight") {
          layers[i].file = "\(component)/\(shardName)"
          layers[i].quantFile = "\(component)/\(shardName)"
        }
      }
    }
  }

  private static func copyAncillaryFiles(
    from sourceURL: URL,
    to outputURL: URL,
    verbose: Bool
  ) throws {
    let fm = FileManager.default
    let contents = try fm.contentsOfDirectory(at: sourceURL, includingPropertiesForKeys: nil)

    let copyExtensions: Set<String> = ["json", "txt", "md"]
    let skipFiles: Set<String> = ["quantization.json"]
    let weightDirs: Set<String> = ["transformer", "text_encoder", "vae"]
    let copyDirs: Set<String> = ["tokenizer", "assets"]

    for file in contents {
      let name = file.lastPathComponent
      let ext = file.pathExtension.lowercased()

      if ext == "safetensors" {
        continue
      }

      if skipFiles.contains(name) {
        continue
      }

      if file.hasDirectoryPath && weightDirs.contains(name) {
        continue
      }

      if file.hasDirectoryPath {
        if copyDirs.contains(name) {
          let destDir = outputURL.appendingPathComponent(name)
          try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
          let subContents = try fm.contentsOfDirectory(at: file, includingPropertiesForKeys: nil)
          for subFile in subContents {
            let subName = subFile.lastPathComponent
            if let data = try? Data(contentsOf: subFile) {
              let destFile = destDir.appendingPathComponent(subName)
              try data.write(to: destFile)
            } else if subFile.hasDirectoryPath {
              let destSubDir = destDir.appendingPathComponent(subName)
              try? fm.removeItem(at: destSubDir)
              try fm.copyItem(at: subFile, to: destSubDir)
            }
          }
        } else if name == "scheduler" {
          let destDir = outputURL.appendingPathComponent(name)
          try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
          let schedulerConfig = file.appendingPathComponent("scheduler_config.json")
          if let data = try? Data(contentsOf: schedulerConfig) {
            let destConfig = destDir.appendingPathComponent("scheduler_config.json")
            try data.write(to: destConfig)
          }
        }
      } else if copyExtensions.contains(ext) {
        if let data = try? Data(contentsOf: file) {
          let destURL = outputURL.appendingPathComponent(name)
          try data.write(to: destURL)
        }
      }
    }
  }

  public static func hasQuantization(at directory: URL) -> Bool {
    let manifestURL = directory.appendingPathComponent("quantization.json")
    return FileManager.default.fileExists(atPath: manifestURL.path)
  }

  public static func applyQuantization(
    to model: Module,
    manifest: ZImageQuantizationManifest,
    availableKeys: Set<String>,
    tensorNameTransform: (String) -> String
  ) {
    let defaultSpec = (manifest.groupSize, manifest.bits, manifest.mode)

    if let transformer = model as? ZImageTransformer2DModel {
      for (i, block) in transformer.layers.enumerated() {
        quantizeBlock(block, prefix: "layers.\(i)", availableKeys: availableKeys,
                      defaultSpec: defaultSpec, manifest: manifest, tensorNameTransform: tensorNameTransform)
      }
      for (i, block) in transformer.noiseRefiner.enumerated() {
        quantizeBlock(block, prefix: "noise_refiner.\(i)", availableKeys: availableKeys,
                      defaultSpec: defaultSpec, manifest: manifest, tensorNameTransform: tensorNameTransform)
      }
      for (i, block) in transformer.contextRefiner.enumerated() {
        quantizeBlock(block, prefix: "context_refiner.\(i)", availableKeys: availableKeys,
                      defaultSpec: defaultSpec, manifest: manifest, tensorNameTransform: tensorNameTransform)
      }
      return
    }

    if let controlTransformer = model as? ZImageControlTransformer2DModel {
      for (i, block) in controlTransformer.layers.enumerated() {
        quantizeBlock(block, prefix: "layers.\(i)", availableKeys: availableKeys,
                      defaultSpec: defaultSpec, manifest: manifest, tensorNameTransform: tensorNameTransform)
      }
      for (i, block) in controlTransformer.noiseRefiner.enumerated() {
        quantizeBlock(block, prefix: "noise_refiner.\(i)", availableKeys: availableKeys,
                      defaultSpec: defaultSpec, manifest: manifest, tensorNameTransform: tensorNameTransform)
      }
      for (i, block) in controlTransformer.contextRefiner.enumerated() {
        quantizeBlock(block, prefix: "context_refiner.\(i)", availableKeys: availableKeys,
                      defaultSpec: defaultSpec, manifest: manifest, tensorNameTransform: tensorNameTransform)
      }
      return
    }

    MLXNN.quantize(model: model) { path, _ in
      let tensorName = tensorNameTransform(path)
      let scalesKey = "\(tensorName).scales"
      guard availableKeys.contains(scalesKey) else { return nil }

      let (groupSize, bits, modeStr) =
        manifest.layers.first(where: { $0.name == tensorName }).map {
          ($0.groupSize ?? defaultSpec.0, $0.bits ?? defaultSpec.1, $0.mode ?? defaultSpec.2)
        } ?? defaultSpec
      let mode: QuantizationMode = modeStr == "mxfp4" ? .mxfp4 : .affine
      return (groupSize, bits, mode)
    }
  }

  private static func quantizeBlock(
    _ block: Module,
    prefix: String,
    availableKeys: Set<String>,
    defaultSpec: (Int, Int, String),
    manifest: ZImageQuantizationManifest,
    tensorNameTransform: (String) -> String
  ) {
    MLXNN.quantize(model: block) { path, _ in
      let fullPath = "\(prefix).\(path)"
      let tensorName = tensorNameTransform(fullPath)
      let scalesKey = "\(tensorName).scales"
      guard availableKeys.contains(scalesKey) else { return nil }

      let (groupSize, bits, modeStr) =
        manifest.layers.first(where: { $0.name == tensorName }).map {
          ($0.groupSize ?? defaultSpec.0, $0.bits ?? defaultSpec.1, $0.mode ?? defaultSpec.2)
        } ?? defaultSpec
      let mode: QuantizationMode = modeStr == "mxfp4" ? .mxfp4 : .affine
      return (groupSize, bits, mode)
    }
  }

  public static func transformerTensorName(_ path: String) -> String {
    path
  }

  public static func textEncoderTensorName(_ path: String) -> String {
    if path.hasPrefix("encoder.") {
      let suffix = path.dropFirst("encoder.".count)
      return "model." + suffix
    }
    return path
  }

  private static func findSafetensorFiles(in directory: URL) -> [URL] {
    let fm = FileManager.default
    var safetensorFiles: [URL] = []

    if let contents = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
      for item in contents {
        if item.pathExtension == "safetensors" {
          safetensorFiles.append(item)
        }
      }
    }

    let subdirs = ["transformer", "text_encoder", "vae"]
    for subdir in subdirs {
      let subdirURL = directory.appendingPathComponent(subdir)
      if let contents = try? fm.contentsOfDirectory(at: subdirURL, includingPropertiesForKeys: nil) {
        for item in contents {
          if item.pathExtension == "safetensors" {
            safetensorFiles.append(item)
          }
        }
      }
    }

    return safetensorFiles
  }
  public static func quantizeControlnet(
    from sourceURL: URL,
    to outputURL: URL,
    spec: ZImageQuantizationSpec,
    specificFile: String? = nil,
    verbose: Bool = false
  ) throws {
    guard supportedGroupSizes.contains(spec.groupSize) else {
      throw ZImageQuantizationError.invalidGroupSize(spec.groupSize)
    }
    guard supportedBits.contains(spec.bits) else {
      throw ZImageQuantizationError.invalidBits(spec.bits)
    }

    let fm = FileManager.default
    let resolvedSourceURL = sourceURL.resolvingSymlinksInPath()
    do {
      try fm.createDirectory(at: outputURL, withIntermediateDirectories: true)
    } catch {
      throw ZImageQuantizationError.outputDirectoryCreationFailed(outputURL)
    }
    var safetensorFiles: [URL] = []
    var isDirectory: ObjCBool = false

    if fm.fileExists(atPath: resolvedSourceURL.path, isDirectory: &isDirectory) {
      if isDirectory.boolValue {

        if let specific = specificFile {

          let specificURL = resolvedSourceURL.appendingPathComponent(specific)
          if fm.fileExists(atPath: specificURL.path) {
            safetensorFiles = [specificURL]
            if verbose {
              print("Using specific file: \(specific)")
            }
          } else {
            throw ZImageQuantizationError.noSafetensorsFound(specificURL)
          }
        } else {

          let contents = try fm.contentsOfDirectory(at: resolvedSourceURL, includingPropertiesForKeys: nil)
          safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }
          if safetensorFiles.count > 1 && verbose {
            print("WARNING: Found \(safetensorFiles.count) .safetensors files. Consider using --file to specify one.")
            for file in safetensorFiles {
              print("  - \(file.lastPathComponent)")
            }
          }
        }
      } else if resolvedSourceURL.pathExtension == "safetensors" {

        safetensorFiles = [resolvedSourceURL]
      }
    }

    guard !safetensorFiles.isEmpty else {
      throw ZImageQuantizationError.noSafetensorsFound(resolvedSourceURL)
    }

    if verbose {
      print("Quantizing \(safetensorFiles.count) safetensors file(s)")
    }
    var allWeights: [String: MLXArray] = [:]
    for file in safetensorFiles {
      let weights = try MLX.loadArrays(url: file)
      for (key, value) in weights {
        allWeights[key] = value
      }
    }

    if verbose {
      print("Loaded \(allWeights.count) tensors")
    }
    var quantizedWeights: [String: MLXArray] = [:]
    var quantizedLayers: [ZImageQuantizationManifest.QuantizedLayerInfo] = []
    var quantizedCount = 0
    var skippedCount = 0

    for (key, tensor) in allWeights {

      let isEmbedding = key.contains("embed") || key.contains("embedding")
      let isNorm = key.contains("norm") || key.contains("layernorm")
      if key.hasSuffix(".weight") && tensor.ndim == 2 && !isEmbedding && !isNorm {
        let outDim = tensor.dim(0)
        let inDim = tensor.dim(1)
        if inDim % spec.groupSize == 0 {
          let base = String(key.dropLast(".weight".count))

          var f = tensor
          if f.dtype != .float32 {
            f = f.asType(.float32)
          }

          let (wq, scales, biases) = MLX.quantized(
            f,
            groupSize: spec.groupSize,
            bits: spec.bits,
            mode: spec.mode.mlxMode
          )

          quantizedWeights[key] = wq
          quantizedWeights["\(base).scales"] = scales
          if let b = biases {
            quantizedWeights["\(base).biases"] = b
          }

          quantizedLayers.append(.init(
            name: base,
            shape: [outDim, inDim],
            inDim: inDim,
            outDim: outDim,
            file: "model.safetensors",
            quantFile: "model.safetensors",
            groupSize: spec.groupSize,
            bits: spec.bits,
            mode: spec.mode.rawValue
          ))

          quantizedCount += 1

          if verbose {
            print("  Quantized: \(key) [\(outDim)x\(inDim)]")
          }
        } else {

          quantizedWeights[key] = tensor
          skippedCount += 1
          if verbose {
            print("  Skipped (incompatible dims): \(key)")
          }
        }
      } else {

        quantizedWeights[key] = tensor
        if !key.hasSuffix(".weight") {

        } else {
          skippedCount += 1
          if verbose {
            print("  Skipped: \(key)")
          }
        }
      }
    }

    if verbose {
      print("Quantized \(quantizedCount) layers, skipped \(skippedCount)")
    }
    let outputFile = outputURL.appendingPathComponent("model.safetensors")
    try MLX.save(arrays: quantizedWeights, metadata: [:], url: outputFile)

    if verbose {
      print("Saved quantized weights to \(outputFile.path)")
    }
    let manifest = ZImageQuantizationManifest(
      modelId: nil,
      revision: nil,
      groupSize: spec.groupSize,
      bits: spec.bits,
      mode: spec.mode.rawValue,
      layers: quantizedLayers
    )

    let manifestURL = outputURL.appendingPathComponent("controlnet_quantization.json")
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let manifestData = try encoder.encode(manifest)
    try manifestData.write(to: manifestURL)

    if verbose {
      print("Saved manifest to \(manifestURL.path)")
      print("ControlNet quantization complete!")
    }
  }
  public static func hasControlnetQuantization(at directory: URL) -> Bool {
    let manifestURL = directory.appendingPathComponent("controlnet_quantization.json")
    return FileManager.default.fileExists(atPath: manifestURL.path)
  }
  public static func loadControlnetManifest(from directory: URL) throws -> ZImageQuantizationManifest? {
    let manifestURL = directory.appendingPathComponent("controlnet_quantization.json")
    guard FileManager.default.fileExists(atPath: manifestURL.path) else {
      return nil
    }
    return try ZImageQuantizationManifest.load(from: manifestURL)
  }
  public static func controlnetTensorName(_ path: String) -> String {
    var result = path

    result = result.replacingOccurrences(of: "controlNoiseRefiner", with: "control_noise_refiner")
    result = result.replacingOccurrences(of: "controlLayers", with: "control_layers")
    result = result.replacingOccurrences(of: "controlAllXEmbedder", with: "control_all_x_embedder")
    result = result.replacingOccurrences(of: "feedForward", with: "feed_forward")
    result = result.replacingOccurrences(of: "toQ", with: "to_q")
    result = result.replacingOccurrences(of: "toK", with: "to_k")
    result = result.replacingOccurrences(of: "toV", with: "to_v")
    result = result.replacingOccurrences(of: "toOut", with: "to_out")
    result = result.replacingOccurrences(of: "beforeProj", with: "before_proj")
    result = result.replacingOccurrences(of: "afterProj", with: "after_proj")
    return result
  }
  public static func applyControlnetQuantization(
    to transformer: ZImageControlTransformer2DModel,
    manifest: ZImageQuantizationManifest,
    availableKeys: Set<String>
  ) {
    let defaultSpec = (manifest.groupSize, manifest.bits, manifest.mode)
    for (i, block) in transformer.controlNoiseRefiner.enumerated() {
      let prefix = "control_noise_refiner.\(i)"
      quantizeBlock(block, prefix: prefix, availableKeys: availableKeys,
                    defaultSpec: defaultSpec, manifest: manifest, tensorNameTransform: controlnetTensorName)
    }
    for (i, block) in transformer.controlLayers.enumerated() {
      let prefix = "control_layers.\(i)"
      quantizeControlBlock(block, prefix: prefix, availableKeys: availableKeys,
                           defaultSpec: defaultSpec, manifest: manifest)
    }
  }

  private static func quantizeControlBlock(
    _ block: ZImageControlTransformerBlock,
    prefix: String,
    availableKeys: Set<String>,
    defaultSpec: (Int, Int, String),
    manifest: ZImageQuantizationManifest
  ) {
    MLXNN.quantize(model: block) { path, _ in
      let tensorName = controlnetTensorName("\(prefix).\(path)")
      let scalesKey = "\(tensorName).scales"
      guard availableKeys.contains(scalesKey) else { return nil }

      let (groupSize, bits, modeStr) =
        manifest.layers.first(where: { $0.name == tensorName }).map {
          ($0.groupSize ?? defaultSpec.0, $0.bits ?? defaultSpec.1, $0.mode ?? defaultSpec.2)
        } ?? defaultSpec
      let mode: QuantizationMode = modeStr == "mxfp4" ? .mxfp4 : .affine
      return (groupSize, bits, mode)
    }
  }
}
