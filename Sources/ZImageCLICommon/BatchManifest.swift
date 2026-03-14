import Foundation

public struct BatchManifest: Codable, Sendable {
  public var version: Int
  public var defaults: BatchDefaults?
  public var jobs: [BatchJobRecord]

  public init(version: Int = 1, defaults: BatchDefaults? = nil, jobs: [BatchJobRecord]) {
    self.version = version
    self.defaults = defaults
    self.jobs = jobs
  }

  public static func load(from path: String) throws -> BatchManifest {
    let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    let data = try Data(contentsOf: url)
    let manifest = try JSONDecoder().decode(BatchManifest.self, from: data)
    guard manifest.version == 1 else {
      throw CLIError(message: "Unsupported batch manifest version: \(manifest.version)", usage: .batch)
    }
    return manifest
  }

  public func submissions() throws -> [BatchSubmission] {
    try jobs.enumerated().map { index, job in
      try job.makeSubmission(defaults: defaults, index: index)
    }
  }
}

public struct BatchDefaults: Codable, Sendable {
  public var model: String?
  public var weightsVariant: String?
  public var negativePrompt: String?
  public var width: Int?
  public var height: Int?
  public var steps: Int?
  public var guidance: Float?
  public var cfgNormalization: Bool?
  public var cfgTruncation: Float?
  public var seed: UInt64?
  public var cacheLimit: Int?
  public var maxSequenceLength: Int?
  public var loraPath: String?
  public var loraFile: String?
  public var loraScale: Float?
  public var enhancePrompt: Bool?
  public var enhanceMaxTokens: Int?
  public var noProgress: Bool?
  public var outputPath: String?
  public var forceTransformerOverrideOnly: Bool?
  public var controlScale: Float?
  public var controlnetWeights: String?
  public var controlnetWeightsFile: String?
  public var controlImage: String?
  public var inpaintImage: String?
  public var maskImage: String?
  public var logControlMemory: Bool?
}

public struct BatchJobRecord: Codable, Sendable {
  public enum Kind: String, Codable, Sendable {
    case text
    case control
  }

  public var id: String?
  public var kind: Kind?
  public var argv: [String]?
  public var prompt: String?
  public var negativePrompt: String?
  public var width: Int?
  public var height: Int?
  public var steps: Int?
  public var guidance: Float?
  public var cfgNormalization: Bool?
  public var cfgTruncation: Float?
  public var seed: UInt64?
  public var outputPath: String?
  public var model: String?
  public var weightsVariant: String?
  public var cacheLimit: Int?
  public var maxSequenceLength: Int?
  public var loraPath: String?
  public var loraFile: String?
  public var loraScale: Float?
  public var enhancePrompt: Bool?
  public var enhanceMaxTokens: Int?
  public var noProgress: Bool?
  public var forceTransformerOverrideOnly: Bool?
  public var controlImage: String?
  public var inpaintImage: String?
  public var maskImage: String?
  public var controlScale: Float?
  public var controlnetWeights: String?
  public var controlnetWeightsFile: String?
  public var logControlMemory: Bool?

  func makeSubmission(defaults: BatchDefaults?, index: Int) throws -> BatchSubmission {
    let identifier = id ?? "job-\(index + 1)"
    if let argv {
      return BatchSubmission(
        jobID: identifier,
        job: try GenerationJobInvocationParser.parse(tokens: argv, usage: .batch)
      )
    }

    guard let kind else {
      throw CLIError(message: "Batch job \(identifier) is missing kind", usage: .batch)
    }

    switch kind {
    case .text:
      guard let prompt = prompt else {
        throw CLIError(message: "Batch text job \(identifier) is missing prompt", usage: .batch)
      }
      return BatchSubmission(
        jobID: identifier,
        job: .text(
          TextGenerationOptions(
            prompt: prompt,
            negativePrompt: negativePrompt ?? defaults?.negativePrompt,
            width: width ?? defaults?.width,
            height: height ?? defaults?.height,
            steps: steps ?? defaults?.steps,
            guidance: guidance ?? defaults?.guidance,
            cfgNormalization: cfgNormalization ?? defaults?.cfgNormalization ?? false,
            cfgTruncation: cfgTruncation ?? defaults?.cfgTruncation ?? 1.0,
            seed: seed ?? defaults?.seed,
            outputPath: outputPath ?? defaults?.outputPath ?? "z-image-\(identifier).png",
            model: model ?? defaults?.model,
            weightsVariant: weightsVariant ?? defaults?.weightsVariant,
            cacheLimit: cacheLimit ?? defaults?.cacheLimit,
            maxSequenceLength: maxSequenceLength ?? defaults?.maxSequenceLength,
            loraPath: loraPath ?? defaults?.loraPath,
            loraFile: loraFile ?? defaults?.loraFile,
            loraScale: loraScale ?? defaults?.loraScale,
            enhancePrompt: enhancePrompt ?? defaults?.enhancePrompt ?? false,
            enhanceMaxTokens: enhanceMaxTokens ?? defaults?.enhanceMaxTokens ?? 512,
            noProgress: noProgress ?? defaults?.noProgress ?? false,
            forceTransformerOverrideOnly: forceTransformerOverrideOnly ?? defaults?.forceTransformerOverrideOnly ?? false
          ))
      )

    case .control:
      guard let prompt = prompt else {
        throw CLIError(message: "Batch control job \(identifier) is missing prompt", usage: .batch)
      }
      guard let controlnetWeights = controlnetWeights ?? defaults?.controlnetWeights else {
        throw CLIError(message: "Batch control job \(identifier) is missing controlnetWeights", usage: .batch)
      }
      return BatchSubmission(
        jobID: identifier,
        job: .control(
          ControlGenerationOptions(
            prompt: prompt,
            negativePrompt: negativePrompt ?? defaults?.negativePrompt,
            controlImage: controlImage ?? defaults?.controlImage,
            inpaintImage: inpaintImage ?? defaults?.inpaintImage,
            maskImage: maskImage ?? defaults?.maskImage,
            controlScale: controlScale ?? defaults?.controlScale ?? 0.75,
            controlnetWeights: controlnetWeights,
            controlnetWeightsFile: controlnetWeightsFile ?? defaults?.controlnetWeightsFile,
            width: width ?? defaults?.width,
            height: height ?? defaults?.height,
            steps: steps ?? defaults?.steps,
            guidance: guidance ?? defaults?.guidance,
            cfgNormalization: cfgNormalization ?? defaults?.cfgNormalization ?? false,
            cfgTruncation: cfgTruncation ?? defaults?.cfgTruncation ?? 1.0,
            seed: seed ?? defaults?.seed,
            outputPath: outputPath ?? defaults?.outputPath ?? "z-image-\(identifier).png",
            model: model ?? defaults?.model,
            weightsVariant: weightsVariant ?? defaults?.weightsVariant,
            cacheLimit: cacheLimit ?? defaults?.cacheLimit,
            maxSequenceLength: maxSequenceLength ?? defaults?.maxSequenceLength,
            loraPath: loraPath ?? defaults?.loraPath,
            loraFile: loraFile ?? defaults?.loraFile,
            loraScale: loraScale ?? defaults?.loraScale,
            enhancePrompt: enhancePrompt ?? defaults?.enhancePrompt ?? false,
            enhanceMaxTokens: enhanceMaxTokens ?? defaults?.enhanceMaxTokens ?? 512,
            logControlMemory: logControlMemory ?? defaults?.logControlMemory ?? false,
            noProgress: noProgress ?? defaults?.noProgress ?? false
          ))
      )
    }
  }
}

public struct BatchSubmission: Sendable, Equatable {
  public var jobID: String
  public var job: GenerationJobPayload

  public init(jobID: String, job: GenerationJobPayload) {
    self.jobID = jobID
    self.job = job
  }
}
