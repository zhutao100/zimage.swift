import Foundation

public enum CLIProgramKind: String, Sendable, Equatable {
  case cli = "ZImageCLI"
  case serve = "ZImageServe"

  public var executableName: String { rawValue }
}

public enum CLIUsageTopic: Sendable, Equatable {
  case main
  case quantize
  case quantizeControlnet
  case control
  case serve
}

public struct CLIError: LocalizedError, Sendable {
  public let message: String
  public let usage: CLIUsageTopic?

  public init(message: String, usage: CLIUsageTopic? = nil) {
    self.message = message
    self.usage = usage
  }

  public var errorDescription: String? { message }
}

public struct TextGenerationOptions: Codable, Sendable, Equatable {
  public var prompt: String
  public var negativePrompt: String?
  public var width: Int?
  public var height: Int?
  public var steps: Int?
  public var guidance: Float?
  public var cfgNormalization: Bool
  public var cfgTruncation: Float
  public var seed: UInt64?
  public var outputPath: String
  public var model: String?
  public var weightsVariant: String?
  public var cacheLimit: Int?
  public var maxSequenceLength: Int?
  public var loraPath: String?
  public var loraScale: Float
  public var enhancePrompt: Bool
  public var enhanceMaxTokens: Int
  public var noProgress: Bool
  public var forceTransformerOverrideOnly: Bool

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    width: Int? = nil,
    height: Int? = nil,
    steps: Int? = nil,
    guidance: Float? = nil,
    cfgNormalization: Bool = false,
    cfgTruncation: Float = 1.0,
    seed: UInt64? = nil,
    outputPath: String = "z-image.png",
    model: String? = nil,
    weightsVariant: String? = nil,
    cacheLimit: Int? = nil,
    maxSequenceLength: Int? = nil,
    loraPath: String? = nil,
    loraScale: Float = 1.0,
    enhancePrompt: Bool = false,
    enhanceMaxTokens: Int = 512,
    noProgress: Bool = false,
    forceTransformerOverrideOnly: Bool = false
  ) {
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.width = width
    self.height = height
    self.steps = steps
    self.guidance = guidance
    self.cfgNormalization = cfgNormalization
    self.cfgTruncation = cfgTruncation
    self.seed = seed
    self.outputPath = outputPath
    self.model = model
    self.weightsVariant = weightsVariant
    self.cacheLimit = cacheLimit
    self.maxSequenceLength = maxSequenceLength
    self.loraPath = loraPath
    self.loraScale = loraScale
    self.enhancePrompt = enhancePrompt
    self.enhanceMaxTokens = enhanceMaxTokens
    self.noProgress = noProgress
    self.forceTransformerOverrideOnly = forceTransformerOverrideOnly
  }
}

public struct ControlGenerationOptions: Codable, Sendable, Equatable {
  public var prompt: String
  public var negativePrompt: String?
  public var controlImage: String?
  public var inpaintImage: String?
  public var maskImage: String?
  public var controlScale: Float
  public var controlnetWeights: String
  public var controlnetWeightsFile: String?
  public var width: Int?
  public var height: Int?
  public var steps: Int?
  public var guidance: Float?
  public var cfgNormalization: Bool
  public var cfgTruncation: Float
  public var seed: UInt64?
  public var outputPath: String
  public var model: String?
  public var weightsVariant: String?
  public var cacheLimit: Int?
  public var maxSequenceLength: Int?
  public var loraPath: String?
  public var loraScale: Float
  public var enhancePrompt: Bool
  public var enhanceMaxTokens: Int
  public var logControlMemory: Bool
  public var noProgress: Bool

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    controlImage: String? = nil,
    inpaintImage: String? = nil,
    maskImage: String? = nil,
    controlScale: Float = 0.75,
    controlnetWeights: String,
    controlnetWeightsFile: String? = nil,
    width: Int? = nil,
    height: Int? = nil,
    steps: Int? = nil,
    guidance: Float? = nil,
    cfgNormalization: Bool = false,
    cfgTruncation: Float = 1.0,
    seed: UInt64? = nil,
    outputPath: String = "z-image-control.png",
    model: String? = nil,
    weightsVariant: String? = nil,
    cacheLimit: Int? = nil,
    maxSequenceLength: Int? = nil,
    loraPath: String? = nil,
    loraScale: Float = 1.0,
    enhancePrompt: Bool = false,
    enhanceMaxTokens: Int = 512,
    logControlMemory: Bool = false,
    noProgress: Bool = false
  ) {
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.controlImage = controlImage
    self.inpaintImage = inpaintImage
    self.maskImage = maskImage
    self.controlScale = controlScale
    self.controlnetWeights = controlnetWeights
    self.controlnetWeightsFile = controlnetWeightsFile
    self.width = width
    self.height = height
    self.steps = steps
    self.guidance = guidance
    self.cfgNormalization = cfgNormalization
    self.cfgTruncation = cfgTruncation
    self.seed = seed
    self.outputPath = outputPath
    self.model = model
    self.weightsVariant = weightsVariant
    self.cacheLimit = cacheLimit
    self.maxSequenceLength = maxSequenceLength
    self.loraPath = loraPath
    self.loraScale = loraScale
    self.enhancePrompt = enhancePrompt
    self.enhanceMaxTokens = enhanceMaxTokens
    self.logControlMemory = logControlMemory
    self.noProgress = noProgress
  }
}

public struct QuantizeOptions: Sendable, Equatable {
  public var input: String
  public var output: String
  public var bits: Int
  public var groupSize: Int
  public var verbose: Bool

  public init(input: String, output: String, bits: Int = 8, groupSize: Int = 32, verbose: Bool = false) {
    self.input = input
    self.output = output
    self.bits = bits
    self.groupSize = groupSize
    self.verbose = verbose
  }
}

public struct QuantizeControlnetOptions: Sendable, Equatable {
  public var input: String
  public var output: String
  public var specificFile: String?
  public var bits: Int
  public var groupSize: Int
  public var verbose: Bool

  public init(
    input: String,
    output: String,
    specificFile: String? = nil,
    bits: Int = 8,
    groupSize: Int = 32,
    verbose: Bool = false
  ) {
    self.input = input
    self.output = output
    self.specificFile = specificFile
    self.bits = bits
    self.groupSize = groupSize
    self.verbose = verbose
  }
}

public struct ServeOptions: Sendable, Equatable {
  public var socketPath: String?

  public init(socketPath: String? = nil) {
    self.socketPath = socketPath
  }
}

public enum GenerationJobPayload: Sendable, Equatable {
  case text(TextGenerationOptions)
  case control(ControlGenerationOptions)
}

public enum CLIParsedCommand: Sendable, Equatable {
  case help(CLIUsageTopic)
  case generate(TextGenerationOptions)
  case control(ControlGenerationOptions)
  case quantize(QuantizeOptions)
  case quantizeControlnet(QuantizeControlnetOptions)
}

public enum ServeParsedCommand: Sendable, Equatable {
  case help(CLIUsageTopic)
  case serve(ServeOptions)
  case submit(socketPath: String?, job: GenerationJobPayload)
  case quantize(QuantizeOptions)
  case quantizeControlnet(QuantizeControlnetOptions)
}

public struct JobProgressUpdate: Codable, Sendable, Equatable {
  public var stage: String
  public var stepIndex: Int
  public var totalSteps: Int
  public var fractionCompleted: Double
  public var enhancedPrompt: String?

  public init(
    stage: String,
    stepIndex: Int,
    totalSteps: Int,
    fractionCompleted: Double,
    enhancedPrompt: String? = nil
  ) {
    self.stage = stage
    self.stepIndex = stepIndex
    self.totalSteps = totalSteps
    self.fractionCompleted = fractionCompleted
    self.enhancedPrompt = enhancedPrompt
  }
}

public enum CLIErrors {
  public static func describe(_ error: Error) -> String {
    if let localizedError = error as? LocalizedError,
      let message = localizedError.errorDescription
    {
      return message
    }
    return String(describing: error)
  }
}
