import Foundation
import ZImage

public enum CLICompatParser {
  private static let minimumImageDimension = 64
  private static let requiredImageDimensionMultiple = 16

  public static func parseCLI(_ args: [String]) throws -> CLIParsedCommand {
    if let first = args.first {
      switch first {
      case "quantize":
        return try parseQuantize(Array(args.dropFirst()))
      case "quantize-controlnet":
        return try parseQuantizeControlnet(Array(args.dropFirst()))
      case "control":
        return try parseControl(Array(args.dropFirst()))
      default:
        break
      }
    }

    if args.contains("--help") || args.contains("-h") {
      return .help(.main)
    }

    return .generate(try parseTextGeneration(args))
  }

  public static func parseServe(_ args: [String]) throws -> ServeParsedCommand {
    var socketPath: String?
    var remaining = args[...]
    while let first = remaining.first, first == "--socket" || first == "-S" {
      remaining = remaining.dropFirst()
      guard let value = remaining.first, !value.hasPrefix("-") else {
        throw CLIError(message: "Missing value for \(first)", usage: .serve)
      }
      socketPath = value
      remaining = remaining.dropFirst()
    }

    let finalArgs = Array(remaining)
    if finalArgs.isEmpty || finalArgs == ["--help"] || finalArgs == ["-h"] {
      return .help(.main)
    }

    if let first = finalArgs.first {
      switch first {
      case "serve":
        return try parseServeCommand(Array(finalArgs.dropFirst()), socketPath: socketPath)
      case "status":
        return try parseStatusCommand(Array(finalArgs.dropFirst()), socketPath: socketPath)
      case "cancel":
        return try parseCancelCommand(Array(finalArgs.dropFirst()), socketPath: socketPath)
      case "shutdown":
        return try parseShutdownCommand(Array(finalArgs.dropFirst()), socketPath: socketPath)
      case "batch":
        return try parseBatchCommand(Array(finalArgs.dropFirst()), socketPath: socketPath)
      case "markdown":
        return try parseMarkdownCommand(Array(finalArgs.dropFirst()), socketPath: socketPath)
      case "quantize":
        return .quantize(try parseQuantizeOptions(Array(finalArgs.dropFirst()), usage: .quantize))
      case "quantize-controlnet":
        return .quantizeControlnet(
          try parseQuantizeControlnetOptions(Array(finalArgs.dropFirst()), usage: .quantizeControlnet))
      case "control":
        return .submit(socketPath: socketPath, job: .control(try parseControlOptions(Array(finalArgs.dropFirst()))))
      default:
        break
      }
    }

    return .submit(socketPath: socketPath, job: .text(try parseTextGeneration(finalArgs)))
  }

  private static func parseServeCommand(_ args: [String], socketPath: String?) throws -> ServeParsedCommand {
    if args.isEmpty || args.contains("--help") || args.contains("-h") {
      return .help(.serve)
    }

    var iterator = args.makeIterator()
    var resolvedSocketPath = socketPath
    var residencyPolicy: ModuleResidencyPolicy = .adaptive
    var warmModel: String?
    var warmWeightsVariant: String?
    var warmControlnetWeights: String?
    var warmControlnetFile: String?
    var warmMaxSequenceLength = 512
    var idleTimeoutSeconds: TimeInterval = 300
    while let arg = iterator.next() {
      switch arg {
      case "--socket", "-S":
        resolvedSocketPath = try nextValue(for: arg, iterator: &iterator, usage: .serve)
      case "--residency-policy":
        residencyPolicy = try residencyPolicyValue(for: arg, iterator: &iterator, usage: .serve)
      case "--warm-model":
        warmModel = try nextValue(for: arg, iterator: &iterator, usage: .serve)
      case "--weights-variant":
        warmWeightsVariant = try nextValue(for: arg, iterator: &iterator, usage: .serve)
      case "--warm-controlnet-weights":
        warmControlnetWeights = try nextValue(for: arg, iterator: &iterator, usage: .serve)
      case "--warm-control-file":
        warmControlnetFile = try nextValue(for: arg, iterator: &iterator, usage: .serve)
      case "--max-sequence-length":
        warmMaxSequenceLength = try intValue(for: arg, iterator: &iterator, minimum: 64, usage: .serve)
      case "--idle-timeout":
        idleTimeoutSeconds = TimeInterval(try floatValue(for: arg, iterator: &iterator, minimum: 0, usage: .serve))
      default:
        throw CLIError(message: "Unknown serve argument: \(arg)", usage: .serve)
      }
    }

    return .serve(
      ServeOptions(
        socketPath: resolvedSocketPath,
        residencyPolicy: residencyPolicy,
        warmModel: warmModel,
        warmWeightsVariant: warmWeightsVariant,
        warmControlnetWeights: warmControlnetWeights,
        warmControlnetFile: warmControlnetFile,
        warmMaxSequenceLength: warmMaxSequenceLength,
        idleTimeoutSeconds: idleTimeoutSeconds
      ))
  }

  private static func parseStatusCommand(_ args: [String], socketPath: String?) throws -> ServeParsedCommand {
    if args.contains("--help") || args.contains("-h") {
      return .help(.status)
    }
    guard args.isEmpty else {
      throw CLIError(message: "Unknown status argument: \(args[0])", usage: .status)
    }
    return .status(socketPath: socketPath)
  }

  private static func parseCancelCommand(_ args: [String], socketPath: String?) throws -> ServeParsedCommand {
    if args.isEmpty || args.contains("--help") || args.contains("-h") {
      return .help(.cancel)
    }
    guard args.count == 1 else {
      throw CLIError(message: "Expected a single <job-id> argument", usage: .cancel)
    }
    return .cancel(socketPath: socketPath, jobID: args[0])
  }

  private static func parseShutdownCommand(_ args: [String], socketPath: String?) throws -> ServeParsedCommand {
    if args.contains("--help") || args.contains("-h") {
      return .help(.shutdown)
    }
    guard args.isEmpty else {
      throw CLIError(message: "Unknown shutdown argument: \(args[0])", usage: .shutdown)
    }
    return .shutdown(socketPath: socketPath)
  }

  private static func parseBatchCommand(_ args: [String], socketPath: String?) throws -> ServeParsedCommand {
    if args.isEmpty || args.contains("--help") || args.contains("-h") {
      return .help(.batch)
    }
    guard args.count == 1 else {
      throw CLIError(message: "Expected a single <jobs.json> argument", usage: .batch)
    }
    return .batch(socketPath: socketPath, manifestPath: args[0])
  }

  private static func parseMarkdownCommand(_ args: [String], socketPath: String?) throws -> ServeParsedCommand {
    if args.isEmpty || args.contains("--help") || args.contains("-h") {
      return .help(.markdown)
    }
    guard args.count == 1 else {
      throw CLIError(message: "Expected a single <prompts.md> argument", usage: .markdown)
    }
    return .markdown(socketPath: socketPath, markdownPath: args[0])
  }

  private static func parseTextGeneration(_ args: [String]) throws -> TextGenerationOptions {
    let options = try parseTextGenerationOptions(args)
    guard !options.prompt.isEmpty else {
      throw CLIError(message: "Missing required --prompt argument", usage: .main)
    }
    return options
  }

  private static func parseTextGenerationOptions(_ args: [String]) throws -> TextGenerationOptions {
    var prompt: String?
    var negativePrompt: String?
    var width: Int?
    var height: Int?
    var steps: Int?
    var guidance: Float?
    var cfgNormalization = false
    var cfgTruncation: Float = 1.0
    var seed: UInt64?
    var outputPath = "z-image.png"
    var model: String?
    var weightsVariant: String?
    var cacheLimit: Int?
    var maxSequenceLength: Int?
    var loraPath: String?
    var loraFile: String?
    var loraScale: Float?
    var enhancePrompt = false
    var enhanceMaxTokens = 512
    var noProgress = false
    var forceTransformerOverrideOnly = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p":
        prompt = try nextValue(for: arg, iterator: &iterator, usage: .main)
      case "--negative-prompt", "--np":
        negativePrompt = try nextValue(for: arg, iterator: &iterator, usage: .main)
      case "--width", "-W":
        width = try imageDimensionValue(for: arg, iterator: &iterator, usage: .main)
      case "--height", "-H":
        height = try imageDimensionValue(for: arg, iterator: &iterator, usage: .main)
      case "--steps", "-s":
        steps = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: .main)
      case "--guidance", "-g":
        guidance = try floatValue(for: arg, iterator: &iterator, minimum: 0, usage: .main)
      case "--cfg-normalization":
        cfgNormalization = true
      case "--cfg-truncation":
        cfgTruncation = try floatValue(for: arg, iterator: &iterator, minimum: 0, maximum: 1, usage: .main)
      case "--seed":
        seed = try uint64Value(for: arg, iterator: &iterator, usage: .main)
      case "--output", "-o":
        outputPath = try nextValue(for: arg, iterator: &iterator, usage: .main)
      case "--model", "-m":
        model = try nextValue(for: arg, iterator: &iterator, usage: .main)
      case "--weights-variant":
        weightsVariant = try nextValue(for: arg, iterator: &iterator, usage: .main)
      case "--force-transformer-override-only":
        forceTransformerOverrideOnly = true
      case "--cache-limit":
        cacheLimit = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: .main)
      case "--max-sequence-length":
        maxSequenceLength = try intValue(for: arg, iterator: &iterator, minimum: 64, usage: .main)
      case "--lora", "-l":
        loraPath = try nextValue(for: arg, iterator: &iterator, usage: .main)
      case "--lora-file":
        loraFile = try nextValue(for: arg, iterator: &iterator, usage: .main)
      case "--lora-scale":
        loraScale = try floatValue(for: arg, iterator: &iterator, usage: .main)
      case "--enhance", "-e":
        enhancePrompt = true
      case "--enhance-max-tokens":
        enhanceMaxTokens = try intValue(for: arg, iterator: &iterator, minimum: 64, usage: .main)
      case "--no-progress":
        noProgress = true
      case "--help", "-h":
        return TextGenerationOptions(prompt: "")
      default:
        throw CLIError(message: "Unknown argument: \(arg)", usage: .main)
      }
    }

    return TextGenerationOptions(
      prompt: prompt ?? "",
      negativePrompt: negativePrompt,
      width: width,
      height: height,
      steps: steps,
      guidance: guidance,
      cfgNormalization: cfgNormalization,
      cfgTruncation: cfgTruncation,
      seed: seed,
      outputPath: outputPath,
      model: model,
      weightsVariant: weightsVariant,
      cacheLimit: cacheLimit,
      maxSequenceLength: maxSequenceLength,
      loraPath: loraPath,
      loraFile: loraFile,
      loraScale: loraScale,
      enhancePrompt: enhancePrompt,
      enhanceMaxTokens: enhanceMaxTokens,
      noProgress: noProgress,
      forceTransformerOverrideOnly: forceTransformerOverrideOnly
    )
  }

  private static func parseControl(_ args: [String]) throws -> CLIParsedCommand {
    if args.contains("--help") || args.contains("-h") {
      return .help(.control)
    }
    return .control(try parseControlOptions(args))
  }

  private static func parseControlOptions(_ args: [String]) throws -> ControlGenerationOptions {
    var prompt: String?
    var negativePrompt: String?
    var controlImage: String?
    var inpaintImage: String?
    var maskImage: String?
    var controlScale: Float = 0.75
    var controlnetWeights: String?
    var controlnetWeightsFile: String?
    var width: Int?
    var height: Int?
    var steps: Int?
    var guidance: Float?
    var cfgNormalization = false
    var cfgTruncation: Float = 1.0
    var seed: UInt64?
    var outputPath = "z-image-control.png"
    var model: String?
    var weightsVariant: String?
    var cacheLimit: Int?
    var maxSequenceLength: Int?
    var loraPath: String?
    var loraFile: String?
    var loraScale: Float?
    var enhancePrompt = false
    var enhanceMaxTokens = 512
    var logControlMemory = false
    var noProgress = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p":
        prompt = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--negative-prompt", "--np":
        negativePrompt = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--control-image", "-c":
        controlImage = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--inpaint-image", "-i":
        inpaintImage = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--mask", "--mask-image":
        maskImage = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--control-scale", "--cs":
        controlScale = try floatValue(for: arg, iterator: &iterator, minimum: 0, usage: .control)
      case "--controlnet-weights", "--cw":
        controlnetWeights = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--control-file", "--cf":
        controlnetWeightsFile = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--width", "-W":
        width = try imageDimensionValue(for: arg, iterator: &iterator, usage: .control)
      case "--height", "-H":
        height = try imageDimensionValue(for: arg, iterator: &iterator, usage: .control)
      case "--steps", "-s":
        steps = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: .control)
      case "--guidance", "-g":
        guidance = try floatValue(for: arg, iterator: &iterator, minimum: 0, usage: .control)
      case "--cfg-normalization":
        cfgNormalization = true
      case "--cfg-truncation":
        cfgTruncation = try floatValue(for: arg, iterator: &iterator, minimum: 0, maximum: 1, usage: .control)
      case "--seed":
        seed = try uint64Value(for: arg, iterator: &iterator, usage: .control)
      case "--output", "-o":
        outputPath = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--model", "-m":
        model = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--weights-variant":
        weightsVariant = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--cache-limit":
        cacheLimit = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: .control)
      case "--max-sequence-length":
        maxSequenceLength = try intValue(for: arg, iterator: &iterator, minimum: 64, usage: .control)
      case "--lora", "-l":
        loraPath = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--lora-file":
        loraFile = try nextValue(for: arg, iterator: &iterator, usage: .control)
      case "--lora-scale":
        loraScale = try floatValue(for: arg, iterator: &iterator, usage: .control)
      case "--enhance", "-e":
        enhancePrompt = true
      case "--enhance-max-tokens":
        enhanceMaxTokens = try intValue(for: arg, iterator: &iterator, minimum: 64, usage: .control)
      case "--log-control-memory":
        logControlMemory = true
      case "--no-progress":
        noProgress = true
      case "--help", "-h":
        throw CLIError(message: "", usage: .control)
      default:
        throw CLIError(message: "Unknown control argument: \(arg)", usage: .control)
      }
    }

    guard let prompt else {
      throw CLIError(message: "Missing required --prompt argument", usage: .control)
    }
    if controlImage == nil, inpaintImage == nil, maskImage == nil {
      throw CLIError(
        message: "At least one of --control-image, --inpaint-image, or --mask must be provided",
        usage: .control
      )
    }
    guard let controlnetWeights else {
      throw CLIError(message: "Missing required --controlnet-weights argument", usage: .control)
    }

    return ControlGenerationOptions(
      prompt: prompt,
      negativePrompt: negativePrompt,
      controlImage: controlImage,
      inpaintImage: inpaintImage,
      maskImage: maskImage,
      controlScale: controlScale,
      controlnetWeights: controlnetWeights,
      controlnetWeightsFile: controlnetWeightsFile,
      width: width,
      height: height,
      steps: steps,
      guidance: guidance,
      cfgNormalization: cfgNormalization,
      cfgTruncation: cfgTruncation,
      seed: seed,
      outputPath: outputPath,
      model: model,
      weightsVariant: weightsVariant,
      cacheLimit: cacheLimit,
      maxSequenceLength: maxSequenceLength,
      loraPath: loraPath,
      loraFile: loraFile,
      loraScale: loraScale,
      enhancePrompt: enhancePrompt,
      enhanceMaxTokens: enhanceMaxTokens,
      logControlMemory: logControlMemory,
      noProgress: noProgress
    )
  }

  private static func parseQuantize(_ args: [String]) throws -> CLIParsedCommand {
    if args.contains("--help") || args.contains("-h") {
      return .help(.quantize)
    }
    return .quantize(try parseQuantizeOptions(args, usage: .quantize))
  }

  private static func parseQuantizeOptions(_ args: [String], usage: CLIUsageTopic) throws -> QuantizeOptions {
    var input: String?
    var output: String?
    var bits = 8
    var groupSize = 32
    var verbose = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--input", "-i":
        input = try nextValue(for: arg, iterator: &iterator, usage: usage)
      case "--output", "-o":
        output = try nextValue(for: arg, iterator: &iterator, usage: usage)
      case "--bits":
        bits = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: usage)
      case "--group-size":
        groupSize = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: usage)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        throw CLIError(message: "", usage: usage)
      default:
        throw CLIError(message: "Unknown quantize argument: \(arg)", usage: usage)
      }
    }

    guard let input else {
      throw CLIError(message: "Missing required --input argument", usage: usage)
    }
    guard let output else {
      throw CLIError(message: "Missing required --output argument", usage: usage)
    }
    return QuantizeOptions(input: input, output: output, bits: bits, groupSize: groupSize, verbose: verbose)
  }

  private static func parseQuantizeControlnet(_ args: [String]) throws -> CLIParsedCommand {
    if args.contains("--help") || args.contains("-h") {
      return .help(.quantizeControlnet)
    }
    return .quantizeControlnet(try parseQuantizeControlnetOptions(args, usage: .quantizeControlnet))
  }

  private static func parseQuantizeControlnetOptions(_ args: [String], usage: CLIUsageTopic) throws
    -> QuantizeControlnetOptions
  {
    var input: String?
    var output: String?
    var specificFile: String?
    var bits = 8
    var groupSize = 32
    var verbose = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--input", "-i":
        input = try nextValue(for: arg, iterator: &iterator, usage: usage)
      case "--output", "-o":
        output = try nextValue(for: arg, iterator: &iterator, usage: usage)
      case "--file", "-f":
        specificFile = try nextValue(for: arg, iterator: &iterator, usage: usage)
      case "--bits":
        bits = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: usage)
      case "--group-size":
        groupSize = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: usage)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        throw CLIError(message: "", usage: usage)
      default:
        throw CLIError(message: "Unknown quantize-controlnet argument: \(arg)", usage: usage)
      }
    }

    guard let input else {
      throw CLIError(message: "Missing required --input argument", usage: usage)
    }
    guard let output else {
      throw CLIError(message: "Missing required --output argument", usage: usage)
    }
    return QuantizeControlnetOptions(
      input: input,
      output: output,
      specificFile: specificFile,
      bits: bits,
      groupSize: groupSize,
      verbose: verbose
    )
  }

  private static func nextValue(
    for arg: String,
    iterator: inout IndexingIterator<[String]>,
    usage: CLIUsageTopic
  ) throws -> String {
    guard let value = iterator.next(), !value.hasPrefix("-") else {
      throw CLIError(message: "Missing value for \(arg)", usage: usage)
    }
    return value
  }

  private static func intValue(
    for arg: String,
    iterator: inout IndexingIterator<[String]>,
    minimum: Int,
    usage: CLIUsageTopic
  ) throws -> Int {
    let rawValue = try nextValue(for: arg, iterator: &iterator, usage: usage)
    guard let value = Int(rawValue), value >= minimum else {
      throw CLIError(message: "Invalid value for \(arg): '\(rawValue)'. Expected an integer >= \(minimum).", usage: usage)
    }
    return value
  }

  private static func imageDimensionValue(
    for arg: String,
    iterator: inout IndexingIterator<[String]>,
    usage: CLIUsageTopic
  ) throws -> Int {
    let value = try intValue(for: arg, iterator: &iterator, minimum: minimumImageDimension, usage: usage)
    guard value % requiredImageDimensionMultiple == 0 else {
      throw CLIError(
        message:
          "Invalid value for \(arg): '\(value)'. Expected a multiple of \(requiredImageDimensionMultiple).",
        usage: usage
      )
    }
    return value
  }

  private static func floatValue(
    for arg: String,
    iterator: inout IndexingIterator<[String]>,
    minimum: Float? = nil,
    maximum: Float? = nil,
    usage: CLIUsageTopic
  ) throws -> Float {
    let rawValue = try nextValue(for: arg, iterator: &iterator, usage: usage)
    guard let value = Float(rawValue) else {
      throw CLIError(message: "Invalid value for \(arg): '\(rawValue)'. Expected a number.", usage: usage)
    }
    if let minimum, value < minimum {
      throw CLIError(message: "Invalid value for \(arg): '\(rawValue)'. Expected a number >= \(minimum).", usage: usage)
    }
    if let maximum, value > maximum {
      throw CLIError(message: "Invalid value for \(arg): '\(rawValue)'. Expected a number <= \(maximum).", usage: usage)
    }
    return value
  }

  private static func uint64Value(
    for arg: String,
    iterator: inout IndexingIterator<[String]>,
    usage: CLIUsageTopic
  ) throws -> UInt64 {
    let rawValue = try nextValue(for: arg, iterator: &iterator, usage: usage)
    guard let value = UInt64(rawValue) else {
      throw CLIError(
        message: "Invalid value for \(arg): '\(rawValue)'. Expected an unsigned integer seed.",
        usage: usage
      )
    }
    return value
  }

  private static func residencyPolicyValue(
    for arg: String,
    iterator: inout IndexingIterator<[String]>,
    usage: CLIUsageTopic
  ) throws -> ModuleResidencyPolicy {
    let rawValue = try nextValue(for: arg, iterator: &iterator, usage: usage)
    guard let policy = ModuleResidencyPolicy(rawValue: rawValue) else {
      throw CLIError(
        message: "Invalid value for \(arg): '\(rawValue)'. Expected one-shot, warm, or adaptive.",
        usage: usage
      )
    }
    return policy
  }
}
