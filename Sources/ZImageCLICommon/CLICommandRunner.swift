import Dispatch
import Foundation
import Logging
import MLX
import ZImage

private final class Box<T>: @unchecked Sendable {
  var value: T

  init(_ value: T) {
    self.value = value
  }
}

public enum CLICommandRunner {
  private static let defaultCacheLimit = Memory.cacheLimit

  public static func run(_ command: CLIParsedCommand, logger: Logger, program: CLIProgramKind = .cli) throws {
    switch command {
    case .help(let usage):
      print(CLIUsageFormatter.usage(for: usage, program: program))
    case .generate(let options):
      try waitForAsync {
        _ = try await executeTextGeneration(options, logger: logger, progressSink: nil)
      }
    case .control(let options):
      try waitForAsync {
        _ = try await executeControlGeneration(options, logger: logger, progressSink: nil)
      }
    case .quantize(let options):
      try runQuantize(options)
    case .quantizeControlnet(let options):
      try runQuantizeControlnet(options)
    }
  }

  public static func executeGenerationJob(
    _ job: GenerationJobPayload,
    logger: Logger,
    progressSink: (@Sendable (JobProgressUpdate) -> Void)? = nil
  ) async throws -> URL {
    switch job {
    case .text(let options):
      return try await executeTextGeneration(options, logger: logger, progressSink: progressSink)
    case .control(let options):
      return try await executeControlGeneration(options, logger: logger, progressSink: progressSink)
    }
  }

  public static func executeTextGeneration(
    _ options: TextGenerationOptions,
    logger: Logger,
    progressSink: (@Sendable (JobProgressUpdate) -> Void)? = nil
  ) async throws -> URL {
    let plan = buildTextRequest(options, logger: logger)
    applyCacheLimit(options.cacheLimit, logger: logger)

    let pipeline = ZImagePipeline(logger: logger)
    let progressRenderer = TerminalProgressRenderer(noProgress: options.noProgress, totalSteps: plan.preset.steps)

    defer {
      progressRenderer.finish()
    }

    let outputURL = try await pipeline.generate(
      plan.request,
      progressHandler: { progress in
        guard progress.stage == .denoising else { return }
        let update = JobProgressUpdate(
          stage: progress.stage.rawValue,
          stepIndex: progress.stepIndex,
          totalSteps: progress.totalSteps,
          fractionCompleted: progress.fractionCompleted
        )
        progressSink?(update)
        progressRenderer.report(update)
      })
    return outputURL
  }

  public static func executeControlGeneration(
    _ options: ControlGenerationOptions,
    logger: Logger,
    progressSink: (@Sendable (JobProgressUpdate) -> Void)? = nil
  ) async throws -> URL {
    let plan = try buildControlRequest(options, logger: logger)
    applyCacheLimit(options.cacheLimit, logger: logger)

    let pipeline = ZImageControlPipeline(logger: logger)
    let progressRenderer = TerminalProgressRenderer(noProgress: options.noProgress, totalSteps: plan.preset.steps)

    defer {
      progressRenderer.finish()
    }

    let outputURL = try await pipeline.generate(
      ZImageControlGenerationRequest(
        prompt: plan.request.prompt,
        negativePrompt: plan.request.negativePrompt,
        controlImage: plan.request.controlImage,
        inpaintImage: plan.request.inpaintImage,
        maskImage: plan.request.maskImage,
        controlContextScale: plan.request.controlContextScale,
        width: plan.request.width,
        height: plan.request.height,
        steps: plan.request.steps,
        guidanceScale: plan.request.guidanceScale,
        cfgNormalization: plan.request.cfgNormalization,
        cfgTruncation: plan.request.cfgTruncation,
        seed: plan.request.seed,
        outputPath: plan.request.outputPath,
        model: plan.request.model,
        weightsVariant: plan.request.weightsVariant,
        controlnetWeights: plan.request.controlnetWeights,
        controlnetWeightsFile: plan.request.controlnetWeightsFile,
        maxSequenceLength: plan.request.maxSequenceLength,
        lora: plan.request.lora,
        progressCallback: { progress in
          guard progress.stage == "Denoising" else {
            if let enhancedPrompt = progress.enhancedPrompt {
              progressSink?(
                JobProgressUpdate(
                  stage: progress.stage,
                  stepIndex: progress.stepIndex,
                  totalSteps: progress.totalSteps,
                  fractionCompleted: progress.fractionCompleted,
                  enhancedPrompt: enhancedPrompt
                ))
            }
            return
          }
          let update = JobProgressUpdate(
            stage: progress.stage,
            stepIndex: progress.stepIndex,
            totalSteps: progress.totalSteps,
            fractionCompleted: progress.fractionCompleted
          )
          progressSink?(update)
          progressRenderer.report(update)
        },
        enhancePrompt: plan.request.enhancePrompt,
        enhanceMaxTokens: plan.request.enhanceMaxTokens,
        runtimeOptions: plan.request.runtimeOptions
      ))
    return outputURL
  }

  public static func runQuantize(_ options: QuantizeOptions) throws {
    let inputURL = URL(fileURLWithPath: options.input)
    let outputURL = URL(fileURLWithPath: options.output)
    var isDirectory: ObjCBool = false

    guard FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
      throw CLIError(message: "Input model directory not found: \(options.input)", usage: .quantize)
    }
    guard ZImageQuantizer.supportedBits.contains(options.bits) else {
      throw CLIError(message: "Invalid bits: \(options.bits). Supported values: 4, 8", usage: .quantize)
    }
    guard ZImageQuantizer.supportedGroupSizes.contains(options.groupSize) else {
      throw CLIError(
        message: "Invalid group size: \(options.groupSize). Supported values: 32, 64, 128",
        usage: .quantize
      )
    }

    let spec = ZImageQuantizationSpec(groupSize: options.groupSize, bits: options.bits, mode: .affine)
    print("Quantizing: \(options.input) -> \(options.output)")
    print("Config: \(options.bits)-bit, group_size=\(options.groupSize)")
    try ZImageQuantizer.quantizeAndSave(from: inputURL, to: outputURL, spec: spec, verbose: options.verbose)
    print("Done: \(outputURL.path)")
  }

  public static func runQuantizeControlnet(_ options: QuantizeControlnetOptions) throws {
    guard ZImageQuantizer.supportedBits.contains(options.bits) else {
      throw CLIError(message: "Invalid bits: \(options.bits). Supported values: 4, 8", usage: .quantizeControlnet)
    }
    guard ZImageQuantizer.supportedGroupSizes.contains(options.groupSize) else {
      throw CLIError(
        message: "Invalid group size: \(options.groupSize). Supported values: 32, 64, 128",
        usage: .quantizeControlnet
      )
    }

    let outputURL = URL(fileURLWithPath: options.output)
    let spec = ZImageQuantizationSpec(groupSize: options.groupSize, bits: options.bits, mode: .affine)
    print("Quantizing ControlNet: \(options.input) -> \(options.output)")
    print("Config: \(options.bits)-bit, group_size=\(options.groupSize)")

    try waitForAsync {
      let sourceURL: URL
      let localURL = URL(fileURLWithPath: options.input)

      if FileManager.default.fileExists(atPath: localURL.path) {
        sourceURL = localURL
        if options.verbose {
          print("Using local ControlNet: \(options.input)")
        }
      } else if ModelResolution.isHuggingFaceModelId(options.input) {
        if options.verbose {
          print("Downloading ControlNet from HuggingFace: \(options.input)")
        }
        sourceURL = try await ModelResolution.resolve(
          modelSpec: options.input,
          filePatterns: ["*.safetensors", "*.json"],
          progressHandler: { progress in
            let percent = Int(progress.fractionCompleted * 100)
            print("Downloading: \(percent)%")
          }
        )
        if options.verbose {
          print("Downloaded to: \(sourceURL.path)")
        }
      } else {
        throw CLIError(
          message: "Input not found: \(options.input). Provide a local path or HuggingFace model ID.",
          usage: .quantizeControlnet
        )
      }

      try ZImageQuantizer.quantizeControlnet(
        from: sourceURL,
        to: outputURL,
        spec: spec,
        specificFile: options.specificFile,
        verbose: options.verbose
      )
      print("Done: \(outputURL.path)")
    }
  }

  private static func buildTextRequest(_ options: TextGenerationOptions, logger: Logger) -> (
    request: ZImageGenerationRequest, preset: ZImagePreset
  ) {
    let preset = ZImagePreset.resolved(
      for: options.model,
      width: options.width,
      height: options.height,
      steps: options.steps,
      guidanceScale: options.guidance,
      maxSequenceLength: options.maxSequenceLength
    )
    let loraConfig = makeLoRAConfiguration(path: options.loraPath, scale: options.loraScale)
    warnIfLoRAUsesModelDefaults(
      loraConfig: loraConfig,
      steps: options.steps,
      guidance: options.guidance,
      preset: preset,
      logger: logger
    )

    let request = ZImageGenerationRequest(
      prompt: options.prompt,
      negativePrompt: options.negativePrompt ?? preset.negativePrompt,
      width: preset.width,
      height: preset.height,
      steps: preset.steps,
      guidanceScale: preset.guidanceScale,
      cfgNormalization: options.cfgNormalization,
      cfgTruncation: options.cfgTruncation,
      seed: options.seed,
      outputPath: URL(fileURLWithPath: options.outputPath),
      model: options.model,
      weightsVariant: options.weightsVariant,
      maxSequenceLength: preset.maxSequenceLength,
      lora: loraConfig,
      enhancePrompt: options.enhancePrompt,
      enhanceMaxTokens: options.enhanceMaxTokens,
      forceTransformerOverrideOnly: options.forceTransformerOverrideOnly
    )
    return (request, preset)
  }

  private static func buildControlRequest(_ options: ControlGenerationOptions, logger: Logger) throws -> (
    request: ZImageControlGenerationRequest, preset: ZImagePreset
  ) {
    let preset = ZImagePreset.resolved(
      for: options.model,
      width: options.width,
      height: options.height,
      steps: options.steps,
      guidanceScale: options.guidance,
      maxSequenceLength: options.maxSequenceLength
    )

    let controlImageURL = try validatedOptionalFileURL(
      path: options.controlImage,
      missingMessage: "Control image not found: "
    )
    let inpaintImageURL = try validatedOptionalFileURL(
      path: options.inpaintImage,
      missingMessage: "Inpaint image not found: "
    )
    let maskImageURL = try validatedOptionalFileURL(
      path: options.maskImage,
      missingMessage: "Mask image not found: "
    )

    let loraConfig = makeLoRAConfiguration(path: options.loraPath, scale: options.loraScale)
    warnIfLoRAUsesModelDefaults(
      loraConfig: loraConfig,
      steps: options.steps,
      guidance: options.guidance,
      preset: preset,
      logger: logger
    )

    let request = ZImageControlGenerationRequest(
      prompt: options.prompt,
      negativePrompt: options.negativePrompt ?? preset.negativePrompt,
      controlImage: controlImageURL,
      inpaintImage: inpaintImageURL,
      maskImage: maskImageURL,
      controlContextScale: options.controlScale,
      width: preset.width,
      height: preset.height,
      steps: preset.steps,
      guidanceScale: preset.guidanceScale,
      cfgNormalization: options.cfgNormalization,
      cfgTruncation: options.cfgTruncation,
      seed: options.seed,
      outputPath: URL(fileURLWithPath: options.outputPath),
      model: options.model,
      weightsVariant: options.weightsVariant,
      controlnetWeights: options.controlnetWeights,
      controlnetWeightsFile: options.controlnetWeightsFile,
      maxSequenceLength: preset.maxSequenceLength,
      lora: loraConfig,
      progressCallback: nil,
      enhancePrompt: options.enhancePrompt,
      enhanceMaxTokens: options.enhanceMaxTokens,
      runtimeOptions: .init(logPhaseMemory: options.logControlMemory)
    )
    return (request, preset)
  }

  private static func validatedOptionalFileURL(path: String?, missingMessage: String) throws -> URL? {
    guard let path else { return nil }
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw CLIError(message: "\(missingMessage)\(path)", usage: .control)
    }
    return url
  }

  private static func applyCacheLimit(_ cacheLimit: Int?, logger: Logger) {
    if let limit = cacheLimit {
      Memory.cacheLimit = limit * 1024 * 1024
      logger.info("GPU cache limit set to \(limit)MB")
    } else if Memory.cacheLimit != defaultCacheLimit {
      Memory.cacheLimit = defaultCacheLimit
    }
  }

  private static func makeLoRAConfiguration(path: String?, scale: Float) -> LoRAConfiguration? {
    guard let path else { return nil }
    if path.hasPrefix("/") || path.hasPrefix("./") || path.hasPrefix("~") {
      return .local(path, scale: scale)
    }
    return .huggingFace(path, scale: scale)
  }

  private static func warnIfLoRAUsesModelDefaults(
    loraConfig: LoRAConfiguration?,
    steps: Int?,
    guidance: Float?,
    preset: ZImagePreset,
    logger: Logger
  ) {
    guard loraConfig != nil, steps == nil || guidance == nil else { return }
    logger.warning(
      "Using model defaults with LoRA (steps=\(preset.steps), guidance=\(preset.guidanceScale)). Adapter-specific sampling can differ; set --steps and --guidance explicitly when the adapter card recommends values."
    )
  }

  private static func waitForAsync(_ operation: @escaping @Sendable () async throws -> Void) throws {
    let semaphore = DispatchSemaphore(value: 0)
    let errorBox = Box<Error?>(nil)
    Task {
      do {
        try await operation()
      } catch {
        errorBox.value = error
      }
      semaphore.signal()
    }
    semaphore.wait()
    if let error = errorBox.value {
      throw error
    }
  }
}
