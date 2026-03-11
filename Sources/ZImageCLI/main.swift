import Darwin
import Dispatch
import Foundation
import Logging
import MLX
import Metal
import ZImage

#if canImport(CoreGraphics)
  import CoreGraphics
  import ImageIO
#endif

LoggingSystem.bootstrap { label in
  var handler = StreamLogHandler.standardError(label: label)
  handler.logLevel = .info
  return handler
}

private final class Box<T>: @unchecked Sendable {
  var value: T
  init(_ value: T) {
    self.value = value
  }
}

enum ZImageCLI {
  private enum UsageTopic {
    case main
    case quantize
    case quantizeControlnet
    case control
  }

  private enum Subcommand: String {
    case quantize
    case quantizeControlnet = "quantize-controlnet"
    case control
  }

  private struct CLIError: LocalizedError {
    let message: String
    let usage: UsageTopic?

    var errorDescription: String? { message }
  }

  static let logger: Logger = {
    var logger = Logger(label: "z-image.cli")
    logger.logLevel = .info
    return logger
  }()

  private static let minimumImageDimension = 64
  private static let requiredImageDimensionMultiple = 16

  static func main() -> Never {
    do {
      try run()
      Darwin.exit(EXIT_SUCCESS)
    } catch let error as CLIError {
      logger.error("\(error.message)")
      if let usage = error.usage {
        printUsage(for: usage)
      }
      Darwin.exit(EXIT_FAILURE)
    } catch {
      logger.error("\(describe(error))")
      Darwin.exit(EXIT_FAILURE)
    }
  }

  // swiftlint:disable:next cyclomatic_complexity
  static func run() throws {
    if let dev = MTLCreateSystemDefaultDevice() {
      logger.info("Metal device: \(dev.name)")
    } else {
      logger.warning("No Metal device detected; MLX will fall back to CPU.")
    }

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
    var loraScale: Float = 1.0
    var enhancePrompt = false
    var enhanceMaxTokens = 512
    var noProgress = false
    var forceTransformerOverrideOnly = false

    let args = Array(CommandLine.arguments.dropFirst())
    if let first = args.first, let subcommand = Subcommand(rawValue: first) {
      switch subcommand {
      case .quantize:
        try runQuantize(args: Array(args.dropFirst()))
      case .quantizeControlnet:
        try runQuantizeControlnet(args: Array(args.dropFirst()))
      case .control:
        try runControl(args: Array(args.dropFirst()))
      }
      return
    }

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
      case "--lora-scale":
        loraScale = try floatValue(for: arg, iterator: &iterator, usage: .main)
      case "--enhance", "-e":
        enhancePrompt = true
      case "--enhance-max-tokens":
        enhanceMaxTokens = try intValue(for: arg, iterator: &iterator, minimum: 64, usage: .main)
      case "--no-progress":
        noProgress = true
      case "--help", "-h":
        printUsage()
        return
      default:
        throw CLIError(message: "Unknown argument: \(arg)", usage: .main)
      }
    }

    guard let prompt else {
      throw CLIError(message: "Missing required --prompt argument", usage: .main)
    }

    let preset = ZImagePreset.resolved(
      for: model,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      maxSequenceLength: maxSequenceLength
    )
    let resolvedNegativePrompt = negativePrompt ?? preset.negativePrompt

    if let limit = cacheLimit {
      Memory.cacheLimit = limit * 1024 * 1024
      logger.info("GPU cache limit set to \(limit)MB")
    }
    let loraConfig: LoRAConfiguration? = loraPath.map { path in
      if path.hasPrefix("/") || path.hasPrefix("./") || path.hasPrefix("~") {
        return .local(path, scale: loraScale)
      } else {
        return .huggingFace(path, scale: loraScale)
      }
    }
    if loraConfig != nil, steps == nil || guidance == nil {
      logger.warning(
        "Using model defaults with LoRA (steps=\(preset.steps), guidance=\(preset.guidanceScale)). Adapter-specific sampling can differ; set --steps and --guidance explicitly when the adapter card recommends values."
      )
    }

    let request = ZImageGenerationRequest(
      prompt: prompt,
      negativePrompt: resolvedNegativePrompt,
      width: preset.width,
      height: preset.height,
      steps: preset.steps,
      guidanceScale: preset.guidanceScale,
      cfgNormalization: cfgNormalization,
      cfgTruncation: cfgTruncation,
      seed: seed,
      outputPath: URL(fileURLWithPath: outputPath),
      model: model,
      weightsVariant: weightsVariant,
      maxSequenceLength: preset.maxSequenceLength,
      lora: loraConfig,
      enhancePrompt: enhancePrompt,
      enhanceMaxTokens: enhanceMaxTokens,
      forceTransformerOverrideOnly: forceTransformerOverrideOnly
    )

    let pipeline = ZImagePipeline(logger: logger)
    let semaphore = DispatchSemaphore(value: 0)
    let errorBox = Box<Error?>(nil)
    let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
    let bar = useBar ? ProgressBar(total: preset.steps) : nil
    Task {
      do {
        _ = try await pipeline.generate(
          request,
          progressHandler: { progress in
            guard !noProgress else { return }
            guard progress.stage == .denoising else { return }
            let completed = min(progress.totalSteps, max(0, progress.stepIndex))

            if let bar {
              bar.update(completed: completed)
              if completed == progress.totalSteps {
                bar.finish(forceNewline: true)
              }
            } else {
              PlainProgress.shared.report(completed: completed, total: progress.totalSteps)
            }
          })
      } catch {
        errorBox.value = error
      }
      if let bar { bar.finish(forceNewline: true) }
      semaphore.signal()
    }
    semaphore.wait()
    if let error = errorBox.value {
      throw error
    }
  }

  private static func printUsage() {
    print(
      """
      Z-Image Swift CLI

      Usage: ZImageCLI --prompt "text" [options]
        --prompt, -p           Text prompt (required)
        --negative-prompt      Negative prompt
        --width, -W            Output width (default \(ZImageModelMetadata.recommendedWidth))
        --height, -H           Output height (default \(ZImageModelMetadata.recommendedHeight))
                              Width and height must be >= \(minimumImageDimension) and divisible by \(requiredImageDimensionMultiple).
        --steps, -s            Inference steps (default: model-aware, 9 for Turbo / 50 for Base)
        --guidance, -g         Guidance scale (default: model-aware, 0.0 for Turbo / 4.0 for Base)
                              Steps count literal denoising iterations / transformer forwards.
        --cfg-normalization    Clamp CFG output norm to the positive-branch norm
        --cfg-truncation       Disable CFG after normalized timestep exceeds this value (default: 1.0)
        --seed                 Random seed
        --output, -o           Output path (default z-image.png)
        --model, -m            Model path or HuggingFace ID (default: \(ZImageRepository.id))
        --weights-variant      Weights precision variant (e.g. fp16, bf16)
        --force-transformer-override-only  Treat a local .safetensors as transformer-only override (disable AIO auto-detect)
        --cache-limit          GPU memory cache limit in MB (default: unlimited)
        --max-sequence-length  Maximum sequence length for text encoding (default: 512)
        --lora, -l             LoRA weights path or HuggingFace ID
        --lora-scale           LoRA scale factor (default: 1.0)
        --enhance, -e          Enhance prompt using LLM (requires ~5GB extra VRAM)
        --enhance-max-tokens   Max tokens for prompt enhancement (default: 512)
        --no-progress          Disable progress output
        --help, -h             Show help

      Known Tongyi-MAI ids, inspectable local or cached snapshots, and common Z-Image aliases apply model-aware presets. Unrecognized models still keep the Turbo-compatible preset unless you override the sampling flags.

      Subcommands:
        quantize               Quantize model weights
          --input, -i          Input model directory (required)
          --output, -o         Output directory (required)
          --bits               Bit width: 4 or 8 (default: 8)
          --group-size         Group size: 32, 64, 128 (default: 32)
          --verbose            Show progress

        quantize-controlnet    Quantize ControlNet weights
          --input, -i          Input ControlNet path or HuggingFace ID (required)
          --output, -o         Output directory (required)
          --bits               Bit width: 4 or 8 (default: 8)
          --group-size         Group size: 32, 64, 128 (default: 32)
          --verbose            Show progress

        control                Generate with ControlNet conditioning
          --prompt, -p         Text prompt (required)
          --control-image, -c  Control image path (optional; one of control image/inpaint image/mask required)
          --controlnet-weights Path to controlnet weights, local file/dir, or HuggingFace ID (required)
          --control-scale      Control scale (default: 0.75)
          Use 'ZImageCLI control --help' for full options

      Examples:
        ZImageCLI -p "a cute cat" -o cat.png
        ZImageCLI -p "a sunset" -m models/z-image-turbo-q8
        ZImageCLI -p "a forest" -m Tongyi-MAI/Z-Image-Turbo
        ZImageCLI -p "a black tiger in a bamboo forest" -m Tongyi-MAI/Z-Image
        ZImageCLI -p "a cute cat" --lora ostris/z_image_turbo_childrens_drawings
        ZImageCLI -p "cat" --enhance  # Enhanced prompt generation
      """)
  }

  private static func printUsage(for usage: UsageTopic) {
    switch usage {
    case .main:
      printUsage()
    case .quantize:
      printQuantizeUsage()
    case .quantizeControlnet:
      printQuantizeControlnetUsage()
    case .control:
      printControlUsage()
    }
  }

  private static func runQuantize(args: [String]) throws {
    var input: String?
    var output: String?
    var bits = 8
    var groupSize = 32
    var verbose = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--input", "-i":
        input = try nextValue(for: arg, iterator: &iterator, usage: .quantize)
      case "--output", "-o":
        output = try nextValue(for: arg, iterator: &iterator, usage: .quantize)
      case "--bits":
        bits = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: .quantize)
      case "--group-size":
        groupSize = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: .quantize)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        printQuantizeUsage()
        return
      default:
        throw CLIError(message: "Unknown quantize argument: \(arg)", usage: .quantize)
      }
    }

    guard let inputPath = input else {
      throw CLIError(message: "Missing required --input argument", usage: .quantize)
    }

    guard let outputPath = output else {
      throw CLIError(message: "Missing required --output argument", usage: .quantize)
    }

    let inputURL = URL(fileURLWithPath: inputPath)
    let outputURL = URL(fileURLWithPath: outputPath)
    var isDirectory: ObjCBool = false

    guard FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
      throw CLIError(message: "Input model directory not found: \(inputPath)", usage: .quantize)
    }

    guard ZImageQuantizer.supportedBits.contains(bits) else {
      throw CLIError(message: "Invalid bits: \(bits). Supported values: 4, 8", usage: .quantize)
    }

    guard ZImageQuantizer.supportedGroupSizes.contains(groupSize) else {
      throw CLIError(
        message: "Invalid group size: \(groupSize). Supported values: 32, 64, 128",
        usage: .quantize
      )
    }

    let spec = ZImageQuantizationSpec(groupSize: groupSize, bits: bits, mode: .affine)

    print("Quantizing: \(inputPath) -> \(outputPath)")
    print("Config: \(bits)-bit, group_size=\(groupSize)")

    try ZImageQuantizer.quantizeAndSave(
      from: inputURL,
      to: outputURL,
      spec: spec,
      verbose: verbose
    )

    print("Done: \(outputURL.path)")
  }

  private static func printQuantizeUsage() {
    print(
      """
      Quantize model weights.

      Usage: ZImageCLI quantize -i <input> -o <output> [options]
        --input, -i          Input model directory (required)
        --output, -o         Output directory (required)
        --bits               Bit width: 4 or 8 (default: 8)
        --group-size         Group size: 32, 64, 128 (default: 32)
        --verbose            Show progress
        --help, -h           Show help

      Example:
        ZImageCLI quantize -i models/z-image-turbo -o models/z-image-turbo-q8 --verbose
      """)
  }

  private static func runQuantizeControlnet(args: [String]) throws {
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
        input = try nextValue(for: arg, iterator: &iterator, usage: .quantizeControlnet)
      case "--output", "-o":
        output = try nextValue(for: arg, iterator: &iterator, usage: .quantizeControlnet)
      case "--file", "-f":
        specificFile = try nextValue(for: arg, iterator: &iterator, usage: .quantizeControlnet)
      case "--bits":
        bits = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: .quantizeControlnet)
      case "--group-size":
        groupSize = try intValue(for: arg, iterator: &iterator, minimum: 1, usage: .quantizeControlnet)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        printQuantizeControlnetUsage()
        return
      default:
        throw CLIError(message: "Unknown quantize-controlnet argument: \(arg)", usage: .quantizeControlnet)
      }
    }

    guard let inputPath = input else {
      throw CLIError(message: "Missing required --input argument", usage: .quantizeControlnet)
    }

    guard let outputPath = output else {
      throw CLIError(message: "Missing required --output argument", usage: .quantizeControlnet)
    }

    guard ZImageQuantizer.supportedBits.contains(bits) else {
      throw CLIError(message: "Invalid bits: \(bits). Supported values: 4, 8", usage: .quantizeControlnet)
    }

    guard ZImageQuantizer.supportedGroupSizes.contains(groupSize) else {
      throw CLIError(
        message: "Invalid group size: \(groupSize). Supported values: 32, 64, 128",
        usage: .quantizeControlnet
      )
    }

    let outputURL = URL(fileURLWithPath: outputPath)
    let spec = ZImageQuantizationSpec(groupSize: groupSize, bits: bits, mode: .affine)

    print("Quantizing ControlNet: \(inputPath) -> \(outputPath)")
    print("Config: \(bits)-bit, group_size=\(groupSize)")

    let semaphore = DispatchSemaphore(value: 0)
    let errorBox = Box<Error?>(nil)
    let capturedVerbose = verbose
    let capturedSpecificFile = specificFile

    Task {
      do {
        let sourceURL: URL
        let localURL = URL(fileURLWithPath: inputPath)

        if FileManager.default.fileExists(atPath: localURL.path) {
          sourceURL = localURL
          if capturedVerbose {
            print("Using local ControlNet: \(inputPath)")
          }
        } else if ModelResolution.isHuggingFaceModelId(inputPath) {
          if capturedVerbose {
            print("Downloading ControlNet from HuggingFace: \(inputPath)")
          }
          sourceURL = try await ModelResolution.resolve(
            modelSpec: inputPath,
            filePatterns: ["*.safetensors", "*.json"],
            progressHandler: { progress in
              let percent = Int(progress.fractionCompleted * 100)
              print("Downloading: \(percent)%")
            }
          )
          if capturedVerbose {
            print("Downloaded to: \(sourceURL.path)")
          }
        } else {
          throw CLIError(
            message: "Input not found: \(inputPath). Provide a local path or HuggingFace model ID.",
            usage: .quantizeControlnet
          )
        }

        try ZImageQuantizer.quantizeControlnet(
          from: sourceURL,
          to: outputURL,
          spec: spec,
          specificFile: capturedSpecificFile,
          verbose: capturedVerbose
        )

        print("Done: \(outputURL.path)")
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

  private static func printQuantizeControlnetUsage() {
    print(
      """
      Quantize ControlNet weights.

      Usage: ZImageCLI quantize-controlnet -i <input> -o <output> [options]
        --input, -i          Input ControlNet path or HuggingFace ID (required)
        --output, -o         Output directory (required)
        --file, -f           Specific .safetensors file to quantize (optional)
        --bits               Bit width: 4 or 8 (default: 8)
        --group-size         Group size: 32, 64, 128 (default: 32)
        --verbose            Show progress
        --help, -h           Show help

      Examples:
        # From HuggingFace
        ZImageCLI quantize-controlnet -i alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
          --file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors -o controlnet-2.1-q8 --verbose

        # From local directory
        ZImageCLI quantize-controlnet -i ./controlnet-union -o ./controlnet-union-q8 --verbose
      """)
  }

  // swiftlint:disable:next cyclomatic_complexity
  private static func runControl(args: [String]) throws {
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
    var noProgress = false
    var logControlMemory = false

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
      case "--log-control-memory":
        logControlMemory = true
      case "--no-progress":
        noProgress = true
      case "--help", "-h":
        printControlUsage()
        return
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

    let preset = ZImagePreset.resolved(
      for: model,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      maxSequenceLength: maxSequenceLength
    )
    let resolvedNegativePrompt = negativePrompt ?? preset.negativePrompt

    var controlImageURL: URL? = nil
    if let controlImage {
      controlImageURL = URL(fileURLWithPath: controlImage)
      guard FileManager.default.fileExists(atPath: controlImageURL!.path) else {
        throw CLIError(message: "Control image not found: \(controlImage)", usage: .control)
      }
    }
    var inpaintImageURL: URL? = nil
    if let inpaintImage {
      inpaintImageURL = URL(fileURLWithPath: inpaintImage)
      guard FileManager.default.fileExists(atPath: inpaintImageURL!.path) else {
        throw CLIError(message: "Inpaint image not found: \(inpaintImage)", usage: .control)
      }
    }
    var maskImageURL: URL? = nil
    if let maskImage {
      maskImageURL = URL(fileURLWithPath: maskImage)
      guard FileManager.default.fileExists(atPath: maskImageURL!.path) else {
        throw CLIError(message: "Mask image not found: \(maskImage)", usage: .control)
      }
    }

    if let limit = cacheLimit {
      Memory.cacheLimit = limit * 1024 * 1024
      logger.info("GPU cache limit set to \(limit)MB")
    }

    let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
    let bar = useBar ? ProgressBar(total: preset.steps) : nil
    let barBox = Box<ProgressBar?>(bar)
    let disableProgress = noProgress
    let progressCallback: ControlProgressCallback?
    if disableProgress {
      progressCallback = nil
    } else {
      progressCallback = { progress in
        guard progress.stage == "Denoising" else { return }
        let completed = min(progress.totalSteps, max(0, progress.stepIndex))
        if let bar = barBox.value {
          bar.update(completed: completed)
          if completed == progress.totalSteps {
            bar.finish(forceNewline: true)
          }
        } else {
          PlainProgress.shared.report(completed: completed, total: progress.totalSteps)
        }
      }
    }

    let request = ZImageControlGenerationRequest(
      prompt: prompt,
      negativePrompt: resolvedNegativePrompt,
      controlImage: controlImageURL,
      inpaintImage: inpaintImageURL,
      maskImage: maskImageURL,
      controlContextScale: controlScale,
      width: preset.width,
      height: preset.height,
      steps: preset.steps,
      guidanceScale: preset.guidanceScale,
      cfgNormalization: cfgNormalization,
      cfgTruncation: cfgTruncation,
      seed: seed,
      outputPath: URL(fileURLWithPath: outputPath),
      model: model,
      weightsVariant: weightsVariant,
      controlnetWeights: controlnetWeights,
      controlnetWeightsFile: controlnetWeightsFile,
      maxSequenceLength: preset.maxSequenceLength,
      progressCallback: progressCallback,
      runtimeOptions: .init(logPhaseMemory: logControlMemory)
    )

    let pipeline = ZImageControlPipeline(logger: logger)
    let semaphore = DispatchSemaphore(value: 0)
    let errorBox = Box<Error?>(nil)
    let finalOutputPath = outputPath
    Task {
      do {
        _ = try await pipeline.generate(request)
        logger.info("Output saved to: \(finalOutputPath)")
      } catch {
        errorBox.value = error
      }
      if let bar = barBox.value { bar.finish(forceNewline: true) }
      semaphore.signal()
    }
    semaphore.wait()
    if let error = errorBox.value {
      throw error
    }
  }

  private static func printControlUsage() {
    print(
      """
      Generate images with ControlNet conditioning (supports v2.0/v2.1 with inpainting).

      Usage: ZImageCLI control --prompt "text" --controlnet-weights <path> [options]
        --prompt, -p              Text prompt (required)
        --negative-prompt, --np   Negative prompt
        --control-image, -c       Control image path - Canny, HED, Depth, Pose, or MLSD
        --inpaint-image, -i       Source image for inpainting (v2.0+)
        --mask, --mask-image      Mask image for inpainting (white=fill, black=preserve)
        --control-scale, --cs     Control context scale (default: 0.75, recommended: 0.65-0.90)
        --controlnet-weights, --cw Path to controlnet safetensors or HuggingFace ID (required)
        --control-file, --cf      Specific safetensors filename within repo (e.g., "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors")
        --width, -W               Output width (default \(ZImageModelMetadata.recommendedWidth))
        --height, -H              Output height (default \(ZImageModelMetadata.recommendedHeight))
                                 Width and height must be >= \(minimumImageDimension) and divisible by \(requiredImageDimensionMultiple).
        --steps, -s               Inference steps (default: model-aware, 9 for Turbo / 50 for Base)
        --guidance, -g            Guidance scale (default: model-aware, 0.0 for Turbo / 4.0 for Base)
                                 Steps count literal denoising iterations / transformer forwards.
        --cfg-normalization       Clamp CFG output norm to the positive-branch norm
        --cfg-truncation          Disable CFG after normalized timestep exceeds this value (default: 1.0)
        --seed                    Random seed
        --output, -o              Output path (default z-image-control.png)
        --model, -m               Model path or HuggingFace ID (default: \(ZImageRepository.id))
        --weights-variant         Weights precision variant (e.g. fp16, bf16)
        --cache-limit             GPU memory cache limit in MB (default: unlimited)
        --max-sequence-length     Maximum sequence length for text encoding (default: 512)
        --log-control-memory      Emit resident and MLX memory markers for control-path phases
        --no-progress             Disable progress output
        --help, -h                Show help

      Note: At least one of --control-image, --inpaint-image, or --mask must be provided.
      Known Tongyi-MAI ids, inspectable local or cached snapshots, and common Z-Image aliases apply model-aware presets. Unrecognized models still keep the Turbo-compatible preset unless you override the sampling flags.

      Control Types:
        The control image should be pre-processed according to the control type:
        - Canny: Edge detection output (white edges on black background)
        - HED: Holistically-nested edge detection output
        - Depth: Depth map (grayscale, closer=brighter or depth estimation output)
        - Pose: OpenPose/DWPose skeleton visualization
        - MLSD: Line segment detection output

      Examples:
        # T2I with pose control using v2.1 weights (recommended)
        ZImageCLI control -p "a woman on a beach" -c pose.jpg \\
          --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
          --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors

        # I2I inpainting with pose control
        ZImageCLI control -p "a dancer" -c pose.jpg -i photo.jpg --mask mask.png \\
          --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
          --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors --cs 0.75 -s 25

        # Inpainting without control guidance
        ZImageCLI control -p "a cat sitting" -i photo.jpg --mask mask.png \\
          --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
          --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors

        # Using local controlnet weights
        ZImageCLI control -p "a forest path" -c depth.jpg --cs 0.7 \\
          --cw ./controlnet-q8 -o forest.png
      """)
  }

  private static func nextValue(
    for arg: String,
    iterator: inout IndexingIterator<[String]>,
    usage: UsageTopic
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
    usage: UsageTopic
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
    usage: UsageTopic
  ) throws -> Int {
    let value = try intValue(
      for: arg,
      iterator: &iterator,
      minimum: minimumImageDimension,
      usage: usage
    )
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
    usage: UsageTopic
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
    usage: UsageTopic
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

  private static func describe(_ error: Error) -> String {
    if let localizedError = error as? LocalizedError,
      let message = localizedError.errorDescription
    {
      return message
    }
    return String(describing: error)
  }
}

// MARK: - Progress Helpers

private final class PlainProgress: @unchecked Sendable {
  static let shared = PlainProgress()
  private let lock = NSLock()
  private var lastPercent: Int = -1
  private var lastEmitTime: Date = .distantPast

  func report(completed: Int, total: Int) {
    guard total > 0 else { return }
    let now = Date()
    let percent = Int((Double(completed) / Double(total)) * 100.0)
    let shouldEmit: Bool
    lock.lock()
    if percent != lastPercent || now.timeIntervalSince(lastEmitTime) >= 0.5 {
      lastPercent = percent
      lastEmitTime = now
      shouldEmit = true
    } else {
      shouldEmit = false
    }
    lock.unlock()

    guard shouldEmit else { return }
    FileHandle.standardError.write("Step \(completed)/\(total) (\(percent)%)\n".data(using: .utf8)!)
  }
}

private final class ProgressBar {
  private let total: Int
  private var lastStepTime: Date?
  private var postWarmupDurations: [Double] = []
  private let windowSize: Int = 5
  private var lastRenderedPercent: Int = -1
  private var isFinished: Bool = false

  init(total: Int) {
    self.total = max(1, total)
  }

  func update(completed: Int) {
    if isFinished { return }
    let now = Date()
    if let last = lastStepTime {
      let dt = now.timeIntervalSince(last)
      postWarmupDurations.append(dt)
      if postWarmupDurations.count > windowSize { postWarmupDurations.removeFirst() }
    }
    lastStepTime = now

    let percent = Int((Double(completed) / Double(total)) * 100.0)
    if percent == lastRenderedPercent { return }
    lastRenderedPercent = percent

    let remaining = max(0, total - completed)
    var etaSeconds: Double? = nil
    if !postWarmupDurations.isEmpty {
      let avg = postWarmupDurations.reduce(0, +) / Double(postWarmupDurations.count)
      etaSeconds = avg * Double(remaining)
    }

    let barWidth = 28
    let filled = Int((Double(completed) / Double(total)) * Double(barWidth))
    let lead = (completed < total) ? ">" : "="
    let tailCount = max(0, barWidth - max(1, filled))
    let bar = String(repeating: "=", count: max(0, filled - 1)) + lead + String(repeating: "-", count: tailCount)

    let etaStr: String
    if let eta = etaSeconds { etaStr = format(seconds: eta) } else { etaStr = "estimating..." }

    let prefix = "\r\u{001B}[2K"
    let line = String(format: "[%@] %3d%%  %d/%d  ETA %@", bar, percent, completed, total, etaStr)
    if let data = (prefix + line).data(using: .utf8) {
      FileHandle.standardError.write(data)
      fflush(stderr)
    }
  }

  func finish(forceNewline: Bool = true) {
    if isFinished { return }
    isFinished = true
    if let data = "\r\u{001B}[2K".data(using: .utf8) {
      FileHandle.standardError.write(data)
    }
    if forceNewline, let nl = "\n".data(using: .utf8) {
      FileHandle.standardError.write(nl)
    }
    fflush(stderr)
  }

  private func format(seconds: Double) -> String {
    var s = Int(seconds.rounded())
    let h = s / 3600
    s %= 3600
    let m = s / 60
    s %= 60
    if h > 0 { return String(format: "%dh%02dm%02ds", h, m, s) }
    if m > 0 { return String(format: "%dm%02ds", m, s) }
    return String(format: "%ds", s)
  }
}

ZImageCLI.main()
