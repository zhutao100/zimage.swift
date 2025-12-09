import Foundation
import Dispatch
import Logging
import Metal
import MLX
import ZImage
import Darwin

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

LoggingSystem.bootstrap { label in
  var handler = StreamLogHandler.standardError(label: label)
  handler.logLevel = .info
  return handler
}

struct ZImageCLI {
  static var logger: Logger = {
    var logger = Logger(label: "z-image.cli")
    logger.logLevel = .info
    return logger
  }()

  static func run() throws {
    if let dev = MTLCreateSystemDefaultDevice() {
      logger.info("Metal device: \(dev.name)")
    } else {
      logger.warning("No Metal device detected; MLX will fall back to CPU.")
    }

    var prompt: String?
    var negativePrompt: String?
    var width = ZImageModelMetadata.recommendedWidth
    var height = ZImageModelMetadata.recommendedHeight
    var steps = ZImageModelMetadata.recommendedInferenceSteps
    var guidance = ZImageModelMetadata.recommendedGuidanceScale
    var seed: UInt64?
    var outputPath = "z-image.png"
    var model: String?
    var cacheLimit: Int?
    var maxSequenceLength = 512
    var loraPath: String?
    var loraScale: Float = 1.0
    var enhance = false
    var enhanceMaxTokens = 512
    var noProgress = false

    let args = Array(CommandLine.arguments.dropFirst())
    var iterator = args.makeIterator()

    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p":
        prompt = nextValue(for: arg, iterator: &iterator)
      case "--negative-prompt", "--np":
        negativePrompt = nextValue(for: arg, iterator: &iterator)
      case "--width", "-W":
        width = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: width)
      case "--height", "-H":
        height = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: height)
      case "--steps", "-s":
        steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: steps)
      case "--guidance", "-g":
        guidance = floatValue(for: arg, iterator: &iterator, fallback: guidance)
      case "--seed":
        seed = UInt64(nextValue(for: arg, iterator: &iterator))
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
      case "--model", "-m":
        model = nextValue(for: arg, iterator: &iterator)
      case "--cache-limit":
        cacheLimit = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: 2048)
      case "--max-sequence-length":
        maxSequenceLength = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: 512)
      case "--lora", "-l":
        loraPath = nextValue(for: arg, iterator: &iterator)
      case "--lora-scale":
        loraScale = floatValue(for: arg, iterator: &iterator, fallback: 1.0)
      case "--enhance", "-e":
        enhance = true
      case "--enhance-max-tokens":
        enhanceMaxTokens = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: 512)
      case "--no-progress":
        noProgress = true
      case "--help", "-h":
        printUsage()
        return
      case "quantize":
        try runQuantize(args: Array(args.dropFirst()))
        return
      case "control":
        try runControl(args: Array(args.dropFirst()))
        return
      default:
        logger.warning("Unknown argument: \(arg)")
      }
    }

    guard let prompt else {
      printUsage()
      return
    }

    if let limit = cacheLimit {
      GPU.set(cacheLimit: limit * 1024 * 1024)
      logger.info("GPU cache limit set to \(limit)MB")
    }

    let request = ZImageGenerationRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      seed: seed,
      outputPath: URL(fileURLWithPath: outputPath),
      model: model,
      maxSequenceLength: maxSequenceLength,
      loraPath: loraPath,
      loraScale: loraScale,
      enhancePrompt: enhance,
      enhanceMaxTokens: enhanceMaxTokens
    )

    let pipeline = ZImagePipeline(logger: logger)
    let semaphore = DispatchSemaphore(value: 0)
    // TTY-aware progress bar
    let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
    let bar = useBar ? ProgressBar(total: steps) : nil
    let disableProgress = noProgress
    Task {
      do {
        _ = try await pipeline.generate(request, progress: { completed, total in
          if let bar {
            bar.update(completed: completed)
            if completed == total { bar.finish(forceNewline: true) }
          } else if !disableProgress {
            PlainProgress.shared.report(completed: completed, total: total)
          }
        })
        if let bar { bar.finish() }
      } catch {
        logger.error("Generation failed: \(error)")
        if let bar { bar.finish(forceNewline: true) }
      }
      semaphore.signal()
    }
    semaphore.wait()
  }

  private static func printUsage() {
    print("""
    Z-Image-Turbo Swift port

    Usage: ZImageCLI --prompt "text" [options]
      --prompt, -p           Text prompt (required)
      --negative-prompt      Negative prompt
      --width, -W            Output width (default \(ZImageModelMetadata.recommendedWidth))
      --height, -H           Output height (default \(ZImageModelMetadata.recommendedHeight))
      --steps, -s            Inference steps (default \(ZImageModelMetadata.recommendedInferenceSteps))
      --guidance, -g         Guidance scale (default \(ZImageModelMetadata.recommendedGuidanceScale))
      --seed                 Random seed
      --output, -o           Output path (default z-image.png)
      --model, -m            Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --cache-limit          GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length  Maximum sequence length for text encoding (default: 512)
      --lora, -l             LoRA weights path or HuggingFace ID
      --lora-scale           LoRA scale factor (default: 1.0)
      --enhance, -e          Enhance prompt using LLM before generation
      --enhance-max-tokens   Max tokens for prompt enhancement (default: 512)
      --no-progress          Disable progress bar output
      --help, -h             Show help

    Subcommands:
      quantize               Quantize model weights
        --input, -i          Input model directory (required)
        --output, -o         Output directory (required)
        --bits               Bit width: 4 or 8 (default: 8)
        --group-size         Group size: 32, 64, 128 (default: 32)
        --verbose            Show progress

      control                Generate with ControlNet conditioning
        --prompt, -p         Text prompt (required)
        --control-image, -c  Control image path (required)
        --controlnet-weights Path to controlnet safetensors (required)
        --control-scale      Control scale (default: 0.75)
        Use 'ZImageCLI control --help' for full options

    Examples:
      ZImageCLI -p "a cute cat" -o cat.png
      ZImageCLI -p "a sunset" -m models/z-image-turbo-q8
      ZImageCLI -p "a forest" -m Tongyi-MAI/Z-Image-Turbo
      ZImageCLI -p "a cut a cat" --lora ostris/z_image_turbo_childrens_drawings
      ZImageCLI -p "a cute cat" --enhance -o enhanced_cat.png
    """)
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
        input = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        output = nextValue(for: arg, iterator: &iterator)
      case "--bits":
        bits = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: bits)
      case "--group-size":
        groupSize = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: groupSize)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        printQuantizeUsage()
        return
      default:
        logger.warning("Unknown quantize argument: \(arg)")
      }
    }

    guard let inputPath = input else {
      logger.error("Missing required --input argument")
      printQuantizeUsage()
      return
    }

    guard let outputPath = output else {
      logger.error("Missing required --output argument")
      printQuantizeUsage()
      return
    }

    let inputURL = URL(fileURLWithPath: inputPath)
    let outputURL = URL(fileURLWithPath: outputPath)

    guard FileManager.default.fileExists(atPath: inputURL.path) else {
      logger.error("Input directory not found: \(inputPath)")
      return
    }

    guard ZImageQuantizer.supportedBits.contains(bits) else {
      logger.error("Invalid bits: \(bits). Supported: 4, 8")
      return
    }

    guard ZImageQuantizer.supportedGroupSizes.contains(groupSize) else {
      logger.error("Invalid group size: \(groupSize). Supported: 32, 64, 128")
      return
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
    print("""
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

  private static func runControl(args: [String]) throws {
    var prompt: String?
    var negativePrompt: String?
    var controlImage: String?
    var controlScale: Float = 0.75
    var controlnetWeights: String?
    var width = ZImageModelMetadata.recommendedWidth
    var height = ZImageModelMetadata.recommendedHeight
    var steps = ZImageModelMetadata.recommendedInferenceSteps
    var guidance = ZImageModelMetadata.recommendedGuidanceScale
    var seed: UInt64?
    var outputPath = "z-image-control.png"
    var model: String?
    var cacheLimit: Int?
    var maxSequenceLength = 512
    var noProgress = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p":
        prompt = nextValue(for: arg, iterator: &iterator)
      case "--negative-prompt", "--np":
        negativePrompt = nextValue(for: arg, iterator: &iterator)
      case "--control-image", "-c":
        controlImage = nextValue(for: arg, iterator: &iterator)
      case "--control-scale", "--cs":
        controlScale = floatValue(for: arg, iterator: &iterator, fallback: 0.75)
      case "--controlnet-weights", "--cw":
        controlnetWeights = nextValue(for: arg, iterator: &iterator)
      case "--width", "-W":
        width = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: width)
      case "--height", "-H":
        height = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: height)
      case "--steps", "-s":
        steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: steps)
      case "--guidance", "-g":
        guidance = floatValue(for: arg, iterator: &iterator, fallback: guidance)
      case "--seed":
        seed = UInt64(nextValue(for: arg, iterator: &iterator))
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
      case "--model", "-m":
        model = nextValue(for: arg, iterator: &iterator)
      case "--cache-limit":
        cacheLimit = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: 2048)
      case "--max-sequence-length":
        maxSequenceLength = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: 512)
      case "--no-progress":
        noProgress = true
      case "--help", "-h":
        printControlUsage()
        return
      default:
        logger.warning("Unknown control argument: \(arg)")
      }
    }

    guard let prompt else {
      logger.error("Missing required --prompt argument")
      printControlUsage()
      return
    }

    guard let controlImage else {
      logger.error("Missing required --control-image argument")
      printControlUsage()
      return
    }

    guard let controlnetWeights else {
      logger.error("Missing required --controlnet-weights argument")
      printControlUsage()
      return
    }

    let controlImageURL = URL(fileURLWithPath: controlImage)
    guard FileManager.default.fileExists(atPath: controlImageURL.path) else {
      logger.error("Control image not found: \(controlImage)")
      return
    }

    if let limit = cacheLimit {
      GPU.set(cacheLimit: limit * 1024 * 1024)
      logger.info("GPU cache limit set to \(limit)MB")
    }

    let request = ZImageControlGenerationRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      controlImage: controlImageURL,
      controlContextScale: controlScale,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      seed: seed,
      outputPath: URL(fileURLWithPath: outputPath),
      model: model,
      controlnetWeights: controlnetWeights,
      maxSequenceLength: maxSequenceLength
    )

    let pipeline = ZImageControlPipeline(logger: logger)
    let semaphore = DispatchSemaphore(value: 0)
    let finalOutputPath = outputPath
    // TTY-aware progress bar
    let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
    let bar = useBar ? ProgressBar(total: steps) : nil
    let disableProgress = noProgress
    Task {
      do {
        _ = try await pipeline.generate(request, progress: { completed, total in
          if let bar {
            bar.update(completed: completed)
            if completed == total { bar.finish(forceNewline: true) }
          } else if !disableProgress {
            PlainProgress.shared.report(completed: completed, total: total)
          }
        })
        if let bar { bar.finish() }
        logger.info("Output saved to: \(finalOutputPath)")
      } catch {
        logger.error("Control generation failed: \(error)")
        if let bar { bar.finish(forceNewline: true) }
      }
      semaphore.signal()
    }
    semaphore.wait()
  }

  private static func printControlUsage() {
    print("""
    Generate images with ControlNet conditioning.

    Usage: ZImageCLI control --prompt "text" --control-image <path> --controlnet-weights <path> [options]
      --prompt, -p              Text prompt (required)
      --negative-prompt, --np   Negative prompt
      --control-image, -c       Control image path (required) - Canny, HED, Depth, Pose, or MLSD
      --control-scale, --cs     Control context scale (default: 0.75, recommended: 0.65-0.80)
      --controlnet-weights, --cw Path to controlnet safetensors file (required)
      --width, -W               Output width (default \(ZImageModelMetadata.recommendedWidth))
      --height, -H              Output height (default \(ZImageModelMetadata.recommendedHeight))
      --steps, -s               Inference steps (default \(ZImageModelMetadata.recommendedInferenceSteps))
      --guidance, -g            Guidance scale (default \(ZImageModelMetadata.recommendedGuidanceScale))
      --seed                    Random seed
      --output, -o              Output path (default z-image-control.png)
      --model, -m               Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --cache-limit             GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length     Maximum sequence length for text encoding (default: 512)
      --no-progress             Disable progress bar output
      --help, -h                Show help

    Control Types:
      The control image should be pre-processed according to the control type:
      - Canny: Edge detection output (white edges on black background)
      - HED: Holistically-nested edge detection output
      - Depth: Depth map (grayscale, closer=brighter or depth estimation output)
      - Pose: OpenPose skeleton visualization
      - MLSD: Line segment detection output

    Examples:
      ZImageCLI control -p "a woman on a beach" -c pose.jpg --cw Z-Image-Turbo-Fun-Controlnet-Union.safetensors
      ZImageCLI control -p "a forest path" -c depth.jpg --cs 0.7 --cw controlnet.safetensors -o forest.png
    """)
  }

  private static func nextValue(for arg: String, iterator: inout IndexingIterator<[String]>) -> String {
    guard let value = iterator.next() else {
      fatalError("Expected value after \(arg)")
    }
    return value
  }

  private static func intValue(for arg: String, iterator: inout IndexingIterator<[String]>, minimum: Int, fallback: Int) -> Int {
    guard let value = Int(nextValue(for: arg, iterator: &iterator)) else { return fallback }
    return max(minimum, value)
  }

  private static func floatValue(for arg: String, iterator: inout IndexingIterator<[String]>, fallback: Float) -> Float {
    Float(nextValue(for: arg, iterator: &iterator)) ?? fallback
  }

}

// MARK: - Progress Helpers

private final class PlainProgress {
  static let shared = PlainProgress()
  private var lastPercent: Int = -1
  private var lastEmitTime: Date = .distantPast

  func report(completed: Int, total: Int) {
    guard total > 0 else { return }
    let now = Date()
    let percent = Int((Double(completed) / Double(total)) * 100.0)
    if percent != lastPercent || now.timeIntervalSince(lastEmitTime) >= 0.5 {
      FileHandle.standardError.write("Step \(completed)/\(total) (\(percent)%)\n".data(using: .utf8)!)
      lastPercent = percent
      lastEmitTime = now
    }
  }
}

private final class ProgressBar {
  private let total: Int
  private var startTime: Date?
  private var lastStepTime: Date?
  private var postWarmupDurations: [Double] = []
  private let windowSize: Int = 5
  private var lastRenderedPercent: Int = -1
  private var isFinished: Bool = false

  init(total: Int) { self.total = max(1, total) }

  func update(completed: Int) {
    if isFinished { return }
    let now = Date()
    if startTime == nil { startTime = now }

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
    let bar = String(repeating: "=", count: max(0, filled - 1)) + lead + String(repeating: "-", count: max(0, barWidth - filled))

    let etaStr: String
    if let eta = etaSeconds { etaStr = format(seconds: eta) } else { etaStr = "estimating..." }

    // Clear entire line, return carriage, then render
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
    // Clear line and move to new line so logs don't overlay
    if let data = ("\r\u{001B}[2K").data(using: .utf8) {
      FileHandle.standardError.write(data)
    }
    if forceNewline, let nl = "\n".data(using: .utf8) {
      FileHandle.standardError.write(nl)
    }
    fflush(stderr)
  }

  private func format(seconds: Double) -> String {
    var s = Int(seconds.rounded())
    let hours = s / 3600
    s %= 3600
    let minutes = s / 60
    s %= 60
    if hours > 0 { return String(format: "%dh %02dm %02ds", hours, minutes, s) }
    if minutes > 0 { return String(format: "%dm %02ds", minutes, s) }
    return String(format: "%ds", s)
  }
}

try ZImageCLI.run()
