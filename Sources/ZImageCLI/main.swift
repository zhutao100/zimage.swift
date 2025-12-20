import Foundation
import Dispatch
import Logging
import Metal
import MLX
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
  init(_ value: T) { self.value = value }
}

struct ZImageCLI {
  static let logger: Logger = {
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
    var enhancePrompt = false

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
        enhancePrompt = true
      case "--help", "-h":
        printUsage()
        return
      case "quantize":
        try runQuantize(args: Array(args.dropFirst()))
        return
      case "quantize-controlnet":
        try runQuantizeControlnet(args: Array(args.dropFirst()))
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
    let loraConfig: LoRAConfiguration? = loraPath.map { path in

      if path.hasPrefix("/") || path.hasPrefix("./") || path.hasPrefix("~") {
        return .local(path, scale: loraScale)
      } else {
        return .huggingFace(path, scale: loraScale)
      }
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
      lora: loraConfig,
      enhancePrompt: enhancePrompt
    )

    let pipeline = ZImagePipeline(logger: logger)
    nonisolated(unsafe) let semaphore = DispatchSemaphore(value: 0)
    Task {
      do {
        _ = try await pipeline.generate(request)
      } catch {
        logger.error("Generation failed: \(error)")
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
      --enhance, -e          Enhance prompt using LLM (requires ~5GB extra VRAM)
      --help, -h             Show help

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
        --control-image, -c  Control image path (required)
        --controlnet-weights Path to controlnet weights dir or HF ID (required)
        --control-scale      Control scale (default: 0.75)
        Use 'ZImageCLI control --help' for full options

    Examples:
      ZImageCLI -p "a cute cat" -o cat.png
      ZImageCLI -p "a sunset" -m models/z-image-turbo-q8
      ZImageCLI -p "a forest" -m Tongyi-MAI/Z-Image-Turbo
      ZImageCLI -p "a cut a cat" --lora ostris/z_image_turbo_childrens_drawings
      ZImageCLI -p "cat" --enhance  # Enhanced prompt generation
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
        input = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        output = nextValue(for: arg, iterator: &iterator)
      case "--file", "-f":
        specificFile = nextValue(for: arg, iterator: &iterator)
      case "--bits":
        bits = intValue(for: arg, iterator: &iterator, minimum: 4, fallback: 8)
      case "--group-size":
        groupSize = intValue(for: arg, iterator: &iterator, minimum: 32, fallback: 32)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        printQuantizeControlnetUsage()
        return
      default:
        logger.warning("Unknown quantize-controlnet argument: \(arg)")
      }
    }

    guard let inputPath = input else {
      logger.error("Missing required --input argument")
      printQuantizeControlnetUsage()
      return
    }

    guard let outputPath = output else {
      logger.error("Missing required --output argument")
      printQuantizeControlnetUsage()
      return
    }

    let outputURL = URL(fileURLWithPath: outputPath)
    let spec = ZImageQuantizationSpec(groupSize: groupSize, bits: bits, mode: .affine)

    print("Quantizing ControlNet: \(inputPath) -> \(outputPath)")
    print("Config: \(bits)-bit, group_size=\(groupSize)")

    nonisolated(unsafe) let semaphore = DispatchSemaphore(value: 0)
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
          logger.error("Input not found: \(inputPath). Provide a local path or HuggingFace model ID.")
          semaphore.signal()
          return
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
        logger.error("Quantization failed: \(error)")
      }
      semaphore.signal()
    }

    semaphore.wait()
    if let error = errorBox.value {
      throw error
    }
  }

  private static func printQuantizeControlnetUsage() {
    print("""
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
      ZImageCLI quantize-controlnet -i alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0 \\
        --file Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors -o controlnet-2.1-q8 --verbose

      # From local directory
      ZImageCLI quantize-controlnet -i ./controlnet-union -o ./controlnet-union-q8 --verbose
    """)
  }

  private static func runControl(args: [String]) throws {
    var prompt: String?
    var negativePrompt: String?
    var controlImage: String?
    var inpaintImage: String?
    var maskImage: String?
    var controlScale: Float = 0.75
    var controlnetWeights: String?
    var controlnetWeightsFile: String?
    var width = ZImageModelMetadata.recommendedWidth
    var height = ZImageModelMetadata.recommendedHeight
    var steps = ZImageModelMetadata.recommendedInferenceSteps
    var guidance = ZImageModelMetadata.recommendedGuidanceScale
    var seed: UInt64?
    var outputPath = "z-image-control.png"
    var model: String?
    var cacheLimit: Int?
    var maxSequenceLength = 512

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p":
        prompt = nextValue(for: arg, iterator: &iterator)
      case "--negative-prompt", "--np":
        negativePrompt = nextValue(for: arg, iterator: &iterator)
      case "--control-image", "-c":
        controlImage = nextValue(for: arg, iterator: &iterator)
      case "--inpaint-image", "-i":
        inpaintImage = nextValue(for: arg, iterator: &iterator)
      case "--mask", "--mask-image":
        maskImage = nextValue(for: arg, iterator: &iterator)
      case "--control-scale", "--cs":
        controlScale = floatValue(for: arg, iterator: &iterator, fallback: 0.75)
      case "--controlnet-weights", "--cw":
        controlnetWeights = nextValue(for: arg, iterator: &iterator)
      case "--control-file", "--cf":
        controlnetWeightsFile = nextValue(for: arg, iterator: &iterator)
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
    if controlImage == nil && inpaintImage == nil && maskImage == nil {
      logger.error("At least one of --control-image, --inpaint-image, or --mask must be provided")
      printControlUsage()
      return
    }

    guard let controlnetWeights else {
      logger.error("Missing required --controlnet-weights argument")
      printControlUsage()
      return
    }
    var controlImageURL: URL? = nil
    if let controlImage {
      controlImageURL = URL(fileURLWithPath: controlImage)
      guard FileManager.default.fileExists(atPath: controlImageURL!.path) else {
        logger.error("Control image not found: \(controlImage)")
        return
      }
    }
    var inpaintImageURL: URL? = nil
    if let inpaintImage {
      inpaintImageURL = URL(fileURLWithPath: inpaintImage)
      guard FileManager.default.fileExists(atPath: inpaintImageURL!.path) else {
        logger.error("Inpaint image not found: \(inpaintImage)")
        return
      }
    }
    var maskImageURL: URL? = nil
    if let maskImage {
      maskImageURL = URL(fileURLWithPath: maskImage)
      guard FileManager.default.fileExists(atPath: maskImageURL!.path) else {
        logger.error("Mask image not found: \(maskImage)")
        return
      }
    }

    if let limit = cacheLimit {
      GPU.set(cacheLimit: limit * 1024 * 1024)
      logger.info("GPU cache limit set to \(limit)MB")
    }

    let request = ZImageControlGenerationRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      controlImage: controlImageURL,
      inpaintImage: inpaintImageURL,
      maskImage: maskImageURL,
      controlContextScale: controlScale,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      seed: seed,
      outputPath: URL(fileURLWithPath: outputPath),
      model: model,
      controlnetWeights: controlnetWeights,
      controlnetWeightsFile: controlnetWeightsFile,
      maxSequenceLength: maxSequenceLength
    )

    let pipeline = ZImageControlPipeline(logger: logger)
    nonisolated(unsafe) let semaphore = DispatchSemaphore(value: 0)
    let finalOutputPath = outputPath
    Task {
      do {
        _ = try await pipeline.generate(request)
        logger.info("Output saved to: \(finalOutputPath)")
      } catch {
        logger.error("Control generation failed: \(error)")
      }
      semaphore.signal()
    }
    semaphore.wait()
  }

  private static func printControlUsage() {
    print("""
    Generate images with ControlNet conditioning (supports v2.0/v2.1 with inpainting).

    Usage: ZImageCLI control --prompt "text" --controlnet-weights <path> [options]
      --prompt, -p              Text prompt (required)
      --negative-prompt, --np   Negative prompt
      --control-image, -c       Control image path - Canny, HED, Depth, Pose, or MLSD
      --inpaint-image, -i       Source image for inpainting (v2.0+)
      --mask, --mask-image      Mask image for inpainting (white=fill, black=preserve)
      --control-scale, --cs     Control context scale (default: 0.75, recommended: 0.65-0.90)
      --controlnet-weights, --cw Path to controlnet safetensors or HuggingFace ID (required)
      --control-file, --cf      Specific safetensors filename within repo (e.g., "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors")
      --width, -W               Output width (default \(ZImageModelMetadata.recommendedWidth))
      --height, -H              Output height (default \(ZImageModelMetadata.recommendedHeight))
      --steps, -s               Inference steps (default \(ZImageModelMetadata.recommendedInferenceSteps), increase for higher control scale)
      --guidance, -g            Guidance scale (default \(ZImageModelMetadata.recommendedGuidanceScale))
      --seed                    Random seed
      --output, -o              Output path (default z-image-control.png)
      --model, -m               Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --cache-limit             GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length     Maximum sequence length for text encoding (default: 512)
      --help, -h                Show help

    Note: At least one of --control-image, --inpaint-image, or --mask must be provided.

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
        --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0 \\
        --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors

      # I2I inpainting with pose control
      ZImageCLI control -p "a dancer" -c pose.jpg -i photo.jpg --mask mask.png \\
        --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0 \\
        --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors --cs 0.75 -s 25

      # Inpainting without control guidance
      ZImageCLI control -p "a cat sitting" -i photo.jpg --mask mask.png \\
        --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0 \\
        --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors

      # Using local controlnet weights
      ZImageCLI control -p "a forest path" -c depth.jpg --cs 0.7 \\
        --cw ./controlnet-q8 -o forest.png
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

try ZImageCLI.run()
