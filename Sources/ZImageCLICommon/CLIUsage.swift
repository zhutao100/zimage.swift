import ZImage

public enum CLIUsageFormatter {
  private static let minimumImageDimension = 64
  private static let requiredImageDimensionMultiple = 16

  public static func usage(for topic: CLIUsageTopic, program: CLIProgramKind) -> String {
    switch topic {
    case .main:
      mainUsage(program: program)
    case .quantize:
      quantizeUsage(program: program)
    case .quantizeControlnet:
      quantizeControlnetUsage(program: program)
    case .control:
      controlUsage(program: program)
    case .serve:
      serveUsage()
    }
  }

  public static func mainUsage(program: CLIProgramKind) -> String {
    let executable = program.executableName
    if program == .serve {
      return """
        Z-Image staging service client

        Usage:
          \(executable) serve [options]
          \(executable) --prompt "text" [generation options]
          \(executable) control --prompt "text" --controlnet-weights <path> [options]
          \(executable) quantize -i <input> -o <output> [options]
          \(executable) quantize-controlnet -i <input> -o <output> [options]

        Notes:
          - text-to-image and control requests use the same flags as ZImageCLI and are submitted to the local daemon
          - quantize commands still run locally as one-shot commands
          - use '\(executable) serve --help' for daemon options

        Examples:
          \(executable) serve
          \(executable) -p "a mountain lake at sunrise" -o lake.png
          \(executable) control -p "a dancer" -c pose.png --cw ./controlnet -o dancer.png
      """
    }

    return """
      Z-Image Swift CLI

      Usage: \(executable) --prompt "text" [options]
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
          Use '\(executable) control --help' for full options

      Examples:
        \(executable) -p "a cute cat" -o cat.png
        \(executable) -p "a sunset" -m models/z-image-turbo-q8
        \(executable) -p "a forest" -m Tongyi-MAI/Z-Image-Turbo
        \(executable) -p "a black tiger in a bamboo forest" -m Tongyi-MAI/Z-Image
        \(executable) -p "a cute cat" --lora ostris/z_image_turbo_childrens_drawings
        \(executable) -p "cat" --enhance  # Enhanced prompt generation
      """
  }

  public static func quantizeUsage(program: CLIProgramKind) -> String {
    """
      Quantize model weights.

      Usage: \(program.executableName) quantize -i <input> -o <output> [options]
        --input, -i          Input model directory (required)
        --output, -o         Output directory (required)
        --bits               Bit width: 4 or 8 (default: 8)
        --group-size         Group size: 32, 64, 128 (default: 32)
        --verbose            Show progress
        --help, -h           Show help

      Example:
        \(program.executableName) quantize -i models/z-image-turbo -o models/z-image-turbo-q8 --verbose
      """
  }

  public static func quantizeControlnetUsage(program: CLIProgramKind) -> String {
    """
      Quantize ControlNet weights.

      Usage: \(program.executableName) quantize-controlnet -i <input> -o <output> [options]
        --input, -i          Input ControlNet path or HuggingFace ID (required)
        --output, -o         Output directory (required)
        --file, -f           Specific .safetensors file to quantize (optional)
        --bits               Bit width: 4 or 8 (default: 8)
        --group-size         Group size: 32, 64, 128 (default: 32)
        --verbose            Show progress
        --help, -h           Show help

      Examples:
        # From HuggingFace
        \(program.executableName) quantize-controlnet -i alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
          --file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors -o controlnet-2.1-q8 --verbose

        # From local directory
        \(program.executableName) quantize-controlnet -i ./controlnet-union -o ./controlnet-union-q8 --verbose
      """
  }

  public static func controlUsage(program: CLIProgramKind) -> String {
    let executable = program.executableName
    return """
      Generate images with ControlNet conditioning (supports v2.0/v2.1 with inpainting).

      Usage: \(executable) control --prompt "text" --controlnet-weights <path> [options]
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
        --lora, -l                LoRA weights path or HuggingFace ID
        --lora-scale              LoRA scale factor (default: 1.0)
        --enhance, -e             Enhance prompt using LLM (requires ~5GB extra VRAM)
        --enhance-max-tokens      Max tokens for prompt enhancement (default: 512)
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
        \(executable) control -p "a woman on a beach" -c pose.jpg \\
          --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
          --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors

        # I2I inpainting with pose control
        \(executable) control -p "a dancer" -c pose.jpg -i photo.jpg --mask mask.png \\
          --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
          --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors --cs 0.75 -s 25

        # Inpainting without control guidance
        \(executable) control -p "a cat sitting" -i photo.jpg --mask mask.png \\
          --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
          --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors

        # Using local controlnet weights
        \(executable) control -p "a forest path" -c depth.jpg --cs 0.7 \\
          --cw ./controlnet-q8 -o forest.png

        # Depth control with a LoRA adapter
        \(executable) control -p "a modern hallway interior" -c depth.jpg \\
          --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
          --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \\
          --lora F16/z-image-turbo-flow-dpo --lora-scale 1.0 -s 9 -g 1.0
      """
  }

  public static func serveUsage() -> String {
    """
      Run the local Z-Image staging daemon.

      Usage: ZImageServe serve [options]
        --socket, -S         Unix domain socket path (default: user cache directory)
        --help, -h           Show help

      Submit jobs with the same generation flags as ZImageCLI:
        ZImageServe -p "a mountain lake at sunrise" -o lake.png
        ZImageServe control -p "a dancer" -c pose.png --cw ./controlnet -o dancer.png
      """
  }
}
