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
  public static let transformerWeights = [
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
  ]
  public static let transformerIndex = "transformer/diffusion_pytorch_model.safetensors.index.json"

  public static let textEncoderConfig = "text_encoder/config.json"
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
}
