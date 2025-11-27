import Foundation

public struct ZImageTransformerConfig: Decodable {
  public let inChannels: Int
  public let dim: Int
  public let nLayers: Int
  public let nRefinerLayers: Int
  public let nHeads: Int
  public let nKVHeads: Int
  public let normEps: Float
  public let qkNorm: Bool
  public let capFeatDim: Int
  public let ropeTheta: Float
  public let tScale: Float
  public let axesDims: [Int]
  public let axesLens: [Int]

  enum CodingKeys: String, CodingKey {
    case inChannels = "in_channels"
    case dim
    case nLayers = "n_layers"
    case nRefinerLayers = "n_refiner_layers"
    case nHeads = "n_heads"
    case nKVHeads = "n_kv_heads"
    case normEps = "norm_eps"
    case qkNorm = "qk_norm"
    case capFeatDim = "cap_feat_dim"
    case ropeTheta = "rope_theta"
    case tScale = "t_scale"
    case axesDims = "axes_dims"
    case axesLens = "axes_lens"
  }
}

public struct ZImageVAEConfig: Decodable {
  public let blockOutChannels: [Int]
  public let latentChannels: Int
  public let scalingFactor: Float
  public let shiftFactor: Float
  public let sampleSize: Int
  public let inChannels: Int
  public let outChannels: Int
  public let layersPerBlock: Int
  public let normNumGroups: Int
  public let midBlockAddAttention: Bool
  public let usePostQuantConv: Bool?
  public let useQuantConv: Bool?

  enum CodingKeys: String, CodingKey {
    case blockOutChannels = "block_out_channels"
    case latentChannels = "latent_channels"
    case scalingFactor = "scaling_factor"
    case shiftFactor = "shift_factor"
    case sampleSize = "sample_size"
    case inChannels = "in_channels"
    case outChannels = "out_channels"
    case layersPerBlock = "layers_per_block"
    case normNumGroups = "norm_num_groups"
    case midBlockAddAttention = "mid_block_add_attention"
    case usePostQuantConv = "use_post_quant_conv"
    case useQuantConv = "use_quant_conv"
  }

  public var vaeScaleFactor: Int {
    max(1, 1 << max(0, blockOutChannels.count - 1))
  }

  public var latentDivisor: Int {
    vaeScaleFactor  // 8 for Z-Image-Turbo (4 downsampling stages with factor 2 each)
  }
}

public struct ZImageSchedulerConfig: Decodable {
  public let numTrainTimesteps: Int
  public let shift: Float
  public let useDynamicShifting: Bool
  public let baseShift: Float?
  public let maxShift: Float?
  public let baseImageSeqLen: Int?
  public let maxImageSeqLen: Int?

  enum CodingKeys: String, CodingKey {
    case numTrainTimesteps = "num_train_timesteps"
    case shift
    case useDynamicShifting = "use_dynamic_shifting"
    case baseShift = "base_shift"
    case maxShift = "max_shift"
    case baseImageSeqLen = "base_image_seq_len"
    case maxImageSeqLen = "max_image_seq_len"
  }
}

public struct ZImageTextEncoderConfig: Decodable {
  public let hiddenSize: Int
  public let numHiddenLayers: Int
  public let numAttentionHeads: Int
  public let numKeyValueHeads: Int
  public let intermediateSize: Int
  public let maxPositionEmbeddings: Int
  public let ropeTheta: Float
  public let vocabSize: Int
  public let rmsNormEps: Float
  public let headDim: Int

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case intermediateSize = "intermediate_size"
    case maxPositionEmbeddings = "max_position_embeddings"
    case ropeTheta = "rope_theta"
    case vocabSize = "vocab_size"
    case rmsNormEps = "rms_norm_eps"
    case headDim = "head_dim"
  }
}

public struct ZImageModelConfigs {
  public let transformer: ZImageTransformerConfig
  public let vae: ZImageVAEConfig
  public let scheduler: ZImageSchedulerConfig
  public let textEncoder: ZImageTextEncoderConfig

  public static func load(from snapshot: URL) throws -> ZImageModelConfigs {
    let decoder = JSONDecoder()
    func loadJSON<T: Decodable>(_ relativePath: String, as type: T.Type) throws -> T {
      let url = snapshot.appending(path: relativePath)
      let data = try Data(contentsOf: url)
      return try decoder.decode(T.self, from: data)
    }

    let transformer = try loadJSON(ZImageFiles.transformerConfig, as: ZImageTransformerConfig.self)
    let vae = try loadJSON(ZImageFiles.vaeConfig, as: ZImageVAEConfig.self)
    let scheduler = try loadJSON(ZImageFiles.schedulerConfig, as: ZImageSchedulerConfig.self)
    let textEncoder = try loadJSON(ZImageFiles.textEncoderConfig, as: ZImageTextEncoderConfig.self)
    return ZImageModelConfigs(transformer: transformer, vae: vae, scheduler: scheduler, textEncoder: textEncoder)
  }
}
