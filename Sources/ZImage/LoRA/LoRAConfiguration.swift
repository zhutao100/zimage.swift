import Foundation
@preconcurrency import MLX

public enum LoRASource: Sendable, Equatable {
    case local(URL)
    case huggingFace(modelId: String, filename: String?)

    public var displayName: String {
        switch self {
        case .local(let url):
            return url.lastPathComponent
        case .huggingFace(let modelId, _):
            return modelId.components(separatedBy: "/").last ?? modelId
        }
    }

    public var isLocal: Bool {
        if case .local = self { return true }
        return false
    }
}

public struct LoRAConfiguration: Sendable, Equatable {
    public let source: LoRASource
    public let scale: Float

    public init(source: LoRASource, scale: Float = 1.0) {
        self.source = source
        self.scale = max(0.0, min(1.0, scale))
    }

    public static func local(_ path: String, scale: Float = 1.0) -> LoRAConfiguration {
        LoRAConfiguration(source: .local(URL(fileURLWithPath: path)), scale: scale)
    }

    public static func local(_ url: URL, scale: Float = 1.0) -> LoRAConfiguration {
        LoRAConfiguration(source: .local(url), scale: scale)
    }

    public static func huggingFace(_ modelId: String, filename: String? = nil, scale: Float = 1.0) -> LoRAConfiguration {
        LoRAConfiguration(source: .huggingFace(modelId: modelId, filename: filename), scale: scale)
    }
}
public struct LoRAWeights: @unchecked Sendable {
    public let weights: [String: (down: MLXArray, up: MLXArray)]
    public let rank: Int
    public let alpha: Float

    public init(weights: [String: (down: MLXArray, up: MLXArray)], rank: Int, alpha: Float? = nil) {
        self.weights = weights
        self.rank = rank
        self.alpha = alpha ?? Float(rank)
    }

    public var effectiveScale: Float {
        guard rank > 0 else { return 1.0 }
        return alpha / Float(rank)
    }

    public var layerCount: Int {
        weights.count
    }
}

public enum LoRAError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidFormat(String)
    case incompatibleWeights(String)
    case downloadFailed(String, Error)
    case noSafetensorsFound(URL)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "LoRA file not found: \(path)"
        case .invalidFormat(let message):
            return "Invalid LoRA format: \(message)"
        case .incompatibleWeights(let message):
            return "Incompatible LoRA weights: \(message)"
        case .downloadFailed(let modelId, let error):
            return "Failed to download LoRA '\(modelId)': \(error.localizedDescription)"
        case .noSafetensorsFound(let url):
            return "No .safetensors files found in \(url.path)"
        }
    }
}

public struct LoRAValidationResult: Sendable, Equatable {
    public let isValid: Bool
    public let rank: Int
    public let targetLayers: [String]
    public let estimatedMemoryMB: Int
    public let errorMessage: String?

    public init(isValid: Bool, rank: Int, targetLayers: [String], estimatedMemoryMB: Int, errorMessage: String? = nil) {
        self.isValid = isValid
        self.rank = rank
        self.targetLayers = targetLayers
        self.estimatedMemoryMB = estimatedMemoryMB
        self.errorMessage = errorMessage
    }

    public static func invalid(_ message: String) -> LoRAValidationResult {
        LoRAValidationResult(isValid: false, rank: 0, targetLayers: [], estimatedMemoryMB: 0, errorMessage: message)
    }
}
