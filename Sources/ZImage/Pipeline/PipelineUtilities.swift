import Foundation
import MLX
import MLXNN

public enum PipelineUtilities {
    public enum UtilityError: Error {
        case textEncoderFailed
        case emptyEmbeddings
    }
    public static func encodePrompt(
        _ prompt: String,
        tokenizer: QwenTokenizer,
        textEncoder: QwenTextEncoder,
        maxLength: Int
    ) throws -> (embeddings: MLXArray, mask: MLXArray) {
        let encoded = try tokenizer.encodeChat(prompts: [prompt], maxLength: maxLength)
        let embeddingsList = textEncoder.encodeForZImage(
            inputIds: encoded.inputIds,
            attentionMask: encoded.attentionMask
        )

        guard let firstEmbeds = embeddingsList.first else {
            throw UtilityError.emptyEmbeddings
        }

        let embedsBatch = firstEmbeds.expandedDimensions(axis: 0)
        let mask = MLX.ones([1, firstEmbeds.dim(0)], dtype: .int32)

        return (embedsBatch, mask)
    }

    public static func decodeLatents(
        _ latents: MLXArray,
        vae: AutoencoderKL,
        height: Int,
        width: Int
    ) -> MLXArray {
        let (decoded, _) = vae.decode(latents)
        var image = decoded
        if height != decoded.dim(2) || width != decoded.dim(3) {
            var nhwc = image.transposed(0, 2, 3, 1)
            let hScale = Float(height) / Float(decoded.dim(2))
            let wScale = Float(width) / Float(decoded.dim(3))
            nhwc = MLXNN.Upsample(scaleFactor: .array([hScale, wScale]), mode: .nearest)(nhwc)
            image = nhwc.transposed(0, 3, 1, 2)
        }

        image = QwenImageIO.denormalizeFromDecoder(image)
        return MLX.clip(image, min: 0, max: 1)
    }

    public static func calculateShift(
        imageSeqLen: Int,
        baseSeqLen: Int,
        maxSeqLen: Int,
        baseShift: Float,
        maxShift: Float
    ) -> Float {
        let m = (maxShift - baseShift) / Float(maxSeqLen - baseSeqLen)
        let b = baseShift - m * Float(baseSeqLen)
        return Float(imageSeqLen) * m + b
    }

    public static func prepareSnapshot(
        model: String?,
        defaultModelId: String,
        defaultRevision: String,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let resolvedURL = try await ModelResolution.resolveOrDefault(
            modelSpec: model,
            defaultModelId: defaultModelId,
            defaultRevision: defaultRevision,
            progressHandler: progressHandler
        )

        return resolvedURL
    }
    public static func validateDimensions(
        width: Int,
        height: Int,
        vaeScale: Int = 16
    ) throws {
        if width % vaeScale != 0 {
            throw DimensionError.widthNotDivisible(width: width, scale: vaeScale)
        }
        if height % vaeScale != 0 {
            throw DimensionError.heightNotDivisible(height: height, scale: vaeScale)
        }
    }
    public enum DimensionError: Error, LocalizedError {
        case widthNotDivisible(width: Int, scale: Int)
        case heightNotDivisible(height: Int, scale: Int)

        public var errorDescription: String? {
            switch self {
            case .widthNotDivisible(let width, let scale):
                return "Width must be divisible by \(scale) (got \(width)). Please adjust to a multiple of \(scale)."
            case .heightNotDivisible(let height, let scale):
                return "Height must be divisible by \(scale) (got \(height)). Please adjust to a multiple of \(scale)."
            }
        }
    }
}
