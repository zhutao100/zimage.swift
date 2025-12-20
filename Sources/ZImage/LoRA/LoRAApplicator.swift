import Foundation
import MLX
import MLXNN
import Logging

public struct LoRAApplicator {

    public static func mergeWeights(
        baseWeights: [String: MLXArray],
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) -> [String: MLXArray] {
        var merged = baseWeights
        let effectiveScale = scale * loraWeights.effectiveScale

        logger?.info("Merging LoRA weights with scale=\(scale), alpha=\(loraWeights.alpha), rank=\(loraWeights.rank), effective_scale=\(effectiveScale)")

        var appliedCount = 0
        var skippedCount = 0

        for (keyPath, (down, up)) in loraWeights.weights {
            let weightKey = keyPath.hasSuffix(".weight") ? keyPath : keyPath + ".weight"

            guard let baseWeight = merged[weightKey] else {
                logger?.debug("LoRA key '\(weightKey)' not found in base weights, skipping")
                skippedCount += 1
                continue
            }

            guard let delta = computeDelta(up: up, down: down) else {
                logger?.warning("LoRA weight shapes incompatible for '\(weightKey)': up=\(up.shape), down=\(down.shape)")
                skippedCount += 1
                continue
            }

            guard let alignedDelta = alignShape(delta, to: baseWeight.shape) else {
                logger?.warning("LoRA delta shape \(delta.shape) doesn't match base \(baseWeight.shape) for '\(weightKey)'")
                skippedCount += 1
                continue
            }

            merged[weightKey] = baseWeight + (alignedDelta * effectiveScale).asType(baseWeight.dtype)
            appliedCount += 1
            logger?.debug("Applied LoRA to '\(weightKey)'")
        }

        logger?.info("LoRA merge complete: applied=\(appliedCount), skipped=\(skippedCount)")
        return merged
    }

    public static func applyToTransformer(
        _ transformer: ZImageTransformer2DModel,
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) {
        let effectiveScale = scale * loraWeights.effectiveScale

        logger?.info("Applying LoRA to transformer with scale=\(scale), effective_scale=\(effectiveScale)")

        var appliedCount = 0
        var quantizedCount = 0
        var layerUpdates: [String: MLXArray] = [:]

        for (key, module) in transformer.namedModules() {
            let loraKeyBase = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
            let loraKey = loraKeyBase + ".weight"

            guard let (down, up) = loraWeights.weights[loraKey] ?? loraWeights.weights[loraKeyBase] else {
                continue
            }

            guard let delta = computeDelta(up: up, down: down) else {
                logger?.debug("LoRA shape mismatch for \(key): up=\(up.shape), down=\(down.shape)")
                continue
            }

            if let quantizedLinear = module as? QuantizedLinear {
                let dequantizedWeight = MLX.dequantized(
                    quantizedLinear.weight,
                    scales: quantizedLinear.scales,
                    biases: quantizedLinear.biases,
                    groupSize: quantizedLinear.groupSize,
                    bits: quantizedLinear.bits
                )

                guard let alignedDelta = alignShape(delta, to: dequantizedWeight.shape) else {
                    logger?.debug("LoRA delta shape \(delta.shape) doesn't match dequantized \(dequantizedWeight.shape) for \(key)")
                    continue
                }

                let fusedWeight = dequantizedWeight + (alignedDelta * effectiveScale).asType(dequantizedWeight.dtype)

                let (newQuantizedWeight, newScales, newBiases) = MLX.quantized(
                    fusedWeight,
                    groupSize: quantizedLinear.groupSize,
                    bits: quantizedLinear.bits
                )

                layerUpdates[key + ".weight"] = newQuantizedWeight
                layerUpdates[key + ".scales"] = newScales
                if let biases = newBiases {
                    layerUpdates[key + ".biases"] = biases
                }

                appliedCount += 1
                quantizedCount += 1

            } else if let linear = module as? Linear {
                let currentWeight = linear.weight

                guard let alignedDelta = alignShape(delta, to: currentWeight.shape) else {
                    logger?.debug("LoRA delta shape \(delta.shape) doesn't match weight \(currentWeight.shape) for \(key)")
                    continue
                }

                layerUpdates[key + ".weight"] = currentWeight + (alignedDelta * effectiveScale).asType(currentWeight.dtype)
                appliedCount += 1
            }
        }

        if !layerUpdates.isEmpty {
            transformer.update(parameters: ModuleParameters.unflattened(layerUpdates))
        }

        if quantizedCount > 0 {
            logger?.info("LoRA applied to transformer: \(appliedCount) layers modified (\(quantizedCount) quantized)")
        } else {
            logger?.info("LoRA applied to transformer: \(appliedCount) layers modified")
        }
    }

    public static func removeFromWeights(
        mergedWeights: [String: MLXArray],
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) -> [String: MLXArray] {
        var restored = mergedWeights
        let effectiveScale = scale * loraWeights.effectiveScale

        for (keyPath, (down, up)) in loraWeights.weights {
            let weightKey = keyPath.hasSuffix(".weight") ? keyPath : keyPath + ".weight"

            guard let currentWeight = restored[weightKey],
                  let delta = computeDelta(up: up, down: down),
                  let alignedDelta = alignShape(delta, to: currentWeight.shape) else {
                continue
            }

            restored[weightKey] = currentWeight - (alignedDelta * effectiveScale).asType(currentWeight.dtype)
        }

        return restored
    }

    private static func computeDelta(up: MLXArray, down: MLXArray) -> MLXArray? {
        if up.dim(1) == down.dim(0) {
            return MLX.matmul(up, down)
        } else if up.dim(0) == down.dim(1) {
            return MLX.matmul(up.T, down.T)
        } else if up.dim(1) == down.dim(1) {
            return MLX.matmul(up, down.T)
        }
        return nil
    }

    private static func alignShape(_ delta: MLXArray, to targetShape: [Int]) -> MLXArray? {
        if delta.shape == targetShape {
            return delta
        } else if delta.T.shape == targetShape {
            return delta.T
        }
        return nil
    }
    public static func applyDynamically<T: Module>(
        to transformer: T,
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) {
        let effectiveScale = scale * loraWeights.effectiveScale

        logger?.info("Applying dynamic LoRA with scale=\(scale), effective_scale=\(effectiveScale)")

        var moduleUpdates: [(String, Module)] = []
        var appliedCount = 0
        var quantizedCount = 0

        for (key, module) in transformer.namedModules() {
            let loraKey = key.hasSuffix(".weight") ? key : key + ".weight"
            guard let (down, up) = loraWeights.weights[loraKey] ?? loraWeights.weights[key] else {
                continue
            }
            if let loraLinear = module as? LoRALinear {
                loraLinear.setLoRA(down: down, up: up, scale: effectiveScale)
                appliedCount += 1
                continue
            }
            if let loraQuantized = module as? LoRAQuantizedLinear {
                loraQuantized.setLoRA(down: down, up: up, scale: effectiveScale)
                appliedCount += 1
                quantizedCount += 1
                continue
            }
            if let quantizedLinear = module as? QuantizedLinear {
                let loraQuantized = LoRAQuantizedLinear(from: quantizedLinear)
                loraQuantized.setLoRA(down: down, up: up, scale: effectiveScale)
                moduleUpdates.append((key, loraQuantized))
                appliedCount += 1
                quantizedCount += 1
                continue
            }
            if let linear = module as? Linear {
                let loraLinear = LoRALinear(from: linear)
                loraLinear.setLoRA(down: down, up: up, scale: effectiveScale)
                moduleUpdates.append((key, loraLinear))
                appliedCount += 1
                continue
            }
        }

        if !moduleUpdates.isEmpty {
            transformer.update(modules: ModuleChildren.unflattened(moduleUpdates))
        }

        if quantizedCount > 0 {
            logger?.info("Dynamic LoRA applied to \(appliedCount) layers (\(quantizedCount) quantized)")
        } else {
            logger?.info("Dynamic LoRA applied to \(appliedCount) layers")
        }
    }

    public static func clearDynamicLoRA<T: Module>(
        from transformer: T,
        logger: Logger? = nil
    ) {
        var clearedCount = 0

        for (_, module) in transformer.namedModules() {
            if let loraLinear = module as? LoRALinear {
                loraLinear.clearLoRA()
                clearedCount += 1
            } else if let loraQuantized = module as? LoRAQuantizedLinear {
                loraQuantized.clearLoRA()
                clearedCount += 1
            }
        }

        logger?.info("Cleared dynamic LoRA from \(clearedCount) layers")
    }

    public static func hasDynamicLoRA<T: Module>(in transformer: T) -> Bool {
        for (_, module) in transformer.namedModules() {
            if let loraLinear = module as? LoRALinear, loraLinear.hasLoRA {
                return true
            }
            if let loraQuantized = module as? LoRAQuantizedLinear, loraQuantized.hasLoRA {
                return true
            }
        }
        return false
    }
}
