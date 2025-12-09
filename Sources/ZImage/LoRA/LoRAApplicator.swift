import Foundation
import MLX
import MLXNN
import Logging

public struct LoRAApplicator {

    private static func linearDims(for module: Module) -> (out: Int, in: Int)? {
        if let qlin = module as? QuantizedLinear {
            let outDim = qlin.weight.dim(max(0, qlin.weight.ndim - 2))
            let inDim = qlin.weight.dim(max(0, qlin.weight.ndim - 1))
            return (out: outDim, in: inDim)
        }
        if let lin = module as? Linear {
            let outDim = lin.weight.dim(max(0, lin.weight.ndim - 2))
            let inDim = lin.weight.dim(max(0, lin.weight.ndim - 1))
            return (out: outDim, in: inDim)
        }
        return nil
    }

    private static func normalizeLoRAPair(
        down: MLXArray,
        up: MLXArray,
        inFeatures: Int,
        outFeatures: Int
    ) -> (down: MLXArray, up: MLXArray)? {
        guard down.ndim == 2, up.ndim == 2 else { return nil }

        let d0 = down.dim(0)
        let d1 = down.dim(1)
        let u0 = up.dim(0)
        let u1 = up.dim(1)

        // target: down=[rank, in], up=[out, rank]
        if d1 == inFeatures, u0 == outFeatures, d0 == u1 {
            return (down: down, up: up)
        }
        // both transposed: down=[in, rank], up=[rank, out]
        if d0 == inFeatures, u1 == outFeatures, d1 == u0 {
            return (down: down.T, up: up.T)
        }
        // down transposed only: down=[in, rank], up=[out, rank]
        if d0 == inFeatures, u0 == outFeatures, d1 == u1 {
            return (down: down.T, up: up)
        }
        // up transposed only: down=[rank, in], up=[rank, out]
        if d1 == inFeatures, u1 == outFeatures, d0 == u0 {
            return (down: down, up: up.T)
        }
        return nil
    }

    private static func normalizeQKVLoRAPair(
        down: MLXArray,
        up: MLXArray,
        inFeatures: Int,
        outFeatures: Int,
        projectionIndex: Int
    ) -> (down: MLXArray, up: MLXArray)? {
        guard projectionIndex >= 0, projectionIndex < 3 else { return nil }

        guard let normalized = normalizeLoRAPair(
            down: down,
            up: up,
            inFeatures: inFeatures,
            outFeatures: outFeatures * 3
        ) else { return nil }

        let start = projectionIndex * outFeatures
        let end = start + outFeatures
        guard normalized.up.ndim == 2, normalized.up.dim(0) >= end else { return nil }

        let slicedUp = normalized.up[start..<end, 0...]
        return (down: normalized.down, up: slicedUp)
    }

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

    public static func applyLoKr<T: Module>(
        to transformer: T,
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) {
        applyLoKrInternal(to: transformer, loraWeights: loraWeights, signedScale: scale, logger: logger)
    }

    public static func removeLoKr<T: Module>(
        from transformer: T,
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) {
        applyLoKrInternal(to: transformer, loraWeights: loraWeights, signedScale: -scale, logger: logger)
    }

    private static func applyLoKrInternal<T: Module>(
        to transformer: T,
        loraWeights: LoRAWeights,
        signedScale: Float,
        logger: Logger? = nil
    ) {
        guard !loraWeights.lokrWeights.isEmpty else { return }

        var layerUpdates: [String: MLXArray] = [:]
        var appliedCount = 0
        var quantizedCount = 0

        func kron2D(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            let a0 = a.dim(0), a1 = a.dim(1)
            let b0 = b.dim(0), b1 = b.dim(1)
            let aExp = a.reshaped(a0, 1, a1, 1)
            let bExp = b.reshaped(1, b0, 1, b1)
            return (aExp * bExp).reshaped(a0 * b0, a1 * b1)
        }

        for (key, module) in transformer.namedModules() {
            guard let lokr = loraWeights.lokrWeights[key] else { continue }
            guard let dims = linearDims(for: module) else { continue }
            guard lokr.w1.ndim == 2, lokr.w2.ndim == 2 else { continue }

            let expectedOut = lokr.w1.dim(0) * lokr.w2.dim(0)
            let expectedIn = lokr.w1.dim(1) * lokr.w2.dim(1)
            if expectedOut != dims.out || expectedIn != dims.in {
                logger?.debug("Skipping LoKr for \(key): kron (\(expectedOut)x\(expectedIn)) vs weight (\(dims.out)x\(dims.in))")
                continue
            }

            let alphaScale = lokr.alpha ?? 1.0
            let effectiveScale = signedScale * alphaScale

            if let qlin = module as? QuantizedLinear {
                let dequantizedWeight = MLX.dequantized(
                    qlin.weight,
                    scales: qlin.scales,
                    biases: qlin.biases,
                    groupSize: qlin.groupSize,
                    bits: qlin.bits
                )

                var delta = kron2D(lokr.w1, lokr.w2)
                if let aligned = alignShape(delta, to: dequantizedWeight.shape) {
                    delta = aligned
                } else {
                    logger?.debug("Skipping LoKr for \(key): delta=\(delta.shape) weight=\(dequantizedWeight.shape)")
                    continue
                }

                if delta.dtype != dequantizedWeight.dtype { delta = delta.asType(dequantizedWeight.dtype) }
                let fusedWeight = dequantizedWeight + (delta * effectiveScale).asType(dequantizedWeight.dtype)
                let (newQuantizedWeight, newScales, newBiases) = MLX.quantized(
                    fusedWeight,
                    groupSize: qlin.groupSize,
                    bits: qlin.bits
                )

                layerUpdates[key + ".weight"] = newQuantizedWeight
                layerUpdates[key + ".scales"] = newScales
                if let biases = newBiases {
                    layerUpdates[key + ".biases"] = biases
                }

                appliedCount += 1
                quantizedCount += 1
                continue
            }

            if let lin = module as? Linear {
                let currentWeight = lin.weight

                var delta = kron2D(lokr.w1, lokr.w2)
                if let aligned = alignShape(delta, to: currentWeight.shape) {
                    delta = aligned
                } else {
                    logger?.debug("Skipping LoKr for \(key): delta=\(delta.shape) weight=\(currentWeight.shape)")
                    continue
                }

                if delta.dtype != currentWeight.dtype { delta = delta.asType(currentWeight.dtype) }
                layerUpdates[key + ".weight"] = currentWeight + (delta * effectiveScale).asType(currentWeight.dtype)
                appliedCount += 1
            }
        }

        if !layerUpdates.isEmpty {
            transformer.update(parameters: ModuleParameters.unflattened(layerUpdates))
        }

        if appliedCount > 0 {
            if quantizedCount > 0 {
                logger?.info("LoKr applied to \(appliedCount) layers (\(quantizedCount) quantized)")
            } else {
                logger?.info("LoKr applied to \(appliedCount) layers")
            }
        }
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
            guard let dims = linearDims(for: module) else { continue }

            let loraKey = key.hasSuffix(".weight") ? key : key + ".weight"
            var pair: (down: MLXArray, up: MLXArray)? = loraWeights.weights[loraKey] ?? loraWeights.weights[key]

            if pair == nil {
                // Fallback: some LoRA packs store combined qkv deltas under attention.qkv.*
                if key.contains(".attention.to_q") || key.contains(".attention.to_k") || key.contains(".attention.to_v") {
                    let qkvKey: String?
                    let projectionIndex: Int

                    if let range = key.range(of: ".attention.to_q") {
                        qkvKey = key.replacingCharacters(in: range, with: ".attention.qkv")
                        projectionIndex = 0
                    } else if let range = key.range(of: ".attention.to_k") {
                        qkvKey = key.replacingCharacters(in: range, with: ".attention.qkv")
                        projectionIndex = 1
                    } else if let range = key.range(of: ".attention.to_v") {
                        qkvKey = key.replacingCharacters(in: range, with: ".attention.qkv")
                        projectionIndex = 2
                    } else {
                        qkvKey = nil
                        projectionIndex = 0
                    }

                    if let qkvKey,
                       let qkvPair = loraWeights.weights[qkvKey + ".weight"] ?? loraWeights.weights[qkvKey],
                       let normalized = normalizeQKVLoRAPair(
                        down: qkvPair.down,
                        up: qkvPair.up,
                        inFeatures: dims.in,
                        outFeatures: dims.out,
                        projectionIndex: projectionIndex
                       ) {
                        pair = normalized
                    }
                }
            }

            guard let pair else { continue }

            guard let normalized = normalizeLoRAPair(
                down: pair.down,
                up: pair.up,
                inFeatures: dims.in,
                outFeatures: dims.out
            ) else {
                logger?.debug("Skipping LoRA for \(key): down=\(pair.down.shape) up=\(pair.up.shape) vs weight (\(dims.out)x\(dims.in))")
                continue
            }

            let down = normalized.down
            let up = normalized.up

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

        if loraWeights.hasLoKr {
            applyLoKr(to: transformer, loraWeights: loraWeights, scale: scale, logger: logger)
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
