import Foundation
import MLX
import Hub

public final class LoRAWeightLoader {

    private static let loraPatterns: [(down: String, up: String)] = [
        (".lora_down.", ".lora_up."),
        (".lora_A.", ".lora_B.")
    ]

    private static let prefixesToRemove = [
        "base_model.model.",
        "diffusion_model.",
        "lora_unet_",
        "lora_te_",
        "transformer.",
        "text_encoder."
    ]

    public static func load(from config: LoRAConfiguration) async throws -> LoRAWeights {
        let url = try await resolveSource(config.source)
        return try load(from: url)
    }

    public static func load(from url: URL) throws -> LoRAWeights {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoRAError.fileNotFound(url.path)
        }

        let allWeights = try MLX.loadArrays(url: url)
        var loraWeights: [String: (down: MLXArray, up: MLXArray)] = [:]
        var processedKeys = Set<String>()

        for key in allWeights.keys {
            if processedKeys.contains(key) { continue }

            guard let (downKey, upKey, baseKey) = resolveKeyPair(key) else { continue }
            guard let downWeight = allWeights[downKey],
                  let upWeight = allWeights[upKey] else { continue }

            let mappedKey = LoRAKeyMapper.mapToZImageKey(baseKey)
            loraWeights[mappedKey] = (down: downWeight, up: upWeight)

            processedKeys.insert(downKey)
            processedKeys.insert(upKey)
        }

        guard !loraWeights.isEmpty else {
            throw LoRAError.invalidFormat("No valid LoRA weight pairs found. Expected keys with .lora_down/.lora_up or .lora_A/.lora_B patterns.")
        }

        let rank = inferRank(from: loraWeights)
        let alpha = loadAlpha(from: url.deletingLastPathComponent())

        return LoRAWeights(weights: loraWeights, rank: rank, alpha: alpha)
    }

    public static func resolveSource(_ source: LoRASource) async throws -> URL {
        switch source {
        case .local(let url):
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw LoRAError.fileNotFound(url.path)
            }
            return url

        case .huggingFace(let modelId, let filename):
            return try await downloadFromHuggingFace(modelId: modelId, filename: filename)
        }
    }

    public static func validate(at url: URL) throws -> LoRAValidationResult {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .invalid("File not found: \(url.path)")
        }

        do {
            let allWeights = try MLX.loadArrays(url: url)
            let keys = Array(allWeights.keys)

            let hasDownWeights = keys.contains { key in loraPatterns.contains { key.contains($0.down) } }
            let hasUpWeights = keys.contains { key in loraPatterns.contains { key.contains($0.up) } }

            guard hasDownWeights && hasUpWeights else {
                return .invalid("Not a valid LoRA file: missing lora_down/lora_up weight pairs")
            }

            var targetLayers: [String] = []
            var rank = 0

            for key in keys {
                let matchedDownPattern = loraPatterns.first { key.contains($0.down) }?.down
                guard let downPattern = matchedDownPattern,
                      let tensor = allWeights[key] else { continue }

                let layerName = extractBaseKey(key, pattern: downPattern) ?? key
                targetLayers.append(layerName)

                if tensor.ndim == 2 {
                    rank = max(rank, min(tensor.dim(0), tensor.dim(1)))
                }
            }

            let estimatedMemoryMB = (rank * 3840 * 2 * 4 * targetLayers.count) / (1024 * 1024)

            return LoRAValidationResult(
                isValid: true,
                rank: rank,
                targetLayers: targetLayers,
                estimatedMemoryMB: estimatedMemoryMB
            )
        } catch {
            return .invalid("Failed to read LoRA file: \(error.localizedDescription)")
        }
    }

    private static func resolveKeyPair(_ key: String) -> (downKey: String, upKey: String, baseKey: String)? {
        for (downPattern, upPattern) in loraPatterns {
            if key.contains(downPattern) {
                guard let base = extractBaseKey(key, pattern: downPattern) else { return nil }
                let upKey = key.replacingOccurrences(of: downPattern, with: upPattern)
                return (key, upKey, base)
            } else if key.contains(upPattern) {
                guard let base = extractBaseKey(key, pattern: upPattern) else { return nil }
                let downKey = key.replacingOccurrences(of: upPattern, with: downPattern)
                return (downKey, key, base)
            }
        }
        return nil
    }

    private static func extractBaseKey(_ key: String, pattern: String) -> String? {
        guard let range = key.range(of: pattern) else { return nil }
        var base = String(key[..<range.lowerBound])

        for prefix in prefixesToRemove {
            if base.hasPrefix(prefix) {
                base = String(base.dropFirst(prefix.count))
                break
            }
        }

        return base
    }

    private static func inferRank(from weights: [String: (down: MLXArray, up: MLXArray)]) -> Int {
        for (_, pair) in weights {
            let downShape = pair.down.shape
            if downShape.count == 2 {
                return min(downShape[0], downShape[1])
            }
        }
        return 16
    }

    private static func loadAlpha(from directory: URL) -> Float? {
        let configPath = directory.appendingPathComponent("adapter_config.json")

        guard FileManager.default.fileExists(atPath: configPath.path),
              let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let alpha = json["lora_alpha"] as? NSNumber else {
            return nil
        }

        return alpha.floatValue
    }

    private static func downloadFromHuggingFace(modelId: String, filename: String?) async throws -> URL {
        let filePatterns = filename.map { [$0] } ?? ["*.safetensors"]

        do {
            let snapshotURL = try await ModelResolution.resolve(
                modelSpec: modelId,
                filePatterns: filePatterns
            )

            let fm = FileManager.default
            let contents = try fm.contentsOfDirectory(at: snapshotURL, includingPropertiesForKeys: nil)

            if let filename = filename {
                let targetURL = snapshotURL.appendingPathComponent(filename)
                if fm.fileExists(atPath: targetURL.path) {
                    return targetURL
                }
            }

            if let safetensorFile = contents.first(where: { $0.pathExtension == "safetensors" }) {
                return safetensorFile
            }

            throw LoRAError.noSafetensorsFound(snapshotURL)
        } catch let error as LoRAError {
            throw error
        } catch {
            throw LoRAError.downloadFailed(modelId, error)
        }
    }
}
