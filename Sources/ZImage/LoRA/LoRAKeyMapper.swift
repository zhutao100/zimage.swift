import Foundation

public struct LoRAKeyMapper {

    private static let componentMappings: [String: String] = [
        "attn.to_q": "attention.to_q",
        "attn.to_k": "attention.to_k",
        "attn.to_v": "attention.to_v",
        "attn.to_out.0": "attention.to_out.0",
        "attn1.to_q": "attention.to_q",
        "attn1.to_k": "attention.to_k",
        "attn1.to_v": "attention.to_v",
        "attn1.to_out.0": "attention.to_out.0",
        "self_attn.q_proj": "attention.to_q",
        "self_attn.k_proj": "attention.to_k",
        "self_attn.v_proj": "attention.to_v",
        "self_attn.o_proj": "attention.to_out.0",
        "ff.net.0.proj": "feed_forward.w1",
        "ff.net.2": "feed_forward.w2",
        "ff_net_0_proj": "feed_forward.w1",
        "ff_net_2": "feed_forward.w2",
        "mlp.gate_proj": "feed_forward.w1",
        "mlp.up_proj": "feed_forward.w3",
        "mlp.down_proj": "feed_forward.w2",
    ]

    private static let blockMappings: [String: String] = [
        "transformer_blocks": "layers",
        "single_transformer_blocks": "layers",
        "down_blocks": "layers",
        "up_blocks": "layers",
        "noise_refiner": "noise_refiner",
        "context_refiner": "context_refiner",
    ]

    private static let prefixesToRemove = [
        "base_model.model.",
        "diffusion_model.",
        "lora_unet_",
        "lora_te_",
        "transformer.",
        "text_encoder.",
        "model.",
    ]

    private static let allMappingKeys: Set<String> = {
        var keys = Set<String>()
        keys.formUnion(componentMappings.keys)
        keys.formUnion(blockMappings.keys)
        return keys
    }()

    private static let cachedTargetPaths: [String] = {
        var paths: [String] = []
        paths.reserveCapacity(238)

        for i in 0..<30 {
            paths.append(contentsOf: [
                "layers.\(i).attention.to_q.weight",
                "layers.\(i).attention.to_k.weight",
                "layers.\(i).attention.to_v.weight",
                "layers.\(i).attention.to_out.0.weight",
                "layers.\(i).feed_forward.w1.weight",
                "layers.\(i).feed_forward.w2.weight",
                "layers.\(i).feed_forward.w3.weight",
            ])
        }

        for i in 0..<2 {
            paths.append(contentsOf: [
                "noise_refiner.\(i).attention.to_q.weight",
                "noise_refiner.\(i).attention.to_k.weight",
                "noise_refiner.\(i).attention.to_v.weight",
                "noise_refiner.\(i).attention.to_out.0.weight",
                "noise_refiner.\(i).feed_forward.w1.weight",
                "noise_refiner.\(i).feed_forward.w2.weight",
                "noise_refiner.\(i).feed_forward.w3.weight",
            ])
        }

        for i in 0..<2 {
            paths.append(contentsOf: [
                "context_refiner.\(i).attention.to_q.weight",
                "context_refiner.\(i).attention.to_k.weight",
                "context_refiner.\(i).attention.to_v.weight",
                "context_refiner.\(i).attention.to_out.0.weight",
                "context_refiner.\(i).feed_forward.w1.weight",
                "context_refiner.\(i).feed_forward.w2.weight",
                "context_refiner.\(i).feed_forward.w3.weight",
            ])
        }

        return paths
    }()

    private static let validTargetPathsSet: Set<String> = Set(cachedTargetPaths)

    public static func mapToZImageKey(_ loraKey: String) -> String {
        var key = loraKey

        for prefix in prefixesToRemove {
            if key.hasPrefix(prefix) {
                key = String(key.dropFirst(prefix.count))
                break
            }
        }

        if key.contains("_") && !key.contains(".") {
            key = convertUnderscoreToDot(key)
        }

        for (from, to) in blockMappings {
            key = key.replacingOccurrences(of: "\(from).", with: "\(to).")
        }

        for (from, to) in componentMappings {
            if key.hasSuffix(from) {
                key = String(key.dropLast(from.count)) + to
                break
            } else if key.contains(".\(from).") {
                key = key.replacingOccurrences(of: ".\(from).", with: ".\(to).")
                break
            }
        }

        if !key.hasSuffix(".weight") && !key.hasSuffix(".bias") {
            key = key + ".weight"
        }

        return key
    }

    private static func convertUnderscoreToDot(_ key: String) -> String {
        var result = ""
        var components = key.split(separator: "_").map(String.init)

        var i = 0
        while i < components.count {
            let component = components[i]

            if Int(component) != nil {
                if !result.isEmpty { result += "." }
                result += component
            } else {
                var matched = false
                for length in stride(from: min(4, components.count - i), to: 0, by: -1) {
                    let combined = components[i..<(i + length)].joined(separator: "_")
                    if allMappingKeys.contains(combined) {
                        if !result.isEmpty { result += "." }
                        result += combined
                        i += length - 1
                        matched = true
                        break
                    }
                }

                if !matched {
                    if !result.isEmpty { result += "." }
                    result += component
                }
            }

            i += 1
        }

        return result
    }

    public static var supportedTargetPaths: [String] {
        cachedTargetPaths
    }

    public static func isValidTarget(_ keyPath: String) -> Bool {
        let normalized = keyPath.hasSuffix(".weight") ? keyPath : keyPath + ".weight"
        return validTargetPathsSet.contains(normalized)
    }
}
