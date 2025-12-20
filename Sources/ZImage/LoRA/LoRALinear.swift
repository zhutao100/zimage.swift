import Foundation
import MLX
import MLXNN

public protocol DynamicLoRACapable: AnyObject {

    var loraDown: MLXArray? { get set }
    var loraUp: MLXArray? { get set }
    var loraScale: Float { get set }
}

extension DynamicLoRACapable {

    public func setLoRA(down: MLXArray, up: MLXArray, scale: Float) {
        self.loraDown = down
        self.loraUp = up
        self.loraScale = scale
    }
    public func clearLoRA() {
        self.loraDown = nil
        self.loraUp = nil
        self.loraScale = 0.0
    }
    public var hasLoRA: Bool {
        loraDown != nil && loraUp != nil && loraScale > 0
    }
    public func computeLoRAContribution(_ x: MLXArray) -> MLXArray? {
        guard let down = loraDown, let up = loraUp, loraScale > 0 else {
            return nil
        }
        let loraHidden = MLX.matmul(x, down.T)
        let loraOut = MLX.matmul(loraHidden, up.T)
        return loraOut * loraScale
    }
}
public class LoRALinear: Linear, DynamicLoRACapable {
    public var loraDown: MLXArray?
    public var loraUp: MLXArray?
    public var loraScale: Float = 0.0
    public convenience init(from linear: Linear) {
        self.init(weight: linear.weight, bias: linear.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {

        var result: MLXArray
        if let bias = bias {
            result = MLX.addMM(bias, x, weight.T)
        } else {
            result = MLX.matmul(x, weight.T)
        }
        if let down = loraDown, let up = loraUp, loraScale > 0 {
            let loraHidden = MLX.matmul(x, down.T)
            let loraOut = MLX.matmul(loraHidden, up.T)
            result = result + (loraOut * loraScale).asType(result.dtype)
        }

        return result
    }
}
public class LoRAQuantizedLinear: QuantizedLinear, DynamicLoRACapable {
    public var loraDown: MLXArray?
    public var loraUp: MLXArray?
    public var loraScale: Float = 0.0
    public convenience init(from quantizedLinear: QuantizedLinear) {
        self.init(
            weight: quantizedLinear.weight,
            bias: quantizedLinear.bias,
            scales: quantizedLinear.scales,
            biases: quantizedLinear.biases,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {

        var result = MLX.quantizedMatmul(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits
        )

        if let bias = bias {
            result = result + bias
        }
        if let down = loraDown, let up = loraUp, loraScale > 0 {

            let loraHidden = MLX.matmul(x, down.T)
            let loraOut = MLX.matmul(loraHidden, up.T)
            result = result + (loraOut * loraScale).asType(result.dtype)
        }

        return result
    }
}
