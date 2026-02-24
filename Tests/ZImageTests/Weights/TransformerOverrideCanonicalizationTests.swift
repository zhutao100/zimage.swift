import Logging
import MLX
import XCTest
@testable import ZImage

final class TransformerOverrideCanonicalizationTests: XCTestCase {
  func testStripsModelDiffusionModelPrefix() {
    let w = MLXArray([Float(0.0)])
    let input = ["model.diffusion_model.layers.0.attention_norm1.weight": w]

    let result = ZImageTransformerOverride.canonicalize(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.0.attention_norm1.weight"])
  }

  func testStripsDiffusionModelPrefix() {
    let w = MLXArray([Float(0.0)])
    let input = ["diffusion_model.layers.1.ffn_norm1.weight": w]

    let result = ZImageTransformerOverride.canonicalize(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.1.ffn_norm1.weight"])
  }

  func testStripsModelPrefixOnly() {
    let w = MLXArray([Float(0.0)])
    let input = ["model.layers.2.attention_norm1.weight": w]

    let result = ZImageTransformerOverride.canonicalize(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.2.attention_norm1.weight"])
  }

  func testStripsTransformerPrefix() {
    let w = MLXArray([Float(0.0)])
    let input = ["transformer.layers.3.attention_norm1.weight": w]

    let result = ZImageTransformerOverride.canonicalize(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.3.attention_norm1.weight"])
  }

  func testQKVSplitUsesNormalizedKey() {
    let dim = 2
    let values = (0 ..< (dim * 3 * dim)).map { Float($0) }
    let qkv = MLXArray(values, [dim * 3, dim])
    let input = ["model.diffusion_model.layers.0.attention.qkv.weight": qkv]

    let result = ZImageTransformerOverride.canonicalize(input, dim: dim, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.0.attention.to_q.weight"])
    XCTAssertNotNil(result["layers.0.attention.to_k.weight"])
    XCTAssertNotNil(result["layers.0.attention.to_v.weight"])
  }

  func testFinalLayerRemapAfterPrefixStrip() {
    let w = MLXArray([Float(0.0)])
    let input = ["model.diffusion_model.final_layer.adaLN_modulation.1.weight": w]

    let result = ZImageTransformerOverride.canonicalize(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["all_final_layer.2-1.adaLN_modulation.1.weight"])
  }

  func testRenamesQKNormKeys() {
    let w = MLXArray([Float(0.0), Float(1.0)]).asType(.bfloat16)
    let input: [String: MLXArray] = [
      "model.diffusion_model.layers.0.attention.q_norm.weight": w,
      "model.diffusion_model.layers.0.attention.k_norm.weight": w,
    ]

    let result = ZImageTransformerOverride.canonicalize(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.0.attention.norm_q.weight"])
    XCTAssertNotNil(result["layers.0.attention.norm_k.weight"])
  }
}
