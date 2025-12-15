import XCTest
import MLX
import Logging
@testable import ZImage

final class TransformerOverrideCanonicalizationTests: XCTestCase {

  func testStripsModelDiffusionModelPrefix() {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let w = MLXArray([Float(0.0)])
    let input = ["model.diffusion_model.layers.0.attention_norm1.weight": w]

    let result = pipeline.canonicalizeTransformerOverride(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.0.attention_norm1.weight"])
  }

  func testStripsDiffusionModelPrefix() {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let w = MLXArray([Float(0.0)])
    let input = ["diffusion_model.layers.1.ffn_norm1.weight": w]

    let result = pipeline.canonicalizeTransformerOverride(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.1.ffn_norm1.weight"])
  }

  func testStripsModelPrefixOnly() {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let w = MLXArray([Float(0.0)])
    let input = ["model.layers.2.attention_norm1.weight": w]

    let result = pipeline.canonicalizeTransformerOverride(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.2.attention_norm1.weight"])
  }

  func testStripsTransformerPrefix() {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let w = MLXArray([Float(0.0)])
    let input = ["transformer.layers.3.attention_norm1.weight": w]

    let result = pipeline.canonicalizeTransformerOverride(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.3.attention_norm1.weight"])
  }

  func testQKVSplitUsesNormalizedKey() {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let dim = 2
    let values = (0..<(dim * 3 * dim)).map { Float($0) }
    let qkv = MLXArray(values, [dim * 3, dim])
    let input = ["model.diffusion_model.layers.0.attention.qkv.weight": qkv]

    let result = pipeline.canonicalizeTransformerOverride(input, dim: dim, logger: Logger(label: "test"))
    XCTAssertNotNil(result["layers.0.attention.to_q.weight"])
    XCTAssertNotNil(result["layers.0.attention.to_k.weight"])
    XCTAssertNotNil(result["layers.0.attention.to_v.weight"])
  }

  func testFinalLayerRemapAfterPrefixStrip() {
    let pipeline = ZImagePipeline(logger: Logger(label: "test"))
    let w = MLXArray([Float(0.0)])
    let input = ["model.diffusion_model.final_layer.adaLN_modulation.1.weight": w]

    let result = pipeline.canonicalizeTransformerOverride(input, dim: 2, logger: Logger(label: "test"))
    XCTAssertNotNil(result["all_final_layer.2-1.adaLN_modulation.1.weight"])
  }
}

