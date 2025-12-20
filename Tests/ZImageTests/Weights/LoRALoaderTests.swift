import XCTest
@testable import ZImage

final class LoRALoaderTests: XCTestCase {

  func testLoRAUnetPrefixRemoval() {
    let input = "lora_unet_transformer_blocks.0.attn.to_q"
    let expected = "layers.0.attention.to_q.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testDiffusionModelPrefixRemoval() {
    let input = "diffusion_model.layers.0.attention.to_q"
    let expected = "layers.0.attention.to_q.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testNoPrefixUnchanged() {
    let input = "layers.0.attention.to_q.weight"
    let expected = "layers.0.attention.to_q.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testFFLayerMapping_Net0() {
    let input = "transformer_blocks.5.ff.net.0.proj"
    let expected = "layers.5.feed_forward.w1.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testFFLayerMapping_Net2() {
    let input = "transformer_blocks.5.ff.net.2"
    let expected = "layers.5.feed_forward.w2.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testAttentionKeyMapping() {
    let testCases = [
      ("transformer_blocks.0.attn.to_q", "layers.0.attention.to_q.weight"),
      ("transformer_blocks.0.attn.to_k", "layers.0.attention.to_k.weight"),
      ("transformer_blocks.0.attn.to_v", "layers.0.attention.to_v.weight"),
      ("transformer_blocks.0.attn.to_out.0", "layers.0.attention.to_out.0.weight"),
    ]

    for (input, expected) in testCases {
      let result = LoRAKeyMapper.mapToZImageKey(input)
      XCTAssertEqual(result, expected, "Failed for input: \(input)")
    }
  }

  func testLoRAKeyWithLoraunnetPrefix() {
    let testCases = [
      ("lora_unet_transformer_blocks.0.attn.to_q", "layers.0.attention.to_q.weight"),
      ("lora_unet_transformer_blocks.5.ff.net.0.proj", "layers.5.feed_forward.w1.weight"),
    ]

    for (input, expected) in testCases {
      let result = LoRAKeyMapper.mapToZImageKey(input)
      XCTAssertEqual(result, expected, "Failed for input: \(input)")
    }
  }

  func testNoiseRefinerKeys() {
    let input = "noise_refiner.0.attn.to_q"
    let expected = "noise_refiner.0.attention.to_q.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testValidTargetPaths() {
    XCTAssertTrue(LoRAKeyMapper.isValidTarget("layers.0.attention.to_q.weight"))
    XCTAssertTrue(LoRAKeyMapper.isValidTarget("layers.0.feed_forward.w1.weight"))
    XCTAssertTrue(LoRAKeyMapper.isValidTarget("noise_refiner.0.attention.to_q.weight"))
    XCTAssertTrue(LoRAKeyMapper.isValidTarget("context_refiner.0.attention.to_k.weight"))
  }

  func testInvalidTargetPaths() {
    XCTAssertFalse(LoRAKeyMapper.isValidTarget("invalid.path.weight"))
    XCTAssertFalse(LoRAKeyMapper.isValidTarget("layers.99.attention.to_q.weight"))
  }

  func testLoRAErrorFileNotFound() {
    let error = LoRAError.fileNotFound("/nonexistent/path")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("/nonexistent/path"))
  }

  func testLoRAErrorInvalidFormat() {
    let error = LoRAError.invalidFormat("missing keys")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("missing keys"))
  }

  func testLoRAErrorIncompatibleWeights() {
    let error = LoRAError.incompatibleWeights("Shape mismatch")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("Shape mismatch"))
  }

  func testLoRAErrorNoSafetensorsFound() {
    let url = URL(fileURLWithPath: "/some/path")
    let error = LoRAError.noSafetensorsFound(url)
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("/some/path"))
  }

  func testSupportedTargetPathsCount() {

    let paths = LoRAKeyMapper.supportedTargetPaths
    XCTAssertEqual(paths.count, 238)
  }
}
