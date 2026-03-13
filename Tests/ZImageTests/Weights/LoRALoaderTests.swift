import Foundation
import MLX
import XCTest

@testable import ZImage

final class LoRALoaderTests: MLXTestCase {

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

  func testDistillUnderscoreAttentionKeyMapping() {
    let input = "_layers_0_attention_to_q"
    let expected = "layers.0.attention.to_q.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testDistillUnderscoreFeedForwardKeyMapping() {
    let input = "_noise_refiner_1_feed_forward_w2"
    let expected = "noise_refiner.1.feed_forward.w2.weight"

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

  func testLoaderRejectsAmbiguousLocalDirectoryWithoutFilename() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("lora_dir_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let firstFile = directory.appendingPathComponent("first.safetensors")
    let secondFile = directory.appendingPathComponent("second.safetensors")
    try Data().write(to: firstFile)
    try Data().write(to: secondFile)

    XCTAssertThrowsError(try LoRAWeightLoader.load(from: directory)) { error in
      guard case LoRAError.ambiguousSafetensorsSource(let url, let files) = error else {
        return XCTFail("Unexpected error: \(error)")
      }
      XCTAssertEqual(url.standardizedFileURL.path, directory.standardizedFileURL.path)
      XCTAssertEqual(files.sorted(), ["first.safetensors", "second.safetensors"])
    }
  }

  func testLoaderRejectsLoRAThatMapsZeroValidTargets() throws {
    let fileURL = try makeLoRAFile(
      named: "invalid_targets",
      arrays: [
        "lora_unet__layers_99_attention_to_q.lora_down.weight": MLXArray([Float(0), 1, 2, 3], [2, 2]).asType(.bfloat16),
        "lora_unet__layers_99_attention_to_q.lora_up.weight": MLXArray([Float(0), 1, 2, 3], [2, 2]).asType(.bfloat16),
      ]
    )
    defer { try? FileManager.default.removeItem(at: fileURL) }

    XCTAssertThrowsError(try LoRAWeightLoader.load(from: fileURL)) { error in
      guard case LoRAError.incompatibleWeights(let message) = error else {
        return XCTFail("Unexpected error: \(error)")
      }
      XCTAssertTrue(message.contains("zero valid target layers"))
      XCTAssertTrue(message.contains("layers.99.attention.to_q.weight"))
    }
  }

  func testLoaderAcceptsDistillUnderscoreTargets() throws {
    let fileURL = try makeLoRAFile(
      named: "distill_targets",
      arrays: [
        "lora_unet__layers_0_attention_to_q.lora_down.weight": MLXArray([Float(0), 1, 2, 3], [2, 2]).asType(.bfloat16),
        "lora_unet__layers_0_attention_to_q.lora_up.weight": MLXArray([Float(0), 1, 2, 3], [2, 2]).asType(.bfloat16),
        "lora_unet__noise_refiner_1_feed_forward_w2.lora_down.weight": MLXArray([Float(0), 1, 2, 3], [2, 2]).asType(.bfloat16),
        "lora_unet__noise_refiner_1_feed_forward_w2.lora_up.weight": MLXArray([Float(0), 1, 2, 3], [2, 2]).asType(.bfloat16),
      ]
    )
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let weights = try LoRAWeightLoader.load(from: fileURL)
    XCTAssertEqual(weights.layerCount, 2)
    XCTAssertTrue(weights.weights.keys.contains("layers.0.attention.to_q.weight"))
    XCTAssertTrue(weights.weights.keys.contains("noise_refiner.1.feed_forward.w2.weight"))
  }

  func testResolveSourceRequiresExplicitFilenameForKnownDistillRepo() async throws {
    do {
      _ = try await LoRAWeightLoader.resolveSource(.huggingFace(modelId: "alibaba-pai/Z-Image-Fun-Lora-Distill", filename: nil))
      XCTFail("Expected explicit filename requirement")
    } catch let error as LoRAError {
      guard case .explicitFilenameRequired(let modelId, let suggestedFilename) = error else {
        return XCTFail("Unexpected error: \(error)")
      }
      XCTAssertEqual(modelId, "alibaba-pai/Z-Image-Fun-Lora-Distill")
      XCTAssertEqual(suggestedFilename, "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors")
    }
  }

  private func makeLoRAFile(named stem: String, arrays: [String: MLXArray]) throws -> URL {
    let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("\(stem)_\(UUID().uuidString).safetensors")
    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)
    return fileURL
  }
}
