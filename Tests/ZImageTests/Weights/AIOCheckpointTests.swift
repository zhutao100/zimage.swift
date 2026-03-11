import Logging
import MLX
import XCTest

@testable import ZImage

final class AIOCheckpointTests: MLXTestCase {

  func testInspectDetectsAIOCheckpoint() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("aio_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let arrays: [String: MLXArray] = [
      "model.diffusion_model.cap_embedder.0.weight": MLXArray([Float(0.0)], [1]).asType(.bfloat16),
      "text_encoders.qwen3_4b.transformer.model.embed_tokens.weight": MLXArray([Float(0.0)], [1, 1]).asType(.bfloat16),
      "vae.decoder.conv_in.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
      "vae.decoder.conv_out.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
      "vae.decoder.mid.block_1.conv1.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
      "vae.decoder.up.0.block.0.conv1.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
    ]
    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let inspection = ZImageAIOCheckpoint.inspect(fileURL: fileURL)
    XCTAssertTrue(inspection.isAIO)
    XCTAssertEqual(inspection.textEncoderPrefix, "text_encoders.qwen3_4b.transformer.")
  }

  func testInspectRejectsCheckpointMissingTextEncoder() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("aio_missing_te_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let arrays: [String: MLXArray] = [
      "model.diffusion_model.cap_embedder.0.weight": MLXArray([Float(0.0)], [1]).asType(.bfloat16),
      "vae.decoder.conv_in.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
      "vae.decoder.conv_out.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
      "vae.decoder.mid.block_1.conv1.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
      "vae.decoder.up.0.block.0.conv1.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
    ]
    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let inspection = ZImageAIOCheckpoint.inspect(fileURL: fileURL)
    XCTAssertFalse(inspection.isAIO)
    XCTAssertTrue(inspection.diagnostics.contains(where: { $0.contains("text encoder") }))
  }

  func testResolveModelSelectionUsesAIOByDefaultUnlessForced() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("aio_select_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let arrays: [String: MLXArray] = [
      "model.diffusion_model.cap_embedder.0.weight": MLXArray([Float(0.0)], [1]).asType(.bfloat16),
      "text_encoders.qwen3_4b.transformer.model.embed_tokens.weight": MLXArray([Float(0.0)], [1, 1]).asType(.bfloat16),
      "vae.decoder.conv_in.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
      "vae.decoder.conv_out.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
      "vae.decoder.mid.block_1.conv1.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
      "vae.decoder.up.0.block.0.conv1.weight": MLXArray([Float(0.0)], [1, 1, 1, 1]).asType(.bfloat16),
    ]
    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let pipeline = ZImagePipeline(logger: Logger(label: "test"))

    let auto = try pipeline.resolveModelSelection(fileURL.path, forceTransformerOverrideOnly: false)
    XCTAssertEqual(auto.aioCheckpointURL?.standardizedFileURL.path, fileURL.standardizedFileURL.path)
    XCTAssertNil(auto.transformerOverrideURL)

    let forced = try pipeline.resolveModelSelection(fileURL.path, forceTransformerOverrideOnly: true)
    XCTAssertEqual(forced.transformerOverrideURL?.standardizedFileURL.path, fileURL.standardizedFileURL.path)
    XCTAssertNil(forced.aioCheckpointURL)
  }

  func testResolveModelSelectionRejectsInvalidLocalDirectory() throws {
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("invalid_model_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let pipeline = ZImagePipeline(logger: Logger(label: "test"))

    XCTAssertThrowsError(try pipeline.resolveModelSelection(tempDir.path, forceTransformerOverrideOnly: false)) {
      error in
      guard case ZImagePipeline.PipelineError.invalidModelPath(let message) = error else {
        XCTFail("Unexpected error: \(error)")
        return
      }
      XCTAssertTrue(message.contains("Invalid local model directory"))
    }
  }

  func testResolveModelSelectionRejectsUnsupportedLocalFile() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("invalid_model_\(UUID().uuidString).txt")
    defer { try? FileManager.default.removeItem(at: fileURL) }
    try Data("not a model".utf8).write(to: fileURL)

    let pipeline = ZImagePipeline(logger: Logger(label: "test"))

    XCTAssertThrowsError(try pipeline.resolveModelSelection(fileURL.path, forceTransformerOverrideOnly: false)) {
      error in
      guard case ZImagePipeline.PipelineError.invalidModelPath(let message) = error else {
        XCTFail("Unexpected error: \(error)")
        return
      }
      XCTAssertTrue(message.contains("Unsupported local model file"))
    }
  }
}
