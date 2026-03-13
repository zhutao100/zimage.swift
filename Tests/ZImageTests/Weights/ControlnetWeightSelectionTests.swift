import Foundation
import MLX
import XCTest

@testable import ZImage

final class ControlnetWeightSelectionTests: MLXTestCase {
  func testResolveControlnetWeightFilesRejectsAmbiguousDirectoryWithoutPreferredFile() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("controlnet_dir_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    try Data().write(to: directory.appendingPathComponent("union.safetensors"))
    try Data().write(to: directory.appendingPathComponent("union-lite.safetensors"))

    XCTAssertThrowsError(try ZImageControlPipeline.resolveControlnetWeightFiles(in: directory)) { error in
      guard case ZImageControlPipeline.PipelineError.weightsMissing(let message) = error else {
        return XCTFail("Unexpected error: \(error)")
      }
      XCTAssertTrue(message.contains("Specify --control-file"))
      XCTAssertTrue(message.contains("union.safetensors"))
      XCTAssertTrue(message.contains("union-lite.safetensors"))
    }
  }

  func testResolveControlnetWeightFilesHonorsPreferredFile() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("controlnet_dir_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let selected = directory.appendingPathComponent("union.safetensors")
    try Data().write(to: selected)
    try Data().write(to: directory.appendingPathComponent("union-lite.safetensors"))

    let files = try ZImageControlPipeline.resolveControlnetWeightFiles(
      in: directory,
      preferredFile: "union.safetensors"
    )
    XCTAssertEqual(
      files.map(\.standardizedFileURL.path),
      [selected].map(\.standardizedFileURL.path)
    )
  }

  func testValidateSupportedControlnetWeightsAcceptsFullUnionLayout() throws {
    let weights = makeWeights(controlLayerBlockCount: 15, refinerBlockCount: 2, patchInputFeatures: 132)
    XCTAssertNoThrow(
      try ZImageControlPipeline.validateSupportedControlnetWeights(
        weights,
        sourceName: "Z-Image-Fun-Controlnet-Union-2.1.safetensors"
      ))
  }

  func testValidateSupportedControlnetWeightsRejectsLiteLayout() throws {
    let weights = makeWeights(controlLayerBlockCount: 3, refinerBlockCount: 2, patchInputFeatures: 132)

    XCTAssertThrowsError(
      try ZImageControlPipeline.validateSupportedControlnetWeights(
        weights,
        sourceName: "Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors"
      )
    ) { error in
      guard case ZImageControlPipeline.PipelineError.weightsMissing(let message) = error else {
        return XCTFail("Unexpected error: \(error)")
      }
      XCTAssertTrue(message.contains("Expected 15 control layer blocks"))
    }
  }

  func testValidateSupportedControlnetSelectionRejectsTileVariant() throws {
    let file = URL(fileURLWithPath: "/tmp/Z-Image-Fun-Controlnet-Tile-2.1.safetensors")

    XCTAssertThrowsError(try ZImageControlPipeline.validateSupportedControlnetSelection(file: file)) { error in
      guard case ZImageControlPipeline.PipelineError.weightsMissing(let message) = error else {
        return XCTFail("Unexpected error: \(error)")
      }
      XCTAssertTrue(message.contains("Tile variants are not supported yet"))
    }
  }

  func testValidateSupportedControlnetSelectionRejectsLiteVariant() throws {
    let file = URL(fileURLWithPath: "/tmp/Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors")

    XCTAssertThrowsError(try ZImageControlPipeline.validateSupportedControlnetSelection(file: file)) { error in
      guard case ZImageControlPipeline.PipelineError.weightsMissing(let message) = error else {
        return XCTFail("Unexpected error: \(error)")
      }
      XCTAssertTrue(message.contains("Lite variants are not supported yet"))
    }
  }

  func testPreflightRequiresExplicitFileForKnownControlnetRepo() async throws {
    do {
      try await ZImageControlPipeline.preflightControlnetSelection(
        controlnetSpec: "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1",
        preferredFile: nil
      )
      XCTFail("Expected explicit control file requirement")
    } catch let error as ZImageControlPipeline.PipelineError {
      guard case .weightsMissing(let message) = error else {
        return XCTFail("Unexpected error: \(error)")
      }
      XCTAssertTrue(message.contains("requires an explicit --control-file"))
      XCTAssertTrue(message.contains("Z-Image-Fun-Controlnet-Union-2.1.safetensors"))
    }
  }

  func testPreflightRejectsUnsupportedLiteSelectionBeforeLoading() async throws {
    do {
      try await ZImageControlPipeline.preflightControlnetSelection(
        controlnetSpec: "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1",
        preferredFile: "Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors"
      )
      XCTFail("Expected unsupported lite selection")
    } catch let error as ZImageControlPipeline.PipelineError {
      guard case .weightsMissing(let message) = error else {
        return XCTFail("Unexpected error: \(error)")
      }
      XCTAssertTrue(message.contains("Lite variants are not supported yet"))
    }
  }

  private func makeWeights(
    controlLayerBlockCount: Int,
    refinerBlockCount: Int,
    patchInputFeatures: Int
  ) -> [String: MLXArray] {
    var weights: [String: MLXArray] = [
      "control_all_x_embedder.2-1.weight": MLXArray(
        Array(repeating: Float(0), count: patchInputFeatures),
        [1, patchInputFeatures]
      ).asType(.bfloat16)
    ]

    for index in 0..<controlLayerBlockCount {
      weights["control_layers.\(index).proj.weight"] = MLXArray([Float(0)], [1]).asType(.bfloat16)
    }

    for index in 0..<refinerBlockCount {
      weights["control_noise_refiner.\(index).proj.weight"] = MLXArray([Float(0)], [1]).asType(.bfloat16)
    }

    return weights
  }
}
