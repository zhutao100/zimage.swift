import Foundation
import XCTest
@testable import ZImage

final class WeightsVariantResolutionTests: XCTestCase {
  func testSelectsVariantIndexWhenProvided() throws {
    let fixtureSnapshot = TestFixtures.snapshot(named: "ZImageTurbo")
    try TestFixtures.withTemporaryCopy(of: fixtureSnapshot) { snapshot in
      let transformerNonVariant = [
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
      ]
      let transformerFP16 = [
        "transformer/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
      ]
      let textEncoderNonVariant = [
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
      ]
      let textEncoderFP16 = [
        "text_encoder/model.fp16-00001-of-00002.safetensors",
        "text_encoder/model.fp16-00002-of-00002.safetensors",
      ]

      for relative in transformerNonVariant + transformerFP16 + textEncoderNonVariant + textEncoderFP16 {
        try TestFixtures.createEmptyFile(at: snapshot.appending(path: relative))
      }

      let transformerIndex = snapshot.appending(path: ZImageFiles.transformerIndex)
      let transformerNonVariantIndex = """
      { "weight_map": { "a": "diffusion_pytorch_model-00001-of-00002.safetensors", "b": "diffusion_pytorch_model-00002-of-00002.safetensors" } }
      """
      try transformerNonVariantIndex.data(using: .utf8)!.write(to: transformerIndex)

      let transformerFP16Index = snapshot.appending(path: "transformer/diffusion_pytorch_model.fp16.safetensors.index.json")
      let transformerFP16IndexText = """
      { "weight_map": { "a": "diffusion_pytorch_model.fp16-00001-of-00002.safetensors", "b": "diffusion_pytorch_model.fp16-00002-of-00002.safetensors" } }
      """
      try transformerFP16IndexText.data(using: .utf8)!.write(to: transformerFP16Index)

      let textEncoderIndex = snapshot.appending(path: ZImageFiles.textEncoderIndex)
      let textEncoderNonVariantIndex = """
      { "weight_map": { "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors" } }
      """
      try textEncoderNonVariantIndex.data(using: .utf8)!.write(to: textEncoderIndex)

      let textEncoderFP16Index = snapshot.appending(path: "text_encoder/model.fp16.safetensors.index.json")
      let textEncoderFP16IndexText = """
      { "weight_map": { "a": "model.fp16-00001-of-00002.safetensors", "b": "model.fp16-00002-of-00002.safetensors" } }
      """
      try textEncoderFP16IndexText.data(using: .utf8)!.write(to: textEncoderFP16Index)

      XCTAssertEqual(ZImageFiles.resolveTransformerWeights(at: snapshot), transformerNonVariant)
      XCTAssertEqual(ZImageFiles.resolveTransformerWeights(at: snapshot, weightsVariant: "fp16"), transformerFP16)
      XCTAssertEqual(ZImageFiles.resolveTextEncoderWeights(at: snapshot), textEncoderNonVariant)
      XCTAssertEqual(ZImageFiles.resolveTextEncoderWeights(at: snapshot, weightsVariant: "fp16"), textEncoderFP16)
    }
  }

  func testRejectsMixedIndexShardSetWhenVariantRequested() throws {
    let fixtureSnapshot = TestFixtures.snapshot(named: "ZImageTurbo")
    try TestFixtures.withTemporaryCopy(of: fixtureSnapshot) { snapshot in
      let transformerNonVariant = [
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
      ]
      let transformerFP16 = [
        "transformer/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
      ]

      for relative in transformerNonVariant + transformerFP16 {
        try TestFixtures.createEmptyFile(at: snapshot.appending(path: relative))
      }

      let transformerIndex = snapshot.appending(path: ZImageFiles.transformerIndex)
      let mixedIndex = """
      {
        "weight_map": {
          "a": "diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
          "b": "diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
          "c": "diffusion_pytorch_model-00001-of-00002.safetensors",
          "d": "diffusion_pytorch_model-00002-of-00002.safetensors"
        }
      }
      """
      try mixedIndex.data(using: .utf8)!.write(to: transformerIndex)

      XCTAssertEqual(ZImageFiles.resolveTransformerWeights(at: snapshot), transformerNonVariant)
      XCTAssertEqual(ZImageFiles.resolveTransformerWeights(at: snapshot, weightsVariant: "fp16"), transformerFP16)
    }
  }

  func testValidateRequiredComponentWeightsThrowsWhenVariantMissing() throws {
    let fixtureSnapshot = TestFixtures.snapshot(named: "ZImageTurbo")
    try TestFixtures.withTemporaryCopy(of: fixtureSnapshot) { snapshot in
      let transformerFP16 = [
        "transformer/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
      ]
      let textEncoderFP16 = [
        "text_encoder/model.fp16-00001-of-00002.safetensors",
        "text_encoder/model.fp16-00002-of-00002.safetensors",
      ]
      for relative in transformerFP16 + textEncoderFP16 {
        try TestFixtures.createEmptyFile(at: snapshot.appending(path: relative))
      }

      let transformerFP16Index = snapshot.appending(path: "transformer/diffusion_pytorch_model.fp16.safetensors.index.json")
      let transformerFP16IndexText = """
      { "weight_map": { "a": "diffusion_pytorch_model.fp16-00001-of-00002.safetensors", "b": "diffusion_pytorch_model.fp16-00002-of-00002.safetensors" } }
      """
      try transformerFP16IndexText.data(using: .utf8)!.write(to: transformerFP16Index)

      let textEncoderFP16Index = snapshot.appending(path: "text_encoder/model.fp16.safetensors.index.json")
      let textEncoderFP16IndexText = """
      { "weight_map": { "a": "model.fp16-00001-of-00002.safetensors", "b": "model.fp16-00002-of-00002.safetensors" } }
      """
      try textEncoderFP16IndexText.data(using: .utf8)!.write(to: textEncoderFP16Index)

      // VAE fp16 is missing; only non-variant weight exists.
      try TestFixtures.createEmptyFile(at: snapshot.appending(path: "vae/diffusion_pytorch_model.safetensors"))

      XCTAssertThrowsError(try ZImageFiles.validateRequiredComponentWeights(at: snapshot, weightsVariant: "fp16")) { error in
        guard case let ZImageFiles.WeightsVariantError.missingRequiredComponentWeights(_, missingComponents, _) = error else {
          XCTFail("Unexpected error: \(error)")
          return
        }
        XCTAssertTrue(missingComponents.contains("vae"))
      }
    }
  }

  func testDeterministicDefaultPrefersNonVariantWhenMultiplePresent() throws {
    let fixtureSnapshot = TestFixtures.snapshot(named: "ZImageTurbo")
    try TestFixtures.withTemporaryCopy(of: fixtureSnapshot) { snapshot in
      let transformerNonVariant = [
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
      ]
      let transformerFP16 = [
        "transformer/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
      ]

      for relative in transformerNonVariant + transformerFP16 {
        try TestFixtures.createEmptyFile(at: snapshot.appending(path: relative))
      }

      // Remove indices so directory scan is used.
      try? FileManager.default.removeItem(at: snapshot.appending(path: ZImageFiles.transformerIndex))
      try? FileManager.default.removeItem(at: snapshot.appending(path: "transformer/diffusion_pytorch_model.fp16.safetensors.index.json"))

      XCTAssertEqual(ZImageFiles.resolveTransformerWeights(at: snapshot), transformerNonVariant)
    }
  }
}
