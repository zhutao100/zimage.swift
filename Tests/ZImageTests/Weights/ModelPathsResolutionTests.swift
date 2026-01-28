import XCTest
@testable import ZImage

final class ModelPathsResolutionTests: XCTestCase {
  func testResolveShardListsFromIndexAndFallbackDiscovery() throws {
    let expectedTransformer = [
      "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
      "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
      "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    ]
    let expectedTextEncoder = [
      "text_encoder/model-00001-of-00003.safetensors",
      "text_encoder/model-00002-of-00003.safetensors",
      "text_encoder/model-00003-of-00003.safetensors",
    ]

    for fixtureName in ["ZImageTurbo", "ZImageBase"] {
      let fixtureSnapshot = TestFixtures.snapshot(named: fixtureName)

      try TestFixtures.withTemporaryCopy(of: fixtureSnapshot) { snapshot in
        for relative in expectedTransformer + expectedTextEncoder {
          try TestFixtures.createEmptyFile(at: snapshot.appending(path: relative))
        }

        XCTAssertEqual(
          ZImageFiles.resolveTransformerWeights(at: snapshot),
          expectedTransformer,
          "fixture=\(fixtureName)"
        )
        XCTAssertEqual(
          ZImageFiles.resolveTextEncoderWeights(at: snapshot),
          expectedTextEncoder,
          "fixture=\(fixtureName)"
        )

        // Index weight_map values may already include the component prefix; accept either style.
        let transformerIndex = snapshot.appending(path: ZImageFiles.transformerIndex)
        let prefixedTransformerIndex = """
        {
          "weight_map": {
            "model.diffusion_model.layers.0.weight": "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "model.diffusion_model.layers.1.weight": "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "model.diffusion_model.layers.2.weight": "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
          }
        }
        """
        try prefixedTransformerIndex.data(using: .utf8)!.write(to: transformerIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTransformerWeights(at: snapshot),
          expectedTransformer,
          "fixture=\(fixtureName)"
        )

        // If the index exists but points to missing shards, fall back to directory discovery.
        let missingTransformerIndex = """
        { "weight_map": { "missing.weight": "diffusion_pytorch_model-99999-of-99999.safetensors" } }
        """
        try missingTransformerIndex.data(using: .utf8)!.write(to: transformerIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTransformerWeights(at: snapshot),
          expectedTransformer,
          "fixture=\(fixtureName)"
        )

        try FileManager.default.removeItem(at: transformerIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTransformerWeights(at: snapshot),
          expectedTransformer,
          "fixture=\(fixtureName)"
        )

        let textEncoderIndex = snapshot.appending(path: ZImageFiles.textEncoderIndex)
        let missingTextEncoderIndex = """
        { "weight_map": { "missing.weight": "model-99999-of-99999.safetensors" } }
        """
        try missingTextEncoderIndex.data(using: .utf8)!.write(to: textEncoderIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTextEncoderWeights(at: snapshot),
          expectedTextEncoder,
          "fixture=\(fixtureName)"
        )

        try FileManager.default.removeItem(at: textEncoderIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTextEncoderWeights(at: snapshot),
          expectedTextEncoder,
          "fixture=\(fixtureName)"
        )
      }
    }
  }
}
