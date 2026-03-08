import XCTest

@testable import ZImage

final class ModelPathsResolutionTests: XCTestCase {
  func testResolveShardListsFromIndexAndFallbackDiscovery() throws {
    let expectedLayouts: [(fixtureName: String, transformer: [String], textEncoder: [String])] = [
      (
        fixtureName: "ZImageTurbo",
        transformer: [
          "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
          "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
          "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        ],
        textEncoder: [
          "text_encoder/model-00001-of-00003.safetensors",
          "text_encoder/model-00002-of-00003.safetensors",
          "text_encoder/model-00003-of-00003.safetensors",
        ]
      ),
      (
        fixtureName: "ZImageBase",
        transformer: [
          "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
          "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        ],
        textEncoder: [
          "text_encoder/model-00001-of-00003.safetensors",
          "text_encoder/model-00002-of-00003.safetensors",
          "text_encoder/model-00003-of-00003.safetensors",
        ]
      ),
    ]

    for expected in expectedLayouts {
      let fixtureName = expected.fixtureName
      let fixtureSnapshot = TestFixtures.snapshot(named: fixtureName)

      try TestFixtures.withTemporaryCopy(of: fixtureSnapshot) { snapshot in
        for relative in expected.transformer + expected.textEncoder {
          try TestFixtures.createEmptyFile(at: snapshot.appending(path: relative))
        }

        XCTAssertEqual(
          ZImageFiles.resolveTransformerWeights(at: snapshot),
          expected.transformer,
          "fixture=\(fixtureName)"
        )
        XCTAssertEqual(
          ZImageFiles.resolveTextEncoderWeights(at: snapshot),
          expected.textEncoder,
          "fixture=\(fixtureName)"
        )

        // Index weight_map values may already include the component prefix; accept either style.
        let transformerIndex = snapshot.appending(path: ZImageFiles.transformerIndex)
        let prefixedTransformerEntries = expected.transformer.enumerated().map { offset, path in
          "\"model.diffusion_model.layers.\(offset).weight\": \"\(path)\""
        }.joined(separator: ",\n      ")
        let prefixedTransformerIndex = """
          {
            "weight_map": {
              \(prefixedTransformerEntries)
            }
          }
          """
        try prefixedTransformerIndex.data(using: .utf8)!.write(to: transformerIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTransformerWeights(at: snapshot),
          expected.transformer,
          "fixture=\(fixtureName)"
        )

        // If the index exists but points to missing shards, fall back to directory discovery.
        let missingTransformerIndex = """
          { "weight_map": { "missing.weight": "diffusion_pytorch_model-99999-of-99999.safetensors" } }
          """
        try missingTransformerIndex.data(using: .utf8)!.write(to: transformerIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTransformerWeights(at: snapshot),
          expected.transformer,
          "fixture=\(fixtureName)"
        )

        try FileManager.default.removeItem(at: transformerIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTransformerWeights(at: snapshot),
          expected.transformer,
          "fixture=\(fixtureName)"
        )

        let textEncoderIndex = snapshot.appending(path: ZImageFiles.textEncoderIndex)
        let missingTextEncoderIndex = """
          { "weight_map": { "missing.weight": "model-99999-of-99999.safetensors" } }
          """
        try missingTextEncoderIndex.data(using: .utf8)!.write(to: textEncoderIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTextEncoderWeights(at: snapshot),
          expected.textEncoder,
          "fixture=\(fixtureName)"
        )

        try FileManager.default.removeItem(at: textEncoderIndex)
        XCTAssertEqual(
          ZImageFiles.resolveTextEncoderWeights(at: snapshot),
          expected.textEncoder,
          "fixture=\(fixtureName)"
        )
      }
    }
  }
}
