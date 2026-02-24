import Logging
import MLX
import XCTest
@testable import ZImage

final class AIOTransformerGuardsTests: XCTestCase {
  private func makeConfig(qkNorm: Bool) -> ZImageTransformerConfig {
    ZImageTransformerConfig(
      inChannels: 4,
      dim: 4,
      nLayers: 1,
      nRefinerLayers: 0,
      nHeads: 1,
      nKVHeads: 1,
      normEps: 1e-5,
      qkNorm: qkNorm,
      capFeatDim: 4,
      ropeTheta: 10000,
      tScale: 1.0,
      axesDims: [2],
      axesLens: [1]
    )
  }

  func testStrictAIORequiresQKNormSentinelsWhenEnabled() {
    let w = MLXArray([Float(0.0)])
    let weights: [String: MLXArray] = [
      "layers.0.attention.to_q.weight": w,
      "layers.0.attention.to_out.0.weight": w,
    ]

    let missing = ZImageAIOTransformerValidation.missingStrictRequiredKeys(in: weights, config: makeConfig(qkNorm: true))
    XCTAssertTrue(missing.contains("layers.0.attention.norm_q.weight"))
    XCTAssertTrue(missing.contains("layers.0.attention.norm_k.weight"))
  }

  func testStrictAIOAllowsMissingQKNormWhenDisabled() {
    let w = MLXArray([Float(0.0)])
    let weights: [String: MLXArray] = [
      "layers.0.attention.to_q.weight": w,
      "layers.0.attention.to_out.0.weight": w,
    ]

    let missing = ZImageAIOTransformerValidation.missingStrictRequiredKeys(in: weights, config: makeConfig(qkNorm: false))
    XCTAssertTrue(missing.isEmpty)
  }

  func testAIOTransformerCoverageThrowsWhenTooLow() {
    let config = makeConfig(qkNorm: true)
    let transformer = ZImageTransformer2DModel(configuration: config)
    let w = MLXArray([Float(0.0)])
    let weights: [String: MLXArray] = [
      "layers.0.attention.to_q.weight": w,
      "layers.0.attention.to_out.0.weight": w,
      "layers.0.attention.norm_q.weight": w,
      "layers.0.attention.norm_k.weight": w,
    ]

    let (coverage, _) = aioTransformerCoverage(weights: weights, transformer: transformer)
    XCTAssertLessThan(coverage, 0.99)
  }

  func testAIOTransformerCoveragePassesWhenFull() {
    let config = makeConfig(qkNorm: true)
    let transformer = ZImageTransformer2DModel(configuration: config)
    let placeholder = MLXArray([Float(0.0)])

    var weights: [String: MLXArray] = [:]
    for (key, _) in transformer.parameters().flattened() {
      weights[key] = placeholder
    }

    let (coverage, total) = aioTransformerCoverage(weights: weights, transformer: transformer)
    XCTAssertGreaterThanOrEqual(total, 1)
    XCTAssertGreaterThanOrEqual(coverage, 0.99)
  }

  func testAIOTransformerCoverageAcceptsCapEmbedderAliases() {
    let config = makeConfig(qkNorm: true)
    let transformer = ZImageTransformer2DModel(configuration: config)
    let placeholder = MLXArray([Float(0.0)])

    var weights: [String: MLXArray] = [:]
    for (key, _) in transformer.parameters().flattened() {
      weights[key] = placeholder
    }

    XCTAssertNotNil(weights.removeValue(forKey: "capEmbedNorm.weight"))
    XCTAssertNotNil(weights.removeValue(forKey: "capEmbedLinear.weight"))
    XCTAssertNotNil(weights.removeValue(forKey: "capEmbedLinear.bias"))
    weights["cap_embedder.0.weight"] = placeholder
    weights["cap_embedder.1.weight"] = placeholder
    weights["cap_embedder.1.bias"] = placeholder

    let (coverage, _) = aioTransformerCoverage(weights: weights, transformer: transformer)
    XCTAssertGreaterThanOrEqual(coverage, 0.99)
  }

  private func aioTransformerCoverage(
    weights: [String: MLXArray],
    transformer: ZImageTransformer2DModel
  ) -> (coverage: Double, total: Int) {
    let auditWeights = ZImageAIOTransformerValidation.coverageAuditWeights(weights)
    var logger = Logger(label: "test")
    logger.logLevel = .error
    let audit = WeightsAudit.audit(module: transformer, weights: auditWeights, logger: logger, sample: 10)
    let total = audit.matched + audit.missing.count
    let coverage = total > 0 ? Double(audit.matched) / Double(total) : 0.0
    return (coverage: coverage, total: total)
  }
}
