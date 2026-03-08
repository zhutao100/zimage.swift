import MLX
import XCTest

@testable import ZImage

final class PipelineCFGTests: XCTestCase {
  override func setUpWithError() throws {
    try super.setUpWithError()
    try ensureMLXMetalLibraryColocated(for: type(of: self))
  }

  func testUsesClassifierFreeGuidanceTreatsPositiveScaleAsEnabled() {
    XCTAssertFalse(PipelineUtilities.usesClassifierFreeGuidance(guidanceScale: 0.0))
    XCTAssertTrue(PipelineUtilities.usesClassifierFreeGuidance(guidanceScale: 0.5))
  }

  func testEffectiveGuidanceScaleHonorsCfgTruncation() {
    XCTAssertEqual(
      PipelineUtilities.effectiveGuidanceScale(
        guidanceScale: 4.0,
        normalizedTimestep: 0.4,
        cfgTruncation: 0.5
      ),
      4.0,
      accuracy: 1e-6
    )
    XCTAssertEqual(
      PipelineUtilities.effectiveGuidanceScale(
        guidanceScale: 4.0,
        normalizedTimestep: 0.6,
        cfgTruncation: 0.5
      ),
      0.0,
      accuracy: 1e-6
    )
    XCTAssertEqual(
      PipelineUtilities.effectiveGuidanceScale(
        guidanceScale: 4.0,
        normalizedTimestep: 0.9,
        cfgTruncation: 1.2
      ),
      4.0,
      accuracy: 1e-6
    )
  }

  func testCFGNormalizationScaleClampsToPositiveNorm() {
    let scale = PipelineUtilities.cfgNormalizationScale(
      positiveNorm: 5.0,
      predictionNorm: 10.0,
      cfgNormalization: true
    )

    XCTAssertEqual(scale, 0.5, accuracy: 1e-6)
  }

  func testCFGNormalizationScaleLeavesPredictionUntouchedWhenDisabled() {
    let scale = PipelineUtilities.cfgNormalizationScale(
      positiveNorm: 5.0,
      predictionNorm: 10.0,
      cfgNormalization: false
    )

    XCTAssertEqual(scale, 1.0, accuracy: 1e-6)
  }

  func testGuidedNoisePredictionMatchesBasicCFGWhenNormalizationDisabled() {
    let positive = MLXArray([Float(2), 0], [1, 1, 1, 2])
    let negative = MLXArray([Float(1), 0], [1, 1, 1, 2])

    let prediction = PipelineUtilities.guidedNoisePrediction(
      positive: positive,
      negative: negative,
      guidanceScale: 2.0,
      cfgNormalization: false
    )

    MLX.eval(prediction)
    let values = prediction.asArray(Float.self)
    XCTAssertEqual(values[0], 4.0, accuracy: 1e-6)
    XCTAssertEqual(values[1], 0.0, accuracy: 1e-6)
  }

  func testGuidedNoisePredictionClampsNormWhenNormalizationEnabled() {
    let positive = MLXArray([Float(3), 4], [1, 1, 1, 2])
    let negative = MLXArray([Float(0), 0], [1, 1, 1, 2])

    let prediction = PipelineUtilities.guidedNoisePrediction(
      positive: positive,
      negative: negative,
      guidanceScale: 1.0,
      cfgNormalization: true
    )

    MLX.eval(prediction)
    let values = prediction.asArray(Float.self)
    XCTAssertEqual(values[0], 3.0, accuracy: 1e-5)
    XCTAssertEqual(values[1], 4.0, accuracy: 1e-5)
  }

  func testGenerationRequestDefaultsExposeCfgParityControls() {
    let request = ZImageGenerationRequest(prompt: "test")

    XCTAssertFalse(request.cfgNormalization)
    XCTAssertEqual(request.cfgTruncation, 1.0, accuracy: 1e-6)
  }

  func testControlRequestDefaultsExposeCfgParityControls() {
    let request = ZImageControlGenerationRequest(prompt: "test")

    XCTAssertFalse(request.cfgNormalization)
    XCTAssertEqual(request.cfgTruncation, 1.0, accuracy: 1e-6)
  }
}
