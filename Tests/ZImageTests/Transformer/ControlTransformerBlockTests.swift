import MLX
import XCTest

@testable import ZImage

final class ControlTransformerBlockTests: XCTestCase {
  private func makeBlock(blockId: Int, dim: Int = 16) -> ZImageControlTransformerBlock {
    ZImageControlTransformerBlock(
      blockId: blockId,
      dim: dim,
      nHeads: 4,
      nKvHeads: 4,
      normEps: 1e-5,
      qkNorm: true
    )
  }

  private func assertArraysClose(
    _ lhs: MLXArray,
    _ rhs: MLXArray,
    accuracy: Float = 1e-6,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    MLX.eval(lhs, rhs)
    XCTAssertEqual(lhs.shape, rhs.shape, file: file, line: line)

    let lhsValues = lhs.asArray(Float.self)
    let rhsValues = rhs.asArray(Float.self)
    XCTAssertEqual(lhsValues.count, rhsValues.count, file: file, line: line)

    let maxDifference = zip(lhsValues, rhsValues).reduce(Float.zero) { current, pair in
      max(current, abs(pair.0 - pair.1))
    }
    XCTAssertLessThan(maxDifference, accuracy, file: file, line: line)
  }

  func testStateAccumulationMatchesLegacyStackingAcrossBlocks() {
    let block0 = makeBlock(blockId: 0)
    let block1 = makeBlock(blockId: 1)

    let x = MLXRandom.normal([1, 4, 16])
    let initialControl = MLXRandom.normal([1, 4, 16])

    let stateAfterBlock0 = block0(
      ZImageControlHintState(control: initialControl),
      x: x,
      attnMask: nil,
      freqsCis: nil,
      adalnInput: nil
    )
    let legacyAfterBlock0 = block0(
      initialControl,
      x: x,
      attnMask: nil,
      freqsCis: nil,
      adalnInput: nil
    )
    assertArraysClose(stateAfterBlock0.stacked(), legacyAfterBlock0)

    let stateAfterBlock1 = block1(
      stateAfterBlock0,
      x: x,
      attnMask: nil,
      freqsCis: nil,
      adalnInput: nil
    )
    let legacyAfterBlock1 = block1(
      legacyAfterBlock0,
      x: x,
      attnMask: nil,
      freqsCis: nil,
      adalnInput: nil
    )
    assertArraysClose(stateAfterBlock1.stacked(), legacyAfterBlock1)
    XCTAssertEqual(stateAfterBlock1.hints.count, 2)
  }

  func testScaledHintsPreservesLayerOrderingAndScale() {
    let hint0 = MLXRandom.normal([1, 4, 8])
    let hint1 = MLXRandom.normal([1, 4, 8])
    let control = MLXRandom.normal([1, 4, 8])

    let state = ZImageControlHintState(control: control, hints: [hint0, hint1])
    let samples = state.scaledHints(layerPlaces: [2, 5], conditioningScale: 0.75)

    XCTAssertEqual(Set(samples.keys), Set([2, 5]))
    assertArraysClose(samples[2]!, hint0 * 0.75)
    assertArraysClose(samples[5]!, hint1 * 0.75)
  }
}
