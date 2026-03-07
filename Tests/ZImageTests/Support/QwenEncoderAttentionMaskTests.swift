import MLX
import XCTest

@testable import ZImage

final class QwenEncoderAttentionMaskTests: XCTestCase {
  private func makeEncoder() -> QwenEncoder {
    QwenEncoder(
      configuration: .init(
        vocabSize: 32,
        hiddenSize: 8,
        numHiddenLayers: 1,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        intermediateSize: 16,
        maxPositionEmbeddings: 16,
        promptDropIndex: 0,
        headDim: 8
      )
    )
  }

  func testCreateAttentionMaskReturnsCausalModeWithoutPaddingMask() {
    let encoder = makeEncoder()
    let hiddenStates = MLX.zeros([1, 4, 8])

    let mask = encoder.createAttentionMask(h: hiddenStates, attentionMask: nil)

    guard case .causal = mask else {
      XCTFail("Expected causal mask when no prompt attention mask is provided")
      return
    }
  }

  func testCreateAttentionMaskCombinesCausalAndPaddingAsBoolMask() {
    let encoder = makeEncoder()
    let hiddenStates = MLX.zeros([2, 4, 8])
    let attentionMask = MLXArray(
      [Int32(1), 1, 0, 0,
       1, 0, 0, 0],
      [2, 4]
    )

    let maskMode = encoder.createAttentionMask(h: hiddenStates, attentionMask: attentionMask)

    guard case .array(let mask) = maskMode else {
      XCTFail("Expected explicit array mask when prompt attention mask is provided")
      return
    }

    XCTAssertEqual(mask.dtype, .bool)
    XCTAssertEqual(mask.shape, [2, 1, 4, 4])

    let actual = mask.asType(.int32).reshaped(-1).asArray(Int32.self)
    let expected: [Int32] = [
      1, 0, 0, 0,
      1, 1, 0, 0,
      1, 1, 0, 0,
      1, 1, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
    ]

    XCTAssertEqual(actual, expected)
  }
}
