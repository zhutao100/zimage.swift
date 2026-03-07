import MLX
import XCTest

@testable import ZImage

final class VAEAttentionTests: XCTestCase {
  func testChunkedScaledDotProductAttentionMatchesFullAttention() {
    let shape = [1, 1, 17, 8]
    let queries = MLXArray(0 ..< 136, shape).asType(.float32)
    let keys = (MLXArray(0 ..< 136, shape).asType(.float32) + 1) / MLXArray(Float(7))
    let values = (MLXArray(0 ..< 136, shape).asType(.float32) + 3) / MLXArray(Float(11))
    let scale = Float(1) / sqrt(Float(shape[3]))

    let full = VAEAttention.scaledDotProductAttention(
      queries: queries,
      keys: keys,
      values: values,
      scale: scale,
      queryChunkSize: nil
    )
    let chunked = VAEAttention.scaledDotProductAttention(
      queries: queries,
      keys: keys,
      values: values,
      scale: scale,
      queryChunkSize: 4
    )
    MLX.eval(full, chunked)

    XCTAssertEqual(full.shape, chunked.shape)

    let fullValues = full.asArray(Float.self)
    let chunkedValues = chunked.asArray(Float.self)
    XCTAssertEqual(fullValues.count, chunkedValues.count)

    let maxDifference = zip(fullValues, chunkedValues).reduce(Float.zero) { current, pair in
      max(current, abs(pair.0 - pair.1))
    }
    XCTAssertLessThan(maxDifference, 1e-4)
  }
}
