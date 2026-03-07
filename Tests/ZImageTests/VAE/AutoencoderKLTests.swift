import MLX
import XCTest

@testable import ZImage

final class AutoencoderKLTests: XCTestCase {
  func testDisableMidBlockAttentionPreservesEncodeShape() {
    let vae = AutoencoderKL(
      configuration: .init(
        inChannels: 3,
        outChannels: 3,
        latentChannels: 4,
        blockOutChannels: [8, 16],
        layersPerBlock: 1,
        normNumGroups: 1,
        sampleSize: 32,
        midBlockAddAttention: true
      )
    )
    let image = MLXArray.ones([1, 3, 32, 32], dtype: .float32)

    let baseline = vae.encode(image)
    let diagnostic = vae.encode(image, disableMidBlockAttention: true)
    MLX.eval(baseline, diagnostic)

    XCTAssertEqual(baseline.shape, diagnostic.shape)

    let delta = zip(baseline.asArray(Float.self), diagnostic.asArray(Float.self)).reduce(Float.zero) {
      partial, pair in
      partial + abs(pair.0 - pair.1)
    }
    XCTAssertGreaterThan(delta, 0)
  }
}
