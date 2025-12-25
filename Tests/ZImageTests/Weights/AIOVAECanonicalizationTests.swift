import XCTest
import MLX
@testable import ZImage

final class AIOVAECanonicalizationTests: XCTestCase {

  func testCanonicalizeComfyUIVAENamesToDiffusers() {
    let linearLike = MLXArray([Float](repeating: 0.0, count: 4), [2, 2, 1, 1]).asType(.bfloat16)
    let convLike = MLXArray([Float](repeating: 0.0, count: 4), [2, 2, 1, 1]).asType(.bfloat16)
    let weights: [String: MLXArray] = [
      "decoder.mid.attn_1.q.weight": linearLike,
      "decoder.mid.attn_1.norm.weight": MLXArray([Float](repeating: 0.0, count: 2), [2]).asType(.bfloat16),
      "decoder.mid.attn_1.proj_out.weight": linearLike,
      "decoder.mid.block_1.conv1.weight": convLike,
      "decoder.mid.block_2.norm2.bias": MLXArray([Float](repeating: 0.0, count: 2), [2]).asType(.bfloat16),
      "decoder.norm_out.weight": MLXArray([Float](repeating: 0.0, count: 2), [2]).asType(.bfloat16),
      "decoder.up.3.upsample.conv.weight": convLike,
      "decoder.up.0.block.0.conv1.weight": convLike,
      "decoder.up.0.block.0.nin_shortcut.weight": convLike,
    ]

    let mapped = ZImageAIOCheckpoint.canonicalizeVAEWeights(weights, expectedUpBlocks: 4, logger: nil)

    XCTAssertEqual(mapped["decoder.mid_block.attentions.0.to_q.weight"]?.shape, [2, 2])
    XCTAssertNotNil(mapped["decoder.mid_block.attentions.0.group_norm.weight"])
    XCTAssertEqual(mapped["decoder.mid_block.attentions.0.to_out.0.weight"]?.shape, [2, 2])
    XCTAssertNotNil(mapped["decoder.mid_block.resnets.0.conv1.weight"])
    XCTAssertNotNil(mapped["decoder.mid_block.resnets.1.norm2.bias"])
    XCTAssertNotNil(mapped["decoder.conv_norm_out.weight"])

    // Up blocks are reversed: up.3 -> up_blocks.0 and up.0 -> up_blocks.3
    XCTAssertEqual(mapped["decoder.up_blocks.0.upsamplers.0.conv.weight"]?.shape, [2, 2, 1, 1])
    XCTAssertNotNil(mapped["decoder.up_blocks.3.resnets.0.conv1.weight"])
    XCTAssertEqual(mapped["decoder.up_blocks.3.resnets.0.conv_shortcut.weight"]?.shape, [2, 2, 1, 1])

    XCTAssertNil(mapped["decoder.mid.attn_1.q.weight"])
    XCTAssertNil(mapped["decoder.norm_out.weight"])
    XCTAssertNil(mapped["decoder.up.3.upsample.conv.weight"])
    XCTAssertNil(mapped["decoder.up.0.block.0.nin_shortcut.weight"])
  }

  func testCanonicalizeNoOpForDiffusersStyleKeys() {
    let dummy = MLXArray([Float(0.0)], [1]).asType(.bfloat16)
    let weights: [String: MLXArray] = [
      "decoder.mid_block.attentions.0.to_q.weight": dummy,
      "decoder.up_blocks.0.upsamplers.0.conv.weight": dummy,
    ]

    let mapped = ZImageAIOCheckpoint.canonicalizeVAEWeights(weights, expectedUpBlocks: 4, logger: nil)
    XCTAssertEqual(Set(mapped.keys), Set(weights.keys))
  }
}
