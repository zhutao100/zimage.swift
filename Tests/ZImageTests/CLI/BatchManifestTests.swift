import XCTest
@testable import ZImageCLICommon

final class BatchManifestTests: XCTestCase {
  func testStructuredManifestAppliesDefaults() throws {
    let manifest = try decodeManifest(
      """
      {
        "version": 1,
        "defaults": {
          "model": "mzbac/z-image-turbo-8bit",
          "width": 256,
          "height": 256,
          "steps": 9,
          "loraPath": "alibaba-pai/Z-Image-Fun-Lora-Distill",
          "loraFile": "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors"
        },
        "jobs": [
          {
            "id": "lake",
            "kind": "text",
            "prompt": "a mountain lake",
            "outputPath": "lake.png"
          }
        ]
      }
      """
    )

    let submissions = try manifest.submissions()
    XCTAssertEqual(submissions.count, 1)

    guard case .text(let options) = submissions[0].job else {
      return XCTFail("Expected text job")
    }
    XCTAssertEqual(submissions[0].jobID, "lake")
    XCTAssertEqual(options.prompt, "a mountain lake")
    XCTAssertEqual(options.model, "mzbac/z-image-turbo-8bit")
    XCTAssertEqual(options.width, 256)
    XCTAssertEqual(options.height, 256)
    XCTAssertEqual(options.steps, 9)
    XCTAssertEqual(options.loraPath, "alibaba-pai/Z-Image-Fun-Lora-Distill")
    XCTAssertEqual(options.loraFile, "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors")
    XCTAssertNil(options.loraScale)
    XCTAssertEqual(options.outputPath, "lake.png")
  }

  func testArgvJobAcceptsLeadingExecutableName() throws {
    let manifest = try decodeManifest(
      """
      {
        "version": 1,
        "jobs": [
          {
            "id": "argv-job",
            "argv": [
              "ZImageServe",
              "--prompt", "test prompt",
              "--output", "argv.png"
            ]
          }
        ]
      }
      """
    )

    let submissions = try manifest.submissions()
    XCTAssertEqual(submissions.count, 1)

    guard case .text(let options) = submissions[0].job else {
      return XCTFail("Expected text job")
    }
    XCTAssertEqual(submissions[0].jobID, "argv-job")
    XCTAssertEqual(options.prompt, "test prompt")
    XCTAssertEqual(options.outputPath, "argv.png")
  }

  private func decodeManifest(_ json: String) throws -> BatchManifest {
    try JSONDecoder().decode(BatchManifest.self, from: Data(json.utf8))
  }
}
