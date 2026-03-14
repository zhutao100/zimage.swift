import Logging
import XCTest
@testable import ZImage
@testable import ZImageCLICommon

final class CLICommandRunnerTests: XCTestCase {
  func testBuildTextExecutionPlanAutoAppliesKnownDistillRecipeWhenUnset() {
    let store = TestLogStore()
    let logger = Logger(label: "test.auto-apply") { _ in TestLogHandler(store: store) }

    let plan = CLICommandRunner.buildTextExecutionPlan(
      TextGenerationOptions(
        prompt: "a mountain lake",
        outputPath: "lake.png",
        model: "Tongyi-MAI/Z-Image",
        loraPath: "alibaba-pai/Z-Image-Fun-Lora-Distill"
      ),
      logger: logger
    )

    XCTAssertEqual(plan.request.steps, 8)
    XCTAssertEqual(plan.request.guidanceScale, 1.0, accuracy: 0.0001)
    XCTAssertEqual(Double(plan.request.lora?.scale ?? -1), 0.8, accuracy: 0.0001)
    XCTAssertTrue(store.messages.contains { $0.contains("Auto-applying known adapter recipe") })
  }

  func testDistillAdapterWarningMentionsExplicitOverrides() {
    let warning = CLICommandRunner.loraSamplingWarning(
      loraConfig: .huggingFace(
        "alibaba-pai/Z-Image-Fun-Lora-Distill",
        filename: "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors",
        scale: 1.0
      ),
      steps: 9,
      guidance: 1.5,
      loraScale: 1.0,
      preset: .zImage
    )

    XCTAssertNotNil(warning)
    XCTAssertTrue(warning?.contains("--steps 8 --guidance 1.0 --lora-scale 0.8") == true)
    XCTAssertTrue(warning?.contains("Explicit overrides are in effect") == true)
    XCTAssertTrue(warning?.contains("--steps 9") == true)
    XCTAssertTrue(warning?.contains("--guidance 1.5") == true)
  }

  func testDistillAdapterWarningClearsWhenRecipeAlreadyMatches() {
    let warning = CLICommandRunner.loraSamplingWarning(
      loraConfig: .local(
        "/tmp/Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors",
        scale: 0.8
      ),
      steps: 8,
      guidance: 1.0,
      loraScale: 0.8,
      preset: .zImage
    )

    XCTAssertNil(warning)
  }

  func testGenericLoRAWarningStillUsesModelDefaultsMessage() {
    let warning = CLICommandRunner.loraSamplingWarning(
      loraConfig: .huggingFace("ostris/z_image_turbo_childrens_drawings"),
      steps: nil,
      guidance: nil,
      loraScale: nil,
      preset: .zImageTurbo
    )

    XCTAssertNotNil(warning)
    XCTAssertTrue(warning?.contains("Using model defaults with LoRA") == true)
  }
}

private final class TestLogStore: @unchecked Sendable {
  var messages: [String] = []
}

private struct TestLogHandler: LogHandler {
  let store: TestLogStore
  var metadata: Logger.Metadata = [:]
  var logLevel: Logger.Level = .trace

  subscript(metadataKey key: String) -> Logger.Metadata.Value? {
    get { metadata[key] }
    set { metadata[key] = newValue }
  }

  func log(
    level: Logger.Level,
    message: Logger.Message,
    metadata _: Logger.Metadata?,
    source _: String,
    file _: String,
    function _: String,
    line _: UInt
  ) {
    store.messages.append("[\(level)] \(message)")
  }
}
