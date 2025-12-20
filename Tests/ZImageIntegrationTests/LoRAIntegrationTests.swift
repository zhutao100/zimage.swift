import XCTest
import MLX
@testable import ZImage

final class LoRAIntegrationTests: XCTestCase {
  nonisolated(unsafe) private static var sharedPipeline: ZImagePipeline?
  private static let outputDir: URL = {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("ZImageIntegrationTests")
      .appendingPathComponent("LoRAIntegrationTests")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }()
  override class func setUp() {
    super.setUp()

    if ProcessInfo.processInfo.environment["CI"] == nil {
      sharedPipeline = ZImagePipeline()
    }
  }

  override class func tearDown() {

    sharedPipeline = nil

    try? FileManager.default.removeItem(at: outputDir)
    super.tearDown()
  }
  private func getPipeline() throws -> ZImagePipeline {
    guard let pipeline = Self.sharedPipeline else {
      throw XCTSkip("Pipeline not available (likely CI environment)")
    }
    return pipeline
  }
  func testLoRAStyleApplication() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let loraConfig = getTestLoRAConfiguration()
    let tempOutput = Self.outputDir.appendingPathComponent("test_lora.png")

    let request = ZImageGenerationRequest(
      prompt: "a lion",
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      lora: loraConfig
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  func testLoRAProducesDifferentOutput() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let loraConfig = getTestLoRAConfiguration()
    let seed: UInt64 = 123456
    let noLoraOutput = Self.outputDir.appendingPathComponent("test_no_lora.png")

    let requestNoLora = ZImageGenerationRequest(
      prompt: "a lion",
      width: 256,
      height: 256,
      steps: 9,
      seed: seed,
      outputPath: noLoraOutput,
      model: "mzbac/z-image-turbo-8bit"
    )
    _ = try await pipeline.generate(requestNoLora)
    let withLoraOutput = Self.outputDir.appendingPathComponent("test_with_lora.png")

    let requestWithLora = ZImageGenerationRequest(
      prompt: "a lion",
      width: 256,
      height: 256,
      steps: 9,
      seed: seed,
      outputPath: withLoraOutput,
      model: "mzbac/z-image-turbo-8bit",
      lora: loraConfig
    )
    _ = try await pipeline.generate(requestWithLora)
    XCTAssertTrue(FileManager.default.fileExists(atPath: noLoraOutput.path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: withLoraOutput.path))
    let dataNoLora = try Data(contentsOf: noLoraOutput)
    let dataWithLora = try Data(contentsOf: withLoraOutput)
    XCTAssertNotEqual(dataNoLora, dataWithLora, "LoRA should produce different output")
  }

  func testInvalidLoRAPath() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let tempOutput = Self.outputDir.appendingPathComponent("test_invalid_lora.png")

    let request = ZImageGenerationRequest(
      prompt: "test",
      width: 256,
      height: 256,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      lora: .local("/nonexistent/path/to/lora")
    )
    do {
      _ = try await pipeline.generate(request)
      XCTFail("Should have thrown an error for invalid LoRA path")
    } catch {

      XCTAssertTrue(true)
    }
  }

  func testLoRAConfigurationLocal() {
    let config = LoRAConfiguration.local("/path/to/lora.safetensors")
    XCTAssertEqual(config.scale, 1.0)
    XCTAssertTrue(config.source.isLocal)
  }

  func testLoRAConfigurationHuggingFace() {
    let config = LoRAConfiguration.huggingFace("ostris/z_image_turbo_childrens_drawings")
    XCTAssertEqual(config.scale, 1.0)
    XCTAssertFalse(config.source.isLocal)
  }

  func testLoRAConfigurationWithScale() {
    let config = LoRAConfiguration.local("/path/to/lora", scale: 0.5)
    XCTAssertEqual(config.scale, 0.5)
  }

  func testLoRAConfigurationScaleClamped() {
    let configLow = LoRAConfiguration.local("/path/to/lora", scale: -0.5)
    XCTAssertEqual(configLow.scale, 0.0)

    let configHigh = LoRAConfiguration.local("/path/to/lora", scale: 1.5)
    XCTAssertEqual(configHigh.scale, 1.0)
  }

  private func getTestLoRAConfiguration() -> LoRAConfiguration {

    if let envPath = ProcessInfo.processInfo.environment["ZIMAGE_TEST_LORA_PATH"] {
      return .local(envPath)
    }
    return .huggingFace("ostris/z_image_turbo_childrens_drawings")
  }

  private func skipIfNoGPU() throws {
    if ProcessInfo.processInfo.environment["CI"] != nil {
      throw XCTSkip("Skipping GPU-intensive test in CI environment")
    }
  }
}
