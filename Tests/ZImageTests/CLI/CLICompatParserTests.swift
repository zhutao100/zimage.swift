import XCTest
@testable import ZImageCLICommon
@testable import ZImageServeCore

final class CLICompatParserTests: XCTestCase {
  func testParseCLITextGenerationPreservesKeyOptions() throws {
    let command = try CLICompatParser.parseCLI([
      "--prompt", "a mountain lake",
      "--output", "lake.png",
      "--steps", "12",
      "--guidance", "1.5",
      "--no-progress",
    ])

    guard case .generate(let options) = command else {
      return XCTFail("Expected text generation command")
    }

    XCTAssertEqual(options.prompt, "a mountain lake")
    XCTAssertEqual(options.outputPath, "lake.png")
    XCTAssertEqual(options.steps, 12)
    XCTAssertEqual(options.guidance, 1.5)
    XCTAssertTrue(options.noProgress)
  }

  func testParseCLIControlRequiresWeights() throws {
    XCTAssertThrowsError(
      try CLICompatParser.parseCLI([
        "control",
        "--prompt", "test",
        "--control-image", "/tmp/control.png",
      ])
    ) { error in
      guard let cliError = error as? CLIError else {
        return XCTFail("Expected CLIError")
      }
      XCTAssertEqual(cliError.message, "Missing required --controlnet-weights argument")
      XCTAssertEqual(cliError.usage, .control)
    }
  }

  func testServeParserParsesServeCommandWithSocket() throws {
    let command = try CLICompatParser.parseServe([
      "serve",
      "--socket", "/tmp/zimage-serve.sock",
    ])

    guard case .serve(let options) = command else {
      return XCTFail("Expected serve command")
    }

    XCTAssertEqual(options.socketPath, "/tmp/zimage-serve.sock")
  }

  func testServeParserParsesAdHocSubmissionWithGlobalSocket() throws {
    let command = try CLICompatParser.parseServe([
      "--socket", "/tmp/zimage.sock",
      "--prompt", "a portrait",
      "--output", "portrait.png",
    ])

    guard case .submit(let socketPath, let job) = command else {
      return XCTFail("Expected submission command")
    }

    XCTAssertEqual(socketPath, "/tmp/zimage.sock")
    guard case .text(let options) = job else {
      return XCTFail("Expected text job")
    }
    XCTAssertEqual(options.prompt, "a portrait")
    XCTAssertEqual(options.outputPath, "portrait.png")
  }

  func testServiceEnvelopeRoundTripsTextJob() throws {
    let request = ServiceRequestEnvelope(
      type: .submit,
      submission: ServiceSubmissionPayload(
        jobID: "job-1",
        job: .from(.text(TextGenerationOptions(prompt: "test", outputPath: "out.png")))
      )
    )

    let data = try JSONEncoder().encode(request)
    let decoded = try JSONDecoder().decode(ServiceRequestEnvelope.self, from: data)

    XCTAssertEqual(decoded, request)
    XCTAssertEqual(try decoded.submission?.job.asPayload(), .text(TextGenerationOptions(prompt: "test", outputPath: "out.png")))
  }
}
