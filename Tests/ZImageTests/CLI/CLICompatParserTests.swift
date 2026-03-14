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
      "--lora", "alibaba-pai/Z-Image-Fun-Lora-Distill",
      "--lora-file", "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors",
      "--no-progress",
    ])

    guard case .generate(let options) = command else {
      return XCTFail("Expected text generation command")
    }

    XCTAssertEqual(options.prompt, "a mountain lake")
    XCTAssertEqual(options.outputPath, "lake.png")
    XCTAssertEqual(options.steps, 12)
    XCTAssertEqual(options.guidance, 1.5)
    XCTAssertEqual(options.loraPath, "alibaba-pai/Z-Image-Fun-Lora-Distill")
    XCTAssertEqual(options.loraFile, "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors")
    XCTAssertNil(options.loraScale)
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

  func testServeParserParsesWarmServeOptions() throws {
    let command = try CLICompatParser.parseServe([
      "serve",
      "--residency-policy", "warm",
      "--warm-model", "mzbac/z-image-turbo-8bit",
      "--weights-variant", "bf16",
      "--idle-timeout", "45",
      "--max-sequence-length", "1024",
    ])

    guard case .serve(let options) = command else {
      return XCTFail("Expected serve command")
    }

    XCTAssertEqual(options.residencyPolicy, .warm)
    XCTAssertEqual(options.warmModel, "mzbac/z-image-turbo-8bit")
    XCTAssertEqual(options.warmWeightsVariant, "bf16")
    XCTAssertEqual(options.idleTimeoutSeconds, 45, accuracy: 0.001)
    XCTAssertEqual(options.warmMaxSequenceLength, 1024)
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

  func testServeParserParsesOperationalCommands() throws {
    let status = try CLICompatParser.parseServe(["--socket", "/tmp/zimage.sock", "status"])
    guard case .status(let statusSocket) = status else {
      return XCTFail("Expected status command")
    }
    XCTAssertEqual(statusSocket, "/tmp/zimage.sock")

    let cancel = try CLICompatParser.parseServe(["cancel", "job-42"])
    guard case .cancel(let cancelSocket, let jobID) = cancel else {
      return XCTFail("Expected cancel command")
    }
    XCTAssertNil(cancelSocket)
    XCTAssertEqual(jobID, "job-42")

    let shutdown = try CLICompatParser.parseServe(["shutdown"])
    guard case .shutdown(let shutdownSocket) = shutdown else {
      return XCTFail("Expected shutdown command")
    }
    XCTAssertNil(shutdownSocket)
  }

  func testServeParserParsesBatchAndMarkdownCommands() throws {
    let batch = try CLICompatParser.parseServe(["batch", "jobs.json"])
    guard case .batch(let batchSocket, let manifestPath) = batch else {
      return XCTFail("Expected batch command")
    }
    XCTAssertNil(batchSocket)
    XCTAssertEqual(manifestPath, "jobs.json")

    let markdown = try CLICompatParser.parseServe(["markdown", "prompts.md"])
    guard case .markdown(let markdownSocket, let markdownPath) = markdown else {
      return XCTFail("Expected markdown command")
    }
    XCTAssertNil(markdownSocket)
    XCTAssertEqual(markdownPath, "prompts.md")
  }

  func testServiceEnvelopeRoundTripsTextJob() throws {
    let request = ServiceRequestEnvelope(
      type: .submit,
      submission: ServiceSubmissionPayload(
        jobID: "job-1",
        job: .from(
          .text(
            TextGenerationOptions(
              prompt: "test",
              outputPath: "out.png",
              loraPath: "alibaba-pai/Z-Image-Fun-Lora-Distill",
              loraFile: "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors"
            )))
      )
    )

    let data = try JSONEncoder().encode(request)
    let decoded = try JSONDecoder().decode(ServiceRequestEnvelope.self, from: data)

    XCTAssertEqual(decoded, request)
    XCTAssertEqual(
      try decoded.submission?.job.asPayload(),
      .text(
        TextGenerationOptions(
          prompt: "test",
          outputPath: "out.png",
          loraPath: "alibaba-pai/Z-Image-Fun-Lora-Distill",
          loraFile: "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors"
        )))
  }

  func testServiceEnvelopeRoundTripsStatusAndCancelPayloads() throws {
    let statusRequest = ServiceRequestEnvelope(type: .status)
    let statusData = try JSONEncoder().encode(statusRequest)
    XCTAssertEqual(try JSONDecoder().decode(ServiceRequestEnvelope.self, from: statusData), statusRequest)

    let cancelRequest = ServiceRequestEnvelope(
      type: .cancel,
      cancellation: ServiceCancellationPayload(jobID: "job-9")
    )
    let cancelData = try JSONEncoder().encode(cancelRequest)
    XCTAssertEqual(try JSONDecoder().decode(ServiceRequestEnvelope.self, from: cancelData), cancelRequest)
  }
}
