import Darwin
import Foundation
import Logging
import Metal
import ZImage
import ZImageCLICommon
import ZImageServeCore

LoggingSystem.bootstrap { label in
  var handler = StreamLogHandler.standardError(label: label)
  handler.logLevel = .info
  return handler
}

enum ZImageServe {
  static let logger: Logger = {
    var logger = Logger(label: "z-image.serve")
    logger.logLevel = .info
    return logger
  }()

  static func main() -> Never {
    do {
      try run()
      Darwin.exit(EXIT_SUCCESS)
    } catch let error as CLIError {
      if !error.message.isEmpty {
        logger.error("\(error.message)")
      }
      if let usage = error.usage {
        print(CLIUsageFormatter.usage(for: usage, program: .serve))
      }
      Darwin.exit(EXIT_FAILURE)
    } catch {
      logger.error("\(CLIErrors.describe(error))")
      Darwin.exit(EXIT_FAILURE)
    }
  }

  static func run() throws {
    let command = try CLICompatParser.parseServe(Array(CommandLine.arguments.dropFirst()))
    switch command {
    case .help(let usage):
      print(CLIUsageFormatter.usage(for: usage, program: .serve))
    case .serve(let options):
      logMetalDevice()
      let daemon = StagingServiceDaemon(options: options, logger: logger)
      try daemon.run()
    case .submit(let socketPath, let job):
      let client = ServiceClient(socketPath: socketPath)
      let renderer = makeClientRenderer(for: job)
      defer {
        renderer.finish()
      }

      try client.submit(job: job) { event in
        switch event.type {
        case .accepted:
          if let queuePosition = event.queuePosition, queuePosition > 0, let jobID = event.jobID {
            logger.info("Queued job \(jobID) at position \(queuePosition)")
          }
        case .progress:
          if let progress = event.progress {
            renderer.report(progress)
          }
        case .completed:
          if let outputPath = event.outputPath {
            logger.info("Output saved to: \(outputPath)")
          }
        case .failed:
          throw CLIError(message: event.message ?? "Service request failed", usage: nil)
        }
      }
    case .quantize(let options):
      logMetalDevice()
      try CLICommandRunner.runQuantize(options)
    case .quantizeControlnet(let options):
      logMetalDevice()
      try CLICommandRunner.runQuantizeControlnet(options)
    }
  }

  private static func logMetalDevice() {
    if let dev = MTLCreateSystemDefaultDevice() {
      logger.info("Metal device: \(dev.name)")
    } else {
      logger.warning("No Metal device detected; MLX will fall back to CPU.")
    }
  }

  private static func makeClientRenderer(for job: GenerationJobPayload) -> TerminalProgressRenderer {
    let noProgress: Bool
    let totalSteps: Int
    switch job {
    case .text(let options):
      let preset = ZImagePreset.resolved(
        for: options.model,
        width: options.width,
        height: options.height,
        steps: options.steps,
        guidanceScale: options.guidance,
        maxSequenceLength: options.maxSequenceLength
      )
      noProgress = options.noProgress
      totalSteps = preset.steps
    case .control(let options):
      let preset = ZImagePreset.resolved(
        for: options.model,
        width: options.width,
        height: options.height,
        steps: options.steps,
        guidanceScale: options.guidance,
        maxSequenceLength: options.maxSequenceLength
      )
      noProgress = options.noProgress
      totalSteps = preset.steps
    }
    return TerminalProgressRenderer(noProgress: noProgress, totalSteps: totalSteps)
  }
}

ZImageServe.main()
