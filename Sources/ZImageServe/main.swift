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
      try submitJob(BatchSubmission(jobID: UUID().uuidString, job: job), socketPath: socketPath)
    case .status(let socketPath):
      let snapshot = try ServiceClient(socketPath: socketPath).status()
      print(renderStatus(snapshot))
    case .cancel(let socketPath, let jobID):
      let event = try ServiceClient(socketPath: socketPath).cancel(jobID: jobID)
      logger.info("\(event.message ?? "Cancelled \(jobID)")")
    case .shutdown(let socketPath):
      let event = try ServiceClient(socketPath: socketPath).shutdown()
      logger.info("\(event.message ?? "Shutdown acknowledged")")
    case .batch(let socketPath, let manifestPath):
      let manifest = try BatchManifest.load(from: manifestPath)
      try submitBatch(manifest.submissions(), socketPath: socketPath, sourceLabel: manifestPath)
    case .markdown(let socketPath, let markdownPath):
      let submissions = try MarkdownCommandExtractor.submissions(fromPath: markdownPath)
      try submitMarkdownBatch(submissions, socketPath: socketPath, sourceLabel: markdownPath)
    case .quantize(let options):
      logMetalDevice()
      try CLICommandRunner.runQuantize(options)
    case .quantizeControlnet(let options):
      logMetalDevice()
      try CLICommandRunner.runQuantizeControlnet(options)
    }
  }

  private static func submitBatch(_ submissions: [BatchSubmission], socketPath: String?, sourceLabel: String) throws {
    try submitResolvedBatch(
      submissions,
      socketPath: socketPath,
      sourceLabel: sourceLabel,
      jobID: \.jobID
    ) { $0 }
  }

  private static func submitMarkdownBatch(
    _ submissions: [MarkdownSubmission],
    socketPath: String?,
    sourceLabel: String
  ) throws {
    try submitResolvedBatch(
      submissions,
      socketPath: socketPath,
      sourceLabel: sourceLabel,
      jobID: \.jobID
    ) { submission in
      BatchSubmission(jobID: submission.jobID, job: try submission.resolveJob())
    }
  }

  private static func submitResolvedBatch<Submission>(
    _ submissions: [Submission],
    socketPath: String?,
    sourceLabel: String,
    jobID: KeyPath<Submission, String>,
    resolve: (Submission) throws -> BatchSubmission
  ) throws {
    let total = submissions.count
    var failures: [String] = []

    for (index, submission) in submissions.enumerated() {
      let resolvedSubmission: BatchSubmission
      do {
        resolvedSubmission = try resolve(submission)
      } catch {
        let message = CLIErrors.describe(error)
        failures.append("\(submission[keyPath: jobID]): \(message)")
        logger.error("\(message)")
        continue
      }

      logger.info("Submitting \(resolvedSubmission.jobID) (\(index + 1)/\(total)) from \(sourceLabel)")
      do {
        try submitJob(resolvedSubmission, socketPath: socketPath)
      } catch {
        let message = CLIErrors.describe(error)
        failures.append("\(resolvedSubmission.jobID): \(message)")
        logger.error("\(message)")
      }
    }

    guard failures.isEmpty else {
      throw CLIError(
        message: "Completed with \(failures.count) failed staged job(s): \(failures.joined(separator: "; "))"
      )
    }
  }

  private static func submitJob(_ submission: BatchSubmission, socketPath: String?) throws {
    let client = ServiceClient(socketPath: socketPath)
    let renderer = makeClientRenderer(for: submission.job)
    defer {
      renderer.finish()
    }

    try client.submit(job: submission.job, jobID: submission.jobID) { event in
      switch event.type {
      case .accepted:
        if let jobID = event.jobID {
          if let queuePosition = event.queuePosition, queuePosition > 0 {
            logger.info("Queued job \(jobID) at position \(queuePosition)")
          } else {
            logger.info("Accepted job \(jobID)")
          }
        }
      case .progress:
        if let progress = event.progress {
          renderer.report(progress)
        }
      case .completed:
        if let outputPath = event.outputPath {
          logger.info("Output saved to: \(outputPath)")
        }
      case .cancelled:
        throw CLIError(message: event.message ?? "Service request was cancelled")
      case .failed:
        throw CLIError(message: event.message ?? "Service request failed")
      case .status, .shutdownAcknowledged:
        break
      }
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

  private static func renderStatus(_ snapshot: ServiceStatusSnapshot) -> String {
    var lines = [
      "Socket: \(snapshot.socketPath)",
      "Residency policy: \(snapshot.residencyPolicy.rawValue)",
      "Idle timeout: \(Int(snapshot.idleTimeoutSeconds.rounded()))s",
      "Executing: \(snapshot.isExecuting ? "yes" : "no")",
      "Shutting down: \(snapshot.isShuttingDown ? "yes" : "no")",
      "Active job: \(snapshot.activeJobID ?? "none")",
    ]

    if snapshot.queuedJobIDs.isEmpty {
      lines.append("Queued jobs: none")
    } else {
      lines.append("Queued jobs: \(snapshot.queuedJobIDs.joined(separator: ", "))")
    }

    if let worker = snapshot.residentWorker {
      lines.append(
        "Resident worker: \(worker.kind) model=\(worker.model ?? ZImageRepository.id) residency=\(worker.residencyPolicy.rawValue)"
      )
    } else {
      lines.append("Resident worker: none")
    }

    return lines.joined(separator: "\n")
  }
}

ZImageServe.main()
