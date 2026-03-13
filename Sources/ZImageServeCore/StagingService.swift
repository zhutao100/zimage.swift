import Darwin
import Dispatch
import Foundation
import Logging
import ZImageCLICommon

private final class AsyncBox<T>: @unchecked Sendable {
  var value: T

  init(_ value: T) {
    self.value = value
  }
}

public final class ServiceClient {
  private let socketPath: String
  private let decoder = JSONDecoder()
  private let encoder = JSONEncoder()

  public init(socketPath: String? = nil) {
    self.socketPath = ServiceSocketPath.resolve(socketPath)
  }

  public func submit(
    job: GenerationJobPayload,
    jobID: String = UUID().uuidString,
    eventHandler: (ServiceEventEnvelope) throws -> Void
  ) throws {
    let connection = try UnixDomainSocket.connect(path: socketPath)
    let request = ServiceRequestEnvelope(
      type: .submit,
      submission: ServiceSubmissionPayload(jobID: jobID, job: CodableGenerationJob.from(job))
    )
    let payload = try encoder.encode(request)
    guard let line = String(data: payload, encoding: .utf8) else {
      throw ServiceTransportError.invalidMessage("Failed to encode service request")
    }
    try connection.writeLine(line)

    while let responseLine = try connection.readLine() {
      let event = try decoder.decode(ServiceEventEnvelope.self, from: Data(responseLine.utf8))
      try eventHandler(event)
      if event.type == .completed || event.type == .failed {
        return
      }
    }
  }
}

public final class StagingServiceDaemon {
  private let socketPath: String
  private let logger: Logger
  private let coordinator: SerialServiceCoordinator

  public init(options: ServeOptions, logger: Logger) {
    self.socketPath = ServiceSocketPath.resolve(options.socketPath)
    self.logger = logger
    self.coordinator = SerialServiceCoordinator(logger: logger)
  }

  public func run() throws {
    let serverFD = try UnixDomainSocket.makeServerSocket(path: socketPath)
    defer {
      close(serverFD)
      unlink(socketPath)
    }

    logger.info("Staging daemon listening on \(socketPath)")

    while true {
      let connection: SocketConnection
      do {
        connection = try UnixDomainSocket.acceptClient(serverFD: serverFD)
      } catch let error as ServiceTransportError {
        if case .acceptFailed(let message) = error, message.contains("Bad file descriptor") {
          return
        }
        throw error
      }

      let coordinator = self.coordinator
      let logger = self.logger
      do {
        try waitForAsync {
          try await Self.handleConnection(connection, coordinator: coordinator, logger: logger)
        }
      } catch {
        logger.error("\(CLIErrors.describe(error))")
      }
    }
  }

  private static func handleConnection(
    _ connection: SocketConnection,
    coordinator: SerialServiceCoordinator,
    logger: Logger
  ) async throws {
    let decoder = JSONDecoder()
    let encoder = JSONEncoder()

    guard let line = try connection.readLine() else {
      throw ServiceTransportError.invalidMessage("Missing request payload")
    }
    let request = try decoder.decode(ServiceRequestEnvelope.self, from: Data(line.utf8))
    switch request.type {
    case .submit:
      guard let submission = request.submission else {
        throw ServiceTransportError.invalidMessage("Missing submission payload")
      }
      try await coordinator.submit(submission) { event in
        let data = try encoder.encode(event)
        guard let line = String(data: data, encoding: .utf8) else {
          throw ServiceTransportError.invalidMessage("Failed to encode event payload")
        }
        try connection.writeLine(line)
      }
    }
  }
}

private func waitForAsync(_ operation: @escaping @Sendable () async throws -> Void) throws {
  let semaphore = DispatchSemaphore(value: 0)
  let errorBox = AsyncBox<Error?>(nil)
  Task {
    do {
      try await operation()
    } catch {
      errorBox.value = error
    }
    semaphore.signal()
  }
  semaphore.wait()
  if let error = errorBox.value {
    throw error
  }
}

actor SerialServiceCoordinator {
  private let logger: Logger
  private var isBusy = false
  private var waiters: [CheckedContinuation<Void, Never>] = []

  init(logger: Logger) {
    self.logger = logger
  }

  func submit(
    _ submission: ServiceSubmissionPayload,
    eventSink: @escaping @Sendable (ServiceEventEnvelope) throws -> Void
  ) async throws {
    let queuePosition = isBusy ? waiters.count + 1 : 0
    try eventSink(.init(type: .accepted, jobID: submission.jobID, queuePosition: queuePosition))

    if isBusy {
      await withCheckedContinuation { continuation in
        waiters.append(continuation)
      }
    } else {
      isBusy = true
    }

    defer {
      if waiters.isEmpty {
        isBusy = false
      } else {
        let next = waiters.removeFirst()
        next.resume()
      }
    }

    let job = try submission.job.asPayload()
    do {
      let outputURL = try await CLICommandRunner.executeGenerationJob(
        job,
        logger: logger,
        progressSink: { update in
          try? eventSink(.init(type: .progress, jobID: submission.jobID, progress: update))
        }
      )
      try eventSink(.init(type: .completed, jobID: submission.jobID, outputPath: outputURL.path))
    } catch {
      try eventSink(
        .init(type: .failed, jobID: submission.jobID, message: CLIErrors.describe(error))
      )
    }
  }
}
