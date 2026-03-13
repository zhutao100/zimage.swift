import Foundation
import ZImageCLICommon

public struct ServiceSubmissionPayload: Codable, Sendable, Equatable {
  public var jobID: String
  public var job: CodableGenerationJob

  public init(jobID: String, job: CodableGenerationJob) {
    self.jobID = jobID
    self.job = job
  }
}

public struct CodableGenerationJob: Codable, Sendable, Equatable {
  public enum Kind: String, Codable, Sendable {
    case text
    case control
  }

  public var kind: Kind
  public var text: TextGenerationOptions?
  public var control: ControlGenerationOptions?

  public static func from(_ payload: GenerationJobPayload) -> CodableGenerationJob {
    switch payload {
    case .text(let options):
      return .init(kind: .text, text: options, control: nil)
    case .control(let options):
      return .init(kind: .control, text: nil, control: options)
    }
  }

  public func asPayload() throws -> GenerationJobPayload {
    switch kind {
    case .text:
      guard let text else {
        throw ServiceTransportError.invalidMessage("Missing text job payload")
      }
      return .text(text)
    case .control:
      guard let control else {
        throw ServiceTransportError.invalidMessage("Missing control job payload")
      }
      return .control(control)
    }
  }
}

public enum ServiceRequestType: String, Codable, Sendable {
  case submit
}

public struct ServiceRequestEnvelope: Codable, Sendable, Equatable {
  public var type: ServiceRequestType
  public var submission: ServiceSubmissionPayload?

  public init(type: ServiceRequestType, submission: ServiceSubmissionPayload? = nil) {
    self.type = type
    self.submission = submission
  }
}

public enum ServiceEventType: String, Codable, Sendable {
  case accepted
  case progress
  case completed
  case failed
}

public struct ServiceEventEnvelope: Codable, Sendable, Equatable {
  public var type: ServiceEventType
  public var jobID: String?
  public var queuePosition: Int?
  public var progress: JobProgressUpdate?
  public var outputPath: String?
  public var message: String?

  public init(
    type: ServiceEventType,
    jobID: String? = nil,
    queuePosition: Int? = nil,
    progress: JobProgressUpdate? = nil,
    outputPath: String? = nil,
    message: String? = nil
  ) {
    self.type = type
    self.jobID = jobID
    self.queuePosition = queuePosition
    self.progress = progress
    self.outputPath = outputPath
    self.message = message
  }
}

public enum ServiceTransportError: LocalizedError, Sendable {
  case socketPathTooLong(String)
  case connectionFailed(String)
  case bindFailed(String)
  case listenFailed(String)
  case acceptFailed(String)
  case readFailed(String)
  case writeFailed(String)
  case invalidMessage(String)

  public var errorDescription: String? {
    switch self {
    case .socketPathTooLong(let message),
      .connectionFailed(let message),
      .bindFailed(let message),
      .listenFailed(let message),
      .acceptFailed(let message),
      .readFailed(let message),
      .writeFailed(let message),
      .invalidMessage(let message):
      return message
    }
  }
}

public enum ServiceSocketPath {
  public static var defaultPath: String {
    let cacheURL =
      FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
      ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Caches")
    return cacheURL.appendingPathComponent("zimage/staging.sock").path
  }

  public static func resolve(_ rawPath: String?) -> String {
    let path = rawPath ?? defaultPath
    return NSString(string: path).expandingTildeInPath
  }
}
