import Foundation

public enum GenerationJobInvocationParser {
  public static func programKind(for token: String) -> CLIProgramKind? {
    switch NSString(string: token).lastPathComponent {
    case CLIProgramKind.cli.executableName:
      return .cli
    case CLIProgramKind.serve.executableName:
      return .serve
    default:
      return nil
    }
  }

  public static func supportsProgram(_ token: String) -> Bool {
    programKind(for: token) != nil
  }

  public static func parse(tokens: [String], usage: CLIUsageTopic) throws -> GenerationJobPayload {
    guard !tokens.isEmpty else {
      throw CLIError(message: "Missing staged generation command", usage: usage)
    }

    switch programKind(for: tokens[0]) {
    case .cli:
      return try parseCLI(tokens: Array(tokens.dropFirst()), usage: usage)
    case .serve:
      return try parseServe(tokens: Array(tokens.dropFirst()), usage: usage)
    case nil:
      return try parseServe(tokens: tokens, usage: usage)
    }
  }

  private static func parseCLI(tokens: [String], usage: CLIUsageTopic) throws -> GenerationJobPayload {
    let command = try CLICompatParser.parseCLI(tokens)
    switch command {
    case .generate(let options):
      return .text(options)
    case .control(let options):
      return .control(options)
    case .help, .quantize, .quantizeControlnet:
      throw CLIError(message: "Command does not resolve to a generation request", usage: usage)
    }
  }

  private static func parseServe(tokens: [String], usage: CLIUsageTopic) throws -> GenerationJobPayload {
    let command = try CLICompatParser.parseServe(tokens)
    switch command {
    case .submit(_, let job):
      return job
    case .help, .serve, .status, .cancel, .shutdown, .batch, .markdown, .quantize, .quantizeControlnet:
      throw CLIError(message: "Command does not resolve to a generation request", usage: usage)
    }
  }
}
