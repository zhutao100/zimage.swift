import Foundation

public enum MarkdownFenceShell: String, Sendable, Equatable {
  case bash
  case sh
  case zsh

  var executablePath: String { "/bin/\(rawValue)" }
}

public struct MarkdownSubmission: Sendable, Equatable {
  public var jobID: String
  public var program: CLIProgramKind
  public var shell: MarkdownFenceShell
  public var commandTail: String

  public init(jobID: String, program: CLIProgramKind, shell: MarkdownFenceShell, commandTail: String) {
    self.jobID = jobID
    self.program = program
    self.shell = shell
    self.commandTail = commandTail
  }

  public func resolveJob() throws -> GenerationJobPayload {
    let tokens = try MarkdownCommandResolver.captureExpandedTokens(for: self)
    return try GenerationJobInvocationParser.parse(tokens: tokens, usage: .markdown)
  }
}

private enum MarkdownCommandResolver {
  static func captureExpandedTokens(for submission: MarkdownSubmission) throws -> [String] {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: submission.shell.executablePath)
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

    let wrapperName: String
    let programName: String
    switch submission.program {
    case .cli:
      wrapperName = "__zimage_capture_cli"
      programName = CLIProgramKind.cli.executableName
    case .serve:
      wrapperName = "__zimage_capture_serve"
      programName = CLIProgramKind.serve.executableName
    }

    process.arguments = [
      "-lc",
      shellScript(wrapperName: wrapperName, programName: programName, commandTail: submission.commandTail),
    ]

    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr

    try process.run()
    process.waitUntilExit()

    let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
    let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()

    guard process.terminationStatus == 0 else {
      let stderrText = String(data: stderrData, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
      let message = stderrText?.isEmpty == false
        ? "Command substitution failed: \(stderrText!)"
        : "Command substitution failed with exit code \(process.terminationStatus)"
      throw CLIError(message: message, usage: .markdown)
    }

    let tokens = stdoutData.split(separator: 0).map { String(decoding: $0, as: UTF8.self) }
    guard !tokens.isEmpty else {
      throw CLIError(message: "Markdown command did not resolve to a staged generation command", usage: .markdown)
    }
    return tokens
  }

  private static func shellScript(wrapperName: String, programName: String, commandTail: String) -> String {
    """
    set -f
    \(wrapperName)() {
      printf '%s\\0' \(singleQuoted(programName))
      for arg in "$@"; do
        printf '%s\\0' "$arg"
      done
    }
    \(wrapperName)\(commandTail)
    """
  }

  private static func singleQuoted(_ value: String) -> String {
    let escaped = value.replacingOccurrences(of: "'", with: "'\"'\"'")
    return "'\(escaped)'"
  }
}
