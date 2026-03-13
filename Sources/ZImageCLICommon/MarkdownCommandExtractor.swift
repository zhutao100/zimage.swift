import Foundation

public enum MarkdownCommandExtractor {
  private struct Fence {
    let marker: Character
    let count: Int
  }

  private static let supportedPrograms = ["ZImageCLI", "ZImageServe"]

  private static func parseOpeningFence(from trimmedLine: String) -> (fence: Fence, shell: MarkdownFenceShell?)? {
    guard let marker = trimmedLine.first, marker == "`" || marker == "~" else { return nil }

    let count = trimmedLine.prefix { $0 == marker }.count
    guard count >= 3 else { return nil }

    let info = trimmedLine.dropFirst(count).trimmingCharacters(in: .whitespacesAndNewlines)
    let shell = info.split(separator: " ").first.flatMap { MarkdownFenceShell(rawValue: String($0).lowercased()) }
    return (Fence(marker: marker, count: count), shell)
  }

  private static func isClosingFence(_ trimmedLine: String, matching fence: Fence) -> Bool {
    guard trimmedLine.allSatisfy({ $0 == fence.marker }) else { return false }
    return trimmedLine.count >= fence.count
  }

  private static func containsSupportedProgramReference(_ block: String) -> Bool {
    supportedPrograms.contains { block.contains($0) }
  }

  private static func containsWrappedSupportedProgram(_ tokens: [ShellWordsLexer.LexedToken]) -> Bool {
    tokens.dropFirst().contains { GenerationJobInvocationParser.supportsProgram($0.text) }
  }

  public static func submissions(fromPath path: String) throws -> [MarkdownSubmission] {
    let expandedPath = NSString(string: path).expandingTildeInPath
    let url = URL(fileURLWithPath: expandedPath)
    let markdown = try String(contentsOf: url, encoding: .utf8)
    return try submissions(from: markdown)
  }

  public static func submissions(from markdown: String) throws -> [MarkdownSubmission] {
    var submissions: [MarkdownSubmission] = []
    var currentFence: Fence?
    var currentShell: MarkdownFenceShell?
    var currentLines: [String] = []
    var currentFenceLine = 0

    for (lineIndex, rawLine) in markdown.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
      let line = String(rawLine)
      let trimmed = line.trimmingCharacters(in: .whitespaces)

      if currentFence == nil {
        guard let (fence, shell) = parseOpeningFence(from: trimmed) else { continue }
        if let shell {
          currentFence = fence
          currentShell = shell
          currentLines.removeAll(keepingCapacity: true)
          currentFenceLine = lineIndex + 1
        }
        continue
      }

      if let activeFence = currentFence, isClosingFence(trimmed, matching: activeFence) {
        let block = currentLines.joined(separator: "\n")
        if containsSupportedProgramReference(block) {
          let tokens = try ShellWordsLexer.lexSingleCommandTokens(block)
          if let firstToken = tokens.first,
            let program = GenerationJobInvocationParser.programKind(for: firstToken.text)
          {
            guard let currentShell else {
              throw CLIError(message: "Missing markdown shell language for fenced command", usage: .markdown)
            }
            submissions.append(
              MarkdownSubmission(
                jobID: "markdown-\(submissions.count + 1)",
                program: program,
                shell: currentShell,
                commandTail: String(block[firstToken.rawRange.upperBound...])
              ))
          } else if containsWrappedSupportedProgram(tokens) {
            throw CLIError(
              message:
                "Markdown generation commands must invoke ZImageCLI or ZImageServe directly (fence starting at line \(currentFenceLine))",
              usage: .markdown
            )
          }
        }
        currentFence = nil
        currentShell = nil
        currentLines.removeAll(keepingCapacity: true)
        currentFenceLine = 0
        continue
      }

      currentLines.append(line)
    }

    if currentFence != nil {
      throw CLIError(message: "Unterminated fenced shell block in markdown file", usage: .markdown)
    }

    guard !submissions.isEmpty else {
      throw CLIError(
        message: "No fenced ZImageCLI or ZImageServe generation commands were found in the markdown file",
        usage: .markdown
      )
    }
    return submissions
  }
}
