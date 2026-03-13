import Foundation

public enum ShellWordsLexer {
  private enum QuoteMode {
    case none
    case single
    case double
  }

  struct LexedToken {
    var text: String
    var rawRange: Range<String.Index>
  }

  public static func lexSingleCommand(_ script: String, usage: CLIUsageTopic = .markdown) throws -> [String] {
    try lexSingleCommandTokens(script, usage: usage).map(\.text)
  }

  static func lexSingleCommandTokens(_ script: String, usage: CLIUsageTopic = .markdown) throws -> [LexedToken] {
    var tokens: [LexedToken] = []
    var current = ""
    var mode: QuoteMode = .none
    var index = script.startIndex
    var pendingCommandBreak = false
    var tokenStart: String.Index?

    func markTokenStartIfNeeded(at position: String.Index) {
      if tokenStart == nil {
        tokenStart = position
      }
    }

    func finishToken(at endIndex: String.Index) {
      guard let start = tokenStart else {
        current.removeAll(keepingCapacity: true)
        return
      }
      tokens.append(LexedToken(text: current, rawRange: start..<endIndex))
      current.removeAll(keepingCapacity: true)
      tokenStart = nil
    }

    func rejectDollarSequence(at index: String.Index) throws {
      let nextIndex = script.index(after: index)
      guard nextIndex < script.endIndex else { return }

      let next = script[nextIndex]
      if next == "(" {
        let afterOpenIndex = script.index(after: nextIndex)
        if afterOpenIndex < script.endIndex, script[afterOpenIndex] == "(" {
          throw CLIError(message: "Shell expansion is not supported in markdown commands", usage: usage)
        }
        return
      }
      if next == "{" || next == "_" || next.isLetter || next.isNumber
        || next == "@" || next == "*" || next == "#" || next == "?" || next == "-" || next == "!"
        || next == "$"
      {
        throw CLIError(message: "Shell expansion is not supported in markdown commands", usage: usage)
      }
    }

    func commandSubstitutionEnd(startingAt startIndex: String.Index) throws -> String.Index {
      let character = script[startIndex]
      if character == "`" {
        var cursor = script.index(after: startIndex)
        while cursor < script.endIndex {
          let currentCharacter = script[cursor]
          if currentCharacter == "\\" {
            cursor = script.index(after: cursor)
            if cursor < script.endIndex {
              cursor = script.index(after: cursor)
            }
            continue
          }
          if currentCharacter == "`" {
            return script.index(after: cursor)
          }
          cursor = script.index(after: cursor)
        }
        throw CLIError(message: "Unterminated command substitution in markdown command", usage: usage)
      }

      let openParenIndex = script.index(after: startIndex)
      guard openParenIndex < script.endIndex, script[openParenIndex] == "(" else {
        throw CLIError(message: "Unterminated command substitution in markdown command", usage: usage)
      }

      var cursor = script.index(after: openParenIndex)
      var nestedDepth = 1
      var nestedMode: QuoteMode = .none

      while cursor < script.endIndex {
        let currentCharacter = script[cursor]

        switch nestedMode {
        case .single:
          if currentCharacter == "'" {
            nestedMode = .none
          }
          cursor = script.index(after: cursor)

        case .double:
          if currentCharacter == "\\" {
            cursor = script.index(after: cursor)
            if cursor < script.endIndex {
              cursor = script.index(after: cursor)
            }
            continue
          }
          if currentCharacter == "\"" {
            nestedMode = .none
          }
          cursor = script.index(after: cursor)

        case .none:
          if currentCharacter == "\\" {
            cursor = script.index(after: cursor)
            if cursor < script.endIndex {
              cursor = script.index(after: cursor)
            }
            continue
          }
          if currentCharacter == "'" {
            nestedMode = .single
            cursor = script.index(after: cursor)
            continue
          }
          if currentCharacter == "\"" {
            nestedMode = .double
            cursor = script.index(after: cursor)
            continue
          }
          if currentCharacter == "`" {
            cursor = try commandSubstitutionEnd(startingAt: cursor)
            continue
          }
          if currentCharacter == "$" {
            let nextIndex = script.index(after: cursor)
            if nextIndex < script.endIndex, script[nextIndex] == "(" {
              let afterOpenIndex = script.index(after: nextIndex)
              if afterOpenIndex < script.endIndex, script[afterOpenIndex] == "(" {
                throw CLIError(message: "Shell expansion is not supported in markdown commands", usage: usage)
              }
              nestedDepth += 1
              cursor = script.index(after: nextIndex)
              continue
            }
          }
          if currentCharacter == ")" {
            nestedDepth -= 1
            let nextCursor = script.index(after: cursor)
            if nestedDepth == 0 {
              return nextCursor
            }
            cursor = nextCursor
            continue
          }
          cursor = script.index(after: cursor)
        }
      }

      throw CLIError(message: "Unterminated command substitution in markdown command", usage: usage)
    }

    func appendCommandSubstitution() throws {
      markTokenStartIfNeeded(at: index)
      let endIndex = try commandSubstitutionEnd(startingAt: index)
      current.append(contentsOf: script[index..<endIndex])
      index = endIndex
    }

    while index < script.endIndex {
      let character = script[index]

      switch mode {
      case .single:
        markTokenStartIfNeeded(at: index)
        if character == "'" {
          mode = .none
        } else {
          current.append(character)
        }
        index = script.index(after: index)

      case .double:
        if character == "\"" {
          mode = .none
          index = script.index(after: index)
          continue
        }
        if character == "\\" {
          markTokenStartIfNeeded(at: index)
          let nextIndex = script.index(after: index)
          guard nextIndex < script.endIndex else {
            throw CLIError(message: "Unterminated escape sequence in markdown command", usage: usage)
          }
          let escaped = script[nextIndex]
          if escaped == "\n" {
            index = script.index(after: nextIndex)
            continue
          }
          current.append(escaped)
          index = script.index(after: nextIndex)
          continue
        }
        if character == "`" {
          try appendCommandSubstitution()
          continue
        }
        if character == "$" {
          try rejectDollarSequence(at: index)
          let nextIndex = script.index(after: index)
          if nextIndex < script.endIndex, script[nextIndex] == "(" {
            try appendCommandSubstitution()
            continue
          }
        }
        markTokenStartIfNeeded(at: index)
        current.append(character)
        index = script.index(after: index)

      case .none:
        if character == "\\" {
          markTokenStartIfNeeded(at: index)
          let nextIndex = script.index(after: index)
          guard nextIndex < script.endIndex else {
            throw CLIError(message: "Unterminated escape sequence in markdown command", usage: usage)
          }
          let escaped = script[nextIndex]
          if escaped == "\n" {
            index = script.index(after: nextIndex)
            continue
          }
          current.append(escaped)
          index = script.index(after: nextIndex)
          continue
        }

        if character == "'" {
          markTokenStartIfNeeded(at: index)
          mode = .single
          pendingCommandBreak = false
          index = script.index(after: index)
          continue
        }
        if character == "\"" {
          markTokenStartIfNeeded(at: index)
          mode = .double
          pendingCommandBreak = false
          index = script.index(after: index)
          continue
        }
        if character == "`" {
          try appendCommandSubstitution()
          pendingCommandBreak = false
          continue
        }
        if character == "$" {
          try rejectDollarSequence(at: index)
          let nextIndex = script.index(after: index)
          if nextIndex < script.endIndex, script[nextIndex] == "(" {
            try appendCommandSubstitution()
            pendingCommandBreak = false
            continue
          }
          markTokenStartIfNeeded(at: index)
          current.append(character)
          index = script.index(after: index)
          continue
        }
        if character == "|" || character == ";" || character == ">" || character == "<"
          || character == "&" || character == "(" || character == ")"
        {
          throw CLIError(message: "Shell control operators are not allowed in markdown commands", usage: usage)
        }
        if character == "#" && current.isEmpty {
          while index < script.endIndex, script[index] != "\n" {
            index = script.index(after: index)
          }
          continue
        }
        if character.isWhitespace {
          finishToken(at: index)
          if character == "\n" || character == "\r" {
            if !tokens.isEmpty {
              pendingCommandBreak = true
            }
          }
          index = script.index(after: index)
          continue
        }
        if pendingCommandBreak {
          throw CLIError(message: "Each markdown fence must contain exactly one command", usage: usage)
        }
        markTokenStartIfNeeded(at: index)
        current.append(character)
        index = script.index(after: index)
      }
    }

    guard mode == .none else {
      throw CLIError(message: "Unterminated quote in markdown command", usage: usage)
    }

    finishToken(at: script.endIndex)
    return tokens
  }
}
