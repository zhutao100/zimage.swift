import XCTest
@testable import ZImageCLICommon

final class MarkdownCommandExtractorTests: XCTestCase {
  func testExtractsServeInvocationFromMarkdownFence() throws {
    let submissions = try MarkdownCommandExtractor.submissions(
      from:
        """
        # Prompt draft

        ```bash
        ZImageServe --prompt "a mountain lake" --model /tmp/model --output lake.png
        ```
        """
    )

    XCTAssertEqual(submissions.count, 1)
    XCTAssertEqual(submissions[0].jobID, "markdown-1")
    XCTAssertEqual(submissions[0].program, .serve)
    XCTAssertEqual(submissions[0].shell, .bash)

    guard case .text(let options) = try submissions[0].resolveJob() else {
      return XCTFail("Expected text job")
    }
    XCTAssertEqual(options.prompt, "a mountain lake")
    XCTAssertEqual(options.model, "/tmp/model")
    XCTAssertEqual(options.outputPath, "lake.png")
  }

  func testExtractsCLIControlInvocationFromMarkdownFence() throws {
    let submissions = try MarkdownCommandExtractor.submissions(
      from:
        """
        ```zsh
        ZImageCLI control --prompt "a dancer" --control-image pose.png --controlnet-weights ./controlnet --output dancer.png
        ```
        """
    )

    XCTAssertEqual(submissions.count, 1)
    XCTAssertEqual(submissions[0].program, .cli)
    XCTAssertEqual(submissions[0].shell, .zsh)

    guard case .control(let options) = try submissions[0].resolveJob() else {
      return XCTFail("Expected control job")
    }
    XCTAssertEqual(options.prompt, "a dancer")
    XCTAssertEqual(options.controlImage, "pose.png")
    XCTAssertEqual(options.controlnetWeights, "./controlnet")
    XCTAssertEqual(options.outputPath, "dancer.png")
  }

  func testExtractsInvocationWhenExecutableUsesExplicitPath() throws {
    let submissions = try MarkdownCommandExtractor.submissions(
      from:
        """
        ```bash
        .build/xcode/Build/Products/Release/ZImageCLI control \
        --prompt "a dancer" \
        --control-image pose.png \
        --controlnet-weights ./controlnet \
        --output dancer.png
        ```
        """
    )

    XCTAssertEqual(submissions.count, 1)
    XCTAssertEqual(submissions[0].program, .cli)

    guard case .control(let options) = try submissions[0].resolveJob() else {
      return XCTFail("Expected control job")
    }
    XCTAssertEqual(options.prompt, "a dancer")
    XCTAssertEqual(options.controlImage, "pose.png")
    XCTAssertEqual(options.controlnetWeights, "./controlnet")
    XCTAssertEqual(options.outputPath, "dancer.png")
  }

  func testRejectsShellControlOperators() throws {
    XCTAssertThrowsError(
      try MarkdownCommandExtractor.submissions(
        from:
          """
          ```bash
          ZImageServe --prompt "a lake" | tee out.txt
          ```
          """
      )
    ) { error in
      guard let cliError = error as? CLIError else {
        return XCTFail("Expected CLIError")
      }
      XCTAssertTrue(cliError.message.contains("Shell control operators"))
      XCTAssertEqual(cliError.usage, .markdown)
    }
  }

  func testRejectsWrappedGenerationInvocation() throws {
    XCTAssertThrowsError(
      try MarkdownCommandExtractor.submissions(
        from:
          """
          ```bash
          env FOO=1 /tmp/build/ZImageServe --prompt "a lake" --model /tmp/model --output lake.png
          ```
          """
      )
    ) { error in
      guard let cliError = error as? CLIError else {
        return XCTFail("Expected CLIError")
      }
      XCTAssertTrue(cliError.message.contains("invoke ZImageCLI or ZImageServe directly"))
      XCTAssertEqual(cliError.usage, .markdown)
    }
  }

  func testRejectsMultipleCommandsInSingleFence() throws {
    XCTAssertThrowsError(
      try MarkdownCommandExtractor.submissions(
        from:
          """
          ```bash
          ZImageServe --prompt "a lake"
          ZImageServe --prompt "a forest"
          ```
          """
      )
    ) { error in
      guard let cliError = error as? CLIError else {
        return XCTFail("Expected CLIError")
      }
      XCTAssertTrue(cliError.message.contains("exactly one command"))
      XCTAssertEqual(cliError.usage, .markdown)
    }
  }

  func testResolvesDollarParenCommandSubstitutionAtRuntime() throws {
    let submissions = try MarkdownCommandExtractor.submissions(
      from:
        """
        ```bash
        /tmp/build/ZImageServe --prompt "a lake" --model /tmp/model --output "lake-$(printf runtime).png"
        ```
        """
    )

    XCTAssertEqual(submissions.count, 1)

    guard case .text(let options) = try submissions[0].resolveJob() else {
      return XCTFail("Expected text job")
    }
    XCTAssertEqual(options.outputPath, "lake-runtime.png")
  }

  func testResolvesBacktickCommandSubstitutionAtRuntime() throws {
    let submissions = try MarkdownCommandExtractor.submissions(
      from:
        """
        ```bash
        /tmp/build/ZImageServe --prompt "a lake" --model /tmp/model --output "lake-`printf runtime`.png"
        ```
        """
    )

    XCTAssertEqual(submissions.count, 1)

    guard case .text(let options) = try submissions[0].resolveJob() else {
      return XCTFail("Expected text job")
    }
    XCTAssertEqual(options.outputPath, "lake-runtime.png")
  }

  func testRejectsUnsupportedShellExpansionOutsideCommandSubstitution() throws {
    XCTAssertThrowsError(
      try MarkdownCommandExtractor.submissions(
        from:
          """
          ```bash
          /tmp/build/ZImageServe --prompt "a lake" --model /tmp/model --output "$HOME/lake.png"
          ```
          """
      )
    ) { error in
      guard let cliError = error as? CLIError else {
        return XCTFail("Expected CLIError")
      }
      XCTAssertTrue(cliError.message.contains("Shell expansion"))
      XCTAssertEqual(cliError.usage, .markdown)
    }
  }

  func testRejectsUnterminatedFence() throws {
    XCTAssertThrowsError(
      try MarkdownCommandExtractor.submissions(
        from:
          """
          ```bash
          ZImageServe --prompt "a lake"
          """
      )
    ) { error in
      guard let cliError = error as? CLIError else {
        return XCTFail("Expected CLIError")
      }
      XCTAssertTrue(cliError.message.contains("Unterminated fenced shell block"))
      XCTAssertEqual(cliError.usage, .markdown)
    }
  }

  func testAcceptsTildeFencedShellBlock() throws {
    let submissions = try MarkdownCommandExtractor.submissions(
      from:
        """
        ~~~bash
        /tmp/build/ZImageServe --prompt "a mountain lake" --model /tmp/model --output lake.png
        ~~~
        """
    )

    XCTAssertEqual(submissions.count, 1)

    guard case .text(let options) = try submissions[0].resolveJob() else {
      return XCTFail("Expected text job")
    }
    XCTAssertEqual(options.prompt, "a mountain lake")
    XCTAssertEqual(options.model, "/tmp/model")
    XCTAssertEqual(options.outputPath, "lake.png")
  }
}
