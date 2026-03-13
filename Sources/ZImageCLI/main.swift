import Darwin
import Foundation
import Logging
import Metal
import ZImageCLICommon

LoggingSystem.bootstrap { label in
  var handler = StreamLogHandler.standardError(label: label)
  handler.logLevel = .info
  return handler
}

enum ZImageCLI {
  static let logger: Logger = {
    var logger = Logger(label: "z-image.cli")
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
        print(CLIUsageFormatter.usage(for: usage, program: .cli))
      }
      Darwin.exit(EXIT_FAILURE)
    } catch {
      logger.error("\(CLIErrors.describe(error))")
      Darwin.exit(EXIT_FAILURE)
    }
  }

  static func run() throws {
    logMetalDevice()
    let command = try CLICompatParser.parseCLI(Array(CommandLine.arguments.dropFirst()))
    try CLICommandRunner.run(command, logger: logger, program: .cli)
  }

  private static func logMetalDevice() {
    if let dev = MTLCreateSystemDefaultDevice() {
      logger.info("Metal device: \(dev.name)")
    } else {
      logger.warning("No Metal device detected; MLX will fall back to CPU.")
    }
  }
}

ZImageCLI.main()
