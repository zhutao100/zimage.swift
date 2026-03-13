import Darwin
import Foundation

public final class TerminalProgressRenderer: @unchecked Sendable {
  private let enabled: Bool
  private let totalSteps: Int
  private var bar: ProgressBar?
  private var plainProgress: PlainProgress?

  public init(noProgress: Bool, totalSteps: Int) {
    enabled = !noProgress
    self.totalSteps = max(1, totalSteps)
    guard enabled else { return }
    if isatty(STDERR_FILENO) != 0 {
      bar = ProgressBar(total: totalSteps)
    } else {
      plainProgress = PlainProgress.shared
    }
  }

  public func report(_ update: JobProgressUpdate) {
    guard enabled, update.totalSteps > 0 else { return }
    let completed = min(update.totalSteps, max(0, update.stepIndex))
    if let bar {
      bar.update(completed: completed)
      if completed == update.totalSteps {
        bar.finish(forceNewline: true)
      }
    } else if let plainProgress {
      plainProgress.report(completed: completed, total: totalSteps)
    }
  }

  public func finish() {
    bar?.finish(forceNewline: true)
  }
}

private final class PlainProgress: @unchecked Sendable {
  static let shared = PlainProgress()
  private let lock = NSLock()
  private var lastPercent = -1
  private var lastEmitTime: Date = .distantPast

  func report(completed: Int, total: Int) {
    guard total > 0 else { return }
    let now = Date()
    let percent = Int((Double(completed) / Double(total)) * 100.0)
    let shouldEmit: Bool
    lock.lock()
    if percent != lastPercent || now.timeIntervalSince(lastEmitTime) >= 0.5 {
      lastPercent = percent
      lastEmitTime = now
      shouldEmit = true
    } else {
      shouldEmit = false
    }
    lock.unlock()

    guard shouldEmit else { return }
    FileHandle.standardError.write("Step \(completed)/\(total) (\(percent)%)\n".data(using: .utf8)!)
  }
}

private final class ProgressBar: @unchecked Sendable {
  private let total: Int
  private var lastStepTime: Date?
  private var postWarmupDurations: [Double] = []
  private let windowSize = 5
  private var lastRenderedPercent = -1
  private var isFinished = false

  init(total: Int) {
    self.total = max(1, total)
  }

  func update(completed: Int) {
    if isFinished { return }
    let now = Date()
    if let last = lastStepTime {
      let dt = now.timeIntervalSince(last)
      postWarmupDurations.append(dt)
      if postWarmupDurations.count > windowSize {
        postWarmupDurations.removeFirst()
      }
    }
    lastStepTime = now

    let percent = Int((Double(completed) / Double(total)) * 100.0)
    if percent == lastRenderedPercent { return }
    lastRenderedPercent = percent

    let remaining = max(0, total - completed)
    let etaSeconds: Double?
    if postWarmupDurations.isEmpty {
      etaSeconds = nil
    } else {
      let avg = postWarmupDurations.reduce(0, +) / Double(postWarmupDurations.count)
      etaSeconds = avg * Double(remaining)
    }

    let barWidth = 28
    let filled = Int((Double(completed) / Double(total)) * Double(barWidth))
    let lead = completed < total ? ">" : "="
    let tailCount = max(0, barWidth - max(1, filled))
    let bar =
      String(repeating: "=", count: max(0, filled - 1))
      + lead
      + String(repeating: "-", count: tailCount)
    let etaString = etaSeconds.map(Self.format(seconds:)) ?? "estimating..."
    let prefix = "\r\u{001B}[2K"
    let line = String(format: "[%@] %3d%%  %d/%d  ETA %@", bar, percent, completed, total, etaString)
    if let data = (prefix + line).data(using: .utf8) {
      FileHandle.standardError.write(data)
      fflush(stderr)
    }
  }

  func finish(forceNewline: Bool = true) {
    if isFinished { return }
    isFinished = true
    if let data = "\r\u{001B}[2K".data(using: .utf8) {
      FileHandle.standardError.write(data)
    }
    if forceNewline, let newline = "\n".data(using: .utf8) {
      FileHandle.standardError.write(newline)
    }
    fflush(stderr)
  }

  private static func format(seconds: Double) -> String {
    var secondsValue = Int(seconds.rounded())
    let hours = secondsValue / 3600
    secondsValue %= 3600
    let minutes = secondsValue / 60
    secondsValue %= 60
    if hours > 0 {
      return String(format: "%dh%02dm%02ds", hours, minutes, secondsValue)
    }
    if minutes > 0 {
      return String(format: "%dm%02ds", minutes, secondsValue)
    }
    return String(format: "%ds", secondsValue)
  }
}
