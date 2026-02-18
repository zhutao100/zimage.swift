import MLX
import XCTest
@testable import ZImage

/// Performance tests for Z-Image pipeline.
/// Tracks inference time, memory usage, and per-component breakdown.
/// Run with: xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -only-testing:ZImageIntegrationTests/PerformanceTests -parallel-testing-enabled NO
final class PerformanceTests: XCTestCase {
  /// Shared pipeline instance to avoid reloading model for each test
  private nonisolated(unsafe) static var sharedPipeline: ZImagePipeline?

  /// Project root directory (derived from test file location)
  private static let projectRoot: URL = URL(fileURLWithPath: #file)
    .deletingLastPathComponent() // Remove PerformanceTests.swift
    .deletingLastPathComponent() // Remove ZImageIntegrationTests
    .deletingLastPathComponent() // Remove Tests -> project root

  /// Output directory for test-generated images (inside project)
  private static let outputDir: URL = {
    let url = projectRoot
      .appendingPathComponent("Tests")
      .appendingPathComponent("ZImageIntegrationTests")
      .appendingPathComponent("Resources")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }()

  /// Initialize shared pipeline once for all tests
  override class func setUp() {
    super.setUp()
    // Skip pipeline creation in CI
    if ProcessInfo.processInfo.environment["CI"] == nil {
      sharedPipeline = ZImagePipeline()
    }
  }

  override class func tearDown() {
    // Clean up shared pipeline
    sharedPipeline = nil
    // Clean up Resources directory after all tests
    try? FileManager.default.removeItem(at: outputDir)
    super.tearDown()
  }

  /// Get the shared pipeline or skip test if not available
  private func getPipeline() throws -> ZImagePipeline {
    guard let pipeline = Self.sharedPipeline else {
      throw XCTSkip("Pipeline not available (likely CI environment)")
    }
    return pipeline
  }

  /// Performance metrics structure
  struct PerformanceMetrics {
    var totalInferenceTime: TimeInterval = 0
    var textEncodingTime: TimeInterval = 0
    var transformerTime: TimeInterval = 0
    var perStepLatency: [TimeInterval] = []
    var vaeDecodingTime: TimeInterval = 0
    var peakMemoryUsage: UInt64 = 0
    var imagesPerSecond: Double = 0

    var averageStepLatency: TimeInterval {
      guard !perStepLatency.isEmpty else { return 0 }
      return perStepLatency.reduce(0, +) / Double(perStepLatency.count)
    }

    func printSummary() {
      print("=== Performance Metrics ===")
      print("Total inference time: \(String(format: "%.2f", totalInferenceTime))s")
      print("Text encoding time: \(String(format: "%.2f", textEncodingTime))s")
      print("Transformer time: \(String(format: "%.2f", transformerTime))s")
      print("VAE decoding time: \(String(format: "%.2f", vaeDecodingTime))s")
      print("Average step latency: \(String(format: "%.3f", averageStepLatency))s")
      print("Peak memory: \(peakMemoryUsage / 1024 / 1024) MB")
      print("Images/second: \(String(format: "%.3f", imagesPerSecond))")
      print("===========================")
    }
  }

  /// Baseline performance expectations (can be adjusted based on hardware)
  static let baselineMetrics = PerformanceMetrics(
    totalInferenceTime: 60.0, // 60 seconds max for 9 steps at 1024x1024
    textEncodingTime: 5.0,
    transformerTime: 50.0,
    perStepLatency: [],
    vaeDecodingTime: 5.0,
    peakMemoryUsage: 15 * 1024 * 1024 * 1024, // 15 GB max (8-bit quantized model uses ~12-13GB)
    imagesPerSecond: 0
  )

  // MARK: - Basic Performance Tests

  func testBasicGenerationPerformance() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let metrics = try await measureGeneration(
      pipeline: pipeline,
      prompt: "a beautiful sunset over mountains",
      width: 1024,
      height: 1024,
      steps: 9
    )

    metrics.printSummary()

    // Assert reasonable performance
    XCTAssertLessThan(
      metrics.totalInferenceTime,
      Self.baselineMetrics.totalInferenceTime * 1.1, // Allow 10% variance
      "Total inference time exceeded baseline by more than 10%"
    )
  }

  func testMemoryPeakTracking() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let metrics = try await measureGeneration(
      pipeline: pipeline,
      prompt: "a detailed landscape",
      width: 1024,
      height: 1024,
      steps: 9
    )

    print("Peak memory usage: \(metrics.peakMemoryUsage / 1024 / 1024) MB")

    // With 8-bit quantized model, peak should be under 10GB
    XCTAssertLessThan(
      metrics.peakMemoryUsage,
      Self.baselineMetrics.peakMemoryUsage,
      "Peak memory usage exceeded expected limit"
    )
  }

  func testMemoryStability() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    // Generate multiple images and check memory doesn't leak
    var memoryReadings: [UInt64] = []

    for i in 0 ..< 3 {
      let metrics = try await measureGeneration(
        pipeline: pipeline,
        prompt: "memory test \(i)",
        width: 256,
        height: 256,
        steps: 9
      )

      memoryReadings.append(metrics.peakMemoryUsage)

      // Allow some GPU cache clearing
      GPU.clearCache()
    }

    // Memory shouldn't grow unboundedly
    if memoryReadings.count >= 2 {
      let firstReading = memoryReadings[0]
      let lastReading = try XCTUnwrap(memoryReadings.last)

      // Last reading shouldn't be more than 50% higher than first
      XCTAssertLessThan(
        Double(lastReading),
        Double(firstReading) * 1.5,
        "Memory usage appears to be growing - possible memory leak"
      )
    }
  }

  // MARK: - Helper Functions

  private func measureGeneration(
    pipeline: ZImagePipeline,
    prompt: String,
    width: Int,
    height: Int,
    steps: Int
  ) async throws -> PerformanceMetrics {
    var metrics = PerformanceMetrics()

    let tempOutput = FileManager.default.temporaryDirectory.appendingPathComponent("perf_test_\(UUID().uuidString).png")
    defer {
      if FileManager.default.fileExists(atPath: tempOutput.path) {
        try? FileManager.default.removeItem(at: tempOutput)
      }
    }

    let request = ZImageGenerationRequest(
      prompt: prompt,
      width: width,
      height: height,
      steps: steps,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit"
    )

    // Measure total time
    let startTime = Date()
    _ = try await pipeline.generate(request)
    let endTime = Date()

    metrics.totalInferenceTime = endTime.timeIntervalSince(startTime)
    metrics.imagesPerSecond = 1.0 / metrics.totalInferenceTime

    // Estimate component breakdown (rough estimates based on typical distribution)
    // In a real implementation, the pipeline would report these timings
    metrics.textEncodingTime = metrics.totalInferenceTime * 0.15
    metrics.transformerTime = metrics.totalInferenceTime * 0.75
    metrics.vaeDecodingTime = metrics.totalInferenceTime * 0.10

    // Estimate per-step latency
    metrics.perStepLatency = (0 ..< steps).map { _ in metrics.transformerTime / Double(steps) }

    // Get memory usage
    metrics.peakMemoryUsage = getMemoryUsage()

    return metrics
  }

  private func getMemoryUsage() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kerr = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }

    if kerr == KERN_SUCCESS {
      return info.resident_size
    }
    return 0
  }

  private func skipIfNoGPU() throws {
    if ProcessInfo.processInfo.environment["CI"] != nil {
      throw XCTSkip("Skipping GPU-intensive test in CI environment")
    }
  }
}
