import MLX
import XCTest
@testable import ZImage

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

final class ControlNetIntegrationTests: XCTestCase {
  private nonisolated(unsafe) static var sharedPipeline: ZImageControlPipeline?
  private static let projectRoot: URL = URL(fileURLWithPath: #file)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

  private static let outputDir: URL = {
    let url = projectRoot
      .appendingPathComponent("Tests")
      .appendingPathComponent("ZImageIntegrationTests")
      .appendingPathComponent("Resources")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }()

  override class func setUp() {
    super.setUp()

    if ProcessInfo.processInfo.environment["CI"] == nil {
      sharedPipeline = ZImageControlPipeline()
    }
  }

  override class func tearDown() {
    sharedPipeline = nil

    try? FileManager.default.removeItem(at: outputDir)
    super.tearDown()
  }

  private func getPipeline() throws -> ZImageControlPipeline {
    guard let pipeline = Self.sharedPipeline else {
      throw XCTSkip("Pipeline not available (likely CI environment)")
    }
    return pipeline
  }

  func testCannyControlGeneration() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let controlImage = try createCannyEdgeImage()
    let tempOutput = Self.outputDir.appendingPathComponent("test_canny.png")

    let request = ZImageControlGenerationRequest(
      prompt: "a detailed building based on the edge map",
      controlImage: controlImage,
      controlContextScale: 0.75,
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      controlnetWeights: "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
      controlnetWeightsFile: "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors"
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  func testDepthControlGeneration() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let controlImage = try createDepthMapImage()
    let tempOutput = Self.outputDir.appendingPathComponent("test_depth.png")

    let request = ZImageControlGenerationRequest(
      prompt: "a 3D scene with clear depth",
      controlImage: controlImage,
      controlContextScale: 0.75,
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      controlnetWeights: "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
      controlnetWeightsFile: "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors"
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  func testPoseControlGeneration() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let controlImage = try createPoseImage()
    let tempOutput = Self.outputDir.appendingPathComponent("test_pose.png")

    let request = ZImageControlGenerationRequest(
      prompt: "a person in the shown pose",
      controlImage: controlImage,
      controlContextScale: 0.75,
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      controlnetWeights: "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
      controlnetWeightsFile: "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors"
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  func testControlNetV21PoseGeneration() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let poseImagePath = Self.projectRoot
      .appendingPathComponent("temp")
      .appendingPathComponent("Archive")
      .appendingPathComponent("asset")
      .appendingPathComponent("pose.jpg")
    let controlImage: URL
    if FileManager.default.fileExists(atPath: poseImagePath.path) {
      controlImage = poseImagePath
    } else {
      controlImage = try createPoseImage()
    }

    let tempOutput = Self.outputDir.appendingPathComponent("test_controlnet_v21_pose.png")

    let request = ZImageControlGenerationRequest(
      prompt: "a young woman standing on a sunny coastline, photorealistic",
      controlImage: controlImage,
      controlContextScale: 0.75,
      width: 576,
      height: 1024,
      steps: 25,
      seed: 43,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      controlnetWeights: "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
      controlnetWeightsFile: "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors"
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
    guard let imageSource = CGImageSourceCreateWithURL(outputURL as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
    else {
      XCTFail("Failed to load output image")
      return
    }

    XCTAssertEqual(cgImage.width, 576, "Output width should match request")
    XCTAssertEqual(cgImage.height, 1024, "Output height should match request")
  }

  private func createCannyEdgeImage() throws -> URL {
    let width = 512
    let height = 512

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      throw TestError.contextCreationFailed
    }
    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.setLineWidth(3)
    context.stroke(CGRect(x: 100, y: 100, width: 312, height: 312))
    context.move(to: CGPoint(x: 100, y: 100))
    context.addLine(to: CGPoint(x: 412, y: 412))
    context.strokePath()

    guard let image = context.makeImage() else {
      throw TestError.imageCreationFailed
    }

    return try saveImage(image, name: "canny_edge")
  }

  private func createDepthMapImage() throws -> URL {
    let width = 512
    let height = 512

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      throw TestError.contextCreationFailed
    }
    for y in 0 ..< height {
      let depth = CGFloat(y) / CGFloat(height)
      context.setFillColor(CGColor(red: depth, green: depth, blue: depth, alpha: 1))
      context.fill(CGRect(x: 0, y: y, width: width, height: 1))
    }

    guard let image = context.makeImage() else {
      throw TestError.imageCreationFailed
    }

    return try saveImage(image, name: "depth_map")
  }

  private func createPoseImage() throws -> URL {
    let width = 512
    let height = 512

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      throw TestError.contextCreationFailed
    }
    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    let centerX = CGFloat(width / 2)
    let headY = CGFloat(100)
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fillEllipse(in: CGRect(x: centerX - 20, y: headY - 20, width: 40, height: 40))
    context.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.setLineWidth(3)
    context.move(to: CGPoint(x: centerX, y: headY + 20))
    context.addLine(to: CGPoint(x: centerX, y: headY + 150))
    context.strokePath()
    context.move(to: CGPoint(x: centerX - 80, y: headY + 60))
    context.addLine(to: CGPoint(x: centerX + 80, y: headY + 60))
    context.strokePath()
    context.move(to: CGPoint(x: centerX, y: headY + 150))
    context.addLine(to: CGPoint(x: centerX - 50, y: headY + 280))
    context.strokePath()

    context.move(to: CGPoint(x: centerX, y: headY + 150))
    context.addLine(to: CGPoint(x: centerX + 50, y: headY + 280))
    context.strokePath()

    guard let image = context.makeImage() else {
      throw TestError.imageCreationFailed
    }

    return try saveImage(image, name: "pose")
  }

  private func saveImage(_ image: CGImage, name: String) throws -> URL {
    let url = Self.outputDir.appendingPathComponent("\(name).png")

    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
      throw TestError.imageCreationFailed
    }

    CGImageDestinationAddImage(destination, image, nil)

    guard CGImageDestinationFinalize(destination) else {
      throw TestError.imageCreationFailed
    }

    return url
  }

  private func skipIfNoGPU() throws {
    if ProcessInfo.processInfo.environment["CI"] != nil {
      throw XCTSkip("Skipping GPU-intensive test in CI environment")
    }
  }

  enum TestError: Error {
    case contextCreationFailed
    case imageCreationFailed
  }
}
#endif
