import MLX
import XCTest

@testable import ZImage

#if canImport(CoreGraphics)
  import CoreGraphics

  final class ImageIOTests: XCTestCase {

    // MARK: - Array from CGImage Tests

    func testArrayFromCGImage() throws {
      let cgImage = try createTestCGImage(width: 64, height: 64)
      let array = try QwenImageIO.array(from: cgImage, addBatchDimension: true)

      // Should have shape [1, 3, 64, 64] with batch dimension
      XCTAssertEqual(array.shape, [1, 3, 64, 64])
      XCTAssertEqual(array.dtype, .float32)
    }

    func testArrayFromCGImageNoBatch() throws {
      let cgImage = try createTestCGImage(width: 64, height: 64)
      let array = try QwenImageIO.array(from: cgImage, addBatchDimension: false)

      // Should have shape [3, 64, 64] without batch dimension
      XCTAssertEqual(array.shape, [3, 64, 64])
    }

    func testArrayFromCGImageDtype() throws {
      let cgImage = try createTestCGImage(width: 32, height: 32)

      let float32Array = try QwenImageIO.array(from: cgImage, dtype: .float32)
      XCTAssertEqual(float32Array.dtype, .float32)

      let bfloat16Array = try QwenImageIO.array(from: cgImage, dtype: .bfloat16)
      XCTAssertEqual(bfloat16Array.dtype, .bfloat16)
    }

    func testArrayValuesInRange() throws {
      let cgImage = try createTestCGImage(width: 32, height: 32)
      let array = try QwenImageIO.array(from: cgImage, addBatchDimension: false)
      MLX.eval(array)

      let values = array.asArray(Float.self)

      // All values should be in [0, 1] range
      for value in values {
        XCTAssertGreaterThanOrEqual(value, 0.0)
        XCTAssertLessThanOrEqual(value, 1.0)
      }
    }

    // MARK: - Image from Array Tests

    func testImageFromArray() throws {
      // Create a test array with shape [3, 64, 64]
      let values = (0..<(3 * 64 * 64)).map { Float($0 % 256) / 255.0 }
      let array = MLXArray(values, [3, 64, 64])

      let cgImage = try QwenImageIO.image(from: array)

      XCTAssertEqual(cgImage.width, 64)
      XCTAssertEqual(cgImage.height, 64)
    }

    func testImageFromArrayWithBatch() throws {
      // Create a test array with shape [1, 3, 64, 64]
      let values = (0..<(1 * 3 * 64 * 64)).map { Float($0 % 256) / 255.0 }
      let array = MLXArray(values, [1, 3, 64, 64])

      let cgImage = try QwenImageIO.image(from: array)

      XCTAssertEqual(cgImage.width, 64)
      XCTAssertEqual(cgImage.height, 64)
    }

    func testImageFromArrayClipping() throws {
      // Create array with values outside [0, 1] range
      let values: [Float] = [-0.5, 0.5, 1.5, 0.0, 1.0, 2.0]  // 6 values for 1x2 with 3 channels
      let array = MLXArray(values, [3, 1, 2])

      // Should not throw - values should be clipped internally
      let cgImage = try QwenImageIO.image(from: array)

      XCTAssertEqual(cgImage.width, 2)
      XCTAssertEqual(cgImage.height, 1)
    }

    // MARK: - Normalization Tests

    func testNormalizeForEncoder() {
      // Create array with values in [0, 1]
      let values: [Float] = [0.0, 0.5, 1.0, 0.25, 0.75, 0.0]
      let array = MLXArray(values, [3, 1, 2])

      let normalized = QwenImageIO.normalizeForEncoder(array)
      MLX.eval(normalized)

      let result = normalized.asArray(Float.self)

      // Expected: value * 2 - 1
      // 0.0 -> -1.0, 0.5 -> 0.0, 1.0 -> 1.0
      XCTAssertEqual(result[0], -1.0, accuracy: 1e-5)
      XCTAssertEqual(result[1], 0.0, accuracy: 1e-5)
      XCTAssertEqual(result[2], 1.0, accuracy: 1e-5)
    }

    func testNormalizeForEncoderPreservesInputDtype() {
      let values: [Float] = [0.0, 0.5, 1.0, 0.25, 0.75, 0.0]
      let array = MLXArray(values, [3, 1, 2]).asType(.bfloat16)

      let normalized = QwenImageIO.normalizeForEncoder(array)

      XCTAssertEqual(normalized.dtype, .bfloat16)
    }

    func testDenormalizeFromDecoder() {
      // Create array with values in [-1, 1]
      let values: [Float] = [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0]
      let array = MLXArray(values, [3, 1, 2])

      let denormalized = QwenImageIO.denormalizeFromDecoder(array)
      MLX.eval(denormalized)

      let result = denormalized.asArray(Float.self)

      // Expected: (value + 1) / 2
      // -1.0 -> 0.0, 0.0 -> 0.5, 1.0 -> 1.0
      XCTAssertEqual(result[0], 0.0, accuracy: 1e-5)
      XCTAssertEqual(result[1], 0.5, accuracy: 1e-5)
      XCTAssertEqual(result[2], 1.0, accuracy: 1e-5)
    }

    func testDenormalizeFromDecoderPreservesInputDtype() {
      let values: [Float] = [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0]
      let array = MLXArray(values, [3, 1, 2]).asType(.bfloat16)

      let denormalized = QwenImageIO.denormalizeFromDecoder(array)

      XCTAssertEqual(denormalized.dtype, .bfloat16)
    }

    func testNormalizeDenormalizeRoundTrip() {
      let values: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0, 0.33]
      let original = MLXArray(values, [3, 1, 2])

      let normalized = QwenImageIO.normalizeForEncoder(original)
      let denormalized = QwenImageIO.denormalizeFromDecoder(normalized)
      MLX.eval(denormalized)

      let result = denormalized.asArray(Float.self)
      let originalValues = original.asArray(Float.self)

      // Round trip should preserve values
      for i in 0..<values.count {
        XCTAssertEqual(result[i], originalValues[i], accuracy: 1e-5)
      }
    }

    // MARK: - Resize Tests

    func testResizedCGImage() throws {
      let cgImage = try createTestCGImage(width: 128, height: 128)
      let resized = try QwenImageIO.resizedCGImage(from: cgImage, width: 64, height: 64)

      XCTAssertEqual(resized.width, 64)
      XCTAssertEqual(resized.height, 64)
    }

    func testResizedCGImageUpscale() throws {
      let cgImage = try createTestCGImage(width: 32, height: 32)
      let resized = try QwenImageIO.resizedCGImage(from: cgImage, width: 128, height: 128)

      XCTAssertEqual(resized.width, 128)
      XCTAssertEqual(resized.height, 128)
    }

    func testResizedCGImageNonSquare() throws {
      let cgImage = try createTestCGImage(width: 128, height: 64)
      let resized = try QwenImageIO.resizedCGImage(from: cgImage, width: 256, height: 128)

      XCTAssertEqual(resized.width, 256)
      XCTAssertEqual(resized.height, 128)
    }

    func testResizedCGImageInvalidDimensions() throws {
      let cgImage = try createTestCGImage(width: 64, height: 64)

      XCTAssertThrowsError(try QwenImageIO.resizedCGImage(from: cgImage, width: 0, height: 64)) { error in
        XCTAssertTrue(error is QwenImageIOError)
      }

      XCTAssertThrowsError(try QwenImageIO.resizedCGImage(from: cgImage, width: 64, height: 0)) { error in
        XCTAssertTrue(error is QwenImageIOError)
      }
    }

    func testResizedPixelArray() throws {
      let cgImage = try createTestCGImage(width: 128, height: 128)
      let array = try QwenImageIO.resizedPixelArray(
        from: cgImage,
        width: 64,
        height: 64,
        addBatchDimension: true
      )

      XCTAssertEqual(array.shape, [1, 3, 64, 64])
    }

    // MARK: - Resize RGB Array Tests

    func testResizeRGBArray() throws {
      // Create test array [3, 32, 32]
      let values = (0..<(3 * 32 * 32)).map { Float($0 % 256) / 255.0 }
      let array = MLXArray(values, [3, 32, 32])

      let resized = try QwenImageIO.resize(rgbArray: array, targetHeight: 64, targetWidth: 64)

      XCTAssertEqual(resized.shape, [3, 64, 64])
    }

    func testResizeRGBArraySameDimensions() throws {
      let values = (0..<(3 * 32 * 32)).map { Float($0 % 256) / 255.0 }
      let array = MLXArray(values, [3, 32, 32])

      // Same dimensions should return input unchanged
      let resized = try QwenImageIO.resize(rgbArray: array, targetHeight: 32, targetWidth: 32)

      XCTAssertEqual(resized.shape, [3, 32, 32])
    }

    func testResizeRGBArrayDownscale() throws {
      let values = (0..<(3 * 128 * 128)).map { Float($0 % 256) / 255.0 }
      let array = MLXArray(values, [3, 128, 128])

      let resized = try QwenImageIO.resize(rgbArray: array, targetHeight: 32, targetWidth: 32)

      XCTAssertEqual(resized.shape, [3, 32, 32])
    }

    // MARK: - Various Dimensions Tests

    func testVariousDimensions() throws {
      let testDimensions = [(512, 512), (768, 768), (1024, 1024), (256, 512)]

      for (width, height) in testDimensions {
        let cgImage = try createTestCGImage(width: width, height: height)
        let array = try QwenImageIO.array(from: cgImage, addBatchDimension: true)

        XCTAssertEqual(array.shape, [1, 3, height, width], "Failed for dimensions \(width)x\(height)")
      }
    }

    // MARK: - Save Image Tests

    func testSaveImage() throws {
      let values = (0..<(3 * 64 * 64)).map { Float($0 % 256) / 255.0 }
      let array = MLXArray(values, [3, 64, 64])

      let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_output.png")
      defer { try? FileManager.default.removeItem(at: tempURL) }

      try QwenImageIO.saveImage(array: array, to: tempURL)

      XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
    }

    func testSaveImageWithBatch() throws {
      let values = (0..<(1 * 3 * 64 * 64)).map { Float($0 % 256) / 255.0 }
      let array = MLXArray(values, [1, 3, 64, 64])

      let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_output_batch.png")
      defer { try? FileManager.default.removeItem(at: tempURL) }

      try QwenImageIO.saveImage(array: array, to: tempURL)

      XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
    }

    // MARK: - Round Trip Tests

    func testArrayImageRoundTrip() throws {
      // Create array -> image -> array and verify dimensions
      let originalValues = (0..<(3 * 64 * 64)).map { _ in Float.random(in: 0...1) }
      let originalArray = MLXArray(originalValues, [3, 64, 64])

      let cgImage = try QwenImageIO.image(from: originalArray)
      let reconstructed = try QwenImageIO.array(from: cgImage, addBatchDimension: false)

      XCTAssertEqual(reconstructed.shape, [3, 64, 64])

      // Values should be approximately preserved (some loss due to quantization to 8-bit)
      MLX.eval(reconstructed)
      let reconstructedValues = reconstructed.asArray(Float.self)

      var maxDiff: Float = 0
      for i in 0..<originalValues.count {
        let diff = abs(reconstructedValues[i] - originalValues[i])
        maxDiff = max(maxDiff, diff)
      }

      // 8-bit quantization error should be at most 1/255 ~= 0.004
      XCTAssertLessThan(maxDiff, 0.01, "Round trip error too large: \(maxDiff)")
    }

    // MARK: - Helper Functions

    private func createTestCGImage(width: Int, height: Int) throws -> CGImage {
      let colorSpace = CGColorSpaceCreateDeviceRGB()
      let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

      guard
        let context = CGContext(
          data: nil,
          width: width,
          height: height,
          bitsPerComponent: 8,
          bytesPerRow: width * 4,
          space: colorSpace,
          bitmapInfo: bitmapInfo
        )
      else {
        throw TestError.contextCreationFailed
      }

      // Fill with gradient for variety
      for y in 0..<height {
        for x in 0..<width {
          let r = CGFloat(x) / CGFloat(width)
          let g = CGFloat(y) / CGFloat(height)
          let b = CGFloat(x + y) / CGFloat(width + height)
          context.setFillColor(red: r, green: g, blue: b, alpha: 1.0)
          context.fill(CGRect(x: x, y: y, width: 1, height: 1))
        }
      }

      guard let image = context.makeImage() else {
        throw TestError.imageCreationFailed
      }

      return image
    }

    enum TestError: Error {
      case contextCreationFailed
      case imageCreationFailed
    }
  }
#endif
