import Foundation
import MLX

public struct SafeTensorMetadata: Sendable {
  public let name: String
  public let dtype: DType
  public let shape: [Int]
  public let dataOffset: Int
  public let byteCount: Int

  public var elementCount: Int {
    shape.reduce(1, *)
  }
}

public enum SafeTensorsReaderError: Error {
  case fileTooSmall(URL)
  case invalidHeaderLength(URL)
  case malformedHeader(URL)
  case tensorMetadataMissing(String)
  case unsupportedDType(String)
  case invalidOffsets(name: String)
  case invalidShape(name: String)
  case tensorNotFound(String)
}

public final class SafeTensorsReader {
  public let fileURL: URL
  private let mappedData: Data
  private let tensors: [String: SafeTensorMetadata]
  public let fileMetadata: [String: String]

  public init(fileURL: URL) throws {
    self.fileURL = fileURL
    self.mappedData = try Data(contentsOf: fileURL, options: [.mappedIfSafe])

    guard mappedData.count >= MemoryLayout<UInt64>.size else {
      throw SafeTensorsReaderError.fileTooSmall(fileURL)
    }

    let headerLength = mappedData.prefix(8).withUnsafeBytes { rawBuffer -> Int in
      let value = rawBuffer.load(as: UInt64.self)
      return Int(UInt64(littleEndian: value))
    }

    let headerStart = 8
    let headerEnd = headerStart + headerLength
    guard headerEnd <= mappedData.count else {
      throw SafeTensorsReaderError.invalidHeaderLength(fileURL)
    }

    let headerData = mappedData.subdata(in: headerStart..<headerEnd)
    let headerJSON = try JSONSerialization.jsonObject(with: headerData, options: [])

    guard let headerDict = headerJSON as? [String: Any] else {
      throw SafeTensorsReaderError.malformedHeader(fileURL)
    }

    var tensorMetadata: [String: SafeTensorMetadata] = [:]
    var metadataValues: [String: String] = [:]

    let dataStartOffset = headerEnd

    for (key, value) in headerDict {
      if key == "__metadata__" {
        if let dict = value as? [String: Any] {
          for (metaKey, metaValue) in dict {
            if let stringValue = metaValue as? String {
              metadataValues[metaKey] = stringValue
            } else {
              metadataValues[metaKey] = "\(metaValue)"
            }
          }
        }
        continue
      }

      guard let tensorInfo = value as? [String: Any] else {
        throw SafeTensorsReaderError.tensorMetadataMissing(key)
      }

      guard let dtypeString = tensorInfo["dtype"] as? String else {
        throw SafeTensorsReaderError.tensorMetadataMissing(key)
      }

      let dtype = try SafeTensorsReader.mapDType(dtypeString)

      guard let shapeAny = tensorInfo["shape"] as? [Any] else {
        throw SafeTensorsReaderError.tensorMetadataMissing(key)
      }

      let shape: [Int] = try shapeAny.map { element in
        if let number = element as? NSNumber {
          return number.intValue
        } else if let string = element as? String, let intValue = Int(string) {
          return intValue
        } else {
          throw SafeTensorsReaderError.invalidShape(name: key)
        }
      }

      guard let offsetsAny = tensorInfo["data_offsets"] as? [Any], offsetsAny.count == 2 else {
        throw SafeTensorsReaderError.tensorMetadataMissing(key)
      }

      let startOffset = try SafeTensorsReader.parseOffset(offsetsAny[0], tensorName: key)
      let endOffset = try SafeTensorsReader.parseOffset(offsetsAny[1], tensorName: key)

      guard endOffset >= startOffset else {
        throw SafeTensorsReaderError.invalidOffsets(name: key)
      }

      let byteCount = endOffset - startOffset
      let expectedBytes = SafeTensorsReader.expectedByteCount(shape: shape, dtype: dtype)
      guard byteCount == expectedBytes else {
        throw SafeTensorsReaderError.invalidShape(name: key)
      }

      let absoluteOffset = dataStartOffset + startOffset
      guard absoluteOffset + byteCount <= mappedData.count else {
        throw SafeTensorsReaderError.invalidOffsets(name: key)
      }

      tensorMetadata[key] = SafeTensorMetadata(
        name: key,
        dtype: dtype,
        shape: shape,
        dataOffset: absoluteOffset,
        byteCount: byteCount
      )
    }

    self.tensors = tensorMetadata
    self.fileMetadata = metadataValues
  }

  public var tensorNames: [String] {
    Array(tensors.keys)
  }

  public func metadata(for name: String) -> SafeTensorMetadata? {
    tensors[name]
  }

  public func contains(_ name: String) -> Bool {
    tensors[name] != nil
  }

  public func loadAllTensors(as dtype: DType? = nil) throws -> [String: MLXArray] {
    var results: [String: MLXArray] = [:]
    for name in tensorNames {
      var tensor = try self.tensor(named: name)
      if let dtype, tensor.dtype != dtype {
        tensor = tensor.asType(dtype)
      }
      results[name] = tensor
    }
    return results
  }

  public func tensor(named name: String) throws -> MLXArray {
    guard let metadata = tensors[name] else {
      throw SafeTensorsReaderError.tensorNotFound(name)
    }

    return mappedData.withUnsafeBytes { rawBuffer in
      guard let base = rawBuffer.baseAddress else {
        fatalError("Expected non-empty tensor data")
      }

      let startPointer = base.advanced(by: metadata.dataOffset)
      let slicePointer = UnsafeMutableRawPointer(mutating: startPointer)
      let data = Data(bytesNoCopy: slicePointer, count: metadata.byteCount, deallocator: .none)
      return MLXArray(data, metadata.shape, dtype: metadata.dtype)
    }
  }

  public func tensorData(named name: String) throws -> Data {
    guard let metadata = tensors[name] else {
      throw SafeTensorsReaderError.tensorNotFound(name)
    }

    return mappedData.withUnsafeBytes { rawBuffer in
      guard let base = rawBuffer.baseAddress else {
        return Data()
      }
      let startPointer = base.advanced(by: metadata.dataOffset)
      let mutablePointer = UnsafeMutableRawPointer(mutating: startPointer)
      return Data(bytesNoCopy: mutablePointer, count: metadata.byteCount, deallocator: .none)
    }
  }

  public func allMetadata() -> [SafeTensorMetadata] {
    Array(tensors.values)
  }

  private static func parseOffset(_ value: Any, tensorName: String) throws -> Int {
    if let number = value as? NSNumber {
      return number.intValue
    }
    if let string = value as? String, let intValue = Int(string) {
      return intValue
    }
    throw SafeTensorsReaderError.invalidOffsets(name: tensorName)
  }

  private static func expectedByteCount(shape: [Int], dtype: DType) -> Int {
    let elements = shape.reduce(1, *)
    return elements * dtype.size
  }

  private static func mapDType(_ value: String) throws -> DType {
    let key = value.uppercased()
    switch key {
    case "F32":
      return .float32
    case "F16":
      return .float16
    case "F64":
      return .float64
    case "BF16":
      return .bfloat16
    case "I64":
      return .int64
    case "I32":
      return .int32
    case "I16":
      return .int16
    case "I8":
      return .int8
    case "U64":
      return .uint64
    case "U32":
      return .uint32
    case "U16":
      return .uint16
    case "U8":
      return .uint8
    case "BOOL":
      return .bool
    default:
      throw SafeTensorsReaderError.unsupportedDType(value)
    }
  }
}
