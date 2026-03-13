import Darwin
import Foundation

final class SocketConnection: @unchecked Sendable {
  private let fileDescriptor: Int32
  private var buffer = Data()

  init(fileDescriptor: Int32) {
    self.fileDescriptor = fileDescriptor
  }

  deinit {
    close(fileDescriptor)
  }

  func writeLine(_ line: String) throws {
    guard let data = (line + "\n").data(using: .utf8) else {
      throw ServiceTransportError.writeFailed("Failed to encode socket payload")
    }

    try data.withUnsafeBytes { rawBuffer in
      guard let baseAddress = rawBuffer.baseAddress else { return }
      var bytesWritten = 0
      while bytesWritten < data.count {
        let written = Darwin.write(
          fileDescriptor,
          baseAddress.advanced(by: bytesWritten),
          data.count - bytesWritten
        )
        if written < 0 {
          throw ServiceTransportError.writeFailed(String(cString: strerror(errno)))
        }
        bytesWritten += written
      }
    }
  }

  func readLine() throws -> String? {
    while true {
      if let newlineIndex = buffer.firstIndex(of: 0x0A) {
        let line = buffer.prefix(upTo: newlineIndex)
        buffer.removeSubrange(...newlineIndex)
        return String(data: line, encoding: .utf8)
      }

      var temp = [UInt8](repeating: 0, count: 4096)
      let count = Darwin.read(fileDescriptor, &temp, temp.count)
      if count < 0 {
        if errno == EINTR {
          continue
        }
        throw ServiceTransportError.readFailed(String(cString: strerror(errno)))
      }
      if count == 0 {
        if buffer.isEmpty {
          return nil
        }
        let line = buffer
        buffer.removeAll(keepingCapacity: true)
        return String(data: line, encoding: .utf8)
      }
      buffer.append(contentsOf: temp.prefix(count))
    }
  }
}

enum UnixDomainSocket {
  private static let maxPathLength = 103

  static func connect(path: String) throws -> SocketConnection {
    let resolvedPath = ServiceSocketPath.resolve(path)
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else {
      throw ServiceTransportError.connectionFailed(String(cString: strerror(errno)))
    }

    do {
      try withSockAddr(path: resolvedPath) { address, length in
        if Darwin.connect(fd, address, length) != 0 {
          throw ServiceTransportError.connectionFailed(
            "Failed to connect to \(resolvedPath): \(String(cString: strerror(errno)))")
        }
      }
      return SocketConnection(fileDescriptor: fd)
    } catch {
      close(fd)
      throw error
    }
  }

  static func makeServerSocket(path: String) throws -> Int32 {
    let resolvedPath = ServiceSocketPath.resolve(path)
    let parentDirectory = URL(fileURLWithPath: resolvedPath).deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parentDirectory, withIntermediateDirectories: true)

    if FileManager.default.fileExists(atPath: resolvedPath) {
      unlink(resolvedPath)
    }

    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else {
      throw ServiceTransportError.bindFailed(String(cString: strerror(errno)))
    }

    var reuse: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

    do {
      try withSockAddr(path: resolvedPath) { address, length in
        if bind(fd, address, length) != 0 {
          throw ServiceTransportError.bindFailed(
            "Failed to bind \(resolvedPath): \(String(cString: strerror(errno)))")
        }
      }
      if listen(fd, SOMAXCONN) != 0 {
        throw ServiceTransportError.listenFailed(
          "Failed to listen on \(resolvedPath): \(String(cString: strerror(errno)))")
      }
      return fd
    } catch {
      close(fd)
      throw error
    }
  }

  static func acceptClient(serverFD: Int32) throws -> SocketConnection {
    let clientFD = accept(serverFD, nil, nil)
    guard clientFD >= 0 else {
      throw ServiceTransportError.acceptFailed(String(cString: strerror(errno)))
    }
    return SocketConnection(fileDescriptor: clientFD)
  }

  private static func withSockAddr<T>(path: String, _ body: (UnsafePointer<sockaddr>, socklen_t) throws -> T) throws
    -> T
  {
    guard path.utf8.count <= maxPathLength else {
      throw ServiceTransportError.socketPathTooLong("Unix socket path exceeds \(maxPathLength) bytes: \(path)")
    }

    var address = sockaddr_un()
    address.sun_family = sa_family_t(AF_UNIX)

    let pathBytes = Array(path.utf8)
    withUnsafeMutableBytes(of: &address.sun_path) { rawBuffer in
      rawBuffer.initializeMemory(as: CChar.self, repeating: 0)
      for (index, byte) in pathBytes.enumerated() {
        rawBuffer[index] = byte
      }
    }

    return try withUnsafePointer(to: &address) {
      try $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { pointer in
        try body(pointer, socklen_t(MemoryLayout<sockaddr_un>.size))
      }
    }
  }
}
