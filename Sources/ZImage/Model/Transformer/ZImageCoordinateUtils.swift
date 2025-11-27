import Foundation
import MLX

enum ZImageCoordinateUtils {

  static func createCoordinateGrid(
    size: (Int, Int, Int),
    start: (Int, Int, Int)
  ) -> MLXArray {
    let (fSize, hSize, wSize) = size
    let (fStart, hStart, wStart) = start

    var values: [Int32] = []
    values.reserveCapacity(fSize * hSize * wSize * 3)

    for f in 0..<fSize {
      for h in 0..<hSize {
        for w in 0..<wSize {
          values.append(Int32(fStart + f))
          values.append(Int32(hStart + h))
          values.append(Int32(wStart + w))
        }
      }
    }

    return MLXArray(values.map(Float32.init), [fSize, hSize, wSize, 3]).asType(.int32)
  }
}
