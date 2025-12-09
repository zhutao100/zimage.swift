import Foundation
import Logging

/// Skeleton loader that will eventually map safetensor shards into Transformer/VAE/Text encoder.
/// For now this just ensures the snapshot exists and logs the discovered files.
public final class ZImageWeightsLoader {
  private let snapshot: URL
  private let logger: Logger

  public init(snapshot: URL, logger: Logger) {
    self.snapshot = snapshot
    self.logger = logger
  }

  public func listComponents() {
    let fm = FileManager.default
    let files = ZImageFiles.resolveTransformerWeights(at: snapshot)
      + ZImageFiles.resolveTextEncoderWeights(at: snapshot)
      + ZImageFiles.vaeWeights
    for file in files {
      let url = snapshot.appending(path: file)
      let exists = fm.fileExists(atPath: url.path)
      logger.info("\(file): \(exists ? "found" : "missing")")
    }
  }
}
