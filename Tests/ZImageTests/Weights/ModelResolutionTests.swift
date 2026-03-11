import XCTest

@testable import ZImage

final class ModelResolutionTests: XCTestCase {
  func testResolveRejectsMissingObviousLocalPath() async {
    let missingPath = "./missing-model-\(UUID().uuidString)"

    do {
      _ = try await ModelResolution.resolve(modelSpec: missingPath)
      XCTFail("Expected local path resolution to fail")
    } catch let error as ModelResolutionError {
      guard case .localPathNotFound(let path) = error else {
        XCTFail("Unexpected error: \(error)")
        return
      }
      XCTAssertTrue(path.contains("missing-model-"))
    } catch {
      XCTFail("Unexpected error: \(error)")
    }
  }

  func testAuthorizationRequiredMessageMentionsHFToken() {
    let error = ModelResolutionError.authorizationRequired("example/private-model")

    XCTAssertTrue(error.localizedDescription.contains("HF_TOKEN"))
    XCTAssertTrue(error.localizedDescription.contains("local snapshot"))
  }
}
