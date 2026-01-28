import XCTest
@testable import ZImage

final class ZImageModelRegistryTests: XCTestCase {
  func testPresetDefaultsDifferBetweenTurboAndBase() {
    let turbo = ZImagePreset.defaults(for: ZImageKnownModel.zImageTurbo.id)
    let base = ZImagePreset.defaults(for: ZImageKnownModel.zImage.id)

    XCTAssertNotEqual(turbo.steps, base.steps)
    XCTAssertNotEqual(turbo.guidanceScale, base.guidanceScale)
  }

  func testDefaultCacheDirectoryChangesWithModelId() {
    let root = URL(fileURLWithPath: "/tmp/models")

    let turbo = ZImageRepository.defaultCacheDirectory(for: ZImageKnownModel.zImageTurbo.id, base: root)
    let base = ZImageRepository.defaultCacheDirectory(for: ZImageKnownModel.zImage.id, base: root)

    XCTAssertEqual(turbo.lastPathComponent, "z-image-turbo")
    XCTAssertEqual(base.lastPathComponent, "z-image")
    XCTAssertNotEqual(turbo.path, base.path)
    XCTAssertEqual(ZImageRepository.defaultCacheDirectory(base: root).path, turbo.path)
  }

  func testAreZImageVariantsRecognizesTurboAndBase() {
    XCTAssertTrue(ZImageModelRegistry.areZImageVariants(ZImageKnownModel.zImageTurbo.id, ZImageKnownModel.zImage.id))
    XCTAssertTrue(ZImageModelRegistry.areZImageVariants("Tongyi-MAI/Z-Image:main", "Tongyi-MAI/Z-Image-Turbo:main"))
  }
}
