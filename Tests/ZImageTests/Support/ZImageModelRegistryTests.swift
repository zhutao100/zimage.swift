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

  func testResolvedPresetUsesKnownBaseDefaults() {
    let base = ZImagePreset.resolved(for: "Tongyi-MAI/Z-Image:main")

    XCTAssertEqual(base.width, 1024)
    XCTAssertEqual(base.height, 1024)
    XCTAssertEqual(base.steps, 50)
    XCTAssertEqual(base.guidanceScale, 4.0)
  }

  func testResolvedPresetPreservesExplicitOverrides() {
    let resolved = ZImagePreset.resolved(
      for: ZImageKnownModel.zImage.id,
      steps: 28,
      guidanceScale: 3.5
    )

    XCTAssertEqual(resolved.width, 1024)
    XCTAssertEqual(resolved.height, 1024)
    XCTAssertEqual(resolved.steps, 28)
    XCTAssertEqual(resolved.guidanceScale, 3.5)
  }

  func testResolvedPresetFallsBackToTurboForUnknownModelId() {
    let resolved = ZImagePreset.resolved(for: "example/local-base-path")

    XCTAssertEqual(resolved.steps, ZImagePreset.zImageTurbo.steps)
    XCTAssertEqual(resolved.guidanceScale, ZImagePreset.zImageTurbo.guidanceScale)
  }

  func testResolvedPresetUsesLocalBaseSnapshotDefaults() {
    let snapshot = TestFixtures.snapshot(named: "ZImageBase")

    let resolved = ZImagePreset.resolved(for: snapshot.path)

    XCTAssertEqual(resolved.steps, ZImagePreset.zImage.steps)
    XCTAssertEqual(resolved.guidanceScale, ZImagePreset.zImage.guidanceScale)
  }

  func testResolvedPresetUsesLocalTurboSnapshotDefaults() {
    let snapshot = TestFixtures.snapshot(named: "ZImageTurbo")

    let resolved = ZImagePreset.resolved(for: snapshot.path)

    XCTAssertEqual(resolved.steps, ZImagePreset.zImageTurbo.steps)
    XCTAssertEqual(resolved.guidanceScale, ZImagePreset.zImageTurbo.guidanceScale)
  }

  func testResolvedPresetUsesParentSnapshotMetadataForLocalCheckpointFile() throws {
    let fixtureSnapshot = TestFixtures.snapshot(named: "ZImageBase")

    try TestFixtures.withTemporaryCopy(of: fixtureSnapshot) { snapshot in
      let checkpoint = snapshot.appending(path: "transformer/diffusion_pytorch_model.safetensors")
      try TestFixtures.createEmptyFile(at: checkpoint)

      let resolved = ZImagePreset.resolved(for: checkpoint.path)

      XCTAssertEqual(resolved.steps, ZImagePreset.zImage.steps)
      XCTAssertEqual(resolved.guidanceScale, ZImagePreset.zImage.guidanceScale)
    }
  }

  func testResolvedPresetUsesHeuristicForUnknownBaseAlias() {
    let resolved = ZImagePreset.resolved(for: "example/z-image-community-fork")

    XCTAssertEqual(resolved.steps, ZImagePreset.zImage.steps)
    XCTAssertEqual(resolved.guidanceScale, ZImagePreset.zImage.guidanceScale)
  }

  func testResolvedPresetUsesHeuristicForUnknownTurboAlias() {
    let resolved = ZImagePreset.resolved(for: "example/z-image-turbo-remix")

    XCTAssertEqual(resolved.steps, ZImagePreset.zImageTurbo.steps)
    XCTAssertEqual(resolved.guidanceScale, ZImagePreset.zImageTurbo.guidanceScale)
  }

  func testResolvedPresetUsesCachedSnapshotMetadataForUnknownAlias() throws {
    let tempCache = FileManager.default.temporaryDirectory.appendingPathComponent("hf_cache_\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: tempCache) }

    let repoRoot = tempCache.appendingPathComponent("models--example--cached-z-image")
    let snapshot = repoRoot.appendingPathComponent("snapshots").appendingPathComponent("commit123")
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    let modelIndex = snapshot.appendingPathComponent("model_index.json")
    try Data(
      """
      {
        "_class_name": "ZImagePipeline",
        "_diffusers_version": "0.0.0-test-fixture",
        "model_type": "z-image"
      }
      """.utf8
    ).write(to: modelIndex)

    let previousHFHubCache = ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
    setenv("HF_HUB_CACHE", tempCache.path, 1)
    defer {
      if let previousHFHubCache {
        setenv("HF_HUB_CACHE", previousHFHubCache, 1)
      } else {
        unsetenv("HF_HUB_CACHE")
      }
    }

    let resolved = ZImagePreset.resolved(for: "example/cached-z-image")

    XCTAssertEqual(resolved.steps, ZImagePreset.zImage.steps)
    XCTAssertEqual(resolved.guidanceScale, ZImagePreset.zImage.guidanceScale)
  }
}
