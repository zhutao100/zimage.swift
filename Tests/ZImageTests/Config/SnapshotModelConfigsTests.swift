import XCTest
@testable import ZImage

final class SnapshotModelConfigsTests: XCTestCase {
  func testLoadModelConfigsFromTurboSnapshotFixture() throws {
    let snapshot = TestFixtures.snapshot(named: "ZImageTurbo")

    let configs = try ZImageModelConfigs.load(from: snapshot)

    XCTAssertEqual(configs.transformer.dim, 3840)
    XCTAssertEqual(configs.transformer.axesDims, [32, 48, 48])

    XCTAssertEqual(configs.scheduler.numTrainTimesteps, 1000)
    XCTAssertTrue(configs.scheduler.useDynamicShifting)
    XCTAssertEqual(configs.scheduler.baseShift, 0.5)
    XCTAssertEqual(configs.scheduler.maxShift, 1.15)

    XCTAssertEqual(configs.textEncoder.hiddenSize, 2560)
    XCTAssertEqual(configs.vae.latentChannels, 16)
  }

  func testLoadModelConfigsFromBaseSnapshotFixture() throws {
    let snapshot = TestFixtures.snapshot(named: "ZImageBase")

    let configs = try ZImageModelConfigs.load(from: snapshot)

    XCTAssertEqual(configs.transformer.dim, 4096)
    XCTAssertEqual(configs.scheduler.shift, 3.0)
    XCTAssertFalse(configs.scheduler.useDynamicShifting)
    XCTAssertNil(configs.scheduler.baseShift)
    XCTAssertNil(configs.scheduler.maxShift)
  }
}
