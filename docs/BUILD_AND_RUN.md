# Build and Run Guide

## Prerequisites
- **Hardware**: Apple Silicon Mac (M1/M2/M3).
- **OS**: macOS 14.0+.
- **Tools**: Xcode 15+ (Swift 5.9+).

## Building

### CLI (Release Build)
To build the optimized release binary:
```bash
xcodebuild -scheme ZImageCLI \
  -configuration Release \
  -destination 'platform=macOS' \
  -derivedDataPath .build/xcode
```
The binary will be located at `.build/xcode/Build/Products/Release/ZImageCLI`.

### Library (Package)
To build the library via SwiftPM:
```bash
swift build -c release
```

## Running

### Basic Generation
```bash
./.build/xcode/Build/Products/Release/ZImageCLI \
  -p "A futuristic city at sunset" \
  -o output.png
```

### With Quantized Model
1. Download or quantize a model (see below).
2. Point to the model directory:
```bash
./ZImageCLI -p "..." -m path/to/quantized/model
```

### With ControlNet
```bash
./ZImageCLI control \
  -p "A cat" \
  -c path/to/canny_edge.png \
  --cw path/to/controlnet.safetensors
```

### Quantizing a Model
To convert a full-precision model to 8-bit or 4-bit:
```bash
./ZImageCLI quantize \
  -i path/to/original/model \
  -o path/to/output/model \
  --bits 8 \
  --group-size 32
```

## Testing
Run the full test suite (requires GPU):
```bash
xcodebuild test \
  -scheme zimage.swift \
  -destination 'platform=macOS' \
  -enableCodeCoverage NO
```
**Note**: End-to-End tests will attempt to download the default model (~7.5GB for 8-bit) if not cached.
